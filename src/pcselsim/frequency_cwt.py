"""Fixed-geometry, real-frequency cold-cavity closure for layered four-wave CWT.

This solves the scalar, weak-envelope CWT frequency problem, not a full-wave
Maxwell pole or a nonlinear gain-pulled laser frequency. At every trial wavelength
the vertical TE field, C1D/Crad/C2D, and the finite eigenfield are re-solved.
The period and every physical layer thickness remain fixed throughout a solve.

中文：固定实际几何后逐波长重建纵向模式、耦合矩阵和有限腔本征场，求实频
冷腔闭合；这不等于全波 Maxwell 复极点，也不包含激光增益牵引频率。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable
from time import perf_counter

import numpy as np
from scipy.optimize import brentq

from .custom_analysis import FourWaveOpticalSpec, solve_finite_modes, finite_area_eigenproblem
from .reflector_cwt import build_reflector_coupling


@dataclass(frozen=True)
class FrequencyConsistentMode:
    name: str
    wavelength_nm: float
    optical: object
    finite_mode: object
    spec: FourWaveOpticalSpec
    background_detuning_per_m: float
    frequency_residual_per_m: float
    reference_overlap: float
    evaluations: tuple[dict, ...]
    converged: bool = True
    elapsed_seconds: float = 0.0


def background_detuning(wavelength_nm: float, effective_index: float, period_nm: float) -> float:
    """(beta_TE(omega)^2-beta0^2)/(2 beta0), in the retained CWT order.

    Using the squared expression follows the underlying scalar Helmholtz
    reduction before neglecting second derivatives of the in-plane envelope.
    It reduces to ng*(omega-omega_B)/c to first order near Bragg.

    中文：输入波长和周期为 nm，换成米后返回失谐量 m^-1；保留平方形式与
    本模型 Helmholtz 降阶一致，不用经验线性偏移替代实际频率闭合。
    """
    beta = 2 * np.pi * effective_index / (wavelength_nm * 1e-9)
    beta0 = 2 * np.pi / (period_nm * 1e-9)
    return float((beta * beta - beta0 * beta0) / (2 * beta0))


def solve_frequency_consistent_modes(
    cell,
    guided_stack,
    radiation_stack,
    pc_layer_name: str,
    period_nm: float,
    wavelength_reference_nm: float,
    settings,
    device_size_um: float,
    internal_power_loss_cm: float = 0.0,
    grid_points: int = 17,
    *,
    wavelength_interval_nm: tuple[float, float] | None = None,
    tolerance_nm: float = 2e-7,
    residual_tolerance_per_m: float = 0.05,
    optical_factory: Callable[[float], object] | None = None,
    mode_names: tuple[str, ...] = ("A", "B", "C", "D"),
    discretization: str | None = None,
    reference_dense_limit: int = 1400,
) -> dict[str, FrequencyConsistentMode]:
    """Close Re(delta_TE(omega))=Re(lambda_finite[C(omega)]) per branch.

    ``optical_factory`` may supply measured dispersive real-index stacks, but
    must preserve the period and physical geometry. Absorbing-film CWT remains
    explicitly rejected by build_reflector_coupling. The default uses the stated
    constant material indices; frequency closure does not validate that input.

    Each branch follows its reference finite eigenpair using shift-invert near
    its complex eigenvalue and complex-field overlap. Weak matches and unbracketed roots fail instead of returning a
    frozen-frequency value labelled converged. These are four retained finite
    branches, not a certification that every possible transverse mode was found.

    中文：同时检查本征方程残差和频率闭合残差；只跟踪指定参考分支，不把
    四条已求分支当作完整横模谱。扫描波长时禁止改变周期或层厚来凑目标值。
    """
    if (not np.isfinite([period_nm, wavelength_reference_nm, tolerance_nm]).all()
            or min(period_nm, wavelength_reference_nm, tolerance_nm) <= 0):
        raise ValueError("Positive finite period, reference wavelength and tolerance required")
    interval = wavelength_interval_nm or (wavelength_reference_nm - 12, wavelength_reference_nm + 12)
    if not 0 < interval[0] < wavelength_reference_nm < interval[1]:
        raise ValueError("wavelength_interval_nm must enclose the positive reference wavelength")

    def optical_at(wavelength):
        optical = (optical_factory(wavelength) if optical_factory is not None else
                   build_reflector_coupling(
                       cell, guided_stack, radiation_stack, pc_layer_name, period_nm,
                       wavelength_reference_nm, settings, fixed_wavelength_nm=wavelength))
        if not np.isclose(optical.result.lattice_constant_nm, period_nm, rtol=0, atol=1e-10):
            raise ValueError("Frequency closure cannot redesign the physical period")
        if not np.isclose(optical.result.bragg_wavelength_nm, wavelength, rtol=0, atol=1e-9):
            raise ValueError("optical_factory returned a Bragg-locked rather than requested frequency")
        return optical

    def finite_at(optical):
        r = optical.result
        spec = FourWaveOpticalSpec(r.bragg_wavelength_nm, r.effective_index, r.group_index,
                                  device_size_um, internal_power_loss_cm,
                                  tuple(r.eigenvalues_m.real / 100),
                                  tuple(r.eigenvalues_m.imag / 100), grid_points)
        if discretization is not None:
            spec = replace(spec, discretization=discretization)
        spec = replace(spec,dense_eigenproblem_limit=reference_dense_limit,eigen_maxiter=500)
        modes = solve_finite_modes(spec, r.coupling_m, r.eigenvectors, r.radiation_fields)
        return spec, modes

    reference_optical = optical_at(float(wavelength_reference_nm))
    reference_spec, reference = finite_at(reference_optical)
    labels = tuple(reference)
    if any(name not in reference for name in mode_names):
        raise ValueError("mode_names must select retained finite branches")
    cache = {}
    fallback_evaluations = set()

    def evaluate(wavelength, name):
        key = (name, float(wavelength))
        if key not in cache:
            optical = optical_at(float(wavelength))
            r = optical.result
            spec = replace(reference_spec, wavelength_nm=float(wavelength),
                           effective_index=r.effective_index, group_index=r.group_index)
            problem = finite_area_eigenproblem(spec, r.coupling_m)
            seed = reference[name]
            target = complex(seed.delta_per_m, seed.alpha_per_m)
            flat_seed = seed.fields.reshape(4, -1)
            change = r.coupling_m-reference_optical.result.coupling_m
            target += np.einsum("ik,ij,jk->", flat_seed.conj(), change, flat_seed) / np.sum(abs(flat_seed)**2)
            try:
                values, fields = problem.near(target, 4, seed.fields.ravel(), tol=1e-10,maxiter=400)
                overlaps = abs(seed.fields.ravel().conj() @ fields) ** 2 / np.sum(abs(seed.fields)**2)
                selected = int(np.argmax(overlaps))
                field = fields[:, selected].reshape(seed.fields.shape)
                value = values[selected]
                if value.imag < -1e-7:
                    raise RuntimeError("Tracked passive frequency mode acquired negative loss")
                rx, ry = r.radiation_fields(field)
                operator = problem.center_operator()
                from .finite_discretization import relative_residual
                mode = replace(seed, delta_per_m=float(value.real), alpha_per_m=float(max(value.imag, 0)),
                               fields=field, radiation_x=rx, radiation_y=ry,
                               eigen_residual=relative_residual(operator, field, value))
                selected_overlap = float(overlaps[selected])
            except (RuntimeError, np.linalg.LinAlgError):
                # Bounded sparse failure is followed by the same K/M full
                # candidate search; no tolerance or physical loss is relaxed.
                selected_overlap = 0.0
            if selected_overlap < .1:
                # A dense lateral cluster can move past the small predictor
                # window, especially for weak-PC-overlap higher TE families.
                # Re-solve the independent full band-edge candidate search,
                # then match its actual fields. Never accept a low-overlap
                # eigenpair merely to make the scalar root finder converge.
                _, candidates = finite_at(optical)
                overlap_by_name = {
                    label: abs(np.vdot(seed.fields, candidate.fields))**2
                    / (np.sum(abs(seed.fields)**2)*np.sum(abs(candidate.fields)**2))
                    for label,candidate in candidates.items()}
                label = max(overlap_by_name,key=overlap_by_name.get)
                mode = candidates[label]
                selected_overlap = float(overlap_by_name[label])
                fallback_evaluations.add(key)
            detuning = background_detuning(float(wavelength), r.effective_index, period_nm)
            cache[key] = optical, spec, (mode, selected_overlap), detuning
        return cache[key]

    # Seed reuse is exact, saving one expensive finite solve.
    for name, mode in reference.items():
        cache[(name, float(wavelength_reference_nm))] = (
            reference_optical, reference_spec, (mode, 1.0),
            background_detuning(wavelength_reference_nm, reference_optical.result.effective_index, period_nm))
    results = {}
    for name in mode_names:
        branch_start = perf_counter()
        trace = []

        def gap(wavelength):
            evaluation_start = perf_counter()
            cached = (name,float(wavelength)) in cache
            optical, spec, matched, detuning = evaluate(wavelength, name)
            mode, overlap = matched
            if overlap < 0.1:
                raise RuntimeError(f"Frequency branch {name} lost its reference field (overlap={overlap:g})")
            residual = detuning - mode.delta_per_m
            trace.append(dict(wavelength_nm=float(wavelength), effective_index=optical.result.effective_index,
                              group_index=optical.result.group_index, background_detuning_per_m=detuning,
                              finite_detuning_per_m=mode.delta_per_m, residual_per_m=float(residual),
                              field_overlap=overlap, optical_theorem_residual=optical.optical_theorem_residual,
                              eigenpair_tracking="full_band_cluster_fallback" if (name,float(wavelength)) in fallback_evaluations else "four_candidate_predictor_continuation",
                              cache_hit=cached,evaluation_elapsed_seconds=perf_counter()-evaluation_start,
                              cumulative_branch_elapsed_seconds=perf_counter()-branch_start))
            return residual

        # A narrow local bracket avoids crossing unrelated branches/poles. The
        # bounded expansion fails honestly when the stated spectral range ends.
        centre = wavelength_reference_nm
        half = min(0.5, (interval[1] - interval[0]) / 4)
        while True:
            left, right = max(interval[0], centre - half), min(interval[1], centre + half)
            gl, gr = gap(left), gap(right)
            if gl * gr <= 0:
                break
            if left == interval[0] and right == interval[1]:
                raise RuntimeError(f"No frequency root for {name} in calibrated interval {interval}")
            half *= 2
        wavelength = float(brentq(gap, left, right, xtol=tolerance_nm, rtol=1e-12))
        residual = gap(wavelength)
        if abs(residual) > residual_tolerance_per_m:
            raise RuntimeError(f"Frequency residual for {name} is {residual:g} /m")
        optical, spec, matched, detuning = evaluate(wavelength, name)
        mode, overlap = matched
        results[name] = FrequencyConsistentMode(
            name, wavelength, optical, replace(mode, name=name), spec,
            detuning, float(residual), overlap, tuple(trace),elapsed_seconds=perf_counter()-branch_start)
    return results
