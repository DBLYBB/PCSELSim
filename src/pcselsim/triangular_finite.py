"""Finite-device companion solver for the triangular six-wave 3-D CWT model.

The six envelopes propagate along Liang Eq. (5.17)'s characteristic directions.
The default generalized midpoint/link discretization implements the averages
and differences of Eqs. (5.18)-(5.19), with zero incoming boundary amplitudes.
The former first-order upwind operator is available as ``legacy_upwind``.
Reported eigenvalues and fields come from the same actual finest-grid solve;
mesh sensitivity is reported separately, without replacing eigenvalues by fits.

中文：六个包络沿三组特征线传播，默认使用Liang广义交错平均/差分并在入射边界取零。
圆形/正六边形/方形边界仍有阶梯几何误差，必须细化验证。远场对真实坐标作Fourier求和，
没有把斜坐标数组错误地当成笛卡尔像素做 FFT，也没有人为乘涡旋或高斯因子。
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle, Polygon, Rectangle
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from .triangular_six_wave import (
    TRIANGULAR_MODE_NAMES,
    TRIANGULAR_DIRECTIONS, TRIANGULAR_POLARIZATIONS,
    TriangularCouplingResult,
)
from .numerical_quality import extrapolate_grid_loss
from .finite_discretization import (build_finite_eigenproblem, relative_residual,
                                    staggered_grid_diagnostics)


@dataclass(frozen=True)
class TriangularFiniteSpec:
    """Finite-device discretization parameters.

    ``radius_um`` is the circle/hexagon circumradius or square half-width.
    The characteristic grid remains aligned with the six Bloch-wave directions;
    only the physical aperture mask changes.

    中文：radius_um 对方形表示半宽，对圆形/正六边形表示外接圆半径。
    内部特征网格与物理外边界分开定义；300 μm 方形腔应指定半宽 150 μm。
    """

    radius_um: float = 30.0
    radius_cells: int = 9
    # Legacy optional *field-amplitude* offset, added as +i*100*value to C.
    # Current workflows solve the passive cavity with 0, then add the separate
    # material intensity loss once in (2*alpha_field + alpha_internal)/overlap.
    internal_loss_cm: float = 0.0
    eigensolutions_per_target: int = 8
    aperture_shape: str = "circle"
    discretization: str = "staggered"


@dataclass(frozen=True)
class HexagonalGrid:
    """Axial grid embedded in Cartesian coordinates."""

    radius_cells: int
    radius_um: float
    axial_j: np.ndarray
    axial_k: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    spacing_m: float
    index: dict[tuple[int, int], int]
    aperture_shape: str = "circle"

    @property
    def points(self) -> int:
        return int(self.axial_j.size)


@dataclass
class TriangularFiniteMode:
    """One finite six-wave eigenmode on the chosen physical aperture.

    fields has shape (6, grid.points) and stores actual coefficient-point
    envelopes. alpha_per_m is field-amplitude loss; intensity sums envelope
    powers and does not include unit-cell Bloch interference.
    中文：fields 为 (6, 网格点数) 的实际系数点包络；alpha_per_m 是振幅损耗。
    intensity 为六包络功率之和，不含单胞内 Bloch 波之间的微观干涉图样。
    """

    name: str
    delta_per_m: float
    alpha_per_m: float
    fields: np.ndarray
    band_overlap: float
    grid: HexagonalGrid
    radiation_x: np.ndarray
    radiation_y: np.ndarray
    grid_alpha_per_m: float | None = None
    extrapolation_uncertainty_per_m: float | None = None
    extrapolation_status: str = "single_grid"
    discretization: str = "legacy_upwind"
    eigen_residual: float | None = None

    @property
    def intensity(self) -> np.ndarray:
        value = np.sum(np.abs(self.fields) ** 2, axis=0)
        return value / max(float(value.max()), np.finfo(float).eps)

    @property
    def alpha_l(self) -> float:
        return float(self.alpha_per_m * self.grid.radius_um * 1e-6)


@dataclass(frozen=True)
class TriangularFarField:
    """Jones fields and window-qualified intensity metrics / Jones 远场及角窗内强度指标。

    field_x/field_y share their complex scale; power components share one
    intensity peak normalization. ellipticity is the beam's second-moment axis
    ratio, not its polarization ellipticity angle.
    中文：两 Jones 分量共用复数尺度，功率分量共用总强度峰值归一化。
    ellipticity 是光斑二阶矩轴比，与偏振椭圆率角不同。
    """
    angle_deg: np.ndarray
    field_x: np.ndarray
    field_y: np.ndarray
    power: np.ndarray
    power_x: np.ndarray
    power_y: np.ndarray
    full_rms_divergence_deg: float
    center_to_peak: float
    peak_offset_deg: float
    centroid_offset_deg: float
    ellipticity: float
    encircled_power_0p5deg: float
    encircled_power_1deg: float
    requested_view_deg: float = 3.0
    evaluated_view_deg: float = 3.0
    alias_free_square_view_deg: float = 3.0
    energy_normalization: str = "evaluated_angular_window"
    moment_reference: str = "surface_normal_radial_second_moment"
    samples: int = 121


def build_hexagonal_grid(radius_cells: int, radius_um: float) -> HexagonalGrid:
    """Backward-compatible circular aperture on the axial characteristic grid."""
    return build_device_grid(radius_cells, radius_um, "circle")


def build_device_grid(
    radius_cells: int,
    radius_um: float,
    aperture_shape: str = "circle",
) -> HexagonalGrid:
    """Build a circular, regular-hexagonal, or square physical aperture.

    The square uses ``radius_um`` as half-width.  The regular hexagon uses it
    as circumradius.  All three masks therefore fit inside the same 300-um
    bounding box when ``radius_um=150``.
    """
    radius_cells = int(radius_cells)
    if radius_cells < 3:
        raise ValueError("radius_cells must be >= 3")
    if radius_um <= 0.0:
        raise ValueError("radius_um must be positive")
    shape = str(aperture_shape).lower()
    if shape not in {"circle", "hexagon", "square"}:
        raise ValueError("aperture_shape must be 'circle', 'hexagon', or 'square'")
    if shape == "square":
        extent = 2 * radius_cells
    elif shape == "circle":
        # A metric circle reaches |j|,|k|=2N/sqrt(3), just like the
        # reciprocal-shell bound. Limiting either index to N clips six arcs
        # and breaks C6 for N>=7 even though every retained node is in-circle.
        extent = int(np.ceil(2*radius_cells/np.sqrt(3)))
    else:
        extent = radius_cells
    nodes = []
    for j in range(-extent, extent + 1):
        for k in range(-extent, extent + 1):
            x_index = 0.5 * (j + k)
            y_index = 0.5 * np.sqrt(3.0) * (k - j)
            if shape == "circle":
                inside = j * j - j * k + k * k <= radius_cells * radius_cells
            elif shape == "hexagon":
                inside = max(abs(j), abs(k), abs(j-k)) <= radius_cells
            else:
                inside = (
                    abs(x_index) <= radius_cells + 1e-12
                    and abs(y_index) <= radius_cells + 1e-12
                )
            if inside:
                nodes.append((j, k))
    nodes.sort()
    axial_j = np.asarray([node[0] for node in nodes], dtype=int)
    axial_k = np.asarray([node[1] for node in nodes], dtype=int)
    spacing_um = radius_um / radius_cells
    x_um = 0.5 * spacing_um * (axial_j + axial_k)
    y_um = 0.5 * np.sqrt(3.0) * spacing_um * (axial_k - axial_j)
    return HexagonalGrid(
        radius_cells=radius_cells,
        radius_um=float(radius_um),
        axial_j=axial_j,
        axial_k=axial_k,
        x_um=x_um,
        y_um=y_um,
        spacing_m=spacing_um * 1e-6,
        index={node: position for position, node in enumerate(nodes)},
        aperture_shape=shape,
    )


def _directed_upwind(
    grid: HexagonalGrid,
    step: tuple[int, int],
    positive_direction: bool,
) -> sparse.csr_matrix:
    """Directional derivative with a zero incoming value outside the disk."""
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    dj, dk = step
    for row, (j, k) in enumerate(zip(grid.axial_j, grid.axial_k, strict=True)):
        rows.append(row)
        columns.append(row)
        data.append(1.0 / grid.spacing_m)
        neighbour = (j - dj, k - dk) if positive_direction else (j + dj, k + dk)
        column = grid.index.get(neighbour)
        if column is not None:
            rows.append(row)
            columns.append(column)
            data.append(-1.0 / grid.spacing_m)
    return sparse.csr_matrix(
        (data, (rows, columns)), shape=(grid.points, grid.points)
    )


def triangular_finite_eigenproblem(
    result: TriangularCouplingResult,
    spec: TriangularFiniteSpec,
):
    """Liang (5.18) generalized staggered operator on characteristic links."""
    grid = build_device_grid(
        spec.radius_cells, spec.radius_um, spec.aperture_shape
    )
    local_matrix = result.coupling_m + 1j * 100.0 * spec.internal_loss_cm * np.eye(6)
    directions = ((1, 0), (1, 0), (0, 1), (0, 1), (1, 1), (1, 1))
    signs = (True, False, True, False, True, False)
    derivatives = tuple(_directed_upwind(grid, step, positive)
                        for step, positive in zip(directions, signs, strict=True))
    problem = build_finite_eigenproblem(local_matrix, derivatives, grid.spacing_m,
                                       spec.discretization)
    return problem, grid


def triangular_finite_operator(result: TriangularCouplingResult, spec: TriangularFiniteSpec):
    """Equivalent operator on common coefficient-point fields (not link fields)."""
    problem, grid = triangular_finite_eigenproblem(result, spec)
    return problem.center_operator(), grid


def _hex_roughness(fields: np.ndarray, grid: HexagonalGrid) -> float:
    numerator = 0.0
    samples = 0
    for step in ((1, 0), (0, 1), (1, 1)):
        dj, dk = step
        for left, (j, k) in enumerate(zip(grid.axial_j, grid.axial_k, strict=True)):
            right = grid.index.get((j + dj, k + dk))
            if right is None:
                continue
            numerator += float(np.sum(np.abs(fields[:, left] - fields[:, right]) ** 2))
            samples += 1
    denominator = max(float(np.mean(np.abs(fields) ** 2)), np.finfo(float).eps)
    return numerator / max(samples * denominator, np.finfo(float).eps)


def _transform_six_wave_fields(fields: np.ndarray, grid: HexagonalGrid,
                                transform: np.ndarray) -> np.ndarray | None:
    """Transform coordinates, propagation directions and physical TE vectors.

    Return None if the actual aperture is not invariant. Acceptance as an
    eigenpartner additionally requires a residual check against its operator.
    """
    direct = np.asarray(((.5, .5), (-np.sqrt(3)/2, np.sqrt(3)/2)))
    axial_transform = np.linalg.solve(direct, transform @ direct)
    transformed = np.rint(axial_transform @ np.vstack((grid.axial_j, grid.axial_k))).astype(int)
    destinations = [grid.index.get(tuple(point)) for point in transformed.T]
    if any(index is None for index in destinations):
        return None
    answer = np.empty_like(fields)
    for channel, direction in enumerate(TRIANGULAR_DIRECTIONS):
        new_direction = transform @ direction
        target = int(np.argmin(np.linalg.norm(TRIANGULAR_DIRECTIONS-new_direction, axis=1)))
        sign = float(TRIANGULAR_POLARIZATIONS[target] @ (transform @ TRIANGULAR_POLARIZATIONS[channel]))
        answer[target, destinations] = sign * fields[channel]
    return answer


def solve_triangular_finite_modes(
    result: TriangularCouplingResult,
    spec: TriangularFiniteSpec,
) -> dict[str, TriangularFiniteMode]:
    """Find the smooth fundamental finite state connected to every Gamma band edge.

    中文：这是按 Γ 点模族选取平滑基态的便捷接口，不返回完整横向谱。
    声称最低阈值前应使用额外复移位/完整粗谱挑战，并对候选逐条闭合实频。
    """
    problem, grid = triangular_finite_eigenproblem(result, spec)
    operator = problem.center_operator()
    targets = np.asarray(result.eigenvalues_m)
    basis = np.asarray(result.eigenvectors)
    sample = np.arange(operator.shape[0], dtype=float)
    phase = 2.0 * np.pi * sample / operator.shape[0]
    v0 = 1.0 + 0.19 * np.cos(phase) + 1j * (0.17 * np.sin(3.0 * phase))

    # Solve once for every distinct band edge.  Degenerate B and D pairs share
    # one candidate pool and are assigned globally by projected overlap.
    groups: list[list[int]] = []
    for mode_index, target in enumerate(targets):
        for group in groups:
            reference = targets[group[0]]
            if abs(target - reference) <= 1e-7 * max(abs(reference), 1.0):
                group.append(mode_index)
                break
        else:
            groups.append([mode_index])

    selected_modes: dict[str, TriangularFiniteMode] = {}
    for group in groups:
        target = complex(np.mean(targets[group]))
        requested = max(spec.eigensolutions_per_target, 3 * len(group) + 2)
        requested = min(requested, operator.shape[0] - 2)
        values, vectors = problem.near(target, requested, v0,
                                      maxiter=max(8000, 6 * operator.shape[0]))
        requested = len(values)
        normalized_vectors = vectors / np.linalg.norm(vectors, axis=0, keepdims=True)
        # Gamma eigenvectors inside a degenerate eigenspace are arbitrary and
        # may even be nonorthogonal. Project onto its QR-orthonormalized whole
        # subspace; individual-vector overlaps spuriously split circle doublets
        # by selecting different radial orders for the two labels.
        group_basis, _ = np.linalg.qr(basis[:, group])
        costs = np.empty((len(group), requested), dtype=float)
        overlaps = np.empty_like(costs)
        for local_row, band_index in enumerate(group):
            for candidate in range(requested):
                fields = normalized_vectors[:, candidate].reshape(6, grid.points)
                projection = group_basis.conj().T @ fields
                overlap = float(np.sum(np.abs(projection) ** 2))
                overlaps[local_row, candidate] = overlap
                passive_penalty = 20.0 if values[candidate].imag < 0.0 else 0.0
                costs[local_row, candidate] = (
                    abs(values[candidate].real - targets[band_index].real)
                    / max(abs(targets[band_index].real), 2e4)
                    + 0.25 * _hex_roughness(fields, grid)
                    + 0.45 * (1.0 - overlap)
                    + passive_penalty
                )
        rows, columns = linear_sum_assignment(costs)
        assignment = {int(row): int(column) for row, column in zip(rows, columns, strict=True)}
        for local_row, band_index in enumerate(group):
            candidate = assignment[local_row]
            value = values[candidate]
            if value.imag < -1e-8 * max(float(np.linalg.norm(result.coupling_m)), 1.0):
                raise RuntimeError("A selected passive six-wave eigenmode has negative loss")
            fields = normalized_vectors[:, candidate].reshape(6, grid.points)
            radiation_x, radiation_y = result.radiation_amplitudes(fields)
            selected_modes[TRIANGULAR_MODE_NAMES[band_index]] = TriangularFiniteMode(
                name=TRIANGULAR_MODE_NAMES[band_index],
                delta_per_m=float(value.real),
                alpha_per_m=float(max(value.imag, 0.0)),
                fields=fields,
                band_overlap=float(overlaps[local_row, candidate]),
                grid=grid,
                radiation_x=np.asarray(radiation_x),
                radiation_y=np.asarray(radiation_y),
                discretization=spec.discretization,
                eigen_residual=relative_residual(operator, fields, value),
            )
        # A single-vector Arnoldi space need not contain both vectors of an
        # exactly repeated eigenvalue. Recover a physical symmetry partner only
        # when it independently solves this finite operator (not from Gamma
        # degeneracy alone). This prevents different radial orders masquerading
        # as the two members of a circular-aperture doublet.
        if len(group) == 2:
            first_name, second_name = (TRIANGULAR_MODE_NAMES[index] for index in group)
            seed = min((selected_modes[first_name], selected_modes[second_name]),
                       key=lambda mode: _hex_roughness(mode.fields, grid))
            value = seed.delta_per_m + 1j*seed.alpha_per_m
            rotation = np.asarray(((.5, -np.sqrt(3)/2), (np.sqrt(3)/2, .5)))
            for transform in (rotation, np.diag((1., -1.)), np.diag((-1., 1.))):
                transformed = _transform_six_wave_fields(seed.fields, grid, transform)
                if transformed is None:
                    continue
                independent = transformed - np.vdot(seed.fields, transformed)*seed.fields
                norm = float(np.linalg.norm(independent))
                if norm < 1e-3:
                    continue
                independent /= norm
                residual = relative_residual(operator, independent, value)
                if residual < 1e-7:
                    radiation_x, radiation_y = result.radiation_amplitudes(independent)
                    selected_modes[first_name] = replace(seed, name=first_name)
                    selected_modes[second_name] = replace(seed, name=second_name,
                        fields=independent, radiation_x=radiation_x, radiation_y=radiation_y,
                        eigen_residual=residual)
                    break
    return {name: selected_modes[name] for name in TRIANGULAR_MODE_NAMES}


def solve_triangular_finite_modes_converged(
    result: TriangularCouplingResult,
    spec: TriangularFiniteSpec,
    radius_cells: tuple[int, ...],
) -> tuple[dict[str, TriangularFiniteMode], dict[str, np.ndarray]]:
    """Study mesh sensitivity; staggered retains the actual finest eigenpair.

    Legacy_upwind alone retains the historical intercept for comparisons.
    A mesh_checked label is empirical and does not certify the physical model.
    """
    if len(radius_cells) < 3 or any(
        left >= right for left, right in zip(radius_cells, radius_cells[1:])
    ):
        raise ValueError("radius_cells must contain at least three increasing values")
    solutions = [
        solve_triangular_finite_modes(result, replace(spec, radius_cells=cells))
        for cells in radius_cells
    ]
    inverse = 1.0 / np.asarray(radius_cells, dtype=float)
    answer: dict[str, TriangularFiniteMode] = {}
    convergence: dict[str, np.ndarray] = {
        "inverse_grid": inverse,
        "radius_cells": np.asarray(radius_cells, dtype=float),
    }
    for name in TRIANGULAR_MODE_NAMES:
        values = np.asarray([
            solution[name].delta_per_m + 1j * solution[name].alpha_per_m
            for solution in solutions
        ])
        if spec.discretization == "staggered":
            raw_alpha, uncertainty, status = staggered_grid_diagnostics(inverse, values.imag)
            answer[name] = replace(
                solutions[-1][name], grid_alpha_per_m=solutions[-1][name].alpha_per_m,
                extrapolation_uncertainty_per_m=uncertainty, extrapolation_status=status,
            )
            convergence[name] = values
            convergence[f"{name}_alpha_diagnostics"] = np.asarray(
                [raw_alpha, uncertainty, solutions[-1][name].alpha_per_m]
            )
            continue
        delta_limit = float(np.polyfit(inverse, values.real, 1)[1])
        alpha_limit, raw_alpha, uncertainty, status = extrapolate_grid_loss(
            inverse, values.imag
        )
        if status != "provisional":
            warnings.warn(
                f"Six-wave mode {name}: {status}; raw alpha intercept={raw_alpha:.6g} "
                f"m^-1, grid sensitivity={uncertainty:.6g} m^-1. Refine the grid "
                "before using this loss for design ranking.", RuntimeWarning, stacklevel=2,
            )
        answer[name] = replace(
            solutions[-1][name],
            delta_per_m=delta_limit,
            alpha_per_m=alpha_limit,
            grid_alpha_per_m=solutions[-1][name].alpha_per_m,
            extrapolation_uncertainty_per_m=uncertainty,
            extrapolation_status=status,
        )
        convergence[name] = values
        convergence[f"{name}_alpha_diagnostics"] = np.asarray(
            [raw_alpha, uncertainty, solutions[-1][name].alpha_per_m]
        )
    return answer, convergence


def plot_triangular_grid_convergence(
    convergence: dict[str, np.ndarray], radius_um: float, path: Path
) -> None:
    """Plot finite eigenvalue convergence and the reported ``1/N -> 0`` intercept."""
    inverse = convergence["inverse_grid"]
    length_m = radius_um * 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for name in TRIANGULAR_MODE_NAMES:
        values = convergence[name]
        xfit = np.linspace(0.0, float(inverse.max()), 100)
        fit_delta = np.polyfit(inverse, values.real * length_m, 1)
        fit_alpha = np.polyfit(inverse, values.imag * length_m, 1)
        axes[0].plot(inverse, values.real * length_m, "o", label=name)
        axes[0].plot(xfit, np.polyval(fit_delta, xfit), lw=1.0)
        axes[1].plot(inverse, values.imag * length_m, "o", label=name)
        axes[1].plot(xfit, np.polyval(fit_alpha, xfit), lw=1.0)
    axes[0].set(xlabel="1/N", ylabel="delta L", title="frequency convergence")
    axes[1].set(xlabel="1/N", ylabel="alpha L", title="threshold convergence")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(ncol=2)
    fig.suptitle("Finite-device mesh study; staggered reports the actual finest eigenpair")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _cartesian_point_fourier_sum(x_m, y_m, radiation_x, radiation_y, wave):
    """Exact point sum on a square angular axis, with a bounded fast path.

    Characteristic-grid points share Cartesian coordinates. Scatter their
    complex amplitudes onto the *exact* unique-coordinate grid and separate
    exp[-i(kx*x+ky*y)] into two matrix products. This changes summation order
    only: coordinates are neither rounded nor interpolated, and the two Jones
    components keep their common complex scale. Unstructured point clouds use
    the direct chunked sum instead of allocating a dense quadratic grid.

    中文：散点按完全相同的笛卡尔坐标归并后分离指数求和，只改变加法次序；
    不舍入坐标、不插值、不把斜网格当普通像素 FFT，两偏振分量保持共同尺度。
    """
    x, ix = np.unique(x_m, return_inverse=True)
    y, iy = np.unique(y_m, return_inverse=True)
    if len(x)*len(y) <= min(4*len(x_m), 1_000_000):
        phase_x = np.exp(-1j*wave[:, None]*x[None, :])
        phase_y = np.exp(-1j*wave[:, None]*y[None, :])
        result = []
        for radiation in (radiation_x, radiation_y):
            aperture = np.zeros((len(y), len(x)), dtype=np.complex128)
            np.add.at(aperture, (iy, ix), radiation)
            result.append((phase_y @ aperture) @ phase_x.T)
        return result
    kx, ky = np.meshgrid(wave, wave)
    kx, ky = kx.ravel(), ky.ravel()
    output_x = np.empty(kx.size, dtype=np.complex128)
    output_y = np.empty_like(output_x)
    for start in range(0, kx.size, 512):
        stop = min(start+512, kx.size)
        phase = np.exp(-1j*(kx[start:stop, None]*x_m[None, :]
                           + ky[start:stop, None]*y_m[None, :]))
        output_x[start:stop] = phase @ radiation_x
        output_y[start:stop] = phase @ radiation_y
    return output_x.reshape(len(wave), len(wave)), output_y.reshape(len(wave), len(wave))


def triangular_vector_far_field(
    mode: TriangularFiniteMode,
    wavelength_nm: float,
    view_deg: float = 3.0,
    samples: int = 121,
) -> TriangularFarField:
    """Point-sampled aperture FFP, restricted to a nonaliased angular window.

    The coarse characteristic lattice produces reciprocal-lattice copies if
    evaluated beyond its sampling Brillouin zone.  Restrict all four corners
    of the requested square to that zone.  Energy ratios and RMS are relative
    to this evaluated window; they are not whole-hemisphere beam metrics.

    中文：对实际辐射场及物理坐标做 Fourier 点求和，并限制在采样无混叠角窗。
    包围能量、RMS 和轴比仅针对该角窗；增大绘图角度不能绕过源网格 Nyquist 限制。
    """
    if view_deg <= 0.0 or samples < 3 or wavelength_nm <= 0.0:
        raise ValueError("view_deg and wavelength_nm must be positive; samples >= 3")
    if samples % 2 == 0:
        samples += 1  # Keep the physical surface-normal sample exactly at zero.
    wavelength_m = wavelength_nm * 1e-9
    # Hexagonal sampling BZ: |kx +/- ky/sqrt(3)| <= 4*pi/(3*h).
    safe_tangent = 2.0 * wavelength_m / (
        3.0 * mode.grid.spacing_m * (1.0 + 1.0 / np.sqrt(3.0))
    )
    safe_view = float(np.rad2deg(np.arctan(safe_tangent)))
    evaluated_view = min(float(view_deg), 0.98 * safe_view)
    if view_deg > safe_view:
        warnings.warn(
            f"Requested six-wave far-field view +/-{view_deg:g} deg exceeds the "
            f"coarse-grid alias-free square +/-{safe_view:.3g} deg; evaluated "
            f"+/-{evaluated_view:.3g} deg. Encircled energy is window-normalized.",
            RuntimeWarning, stacklevel=2,
        )
    angle = np.linspace(-evaluated_view, evaluated_view, int(samples))
    theta_x, theta_y = np.meshgrid(np.deg2rad(angle), np.deg2rad(angle))
    k0 = 2.0 * np.pi / (wavelength_nm * 1e-9)
    x_m = mode.grid.x_um * 1e-6
    y_m = mode.grid.y_um * 1e-6
    output_x, output_y = _cartesian_point_fourier_sum(
        x_m, y_m, mode.radiation_x, mode.radiation_y,
        k0*np.tan(np.deg2rad(angle)))
    obliquity = np.cos(theta_x) + np.cos(theta_y) - 1.0
    field_x = output_x * obliquity
    field_y = output_y * obliquity
    power_x = np.abs(field_x) ** 2
    power_y = np.abs(field_y) ** 2
    power = power_x + power_y
    scale = max(float(power.max()), np.finfo(float).eps)
    power /= scale
    power_x /= scale
    power_y /= scale
    xx, yy = np.meshgrid(angle, angle)
    total = max(float(power.sum()), np.finfo(float).eps)
    divergence = 2.0 * np.sqrt(float(np.sum((xx * xx + yy * yy) * power) / total))
    center = len(angle) // 2
    peak_row, peak_column = np.unravel_index(int(np.argmax(power)), power.shape)
    peak_offset = float(np.hypot(xx[peak_row, peak_column], yy[peak_row, peak_column]))
    centroid_x = float(np.sum(xx * power) / total)
    centroid_y = float(np.sum(yy * power) / total)
    centroid_offset = float(np.hypot(centroid_x, centroid_y))
    centered_x = xx - centroid_x
    centered_y = yy - centroid_y
    covariance = np.asarray((
        (
            float(np.sum(centered_x * centered_x * power) / total),
            float(np.sum(centered_x * centered_y * power) / total),
        ),
        (
            float(np.sum(centered_x * centered_y * power) / total),
            float(np.sum(centered_y * centered_y * power) / total),
        ),
    ))
    principal_variances = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    ellipticity = float(np.sqrt(
        principal_variances[-1]
        / max(principal_variances[0], np.finfo(float).eps)
    ))
    radial_angle = np.hypot(xx, yy)
    encircled_0p5 = float(np.sum(power[radial_angle <= 0.5]) / total)
    encircled = float(np.sum(power[radial_angle <= 1.0]) / total)
    return TriangularFarField(
        angle_deg=angle,
        field_x=field_x,
        field_y=field_y,
        power=power,
        power_x=power_x,
        power_y=power_y,
        full_rms_divergence_deg=divergence,
        center_to_peak=float(power[center, center]),
        peak_offset_deg=peak_offset,
        centroid_offset_deg=centroid_offset,
        ellipticity=ellipticity,
        encircled_power_0p5deg=encircled_0p5,
        encircled_power_1deg=encircled,
        requested_view_deg=float(view_deg),
        evaluated_view_deg=evaluated_view,
        alias_free_square_view_deg=safe_view,
        samples=int(samples),
    )


def _tripcolor(ax: plt.Axes, mode: TriangularFiniteMode) -> None:
    image = ax.tripcolor(
        mode.grid.x_um,
        mode.grid.y_um,
        mode.intensity,
        shading="gouraud",
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
    )
    radius = mode.grid.radius_um
    if mode.grid.aperture_shape == "circle":
        boundary = Circle((0.0, 0.0), radius, fill=False, ls="--", ec="white")
    elif mode.grid.aperture_shape == "hexagon":
        phase = np.deg2rad(np.arange(0.0, 360.0, 60.0))
        boundary = Polygon(
            np.column_stack((radius * np.cos(phase), radius * np.sin(phase))),
            closed=True, fill=False, ls="--", ec="white",
        )
    else:
        boundary = Rectangle(
            (-radius, -radius), 2.0 * radius, 2.0 * radius,
            fill=False, ls="--", ec="white",
        )
    ax.add_patch(boundary)
    ax.set_aspect("equal")
    return image


def write_triangular_finite_table(
    modes: dict[str, TriangularFiniteMode],
    result: TriangularCouplingResult,
    path: Path,
    view_deg: float = 1.0,
    samples: int = 161,
    far_fields: dict[str, TriangularFarField] | None = None,
) -> None:
    """Save finite eigenvalues and directly calculated far-field metrics."""
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "mode", "detuning_cm-1", "threshold_alpha_cm-1", "alpha_L",
            "band_overlap", "far_field_center_to_peak", "full_rms_divergence_deg",
            "peak_offset_deg", "centroid_offset_deg", "ellipticity",
            "encircled_power_0p5deg", "encircled_power_1deg",
            "requested_half_window_deg", "evaluated_half_window_deg", "angular_samples",
            "energy_normalization", "moment_reference", "discretization", "eigen_residual",
            "mesh_status", "empirical_grid_sensitivity_cm-1",
        ))
        for name in TRIANGULAR_MODE_NAMES:
            mode = modes[name]
            far = (far_fields[name] if far_fields is not None else
                   triangular_vector_far_field(mode, result.bragg_wavelength_nm, view_deg, samples))
            writer.writerow((
                name,
                mode.delta_per_m / 100.0,
                mode.alpha_per_m / 100.0,
                mode.alpha_l,
                mode.band_overlap,
                far.center_to_peak,
                far.full_rms_divergence_deg,
                far.peak_offset_deg,
                far.centroid_offset_deg,
                far.ellipticity,
                far.encircled_power_0p5deg,
                far.encircled_power_1deg,
                far.requested_view_deg, far.evaluated_view_deg, far.samples,
                far.energy_normalization, far.moment_reference, mode.discretization,
                mode.eigen_residual, mode.extrapolation_status,
                None if mode.extrapolation_uncertainty_per_m is None else mode.extrapolation_uncertainty_per_m / 100,
            ))


def plot_triangular_thresholds(
    modes: dict[str, TriangularFiniteMode], path: Path
) -> None:
    names = list(TRIANGULAR_MODE_NAMES)
    losses = [modes[name].alpha_per_m / 100.0 for name in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(names, losses, color="#4c78a8")
    for bar, mode in zip(bars, (modes[name] for name in names), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{mode.alpha_l:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set(
        ylabel=r"cold-cavity amplitude threshold $\alpha$ (cm$^{-1}$)",
        title=r"Finite triangular device; labels show $\alpha L$",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_triangular_mode_atlas(
    modes: dict[str, TriangularFiniteMode],
    result: TriangularCouplingResult,
    path: Path,
    view_deg: float = 3.0,
    samples: int = 161,
    far_fields: dict[str, TriangularFarField] | None = None,
) -> None:
    """Plot whole-device envelope, one-cell state, and physical vector FFP."""
    fig, axes = plt.subplots(6, 3, figsize=(11.6, 18.0), constrained_layout=True)
    for row, name in enumerate(TRIANGULAR_MODE_NAMES):
        mode = modes[name]
        _tripcolor(axes[row, 0], mode)
        axes[row, 0].set(
            title=f"{name}: whole-device envelope",
            xlabel="x (um)",
            ylabel="y (um)",
        )
        central = int(np.argmin(mode.grid.x_um ** 2 + mode.grid.y_um ** 2))
        coefficients = mode.fields[:, central]
        x, y, cell_intensity, _, _ = result.unit_cell_fields(coefficients, points=81)
        axes[row, 1].pcolormesh(x, y, cell_intensity, shading="auto", cmap="turbo")
        axes[row, 1].set_aspect("equal")
        axes[row, 1].set(
            title=f"{name}: one-cell intensity",
            xlabel="x/a",
            ylabel="y/a",
        )
        far = (far_fields[name] if far_fields is not None else triangular_vector_far_field(
            mode, result.bragg_wavelength_nm, view_deg=view_deg, samples=samples))
        extent = (
            far.angle_deg[0], far.angle_deg[-1],
            far.angle_deg[0], far.angle_deg[-1],
        )
        axes[row, 2].imshow(
            far.power,
            origin="lower",
            extent=extent,
            cmap="hot",
            norm=PowerNorm(gamma=0.65, vmin=0.0, vmax=1.0),
        )
        axes[row, 2].set(
            title=(
                f"{name}: FFP C={far.center_to_peak:.2f}, "
                f"O={far.peak_offset_deg:.2f} deg, "
                f"E1={far.encircled_power_1deg:.2f}"
            ),
            xlabel="theta_x (deg)",
            ylabel="theta_y (deg)",
        )
    window = far.evaluated_view_deg
    fig.suptitle(f"Six-wave selected families; FFP +/-{window:.4g} deg, {far.samples} samples; energy window-normalized")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_triangular_far_field_diagnostics(
    mode: TriangularFiniteMode,
    result: TriangularCouplingResult,
    path: Path,
    view_deg: float = 3.0,
    samples: int = 161,
    far: TriangularFarField | None = None,
) -> None:
    """Plot total and polarized far fields plus phase/polarization axes."""
    far = far if far is not None else triangular_vector_far_field(
        mode, result.bragg_wavelength_nm, view_deg=view_deg, samples=samples)
    total = far.power
    phase = np.angle(far.field_x + 1j * far.field_y)
    orientation = polarization_axis_angle(far.field_x, far.field_y)
    extent = (
        far.angle_deg[0], far.angle_deg[-1],
        far.angle_deg[0], far.angle_deg[-1],
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2))
    for ax, values, title in (
        (axes[0, 0], total, "total |Fx|^2+|Fy|^2"),
        (axes[0, 1], far.power_x, "x-polarized |Fx|^2"),
        (axes[1, 0], far.power_y, "y-polarized |Fy|^2"),
    ):
        ax.imshow(
            values, origin="lower", extent=extent, cmap="hot",
            norm=PowerNorm(gamma=0.65, vmin=0.0, vmax=1.0),
        )
        ax.set_title(title)
    axes[1, 1].imshow(
        phase, origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    stride = max(1, len(far.angle_deg) // 13)
    xgrid, ygrid = np.meshgrid(far.angle_deg, far.angle_deg)
    amplitude = np.sqrt(total)
    axes[1, 1].quiver(
        xgrid[::stride, ::stride],
        ygrid[::stride, ::stride],
        amplitude[::stride, ::stride] * np.cos(orientation[::stride, ::stride]),
        amplitude[::stride, ::stride] * np.sin(orientation[::stride, ::stride]),
        color="white",
        pivot="middle",
        scale=10.0,
        width=0.004,
    )
    axes[1, 1].set_title("phase of Fx+iFy and polarization axes")
    for ax in axes.flat:
        ax.set(xlabel="theta_x (deg)", ylabel="theta_y (deg)")
    fig.suptitle(
        f"Mode {mode.name}: center/max={far.center_to_peak:.3f}, "
        f"peak offset={far.peak_offset_deg:.3f} deg, "
        f"E1={far.encircled_power_1deg:.3f}, "
        f"ellipticity={far.ellipticity:.3f}"
        f"\n+/-{far.evaluated_view_deg:.4g} deg, {far.samples} samples; window-normalized energy; radial RMS diameter"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def polarization_axis_angle(field_x: np.ndarray, field_y: np.ndarray) -> np.ndarray:
    """Physical Stokes axis, invariant to common complex field normalization."""
    s1 = np.abs(field_x)**2 - np.abs(field_y)**2
    s2 = 2.0 * np.real(field_x * np.conj(field_y))
    return 0.5 * np.arctan2(s2, s1)

