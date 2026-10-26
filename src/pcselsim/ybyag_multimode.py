"""Shared-inversion, depleted-pump CW competition of supplied PCSEL modes.

Each mode has its own cold-cavity frequency, full-mode gain weights, loss,
velocity and spectroscopic cross sections. They exchange excitations with the
same x-z Yb reservoir. y remains averaged and coherent population gratings,
gain-induced field/frequency changes and dynamical stability are not solved.

中文：这是固定冷腔模式上的静态 CW 互补求解。各模式共同消耗 x-z 反转库，
y 方向已平均；活动模式数不能据此直接解释为时域稳定性或实验单模保证。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import c, h
from scipy.optimize import least_squares, root

from .ybyag_spatial_rates import (
    SpatialPumpConfig, SpatialYbYAGModel, _constants,
    propagate_spatial_pump, spatial_volume_and_gain_weights,
)


@dataclass(frozen=True)
class SpatialMultimodeYbYAG:
    modes: tuple[SpatialYbYAGModel, ...]
    names: tuple[str, ...]

    def __post_init__(self):
        # Shared inversion is meaningful only for identical physical gain cells.
        # 共享反转要求相同的物理增益网格和泵浦介质，不能拼接独立单模器件。
        if not self.modes or len(self.modes) != len(self.names) or len(set(self.names)) != len(self.names):
            raise ValueError("Unique names and at least one spatial mode are required")
        first = self.modes[0]
        shared = ("dopant_density_cm3", "upper_state_lifetime_ms", "pump_wavelength_nm",
                  "pump_absorption_cross_section_cm2", "pump_emission_cross_section_cm2", "pumped_area_um2")
        for model in self.modes:
            if (not np.array_equal(model.cell_widths_m, first.cell_widths_m)
                    or any(getattr(model.medium, key) != getattr(first.medium, key) for key in shared)):
                raise ValueError("Competing modes must share the physical gain cells and pump medium")
            if model.medium.spontaneous_emission_factor != 0:
                raise ValueError("CW complementarity currently requires beta=0; finite-beta competition is not implemented")


@dataclass(frozen=True)
class SpatialMultimodeResult:
    names: tuple[str, ...]
    excited_fraction: np.ndarray
    photon_density_m3: np.ndarray
    modal_gain_per_m: np.ndarray
    unlased_modal_gain_per_m: np.ndarray
    upward_output_W: np.ndarray
    downward_output_W: np.ndarray
    active_modes: tuple[str, ...]
    energy_budget_W: dict[str, float]
    population_residual_per_s: float
    complementarity_residual: float
    maximum_inactive_gain_excess_per_m: float
    iterations: int


def _arrays(system: SpatialMultimodeYbYAG, pump: SpatialPumpConfig):
    # Geometry adapters retain x-z cells even when their rays run along z.
    # 即使顶泵光线沿 z 传播，反转库仍使用相同 x-z 网格，y 保持平均。
    weights_pump = (pump if isinstance(pump, SpatialPumpConfig) else
                    SpatialPumpConfig("edge", 0, 0, pump.device_side_um, pump.x_cells))
    volume, _ = spatial_volume_and_gain_weights(system.modes[0], weights_pump)
    weights = np.array([spatial_volume_and_gain_weights(m, weights_pump)[1] for m in system.modes])
    constants = [_constants(m) for m in system.modes]
    return (volume, weights, constants,
            np.array([m.group_velocity_m_s for m in system.modes]),
            np.array([m.total_power_loss_per_m for m in system.modes]))


def _pump_flow(system, pump, fraction):
    if isinstance(pump, SpatialPumpConfig):
        forward, backward, incident, absorbed = propagate_spatial_pump(system.modes[0], pump, fraction)
        return forward, backward, incident, absorbed, dict(pump_escape_W=incident-absorbed, pump_relay_loss_W=0.)
    from .ybyag_pump_geometry import propagate_geometry_pump
    result = propagate_geometry_pump(system.modes[0], pump, fraction)
    return (result.forward_W_cm2, result.backward_W_cm2, result.incident_internal_power_W,
            result.absorbed_power_W, dict(pump_escape_W=result.escaped_pump_power_W,
                                         pump_relay_loss_W=result.relay_loss_power_W,
                                         pump_flux_residual_W=result.pump_flux_residual_W))


def _inversion(system, pump, photons, initial=None):
    volume, weights, constants, velocities, _ = _arrays(system, pump)
    common = constants[0]
    fraction = np.zeros_like(volume) if initial is None else initial.copy()
    stimulated = velocities[:, None, None] * photons[:, None, None] * weights / volume
    absorption = np.array([v["al"] for v in constants])[:, None, None]
    emission = np.array([v["el"] for v in constants])[:, None, None]
    for iteration in range(1, 1501):
        forward, backward, _, _, _ = _pump_flow(system, pump, fraction)
        # Convert W/cm^2 to W/m^2, then divide by photon energy for photon flux.
        # 泵浦强度先乘 1e4 转为 SI 面密度，再除单光子能量；截面在常数层统一换算。
        flux = (forward + backward) * 1e4 / common["pump_energy"]
        updated = (common["ap"] * flux + np.sum(stimulated * absorption, axis=0)) / (
            (common["ap"] + common["ep"]) * flux + 1 / common["tau"]
            + np.sum(stimulated * (absorption + emission), axis=0))
        difference = np.max(abs(updated - fraction))
        fraction = .65 * updated + .35 * fraction
        if difference < 3e-12:
            return fraction, iteration
    raise RuntimeError("Shared multimode pump/inversion iteration did not converge")


def multimode_spatial_rhs(system: SpatialMultimodeYbYAG, pump: SpatialPumpConfig,
                         fraction: np.ndarray, photons: np.ndarray):
    """Rates and exact excitation-count / effective-energy bookkeeping.

    A common excitation energy h*nu_ref uses the first mode wavelength. The
    frequency difference between that reference and each emitted/absorbed laser
    photon appears as an explicit signed thermalization term. Thus modes at
    different wavelengths do not silently create energy in the budget.

    中文：反转按共同参考激发能记账；不同模式光子能量的差异显式记入热化项。
    入射、吸收、逃逸泵浦与腔损耗分别保存，不能把全部入射泵浦等同于发热。
    """
    photons = np.asarray(photons, float)
    if photons.shape != (len(system.modes),) or not np.isfinite(photons).all() or np.any(photons < 0):
        raise ValueError("One finite nonnegative photon density per mode is required")
    volume, weights, constants, velocity, loss = _arrays(system, pump)
    common = constants[0]
    forward, backward, incident, absorbed, transport_budget = _pump_flow(system, pump, fraction)
    flux = (forward + backward) * 1e4 / common["pump_energy"]
    al = np.array([v["al"] for v in constants])[:, None, None]
    el = np.array([v["el"] for v in constants])[:, None, None]
    local_gain = common["nt"] * ((el + al) * fraction[None, ...] - al)
    stimulated = velocity[:, None, None] * photons[:, None, None] * weights * local_gain / volume
    df = (common["ap"] * flux * (1 - fraction) - common["ep"] * flux * fraction
          - fraction / common["tau"] - stimulated.sum(axis=0) / common["nt"])
    gain = np.sum(weights * local_gain, axis=(1, 2))
    ds = velocity * (gain - loss) * photons
    energies = np.array([v["laser_energy"] for v in constants])
    reference_energy = energies[0]
    gain_volume = system.modes[0].gain_volume_m3
    mean_fraction = float(np.sum(volume * fraction))
    cavity = energies * gain_volume * velocity * loss * photons
    upward = energies * gain_volume * velocity * photons * np.array([m.upward_power_loss_per_m for m in system.modes])
    downward = energies * gain_volume * velocity * photons * np.array([m.downward_power_loss_per_m for m in system.modes])
    fluorescence = reference_energy * gain_volume * common["nt"] * mean_fraction / common["tau"]
    stimulated_counts = gain_volume * velocity * gain * photons
    thermalization = float(np.sum((reference_energy - energies) * stimulated_counts))
    storage = gain_volume * (reference_energy * common["nt"] * np.sum(volume * df) + energies @ ds)
    source = absorbed * reference_energy / common["pump_energy"]
    budget = dict(incident_internal_pump_W=incident, absorbed_pump_W=absorbed,
                  pump_quantum_defect_reference_W=absorbed - source,
                  intermode_stimulated_thermalization_W=thermalization,
                  fluorescence_outside_modes_W=float(fluorescence), upward_output_W=float(upward.sum()),
                  downward_output_W=float(downward.sum()), other_cavity_loss_W=float((cavity-upward-downward).sum()),
                  stored_excitation_plus_photon_energy_rate_W=float(storage),
                  energy_closure_residual_W=float(storage - source + fluorescence + cavity.sum() + thermalization))
    budget.update(transport_budget)
    budget["pump_boundary_flow_residual_W"] = incident-absorbed-budget["pump_escape_W"]-budget["pump_relay_loss_W"]
    return df, ds, gain, upward, downward, budget


def solve_multimode_spatial_ybyag(system: SpatialMultimodeYbYAG, pump: SpatialPumpConfig,
                                *, tolerance: float = 2e-7) -> SpatialMultimodeResult:
    """Solve S>=0, loss-G>=0, S*(loss-G)=0 with a shared inversion.

    A small residual certifies the static gain complementarity of the supplied
    branches only. It does not establish dynamical or coherent single-mode
    stability. No modal power is obtained by adding independent single-mode CW
    predictions; all returned modes deplete the same population simultaneously.

    中文：S 为非负光子密度，未活动模允许增益低于损耗，活动模要求增益钳位。
    小残差只验证所提供模式集合的静态互补条件，不验证时域振荡或相干稳定性。
    """
    _, _, constants, _, loss = _arrays(system, pump)
    nt = constants[0]["nt"]
    unlased, iterations = _inversion(system, pump, np.zeros(len(system.modes)))
    unlased_gain = multimode_spatial_rhs(system, pump, unlased, np.zeros(len(system.modes)))[2]
    scale = nt * 1e-8

    def gain_for(x):
        nonlocal iterations
        fraction, count = _inversion(system, pump, scale * x, unlased)
        iterations += count
        gain = multimode_spatial_rhs(system, pump, fraction, scale*x)[2]
        return gain

    def complementarity(x):
        # Fischer-Burmeister residual enforces both positivity and gain clamping.
        # 同时约束非负光子数和净增益余量，避免把各模独立输出相加。
        slack = (loss - gain_for(x)) / loss
        return np.hypot(x, slack) - x - slack

    if np.all(unlased_gain <= loss):
        photons = np.zeros(len(loss))
        fraction = unlased
        complementarity_residual = 0.0
    else:
        fit = least_squares(complementarity, np.full(len(loss), .1), bounds=(0, np.inf),
                            xtol=2e-12, ftol=2e-12, gtol=2e-12, max_nfev=1500)
        x = fit.x.copy()
        # Numerical inactive reservoirs must not be reported as lasing outputs.
        x[x < max(1e-7, x.max()*1e-7)] = 0
        # Bound-constrained least squares can stop a few micro-units above the
        # S=0 face despite a large negative net gain. Identify that face by its
        # gain slack before polishing the active subsystem; magnitude alone is
        # unreliable when photon reservoirs differ by many orders of magnitude.
        x[(loss-gain_for(x)) > 1e-5*loss] = 0
        active_indices = np.flatnonzero(x > 0)
        if len(active_indices):
            def active_balance(log_photons):
                candidate = np.zeros_like(x)
                candidate[active_indices] = np.exp(np.clip(log_photons, -50, 50))
                return (gain_for(candidate)[active_indices]-loss[active_indices])/loss[active_indices]
            polished = root(active_balance, np.log(x[active_indices]), tol=1e-10)
            if polished.success and np.max(abs(active_balance(polished.x))) < 1e-9:
                candidate = np.zeros_like(x)
                candidate[active_indices] = np.exp(np.clip(polished.x, -50, 50))
                if np.max(abs(complementarity(candidate))) < np.max(abs(complementarity(x))):
                    x = candidate
        complementarity_residual = float(np.max(abs(complementarity(x))))
        if not fit.success or complementarity_residual > tolerance:
            raise RuntimeError(f"Multimode CW complementarity not converged: {complementarity_residual:g}")
        photons = scale*x
        fraction, count = _inversion(system, pump, photons, unlased)
        iterations += count
    df, ds, gain, upward, downward, budget = multimode_spatial_rhs(system, pump, fraction, photons)
    active = photons > 0
    inactive_excess = float(max(0, np.max((gain-loss)[~active], initial=0)))
    return SpatialMultimodeResult(system.names, fraction, photons, gain, unlased_gain,
                                 upward, downward, tuple(np.asarray(system.names)[active]), budget,
                                 float(np.max(abs(df))), complementarity_residual, inactive_excess, iterations)
