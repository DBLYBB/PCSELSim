"""Spatial quasi-three-level Yb:YAG pump/gain closure for an in-plane PCSEL.

The caller supplies the *same* PCSEL mode loss, upward/downward radiation
losses and its real longitudinal energy fractions. This module does not add a
Fabry-Perot longitudinal laser cavity. It connects one modal photon reservoir
to spatial inversion and depleted pump beams in that existing PCSEL.

中文：连接面内反馈PCSEL的同一冷腔损耗、上下辐射和真实纵向场权重。上下泵沿z，
双端侧泵沿x并在入口厚度方向均匀；侧泵保留f(x,z)，横向y方向暂均匀。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.constants import c, h
from scipy.optimize import brentq

from .solid_state import YbYAGMediumConfig


@dataclass(frozen=True)
class SpatialYbYAGModel:
    medium: YbYAGMediumConfig
    cell_widths_m: np.ndarray
    optical_energy_fractions: np.ndarray
    total_power_loss_per_m: float
    upward_power_loss_per_m: float = 0.0
    downward_power_loss_per_m: float = 0.0
    group_velocity_m_s: float | None = None
    optical_gain_weights: np.ndarray | None = None
    lateral_energy_fractions: np.ndarray | None = None

    def __post_init__(self) -> None:
        dz = np.asarray(self.cell_widths_m, dtype=float).copy()
        q = np.asarray(self.optical_energy_fractions, dtype=float).copy()
        if (dz.ndim != 1 or len(dz) == 0 or q.shape != dz.shape
                or not np.isfinite(dz).all() or not np.isfinite(q).all()
                or np.any(dz <= 0.0) or np.any(q < 0.0) or not 0.0 < q.sum() <= 1.0 + 1e-9):
            raise ValueError("Positive active-cell widths and full-mode energy fractions with 0 < sum(q) <= 1 are required")
        loss = np.array([self.total_power_loss_per_m, self.upward_power_loss_per_m,
                         self.downward_power_loss_per_m])
        if (not np.isfinite(loss).all() or np.any(loss < 0.0)
                or self.total_power_loss_per_m <= 0.0
                or self.upward_power_loss_per_m + self.downward_power_loss_per_m > self.total_power_loss_per_m * (1.0 + 1e-10)):
            raise ValueError("Require 0 <= upward+downward losses <= positive total power loss")
        velocity = c / self.medium.refractive_index if self.group_velocity_m_s is None else self.group_velocity_m_s
        if not np.isfinite(velocity) or not 0.0 < velocity <= c:
            raise ValueError("group_velocity_m_s must be positive and no greater than c")
        # Energy residence q_energy is not bulk intensity gain confinement.
        # Uniform core: (c/n_core)*q_energy = vg*q_gain, hence
        # q_gain=(n_g/n_core)*q_energy. For a nonuniform gain index the caller
        # must pass explicitly integrated q_gain values from its optical model.
        gain_weights = (
            c / (self.medium.refractive_index * velocity) * q
            if self.optical_gain_weights is None
            else np.asarray(self.optical_gain_weights, dtype=float).copy()
        )
        if (gain_weights.shape != q.shape or not np.isfinite(gain_weights).all()
                or np.any(gain_weights < 0.0) or gain_weights.sum() <= 0.0
                or np.any((q == 0.0) & (gain_weights > 0.0))):
            raise ValueError("optical_gain_weights must be nonnegative with support inside the optical energy fractions")
        dz.setflags(write=False)
        q.setflags(write=False)
        gain_weights.setflags(write=False)
        lateral = None
        if self.lateral_energy_fractions is not None:
            lateral = np.asarray(self.lateral_energy_fractions, dtype=float).copy()
            if (lateral.ndim != 1 or lateral.size == 0 or not np.isfinite(lateral).all()
                    or np.any(lateral < 0.0) or not np.isclose(lateral.sum(), 1.0, rtol=1e-10, atol=1e-12)):
                raise ValueError("lateral_energy_fractions must be nonnegative cell-integrated fractions summing to one")
            lateral.setflags(write=False)
        object.__setattr__(self, "cell_widths_m", dz)
        object.__setattr__(self, "optical_energy_fractions", q)
        object.__setattr__(self, "group_velocity_m_s", float(velocity))
        object.__setattr__(self, "optical_gain_weights", gain_weights)
        object.__setattr__(self, "lateral_energy_fractions", lateral)

    @property
    def gain_thickness_m(self) -> float:
        return float(self.cell_widths_m.sum())

    @property
    def gain_volume_m3(self) -> float:
        return self.medium.pumped_area_um2 * 1e-12 * self.gain_thickness_m


@dataclass(frozen=True)
class SpatialPumpConfig:
    direction: str = "normal"
    forward_intensity_W_cm2: float = 2.5e4
    backward_intensity_W_cm2: float = 0.0
    edge_length_um: float = 500.0
    edge_cells: int = 41

    def __post_init__(self) -> None:
        if self.direction not in ("normal", "edge"):
            raise ValueError("direction must be normal (bottom/top) or edge (left/right)")
        if (not np.isfinite([self.forward_intensity_W_cm2, self.backward_intensity_W_cm2,
                             self.edge_length_um]).all()
                or min(self.forward_intensity_W_cm2, self.backward_intensity_W_cm2) < 0.0
                or self.edge_length_um <= 0.0 or not isinstance(self.edge_cells, (int, np.integer))
                or self.edge_cells < 1):
            raise ValueError("Pump intensities must be nonnegative and edge length/cell count positive")


@dataclass(frozen=True)
class SpatialYbYAGResult:
    excited_fraction: np.ndarray  # (x cells, active z cells)
    pump_intensity_W_cm2: np.ndarray  # local cell average, forward+backward
    pump_forward_W_cm2: np.ndarray
    pump_backward_W_cm2: np.ndarray
    photon_density_m3: float  # N_photon/V_gain
    modal_gain_per_m: float
    unlased_modal_gain_per_m: float
    lasing_gain_condition: bool
    threshold_pump_scale: float | None
    energy_budget_W: dict[str, float]
    population_residual_per_s: float
    photon_fraction_residual_per_s: float
    iterations: int


def _constants(model: SpatialYbYAGModel) -> dict[str, float]:
    medium = model.medium
    return {
        "nt": medium.dopant_density_cm3 * 1e6,
        "ap": medium.pump_absorption_cross_section_cm2 * 1e-4,
        "ep": medium.pump_emission_cross_section_cm2 * 1e-4,
        "al": medium.laser_absorption_cross_section_cm2 * 1e-4,
        "el": medium.laser_emission_cross_section_cm2 * 1e-4,
        "tau": medium.upper_state_lifetime_ms * 1e-3,
        "pump_energy": h * c / (medium.pump_wavelength_nm * 1e-9),
        "laser_energy": h * c / (medium.laser_wavelength_nm * 1e-9),
    }


def spatial_volume_and_gain_weights(model: SpatialYbYAGModel, pump: SpatialPumpConfig):
    """Volume fractions sum to one; gain weights keep their full-mode norm.

    These weights are q_gain, not energy fractions. Their sum is the effective
    bulk-to-modal gain overlap and need not equal Gamma_energy or one.
    """
    nx = pump.edge_cells if pump.direction == "edge" else 1
    volume = np.broadcast_to(model.cell_widths_m / model.gain_thickness_m / nx,
                             (nx, len(model.cell_widths_m))).copy()
    lateral = model.lateral_energy_fractions
    if lateral is None:
        lateral = np.full(nx, 1.0 / nx)
    elif lateral.size != nx:
        raise ValueError("lateral_energy_fractions length must match pump.edge_cells, or one for normal pumping")
    gain_weights = lateral[:, None] * model.optical_gain_weights[None, :]
    return volume, gain_weights


def _cell_average_factor(optical_depth: np.ndarray) -> np.ndarray:
    result = np.ones_like(optical_depth)
    mask = np.abs(optical_depth) > 1e-8
    np.divide(-np.expm1(-optical_depth), optical_depth, out=result, where=mask)
    result[~mask] = 1.0 - optical_depth[~mask] / 2.0 + optical_depth[~mask]**2 / 6.0
    return result


def propagate_spatial_pump(model: SpatialYbYAGModel, pump: SpatialPumpConfig,
                           excited_fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Exact constant-in-cell Beer transport for a supplied inversion.

    Intensities are those entering the gain medium AFTER interface/DBR pump
    transmission. A reflected pump must be included by the optical caller's
    boundary conditions, not silently counted as an independent extra source.
    """
    volume_weights, _ = spatial_volume_and_gain_weights(model, pump)
    fraction = np.asarray(excited_fraction, dtype=float)
    if fraction.shape != volume_weights.shape or not np.isfinite(fraction).all() or np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ValueError("excited_fraction must be a finite (nx,nz) array in [0,1]")
    values = _constants(model)
    alpha = values["nt"] * (values["ap"] * (1.0 - fraction) - values["ep"] * fraction)
    area = model.medium.pumped_area_um2 * 1e-12
    forward_input = pump.forward_intensity_W_cm2 * 1e4
    backward_input = pump.backward_intensity_W_cm2 * 1e4
    if pump.direction == "normal":
        optical_depth = alpha[0] * model.cell_widths_m
        transmission = np.exp(-optical_depth)
        factor = _cell_average_factor(optical_depth)
        forward_edges = np.empty(len(optical_depth) + 1)
        backward_edges = np.empty_like(forward_edges)
        forward_edges[0] = forward_input
        backward_edges[-1] = backward_input
        for index in range(len(optical_depth)):
            forward_edges[index + 1] = forward_edges[index] * transmission[index]
        for index in range(len(optical_depth) - 1, -1, -1):
            backward_edges[index] = backward_edges[index + 1] * transmission[index]
        forward = (forward_edges[:-1] * factor)[None, :]
        backward = (backward_edges[1:] * factor)[None, :]
        incoming = area * (forward_input + backward_input)
        absorbed = incoming - area * (forward_edges[-1] + backward_edges[0])
    else:
        dx = pump.edge_length_um * 1e-6 / pump.edge_cells
        optical_depth = alpha * dx
        transmission = np.exp(-optical_depth)
        factor = _cell_average_factor(optical_depth)
        nx, nz = alpha.shape
        forward_edges = np.empty((nx + 1, nz))
        backward_edges = np.empty_like(forward_edges)
        forward_edges[0] = forward_input
        backward_edges[-1] = backward_input
        for index in range(nx):
            forward_edges[index + 1] = forward_edges[index] * transmission[index]
        for index in range(nx - 1, -1, -1):
            backward_edges[index] = backward_edges[index + 1] * transmission[index]
        forward = forward_edges[:-1] * factor
        backward = backward_edges[1:] * factor
        width = area / (pump.edge_length_um * 1e-6)
        entrance_area = width * model.gain_thickness_m
        incoming = entrance_area * (forward_input + backward_input)
        absorbed = incoming - width * float(np.sum(model.cell_widths_m * (
            forward_edges[-1] + backward_edges[0]
        )))
    return forward / 1e4, backward / 1e4, float(incoming), float(absorbed)


def spatial_ybyag_rate_rhs(model: SpatialYbYAGModel, pump: SpatialPumpConfig,
                           fraction: np.ndarray, photon_density_m3: float):
    """Instantaneous pump-depleted rates and power budget, including storage."""
    if not np.isfinite(photon_density_m3) or photon_density_m3 < 0.0:
        raise ValueError("photon_density_m3 must be finite and nonnegative")
    fraction = np.asarray(fraction, dtype=float)
    values = _constants(model)
    volume_weights, gain_weights = spatial_volume_and_gain_weights(model, pump)
    forward, backward, incident, absorbed = propagate_spatial_pump(model, pump, fraction)
    flux = (forward + backward) * 1e4 / values["pump_energy"]
    gain = values["nt"] * ((values["el"] + values["al"]) * fraction - values["al"])
    local_gain_factor = gain_weights / volume_weights
    stimulated_local = model.group_velocity_m_s * gain * photon_density_m3 * local_gain_factor
    df = (values["ap"] * flux * (1.0 - fraction) - values["ep"] * flux * fraction
          - fraction / values["tau"] - stimulated_local / values["nt"])
    modal_gain = float(np.sum(gain_weights * gain))
    mean_fraction = float(np.sum(volume_weights * fraction))
    spontaneous = model.medium.spontaneous_emission_factor * values["nt"] * mean_fraction / values["tau"]
    dphotons = model.group_velocity_m_s * (modal_gain - model.total_power_loss_per_m) * photon_density_m3 + spontaneous
    energy_per_density = values["laser_energy"] * model.gain_volume_m3
    loss_power = energy_per_density * model.group_velocity_m_s * model.total_power_loss_per_m * photon_density_m3
    upward = energy_per_density * model.group_velocity_m_s * model.upward_power_loss_per_m * photon_density_m3
    downward = energy_per_density * model.group_velocity_m_s * model.downward_power_loss_per_m * photon_density_m3
    fluorescence = energy_per_density * (1.0 - model.medium.spontaneous_emission_factor) * values["nt"] * mean_fraction / values["tau"]
    storage = energy_per_density * (values["nt"] * float(np.sum(volume_weights * df)) + dphotons)
    source = absorbed * values["laser_energy"] / values["pump_energy"]
    budget = {
        "incident_internal_pump_W": incident,
        "absorbed_pump_W": absorbed,
        "quantum_defect_heat_W": absorbed - source,
        "fluorescence_outside_mode_W": fluorescence,
        "upward_output_W": upward,
        "downward_output_W": downward,
        "other_cavity_loss_W": loss_power - upward - downward,
        "stored_excitation_plus_photon_energy_rate_W": storage,
        "energy_closure_residual_W": storage - source + fluorescence + loss_power,
    }
    return df, float(dphotons), modal_gain, budget


def _steady_inversion(model: SpatialYbYAGModel, pump: SpatialPumpConfig,
                      photons: float, initial: np.ndarray | None = None):
    values = _constants(model)
    volume, weights = spatial_volume_and_gain_weights(model, pump)
    fraction = np.zeros_like(volume) if initial is None else np.asarray(initial).copy()
    local_photons = photons * weights / volume
    stimulated = model.group_velocity_m_s * local_photons
    for iteration in range(1, 1001):
        forward, backward, _, _ = propagate_spatial_pump(model, pump, fraction)
        flux = (forward + backward) * 1e4 / values["pump_energy"]
        updated = (values["ap"] * flux + stimulated * values["al"]) / (
            (values["ap"] + values["ep"]) * flux + 1.0 / values["tau"]
            + stimulated * (values["el"] + values["al"])
        )
        difference = float(np.max(np.abs(updated - fraction)))
        fraction = 0.65 * updated + 0.35 * fraction
        if difference < 2e-12:
            return fraction, iteration
    raise RuntimeError("Spatial pump/inversion fixed point did not converge")


def find_spatial_threshold_pump_scale(model: SpatialYbYAGModel, pump: SpatialPumpConfig,
                                     max_scale: float = 1e7) -> float | None:
    """Scale both incident beams at fixed split; None means no bracketed gain threshold."""
    if not np.isfinite(max_scale) or max_scale <= 0.0:
        raise ValueError("max_scale must be finite and positive")
    values = _constants(model)
    ceiling = values["ap"] / (values["ap"] + values["ep"])
    ceiling_gain = float(model.optical_gain_weights.sum() * values["nt"] * (
        (values["el"] + values["al"]) * ceiling - values["al"]
    ))
    if ceiling_gain <= model.total_power_loss_per_m or pump.forward_intensity_W_cm2 + pump.backward_intensity_W_cm2 == 0.0:
        return None

    def gap(scale: float) -> float:
        scaled = replace(pump, forward_intensity_W_cm2=pump.forward_intensity_W_cm2 * scale,
                         backward_intensity_W_cm2=pump.backward_intensity_W_cm2 * scale)
        fraction, _ = _steady_inversion(model, scaled, 0.0)
        _, _, weights_gain, _ = spatial_ybyag_rate_rhs(model, scaled, fraction, 0.0)
        return weights_gain - model.total_power_loss_per_m

    upper = min(1.0, max_scale)
    while gap(upper) <= 0.0 and upper < max_scale:
        upper = min(2.0 * upper, max_scale)
    if gap(upper) <= 0.0:
        return None
    return float(brentq(gap, 0.0, upper, xtol=1e-9, rtol=1e-9))


def solve_spatial_ybyag_steady(model: SpatialYbYAGModel, pump: SpatialPumpConfig,
                               find_threshold: bool = True) -> SpatialYbYAGResult:
    """Solve local inversion/pump depletion and one shared modal photon pool.

    Finite beta produces subthreshold spontaneous-mode light. The explicit
    lasing_gain_condition uses the zero-photon pump-limited gain, not P>0.
    No extra confinement_factor is taken from medium: q_gain already contains
    the longitudinal energy overlap and local/group-velocity conversion.
    """
    values = _constants(model)
    unlased, iterations = _steady_inversion(model, pump, 0.0)
    _, _, unlased_gain, _ = spatial_ybyag_rate_rhs(model, pump, unlased, 0.0)
    can_lase = unlased_gain > model.total_power_loss_per_m

    def photon_balance(photon_fraction: float) -> float:
        nonlocal iterations
        fraction, count = _steady_inversion(model, pump, photon_fraction * values["nt"], unlased)
        iterations += count
        _, dphotons, _, _ = spatial_ybyag_rate_rhs(model, pump, fraction, photon_fraction * values["nt"])
        return dphotons / values["nt"]

    beta = model.medium.spontaneous_emission_factor
    if not can_lase and beta == 0.0:
        photon_fraction = 0.0
    elif can_lase and beta == 0.0:
        def gain_gap(photon_fraction: float) -> float:
            fraction, _ = _steady_inversion(model, pump, photon_fraction * values["nt"], unlased)
            return spatial_ybyag_rate_rhs(model, pump, fraction, photon_fraction * values["nt"])[2] - model.total_power_loss_per_m
        upper = 1e-10
        while gain_gap(upper) > 0.0:
            upper *= 10.0
            if upper > 1e3:
                raise RuntimeError("No finite photon saturation bracket")
        photon_fraction = float(brentq(gain_gap, 0.0, upper, xtol=1e-30, rtol=2e-10))
    else:
        upper = 1e-24
        while photon_balance(upper) > 0.0:
            upper *= 10.0
            if upper > 1e3:
                raise RuntimeError("No finite photon-balance bracket")
        photon_fraction = float(brentq(photon_balance, 0.0, upper, xtol=1e-30, rtol=2e-10))
    photons = photon_fraction * values["nt"]
    fraction, count = _steady_inversion(model, pump, photons, unlased)
    iterations += count
    forward, backward, _, _ = propagate_spatial_pump(model, pump, fraction)
    df, dphotons, gain, budget = spatial_ybyag_rate_rhs(model, pump, fraction, photons)
    return SpatialYbYAGResult(
        excited_fraction=fraction, pump_intensity_W_cm2=forward + backward,
        pump_forward_W_cm2=forward, pump_backward_W_cm2=backward,
        photon_density_m3=photons, modal_gain_per_m=gain,
        unlased_modal_gain_per_m=unlased_gain, lasing_gain_condition=can_lase,
        threshold_pump_scale=find_spatial_threshold_pump_scale(model, pump) if find_threshold else None,
        energy_budget_W=budget, population_residual_per_s=float(np.max(np.abs(df))),
        photon_fraction_residual_per_s=dphotons / values["nt"], iterations=iterations,
    )
