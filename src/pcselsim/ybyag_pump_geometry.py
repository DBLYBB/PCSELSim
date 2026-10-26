"""Equal-total-power x-z pump comparisons for the existing crystal PCSEL.

This independent adapter preserves SpatialYbYAGModel's optical energy/gain
weights and its one modal photon reservoir S=Nph/Vgain. It does not replace
the old solver, alter the PCSEL cold cavity or introduce an FP laser cavity.

Pump control volumes use exact constant-in-cell Beer transport and its exact
cell-average intensity. Side pumping propagates along x; top pumping has an
independent downward z ray for every x column. y is still averaged. All three
architectures start from a specified *total internal incident power*, not an
intensity held constant while increasing the side entrance area with thickness.

中文：公平比较单端侧泵、双端侧泵和顶部泵。顶部可指定有限程数及返回效率，按非相干
理想重成像处理；不含真实DBR泵边界/驻波、孔层泵衍射、聚焦/NA或热效应。top R/T只换算入口
入口功率，不改变激光模式。增厚后光学场/损耗须由调用者重新求解，不能只改体积。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq

from .ybyag_spatial_rates import (
    SpatialPumpConfig,
    SpatialYbYAGModel,
    SpatialYbYAGResult,
    _cell_average_factor,
    _constants,
    propagate_spatial_pump,
    solve_spatial_ybyag_steady,
    spatial_volume_and_gain_weights,
    spatial_ybyag_rate_rhs,
)


@dataclass(frozen=True)
class PumpGeometryConfig:
    """Specified total internal pump W for a square L by L aperture.

    edge_single: all power enters the left L*t face.
    edge_dual: equal halves enter left/right L*t faces.
    top_single: first input enters the top L*L face and travels downward;
                optional ideal relay passes alternate downward/upward.
    For measured/assumed entrance optics use from_external_top(); external
    input does not imply reflected pump is recycled into the active medium.
    """

    architecture: str = "edge_dual"
    internal_incident_power_W: float = 32.0
    device_side_um: float = 1000.0
    x_cells: int = 41
    external_incident_power_W: float | None = None
    top_transmission: float = 1.0
    top_reflectance: float = 0.0
    top_passes: int = 1
    return_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if self.architecture not in ("edge_single", "edge_dual", "top_single"):
            raise ValueError("architecture must be edge_single, edge_dual or top_single")
        values = (self.internal_incident_power_W, self.device_side_um,
                  self.top_transmission, self.top_reflectance, self.return_efficiency)
        if (not np.isfinite(values).all() or self.internal_incident_power_W < 0.0
                or self.device_side_um <= 0.0 or not isinstance(self.x_cells, (int, np.integer))
                or self.x_cells < 1):
            raise ValueError("Pump power must be nonnegative; device side and integer x_cells positive")
        if (min(self.top_transmission, self.top_reflectance) < 0.0
                or self.top_transmission + self.top_reflectance > 1.0 + 1e-12):
            raise ValueError("Top power T/R must be nonnegative with T+R <= 1")
        if (not isinstance(self.top_passes, (int, np.integer)) or self.top_passes < 1
                or not 0.0 <= self.return_efficiency <= 1.0):
            raise ValueError("top_passes must be a positive integer and return_efficiency in [0,1]")
        if self.architecture != "top_single" and (
                self.external_incident_power_W is not None
                or self.top_transmission != 1.0 or self.top_reflectance != 0.0
                or self.top_passes != 1 or self.return_efficiency != 1.0):
            raise ValueError("Top R/T, finite passes and external-top input apply only to top_single")
        if self.external_incident_power_W is not None:
            if (not np.isfinite(self.external_incident_power_W)
                    or self.external_incident_power_W < 0.0
                    or not np.isclose(self.internal_incident_power_W,
                                      self.external_incident_power_W * self.top_transmission,
                                      rtol=1e-11, atol=1e-12)):
                raise ValueError("Internal top power must equal T times the specified external power")

    @classmethod
    def from_external_top(cls, external_power_W: float, T: float, R: float = 0.0,
                          *, device_side_um: float = 1000.0, x_cells: int = 41,
                          top_passes: int = 1, return_efficiency: float = 1.0):
        """Convert specified external pump to the first internal top-pass power.

        T/R are pump-frequency *power* fractions to the active-layer entrance,
        not field amplitudes or the laser-frequency DBR reflectivity.
        """
        return cls("top_single", external_power_W * T, device_side_um, x_cells,
                   external_power_W, T, R, top_passes, return_efficiency)


@dataclass(frozen=True)
class PumpGeometryTransport:
    forward_W_cm2: np.ndarray
    backward_W_cm2: np.ndarray
    incident_internal_power_W: float
    escaped_pump_power_W: float
    relay_loss_power_W: float
    absorbed_power_W: float
    control_volume_absorbed_power_W: float
    pump_flux_residual_W: float

    @property
    def intensity_W_cm2(self) -> np.ndarray:
        return self.forward_W_cm2 + self.backward_W_cm2


def _geometry_weights(model: SpatialYbYAGModel, pump: PumpGeometryConfig):
    # This zero-pump adapter selects exactly the same x-z volume/gain weights
    # as the original side-pump model, even when new pump rays run along z.
    area_um2 = model.medium.pumped_area_um2
    if not np.isclose(area_um2, pump.device_side_um**2, rtol=1e-10, atol=1e-9):
        raise ValueError("This comparison requires medium.pumped_area_um2 == device_side_um**2")
    zero_pump = SpatialPumpConfig(
        "edge", 0.0, 0.0, pump.device_side_um, pump.x_cells,
    )
    volume, gain = spatial_volume_and_gain_weights(model, zero_pump)
    return zero_pump, volume, gain


def pump_geometry_metadata(model: SpatialYbYAGModel, pump: PumpGeometryConfig) -> dict:
    """Explicit areas/intensities for reporting equal-power comparisons."""
    _geometry_weights(model, pump)
    side_m = pump.device_side_um * 1e-6
    face_area_m2 = side_m**2 if pump.architecture == "top_single" else side_m * model.gain_thickness_m
    beam_count = 2 if pump.architecture == "edge_dual" else 1
    intensity = pump.internal_incident_power_W / beam_count / face_area_m2 / 1e4
    return {
        "architecture": pump.architecture,
        "internal_incident_power_W": pump.internal_incident_power_W,
        "external_incident_power_W": pump.external_incident_power_W,
        "entrance_area_per_beam_m2": face_area_m2,
        "number_of_incident_beams": beam_count,
        "internal_intensity_per_beam_W_cm2": intensity,
        "intensity_reference": "first entrance only; circulated passes are not new incident sources",
        "active_volume_m3": model.gain_volume_m3,
        "gain_thickness_um": model.gain_thickness_m * 1e6,
        "top_transmission": pump.top_transmission,
        "top_reflectance": pump.top_reflectance,
        "top_passes": pump.top_passes,
        "return_efficiency": pump.return_efficiency,
        "pump_scope": "noncoherent_finite_pass_ideal_reimaging_no_real_DBR_pump_boundary_no_thermal_or_diffraction",
    }


def _edge_config(model: SpatialYbYAGModel, pump: PumpGeometryConfig) -> SpatialPumpConfig:
    meta = pump_geometry_metadata(model, pump)
    intensity = meta["internal_intensity_per_beam_W_cm2"]
    return SpatialPumpConfig(
        "edge", intensity, intensity if pump.architecture == "edge_dual" else 0.0,
        pump.device_side_um, pump.x_cells,
    )


def propagate_geometry_pump(model: SpatialYbYAGModel, pump: PumpGeometryConfig,
                            excited_fraction: np.ndarray) -> PumpGeometryTransport:
    """Control-volume pump flow for supplied f(x,z), including pump saturation.

    Exact cell averages make sum(alpha*I_average*V_cell) equal input-output
    power. The top beam is backward (downward in increasing-z coordinates).
    """
    _, volume, _ = _geometry_weights(model, pump)
    fraction = np.asarray(excited_fraction, dtype=float)
    if (fraction.shape != volume.shape or not np.isfinite(fraction).all()
            or np.any((fraction < 0.0) | (fraction > 1.0))):
        raise ValueError("excited_fraction must be a finite (x_cells,nz) array in [0,1]")
    constants = _constants(model)
    alpha = constants["nt"] * (constants["ap"] * (1.0 - fraction) - constants["ep"] * fraction)
    if pump.architecture.startswith("edge"):
        forward, backward, incident, absorbed = propagate_spatial_pump(
            model, _edge_config(model, pump), fraction,
        )
        escaped = incident - absorbed
        relay_loss = 0.0
    else:
        optical_depth = alpha * model.cell_widths_m[None, :]
        average_factor = _cell_average_factor(optical_depth)
        nx = fraction.shape[0]
        area_m2 = model.medium.pumped_area_um2 * 1e-12
        cumulative_depth = np.cumsum(optical_depth, axis=1)
        total_depth = cumulative_depth[:, -1]
        before_depth = np.column_stack((np.zeros(nx), cumulative_depth[:, :-1]))
        after_depth = total_depth[:, None] - cumulative_depth
        full_transmission = np.exp(-total_depth)
        downward_entrance_sum = np.zeros(nx)
        upward_entrance_sum = np.zeros(nx)
        # Ideal relay keeps each x ray in the same control-volume column.
        # Only its surviving power becomes the next pass, never an extra source.
        entrance = np.full(nx, pump.internal_incident_power_W / area_m2)
        relay_loss = 0.0
        for pass_number in range(pump.top_passes):
            if pass_number % 2 == 0:  # first/from-top ray travels downward
                downward_entrance_sum += entrance
            else:
                upward_entrance_sum += entrance
            # f is shared by all passes at a CW fixed-point evaluation, hence
            # each column's whole-film transmission is identical for all passes.
            # Only short nx arrays need a pass loop; no nested z traversal.
            exiting = entrance * full_transmission
            if pass_number + 1 < pump.top_passes:
                relay_loss += area_m2 / nx * float(exiting.sum()) * (1.0 - pump.return_efficiency)
                entrance = exiting * pump.return_efficiency
        backward = downward_entrance_sum[:, None] * np.exp(-after_depth) * average_factor / 1e4
        forward = upward_entrance_sum[:, None] * np.exp(-before_depth) * average_factor / 1e4
        incident = pump.internal_incident_power_W
        escaped = area_m2 / nx * float(exiting.sum())
        absorbed = incident - escaped - relay_loss
    control_absorbed = float(np.sum(alpha * (forward + backward) * 1e4 * volume)) * model.gain_volume_m3
    return PumpGeometryTransport(
        forward, backward, float(incident), float(escaped), float(relay_loss), float(absorbed), control_absorbed,
        float(absorbed - control_absorbed),
    )


def pump_geometry_rate_rhs(model: SpatialYbYAGModel, pump: PumpGeometryConfig,
                           excited_fraction: np.ndarray, photon_density_m3: float):
    """Reuse the original stimulated/photon physics; replace only pump flow.

    Calling its zero-pump RHS avoids maintaining a second stimulated-gain,
    spontaneous, photon-loss or fluorescence formula. Geometry-specific pump
    excitation and matching source/storage terms are then added explicitly.
    """
    zero_pump, volume, _ = _geometry_weights(model, pump)
    fraction = np.asarray(excited_fraction, dtype=float)
    df, dphotons, gain, original_budget = spatial_ybyag_rate_rhs(
        model, zero_pump, fraction, photon_density_m3,
    )
    transport = propagate_geometry_pump(model, pump, fraction)
    constants = _constants(model)
    flux = transport.intensity_W_cm2 * 1e4 / constants["pump_energy"]
    pump_excitation = flux * (constants["ap"] * (1.0 - fraction) - constants["ep"] * fraction)
    df = df + pump_excitation
    added_storage = (constants["laser_energy"] * model.gain_volume_m3 * constants["nt"]
                     * float(np.sum(volume * pump_excitation)))
    laser_equivalent_absorbed = transport.absorbed_power_W * constants["laser_energy"] / constants["pump_energy"]
    budget = dict(original_budget)
    budget.update({
        "incident_internal_pump_W": transport.incident_internal_power_W,
        "pump_escape_W": transport.escaped_pump_power_W,
        "pump_relay_loss_W": transport.relay_loss_power_W,
        "absorbed_pump_W": transport.absorbed_power_W,
        "control_volume_absorbed_pump_W": transport.control_volume_absorbed_power_W,
        "pump_flux_residual_W": transport.pump_flux_residual_W,
        "quantum_defect_heat_W": transport.absorbed_power_W - laser_equivalent_absorbed,
        "stored_excitation_plus_photon_energy_rate_W": original_budget["stored_excitation_plus_photon_energy_rate_W"] + added_storage,
        "energy_closure_residual_W": original_budget["energy_closure_residual_W"] + added_storage - laser_equivalent_absorbed,
    })
    budget["pump_boundary_flow_residual_W"] = (transport.incident_internal_power_W - transport.absorbed_power_W
                                              - transport.relay_loss_power_W - transport.escaped_pump_power_W)
    if pump.top_passes == 1:
        budget["escaped_single_pass_pump_W"] = transport.escaped_pump_power_W
    if pump.external_incident_power_W is not None:
        reflected = pump.external_incident_power_W * pump.top_reflectance
        entrance_loss = pump.external_incident_power_W * max(1.0 - pump.top_reflectance - pump.top_transmission, 0.0)
        budget.update({
            "incident_external_pump_W": pump.external_incident_power_W,
            "entrance_reflected_pump_W": reflected,
            "entrance_untransmitted_unassigned_pump_W": entrance_loss,
            "external_pump_flow_residual_W": pump.external_incident_power_W - reflected - entrance_loss
                                             - transport.escaped_pump_power_W - transport.absorbed_power_W
                                             - transport.relay_loss_power_W,
        })
    return df, dphotons, gain, budget


def _steady_geometry_inversion(model: SpatialYbYAGModel, pump: PumpGeometryConfig,
                               photons: float, initial: np.ndarray | None = None):
    _, volume, gain_weights = _geometry_weights(model, pump)
    constants = _constants(model)
    fraction = np.zeros_like(volume) if initial is None else np.asarray(initial).copy()
    stimulated = model.group_velocity_m_s * photons * gain_weights / volume
    for iteration in range(1, 1001):
        transport = propagate_geometry_pump(model, pump, fraction)
        flux = transport.intensity_W_cm2 * 1e4 / constants["pump_energy"]
        updated = (constants["ap"] * flux + stimulated * constants["al"]) / (
            (constants["ap"] + constants["ep"]) * flux + 1.0 / constants["tau"]
            + stimulated * (constants["al"] + constants["el"])
        )
        difference = float(np.max(np.abs(updated - fraction)))
        fraction = 0.65 * updated + 0.35 * fraction
        if difference < 2e-12:
            return fraction, iteration
    raise RuntimeError("x-z pump/inversion fixed point did not converge")


def _scaled_pump(pump: PumpGeometryConfig, scale: float) -> PumpGeometryConfig:
    external = None if pump.external_incident_power_W is None else pump.external_incident_power_W * scale
    return replace(pump, internal_incident_power_W=pump.internal_incident_power_W * scale,
                   external_incident_power_W=external)


def find_geometry_threshold_pump_power(model: SpatialYbYAGModel, pump: PumpGeometryConfig,
                                      max_power_W: float = 1e8) -> float | None:
    """Return total INTERNAL entrance W at zero-photon gain threshold.

    Keep geometry/split and entrance T/R fixed. None means a pump-gain ceiling,
    zero configured power, zero-T external source, or no bracket by max_power_W.
    This is neither absorbed threshold W nor experimentally calibrated external W.
    """
    _geometry_weights(model, pump)
    if not np.isfinite(max_power_W) or max_power_W <= 0.0:
        raise ValueError("max_power_W must be finite and positive")
    constants = _constants(model)
    ceiling = constants["ap"] / (constants["ap"] + constants["ep"])
    ceiling_gain = float(model.optical_gain_weights.sum() * constants["nt"] * (
        (constants["el"] + constants["al"]) * ceiling - constants["al"]
    ))
    if pump.internal_incident_power_W == 0.0 or ceiling_gain <= model.total_power_loss_per_m:
        return None

    def gap(power_W):
        scaled = _scaled_pump(pump, power_W / pump.internal_incident_power_W)
        fraction, _ = _steady_geometry_inversion(model, scaled, 0.0)
        return pump_geometry_rate_rhs(model, scaled, fraction, 0.0)[2] - model.total_power_loss_per_m

    upper = min(pump.internal_incident_power_W, max_power_W)
    upper_gap = gap(upper)
    while upper_gap <= 0.0 and upper < max_power_W:
        upper = min(2.0 * upper, max_power_W)
        upper_gap = gap(upper)
    if upper_gap <= 0.0:
        return None
    return float(brentq(gap, 0.0, upper, xtol=1e-11, rtol=1e-10))


def solve_ybyag_pump_geometry(model: SpatialYbYAGModel, pump: PumpGeometryConfig,
                              find_threshold: bool = True) -> SpatialYbYAGResult:
    """Matched result interface for single-mode CW, equal-total-power comparison.

    Edge cases call the original solver itself. Top solves spatial x-z pump
    depletion with the same volume, gain, spontaneous and loss definitions.
    A passing gain gate is not proof of stable multimode competition or thermal
    operation, and no new wavelength/DBR phase self-consistency is provided.
    """
    _geometry_weights(model, pump)
    if pump.architecture.startswith("edge"):
        result = solve_spatial_ybyag_steady(model, _edge_config(model, pump), find_threshold=find_threshold)
        budget = pump_geometry_rate_rhs(model, pump, result.excited_fraction, result.photon_density_m3)[3]
        return replace(result, energy_budget_W=budget)

    constants = _constants(model)
    unlased, iterations = _steady_geometry_inversion(model, pump, 0.0)
    _, _, unlased_gain, _ = pump_geometry_rate_rhs(model, pump, unlased, 0.0)
    can_lase = unlased_gain > model.total_power_loss_per_m

    def photon_balance(photon_fraction):
        nonlocal iterations
        fraction, count = _steady_geometry_inversion(model, pump, photon_fraction * constants["nt"], unlased)
        iterations += count
        return pump_geometry_rate_rhs(model, pump, fraction, photon_fraction * constants["nt"])[1] / constants["nt"]

    beta = model.medium.spontaneous_emission_factor
    if not can_lase and beta == 0.0:
        photon_fraction = 0.0
    elif can_lase and beta == 0.0:
        def gain_gap(photon_fraction):
            nonlocal iterations
            fraction, count = _steady_geometry_inversion(model, pump, photon_fraction * constants["nt"], unlased)
            iterations += count
            return pump_geometry_rate_rhs(model, pump, fraction, photon_fraction * constants["nt"])[2] - model.total_power_loss_per_m
        upper = 1e-10
        while gain_gap(upper) > 0.0:
            upper *= 10.0
            if upper > 1e3:
                raise RuntimeError("No finite geometry-photon saturation bracket")
        photon_fraction = float(brentq(gain_gap, 0.0, upper, xtol=1e-30, rtol=2e-10))
    else:
        upper = 1e-24
        while photon_balance(upper) > 0.0:
            upper *= 10.0
            if upper > 1e3:
                raise RuntimeError("No finite geometry photon-balance bracket")
        photon_fraction = float(brentq(photon_balance, 0.0, upper, xtol=1e-30, rtol=2e-10))
    photons = photon_fraction * constants["nt"]
    fraction, count = _steady_geometry_inversion(model, pump, photons, unlased)
    iterations += count
    transport = propagate_geometry_pump(model, pump, fraction)
    df, ds, gain, budget = pump_geometry_rate_rhs(model, pump, fraction, photons)
    threshold_W = find_geometry_threshold_pump_power(model, pump) if find_threshold else None
    return SpatialYbYAGResult(
        excited_fraction=fraction, pump_intensity_W_cm2=transport.intensity_W_cm2,
        pump_forward_W_cm2=transport.forward_W_cm2, pump_backward_W_cm2=transport.backward_W_cm2,
        photon_density_m3=photons, modal_gain_per_m=gain, unlased_modal_gain_per_m=unlased_gain,
        lasing_gain_condition=can_lase,
        threshold_pump_scale=(threshold_W / pump.internal_incident_power_W) if threshold_W is not None else None,
        energy_budget_W=budget, population_residual_per_s=float(np.max(np.abs(df))),
        photon_fraction_residual_per_s=ds / constants["nt"], iterations=iterations,
    )
