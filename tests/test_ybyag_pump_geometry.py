"""Equal-input-power pump geometry, saturation and conservation checks."""

from dataclasses import replace

import numpy as np
import pytest

from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.ybyag_spatial_rates import (
    SpatialPumpConfig, SpatialYbYAGModel, solve_spatial_ybyag_steady,
)
from pcselsim.ybyag_pump_geometry import (
    PumpGeometryConfig, find_geometry_threshold_pump_power,
    propagate_geometry_pump, pump_geometry_metadata, pump_geometry_rate_rhs,
    solve_ybyag_pump_geometry,
)


def model_fixture(thickness_um=8.0, loss=400.0, beta=0.0, uniform_x=False):
    return SpatialYbYAGModel(
        medium=YbYAGMediumConfig(
            dopant_density_cm3=1.38e21, spontaneous_emission_factor=beta,
            pumped_area_um2=1000.0**2,
        ),
        cell_widths_m=np.array([1.0, 1.0, 2.0, 4.0]) * thickness_um / 8.0 * 1e-6,
        optical_energy_fractions=np.array([0.05, 0.1, 0.2, 0.25]),
        total_power_loss_per_m=loss, upward_power_loss_per_m=loss * 0.3,
        downward_power_loss_per_m=loss * 0.1,
        lateral_energy_fractions=np.full(5, 0.2) if uniform_x else np.array([0.05, 0.15, 0.6, 0.15, 0.05]),
    )


@pytest.mark.parametrize("architecture", ["edge_single", "edge_dual"])
def test_edge_adapter_matches_original_spatial_solver(architecture):
    model = model_fixture()
    pump = PumpGeometryConfig(architecture, 32.0, 1000.0, 5)
    intensity = pump_geometry_metadata(model, pump)["internal_intensity_per_beam_W_cm2"]
    original_pump = SpatialPumpConfig(
        "edge", intensity, intensity if architecture == "edge_dual" else 0.0, 1000.0, 5,
    )
    original = solve_spatial_ybyag_steady(model, original_pump)
    adapted = solve_ybyag_pump_geometry(model, pump)
    np.testing.assert_allclose(adapted.excited_fraction, original.excited_fraction, rtol=1e-12)
    np.testing.assert_allclose(adapted.photon_density_m3, original.photon_density_m3, rtol=1e-12)
    np.testing.assert_allclose(adapted.pump_intensity_W_cm2, original.pump_intensity_W_cm2, rtol=1e-12)
    np.testing.assert_allclose(adapted.threshold_pump_scale, original.threshold_pump_scale, rtol=1e-12)
    np.testing.assert_allclose(adapted.energy_budget_W["upward_output_W"], original.energy_budget_W["upward_output_W"], rtol=1e-12)
    threshold_W = find_geometry_threshold_pump_power(model, pump)
    np.testing.assert_allclose(threshold_W, pump.internal_incident_power_W * original.threshold_pump_scale, rtol=1e-8)


def test_thickness_does_not_silently_increase_total_pump_power():
    thin, thick = model_fixture(8.0), model_fixture(16.0)
    for architecture in ("edge_single", "edge_dual", "top_single"):
        pump = PumpGeometryConfig(architecture, 32.0, 1000.0, 5)
        thin_meta = pump_geometry_metadata(thin, pump)
        thick_meta = pump_geometry_metadata(thick, pump)
        assert thin_meta["internal_incident_power_W"] == thick_meta["internal_incident_power_W"] == 32.0
        ratio = thick_meta["internal_intensity_per_beam_W_cm2"] / thin_meta["internal_intensity_per_beam_W_cm2"]
        np.testing.assert_allclose(ratio, 1.0 if architecture == "top_single" else 0.5)
        for model in (thin, thick):
            transport = propagate_geometry_pump(model, pump, np.zeros((5, 4)))
            np.testing.assert_allclose(transport.incident_internal_power_W, 32.0)


def test_top_transport_recovers_beer_lambert_and_depletes_from_top():
    model = model_fixture()
    pump = PumpGeometryConfig("top_single", 1.0, 1000.0, 5)
    result = propagate_geometry_pump(model, pump, np.zeros((5, 4)))
    alpha = model.medium.dopant_density_cm3 * model.medium.pump_absorption_cross_section_cm2 * 100.0
    expected_fraction = -np.expm1(-alpha * model.gain_thickness_m)
    np.testing.assert_allclose(result.absorbed_power_W, expected_fraction, rtol=1e-12)
    assert np.all(result.forward_W_cm2 == 0.0)
    assert np.all(result.backward_W_cm2[:, -1] > result.backward_W_cm2[:, 0])
    assert abs(result.pump_flux_residual_W) < 1e-12


def test_zero_net_pump_absorption_gives_constant_intensity_and_no_heat_source():
    model = model_fixture()
    medium = model.medium
    ceiling = medium.pump_absorption_cross_section_cm2 / (
        medium.pump_absorption_cross_section_cm2 + medium.pump_emission_cross_section_cm2
    )
    pump = PumpGeometryConfig("top_single", 32.0, 1000.0, 5)
    result = propagate_geometry_pump(model, pump, np.full((5, 4), ceiling))
    np.testing.assert_allclose(result.intensity_W_cm2, 3200.0, rtol=1e-12)
    assert abs(result.absorbed_power_W) < 1e-12


@pytest.mark.parametrize("architecture", ["edge_single", "edge_dual", "top_single"])
def test_instantaneous_geometry_rhs_preserves_pump_flow_and_full_energy(architecture):
    model = model_fixture(beta=1e-4)
    pump = PumpGeometryConfig(architecture, 32.0, 1000.0, 5)
    fraction = np.full((5, 4), 0.3)
    fraction[2, -1] = 0.6
    df, ds, _, budget = pump_geometry_rate_rhs(model, pump, fraction, 1e18)
    assert df.shape == (5, 4) and np.isfinite(ds)
    assert abs(budget["pump_flux_residual_W"]) < 1e-10
    assert abs(budget["pump_boundary_flow_residual_W"]) < 1e-10
    assert abs(budget["energy_closure_residual_W"]) < 1e-9


def test_top_zero_photon_saturation_is_pump_emission_limited_not_fully_inverted():
    model = model_fixture(loss=1e6)
    pump = PumpGeometryConfig("top_single", 1e6, 1000.0, 5)
    result = solve_ybyag_pump_geometry(model, pump, find_threshold=False)
    expected = model.medium.pump_absorption_cross_section_cm2 / (
        model.medium.pump_absorption_cross_section_cm2 + model.medium.pump_emission_cross_section_cm2
    )
    np.testing.assert_allclose(result.excited_fraction, expected, rtol=3e-4)
    assert np.all(result.excited_fraction < expected)
    assert not result.lasing_gain_condition
    assert result.photon_density_m3 == 0.0
    assert abs(result.energy_budget_W["pump_flux_residual_W"]) < 1e-7


def test_top_threshold_in_internal_w_and_steady_gain_clamp():
    model = model_fixture()
    pump = PumpGeometryConfig("top_single", 32.0, 1000.0, 5)
    threshold_W = find_geometry_threshold_pump_power(model, pump)
    assert threshold_W is not None and threshold_W > 32.0
    below = solve_ybyag_pump_geometry(model, pump, find_threshold=False)
    assert not below.lasing_gain_condition
    threshold_result = solve_ybyag_pump_geometry(
        model, replace(pump, internal_incident_power_W=threshold_W), find_threshold=False,
    )
    np.testing.assert_allclose(threshold_result.unlased_modal_gain_per_m, model.total_power_loss_per_m, rtol=1e-8)
    above = solve_ybyag_pump_geometry(model, replace(pump, internal_incident_power_W=2.0 * threshold_W))
    assert above.lasing_gain_condition and above.photon_density_m3 > 0.0
    assert above.excited_fraction[2, -1] < above.excited_fraction[0, -1]
    np.testing.assert_allclose(above.modal_gain_per_m, model.total_power_loss_per_m, rtol=1e-8)
    np.testing.assert_allclose(above.threshold_pump_scale, 0.5, rtol=1e-8)
    assert abs(above.energy_budget_W["energy_closure_residual_W"]) < 1e-9
    assert abs(above.energy_budget_W["stored_excitation_plus_photon_energy_rate_W"]) < 1e-8


def test_top_uniform_lateral_limit_matches_original_normal_model():
    model = model_fixture(uniform_x=True)
    new_pump = PumpGeometryConfig("top_single", 400.0, 1000.0, 5)
    original_model = replace(model, lateral_energy_fractions=None)
    original = solve_spatial_ybyag_steady(
        original_model, SpatialPumpConfig("normal", 0.0, 40000.0), find_threshold=False,
    )
    updated = solve_ybyag_pump_geometry(model, new_pump, find_threshold=False)
    np.testing.assert_allclose(updated.excited_fraction, np.repeat(original.excited_fraction, 5, axis=0), rtol=1e-9)
    np.testing.assert_allclose(updated.photon_density_m3, original.photon_density_m3, rtol=1e-9)
    np.testing.assert_allclose(updated.energy_budget_W["upward_output_W"], original.energy_budget_W["upward_output_W"], rtol=1e-9)


def test_external_top_power_conversion_is_distinct_from_absorption():
    model = model_fixture()
    pump = PumpGeometryConfig.from_external_top(40.0, 0.8, 0.1, device_side_um=1000.0, x_cells=5)
    np.testing.assert_allclose(pump.internal_incident_power_W, 32.0)
    result = solve_ybyag_pump_geometry(model, pump, find_threshold=False)
    budget = result.energy_budget_W
    np.testing.assert_allclose(budget["incident_external_pump_W"], 40.0)
    np.testing.assert_allclose(budget["entrance_reflected_pump_W"], 4.0)
    np.testing.assert_allclose(budget["entrance_untransmitted_unassigned_pump_W"], 4.0)
    assert 0.0 < budget["absorbed_pump_W"] < 32.0
    assert abs(budget["external_pump_flow_residual_W"]) < 1e-12


@pytest.mark.parametrize("passes", [1, 2, 20])
def test_incoherent_lossless_relay_beer_limit_not_repeated_new_sources(passes):
    model = model_fixture()
    pump = PumpGeometryConfig("top_single", 32.0, 1000.0, 5, top_passes=passes)
    result = propagate_geometry_pump(model, pump, np.zeros((5, 4)))
    alpha = model.medium.dopant_density_cm3 * model.medium.pump_absorption_cross_section_cm2 * 100.0
    expected = 32.0 * -np.expm1(-alpha * model.gain_thickness_m * passes)
    np.testing.assert_allclose(result.absorbed_power_W, expected, rtol=1e-12)
    assert result.incident_internal_power_W == 32.0
    assert result.relay_loss_power_W == 0.0
    assert abs(result.pump_flux_residual_W) < 1e-11


def test_finite_relay_loss_and_escape_are_not_absorbed_pump():
    model = model_fixture()
    pump = PumpGeometryConfig("top_single", 32.0, 1000.0, 5, top_passes=20, return_efficiency=0.98)
    transport = propagate_geometry_pump(model, pump, np.zeros((5, 4)))
    assert transport.relay_loss_power_W > 0.0
    alpha = model.medium.dopant_density_cm3 * model.medium.pump_absorption_cross_section_cm2 * 100.0
    transmission = np.exp(-alpha * model.gain_thickness_m)
    roundtrip_multiplier = pump.return_efficiency * transmission
    expected_absorbed = 32.0 * (1.0 - transmission) * (1.0 - roundtrip_multiplier**20) / (1.0 - roundtrip_multiplier)
    np.testing.assert_allclose(transport.absorbed_power_W, expected_absorbed, rtol=1e-12)
    np.testing.assert_allclose(transport.absorbed_power_W + transport.relay_loss_power_W + transport.escaped_pump_power_W, 32.0)
    df, ds, _, budget = pump_geometry_rate_rhs(model, pump, np.full((5, 4), 0.4), 1e18)
    assert np.isfinite(df).all() and np.isfinite(ds)
    assert abs(budget["pump_boundary_flow_residual_W"]) < 1e-11
    assert abs(budget["pump_flux_residual_W"]) < 1e-10
    assert abs(budget["energy_closure_residual_W"]) < 1e-9
    assert "escaped_single_pass_pump_W" not in budget


def test_no_return_reduces_twenty_pass_configuration_to_single_pass():
    model = model_fixture()
    fraction = np.full((5, 4), 0.3)
    one = propagate_geometry_pump(model, PumpGeometryConfig("top_single", 32.0, 1000.0, 5), fraction)
    no_return = propagate_geometry_pump(
        model, PumpGeometryConfig("top_single", 32.0, 1000.0, 5, top_passes=20, return_efficiency=0.0), fraction,
    )
    np.testing.assert_allclose(one.intensity_W_cm2, no_return.intensity_W_cm2)
    np.testing.assert_allclose(one.absorbed_power_W, no_return.absorbed_power_W)
    np.testing.assert_allclose(no_return.relay_loss_power_W, one.escaped_pump_power_W)
    assert no_return.escaped_pump_power_W == 0.0


def test_multipass_top_steady_energy_closure_and_threshold():
    model = model_fixture()
    pump = PumpGeometryConfig("top_single", 32.0, 1000.0, 5, top_passes=20, return_efficiency=0.98)
    result = solve_ybyag_pump_geometry(model, pump)
    assert result.lasing_gain_condition
    assert result.threshold_pump_scale is not None and result.threshold_pump_scale < 1.0
    np.testing.assert_allclose(result.modal_gain_per_m, model.total_power_loss_per_m, rtol=1e-8)
    assert abs(result.energy_budget_W["energy_closure_residual_W"]) < 1e-9
    assert result.energy_budget_W["pump_relay_loss_W"] > 0.0


@pytest.mark.parametrize("loss", [400.0, 1e6])
def test_top_finite_beta_solutions_close_energy_above_and_below_threshold(loss):
    model = model_fixture(loss=loss, beta=1e-8)
    pump = PumpGeometryConfig("top_single", 32.0, 1000.0, 5, top_passes=20, return_efficiency=0.98)
    result = solve_ybyag_pump_geometry(model, pump, find_threshold=False)
    assert result.photon_density_m3 > 0.0
    assert result.lasing_gain_condition == (loss == 400.0)
    assert abs(result.energy_budget_W["energy_closure_residual_W"]) < 1e-9


def test_geometry_input_validation_and_unreachable_threshold():
    with pytest.raises(ValueError, match="positive"):
        PumpGeometryConfig(internal_incident_power_W=-1.0)
    with pytest.raises(ValueError, match="T/R"):
        PumpGeometryConfig.from_external_top(10.0, 0.9, 0.2)
    with pytest.raises(ValueError, match="top_single"):
        PumpGeometryConfig("edge_single", top_passes=20)
    with pytest.raises(ValueError, match="positive integer"):
        PumpGeometryConfig("top_single", top_passes=2.5)
    with pytest.raises(ValueError, match="pumped_area"):
        solve_ybyag_pump_geometry(model_fixture(), PumpGeometryConfig(device_side_um=500.0))
    with pytest.raises(ValueError, match="length must match"):
        solve_ybyag_pump_geometry(model_fixture(), PumpGeometryConfig(x_cells=7))
    assert find_geometry_threshold_pump_power(model_fixture(loss=1e6), PumpGeometryConfig(x_cells=5)) is None
    assert find_geometry_threshold_pump_power(model_fixture(), PumpGeometryConfig(internal_incident_power_W=0.0, x_cells=5)) is None
