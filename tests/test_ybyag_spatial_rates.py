"""Physical closure tests for layered inversion coupled to one PCSEL mode."""

from dataclasses import replace

import numpy as np
import pytest

from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.ybyag_spatial_rates import (
    SpatialPumpConfig, SpatialYbYAGModel, find_spatial_threshold_pump_scale,
    propagate_spatial_pump, solve_spatial_ybyag_steady, spatial_volume_and_gain_weights,
    spatial_ybyag_rate_rhs,
)


def model_fixture(beta=0.0, loss=35.0, widths=None, weights=None):
    return SpatialYbYAGModel(
        medium=YbYAGMediumConfig(spontaneous_emission_factor=beta, pumped_area_um2=500.0**2),
        cell_widths_m=np.array([2.0, 3.0, 10.0]) * 1e-6 if widths is None else widths,
        optical_energy_fractions=np.array([0.05, 0.65, 0.1]) if weights is None else weights,
        total_power_loss_per_m=loss, upward_power_loss_per_m=loss * 0.3,
        downward_power_loss_per_m=loss * 0.1,
    )


@pytest.mark.parametrize("direction", ["normal", "edge"])
def test_spatial_energy_and_stimulated_count_closure(direction):
    model = model_fixture(beta=1e-4)
    pump = SpatialPumpConfig(direction=direction, backward_intensity_W_cm2=1e4, edge_cells=5)
    volume, q = spatial_volume_and_gain_weights(model, pump)
    assert np.isclose(volume.sum(), 1.0)
    assert np.isclose(q.sum(), 0.8)
    fraction = np.full(volume.shape, 0.3)
    fraction[:, 1] = 0.6
    df, ds, _, budget = spatial_ybyag_rate_rhs(model, pump, fraction, 1e18)
    scale = max(budget["absorbed_pump_W"], budget["upward_output_W"], 1e-9)
    assert abs(budget["energy_closure_residual_W"]) < scale * 1e-9
    # Strong TE weight in a thin cell must change inversion faster: no
    # equal-thickness average can replace actual optical energy fractions.
    assert abs(df[0, 1]) > abs(df[0, 2])
    assert np.isfinite(ds)


def test_pump_transport_recovers_unsaturated_beer_lambert():
    model = model_fixture()
    pump = SpatialPumpConfig(forward_intensity_W_cm2=123.0, backward_intensity_W_cm2=0.0)
    _, _, incident, absorbed = propagate_spatial_pump(model, pump, np.zeros((1, 3)))
    alpha = model.medium.dopant_density_cm3 * model.medium.pump_absorption_cross_section_cm2 * 100.0
    expected = -np.expm1(-alpha * model.gain_thickness_m)
    np.testing.assert_allclose(absorbed / incident, expected, rtol=1e-10)


@pytest.mark.parametrize("direction", ["normal", "edge"])
def test_spatial_steady_lasing_has_gain_clamp_and_power_budget(direction):
    model = model_fixture(beta=0.0)
    pump = SpatialPumpConfig(direction=direction, backward_intensity_W_cm2=2.5e4, edge_cells=5)
    result = solve_spatial_ybyag_steady(model, pump, find_threshold=False)
    assert result.lasing_gain_condition
    assert result.photon_density_m3 > 0.0
    np.testing.assert_allclose(result.modal_gain_per_m, model.total_power_loss_per_m, rtol=1e-8)
    assert result.population_residual_per_s < 1e-6
    assert abs(result.energy_budget_W["energy_closure_residual_W"]) < 1e-9
    assert abs(result.energy_budget_W["stored_excitation_plus_photon_energy_rate_W"]) < 1e-8
    assert result.energy_budget_W["upward_output_W"] > result.energy_budget_W["downward_output_W"] > 0.0


def test_nonlasing_spontaneous_output_is_not_labelled_lasing():
    model = model_fixture(beta=1e-8, loss=1e4)
    result = solve_spatial_ybyag_steady(model, SpatialPumpConfig(), find_threshold=False)
    assert not result.lasing_gain_condition
    assert result.photon_density_m3 > 0.0
    assert result.energy_budget_W["upward_output_W"] > 0.0
    assert abs(result.energy_budget_W["energy_closure_residual_W"]) < 1e-9


def test_true_optical_weights_are_not_renormalized_to_one():
    original = model_fixture()
    low_overlap = replace(original, optical_energy_fractions=original.optical_energy_fractions * 0.25,
                          optical_gain_weights=original.optical_gain_weights * 0.25)
    pump = SpatialPumpConfig()
    high = solve_spatial_ybyag_steady(original, pump, find_threshold=False)
    low = solve_spatial_ybyag_steady(low_overlap, pump, find_threshold=False)
    np.testing.assert_allclose(low.unlased_modal_gain_per_m, high.unlased_modal_gain_per_m * 0.25)
    assert low.lasing_gain_condition is False


def test_threshold_is_based_on_depleted_unlased_local_gain():
    model = model_fixture()
    pump = SpatialPumpConfig(direction="normal")
    threshold = find_spatial_threshold_pump_scale(model, pump)
    assert threshold is not None and 0.0 < threshold < 1.0
    scaled = replace(pump, forward_intensity_W_cm2=pump.forward_intensity_W_cm2 * threshold)
    result = solve_spatial_ybyag_steady(model, scaled, find_threshold=False)
    np.testing.assert_allclose(result.unlased_modal_gain_per_m, model.total_power_loss_per_m, rtol=1e-8)
    unreachable = replace(model, total_power_loss_per_m=1e4)
    assert find_spatial_threshold_pump_scale(unreachable, pump) is None


def test_equal_double_edge_pump_is_symmetric_for_symmetric_cells():
    model = model_fixture(widths=np.array([5e-6, 5e-6]), weights=np.array([0.4, 0.4]))
    pump = SpatialPumpConfig(direction="edge", forward_intensity_W_cm2=1.25e4,
                             backward_intensity_W_cm2=1.25e4, edge_length_um=5000, edge_cells=9)
    result = solve_spatial_ybyag_steady(model, pump, find_threshold=False)
    np.testing.assert_allclose(result.excited_fraction, result.excited_fraction[::-1], rtol=1e-9)
    assert abs(result.energy_budget_W["energy_closure_residual_W"]) < 1e-9


def test_loss_budget_cannot_double_count_upward_and_downward_radiation():
    with pytest.raises(ValueError, match="upward"):
        replace(model_fixture(), upward_power_loss_per_m=30.0, downward_power_loss_per_m=30.0)


def test_energy_overlap_and_flux_gain_weights_use_local_velocity():
    medium = YbYAGMediumConfig(spontaneous_emission_factor=0.0, refractive_index=1.82)
    velocity = 299792458.0 / 2.0
    model = SpatialYbYAGModel(
        medium=medium, cell_widths_m=np.array([5e-6, 5e-6]),
        optical_energy_fractions=np.array([0.3, 0.4]),
        group_velocity_m_s=velocity, total_power_loss_per_m=40.0,
    )
    np.testing.assert_allclose(model.optical_gain_weights, model.optical_energy_fractions * 2.0 / 1.82)
    fraction = np.array([[0.4, 0.4]])
    df, ds, gain, budget = spatial_ybyag_rate_rhs(model, SpatialPumpConfig(), fraction, 1e18)
    material_gain = medium.dopant_density_cm3 * 1e6 * (
        (medium.laser_emission_cross_section_cm2 + medium.laser_absorption_cross_section_cm2) * 1e-4 * 0.4
        - medium.laser_absorption_cross_section_cm2 * 1e-4
    )
    # Absorption/emission acts at c/n_core locally; a different group index
    # must not silently change that rate for fixed full-mode energy fractions.
    np.testing.assert_allclose(velocity * gain, 299792458.0 / medium.refractive_index * 0.7 * material_gain)
    assert abs(budget["energy_closure_residual_W"]) < 1e-9


def test_finite_beta_above_threshold_still_closes_energy_budget():
    model = model_fixture(beta=1e-4)
    result = solve_spatial_ybyag_steady(model, SpatialPumpConfig(), find_threshold=False)
    assert result.lasing_gain_condition
    assert 0.0 < result.modal_gain_per_m < model.total_power_loss_per_m
    assert abs(result.photon_fraction_residual_per_s) < 1e-7
    assert abs(result.energy_budget_W["energy_closure_residual_W"]) < 1e-9


def test_photon_density_and_spatial_grid_input_validation():
    with pytest.raises(ValueError, match="photon_density"):
        spatial_ybyag_rate_rhs(model_fixture(), SpatialPumpConfig(), np.full((1, 3), 0.3), -1.0)
    with pytest.raises(ValueError, match="cell count"):
        SpatialPumpConfig(edge_cells=4.5)
    with pytest.raises(ValueError, match="max_scale"):
        find_spatial_threshold_pump_scale(model_fixture(), SpatialPumpConfig(), max_scale=0.0)


def test_nonuniform_finite_envelope_keeps_exact_excitation_photon_closure():
    lateral = np.array([0.02, 0.1, 0.76, 0.1, 0.02])
    model = replace(model_fixture(beta=1e-4), lateral_energy_fractions=lateral)
    pump = SpatialPumpConfig(direction="edge", edge_cells=5)
    volume, gain_weights = spatial_volume_and_gain_weights(model, pump)
    np.testing.assert_allclose(gain_weights.sum(axis=1), lateral * model.optical_gain_weights.sum())
    np.testing.assert_allclose(volume.sum(axis=1), np.full(5, 0.2))
    _, _, _, budget = spatial_ybyag_rate_rhs(model, pump, np.full((5, 3), 0.4), 1e18)
    assert abs(budget["energy_closure_residual_W"]) < 1e-9
    with pytest.raises(ValueError, match="length must match"):
        spatial_volume_and_gain_weights(model, SpatialPumpConfig(direction="normal"))


def test_finite_envelope_produces_local_steady_spatial_hole_burning():
    model = replace(model_fixture(), lateral_energy_fractions=np.array([0.02, 0.1, 0.76, 0.1, 0.02]))
    pump = SpatialPumpConfig(direction="edge", edge_cells=5,
                             forward_intensity_W_cm2=2.5e4, backward_intensity_W_cm2=2.5e4)
    result = solve_spatial_ybyag_steady(model, pump, find_threshold=False)
    assert result.lasing_gain_condition
    # Strong central finite-mode envelope must deplete the same layer more
    # strongly at x centre, rather than assuming uniform photon density.
    assert result.excited_fraction[2, 1] < result.excited_fraction[0, 1]
    np.testing.assert_allclose(result.excited_fraction, result.excited_fraction[::-1], rtol=1e-9)
    np.testing.assert_allclose(result.modal_gain_per_m, model.total_power_loss_per_m, rtol=1e-8)
    assert abs(result.energy_budget_W["energy_closure_residual_W"]) < 1e-9
