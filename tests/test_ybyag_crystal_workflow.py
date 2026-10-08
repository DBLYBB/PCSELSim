"""Interface checks without an expensive finite-aperture CWT solve.

These checks prove geometry/normalization/unit consistency, not fabrication
or experimental validation of the proposed Yb:YAG crystal PCSEL.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c
from scipy.integrate import trapezoid

from scripts import run_ybyag_pcsel as workflow
from pcselsim.hybrid_vertical import solve_guided_te_modes
from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.ybyag_spatial_rates import (
    SpatialPumpConfig, SpatialYbYAGModel, propagate_spatial_pump,
)


@pytest.fixture(scope="module")
def vertical_only_candidate():
    cell, guided, full, period = workflow.make_structure()
    mode = guided.solve_te0(workflow.TARGET_WAVELENGTH_NM, workflow.VERTICAL_STEP_NM)
    norm_energy = trapezoid(mode.index**2 * abs(mode.field)**2, mode.z_um * 1e-6)
    result = SimpleNamespace(
        vertical_mode=mode, effective_index=mode.effective_index,
        group_index=norm_energy / mode.effective_index,
    )
    return cell, guided, full, period, result


def test_waveguide_is_separate_from_the_finite_bottom_dbr(vertical_only_candidate):
    _, guided, full, _, _ = vertical_only_candidate
    assert len(full.layers) == len(guided.layers) + 2 * workflow.DBR_PAIRS
    assert all(layer.name.startswith("DBR_") for layer in full.layers[:2 * workflow.DBR_PAIRS])
    assert not any(layer.name.startswith("DBR_") for layer in guided.layers)
    assert [layer.name for layer in full.layers[2 * workflow.DBR_PAIRS:]] == [layer.name for layer in guided.layers]
    assert guided.layers[0].name == "lower SiO2 guide spacer"
    assert guided.layers[0].thickness_nm == workflow.BOTTOM_SPACER_NM
    assert guided.layers[-2].name == "unpatterned SiN upper guide"
    assert guided.layers[-1].name == "SiO2 upper cladding"
    assert guided.layers[-2].refractive_index != guided.layers[0].refractive_index
    assert full.top_index == 1.0


def test_active_energy_and_gain_weights_have_correct_distinct_normalizations(vertical_only_candidate):
    _, guided, _, _, result = vertical_only_candidate
    widths, energy, gain = workflow.active_weights(result, guided)
    assert 0.0 < energy.sum() < 1.0
    np.testing.assert_allclose(
        gain, result.group_index / workflow.YBYAG_INDEX * energy, rtol=1e-13,
    )
    np.testing.assert_allclose(widths.sum(), workflow.GAIN_THICKNESS_NM * 1e-9, rtol=1e-13)
    assert len(widths) == workflow.GAIN_Z_CELLS
    # Refining only the rate cells must not alter exact piecewise-linear
    # longitudinal integrals or renormalize gain confinement to one.
    fine_widths, fine_energy, fine_gain = workflow.active_weights(result, guided, cells=81)
    np.testing.assert_allclose(fine_widths.sum(), widths.sum(), rtol=1e-13)
    np.testing.assert_allclose(fine_energy.sum(), energy.sum(), rtol=1e-13)
    np.testing.assert_allclose(fine_gain.sum(), gain.sum(), rtol=1e-13)


def test_actual_active_thickness_not_legacy_volume_drives_edge_pump_units(vertical_only_candidate):
    _, guided, _, _, result = vertical_only_candidate
    widths, energy, gain = workflow.active_weights(result, guided)
    medium = YbYAGMediumConfig(
        refractive_index=workflow.YBYAG_INDEX, dopant_density_cm3=workflow.YB_DENSITY_CM3,
        pumped_area_um2=workflow.DEVICE_SIZE_UM**2,
        gain_thickness_um=200.0,  # deliberately inconsistent legacy value
    )
    model = SpatialYbYAGModel(medium, widths, energy, total_power_loss_per_m=1.0,
                              group_velocity_m_s=c / result.group_index, optical_gain_weights=gain)
    area_m2 = workflow.DEVICE_SIZE_UM**2 * 1e-12
    thickness_m = workflow.GAIN_THICKNESS_NM * 1e-9
    np.testing.assert_allclose(model.gain_volume_m3, area_m2 * thickness_m, rtol=1e-13)
    pump = SpatialPumpConfig(
        direction="edge", edge_length_um=workflow.DEVICE_SIZE_UM, edge_cells=workflow.PUMP_CELLS,
        forward_intensity_W_cm2=workflow.PUMP_INTENSITY_PER_EDGE_W_CM2,
        backward_intensity_W_cm2=workflow.PUMP_INTENSITY_PER_EDGE_W_CM2,
    )
    _, _, incident, absorbed = propagate_spatial_pump(
        model, pump, np.zeros((workflow.PUMP_CELLS, workflow.GAIN_Z_CELLS)),
    )
    entrance_area_m2 = area_m2 / (workflow.DEVICE_SIZE_UM * 1e-6) * thickness_m
    expected_incident = 2.0 * workflow.PUMP_INTENSITY_PER_EDGE_W_CM2 * 1e4 * entrance_area_m2
    np.testing.assert_allclose(incident, expected_incident, rtol=1e-13)
    assert 0.0 < absorbed < incident


def test_thick_candidate_reports_multiple_te_families_not_single_mode_proof(vertical_only_candidate):
    _, guided, _, period, result = vertical_only_candidate
    families = solve_guided_te_modes(guided, workflow.TARGET_WAVELENGTH_NM,
                                     workflow.VERTICAL_STEP_NM, max_modes=12)
    assert len(families) > 1
    assert all(mode.effective_index > max(guided.top_index, guided.bottom_index) for mode in families)
    assert all(left.effective_index > right.effective_index for left, right in zip(families[:-1], families[1:]))
    np.testing.assert_allclose(period * result.effective_index, workflow.TARGET_WAVELENGTH_NM, rtol=1e-12)


def test_passive_photonic_crystal_does_not_silently_count_as_yb_gain(vertical_only_candidate):
    cell, guided, _, _, _ = vertical_only_candidate
    gain_layers = [layer for layer in guided.layers if layer.name == "Yb:YAG gain"]
    pc_layers = [layer for layer in guided.layers if layer.name == "passive YAG photonic crystal"]
    assert len(gain_layers) == len(pc_layers) == 1
    assert pc_layers[0].thickness_nm == workflow.PC_DEPTH_NM
    np.testing.assert_allclose(pc_layers[0].refractive_index**2, cell.fourier_epsilon(0, 0).real)
