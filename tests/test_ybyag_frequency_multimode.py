from dataclasses import replace

import numpy as np
import pytest

from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.ybyag_spatial_rates import SpatialPumpConfig, SpatialYbYAGModel, solve_spatial_ybyag_steady
from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG, multimode_spatial_rhs, solve_multimode_spatial_ybyag
from pcselsim.ybyag_spectroscopy import YbYAGSpectroscopy


def model(loss=100, weights=(.4, .4), wavelength=1030):
    return SpatialYbYAGModel(
        YbYAGMediumConfig(dopant_density_cm3=1.38e21, spontaneous_emission_factor=0,
                          laser_wavelength_nm=wavelength, pumped_area_um2=1000**2),
        np.array([4e-6, 4e-6]), np.array(weights), loss, loss*.3, loss*.1)


def test_shared_inversion_single_mode_limit_recovers_original():
    one = model()
    pump = SpatialPumpConfig("edge", 2e5, 2e5, 1000, 5)
    expected = solve_spatial_ybyag_steady(one, pump, find_threshold=False)
    actual = solve_multimode_spatial_ybyag(SpatialMultimodeYbYAG((one,), ("one",)), pump)
    np.testing.assert_allclose(actual.photon_density_m3[0], expected.photon_density_m3, rtol=2e-6)
    np.testing.assert_allclose(actual.excited_fraction, expected.excited_fraction, rtol=2e-6)
    assert abs(actual.energy_budget_W["energy_closure_residual_W"]) < 1e-10


def test_same_spatial_modes_compete_for_one_population_lowest_loss_wins():
    system = SpatialMultimodeYbYAG((model(100), model(180)), ("low", "high"))
    result = solve_multimode_spatial_ybyag(system, SpatialPumpConfig("edge", 2e5, 2e5, 1000, 5))
    assert result.active_modes == ("low",)
    assert result.photon_density_m3[1] == 0
    np.testing.assert_allclose(result.modal_gain_per_m[0], 100, rtol=2e-7)
    assert result.maximum_inactive_gain_excess_per_m == 0


def test_separate_spatial_modes_can_coexist_and_gain_is_clamped():
    system = SpatialMultimodeYbYAG((model(100, (.79, .01)), model(110, (.01, .79))), ("bottom", "top"))
    result = solve_multimode_spatial_ybyag(system, SpatialPumpConfig("edge", 2e5, 2e5, 1000, 5))
    assert len(result.active_modes) == 2
    np.testing.assert_allclose(result.modal_gain_per_m, [100, 110], rtol=2e-7)
    assert result.population_residual_per_s < 1e-5


def test_distinct_wavelengths_preserve_excitation_counts_and_effective_energy():
    system = SpatialMultimodeYbYAG((model(100, wavelength=1025), model(110, wavelength=1034)), ("one", "two"))
    pump = SpatialPumpConfig("edge", 0, 0, 1000, 5)
    fraction = np.full((5, 2), .3)
    photons = np.array([1e18, 2e18])
    df, ds, gain, _, _, budget = multimode_spatial_rhs(system, pump, fraction, photons)
    assert abs(budget["energy_closure_residual_W"]) < 1e-10
    assert budget["intermode_stimulated_thermalization_W"] != 0
    nt = 1.38e27
    total = nt*np.mean(df) + ds.sum()
    sinks = -nt*.3/.00095 - sum(m.group_velocity_m_s*m.total_power_loss_per_m*s for m, s in zip(system.modes, photons))
    np.testing.assert_allclose(total, sinks, rtol=1e-11)


def test_material_tables_interpolate_and_reject_unsupported_spectrum_temperature():
    table = YbYAGSpectroscopy(
        np.array([290, 310]), np.array([930, 950]), np.array([1020, 1040]),
        np.full((2, 2), 7e-21), np.full((2, 2), 1e-21),
        np.full((2, 2), 1e-21), np.array([[2e-20, 1e-20], [1.8e-20, .8e-20]]),
        "synthetic regression table, not material evidence")
    medium = table.at(YbYAGMediumConfig(), 1030, 300)
    assert medium.laser_emission_cross_section_cm2 == pytest.approx(1.4e-20)
    with pytest.raises(ValueError, match="extrapolation rejected"):
        table.at(YbYAGMediumConfig(), 1041, 300)
    with pytest.raises(ValueError, match="extrapolation rejected"):
        table.at(YbYAGMediumConfig(), 1030, 320)


def test_fixed_frequency_cwt_does_not_relock_to_bragg():
    from scripts import run_ybyag_pcsel as workflow
    from pcselsim.reflector_cwt import build_reflector_coupling
    from pcselsim.three_d_cwt import ThreeDCWTSettings
    cell, guided, radiation, period = workflow.make_structure()
    outputs = [build_reflector_coupling(cell, guided, radiation, "passive YAG photonic crystal",
                                       period, 1030, ThreeDCWTSettings(2, 10), fixed_wavelength_nm=w)
               for w in (1026, 1030)]
    assert outputs[0].result.bragg_wavelength_nm == 1026
    assert outputs[1].result.bragg_wavelength_nm == 1030
    assert outputs[0].result.lattice_constant_nm == outputs[1].result.lattice_constant_nm == period
    assert abs(outputs[0].result.effective_index-outputs[1].result.effective_index) > 1e-6
    assert np.linalg.norm(outputs[0].result.crad_m-outputs[1].result.crad_m) > 1


@pytest.mark.parametrize("gain_um", [8., 20., 30.])
def test_fixed_frequency_finite_mode_closes_dispersion_and_updates_fields(gain_um):
    from scripts import run_ybyag_pcsel as workflow
    from pcselsim.frequency_cwt import solve_frequency_consistent_modes
    from pcselsim.three_d_cwt import ThreeDCWTSettings
    cell, guided, radiation, period = workflow.make_structure(gain_nm=gain_um*1000)
    solution = solve_frequency_consistent_modes(
        cell, guided, radiation, "passive YAG photonic crystal", period, 1030,
        ThreeDCWTSettings(2, 10), 1000, .1, 9, mode_names=("C",))['C']
    assert abs(solution.frequency_residual_per_m) < .05
    assert abs(solution.wavelength_nm-1030) > 1
    assert len(solution.evaluations) > 4
    assert solution.optical.result.bragg_wavelength_nm == solution.wavelength_nm
    assert solution.reference_overlap > .1
    if gain_um == 8:
        # Independent larger local candidate clusters must reproduce the
        # accelerated four-candidate eigenpair continuation at this benchmark.
        from pcselsim.custom_analysis import solve_finite_modes
        r = solution.optical.result
        candidates = solve_finite_modes(solution.spec, r.coupling_m, r.eigenvectors, r.radiation_fields)
        matches = [abs(np.vdot(solution.finite_mode.fields, m.fields))**2 for m in candidates.values()]
        assert max(matches) > .99


def test_shared_inversion_top_relay_counts_recirculated_pump_once():
    from pcselsim.ybyag_pump_geometry import PumpGeometryConfig
    first = replace(model(100), lateral_energy_fractions=np.full(5, .2))
    second = replace(model(180), lateral_energy_fractions=np.full(5, .2))
    system = SpatialMultimodeYbYAG((first, second), ("low", "high"))
    pump = PumpGeometryConfig("top_single", 32, 1000, 5, top_passes=20, return_efficiency=.98)
    result = solve_multimode_spatial_ybyag(system, pump)
    assert result.active_modes == ("low",)
    budget = result.energy_budget_W
    assert budget["pump_relay_loss_W"] > 0
    assert abs(budget["pump_boundary_flow_residual_W"]) < 1e-11
    assert abs(budget["energy_closure_residual_W"]) < 1e-10


def test_configurable_dense_reference_and_sparse_tracking_solve_identical_poles():
    from pcselsim.custom_analysis import _upwind_1d
    from pcselsim.finite_discretization import build_finite_eigenproblem, relative_residual
    count = 16
    step = 1/count
    problem = build_finite_eigenproblem(np.array([[0.,2.],[2.,0.]]),
        (_upwind_1d(count,step,True),_upwind_1d(count,step,False)),step,"staggered")
    seed = np.ones(2*count,dtype=complex)
    dense_values,dense_fields = problem.near(3.4+1j,4,seed,dense_limit=1400)
    sparse_values,sparse_fields = problem.near(3.4+1j,4,seed,dense_limit=0,maxiter=400)
    assert max(min(abs(value-sparse_values)) for value in dense_values) < 1e-9
    for values,fields in ((dense_values,dense_fields),(sparse_values,sparse_fields)):
        assert max(relative_residual(problem.center_operator(),field,value)
                   for value,field in zip(values,fields.T)) < 1e-10
