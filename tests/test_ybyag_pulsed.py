"""Independent limits and photon/energy bookkeeping of the reduced pulse model."""
from dataclasses import replace

import numpy as np

from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.ybyag_spatial_rates import SpatialYbYAGModel
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig, find_geometry_threshold_pump_power
from pcselsim.ybyag_pulsed import (
    RectangularPumpPulse, simulate_pulsed_ybyag, uniform_top_pump_flow,
    uniform_top_pump_threshold, uniform_top_cw_solution,
)


def model():
    medium = YbYAGMediumConfig(dopant_density_cm3=1.38e21,
        pumped_area_um2=300**2, gain_thickness_um=.5, spontaneous_emission_factor=0)
    return SpatialYbYAGModel(medium, np.array([.5e-6]), np.array([.8]),
        900., 200., 100., optical_gain_weights=np.array([.8]))


def test_threshold_agrees_with_independent_spatial_one_cell_solver():
    m = model()
    p = RectangularPumpPulse(1e-3, 1e-3, top_passes=20)
    expected = find_geometry_threshold_pump_power(m,
        PumpGeometryConfig('top_single', 1., 300., 1, top_passes=20, return_efficiency=.98))
    actual = uniform_top_pump_threshold(m,p)['internal_CW_threshold_W']
    assert np.isclose(actual,expected,rtol=1e-7)


def test_multi_pass_boundary_counts_only_first_entry():
    m = model()
    f = .3
    one = uniform_top_pump_flow(m,f,2.,1)
    twenty = uniform_top_pump_flow(m,f,2.,20,.98)
    assert twenty['absorbed_W'] > one['absorbed_W']
    assert abs(twenty['boundary_residual_W']) < 1e-14
    # Beer-law geometric series independently evaluates the same absorption.
    medium = m.medium
    alpha = medium.dopant_density_cm3*(medium.pump_absorption_cross_section_cm2*(1-f)-medium.pump_emission_cross_section_cm2*f)
    T = np.exp(-alpha*m.gain_thickness_m*100)
    expected = 2*(1-T)*(1-(T*.98)**20)/(1-T*.98)
    assert np.isclose(twenty['absorbed_W'],expected,rtol=1e-12)


def test_infinite_pump_limit_rejects_unreachable_gain():
    m = replace(model(),total_power_loss_per_m=1e5,upward_power_loss_per_m=0.,downward_power_loss_per_m=0.)
    p = RectangularPumpPulse(1.,.001)
    assert uniform_top_pump_threshold(m,p)['external_CW_threshold_W'] is None


def test_external_energy_and_excitation_photon_energy_close():
    m = model()
    p = RectangularPumpPulse(5e-4,2e-5,coupling_efficiency=.7,top_passes=20,after_pulse_s=2e-5)
    r = simulate_pulsed_ybyag(m,p,samples=301)
    assert np.isclose(r.metrics['external_pump_J'],p.external_energy_J,rtol=1e-10)
    assert abs(r.metrics['pump_boundary_residual_J']) < 1e-12
    assert r.metrics['relative_energy_residual'] < 1e-9
    assert r.excited_fraction.min() >= -1e-10
    assert r.excited_fraction.max() <= 1


def test_cw_equilibrium_is_stationary_without_beta():
    m = model()
    p0 = RectangularPumpPulse(1e-3,1e-4,spontaneous_beta=0.,after_pulse_s=0.)
    pth = uniform_top_pump_threshold(m,p0)['external_CW_threshold_W']
    power = 2*pth
    p = replace(p0,external_energy_J=power*p0.duration_s)
    cw = uniform_top_cw_solution(m,p,power)
    ions = m.medium.dopant_density_cm3*1e6*m.gain_volume_m3
    r = simulate_pulsed_ybyag(m,p,initial_excited_fraction=cw['excited_fraction'],
        initial_photons=cw['photon_fraction']*ions,samples=301)
    assert np.allclose(r.upward_power_W,cw['upward_power_W'],rtol=2e-6)
    assert np.isclose(r.metrics['last_tenth_on_average_upward_W'],cw['upward_power_W'],rtol=2e-6)


def test_same_energy_with_short_and_long_duration_changes_inversion():
    m = replace(model(), total_power_loss_per_m=1e5,upward_power_loss_per_m=0.,downward_power_loss_per_m=0.)
    short = RectangularPumpPulse(.001,1e-5,spontaneous_beta=0.,after_pulse_s=0.)
    long = replace(short,duration_s=.01)
    a = simulate_pulsed_ybyag(m,short,samples=101)
    b = simulate_pulsed_ybyag(m,long,samples=101)
    assert a.metrics['final_inversion_fraction'] > 3*b.metrics['final_inversion_fraction']
