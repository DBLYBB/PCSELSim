import numpy as np
import pytest

from pcselsim.band_structure import detuning_to_normalized_frequency, square_four_wave_band_diagram


def test_square_band_diagram_contains_gamma_and_four_tracked_branches() -> None:
    coupling = np.diag(np.asarray((-300.0, -100.0, 120.0, 400.0), dtype=complex))
    diagram = square_four_wave_band_diagram(
        coupling,
        lattice_constant_nm=300.0,
        bragg_wavelength_nm=1000.0,
        effective_index=10.0 / 3.0,
        q_max=0.01,
        points=21,
    )
    assert diagram.normalized_frequency.shape == (4, 21)
    assert np.isclose(diagram.wave_number_2pi_over_a[10], 0.0)
    assert (diagram.left_endpoint, diagram.right_endpoint) == ("M", "X")
    expected = 0.3 + np.diag(coupling).real * 300e-9 / (2.0 * np.pi * (10.0 / 3.0))
    assert np.allclose(diagram.normalized_frequency[:, 10], expected)
    assert np.all(diagram.radiation_constant_cm >= 0.0)


def test_group_index_conversion_matches_positive_time_envelope_frequency():
    from scipy.constants import c
    lattice_nm, wavelength_nm, neff, ng = 300.0, 1000.0, 3.0, 3.6
    detuning = np.asarray([-2000.0, 0.0, 4000.0])
    actual = detuning_to_normalized_frequency(detuning, lattice_nm, wavelength_nm, neff, ng)
    frequency_hz = c/(wavelength_nm*1e-9) + (c/ng)*detuning/(2*np.pi)
    assert np.allclose(actual, frequency_hz*(lattice_nm*1e-9)/c, rtol=1e-14)
    assert actual[-1] > lattice_nm/wavelength_nm  # +i vg delta is a blue shift.


def test_square_band_plot_uses_supplied_group_index():
    coupling = np.diag(np.asarray([-2000., -500., 1000., 3000.], complex))
    diagram = square_four_wave_band_diagram(coupling, 300., 1000., 3.0, points=21, group_index=3.6)
    beta0=2*np.pi/300e-9; beta_ref=2*np.pi*3./1000e-9
    delta_ref=(beta_ref**2-beta0**2)/(2*beta0)
    expected = .3+(np.diag(coupling).real-delta_ref)*300e-9/(2*np.pi*3.6)*(beta0/beta_ref)
    assert np.allclose(diagram.normalized_frequency[:, 10], expected)


def test_fixed_frequency_cwt_root_maps_to_the_reference_frequency():
    # A root of delta_bg(lambda)=Re(mu) must map to lambda itself. Adding
    # mu to a/lambda a second time would double-count its physical detuning.
    period_nm,wavelength_nm,neff,ng=601.,1030.,1.708,1.85
    beta0=2*np.pi/(period_nm*1e-9); beta_ref=2*np.pi*neff/(wavelength_nm*1e-9)
    delta_ref=(beta_ref**2-beta0**2)/(2*beta0)
    coupling=np.diag(delta_ref+np.array([-100.,0.,200.,400.])+1j)
    diagram=square_four_wave_band_diagram(coupling,period_nm,wavelength_nm,neff,group_index=ng,points=21)
    assert diagram.normalized_frequency[1,10]==pytest.approx(period_nm/wavelength_nm,rel=1e-14)


def test_squared_detuning_frequency_derivative_is_used_off_bragg():
    # Independent finite difference of delta(omega) in a constant-index host.
    from scipy.constants import c
    period_nm,wavelength_nm,n=600.,1040.,1.72
    beta0=2*np.pi/(period_nm*1e-9); omega=2*np.pi*c/(wavelength_nm*1e-9)
    shifted=omega*(1+1e-7)
    delta=(n*omega/c)**2/(2*beta0)-beta0/2
    delta_shift=(n*shifted/c)**2/(2*beta0)-beta0/2
    plotted=detuning_to_normalized_frequency(np.array([delta,delta_shift]),period_nm,wavelength_nm,n,n,reference_wavenumber_per_m=beta0)
    observed=plotted[1]-plotted[0]; exact=(shifted-omega)*(period_nm*1e-9)/(2*np.pi*c)
    assert abs(observed/exact-1)<1e-6

