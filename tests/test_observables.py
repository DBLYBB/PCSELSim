import numpy as np
import pytest

from pcselsim.observables import incoherent_wavelength_spectrum, wavelength_spectrum


def test_materially_negative_radiation_is_rejected_instead_of_clipped():
    from pcselsim.config import OpticalConfig
    from pcselsim.observables import radiated_power_W
    field = np.ones((4, 3, 3), dtype=complex)
    with pytest.raises(ValueError, match="negative radiated-power"):
        radiated_power_W(field, -1j*np.eye(4), 1e-6, OpticalConfig())


def test_passive_radiated_power_matches_analytic_energy_decay():
    from pcselsim.config import OpticalConfig
    from pcselsim.constants import c, epsilon_0
    from pcselsim.observables import radiated_power_W
    optical = OpticalConfig()
    field = np.arange(1, 37).reshape(4, 3, 3).astype(complex)
    alpha, dx = 27.0, 2e-6
    # d/dt[2 eps0 neff ng |Phi|^2] = -2 vg alpha times that energy.
    energy = 2*epsilon_0*optical.effective_index*optical.group_index*np.sum(abs(field)**2)*dx**2
    expected = 2*(c/optical.group_index)*alpha*energy
    assert radiated_power_W(field, 1j*alpha*np.eye(4), dx, optical) == pytest.approx(expected)


def test_roundoff_negative_radiation_is_allowed_against_matrix_scale():
    from pcselsim.config import OpticalConfig
    from pcselsim.observables import radiated_power_W
    # A Hermitian real matrix has zero radiation; a 1e-15 imaginary diagonal
    # is below the explicit matrix-scale roundoff allowance.
    coupling = np.eye(4, dtype=complex)*(1-1e-15j)
    assert radiated_power_W(np.ones((4, 3, 3), complex), coupling, 1e-6, OpticalConfig()) == 0.0


def test_reference_frequency_coherent_line_is_preserved() -> None:
    wavelength, power = wavelength_spectrum(np.ones(128, complex), 1e-12, 1030.0, window=False)
    assert wavelength[np.argmax(power)] == pytest.approx(1030.0)
    assert power.max() == pytest.approx(1.0)


def test_complex_frequency_offset_uses_paper_positive_time_convention() -> None:
    count, sample_s = 256, 1e-12
    offset_hz = 7 / (count * sample_s)
    signal = np.exp(2j * np.pi * offset_hz * np.arange(count) * sample_s)
    wavelength, power = wavelength_spectrum(signal, sample_s, 950.65, window=False)
    from scipy.constants import c
    expected_nm = c / (c / 950.65e-9 + offset_hz) * 1e9
    assert wavelength[np.argmax(power)] == pytest.approx(expected_nm)


@pytest.mark.parametrize("signal,dt,wavelength", [(np.ones(1), 1e-12, 950.0), (np.ones(8), 0.0, 950.0), (np.ones(8), 1e-12, -1.0)])
def test_invalid_spectrum_inputs_are_rejected(signal, dt, wavelength) -> None:
    with pytest.raises(ValueError):
        wavelength_spectrum(signal, dt, wavelength)


def test_local_probe_spectrum_recovers_odd_modes_cancelled_by_spatial_integral():
    count, dt_s = 512, 1e-12
    time = np.arange(count)*dt_s
    first = np.exp(2j*np.pi*7/(count*dt_s)*time)
    second = .5*np.exp(-2j*np.pi*11/(count*dt_s)*time)
    # Opposite spatial parity cancels the entire coherent aperture integral.
    probes = np.column_stack((first+second, -first-second))[:, :, None]
    _, coherent = wavelength_spectrum(probes.sum(axis=(1, 2)), dt_s, 950., window=False)
    wavelength, power = incoherent_wavelength_spectrum(probes, dt_s, 950., window=False)
    from scipy.constants import c
    expected = c/(c/950e-9 + np.asarray((7., -11.))/(count*dt_s))*1e9
    assert np.max(coherent) == 0.0
    for target, strength in zip(expected, (1.0, .25)):
        index = np.argmin(abs(wavelength-target))
        assert wavelength[index] == pytest.approx(target)
        assert power[index] == pytest.approx(strength)
    assert np.count_nonzero(power > 1e-10) == 2
