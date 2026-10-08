import numpy as np
import pytest

from pcselsim.observables import wavelength_spectrum


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
