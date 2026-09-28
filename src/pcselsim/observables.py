"""Observables corresponding to Eqs. (9) and (10) of Inoue et al."""

from __future__ import annotations

import numpy as np

from .config import CarrierConfig, OpticalConfig
from .constants import c, epsilon_0, hbar, pi


def photon_density_m3(
    field: np.ndarray, carrier: CarrierConfig, optical: OpticalConfig
) -> np.ndarray:
    """Photon density U in the active layer (Eq. 10), in m^-3."""
    omega = 2.0 * pi * c / (optical.wavelength_nm * 1e-9)
    thickness = carrier.active_thickness_nm * 1e-9
    coefficient = (
        optical.confinement_factor
        * 2.0
        * epsilon_0
        * optical.effective_index
        * optical.group_index
        / (hbar * omega * thickness)
    )
    return coefficient * np.sum(np.abs(field) ** 2, axis=0)


def radiated_power_W(
    field: np.ndarray, coupling_m: np.ndarray, dx_m: float, optical: OpticalConfig
) -> float:
    """Surface-radiated power from Eq. (9)."""
    c_phi = np.einsum("ab,bij->aij", coupling_m, field, optimize=True)
    quadratic = np.sum(np.conj(field) * c_phi, axis=0)
    density = 4.0 * epsilon_0 * optical.effective_index * c * np.imag(quadratic)
    return float(np.sum(np.maximum(density, 0.0)) * dx_m * dx_m)


def wavelength_spectrum(
    signal: np.ndarray, dt_s: float, center_wavelength_nm: float, window: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return wavelength and normalized FFT power, sorted by wavelength."""
    signal = np.asarray(signal, dtype=np.complex128)
    signal = signal - np.mean(signal)
    if window:
        signal = signal * np.hanning(signal.size)
    spectrum = np.abs(np.fft.fft(signal)) ** 2
    frequency_offset = np.fft.fftfreq(signal.size, dt_s)
    center_frequency = c / (center_wavelength_nm * 1e-9)
    valid = center_frequency + frequency_offset > 0.0
    wavelength_nm = c / (center_frequency + frequency_offset[valid]) * 1e9
    power = spectrum[valid]
    if np.max(power) > 0.0:
        power = power / np.max(power)
    order = np.argsort(wavelength_nm)
    return wavelength_nm[order], power[order]

