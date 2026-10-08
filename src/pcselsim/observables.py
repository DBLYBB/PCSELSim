"""Physical observables for the Inoue four-wave normalization.

English: photon density follows Eq. (10), radiated power follows Eq. (9), and
spectra are FFTs of a coherent modal projection.  The FFT linewidth is limited
by the simulated time window and by the approximate spontaneous-noise model.

中文：光子密度对应 Inoue 式 (10)，面辐射功率对应式 (9)，频谱由选定模式的
相干投影做 FFT 得到。谱线宽度同时受时间窗与近似自发辐射噪声限制，不能直接
当作实验绝对线宽。
"""

from __future__ import annotations

import numpy as np

from .config import CarrierConfig, OpticalConfig
from .constants import c, epsilon_0, hbar, pi


def photon_density_m3(
    field: np.ndarray, carrier: CarrierConfig, optical: OpticalConfig
) -> np.ndarray:
    """Photon density ``U`` in the active layer (Eq. 10), in m^-3.

    The ``n_eff*n_g`` factor is part of the paper's field normalization.
    ``n_eff*n_g`` 来自论文的光场归一化，不能任意删去或重复计入。
    """
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
    """Surface-radiated power from Eq. (9) / 由式 (9) 积分得到面发射功率。"""
    c_phi = np.einsum("ab,bij->aij", coupling_m, field, optimize=True)
    quadratic = np.sum(np.conj(field) * c_phi, axis=0)
    density = 4.0 * epsilon_0 * optical.effective_index * c * np.imag(quadratic)
    return float(np.sum(np.maximum(density, 0.0)) * dx_m * dx_m)


def wavelength_spectrum(
    signal: np.ndarray,
    dt_s: float,
    center_wavelength_nm: float,
    window: bool = True,
    remove_mean: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized FFT power versus wavelength, sorted by wavelength.

    Use a sufficiently small saved-sample interval to avoid aliasing modal
    detunings. A constant complex envelope is a real coherent line at the
    reference optical frequency, so it must be retained by default. Mean
    removal is an optional analysis operation, not the physical spectrum.
    保存采样间隔必须足够小，否则带边失谐会发生混叠。恒定复包络对应参考光频
    处的真实相干谱线，默认保留；减均值只作为可选数据分析操作。
    """
    signal = np.asarray(signal, dtype=np.complex128)
    if signal.ndim != 1 or signal.size < 2:
        raise ValueError("signal must be a one-dimensional array with at least two samples")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if not np.isfinite(center_wavelength_nm) or center_wavelength_nm <= 0.0:
        raise ValueError("center_wavelength_nm must be finite and positive")
    if remove_mean:
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

