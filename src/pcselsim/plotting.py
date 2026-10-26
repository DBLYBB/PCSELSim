"""Plots for the Inoue time-domain workflow / Inoue 时域结果绘图。

Plotting never changes solver arrays.  Axis limits and normalization are visual
choices and must not be confused with physics parameters. / 绘图不会修改求解结果；
坐标裁剪与归一化只是显示设置，不是器件物理参数。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .observables import incoherent_wavelength_spectrum, wavelength_spectrum
from .solver import SimulationResult


def plot_transient(result: SimulationResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.2, 5.3), constrained_layout=True)
    axes[0].plot(result.time_ns, result.center_carrier_cm3 / 1e18, color="#1f77b4")
    axes[0].set_ylabel(r"Center $N$ ($10^{18}$ cm$^{-3}$)")
    axes[1].plot(result.time_ns, result.power_W, color="#d62728")
    axes[1].set_ylabel("Radiated power (W)")
    axes[1].set_xlabel("Time (ns)")
    axes[0].set_title(f"I = {result.current_ratio:.2f} I_ref")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_spatial(result: SimulationResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extent = [result.x_um[0], result.x_um[-1], result.y_um[0], result.y_um[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    im0 = axes[0].imshow(result.final_carrier_cm3, origin="lower", extent=extent, cmap="turbo")
    axes[0].set_title(r"Carrier density (cm$^{-3}$)")
    fig.colorbar(im0, ax=axes[0], shrink=0.82)
    im1 = axes[1].imshow(
        result.final_photon_areal_cm2, origin="lower", extent=extent, cmap="inferno"
    )
    axes[1].set_title(r"Photon areal density (cm$^{-2}$)")
    fig.colorbar(im1, ax=axes[1], shrink=0.82)
    for axis in axes:
        axis.set_xlabel("x (um)")
        axis.set_ylabel("y (um)")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_spectra(
    results: list[SimulationResult],
    path: str | Path,
    window_ns: float,
    center_wavelength_nm: float = 950.65,
    span_nm: float = 0.60,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(results), 2, squeeze=False,
        figsize=(11.0, 2.1 * len(results) + 1.2),
    )
    for row, result in enumerate(results):
        count = max(16, int(round(window_ns * 1e-9 / result.dt_s)))
        signal = result.complex_signal[-count:]
        wavelength, coherent_power = wavelength_spectrum(signal, result.dt_s, center_wavelength_nm)
        probe_power = None
        if result.probe_signals is not None and result.probe_signals.shape[0] >= 2:
            probe_wavelength, probe_power = incoherent_wavelength_spectrum(
                result.probe_signals[-count:], result.dt_s, center_wavelength_nm
            )
            if not np.array_equal(probe_wavelength, wavelength):
                raise ValueError("probe and coherent signal sampling must match")
        half_span = 0.5*span_nm
        mask = (
            (wavelength >= center_wavelength_nm-half_span)
            & (wavelength <= center_wavelength_nm+half_span)
        )
        for column, axis in enumerate(axes[row]):
            visible = mask if column == 0 else np.ones(wavelength.shape, dtype=bool)
            if probe_power is not None:
                axis.plot(wavelength[visible], probe_power[visible], color="#222222", lw=1.0,
                          label="Sampled local probe sum")
            axis.plot(wavelength[visible], coherent_power[visible], color="#d95f02", lw=0.9,
                      ls="--", label="Coherent modal projection")
            axis.text(0.02, 0.78, f"{result.current_ratio:.2f} I_ref", transform=axis.transAxes)
            axis.set_ylabel("Norm. FFT power")
            if column == 0:
                axis.set_xlim(center_wavelength_nm-half_span, center_wavelength_nm+half_span)
            if row == len(results)-1:
                axis.set_xlabel("Wavelength (nm)")
    axes[0, 0].set_title("Reference-frequency view")
    axes[0, 1].set_title("Full represented frequency range")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Sampled-channel spectra; not total emitted spectra (each curve normalized)", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

