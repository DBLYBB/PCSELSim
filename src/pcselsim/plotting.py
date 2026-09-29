"""Plots for the Inoue time-domain workflow / Inoue 时域结果绘图。

Plotting never changes solver arrays.  Axis limits and normalization are visual
choices and must not be confused with physics parameters. / 绘图不会修改求解结果；
坐标裁剪与归一化只是显示设置，不是器件物理参数。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .observables import wavelength_spectrum
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
    axes[0].set_title(f"I = {result.current_ratio:.2f} Ith")
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
    fig, axes = plt.subplots(len(results), 1, sharex=True, figsize=(6.4, 1.8 * len(results) + 1.2))
    axes = np.atleast_1d(axes)
    for axis, result in zip(axes, results):
        count = max(16, int(round(window_ns * 1e-9 / result.dt_s)))
        signal = result.complex_signal[-count:]
        wavelength, power = wavelength_spectrum(signal, result.dt_s, center_wavelength_nm)
        half_span = 0.5*span_nm
        mask = (
            (wavelength >= center_wavelength_nm-half_span)
            & (wavelength <= center_wavelength_nm+half_span)
        )
        visible_power = power[mask].copy()
        if visible_power.size and float(visible_power.max()) > 0.0:
            visible_power /= float(visible_power.max())
        axis.plot(wavelength[mask], visible_power, color="#222222", lw=1.0)
        axis.text(0.02, 0.78, f"{result.current_ratio:.2f} Ith", transform=axis.transAxes)
        axis.set_ylabel("Norm.")
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

