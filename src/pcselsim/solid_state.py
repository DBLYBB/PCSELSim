"""Quasi-three-level Yb:YAG rate equations coupled to PCSEL band-edge modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from .constants import c, hbar, pi


@dataclass(frozen=True)
class YbYAGMediumConfig:
    pump_wavelength_nm: float = 940.0
    laser_wavelength_nm: float = 1030.0
    refractive_index: float = 1.82
    upper_state_lifetime_ms: float = 0.95
    dopant_density_cm3: float = 1.38e20
    pump_absorption_cross_section_cm2: float = 0.70e-20
    pump_emission_cross_section_cm2: float = 0.10e-20
    laser_absorption_cross_section_cm2: float = 0.126e-20
    laser_emission_cross_section_cm2: float = 2.0e-20
    confinement_factor: float = 0.80
    spontaneous_emission_factor: float = 1.0e-8
    pump_intensity_W_cm2: float = 2.5e4
    pumped_area_um2: float = 200.0**2
    gain_thickness_um: float = 200.0
    end_time_ms: float = 5.0
    samples: int = 1000


@dataclass(frozen=True)
class YbYAGRateResult:
    time_ms: np.ndarray
    excited_fraction: np.ndarray
    photon_density_m3: np.ndarray
    output_power_W: np.ndarray
    mode_names: tuple[str, ...]
    threshold_fractions: np.ndarray


def _medium_si(config: YbYAGMediumConfig) -> dict[str, float]:
    return {
        "nt": config.dopant_density_cm3*1e6,
        "sigma_ap": config.pump_absorption_cross_section_cm2*1e-4,
        "sigma_ep": config.pump_emission_cross_section_cm2*1e-4,
        "sigma_al": config.laser_absorption_cross_section_cm2*1e-4,
        "sigma_el": config.laser_emission_cross_section_cm2*1e-4,
        "tau": config.upper_state_lifetime_ms*1e-3,
        "pump_energy": 2*pi*hbar*c/(config.pump_wavelength_nm*1e-9),
        "laser_energy": 2*pi*hbar*c/(config.laser_wavelength_nm*1e-9),
        "pump_intensity": config.pump_intensity_W_cm2*1e4,
        "vg": c/config.refractive_index,
        "volume": config.pumped_area_um2*1e-12*config.gain_thickness_um*1e-6,
    }


def gain_per_m(excited_fraction: np.ndarray | float, config: YbYAGMediumConfig):
    """Quasi-three-level net material gain including ground-state reabsorption."""
    values = _medium_si(config)
    fraction = np.asarray(excited_fraction)
    n2 = values["nt"]*fraction
    n1 = values["nt"]-n2
    return values["sigma_el"]*n2-values["sigma_al"]*n1


def threshold_excited_fraction(loss_per_m: np.ndarray, config: YbYAGMediumConfig):
    values = _medium_si(config)
    numerator = loss_per_m/config.confinement_factor/values["nt"] + values["sigma_al"]
    denominator = values["sigma_el"]+values["sigma_al"]
    return numerator/denominator


def solve_ybyag_rates(config: YbYAGMediumConfig, mode_names: tuple[str, ...],
                      total_loss_per_m: np.ndarray, output_loss_per_m: np.ndarray) -> YbYAGRateResult:
    """Integrate one inversion reservoir coupled to multiple PCSEL modes.

    N2 is shared by all modes (mean-field approximation).  This is appropriate
    for first validation of threshold and mode competition; a later 2-D N2(x,y)
    model is needed for spatial hole burning in a real pumped crystal.
    """
    values = _medium_si(config)
    count = len(mode_names)
    threshold = threshold_excited_fraction(total_loss_per_m, config)
    pump_flux = values["pump_intensity"]/values["pump_energy"]
    pump_abs = values["sigma_ap"]*pump_flux
    pump_em = values["sigma_ep"]*pump_flux

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        fraction = float(np.clip(state[0], 0.0, 1.0))
        photons = np.maximum(state[1:], 0.0)
        material_gain = float(gain_per_m(fraction, config))
        stimulated_fraction_rate = (
            values["vg"]*material_gain*np.sum(photons)/values["nt"]
        )
        df_dt = (
            pump_abs*(1.0-fraction)
            - pump_em*fraction
            - fraction/values["tau"]
            - stimulated_fraction_rate
        )
        net_rates = values["vg"]*(config.confinement_factor*material_gain-total_loss_per_m)
        spontaneous = (
            config.spontaneous_emission_factor*values["nt"]*fraction
            / values["tau"]/count
        )
        dphotons_dt = net_rates*photons+spontaneous
        return np.concatenate(([df_dt], dphotons_dt))

    initial = np.concatenate(([0.0], np.full(count, 1.0)))
    time_s = np.linspace(0.0, config.end_time_ms*1e-3, config.samples)
    solution = solve_ivp(
        rhs, (time_s[0], time_s[-1]), initial, t_eval=time_s,
        method="BDF", rtol=2e-6, atol=np.concatenate(([1e-10], np.full(count, 1.0))),
    )
    if not solution.success:
        raise RuntimeError(f"Yb:YAG rate integration failed: {solution.message}")
    fraction = np.clip(solution.y[0], 0.0, 1.0)
    photons = np.maximum(solution.y[1:], 0.0)
    power = (
        values["laser_energy"]*values["vg"]*output_loss_per_m[:, None]
        * photons*values["volume"]
    )
    return YbYAGRateResult(
        time_ms=solution.t*1e3,
        excited_fraction=fraction,
        photon_density_m3=photons,
        output_power_W=power,
        mode_names=mode_names,
        threshold_fractions=threshold,
    )


def pump_scan(config: YbYAGMediumConfig, total_loss_per_m: np.ndarray,
              output_loss_per_m: np.ndarray, points: int = 80):
    """Analytic steady-state single-winning-mode pump scan."""
    values = _medium_si(config)
    threshold = threshold_excited_fraction(total_loss_per_m, config)
    winner = int(np.argmin(threshold))
    intensities = np.linspace(0.0, 2.5*config.pump_intensity_W_cm2, points)
    fractions = np.zeros(points)
    powers = np.zeros(points)
    for index, intensity_cm in enumerate(intensities):
        flux = intensity_cm*1e4/values["pump_energy"]
        w_abs = values["sigma_ap"]*flux
        w_em = values["sigma_ep"]*flux
        no_laser = w_abs/(w_abs+w_em+1.0/values["tau"])
        if no_laser <= threshold[winner] or threshold[winner] >= 1.0:
            fractions[index] = no_laser
            continue
        fractions[index] = threshold[winner]
        numerator = (
            w_abs*(1.0-fractions[index])-w_em*fractions[index]
            - fractions[index]/values["tau"]
        )*values["nt"]
        material_gain = float(gain_per_m(fractions[index], config))
        photons = max(numerator/(values["vg"]*material_gain), 0.0)
        powers[index] = (
            values["laser_energy"]*values["vg"]*output_loss_per_m[winner]
            * photons*values["volume"]
        )
    return intensities, powers, fractions, winner


def plot_gain_curve(config: YbYAGMediumConfig, loss_per_m: np.ndarray,
                    mode_names: tuple[str, ...], path: Path) -> None:
    fraction = np.linspace(0.0, 1.0, 400)
    gain_cm = gain_per_m(fraction, config)/100.0
    threshold = threshold_excited_fraction(loss_per_m, config)
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.plot(fraction, gain_cm, lw=2, label="Yb:YAG net gain")
    for name, value, loss in zip(mode_names, threshold, loss_per_m, strict=True):
        if 0.0 <= value <= 1.0:
            ax.axvline(value, ls="--", lw=1, label=f"{name} threshold f={value:.3f}")
            ax.plot(value, loss/config.confinement_factor/100.0, "o")
        else:
            ax.plot([], [], ls="--", label=f"{name}: unreachable f={value:.3f}")
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set(xlabel="excited-state fraction N2/Nt", ylabel=r"material gain (cm$^{-1}$)",
           title="Yb:YAG quasi-three-level gain and reabsorption")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_rate_dynamics(result: YbYAGRateResult, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True)
    axes[0].plot(result.time_ms, result.excited_fraction, color="tab:blue")
    axes[0].set_ylabel("N2/Nt")
    for name, value in zip(result.mode_names, result.threshold_fractions, strict=True):
        if 0.0 <= value <= 1.0:
            axes[0].axhline(value, ls=":", lw=0.8, label=f"{name} threshold")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(fontsize=7, ncol=2)
    for index, name in enumerate(result.mode_names):
        axes[1].semilogy(result.time_ms, np.maximum(result.photon_density_m3[index], 1.0), label=name)
        axes[2].plot(result.time_ms, result.output_power_W[index], label=name)
    axes[1].set_ylabel(r"photon density (m$^{-3}$)")
    axes[2].set_ylabel("surface/output power (W)")
    axes[2].set_xlabel("time (ms)")
    axes[1].legend(ncol=4)
    axes[2].legend(ncol=4)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Yb:YAG mean-field rate-equation dynamics")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_pump_scan(config: YbYAGMediumConfig, total_loss_per_m: np.ndarray,
                   output_loss_per_m: np.ndarray, mode_names: tuple[str, ...], path: Path) -> None:
    intensity, power, fraction, winner = pump_scan(config, total_loss_per_m, output_loss_per_m)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))
    axes[0].plot(intensity/1e3, power)
    axes[0].set(xlabel=r"pump intensity (kW cm$^{-2}$)", ylabel="output power (W)",
                title=f"predicted winning mode: {mode_names[winner]}")
    axes[1].plot(intensity/1e3, fraction)
    axes[1].axhline(threshold_excited_fraction(total_loss_per_m, config)[winner], ls="--",
                    color="tab:red", label="lasing clamp")
    axes[1].set(xlabel=r"pump intensity (kW cm$^{-2}$)", ylabel="N2/Nt",
                title="inversion and threshold clamping")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
