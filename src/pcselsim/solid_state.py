"""Exploratory quasi-three-level Yb:YAG mean-field rate model.

English: this module replaces semiconductor carriers with one spatially
averaged Yb excited-state reservoir, including ground-state reabsorption.  It
includes separate one-dimensional unlased pump-propagation helpers, but does
**not** solve three-dimensional pump/lasing propagation, temperature, spatial hole burning,
up-conversion, stress, or a validated Yb:YAG photonic-crystal vertical mode;
therefore its outputs are feasibility trends, not quantitative predictions.

中文：本模块用一个空间平均的 Yb 激发态粒子数库替换半导体载流子，并包含基态
再吸收。另有一维起振前单端/双端饱和泵浦传播辅助函数，当前尚未联合求解三维泵浦与
激光传播、热效应、空间烧孔、上转换、应力，也没有经过实验
验证的 Yb:YAG 光子晶体纵向模，因此输出只能视为可行性趋势，不能作为定量设计值。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

from .constants import c, hbar, pi


@dataclass(frozen=True)
class YbYAGMediumConfig:
    """Editable spectroscopic and pump inputs / 可修改的光谱与泵浦参数。"""
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

    def __post_init__(self) -> None:
        positive = (
            self.pump_wavelength_nm, self.laser_wavelength_nm,
            self.refractive_index, self.upper_state_lifetime_ms,
            self.dopant_density_cm3, self.pump_absorption_cross_section_cm2,
            self.pumped_area_um2, self.gain_thickness_um, self.end_time_ms,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Wavelength, density, lifetime, volume and duration must be positive")
        if not 0.0 < self.confinement_factor <= 1.0:
            raise ValueError("confinement_factor must be in (0, 1]")
        if not 0.0 <= self.spontaneous_emission_factor <= 1.0:
            raise ValueError("spontaneous_emission_factor must be in [0, 1]")
        nonnegative = (
            self.pump_emission_cross_section_cm2, self.laser_absorption_cross_section_cm2,
            self.laser_emission_cross_section_cm2, self.pump_intensity_W_cm2,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError("Cross sections and pump intensity cannot be negative")
        if self.laser_emission_cross_section_cm2 + self.laser_absorption_cross_section_cm2 <= 0.0:
            raise ValueError("At least one laser cross section must be positive")
        if self.samples < 2:
            raise ValueError("samples must be at least two")


@dataclass(frozen=True)
class YbYAGRateResult:
    time_ms: np.ndarray
    excited_fraction: np.ndarray
    photon_density_m3: np.ndarray
    output_power_W: np.ndarray
    mode_names: tuple[str, ...]
    threshold_fractions: np.ndarray


@dataclass(frozen=True)
class YbYAGPumpProfile:
    distance_um: np.ndarray
    intensity_W_cm2: np.ndarray
    unlased_excited_fraction: np.ndarray
    net_absorbed_fraction: float


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
    """Net gain including stimulated emission and ground-state reabsorption.

    净增益为 ``sigma_e*N2 - sigma_a*N1``，因此低反转时允许为负。
    """
    values = _medium_si(config)
    fraction = np.asarray(excited_fraction)
    n2 = values["nt"]*fraction
    n1 = values["nt"]-n2
    return values["sigma_el"]*n2-values["sigma_al"]*n1


def threshold_excited_fraction(loss_per_m: np.ndarray, config: YbYAGMediumConfig):
    """Solve ``Gamma*g(f_th)=alpha_total`` for each mode / 求各模式阈值反转率。"""
    values = _medium_si(config)
    numerator = loss_per_m/config.confinement_factor/values["nt"] + values["sigma_al"]
    denominator = values["sigma_el"]+values["sigma_al"]
    return numerator/denominator


def pump_limited_excited_fraction(config: YbYAGMediumConfig,
                                 intensity_W_cm2: np.ndarray | float | None = None):
    """Unlased steady inversion at the *local* pump intensity / 局域泵浦反转上限。

    Pump stimulated emission limits f even at infinite intensity:
    f(infinity) = sigma_ap/(sigma_ap + sigma_ep), generally below one.
    """
    values = _medium_si(config)
    intensity = config.pump_intensity_W_cm2 if intensity_W_cm2 is None else intensity_W_cm2
    flux = np.asarray(intensity) * 1e4 / values["pump_energy"]
    return values["sigma_ap"] * flux / (
        (values["sigma_ap"] + values["sigma_ep"]) * flux + 1.0 / values["tau"]
    )


def threshold_pump_intensity_W_cm2(loss_per_m: np.ndarray, config: YbYAGMediumConfig):
    """Required local intensity; infinity means unreachable with this pump line."""
    values = _medium_si(config)
    fraction = threshold_excited_fraction(np.asarray(loss_per_m), config)
    pump_coefficient = values["sigma_ap"] * (1.0 - fraction) - values["sigma_ep"] * fraction
    result = np.full_like(fraction, np.inf, dtype=float)
    valid = (fraction >= 0.0) & (fraction < 1.0) & (pump_coefficient > 0.0)
    np.divide(
        fraction * values["pump_energy"] / values["tau"] / 1e4,
        pump_coefficient, out=result, where=valid,
    )
    return result


def solve_unlased_pump_propagation(config: YbYAGMediumConfig, path_length_um: float,
                                  samples: int = 200) -> YbYAGPumpProfile:
    """One-dimensional saturated pump absorption before lasing / 起振前饱和泵浦传播。

    dI/dz=-Nt[sigma_ap(1-f)-sigma_ep*f]I, with the local unlased steady
    inversion. This assumes uniform dopant density, no focusing, no scattering,
    and no reflections; it does not determine a spatial lasing inversion.
    """
    if not np.isfinite(path_length_um) or path_length_um <= 0.0 or samples < 2:
        raise ValueError("Pump path must be positive and samples at least two")
    grid = np.linspace(0.0, path_length_um, samples)
    if config.pump_intensity_W_cm2 == 0.0:
        return YbYAGPumpProfile(grid, np.zeros(samples), np.zeros(samples), 0.0)

    def rhs(_distance_um: float, transmission: np.ndarray) -> np.ndarray:
        intensity = config.pump_intensity_W_cm2 * np.maximum(transmission, 0.0)
        fraction = pump_limited_excited_fraction(config, intensity)
        alpha_cm = config.dopant_density_cm3 * (
            config.pump_absorption_cross_section_cm2 * (1.0 - fraction)
            - config.pump_emission_cross_section_cm2 * fraction
        )
        return -alpha_cm * transmission * 1e-4

    result = solve_ivp(rhs, (0.0, path_length_um), [1.0],
                       t_eval=grid, rtol=2e-9, atol=1e-11)
    if not result.success:
        raise RuntimeError(f"Pump propagation failed: {result.message}")
    intensities = config.pump_intensity_W_cm2 * np.maximum(result.y[0], 0.0)
    return YbYAGPumpProfile(
        distance_um=grid, intensity_W_cm2=intensities,
        unlased_excited_fraction=pump_limited_excited_fraction(config, intensities),
        net_absorbed_fraction=float(1.0 - intensities[-1] / config.pump_intensity_W_cm2),
    )


def solve_bidirectional_unlased_pump(config: YbYAGMediumConfig, path_length_um: float,
                                     samples: int = 200) -> YbYAGPumpProfile:
    """Equal split of the SAME total input intensity between opposite ends.

    The local inversion depends on I_forward+I_backward. Boundary conditions
    are I_forward(0)=I_backward(L)=I_total/2. Reflective coatings, focusing and
    lasing depletion are still excluded. 中文：保持总泵功率不变的等分双端泵浦。
    """
    if not np.isfinite(path_length_um) or path_length_um <= 0.0 or samples < 2:
        raise ValueError("Pump path must be positive and samples at least two")
    grid = np.linspace(0.0, path_length_um, samples)
    if config.pump_intensity_W_cm2 == 0.0:
        return YbYAGPumpProfile(grid, np.zeros(samples), np.zeros(samples), 0.0)

    def rhs(_distance: np.ndarray, transmissions: np.ndarray) -> np.ndarray:
        local_intensity = config.pump_intensity_W_cm2 * np.maximum(transmissions.sum(axis=0), 0.0)
        fraction = pump_limited_excited_fraction(config, local_intensity)
        alpha_cm = config.dopant_density_cm3 * (
            config.pump_absorption_cross_section_cm2 * (1.0 - fraction)
            - config.pump_emission_cross_section_cm2 * fraction
        )
        return np.vstack((-alpha_cm * transmissions[0], alpha_cm * transmissions[1])) * 1e-4

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.array([left[0] - 0.5, right[1] - 0.5])

    alpha_small = config.dopant_density_cm3 * config.pump_absorption_cross_section_cm2
    initial = 0.5 * np.vstack((
        np.exp(-alpha_small * grid * 1e-4),
        np.exp(-alpha_small * (path_length_um - grid) * 1e-4),
    ))
    result = solve_bvp(rhs, boundary, grid, initial, tol=2e-7, max_nodes=10_000)
    if not result.success:
        raise RuntimeError(f"Bidirectional pump boundary solve failed: {result.message}")
    transmissions = result.sol(grid)
    intensities = config.pump_intensity_W_cm2 * np.maximum(transmissions.sum(axis=0), 0.0)
    return YbYAGPumpProfile(
        distance_um=grid, intensity_W_cm2=intensities,
        unlased_excited_fraction=pump_limited_excited_fraction(config, intensities),
        net_absorbed_fraction=float(1.0 - transmissions[0, -1] - transmissions[1, 0]),
    )


def ybyag_rate_rhs(state: np.ndarray, config: YbYAGMediumConfig,
                   total_loss_per_m: np.ndarray) -> np.ndarray:
    """Conserve excitation + photon counts under one consistent normalization.

    S_m=N_photon,m/V_gain, *not* density per full optical mode volume.
    Thus stimulated depletion and photon generation both contain Gamma.
    中文：光子数统一除以有源体积，反转消耗和光子增加必须含相同的Gamma。
    """
    values = _medium_si(config)
    fraction = float(np.clip(state[0], 0.0, 1.0))
    photons = np.maximum(state[1:], 0.0)
    material_gain = float(gain_per_m(fraction, config))
    flux = values["pump_intensity"] / values["pump_energy"]
    stimulated = config.confinement_factor * values["vg"] * material_gain * photons
    df_dt = (
        values["sigma_ap"] * flux * (1.0 - fraction)
        - values["sigma_ep"] * flux * fraction - fraction / values["tau"]
        - np.sum(stimulated) / values["nt"]
    )
    spontaneous = config.spontaneous_emission_factor * values["nt"] * fraction / values["tau"]
    dphotons_dt = (
        stimulated - values["vg"] * np.asarray(total_loss_per_m) * photons
        + spontaneous / len(photons)
    )
    return np.concatenate(([df_dt], dphotons_dt))


def ybyag_energy_budget(state: np.ndarray, config: YbYAGMediumConfig,
                        total_loss_per_m: np.ndarray,
                        output_loss_per_m: np.ndarray) -> dict[str, float]:
    """Instantaneous count/energy closure for the local uniform pump model.

    Excitation energy is represented by h*nu_l. The pump quantum defect is
    reported separately. Pump transport and thermal escape remain external.
    """
    values = _medium_si(config)
    fraction = float(state[0])
    photons = np.asarray(state[1:])
    flux = values["pump_intensity"] / values["pump_energy"]
    pump_rate = (
        values["sigma_ap"] * flux * (1.0 - fraction)
        - values["sigma_ep"] * flux * fraction
    ) * values["nt"] * values["volume"]
    absorbed = values["pump_energy"] * pump_rate
    excitation_source = values["laser_energy"] * pump_rate
    fluorescence = (
        (1.0 - config.spontaneous_emission_factor) * values["laser_energy"]
        * values["nt"] * fraction * values["volume"] / values["tau"]
    )
    cavity_loss = (
        values["laser_energy"] * values["vg"] * values["volume"]
        * float(np.sum(np.asarray(total_loss_per_m) * photons))
    )
    output = (
        values["laser_energy"] * values["vg"] * values["volume"]
        * float(np.sum(np.asarray(output_loss_per_m) * photons))
    )
    derivatives = ybyag_rate_rhs(state, config, total_loss_per_m)
    stored_energy_rate = values["laser_energy"] * values["volume"] * (
        values["nt"] * derivatives[0] + np.sum(derivatives[1:])
    )
    return {
        "incident_pump_power_W": config.pump_intensity_W_cm2 * config.pumped_area_um2 * 1e-8,
        "net_local_absorbed_pump_power_W": absorbed,
        "pump_quantum_defect_heat_W": absorbed - excitation_source,
        "fluorescence_outside_retained_modes_W": fluorescence,
        "total_cavity_loss_power_W": cavity_loss,
        "retained_mode_output_power_W": output,
        "stored_excitation_plus_photon_energy_rate_W": float(stored_energy_rate),
        "energy_closure_residual_W": float(stored_energy_rate - excitation_source + fluorescence + cavity_loss),
    }


def solve_ybyag_rates(config: YbYAGMediumConfig, mode_names: tuple[str, ...],
                      total_loss_per_m: np.ndarray, output_loss_per_m: np.ndarray) -> YbYAGRateResult:
    """Integrate one inversion reservoir coupled to multiple PCSEL modes.

    Photon density is N_photon/V_gain, so the stimulated terms in BOTH
    population and photon equations contain Gamma. N2 is shared by all modes
    (mean-field approximation). This is appropriate
    for first validation of threshold and mode competition; a later 2-D N2(x,y)
    model is needed for spatial hole burning in a real pumped crystal.

    所有模式共享同一个平均反转率；真实晶体中的泵浦吸收和空间烧孔需要进一步使用
    二维或三维 ``N2(x,y,z)`` 模型。
    """
    values = _medium_si(config)
    count = len(mode_names)
    total_loss_per_m = np.asarray(total_loss_per_m, dtype=float)
    output_loss_per_m = np.asarray(output_loss_per_m, dtype=float)
    if count == 0 or total_loss_per_m.shape != (count,) or output_loss_per_m.shape != (count,):
        raise ValueError("Each mode must have one total and one output power loss")
    if (not np.isfinite(total_loss_per_m).all() or not np.isfinite(output_loss_per_m).all()
            or np.any(total_loss_per_m < 0.0) or np.any(output_loss_per_m < 0.0)
            or np.any(output_loss_per_m > total_loss_per_m)):
        raise ValueError("Losses must be finite with 0 <= output_loss <= total_loss")
    threshold = threshold_excited_fraction(total_loss_per_m, config)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        return ybyag_rate_rhs(state, config, total_loss_per_m)

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
    """Analytic steady-state scan assuming one winning mode.

    该扫描假设单模胜出并钳位反转率，适合查阈值量级，不描述多模拍频。
    """
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
        photons = max(numerator/(config.confinement_factor*values["vg"]*material_gain), 0.0)
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
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=7, ncol=2)
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
    if axes[0].get_legend_handles_labels()[0]:
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
