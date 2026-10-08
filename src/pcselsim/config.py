"""Typed configuration objects, unit conventions, and YAML loading.

English: names ending in ``_nm``, ``_um``, ``_ns``, ``_cm`` or ``_cm3`` expose
paper-friendly units; numerical kernels convert them to SI.  Unknown YAML keys
are rejected so spelling errors cannot silently change a simulation.

中文：字段名中的单位就是用户输入单位，数值内核再统一换算为 SI。YAML 中出现
未知字段会立即报错，避免拼写错误被静默忽略。自定义器件时应优先改配置或主程序
顶部参数区，不要在求解器内部写死参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OpticalConfig:
    """Optical inputs for the square-lattice four-wave basis / 四波光学参数。"""
    lattice_constant_nm: float = 277.0
    wavelength_nm: float = 950.65
    group_index: float = 3.513
    effective_index: float = 3.429
    active_index: float = 3.584
    internal_loss_cm: float = 5.0
    confinement_factor: float = 0.044
    dn_dN_cm3: float = -5.0e-21
    # Eigenvalues of C in the A/B/C/D band-edge basis.  Real parts set
    # frequency splitting; imaginary parts set radiative amplitude loss.
    modal_detuning_cm: tuple[float, float, float, float] = (0.0, 260.0, 720.0, 880.0)
    modal_radiation_loss_cm: tuple[float, float, float, float] = (2.5, 8.0, 28.0, 32.0)
    carrier_index_change: bool = True
    temporal_index_term: bool = True


@dataclass(frozen=True)
class CarrierConfig:
    """Semiconductor reservoir parameters used in Inoue Eqs. (10)-(11)."""
    maximum_gain_cm: float = 2000.0
    zero_carrier_gain_cm: float = -5000.0
    transparency_density_cm3: float = 1.5e18
    lifetime_ns: float = 1.5
    diffusion_cm2_s: float = 100.0
    active_thickness_nm: float = 30.0
    spontaneous_emission_factor: float = 1.0e-4


@dataclass(frozen=True)
class DeviceConfig:
    """Finite-area device and electrical-contact geometry / 器件与电极几何。"""
    domain_um: float = 400.0
    electrode_um: float = 300.0
    current_spread_um: float = 25.0
    threshold_current_A: float = 0.7
    electrode_shape: str = "square"


@dataclass(frozen=True)
class NumericsConfig:
    """Discretization controls; these change accuracy, not device physics.

    数值参数会影响稳定性和收敛性，不应被当作可拟合的器件物理量。
    """
    points: int = 41
    end_time_ns: float = 10.0
    cfl: float = 0.8
    sample_interval_ps: float = 5.0
    snapshot_interval_ps: float = 100.0
    seed: int = 19
    carrier_substeps: int = 1
    noise: bool = True


@dataclass(frozen=True)
class ReproductionConfig:
    """Current sweep and spectrum-window controls / 电流扫描及频谱窗口。"""
    current_ratios: tuple[float, ...] = (1.05, 1.4, 2.8, 4.2)
    spectrum_window_ns: float = 4.0


@dataclass(frozen=True)
class SimulationConfig:
    optical: OpticalConfig = field(default_factory=OpticalConfig)
    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)

    def quick(self) -> "SimulationConfig":
        """Return a short smoke-test configuration without changing physics."""
        return replace(
            self,
            numerics=replace(
                self.numerics,
                points=31,
                end_time_ns=min(self.numerics.end_time_ns, 1.2),
                sample_interval_ps=2.0,
                snapshot_interval_ps=100.0,
            ),
            reproduction=replace(self.reproduction, current_ratios=(1.05, 2.8)),
        )


_SECTIONS = {
    "optical": OpticalConfig,
    "carrier": CarrierConfig,
    "device": DeviceConfig,
    "numerics": NumericsConfig,
    "reproduction": ReproductionConfig,
}


def _coerce_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    # YAML 1.1 readers can interpret scientific notation without a decimal
    # exponent sign (for example 1.5e18) as text. Accept it explicitly.
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_config(path: str | Path) -> SimulationConfig:
    """Load a simulation configuration from YAML with unknown-key checking."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown_sections = set(raw) - set(_SECTIONS)
    if unknown_sections:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown_sections)}")
    values: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = raw.get(name, {}) or {}
        known = set(cls.__dataclass_fields__)
        unknown = set(section) - known
        if unknown:
            raise ValueError(f"Unknown keys in {name}: {sorted(unknown)}")
        values[name] = cls(**{key: _coerce_tuple(value) for key, value in section.items()})
    config = SimulationConfig(**values)
    validate_config(config)
    return config


def validate_config(config: SimulationConfig) -> None:
    """Reject nonphysical units/signs before they reach the time integrator."""
    positive = {
        "optical.lattice_constant_nm": config.optical.lattice_constant_nm,
        "optical.wavelength_nm": config.optical.wavelength_nm,
        "optical.group_index": config.optical.group_index,
        "optical.effective_index": config.optical.effective_index,
        "optical.active_index": config.optical.active_index,
        "carrier.maximum_gain_cm": config.carrier.maximum_gain_cm,
        "carrier.transparency_density_cm3": config.carrier.transparency_density_cm3,
        "carrier.lifetime_ns": config.carrier.lifetime_ns,
        "carrier.active_thickness_nm": config.carrier.active_thickness_nm,
        "device.domain_um": config.device.domain_um,
        "device.electrode_um": config.device.electrode_um,
        "device.threshold_current_A": config.device.threshold_current_A,
        "numerics.end_time_ns": config.numerics.end_time_ns,
        "numerics.sample_interval_ps": config.numerics.sample_interval_ps,
        "reproduction.spectrum_window_ns": config.reproduction.spectrum_window_ns,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    nonnegative = {
        "optical.internal_loss_cm": config.optical.internal_loss_cm,
        "carrier.diffusion_cm2_s": config.carrier.diffusion_cm2_s,
        "device.current_spread_um": config.device.current_spread_um,
    }
    for name, value in nonnegative.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0.0 < config.optical.confinement_factor <= 1.0:
        raise ValueError("optical.confinement_factor must be in (0, 1]")
    if not math.isfinite(config.optical.dn_dN_cm3):
        raise ValueError("optical.dn_dN_cm3 must be finite")
    if not math.isfinite(config.carrier.zero_carrier_gain_cm) or config.carrier.zero_carrier_gain_cm >= 0.0:
        raise ValueError("carrier.zero_carrier_gain_cm must be finite and negative")
    if not 0.0 <= config.carrier.spontaneous_emission_factor <= 1.0:
        raise ValueError("carrier.spontaneous_emission_factor must be in [0, 1]")
    n = config.numerics.points
    if not isinstance(n, int) or n < 9 or n % 2 == 0:
        raise ValueError("numerics.points must be an odd integer >= 9")
    if config.device.electrode_um > config.device.domain_um:
        raise ValueError("electrode_um cannot exceed domain_um")
    if config.device.electrode_shape not in {"square", "circle"}:
        raise ValueError("device.electrode_shape must be 'square' or 'circle'")
    if not isinstance(config.numerics.carrier_substeps, int) or config.numerics.carrier_substeps < 1:
        raise ValueError("carrier_substeps must be >= 1")
    if not 0.0 < config.numerics.cfl <= 1.0:
        raise ValueError("numerics.cfl must be in (0, 1]")
    if len(config.optical.modal_detuning_cm) != 4:
        raise ValueError("modal_detuning_cm must contain A/B/C/D values")
    if len(config.optical.modal_radiation_loss_cm) != 4:
        raise ValueError("modal_radiation_loss_cm must contain A/B/C/D values")
    if not all(math.isfinite(value) for value in config.optical.modal_detuning_cm):
        raise ValueError("modal_detuning_cm must be finite")
    if not all(math.isfinite(value) and value >= 0.0 for value in config.optical.modal_radiation_loss_cm):
        raise ValueError("modal_radiation_loss_cm must be finite and nonnegative")
    if not all(math.isfinite(value) and value >= 0.0 for value in config.reproduction.current_ratios):
        raise ValueError("current_ratios must be finite and nonnegative")
    # Heun is explicit. The largest eigenvalue of the 2-D five-point diffusion
    # stencil gives D*dt_sub/dx^2 <= 1/4. Advection CFL alone is insufficient.
    # Heun 为显式法；二维扩散限制还需 D*dt_sub/dx^2 <= 1/4。
    from .constants import c
    spacing_m = config.device.domain_um * 1e-6 / (n - 1)
    optical_step_s = config.numerics.cfl * spacing_m * config.optical.group_index / c
    diffusion_cfl = (
        config.carrier.diffusion_cm2_s * 1e-4 * optical_step_s
        / config.numerics.carrier_substeps / spacing_m**2
    )
    if diffusion_cfl > 0.25:
        raise ValueError(
            "Explicit carrier diffusion is unstable: D*dt_sub/dx^2="
            f"{diffusion_cfl:.3g} > 0.25; increase carrier_substeps or reduce cfl"
        )

