"""Typed configuration objects and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OpticalConfig:
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
    maximum_gain_cm: float = 2000.0
    zero_carrier_gain_cm: float = -5000.0
    transparency_density_cm3: float = 1.5e18
    lifetime_ns: float = 1.5
    diffusion_cm2_s: float = 100.0
    active_thickness_nm: float = 30.0
    spontaneous_emission_factor: float = 1.0e-4


@dataclass(frozen=True)
class DeviceConfig:
    domain_um: float = 400.0
    electrode_um: float = 300.0
    current_spread_um: float = 25.0
    threshold_current_A: float = 0.7
    electrode_shape: str = "square"


@dataclass(frozen=True)
class NumericsConfig:
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
    n = config.numerics.points
    if n < 9 or n % 2 == 0:
        raise ValueError("numerics.points must be an odd integer >= 9")
    if config.device.electrode_um > config.device.domain_um:
        raise ValueError("electrode_um cannot exceed domain_um")
    if config.device.electrode_shape not in {"square", "circle"}:
        raise ValueError("device.electrode_shape must be 'square' or 'circle'")
    if config.numerics.carrier_substeps < 1:
        raise ValueError("carrier_substeps must be >= 1")
    if not 0.0 < config.numerics.cfl <= 1.0:
        raise ValueError("numerics.cfl must be in (0, 1]")
    if len(config.optical.modal_detuning_cm) != 4:
        raise ValueError("modal_detuning_cm must contain A/B/C/D values")
    if len(config.optical.modal_radiation_loss_cm) != 4:
        raise ValueError("modal_radiation_loss_cm must contain A/B/C/D values")

