"""Stable planar multilayer scattering and source Green functions.

Layers are ordered **bottom to top**, with z increasing upwards.  The time
convention is exp(+i omega t): an upward outgoing wave is exp(-i kz z), and a
passive refractive index is n - i*kappa.  No growing transfer matrices are
formed; interfaces and decaying propagation are cascaded as scattering ports.

TE uses tangential electric field F and p=1.  TM uses tangential magnetic
field F and p=1/epsilon.  In either case the weighted scalar Helmholtz Green
function solves [d_z p d_z + p*kz**2] G = -delta.  Its uniform TE limit is
exactly Liang's -i exp(-i kz |z-z'|)/(2 kz), including evanescent kz=-i*kappa.

中文：这是独立的平面层状光学模块，不改变旧的自由空间 Green 近似。有限 DBR
具有有限透射，不能称为严格单面输出；反射相位也必须进入 Green 函数和耦合矩阵。
它不包含图形化光子晶体的完整矢量 Maxwell 解，也不求有源激光速率方程。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Polarization = Literal["TE", "TM"]


def _index(value: complex, name: str) -> complex:
    value = complex(value)
    if not np.isfinite(value) or value.real <= 0.0 or value.imag > 1e-12:
        raise ValueError(f"{name} must have Re(n)>0 and Im(n)<=0 (passive exp(+i wt) convention)")
    return value


@dataclass(frozen=True)
class StratifiedLayer:
    """One homogeneous isotropic nonmagnetic finite layer / 一层均匀薄膜。"""

    name: str
    thickness_nm: float
    refractive_index: complex

    def __post_init__(self) -> None:
        if not np.isfinite(self.thickness_nm) or self.thickness_nm <= 0.0:
            raise ValueError("Layer thickness_nm must be finite and positive")
        _index(self.refractive_index, self.name)


@dataclass(frozen=True)
class StratifiedStack:
    """Finite layers, bottom to top, between two semi-infinite exteriors."""

    layers: tuple[StratifiedLayer, ...]
    bottom_index: complex
    top_index: complex

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        _index(self.bottom_index, "bottom_index")
        _index(self.top_index, "top_index")
        if any(not isinstance(layer, StratifiedLayer) for layer in self.layers):
            raise TypeError("layers must contain StratifiedLayer objects")

    @property
    def boundaries_m(self) -> np.ndarray:
        """Layer boundary coordinates, bottom finite-stack surface at z=0."""
        return np.r_[0.0, np.cumsum([layer.thickness_nm * 1e-9 for layer in self.layers])]

    @classmethod
    def from_layer_stack(cls, stack: object) -> "StratifiedStack":
        """Adapt the existing vertical.LayerStack; numerical padding is excluded."""
        return cls(
            tuple(StratifiedLayer(layer.name, layer.thickness_nm, layer.refractive_index)
                  for layer in stack.layers),
            bottom_index=stack.bottom_index,
            top_index=stack.top_index,
        )


@dataclass(frozen=True)
class StackScattering:
    """Scalar field amplitudes; R/T are power ratios only for propagating ports.

    For TM the amplitudes are H, not tangential E.  ``A`` is undefined for an
    absorbing incident exterior because incident/reflected interference then
    prevents the usual 1-R-T identification with finite-layer absorption.
    """

    r: complex
    t: complex
    R: float | None
    T: float | None
    A: float | None
    reflection_phase_rad: float
    transmission_phase_rad: float
    scattering_matrix: np.ndarray
    polarization: Polarization
    incident_side: str
    q_parallel_per_m: float
    kz_by_medium_m: np.ndarray
    admittance_by_medium_m: np.ndarray


def _parameters(stack: StratifiedStack, wavelength_nm: float, q: float,
                polarization: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError("wavelength_nm must be finite and positive")
    if not np.isfinite(q) or q < 0.0:
        raise ValueError("q_parallel_per_m must be finite and nonnegative")
    if polarization not in ("TE", "TM"):
        raise ValueError("polarization must be 'TE' or 'TM'")
    indices = np.asarray([stack.bottom_index, *(layer.refractive_index for layer in stack.layers),
                          stack.top_index], dtype=complex)
    epsilon = indices**2
    k0 = 2.0 * np.pi / (wavelength_nm * 1e-9)
    kz = np.sqrt(epsilon * k0**2 - q**2 + 0j)
    # Outgoing/decaying branch for exp(+i wt), including real-index evanescent ports.
    kz = np.where(kz.imag > 0.0, -kz, kz)
    kz = np.where((np.abs(kz.imag) < 1e-14 * k0) & (kz.real < 0.0), -kz, kz)
    if np.any(np.abs(kz) < 1e-13 * k0):
        raise ValueError("Exactly grazing/cutoff kz=0 has a singular plane-wave basis; use a limiting q")
    p = np.ones_like(epsilon) if polarization == "TE" else 1.0 / epsilon
    return kz, p * kz, p


def _interface(left_y: complex, right_y: complex) -> np.ndarray:
    denominator = left_y + right_y
    if abs(denominator) <= np.finfo(float).eps * max(abs(left_y), abs(right_y)):
        raise ValueError("Singular interface admittance")
    reflection = (left_y - right_y) / denominator
    return np.array([[reflection, 2.0 * right_y / denominator],
                     [2.0 * left_y / denominator, -reflection]], dtype=complex)


def _cascade(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Scalar Redheffer star product; never exponentiates a growing solution."""
    denominator = 1.0 - left[1, 1] * right[0, 0]
    if abs(denominator) < 8.0 * np.finfo(float).eps:
        raise ValueError("Scattering pole unresolved at this real frequency/q; add physical loss or detune")
    return np.array([
        [left[0, 0] + left[0, 1] * right[0, 0] * left[1, 0] / denominator,
         left[0, 1] * right[0, 1] / denominator],
        [right[1, 0] * left[1, 0] / denominator,
         right[1, 1] + right[1, 0] * left[1, 1] * right[0, 1] / denominator],
    ], dtype=complex)


def _network(stack: StratifiedStack, kz: np.ndarray, admittance: np.ndarray,
             first_medium: int, last_medium: int) -> np.ndarray:
    """Connect two medium indices; propagate only intermediate finite layers."""
    result = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    for medium in range(first_medium, last_medium):
        result = _cascade(result, _interface(admittance[medium], admittance[medium + 1]))
        next_medium = medium + 1
        if next_medium < last_medium:
            phase = np.exp(-1j * kz[next_medium] * stack.layers[next_medium - 1].thickness_nm * 1e-9)
            result = _cascade(result, np.array([[0.0, phase], [phase, 0.0]], dtype=complex))
    return result


def solve_stack(stack: StratifiedStack, wavelength_nm: float, *, polarization: Polarization = "TE",
                incident_side: Literal["bottom", "top"] = "bottom", angle_deg: float = 0.0,
                q_parallel_per_m: float | None = None) -> StackScattering:
    """Planar scattering at normal/oblique incidence, or explicit transverse q.

    ``q`` is conserved across interfaces.  Angle is in the incident exterior.
    Explicit evanescent q is supported, but R/T/A=None if incident flux is zero.
    """
    if incident_side not in ("bottom", "top"):
        raise ValueError("incident_side must be 'bottom' or 'top'")
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError("wavelength_nm must be finite and positive")
    if not np.isfinite(angle_deg) or not 0.0 <= angle_deg < 90.0:
        raise ValueError("angle_deg must be in [0, 90)")
    incident_index = complex(stack.bottom_index if incident_side == "bottom" else stack.top_index)
    if q_parallel_per_m is None:
        if angle_deg and abs(incident_index.imag) > 1e-14:
            raise ValueError("Specify real q_parallel_per_m instead of angle in an absorbing incident medium")
        q = 2.0 * np.pi / (wavelength_nm * 1e-9) * incident_index.real * np.sin(np.deg2rad(angle_deg))
    else:
        if angle_deg != 0.0:
            raise ValueError("Specify either angle_deg or q_parallel_per_m, not both")
        q = float(q_parallel_per_m)
    kz, admittance, _ = _parameters(stack, wavelength_nm, q, polarization)
    scattering = _network(stack, kz, admittance, 0, len(stack.layers) + 1)
    input_port, output_port = (0, -1) if incident_side == "bottom" else (-1, 0)
    r, t = ((scattering[0, 0], scattering[1, 0]) if incident_side == "bottom"
            else (scattering[1, 1], scattering[0, 1]))
    incident_flux = float(admittance[input_port].real)
    propagating = incident_flux > 1e-12 * max(abs(admittance[input_port]), 1.0)
    reflectance = float(abs(r)**2) if propagating else None
    transmittance = (float(max(admittance[output_port].real, 0.0) / incident_flux * abs(t)**2)
                     if propagating else None)
    absorption = (float(1.0 - reflectance - transmittance)
                  if propagating and abs(incident_index.imag) < 1e-14 else None)
    return StackScattering(complex(r), complex(t), reflectance, transmittance, absorption,
                           float(np.angle(r)), float(np.angle(t)), scattering, polarization,
                           incident_side, q, kz, admittance)


def quarter_wave_dbr(center_wavelength_nm: float, high_index: float, low_index: float,
                     pairs: int, *, first_layer: Literal["high", "low"] = "high",
                     name_prefix: str = "DBR") -> tuple[StratifiedLayer, ...]:
    """Return finite normal-incidence quarter-wave pairs in bottom-to-top order.

    The **last** returned layer is adjacent to a source layer appended above the
    DBR.  Change first_layer/order deliberately: reflection phase depends on it.
    At oblique incidence these thicknesses are not quarter-wave optical phases.
    """
    if not np.isfinite(center_wavelength_nm) or center_wavelength_nm <= 0.0:
        raise ValueError("center_wavelength_nm must be positive")
    if not np.isfinite(high_index) or not np.isfinite(low_index) or not high_index > low_index > 0.0:
        raise ValueError("Require real high_index > low_index > 0")
    if not isinstance(pairs, (int, np.integer)) or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    if first_layer not in ("high", "low"):
        raise ValueError("first_layer must be 'high' or 'low'")
    order = (("H", high_index), ("L", low_index))
    if first_layer == "low":
        order = order[::-1]
    return tuple(StratifiedLayer(f"{name_prefix}_{pair + 1:02d}_{label}",
                                center_wavelength_nm / (4.0 * index), index)
                 for pair in range(pairs) for label, index in order)


@dataclass(frozen=True)
class OutgoingAmplitudes:
    """Field per unit Helmholtz delta source, evaluated at exterior boundaries.

    Flux weights are Re(p*kz); these are not Watts without a source and Maxwell
    normalization.  Evanescent exterior fields have zero far-zone flux.
    """

    top: np.ndarray
    bottom: np.ndarray
    top_admittance_m: complex
    bottom_admittance_m: complex

    @property
    def top_flux(self) -> np.ndarray:
        return max(float(self.top_admittance_m.real), 0.0) * np.abs(self.top)**2

    @property
    def bottom_flux(self) -> np.ndarray:
        return max(float(self.bottom_admittance_m.real), 0.0) * np.abs(self.bottom)**2


@dataclass(frozen=True)
class LocalGreenKernel:
    """Reciprocal Green function for sources/observations in one finite host.

    The four nonnegative optical path lengths are the direct path, one lower
    reflection, one upper reflection, and a round trip.  Summing their geometric
    series is equivalent to the outgoing Jost/Wronskian construction.  It avoids
    evanescent overflow.  Coordinates are global stack z in metres, not centred
    vertical-solver coordinates and not its numerical padding.
    """

    z_start_m: float
    thickness_m: float
    kz_m: complex
    p_host: complex
    lower_reflection: complex
    upper_reflection: complex
    transmission_to_bottom: complex
    transmission_to_top: complex
    bottom_admittance_m: complex
    top_admittance_m: complex
    lossless: bool

    def _coordinates(self, z_m: np.ndarray | float) -> np.ndarray:
        x = np.asarray(z_m, dtype=float) - self.z_start_m
        tolerance = 64.0 * np.finfo(float).eps * max(self.thickness_m, 1e-12)
        if not np.all(np.isfinite(x)) or np.any(x < -tolerance) or np.any(x > self.thickness_m + tolerance):
            raise ValueError("Source/observation coordinates must be inside the selected host layer")
        return np.clip(x, 0.0, self.thickness_m)

    @property
    def denominator(self) -> complex:
        value = 1.0 - self.lower_reflection * self.upper_reflection * np.exp(-2j * self.kz_m * self.thickness_m)
        if abs(value) < 8.0 * np.finfo(float).eps:
            raise ValueError("Green function at an unresolved guided/cavity pole; detune or add physical loss")
        return complex(value)

    def green(self, z_m: np.ndarray | float, source_z_m: np.ndarray | float) -> np.ndarray:
        """Evaluate broadcasting observation/source coordinates (units: metres)."""
        x, source = np.broadcast_arrays(self._coordinates(z_m), self._coordinates(source_z_m))
        distance = np.abs(x - source)
        d, k, lower, upper = self.thickness_m, self.kz_m, self.lower_reflection, self.upper_reflection
        paths = (np.exp(-1j * k * distance)
                 + lower * np.exp(-1j * k * (x + source))
                 + upper * np.exp(-1j * k * (2.0 * d - x - source))
                 + lower * upper * np.exp(-1j * k * (2.0 * d - distance)))
        return (-1j / (2.0 * self.p_host * k)) * paths / self.denominator

    def matrix(self, z_m: np.ndarray, source_z_m: np.ndarray | None = None) -> np.ndarray:
        """Dense local quadrature kernel G[observation, source]."""
        z = np.asarray(z_m, dtype=float)
        sources = z if source_z_m is None else np.asarray(source_z_m, dtype=float)
        if z.ndim != 1 or sources.ndim != 1:
            raise ValueError("matrix coordinates must be one-dimensional")
        return self.green(z[:, None], sources[None, :])

    def outgoing_amplitudes(self, source_z_m: np.ndarray | float) -> OutgoingAmplitudes:
        """Delta-source amplitude escaping to each exterior, including phase."""
        x = self._coordinates(source_z_m)
        d, k = self.thickness_m, self.kz_m
        prefactor = (-1j / (2.0 * self.p_host * k)) / self.denominator
        top = (self.transmission_to_top * np.exp(-1j * k * (d - x)) * prefactor
               * (1.0 + self.lower_reflection * np.exp(-2j * k * x)))
        bottom = (self.transmission_to_bottom * np.exp(-1j * k * x) * prefactor
                  * (1.0 + self.upper_reflection * np.exp(-2j * k * (d - x))))
        return OutgoingAmplitudes(top, bottom, self.top_admittance_m, self.bottom_admittance_m)

    def quadrature_flux(self, z_m: np.ndarray, source: np.ndarray, weights_m: np.ndarray) -> dict[str, float]:
        """Audit the distributed-source optical theorem with the same quadrature.

        For a real lossless stack: -Im(f^H W G W f) equals top_flux+bottom_flux.
        With absorbing layers the difference is nonnegative absorption, not an
        error; this helper labels whether equality is expected.  This is the
        consistency check needed before constructing a reflector-aware Crad.
        """
        z, f, weights = np.asarray(z_m), np.asarray(source, dtype=complex), np.asarray(weights_m, dtype=float)
        if z.ndim != 1 or f.shape != z.shape or weights.shape != z.shape:
            raise ValueError("z_m, source and weights_m must be matching one-dimensional arrays")
        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(f)):
            raise ValueError("Require finite source and nonnegative finite integration weights")
        weighted = f * weights
        amplitudes = self.outgoing_amplitudes(z)
        top = complex(amplitudes.top @ weighted)
        bottom = complex(amplitudes.bottom @ weighted)
        green_flux = -float(np.imag(weighted.conj() @ self.matrix(z) @ weighted))
        top_flux = max(float(self.top_admittance_m.real), 0.0) * abs(top)**2
        bottom_flux = max(float(self.bottom_admittance_m.real), 0.0) * abs(bottom)**2
        difference = green_flux - top_flux - bottom_flux
        scale = max(abs(green_flux), top_flux + bottom_flux, np.finfo(float).tiny)
        return {"green_flux": green_flux, "top_flux": float(top_flux), "bottom_flux": float(bottom_flux),
                "difference_or_absorption": float(difference), "relative_residual": float(difference / scale),
                "lossless_equality_expected": self.lossless}


def local_green_kernel(stack: StratifiedStack, host_layer_index: int, wavelength_nm: float, *,
                       q_parallel_per_m: float = 0.0, polarization: Polarization = "TE") -> LocalGreenKernel:
    """Build a reflection-aware outgoing Green function in a selected host.

    The host is indexed in ``stack.layers`` (zero-based).  High-order evanescent
    q is supported.  A homogeneous host without index contrast recovers free G.
    An actual guided-mode pole needs a causal loss/detuning, not silent clipping.
    """
    if not isinstance(host_layer_index, (int, np.integer)) or not 0 <= host_layer_index < len(stack.layers):
        raise ValueError("host_layer_index must select a finite stack layer")
    kz, admittance, p = _parameters(stack, wavelength_nm, float(q_parallel_per_m), polarization)
    medium = int(host_layer_index) + 1
    lower = _network(stack, kz, admittance, 0, medium)
    upper = _network(stack, kz, admittance, medium, len(stack.layers) + 1)
    boundaries = stack.boundaries_m
    indices = [stack.bottom_index, *(layer.refractive_index for layer in stack.layers), stack.top_index]
    return LocalGreenKernel(
        float(boundaries[host_layer_index]), stack.layers[host_layer_index].thickness_nm * 1e-9,
        complex(kz[medium]), complex(p[medium]), complex(lower[1, 1]), complex(upper[0, 0]),
        complex(lower[0, 1]), complex(upper[1, 0]), complex(admittance[0]), complex(admittance[-1]),
        all(abs(complex(index).imag) < 1e-14 for index in indices),
    )
