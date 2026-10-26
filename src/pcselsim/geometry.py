"""Unit-cell geometry and analytic in-plane Fourier coefficients.

English: Liang's 3-D CWT consumes Fourier coefficients of ``epsilon(x,y)``.
Ellipses use the exact Bessel form factor and polygons use a boundary integral,
avoiding raster-pixel noise.  Inclusions must not overlap.

中文：Liang 三维耦合波理论以面内介电常数的 Fourier 系数为输入。椭圆采用解析
Bessel 形状因子，多边形采用边界积分，因此不会引入像素化误差。当前实现不自动
扣除重叠区域，用户必须保证多个孔互不重叠。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import j1


def _validate_lossless_epsilon(value: float, name: str) -> None:
    """Reject unsupported absorption before lossless CWT projections.

    Complex dielectric constants require an absorption operator and a lossy
    vertical problem. Taking their real part would silently remove physics.
    本后端仅支持实、正、有限介电常数；吸收/金属需扩展模型，不可悄悄取实部。
    """
    epsilon = complex(value)
    if not np.isfinite(epsilon) or epsilon.real <= 0.0 or epsilon.imag != 0.0:
        raise ValueError(f"{name} must be real, positive and finite; absorbing/metallic inclusions need an extended optical model")


@dataclass(frozen=True)
class Ellipse:
    """Ellipse in fractional-cell coordinates / 以晶格常数归一化的椭圆孔。"""

    center: tuple[float, float]
    radii: tuple[float, float]
    angle_deg: float = 0.0
    epsilon: float = 1.0

    def __post_init__(self) -> None:
        _validate_lossless_epsilon(self.epsilon, "inclusion epsilon")
        if (np.shape(self.center) != (2,) or np.shape(self.radii) != (2,)
                or not np.isfinite(self.center).all() or not np.isfinite(self.radii).all()
                or min(self.radii) <= 0.0 or not np.isfinite(self.angle_deg)):
            raise ValueError("Ellipse requires finite 2-D center/angle and positive finite radii")


@dataclass(frozen=True)
class PolygonInclusion:
    """A simple polygon in fractional unit-cell coordinates.

    Vertices may be clockwise or counter-clockwise.  The Fourier transform is
    evaluated analytically from the polygon boundary, so changing a triangle
    or other polygon changes every coupling coefficient without raster noise.
    """

    vertices: tuple[tuple[float, float], ...]
    epsilon: float = 1.0

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("A polygon inclusion needs at least three vertices")
        _validate_lossless_epsilon(self.epsilon, "inclusion epsilon")
        points = np.asarray(self.vertices, dtype=float)
        if points.shape != (len(self.vertices), 2) or not np.isfinite(points).all():
            raise ValueError("Polygon vertices must be finite 2-D coordinates")
        area = np.sum(points[:, 0]*np.roll(points[:, 1], -1)-np.roll(points[:, 0], -1)*points[:, 1])
        if area == 0.0:
            raise ValueError("Polygon inclusion must have nonzero signed area")


def polygon_fourier_integral(
    vertices: tuple[tuple[float, float], ...], m: int, n: int
) -> complex:
    """Return integral_P exp(i 2 pi (m x+n y)) dxdy for a polygon P."""
    points = np.asarray(vertices, dtype=float)
    signed_area = 0.5*np.sum(
        points[:, 0]*np.roll(points[:, 1], -1)
        - np.roll(points[:, 0], -1)*points[:, 1]
    )
    if signed_area < 0.0:
        points = points[::-1]
        signed_area = -signed_area
    if (m, n) == (0, 0):
        return complex(signed_area)

    wave = 2.0*np.pi*np.asarray([m, n], dtype=float)
    wave2 = float(np.dot(wave, wave))
    # All edge terms use the same exact boundary-integral formula. Vectorize
    # only the independent edge evaluations; do not cache between geometries
    # or change the small-phase limit. The final sum can differ from the former
    # sequential accumulation at floating-point roundoff.
    edges = np.roll(points, -1, axis=0)-points
    phase_step = edges @ wave
    edge_average = np.ones(len(points), dtype=complex)
    regular = abs(phase_step) >= 1e-13
    edge_average[regular] = np.expm1(1j*phase_step[regular])/(1j*phase_step[regular])
    normal_ds = np.column_stack((edges[:, 1], -edges[:, 0]))
    value = np.sum((normal_ds @ wave)*np.exp(1j*(points @ wave))*edge_average)
    return complex(value/(1j*wave2))


@dataclass(frozen=True)
class SquareLatticeCell:
    """Square cell with uniform background and analytic inclusions / 方形晶胞。"""

    background_epsilon: float
    inclusions: tuple[Ellipse | PolygonInclusion, ...]

    def __post_init__(self) -> None:
        _validate_lossless_epsilon(self.background_epsilon, "background epsilon")

    def fourier_epsilon(self, m: int, n: int) -> complex:
        """Return the (m,n) Fourier coefficient of epsilon(x,y).

        Overlap between inclusions is not subtracted; motifs should therefore be
        non-overlapping. Coordinates and radii are fractions of the lattice constant.
        """
        value = self.background_epsilon if (m, n) == (0, 0) else 0.0j
        k = 2.0 * np.pi * np.asarray([m, n], dtype=float)
        for inclusion in self.inclusions:
            if isinstance(inclusion, Ellipse):
                theta = np.deg2rad(inclusion.angle_deg)
                rotation = np.asarray(
                    [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
                )
                local_k = rotation @ k
                rho = np.hypot(
                    inclusion.radii[0] * local_k[0], inclusion.radii[1] * local_k[1]
                )
                area = np.pi * inclusion.radii[0] * inclusion.radii[1]
                form = 1.0 if rho == 0.0 else 2.0 * j1(rho) / rho
                shape_transform = area*form*np.exp(1j*np.dot(k, inclusion.center))
            else:
                shape_transform = polygon_fourier_integral(inclusion.vertices, m, n)
            value += (
                inclusion.epsilon - self.background_epsilon
            ) * shape_transform
        return complex(value)


def inoue_double_lattice_cell(
    background_index: float = 3.554,
    large_radii: tuple[float, float] = (0.12, 0.075),
    small_radii: tuple[float, float] = (0.095, 0.060),
) -> SquareLatticeCell:
    """Idealized two-ellipse motif from Fig. 2(a), not the unpublished SEM reconstruction."""
    return SquareLatticeCell(
        background_epsilon=background_index**2,
        inclusions=(
            Ellipse(center=(-0.125, -0.125), radii=large_radii, angle_deg=-35.0),
            Ellipse(center=(0.125, 0.125), radii=small_radii, angle_deg=-35.0),
        ),
    )

