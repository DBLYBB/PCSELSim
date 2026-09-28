"""Geometry primitives and analytic in-plane Fourier coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import j1


@dataclass(frozen=True)
class Ellipse:
    """An elliptical inclusion in fractional unit-cell coordinates."""

    center: tuple[float, float]
    radii: tuple[float, float]
    angle_deg: float = 0.0
    epsilon: float = 1.0


@dataclass(frozen=True)
class SquareLatticeCell:
    """Square unit cell with a uniform background and elliptical inclusions."""

    background_epsilon: float
    inclusions: tuple[Ellipse, ...]

    def fourier_epsilon(self, m: int, n: int) -> complex:
        """Return the (m,n) Fourier coefficient of epsilon(x,y).

        Overlap between inclusions is not subtracted; motifs should therefore be
        non-overlapping. Coordinates and radii are fractions of the lattice constant.
        """
        value = self.background_epsilon if (m, n) == (0, 0) else 0.0j
        k = 2.0 * np.pi * np.asarray([m, n], dtype=float)
        for inclusion in self.inclusions:
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
            phase = np.exp(1j * np.dot(k, inclusion.center))
            value += (
                inclusion.epsilon - self.background_epsilon
            ) * area * form * phase
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
            Ellipse(center=(-0.125, 0.125), radii=large_radii, angle_deg=-35.0),
            Ellipse(center=(0.125, -0.125), radii=small_radii, angle_deg=-35.0),
        ),
    )

