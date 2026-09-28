"""One-dimensional finite-difference TE mode solver for multilayer waveguides."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import pi
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class Layer:
    name: str
    thickness_nm: float
    refractive_index: float


@dataclass
class VerticalMode:
    z_um: np.ndarray
    field: np.ndarray
    index: np.ndarray
    effective_index: float
    confinement: dict[str, float]


@dataclass(frozen=True)
class LayerStack:
    """Finite layer stack padded by semi-infinite top/bottom media."""

    layers: tuple[Layer, ...]
    top_index: float
    bottom_index: float
    padding_um: float = 2.0

    def solve_te0(self, wavelength_nm: float, dz_nm: float = 2.0) -> VerticalMode:
        """Solve the scalar TE Helmholtz problem for the highest-index guided mode.

        Dirichlet boundaries are placed ``padding_um`` into the top and bottom
        claddings. Increase padding until ``effective_index`` has converged.
        """
        if dz_nm <= 0.0:
            raise ValueError("dz_nm must be positive")
        total_nm = 2.0 * self.padding_um * 1e3 + sum(x.thickness_nm for x in self.layers)
        points = int(np.ceil(total_nm / dz_nm)) + 1
        z_nm = np.linspace(0.0, total_nm, points)
        actual_dz_m = float(z_nm[1] - z_nm[0]) * 1e-9
        index = np.full(points, self.bottom_index, dtype=float)
        interfaces: list[tuple[str, float, float]] = []
        cursor = self.padding_um * 1e3
        index[z_nm < cursor] = self.bottom_index
        for layer in self.layers:
            start, stop = cursor, cursor + layer.thickness_nm
            mask = (z_nm >= start) & (z_nm < stop)
            index[mask] = layer.refractive_index
            interfaces.append((layer.name, start, stop))
            cursor = stop
        index[z_nm >= cursor] = self.top_index

        interior_index = index[1:-1]
        count = interior_index.size
        second = diags(
            [np.ones(count - 1), -2.0 * np.ones(count), np.ones(count - 1)],
            offsets=[-1, 0, 1],
            format="csr",
        ) / actual_dz_m**2
        k0 = 2.0 * pi / (wavelength_nm * 1e-9)
        operator = second + diags((k0 * interior_index) ** 2, format="csr")
        beta2, vectors = eigsh(operator, k=1, which="LA")
        beta = float(np.sqrt(max(beta2[0], 0.0)))
        field = np.zeros(points, dtype=float)
        field[1:-1] = vectors[:, 0]
        norm = np.sqrt(np.trapezoid(np.abs(field) ** 2, z_nm * 1e-9))
        field /= norm
        density = np.abs(field) ** 2
        total = float(np.trapezoid(density, z_nm * 1e-9))
        confinement: dict[str, float] = {}
        for name, start, stop in interfaces:
            mask = (z_nm >= start) & (z_nm <= stop)
            value = float(np.trapezoid(density[mask], z_nm[mask] * 1e-9)) / total
            confinement[name] = confinement.get(name, 0.0) + value
        return VerticalMode(
            z_um=(z_nm - 0.5 * total_nm) * 1e-3,
            field=field,
            index=index,
            effective_index=beta / k0,
            confinement=confinement,
        )


def inoue2019_stack(pc_average_index: float = 3.30) -> LayerStack:
    """Table I layer stack; thick claddings are represented explicitly."""
    return LayerStack(
        layers=(
            Layer("p-clad", 2000.0, 3.297),
            Layer("PC", 190.0, pc_average_index),
            Layer("GaAs", 110.0, 3.554),
            Layer("AlGaAs-spacer", 25.0, 3.269),
            Layer("active-InGaAs", 10.0, 3.584),
            Layer("active-AlGaAs-1", 20.0, 3.445),
            Layer("active-InGaAs", 10.0, 3.584),
            Layer("active-AlGaAs-2", 20.0, 3.445),
            Layer("active-InGaAs", 10.0, 3.584),
            Layer("AlGaAs", 80.0, 3.445),
            Layer("n-clad", 1000.0, 3.122),
        ),
        top_index=3.122,
        bottom_index=3.297,
        padding_um=1.0,
    )

