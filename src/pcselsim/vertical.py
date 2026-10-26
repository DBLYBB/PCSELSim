"""Scalar one-dimensional TE solver for multilayer waveguides.

English: the highest-beta eigenpair of the scalar Helmholtz operator supplies
the vertical field, ``n_eff`` and layer overlaps required by Liang 3-D CWT.
Indices are treated as wavelength-independent unless the caller performs an
external wavelength sweep; vectorial/interface and material-dispersion effects
are outside this baseline model.

中文：本模块求标量 Helmholtz 算子的最高传播常数本征模，输出纵向场、``n_eff``
及逐层限制因子，供 Liang 三维 CWT 使用。折射率默认无色散，且没有完整矢量界面
修正；更换材料时应自行提供波长相关折射率并做 ``dz``、padding 收敛检查。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import pi
from scipy.linalg import eigh_tridiagonal


def sample_vertical_interval(
    axis_m: np.ndarray,
    values: np.ndarray,
    start_m: float,
    stop_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a vertical interval including its exact material interfaces.

    Linear interpolation inserts both endpoints when they fall between mesh
    nodes. Integrating only nodes selected by a layer mask omits parts of the
    layer and can severely underestimate the overlap of thin quantum wells.
    Real densities and complex fields are both supported.
    """
    axis_m = np.asarray(axis_m, dtype=float)
    values = np.asarray(values)
    if axis_m.ndim != 1 or values.shape != axis_m.shape or axis_m.size < 2:
        raise ValueError("axis and values must be matching one-dimensional arrays")
    if np.any(np.diff(axis_m) <= 0.0):
        raise ValueError("vertical coordinates must be strictly increasing")
    if not axis_m[0] <= start_m < stop_m <= axis_m[-1]:
        raise ValueError("layer interval must lie inside the vertical mesh")
    interior = (axis_m > start_m) & (axis_m < stop_m)
    sampled_axis = np.concatenate(([start_m], axis_m[interior], [stop_m]))
    return sampled_axis, np.interp(sampled_axis, axis_m, values)


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
    """Finite stack padded by numerical top/bottom claddings / 数值截断的层结构。"""

    layers: tuple[Layer, ...]
    top_index: float
    bottom_index: float
    padding_um: float = 2.0

    def solve_te0(self, wavelength_nm: float, dz_nm: float = 2.0) -> VerticalMode:
        """Solve the scalar TE Helmholtz problem for the highest-index guided mode.

        Dirichlet boundaries are placed ``padding_um`` into the top and bottom
        claddings. Increase padding until ``effective_index`` has converged.
        边界采用 Dirichlet 截断；必须增大 padding，直到 ``n_eff`` 收敛。
        """
        if dz_nm <= 0.0:
            raise ValueError("dz_nm must be positive")
        if wavelength_nm <= 0.0 or self.padding_um <= 0.0:
            raise ValueError("wavelength_nm and padding_um must be positive")
        if not self.layers or any(
            layer.thickness_nm <= 0.0 or layer.refractive_index <= 0.0
            for layer in self.layers
        ) or min(self.top_index, self.bottom_index) <= 0.0:
            raise ValueError("A layer stack requires positive thicknesses and indices")
        total_nm = 2.0 * self.padding_um * 1e3 + sum(x.thickness_nm for x in self.layers)
        points = int(np.ceil(total_nm / dz_nm)) + 1
        if points < 3:
            raise ValueError("the vertical mesh needs at least one interior point")
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
        k0 = 2.0 * pi / (wavelength_nm * 1e-9)
        diagonal = -2.0 / actual_dz_m**2 + (k0 * interior_index)**2
        off_diagonal = np.full(count - 1, 1.0 / actual_dz_m**2)
        # The FD operator is real symmetric tridiagonal. A direct selected
        # eigenpair avoids an iterative large-box solve at every Bragg step.
        beta2, vectors = eigh_tridiagonal(
            diagonal, off_diagonal, select="i", select_range=(count - 1, count - 1)
        )
        cladding_beta2 = (k0 * max(self.top_index, self.bottom_index)) ** 2
        if beta2[0] <= cladding_beta2:
            raise ValueError(
                "The stack has no bound TE0 mode above both cladding light lines; "
                "a finite-box cladding state cannot be used as a PCSEL guided mode"
            )
        beta = float(np.sqrt(beta2[0]))
        field = np.zeros(points, dtype=float)
        field[1:-1] = vectors[:, 0]
        integrate = getattr(np, "trapezoid", np.trapz)
        norm = np.sqrt(integrate(np.abs(field) ** 2, z_nm * 1e-9))
        field /= norm
        density = np.abs(field) ** 2
        total = float(integrate(density, z_nm * 1e-9))
        confinement: dict[str, float] = {}
        for name, start, stop in interfaces:
            layer_z, layer_density = sample_vertical_interval(
                z_nm * 1e-9, density, start * 1e-9, stop * 1e-9
            )
            value = float(integrate(layer_density, layer_z)) / total
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
            Layer("active-AlGaAs-3", 20.0, 3.445),
            Layer("AlGaAs", 80.0, 3.445),
            Layer("n-clad", 1000.0, 3.122),
        ),
        top_index=3.122,
        bottom_index=3.297,
        padding_um=1.0,
    )

