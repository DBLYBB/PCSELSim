"""Independent, fast multimode TE audit for crystal/hybrid planar waveguides.

Uses the same scalar Helmholtz discretization as vertical.py, but a symmetric
tridiagonal eigensolver. No semiconductor baseline is changed. Material
dispersion, TM and cross-family coupling remain outside this scalar model.
中文：保留旧半导体算法；本模块显式列出多纵向模，不能把最高 beta 模等同于全器件单模。
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh_tridiagonal
from .vertical import LayerStack, VerticalMode


def solve_guided_te_modes(stack, wavelength_nm, dz_nm=5.0, max_modes=12):
    if wavelength_nm <= 0 or dz_nm <= 0 or max_modes < 1:
        raise ValueError("Positive wavelength, step and mode count required")
    total_nm = 2 * stack.padding_um * 1000 + sum(x.thickness_nm for x in stack.layers)
    z_nm = np.linspace(0, total_nm, int(np.ceil(total_nm / dz_nm)) + 1)
    dz_m = (z_nm[1] - z_nm[0]) * 1e-9
    index = np.full(z_nm.shape, stack.bottom_index)
    cursor = stack.padding_um * 1000
    ranges = []
    for layer in stack.layers:
        stop = cursor + layer.thickness_nm
        index[(z_nm >= cursor) & (z_nm < stop)] = layer.refractive_index
        ranges.append((layer.name, cursor, stop))
        cursor = stop
    index[z_nm >= cursor] = stack.top_index
    k0 = 2 * np.pi / (wavelength_nm * 1e-9)
    diagonal = (k0 * index[1:-1]) ** 2 - 2 / dz_m**2
    off = np.full(diagonal.size - 1, 1 / dz_m**2)
    # Select all bound eigenvalues, then keep the highest beta families.
    bound = (k0 * max(stack.top_index, stack.bottom_index)) ** 2
    values, vectors = eigh_tridiagonal(
        diagonal,
        off,
        select="v",
        select_range=(bound, (k0 * max(index)) ** 2),
        check_finite=False,
    )
    integrate = getattr(np, "trapezoid", np.trapz)
    modes = []
    for j in range(values.size - 1, max(-1, values.size - max_modes - 1), -1):
        field = np.zeros(z_nm.size)
        field[1:-1] = vectors[:, j]
        field /= np.sqrt(integrate(field**2, z_nm * 1e-9))
        if field[np.argmax(abs(field))] < 0:
            field *= -1
        confinement = {}
        for name, start, stop in ranges:
            mask = (z_nm >= start) & (z_nm <= stop)
            overlap = float(integrate(field[mask] ** 2, z_nm[mask] * 1e-9))
            confinement[name] = confinement.get(name, 0) + overlap
        modes.append(
            VerticalMode(
                (z_nm - total_nm / 2) * 1e-3,
                field,
                index,
                float(np.sqrt(values[j]) / k0),
                confinement,
            )
        )
    if not modes:
        raise ValueError("No bound TE family above both cladding light lines")
    return modes


@dataclass(frozen=True)
class HybridLayerStack(LayerStack):
    """Select a stated TE family, not an unreported best-looking eigenmode."""

    te_family: int = 0

    def solve_te0(self, wavelength_nm, dz_nm=5.0):
        modes = solve_guided_te_modes(self, wavelength_nm, dz_nm, self.te_family + 1)
        if len(modes) <= self.te_family:
            raise ValueError("Requested TE family is not guided")
        return modes[self.te_family]
