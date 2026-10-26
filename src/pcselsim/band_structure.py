"""Near-Gamma band diagrams for the square four-wave and triangular six-wave models.

The band curves are eigenvalues of the same geometry-derived coupling matrix used by
the corresponding cold-cavity model.  Only the Bloch-wavevector mismatch is varied;
material dispersion and the vertical mode are frozen at the self-consistent Bragg
wavelength.  This is the coupled-wave approximation used for the small wave-number
window around the second-order Gamma point, not a full-Brillouin-zone plane-wave solve.

中文：本模块绘制二阶 Gamma 点附近的耦合波能带。每条曲线都来自同一个几何推导
耦合矩阵，只改变 Bloch 波矢失配；折射率和纵向模冻结在 Bragg 波长。因此它适合
论文图中约 ``|k| <= 0.01 (2*pi/a)`` 的局域色散，不代替全布里渊区 PWEM/FDTD。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class BandDiagram:
    """Tracked complex bands on a one-dimensional path through Gamma."""

    wave_number_2pi_over_a: np.ndarray
    normalized_frequency: np.ndarray
    radiation_constant_cm: np.ndarray
    labels: tuple[str, ...]
    left_endpoint: str
    right_endpoint: str
    approximation: str
    path_extent: str = "local_near_gamma_segment"


def track_complex_bands(
    matrix_builder: Callable[[float], np.ndarray], coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Track right eigenvectors away from Gamma by maximum modal overlap.

    Sorting every wave number independently swaps nearly degenerate branches.  We
    instead sort at Gamma and continue independently to the left and right using a
    global maximum-overlap assignment.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 3:
        raise ValueError("coordinates must be a one-dimensional array with >= 3 points")
    center = int(np.argmin(np.abs(coordinates)))
    if abs(coordinates[center]) > 1e-14:
        raise ValueError("coordinates must contain Gamma (zero)")

    center_values, center_vectors = np.linalg.eig(matrix_builder(float(coordinates[center])))
    order = np.lexsort((center_values.imag, center_values.real))
    center_values = center_values[order]
    center_vectors = center_vectors[:, order]
    center_vectors /= np.linalg.norm(center_vectors, axis=0, keepdims=True)

    modes = center_values.size
    values = np.empty((modes, coordinates.size), dtype=np.complex128)
    vectors = np.empty((modes, modes, coordinates.size), dtype=np.complex128)
    values[:, center] = center_values
    vectors[:, :, center] = center_vectors

    def continue_from(indices: range, previous: np.ndarray) -> None:
        for index in indices:
            current_values, current_vectors = np.linalg.eig(
                matrix_builder(float(coordinates[index]))
            )
            current_vectors /= np.linalg.norm(current_vectors, axis=0, keepdims=True)
            overlap = np.abs(previous.conj().T @ current_vectors) ** 2
            rows, columns = linear_sum_assignment(-overlap)
            assignment = np.empty(modes, dtype=int)
            assignment[rows] = columns
            current_values = current_values[assignment]
            current_vectors = current_vectors[:, assignment]
            # Remove arbitrary eigenvector phase before the next overlap comparison.
            phases = np.einsum("ij,ij->j", previous.conj(), current_vectors)
            phases /= np.where(np.abs(phases) > 0.0, np.abs(phases), 1.0)
            current_vectors /= phases[None, :]
            values[:, index] = current_values
            vectors[:, :, index] = current_vectors
            previous = current_vectors

    continue_from(range(center + 1, coordinates.size), center_vectors.copy())
    continue_from(range(center - 1, -1, -1), center_vectors.copy())
    return values, vectors


def detuning_to_normalized_frequency(
    detuning_per_m: np.ndarray,
    lattice_constant_nm: float,
    bragg_wavelength_nm: float,
    effective_index: float,
    group_index: float | None = None,
    *,
    reference_wavenumber_per_m: float | None = None,
) -> np.ndarray:
    """Convert local CWT detuning to ``a/lambda`` with a stated frequency origin.

    With the positive-time convention, ``exp(+i*(c/n_g)*delta*t)`` has
    positive frequency shift ``Delta f=c*delta/(2*pi*n_g)``. The same
    conversion must be used for band plots and time-domain spectra. Omitting
    ``group_index`` preserves the historical frozen-effective-index conversion
    for external callers; current geometry-driven workflows supply ``n_g``.

    Without ``reference_wavenumber_per_m`` the input is a detuning relative
    to the supplied wavelength, preserving the low-level legacy convention.
    With physical ``beta0`` the input is an absolute CWT eigenvalue. Subtract
    ``delta_ref=(beta_ref**2-beta0**2)/(2*beta0)`` and use its derivative
    ``d delta/d omega=(beta_ref/beta0)*n_g/c``. This is required when C was
    built at a fixed frequency that is not the unperturbed Bragg frequency.
    中文：固定波长不等于 Bragg 点；必须减去同一参考频率的背景失谐，并用
    平方失谐方程的正确导数。这里只是一阶局域色散，不重算完整 C(omega,k)。
    """
    lattice_m = lattice_constant_nm * 1e-9
    base = lattice_constant_nm / bragg_wavelength_nm
    frequency_index = effective_index if group_index is None else group_index
    if not np.isfinite(frequency_index) or frequency_index <= 0.0:
        raise ValueError("the frequency-conversion index must be finite and positive")
    detuning = np.asarray(detuning_per_m).real
    slope_factor = 1.0
    if reference_wavenumber_per_m is not None:
        beta0 = float(reference_wavenumber_per_m)
        if not np.isfinite(beta0) or beta0 <= 0.0:
            raise ValueError("reference_wavenumber_per_m must be finite and positive")
        beta_ref = 2.0*np.pi*effective_index/(bragg_wavelength_nm*1e-9)
        detuning = detuning-(beta_ref*beta_ref-beta0*beta0)/(2.0*beta0)
        slope_factor = beta0/beta_ref
    return base + detuning*lattice_m*slope_factor/(2.0*np.pi*frequency_index)


def square_four_wave_band_diagram(
    coupling_m: np.ndarray,
    lattice_constant_nm: float,
    bragg_wavelength_nm: float,
    effective_index: float,
    q_max: float = 0.01,
    points: int = 201,
    *,
    group_index: float | None = None,
) -> BandDiagram:
    """Return the square-lattice ``M <- Gamma -> X`` four-wave dispersion.

    ``q`` is shown in units of ``2*pi/a``.  The diagonal mismatch follows Liang
    Eq. (4.21): ``(+kx, -kx, +ky, -ky)``.  Negative plot coordinates follow
    Gamma-M with ``kx=ky=|q|/sqrt(2)``; positive coordinates follow Gamma-X
    with ``kx=q`` and ``ky=0``.
    """
    coupling_m = np.asarray(coupling_m, dtype=np.complex128)
    if coupling_m.shape != (4, 4):
        raise ValueError("coupling_m must be 4x4 for the square four-wave model")
    q = np.linspace(-float(q_max), float(q_max), int(points))
    lattice_m = lattice_constant_nm * 1e-9

    def matrix(value: float) -> np.ndarray:
        magnitude = abs(value) * 2.0 * np.pi / lattice_m
        if value < 0.0:  # Gamma-M: square-lattice diagonal
            kx = magnitude / np.sqrt(2.0)
            ky = magnitude / np.sqrt(2.0)
        else:  # Gamma-X: square-lattice axis
            kx = magnitude
            ky = 0.0
        return coupling_m + np.diag((kx, -kx, ky, -ky))

    eigenvalues, _ = track_complex_bands(matrix, q)
    frequency = detuning_to_normalized_frequency(
        eigenvalues, lattice_constant_nm, bragg_wavelength_nm, effective_index,
        group_index=group_index,
        reference_wavenumber_per_m=2.0*np.pi/lattice_m,
    )
    return BandDiagram(
        wave_number_2pi_over_a=q,
        normalized_frequency=frequency,
        radiation_constant_cm=2.0 * np.maximum(eigenvalues.imag, 0.0) / 100.0,
        labels=("A", "B", "C", "D"),
        left_endpoint="M",
        right_endpoint="X",
        approximation=(
            "square four-wave CWT; M-Gamma-X path; frozen vertical mode near Gamma; "
            "physical reference detuning and squared-detuning slope; "
            + ("group-index frequency conversion" if group_index is not None
               else "legacy effective-index frequency conversion")
        ),
    )


def plot_band_diagram(
    diagram: BandDiagram,
    path: Path,
    title: str,
    annotate_modes: bool = True,
) -> None:
    """Plot normalized frequency versus wave number in the paper's local style."""
    q = diagram.wave_number_2pi_over_a
    center = int(np.argmin(np.abs(q)))
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    if len(diagram.labels) == 4:
        annotation_offsets = ((10, -18), (-38, 8), (-38, -9), (12, 12))
    elif len(diagram.labels) == 6:
        # Put the two members of each exact C6 doublet on opposite sides.
        annotation_offsets = (
            (10, -18), (-46, -12), (14, -12), (14, 10), (14, 10), (-46, 10)
        )
    else:
        annotation_offsets = tuple(
            (8 if index % 2 == 0 else -28, 7 if index % 3 else -12)
            for index in range(len(diagram.labels))
        )
    for index, label in enumerate(diagram.labels):
        curve = diagram.normalized_frequency[index]
        ax.plot(q, curve, color="black", lw=1.45)
        ax.plot(q[center], curve[center], "o", color="black", ms=3.6)
        if annotate_modes:
            horizontal, vertical = annotation_offsets[index]
            ax.annotate(
                label,
                xy=(q[center], curve[center]),
                xytext=(horizontal, vertical),
                textcoords="offset points",
                fontsize=10,
                arrowprops={"arrowstyle": "-", "lw": 0.65, "color": "0.25"},
            )
    ax.axvline(0.0, color="0.65", lw=0.8, ls="--")
    ax.set(
        xlabel=r"Wavenumber ($2\pi/a$)",
        ylabel=r"Normalized frequency ($a/\lambda$, in units of $c/a$)",
        title=title,
        xlim=(float(q.min()), float(q.max())),
    )
    # High-symmetry-point labels live in axis coordinates so they stay readable
    # when the frequency range changes with geometry/material parameters.
    axis_x = ax.get_xaxis_transform()
    ax.text(q.min(), 0.02, f"{diagram.left_endpoint} direction", transform=axis_x, ha="left", va="bottom")
    ax.text(0.0, 0.02, r"$\Gamma$", transform=axis_x, ha="center", va="bottom")
    ax.text(q.max(), 0.02, f"{diagram.right_endpoint} direction", transform=axis_x, ha="right", va="bottom")
    ax.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(path, dpi=210)
    plt.close(fig)

