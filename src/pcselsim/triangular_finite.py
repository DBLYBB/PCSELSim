"""Finite-device companion solver for the triangular six-wave 3-D CWT model.

The six envelopes propagate along the three characteristic directions of
Liang Eq. (5.17).  They are sampled on an axial characteristic grid with zero
incoming ghost values at a circular, regular-hexagonal, or square aperture.  The resulting first-order
characteristic-grid operator is the finite-volume counterpart of the staggered
relations in Eqs. (5.18)-(5.19).  As with the square four-wave implementation,
reported losses should be checked by increasing ``radius_cells``.

中文：六个包络分别沿三组三角晶格特征线传播。圆形/正六边形/方形器件外的入射 ghost 值设为零，
从而实现吸收/开放边界。远场直接对每个六角网格点的复辐射场作非均匀 Fourier 求和，
没有把斜坐标数组错误地当成笛卡尔像素做 FFT，也没有人为乘涡旋或高斯因子。
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle, Polygon, Rectangle
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from .triangular_six_wave import (
    TRIANGULAR_MODE_NAMES,
    TriangularCouplingResult,
)
from .numerical_quality import extrapolate_grid_loss


@dataclass(frozen=True)
class TriangularFiniteSpec:
    """Finite-device discretization parameters.

    ``radius_um`` is the circle/hexagon circumradius or square half-width.
    The characteristic grid remains aligned with the six Bloch-wave directions;
    only the physical aperture mask changes.
    """

    radius_um: float = 30.0
    radius_cells: int = 9
    internal_loss_cm: float = 0.0
    eigensolutions_per_target: int = 8
    aperture_shape: str = "circle"


@dataclass(frozen=True)
class HexagonalGrid:
    """Axial grid embedded in Cartesian coordinates."""

    radius_cells: int
    radius_um: float
    axial_j: np.ndarray
    axial_k: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    spacing_m: float
    index: dict[tuple[int, int], int]
    aperture_shape: str = "circle"

    @property
    def points(self) -> int:
        return int(self.axial_j.size)


@dataclass
class TriangularFiniteMode:
    """One finite six-wave eigenmode on the circular hexagonal grid."""

    name: str
    delta_per_m: float
    alpha_per_m: float
    fields: np.ndarray
    band_overlap: float
    grid: HexagonalGrid
    radiation_x: np.ndarray
    radiation_y: np.ndarray
    grid_alpha_per_m: float | None = None
    extrapolation_uncertainty_per_m: float | None = None
    extrapolation_status: str = "single_grid"

    @property
    def intensity(self) -> np.ndarray:
        value = np.sum(np.abs(self.fields) ** 2, axis=0)
        return value / max(float(value.max()), np.finfo(float).eps)

    @property
    def alpha_l(self) -> float:
        return float(self.alpha_per_m * self.grid.radius_um * 1e-6)


@dataclass(frozen=True)
class TriangularFarField:
    angle_deg: np.ndarray
    field_x: np.ndarray
    field_y: np.ndarray
    power: np.ndarray
    power_x: np.ndarray
    power_y: np.ndarray
    full_rms_divergence_deg: float
    center_to_peak: float
    peak_offset_deg: float
    centroid_offset_deg: float
    ellipticity: float
    encircled_power_0p5deg: float
    encircled_power_1deg: float
    requested_view_deg: float = 3.0
    evaluated_view_deg: float = 3.0
    alias_free_square_view_deg: float = 3.0
    energy_normalization: str = "evaluated_angular_window"
    moment_reference: str = "surface_normal_radial_second_moment"


def build_hexagonal_grid(radius_cells: int, radius_um: float) -> HexagonalGrid:
    """Backward-compatible circular aperture on the axial characteristic grid."""
    return build_device_grid(radius_cells, radius_um, "circle")


def build_device_grid(
    radius_cells: int,
    radius_um: float,
    aperture_shape: str = "circle",
) -> HexagonalGrid:
    """Build a circular, regular-hexagonal, or square physical aperture.

    The square uses ``radius_um`` as half-width.  The regular hexagon uses it
    as circumradius.  All three masks therefore fit inside the same 300-um
    bounding box when ``radius_um=150``.
    """
    radius_cells = int(radius_cells)
    if radius_cells < 3:
        raise ValueError("radius_cells must be >= 3")
    if radius_um <= 0.0:
        raise ValueError("radius_um must be positive")
    shape = str(aperture_shape).lower()
    if shape not in {"circle", "hexagon", "square"}:
        raise ValueError("aperture_shape must be 'circle', 'hexagon', or 'square'")
    extent = 2 * radius_cells if shape == "square" else radius_cells
    nodes = []
    for j in range(-extent, extent + 1):
        for k in range(-extent, extent + 1):
            x_index = 0.5 * (j + k)
            y_index = 0.5 * np.sqrt(3.0) * (k - j)
            if shape == "circle":
                inside = j * j - j * k + k * k <= radius_cells * radius_cells
            elif shape == "hexagon":
                inside = max(abs(j), abs(k), abs(j-k)) <= radius_cells
            else:
                inside = (
                    abs(x_index) <= radius_cells + 1e-12
                    and abs(y_index) <= radius_cells + 1e-12
                )
            if inside:
                nodes.append((j, k))
    nodes.sort()
    axial_j = np.asarray([node[0] for node in nodes], dtype=int)
    axial_k = np.asarray([node[1] for node in nodes], dtype=int)
    spacing_um = radius_um / radius_cells
    x_um = 0.5 * spacing_um * (axial_j + axial_k)
    y_um = 0.5 * np.sqrt(3.0) * spacing_um * (axial_k - axial_j)
    return HexagonalGrid(
        radius_cells=radius_cells,
        radius_um=float(radius_um),
        axial_j=axial_j,
        axial_k=axial_k,
        x_um=x_um,
        y_um=y_um,
        spacing_m=spacing_um * 1e-6,
        index={node: position for position, node in enumerate(nodes)},
        aperture_shape=shape,
    )


def _directed_upwind(
    grid: HexagonalGrid,
    step: tuple[int, int],
    positive_direction: bool,
) -> sparse.csr_matrix:
    """Directional derivative with a zero incoming value outside the disk."""
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    dj, dk = step
    for row, (j, k) in enumerate(zip(grid.axial_j, grid.axial_k, strict=True)):
        rows.append(row)
        columns.append(row)
        data.append(1.0 / grid.spacing_m)
        neighbour = (j - dj, k - dk) if positive_direction else (j + dj, k + dk)
        column = grid.index.get(neighbour)
        if column is not None:
            rows.append(row)
            columns.append(column)
            data.append(-1.0 / grid.spacing_m)
    return sparse.csr_matrix(
        (data, (rows, columns)), shape=(grid.points, grid.points)
    )


def triangular_finite_operator(
    result: TriangularCouplingResult,
    spec: TriangularFiniteSpec,
) -> tuple[sparse.csr_matrix, HexagonalGrid]:
    """Discretize ``(delta+i*alpha)V = C V + i D V`` on a hex grid."""
    grid = build_device_grid(
        spec.radius_cells, spec.radius_um, spec.aperture_shape
    )
    identity = sparse.identity(grid.points, format="csr")
    local_matrix = result.coupling_m + 1j * 100.0 * spec.internal_loss_cm * np.eye(6)
    local = sparse.kron(sparse.csr_matrix(local_matrix), identity, format="csr")
    directions = ((1, 0), (1, 0), (0, 1), (0, 1), (1, 1), (1, 1))
    signs = (True, False, True, False, True, False)
    transport = sparse.block_diag(
        tuple(
            _directed_upwind(grid, step, positive)
            for step, positive in zip(directions, signs, strict=True)
        ),
        format="csr",
    )
    return local + 1j * transport, grid


def _hex_roughness(fields: np.ndarray, grid: HexagonalGrid) -> float:
    numerator = 0.0
    samples = 0
    for step in ((1, 0), (0, 1), (1, 1)):
        dj, dk = step
        for left, (j, k) in enumerate(zip(grid.axial_j, grid.axial_k, strict=True)):
            right = grid.index.get((j + dj, k + dk))
            if right is None:
                continue
            numerator += float(np.sum(np.abs(fields[:, left] - fields[:, right]) ** 2))
            samples += 1
    denominator = max(float(np.mean(np.abs(fields) ** 2)), np.finfo(float).eps)
    return numerator / max(samples * denominator, 1.0)


def solve_triangular_finite_modes(
    result: TriangularCouplingResult,
    spec: TriangularFiniteSpec,
) -> dict[str, TriangularFiniteMode]:
    """Find the smooth fundamental finite state connected to every Gamma band edge."""
    operator, grid = triangular_finite_operator(result, spec)
    targets = np.asarray(result.eigenvalues_m)
    basis = np.asarray(result.eigenvectors)
    sample = np.arange(operator.shape[0], dtype=float)
    phase = 2.0 * np.pi * sample / operator.shape[0]
    v0 = 1.0 + 0.19 * np.cos(phase) + 1j * (0.17 * np.sin(3.0 * phase))

    # Solve once for every distinct band edge.  Degenerate B and D pairs share
    # one candidate pool and are assigned globally by projected overlap.
    groups: list[list[int]] = []
    for mode_index, target in enumerate(targets):
        for group in groups:
            reference = targets[group[0]]
            if abs(target - reference) <= 1e-7 * max(abs(reference), 1.0):
                group.append(mode_index)
                break
        else:
            groups.append([mode_index])

    selected_modes: dict[str, TriangularFiniteMode] = {}
    for group in groups:
        target = complex(np.mean(targets[group]))
        requested = max(spec.eigensolutions_per_target, 3 * len(group) + 2)
        requested = min(requested, operator.shape[0] - 2)
        try:
            values, vectors = eigs(
                operator,
                k=requested,
                sigma=target,
                which="LM",
                tol=1e-8,
                maxiter=max(8000, 6 * operator.shape[0]),
                v0=v0,
            )
        except ArpackNoConvergence as error:
            if error.eigenvalues is None or len(error.eigenvalues) < len(group):
                raise
            values = error.eigenvalues
            vectors = error.eigenvectors
            requested = len(values)
        normalized_vectors = vectors / np.linalg.norm(vectors, axis=0, keepdims=True)
        costs = np.empty((len(group), requested), dtype=float)
        overlaps = np.empty_like(costs)
        for local_row, band_index in enumerate(group):
            band_vector = basis[:, band_index]
            for candidate in range(requested):
                fields = normalized_vectors[:, candidate].reshape(6, grid.points)
                projection = np.einsum("a,ap->p", band_vector.conj(), fields)
                overlap = float(np.sum(np.abs(projection) ** 2))
                overlaps[local_row, candidate] = overlap
                passive_penalty = 20.0 if values[candidate].imag < 0.0 else 0.0
                costs[local_row, candidate] = (
                    abs(values[candidate].real - targets[band_index].real)
                    / max(abs(targets[band_index].real), 2e4)
                    + 0.25 * _hex_roughness(fields, grid)
                    + 0.45 * (1.0 - overlap)
                    + passive_penalty
                )
        rows, columns = linear_sum_assignment(costs)
        assignment = {int(row): int(column) for row, column in zip(rows, columns, strict=True)}
        for local_row, band_index in enumerate(group):
            candidate = assignment[local_row]
            value = values[candidate]
            if value.imag < -1e-8 * max(float(np.linalg.norm(result.coupling_m)), 1.0):
                raise RuntimeError("A selected passive six-wave eigenmode has negative loss")
            fields = normalized_vectors[:, candidate].reshape(6, grid.points)
            radiation_x, radiation_y = result.radiation_amplitudes(fields)
            selected_modes[TRIANGULAR_MODE_NAMES[band_index]] = TriangularFiniteMode(
                name=TRIANGULAR_MODE_NAMES[band_index],
                delta_per_m=float(value.real),
                alpha_per_m=float(max(value.imag, 0.0)),
                fields=fields,
                band_overlap=float(overlaps[local_row, candidate]),
                grid=grid,
                radiation_x=np.asarray(radiation_x),
                radiation_y=np.asarray(radiation_y),
            )
    return {name: selected_modes[name] for name in TRIANGULAR_MODE_NAMES}


def solve_triangular_finite_modes_converged(
    result: TriangularCouplingResult,
    spec: TriangularFiniteSpec,
    radius_cells: tuple[int, ...],
) -> tuple[dict[str, TriangularFiniteMode], dict[str, np.ndarray]]:
    """Extrapolate first-order characteristic-grid eigenvalues versus ``1/N``.

    Fields and radiation apertures come from the finest grid.  Only the reported
    complex eigenvalue is replaced by the zero-spacing intercept.
    """
    if len(radius_cells) < 3 or any(
        left >= right for left, right in zip(radius_cells, radius_cells[1:])
    ):
        raise ValueError("radius_cells must contain at least three increasing values")
    solutions = [
        solve_triangular_finite_modes(result, replace(spec, radius_cells=cells))
        for cells in radius_cells
    ]
    inverse = 1.0 / np.asarray(radius_cells, dtype=float)
    answer: dict[str, TriangularFiniteMode] = {}
    convergence: dict[str, np.ndarray] = {
        "inverse_grid": inverse,
        "radius_cells": np.asarray(radius_cells, dtype=float),
    }
    for name in TRIANGULAR_MODE_NAMES:
        values = np.asarray([
            solution[name].delta_per_m + 1j * solution[name].alpha_per_m
            for solution in solutions
        ])
        delta_limit = float(np.polyfit(inverse, values.real, 1)[1])
        alpha_limit, raw_alpha, uncertainty, status = extrapolate_grid_loss(
            inverse, values.imag
        )
        if status != "provisional":
            warnings.warn(
                f"Six-wave mode {name}: {status}; raw alpha intercept={raw_alpha:.6g} "
                f"m^-1, grid sensitivity={uncertainty:.6g} m^-1. Refine the grid "
                "before using this loss for design ranking.", RuntimeWarning, stacklevel=2,
            )
        answer[name] = replace(
            solutions[-1][name],
            delta_per_m=delta_limit,
            alpha_per_m=alpha_limit,
            grid_alpha_per_m=solutions[-1][name].alpha_per_m,
            extrapolation_uncertainty_per_m=uncertainty,
            extrapolation_status=status,
        )
        convergence[name] = values
        convergence[f"{name}_alpha_diagnostics"] = np.asarray(
            [raw_alpha, uncertainty, solutions[-1][name].alpha_per_m]
        )
    return answer, convergence


def plot_triangular_grid_convergence(
    convergence: dict[str, np.ndarray], radius_um: float, path: Path
) -> None:
    """Plot finite eigenvalue convergence and the reported ``1/N -> 0`` intercept."""
    inverse = convergence["inverse_grid"]
    length_m = radius_um * 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for name in TRIANGULAR_MODE_NAMES:
        values = convergence[name]
        xfit = np.linspace(0.0, float(inverse.max()), 100)
        fit_delta = np.polyfit(inverse, values.real * length_m, 1)
        fit_alpha = np.polyfit(inverse, values.imag * length_m, 1)
        axes[0].plot(inverse, values.real * length_m, "o", label=name)
        axes[0].plot(xfit, np.polyval(fit_delta, xfit), lw=1.0)
        axes[1].plot(inverse, values.imag * length_m, "o", label=name)
        axes[1].plot(xfit, np.polyval(fit_alpha, xfit), lw=1.0)
    axes[0].set(xlabel="1/N", ylabel="delta L", title="frequency convergence")
    axes[1].set(xlabel="1/N", ylabel="alpha L", title="threshold convergence")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(ncol=2)
    fig.suptitle("Triangular finite-device grid convergence")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def triangular_vector_far_field(
    mode: TriangularFiniteMode,
    wavelength_nm: float,
    view_deg: float = 3.0,
    samples: int = 121,
) -> TriangularFarField:
    """Point-sampled aperture FFP, restricted to a nonaliased angular window.

    The coarse characteristic lattice produces reciprocal-lattice copies if
    evaluated beyond its sampling Brillouin zone.  Restrict all four corners
    of the requested square to that zone.  Energy ratios and RMS are relative
    to this evaluated window; they are not whole-hemisphere beam metrics.
    """
    if view_deg <= 0.0 or samples < 3 or wavelength_nm <= 0.0:
        raise ValueError("view_deg and wavelength_nm must be positive; samples >= 3")
    if samples % 2 == 0:
        samples += 1  # Keep the physical surface-normal sample exactly at zero.
    wavelength_m = wavelength_nm * 1e-9
    # Hexagonal sampling BZ: |kx +/- ky/sqrt(3)| <= 4*pi/(3*h).
    safe_tangent = 2.0 * wavelength_m / (
        3.0 * mode.grid.spacing_m * (1.0 + 1.0 / np.sqrt(3.0))
    )
    safe_view = float(np.rad2deg(np.arctan(safe_tangent)))
    evaluated_view = min(float(view_deg), 0.98 * safe_view)
    if view_deg > safe_view:
        warnings.warn(
            f"Requested six-wave far-field view +/-{view_deg:g} deg exceeds the "
            f"coarse-grid alias-free square +/-{safe_view:.3g} deg; evaluated "
            f"+/-{evaluated_view:.3g} deg. Encircled energy is window-normalized.",
            RuntimeWarning, stacklevel=2,
        )
    angle = np.linspace(-evaluated_view, evaluated_view, int(samples))
    theta_x, theta_y = np.meshgrid(np.deg2rad(angle), np.deg2rad(angle))
    k0 = 2.0 * np.pi / (wavelength_nm * 1e-9)
    kx = k0 * np.tan(theta_x.ravel())
    ky = k0 * np.tan(theta_y.ravel())
    x_m = mode.grid.x_um * 1e-6
    y_m = mode.grid.y_um * 1e-6
    output_x = np.empty(kx.size, dtype=np.complex128)
    output_y = np.empty_like(output_x)
    chunk = 512
    for start in range(0, kx.size, chunk):
        stop = min(start + chunk, kx.size)
        phase = np.exp(-1j * (
            kx[start:stop, None] * x_m[None, :]
            + ky[start:stop, None] * y_m[None, :]
        ))
        output_x[start:stop] = phase @ mode.radiation_x
        output_y[start:stop] = phase @ mode.radiation_y
    obliquity = np.cos(theta_x) + np.cos(theta_y) - 1.0
    field_x = output_x.reshape(theta_x.shape) * obliquity
    field_y = output_y.reshape(theta_y.shape) * obliquity
    power_x = np.abs(field_x) ** 2
    power_y = np.abs(field_y) ** 2
    power = power_x + power_y
    scale = max(float(power.max()), np.finfo(float).eps)
    power /= scale
    power_x /= scale
    power_y /= scale
    xx, yy = np.meshgrid(angle, angle)
    total = max(float(power.sum()), np.finfo(float).eps)
    divergence = 2.0 * np.sqrt(float(np.sum((xx * xx + yy * yy) * power) / total))
    center = len(angle) // 2
    peak_row, peak_column = np.unravel_index(int(np.argmax(power)), power.shape)
    peak_offset = float(np.hypot(xx[peak_row, peak_column], yy[peak_row, peak_column]))
    centroid_x = float(np.sum(xx * power) / total)
    centroid_y = float(np.sum(yy * power) / total)
    centroid_offset = float(np.hypot(centroid_x, centroid_y))
    centered_x = xx - centroid_x
    centered_y = yy - centroid_y
    covariance = np.asarray((
        (
            float(np.sum(centered_x * centered_x * power) / total),
            float(np.sum(centered_x * centered_y * power) / total),
        ),
        (
            float(np.sum(centered_x * centered_y * power) / total),
            float(np.sum(centered_y * centered_y * power) / total),
        ),
    ))
    principal_variances = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    ellipticity = float(np.sqrt(
        principal_variances[-1]
        / max(principal_variances[0], np.finfo(float).eps)
    ))
    radial_angle = np.hypot(xx, yy)
    encircled_0p5 = float(np.sum(power[radial_angle <= 0.5]) / total)
    encircled = float(np.sum(power[radial_angle <= 1.0]) / total)
    return TriangularFarField(
        angle_deg=angle,
        field_x=field_x,
        field_y=field_y,
        power=power,
        power_x=power_x,
        power_y=power_y,
        full_rms_divergence_deg=divergence,
        center_to_peak=float(power[center, center]),
        peak_offset_deg=peak_offset,
        centroid_offset_deg=centroid_offset,
        ellipticity=ellipticity,
        encircled_power_0p5deg=encircled_0p5,
        encircled_power_1deg=encircled,
        requested_view_deg=float(view_deg),
        evaluated_view_deg=evaluated_view,
        alias_free_square_view_deg=safe_view,
    )


def _tripcolor(ax: plt.Axes, mode: TriangularFiniteMode) -> None:
    image = ax.tripcolor(
        mode.grid.x_um,
        mode.grid.y_um,
        mode.intensity,
        shading="gouraud",
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
    )
    radius = mode.grid.radius_um
    if mode.grid.aperture_shape == "circle":
        boundary = Circle((0.0, 0.0), radius, fill=False, ls="--", ec="white")
    elif mode.grid.aperture_shape == "hexagon":
        phase = np.deg2rad(np.arange(0.0, 360.0, 60.0))
        boundary = Polygon(
            np.column_stack((radius * np.cos(phase), radius * np.sin(phase))),
            closed=True, fill=False, ls="--", ec="white",
        )
    else:
        boundary = Rectangle(
            (-radius, -radius), 2.0 * radius, 2.0 * radius,
            fill=False, ls="--", ec="white",
        )
    ax.add_patch(boundary)
    ax.set_aspect("equal")
    return image


def write_triangular_finite_table(
    modes: dict[str, TriangularFiniteMode],
    result: TriangularCouplingResult,
    path: Path,
) -> None:
    """Save finite eigenvalues and directly calculated far-field metrics."""
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "mode", "detuning_cm-1", "threshold_alpha_cm-1", "alpha_L",
            "band_overlap", "far_field_center_to_peak", "full_rms_divergence_deg",
            "peak_offset_deg", "centroid_offset_deg", "ellipticity",
            "encircled_power_0p5deg", "encircled_power_1deg",
        ))
        for name in TRIANGULAR_MODE_NAMES:
            mode = modes[name]
            far = triangular_vector_far_field(mode, result.bragg_wavelength_nm)
            writer.writerow((
                name,
                mode.delta_per_m / 100.0,
                mode.alpha_per_m / 100.0,
                mode.alpha_l,
                mode.band_overlap,
                far.center_to_peak,
                far.full_rms_divergence_deg,
                far.peak_offset_deg,
                far.centroid_offset_deg,
                far.ellipticity,
                far.encircled_power_0p5deg,
                far.encircled_power_1deg,
            ))


def plot_triangular_thresholds(
    modes: dict[str, TriangularFiniteMode], path: Path
) -> None:
    names = list(TRIANGULAR_MODE_NAMES)
    losses = [modes[name].alpha_per_m / 100.0 for name in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(names, losses, color="#4c78a8")
    for bar, mode in zip(bars, (modes[name] for name in names), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{mode.alpha_l:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set(
        ylabel=r"cold-cavity amplitude threshold $\alpha$ (cm$^{-1}$)",
        title=r"Finite triangular device; labels show $\alpha L$",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_triangular_mode_atlas(
    modes: dict[str, TriangularFiniteMode],
    result: TriangularCouplingResult,
    path: Path,
    view_deg: float = 3.0,
) -> None:
    """Plot whole-device envelope, one-cell state, and physical vector FFP."""
    fig, axes = plt.subplots(6, 3, figsize=(11.6, 18.0), constrained_layout=True)
    for row, name in enumerate(TRIANGULAR_MODE_NAMES):
        mode = modes[name]
        _tripcolor(axes[row, 0], mode)
        axes[row, 0].set(
            title=f"{name}: whole-device envelope",
            xlabel="x (um)",
            ylabel="y (um)",
        )
        central = int(np.argmin(mode.grid.x_um ** 2 + mode.grid.y_um ** 2))
        coefficients = mode.fields[:, central]
        x, y, cell_intensity, _, _ = result.unit_cell_fields(coefficients, points=81)
        axes[row, 1].pcolormesh(x, y, cell_intensity, shading="auto", cmap="turbo")
        axes[row, 1].set_aspect("equal")
        axes[row, 1].set(
            title=f"{name}: one-cell intensity",
            xlabel="x/a",
            ylabel="y/a",
        )
        far = triangular_vector_far_field(
            mode, result.bragg_wavelength_nm, view_deg=view_deg
        )
        extent = (
            far.angle_deg[0], far.angle_deg[-1],
            far.angle_deg[0], far.angle_deg[-1],
        )
        axes[row, 2].imshow(
            far.power,
            origin="lower",
            extent=extent,
            cmap="hot",
            norm=PowerNorm(gamma=0.65, vmin=0.0, vmax=1.0),
        )
        axes[row, 2].set(
            title=(
                f"{name}: FFP C={far.center_to_peak:.2f}, "
                f"O={far.peak_offset_deg:.2f} deg, "
                f"E1={far.encircled_power_1deg:.2f}"
            ),
            xlabel="theta_x (deg)",
            ylabel="theta_y (deg)",
        )
    fig.suptitle("Six-wave finite modes: envelope, cell field and vector far field")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_triangular_far_field_diagnostics(
    mode: TriangularFiniteMode,
    result: TriangularCouplingResult,
    path: Path,
    view_deg: float = 3.0,
) -> None:
    """Plot total and polarized far fields plus phase/polarization axes."""
    far = triangular_vector_far_field(mode, result.bragg_wavelength_nm, view_deg=view_deg)
    total = far.power
    phase = np.angle(far.field_x + 1j * far.field_y)
    s1 = far.power_x - far.power_y
    cross = 2.0 * np.real(far.field_x * np.conj(far.field_y))
    scale = max(float(total.max()), np.finfo(float).eps)
    orientation = 0.5 * np.arctan2(cross / scale, s1)
    extent = (
        far.angle_deg[0], far.angle_deg[-1],
        far.angle_deg[0], far.angle_deg[-1],
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2))
    for ax, values, title in (
        (axes[0, 0], total, "total |Fx|^2+|Fy|^2"),
        (axes[0, 1], far.power_x, "x-polarized |Fx|^2"),
        (axes[1, 0], far.power_y, "y-polarized |Fy|^2"),
    ):
        ax.imshow(
            values, origin="lower", extent=extent, cmap="hot",
            norm=PowerNorm(gamma=0.65, vmin=0.0, vmax=1.0),
        )
        ax.set_title(title)
    axes[1, 1].imshow(
        phase, origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    stride = max(1, len(far.angle_deg) // 13)
    xgrid, ygrid = np.meshgrid(far.angle_deg, far.angle_deg)
    amplitude = np.sqrt(total)
    axes[1, 1].quiver(
        xgrid[::stride, ::stride],
        ygrid[::stride, ::stride],
        amplitude[::stride, ::stride] * np.cos(orientation[::stride, ::stride]),
        amplitude[::stride, ::stride] * np.sin(orientation[::stride, ::stride]),
        color="white",
        pivot="middle",
        scale=10.0,
        width=0.004,
    )
    axes[1, 1].set_title("phase of Fx+iFy and polarization axes")
    for ax in axes.flat:
        ax.set(xlabel="theta_x (deg)", ylabel="theta_y (deg)")
    fig.suptitle(
        f"Mode {mode.name}: center/max={far.center_to_peak:.3f}, "
        f"peak offset={far.peak_offset_deg:.3f} deg, "
        f"E1={far.encircled_power_1deg:.3f}, "
        f"ellipticity={far.ellipticity:.3f}"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)

