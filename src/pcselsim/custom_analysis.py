"""Reusable finite-area and visualization tools for custom PCSEL studies.

The optical state is the same square-lattice four-wave vector used by the
Inoue time-domain solver: (Rx, Sx, Ry, Sy).  This module provides the linear
finite-area companion problem used for thresholds, envelopes and far fields.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle, Ellipse as EllipsePatch, Polygon, Rectangle
from scipy import sparse
from scipy.sparse.linalg import eigs

from .coupling import band_edge_basis, validate_passive_coupling


MODE_NAMES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class LatticeInclusionSpec:
    """One inclusion of a multi-feature unit-cell motif, in dimensional units."""

    shape: str
    radius_x_nm: float
    radius_y_nm: float
    center_x_nm: float = 0.0
    center_y_nm: float = 0.0
    rotation_deg: float = 0.0

    @property
    def area_nm2(self) -> float:
        if self.shape in {"circle", "ellipse"}:
            return float(np.pi*self.radius_x_nm*self.radius_y_nm)
        if self.shape in {"triangle", "rit"}:
            return float(2.0*self.radius_x_nm*self.radius_y_nm)
        raise ValueError(f"Unsupported inclusion shape: {self.shape}")


@dataclass(frozen=True)
class LatticeSpec:
    lattice_type: str
    constant_nm: float
    hole_shape: str
    hole_radius_x_nm: float
    hole_radius_y_nm: float
    hole_rotation_deg: float
    background_index: float
    hole_index: float
    inclusions: tuple[LatticeInclusionSpec, ...] = ()

    @property
    def fill_fraction(self) -> float:
        if self.inclusions:
            area = sum(item.area_nm2 for item in self.inclusions)
        elif self.hole_shape in {"circle", "ellipse"}:
            area = np.pi*self.hole_radius_x_nm*self.hole_radius_y_nm
        elif self.hole_shape in {"triangle", "rit"}:
            area = 2.0*self.hole_radius_x_nm*self.hole_radius_y_nm
        else:
            raise ValueError(f"Unsupported hole_shape: {self.hole_shape}")
        cell_area = self.constant_nm**2
        if self.lattice_type == "triangular":
            cell_area *= np.sqrt(3.0) / 2.0
        return float(area / cell_area)


@dataclass(frozen=True)
class LayerSpec:
    name: str
    thickness_nm: float
    refractive_index: float
    color: str = "#80b1d3"


@dataclass(frozen=True)
class FourWaveOpticalSpec:
    wavelength_nm: float
    effective_index: float
    group_index: float
    domain_um: float
    internal_loss_cm: float
    modal_detuning_cm: tuple[float, float, float, float]
    modal_radiation_loss_cm: tuple[float, float, float, float]
    grid_points: int = 17


@dataclass(frozen=True)
class LinearMode:
    name: str
    delta_per_m: float
    alpha_per_m: float
    fields: np.ndarray
    band_overlap: float
    radiation_x: np.ndarray | None = None
    radiation_y: np.ndarray | None = None

    @property
    def intensity(self) -> np.ndarray:
        value = np.sum(np.abs(self.fields) ** 2, axis=0)
        return value / max(float(value.max()), np.finfo(float).eps)


def ensure_square_four_wave(lattice: LatticeSpec) -> None:
    """Reject unsupported lattices instead of silently applying wrong physics."""
    if lattice.lattice_type != "square":
        raise ValueError(
            "The current Rx/Sx/Ry/Sy four-wave solver is valid for a square lattice. "
            "A triangular lattice requires a six-wave basis. Geometry and k-space "
            "plots may still be generated with the optical-solver steps disabled."
        )


def coupling_from_modal_values(spec: FourWaveOpticalSpec) -> np.ndarray:
    basis = band_edge_basis()
    values = 100.0 * (
        np.asarray(spec.modal_detuning_cm, dtype=float)
        + 1j * np.asarray(spec.modal_radiation_loss_cm, dtype=float)
    )
    matrix = basis @ np.diag(values) @ basis.conj().T
    validate_passive_coupling(matrix)
    return matrix


def _upwind_1d(n: int, spacing_m: float, positive_direction: bool) -> sparse.csr_matrix:
    """Derivative with a zero incoming ghost cell and an outgoing boundary."""
    if positive_direction:
        # d/dx = (u_j-u_{j-1})/dx; incoming u(-dx/2)=0.
        return sparse.diags(
            (-np.ones(n-1)/spacing_m, np.ones(n)/spacing_m),
            offsets=(-1, 0),
            shape=(n, n),
            format="csr",
        )
    # -d/dx = (u_j-u_{j+1})/dx; incoming u(L+dx/2)=0.
    return sparse.diags(
        (np.ones(n)/spacing_m, -np.ones(n-1)/spacing_m),
        offsets=(0, 1),
        shape=(n, n),
        format="csr",
    )


def finite_area_operator(spec: FourWaveOpticalSpec, coupling: np.ndarray) -> sparse.csr_matrix:
    """Discretize (delta+i alpha)Phi = C Phi + i D Phi."""
    n = spec.grid_points
    if n < 9:
        raise ValueError("grid_points must be >= 9")
    spacing_m = spec.domain_um * 1e-6 / n
    identity_n = sparse.identity(n, format="csr")
    d_plus = _upwind_1d(n, spacing_m, True)
    d_minus = _upwind_1d(n, spacing_m, False)
    dx_plus = sparse.kron(identity_n, d_plus, format="csr")
    dx_minus = sparse.kron(identity_n, d_minus, format="csr")
    dy_plus = sparse.kron(d_plus, identity_n, format="csr")
    dy_minus = sparse.kron(d_minus, identity_n, format="csr")
    points = n*n
    local = sparse.kron(
        sparse.csr_matrix(coupling), sparse.identity(points, format="csr"), format="csr"
    )
    transport = sparse.block_diag((dx_plus, dx_minus, dy_plus, dy_minus), format="csr")
    return local + 1j*transport


def _roughness(fields: np.ndarray) -> float:
    denominator = max(float(np.mean(np.abs(fields)**2)), np.finfo(float).eps)
    numerator = sum(float(np.mean(np.abs(np.diff(fields, axis=axis))**2)) for axis in (1, 2))
    return numerator / denominator


def analytic_mode_fields(n: int, mode_name: str, order: int = 0) -> np.ndarray:
    """Smooth single-lobed envelope in the shared A/B/C/D basis."""
    axis = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(axis, axis)
    envelope = np.exp(-2.8*(x*x+y*y))
    if order == 1:
        envelope *= 2.7*x
    elif order == 2:
        envelope *= 2.7*y
    elif order == 3:
        envelope *= 1.0-4.0*x*x
    vector = band_edge_basis()[:, MODE_NAMES.index(mode_name)]
    fields = vector[:, None, None]*envelope[None, :, :]
    return fields / np.sqrt(np.sum(np.abs(fields)**2))


def solve_finite_modes(
    spec: FourWaveOpticalSpec,
    coupling: np.ndarray | None = None,
    band_basis: np.ndarray | None = None,
    radiation_builder=None,
) -> dict[str, LinearMode]:
    """Return fundamental finite-area states near the four infinite band edges.

    ``coupling`` and ``band_basis`` allow the operator to use a geometry-derived
    3-D CWT matrix.  The calibrated A/B/C/D matrix remains the backwards-
    compatible fallback.  The returned fields are the actual eigenvectors;
    they are never replaced by a cosmetic analytic envelope.
    """
    coupling = coupling_from_modal_values(spec) if coupling is None else coupling
    operator = finite_area_operator(spec, coupling)
    if band_basis is None:
        values, vectors = np.linalg.eig(coupling)
        ordering = np.argsort(values.real)
        basis = vectors[:, ordering]
        basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    else:
        basis = np.asarray(band_basis, dtype=np.complex128)
    targets = np.diag(basis.conj().T @ coupling @ basis)
    n = spec.grid_points
    sample = np.arange(operator.shape[0], dtype=float)
    phase = 2.0*np.pi*sample/operator.shape[0]
    # scipy.eigs otherwise chooses an implicit random start.  A deterministic,
    # symmetry-breaking vector makes degenerate-subspace selection reproducible.
    v0 = (
        1.0+0.23*np.cos(phase)+0.11*np.sin(3.0*phase)
        + 1j*(0.19*np.sin(phase)+0.07*np.cos(5.0*phase))
    )
    results: dict[str, LinearMode] = {}
    for mode_index, (name, target) in enumerate(zip(MODE_NAMES, targets, strict=True)):
        values, vectors = eigs(
            operator, k=10, sigma=complex(target), which="LM", tol=1e-7, v0=v0
        )
        candidates: list[tuple[float, int, float]] = []
        degenerate = np.flatnonzero(np.abs(targets-target) < 1e-7*max(abs(target), 1.0))
        for index, value in enumerate(values):
            fields = vectors[:, index].reshape(4, n, n)
            projections = np.einsum(
                "ak,axy->kxy", basis[:, degenerate].conj(), fields
            )
            overlap = float(
                np.sum(np.abs(projections)**2)/np.sum(np.abs(fields)**2)
            )
            passive_penalty = 10.0 if value.imag < 0.0 else 0.0
            score = (
                abs(value.real-target.real)/max(abs(target.real), 2e4)
                + 0.35*_roughness(fields)
                + 0.30*(1.0-overlap)
                + 0.08*max(value.imag, 0.0)/max(abs(target.imag), 2e4)
                + passive_penalty
            )
            candidates.append((score, index, overlap))
        _, selected, overlap = min(candidates)
        value = values[selected]
        fields = vectors[:, selected].reshape(4, n, n)
        fields /= np.sqrt(np.sum(np.abs(fields)**2))
        radiation_x = radiation_y = None
        if radiation_builder is not None:
            radiation_x, radiation_y = radiation_builder(fields)
        results[name] = LinearMode(
            name=name,
            delta_per_m=float(value.real),
            alpha_per_m=float(max(value.imag, 0.0)),
            fields=fields,
            band_overlap=overlap,
            radiation_x=radiation_x,
            radiation_y=radiation_y,
        )
    # Circular/square structures contain exactly degenerate C/D subspaces.
    # ARPACK may return one arbitrary member repeatedly.  Generate its symmetry
    # partner by a 90-degree rotation instead of selecting a transverse overtone.
    if abs(targets[2]-targets[3]) < 1e-7*max(abs(targets[2]), 1.0):
        c_fields = results["C"].fields
        rotated = np.empty_like(c_fields)
        rotated[2] = np.rot90(c_fields[0])  # +x -> +y
        rotated[3] = np.rot90(c_fields[1])  # -x -> -y
        rotated[1] = np.rot90(c_fields[2])  # +y -> -x
        rotated[0] = np.rot90(c_fields[3])  # -y -> +x
        radiation_x = radiation_y = None
        if radiation_builder is not None:
            radiation_x, radiation_y = radiation_builder(rotated)
        results["D"] = replace(
            results["C"], name="D", fields=rotated,
            radiation_x=radiation_x, radiation_y=radiation_y,
        )
    return results


def solve_finite_modes_converged(
    spec: FourWaveOpticalSpec,
    grid_points: tuple[int, ...],
    coupling: np.ndarray,
    band_basis: np.ndarray,
    radiation_builder=None,
) -> tuple[dict[str, LinearMode], dict[str, np.ndarray]]:
    """First-order grid convergence and zero-spacing extrapolation.

    The incoming-boundary finite-volume derivative is first-order accurate.
    Fitting each complex eigenvalue against 1/N removes its leading numerical
    outflow error.  The eigenfield/radiation map is retained from the finest
    grid, while reported delta and alpha use the intercept.
    """
    if len(grid_points) < 3 or tuple(sorted(grid_points)) != grid_points:
        raise ValueError("grid_points must contain at least three increasing sizes")
    solutions = []
    for points in grid_points:
        solutions.append(
            solve_finite_modes(
                replace(spec, grid_points=points), coupling, band_basis,
                radiation_builder,
            )
        )
    inverse_grid = 1.0/np.asarray(grid_points, dtype=float)
    answer: dict[str, LinearMode] = {}
    convergence: dict[str, np.ndarray] = {}
    for name in MODE_NAMES:
        values = np.asarray([
            item[name].delta_per_m+1j*item[name].alpha_per_m for item in solutions
        ])
        delta_limit = float(np.polyfit(inverse_grid, values.real, 1)[1])
        alpha_limit = float(max(np.polyfit(inverse_grid, values.imag, 1)[1], 0.0))
        answer[name] = replace(
            solutions[-1][name], delta_per_m=delta_limit, alpha_per_m=alpha_limit
        )
        convergence[name] = values
    convergence["inverse_grid"] = inverse_grid
    convergence["grid_points"] = np.asarray(grid_points, dtype=float)
    return answer, convergence


def plot_grid_convergence(
    spec: FourWaveOpticalSpec, convergence: dict[str, np.ndarray], path: Path
) -> None:
    inverse_grid = convergence["inverse_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for name in MODE_NAMES:
        values = convergence[name]
        fit_delta = np.polyfit(inverse_grid, values.real*spec.domain_um*1e-6, 1)
        fit_alpha = np.polyfit(inverse_grid, values.imag*spec.domain_um*1e-6, 1)
        xfit = np.linspace(0.0, inverse_grid.max(), 100)
        axes[0].plot(inverse_grid, values.real*spec.domain_um*1e-6, "o", label=name)
        axes[0].plot(xfit, np.polyval(fit_delta, xfit), lw=1)
        axes[1].plot(inverse_grid, values.imag*spec.domain_um*1e-6, "o", label=name)
        axes[1].plot(xfit, np.polyval(fit_alpha, xfit), lw=1)
    axes[0].set(xlabel="1/N", ylabel="delta L", title="frequency convergence")
    axes[1].set(xlabel="1/N", ylabel="alpha L", title="threshold convergence")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(ncol=2)
    fig.suptitle("Finite-area grid convergence; intercept is the reported value")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def unit_cell_field(mode_name: str, points: int = 161) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast Bloch field in one cell from the four basic-wave coefficients."""
    axis = np.linspace(-0.5, 0.5, points)
    x, y = np.meshgrid(axis, axis)
    rx, sx, ry, sy = band_edge_basis()[:, MODE_NAMES.index(mode_name)]
    ey = rx*np.exp(-2j*np.pi*x) + sx*np.exp(2j*np.pi*x)
    ex = ry*np.exp(-2j*np.pi*y) + sy*np.exp(2j*np.pi*y)
    intensity = np.abs(ex)**2 + np.abs(ey)**2
    return axis, intensity/intensity.max(), np.angle(ex+ey)


def unit_cell_field_from_coefficients(
    coefficients: np.ndarray, points: int = 161
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-0.5, 0.5, points)
    x, y = np.meshgrid(axis, axis)
    rx, sx, ry, sy = coefficients
    ey = rx*np.exp(-2j*np.pi*x)+sx*np.exp(2j*np.pi*x)
    ex = ry*np.exp(-2j*np.pi*y)+sy*np.exp(2j*np.pi*y)
    intensity = np.abs(ex)**2+np.abs(ey)**2
    return axis, intensity/max(float(intensity.max()), np.finfo(float).eps), np.angle(ex+ey)


def far_field(intensity_field: np.ndarray, wavelength_nm: float, length_um: float,
              view_deg: float = 3.0, padding: int = 8) -> tuple[np.ndarray, np.ndarray, float]:
    """Return angle grid, normalized FFP, and second-moment divergence angle."""
    n = intensity_field.shape[0]
    padded_n = padding*n
    padded = np.zeros((padded_n, padded_n), dtype=np.complex128)
    start = (padded_n-n)//2
    padded[start:start+n, start:start+n] = intensity_field
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    power = np.abs(spectrum)**2
    power /= max(float(power.max()), np.finfo(float).eps)
    spacing_m = length_um*1e-6/n
    frequency = np.fft.fftshift(np.fft.fftfreq(padded_n, d=spacing_m))
    # Liang Eq. (4.24): kx=k0*tan(theta_x), hence theta=atan(lambda*f).
    angle = np.rad2deg(np.arctan(wavelength_nm*1e-9*frequency))
    keep = np.flatnonzero(np.abs(angle) <= view_deg)
    first, last = int(keep[0]), int(keep[-1])+1
    angle = angle[first:last]
    power = power[first:last, first:last]
    xx, yy = np.meshgrid(angle, angle)
    total = max(float(power.sum()), np.finfo(float).eps)
    rms_radius = np.sqrt(float(np.sum((xx*xx+yy*yy)*power)/total))
    divergence_full_deg = 2.0*rms_radius
    return angle, power, divergence_full_deg


def vector_far_field(
    field_x: np.ndarray,
    field_y: np.ndarray,
    wavelength_nm: float,
    length_um: float,
    view_deg: float = 3.0,
    padding: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Liang Eqs. (4.24)-(4.26) for two complex radiation components."""
    n = field_x.shape[0]
    padded_n = padding*n
    spectra = []
    for field in (field_x, field_y):
        padded = np.zeros((padded_n, padded_n), dtype=np.complex128)
        start = (padded_n-n)//2
        padded[start:start+n, start:start+n] = field
        spectra.append(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded))))
    spacing_m = length_um*1e-6/n
    frequency = np.fft.fftshift(np.fft.fftfreq(padded_n, d=spacing_m))
    angle_full = np.rad2deg(np.arctan(wavelength_nm*1e-9*frequency))
    theta_x, theta_y = np.meshgrid(np.deg2rad(angle_full), np.deg2rad(angle_full))
    obliquity = np.cos(theta_x)+np.cos(theta_y)-1.0
    spectrum_x = spectra[0]*obliquity
    spectrum_y = spectra[1]*obliquity
    power_x = np.abs(spectrum_x)**2
    power_y = np.abs(spectrum_y)**2
    power = power_x+power_y
    scale = max(float(power.max()), np.finfo(float).eps)
    power, power_x, power_y = power/scale, power_x/scale, power_y/scale
    keep = np.flatnonzero(np.abs(angle_full) <= view_deg)
    first, last = int(keep[0]), int(keep[-1])+1
    angle = angle_full[first:last]
    power = power[first:last, first:last]
    power_x = power_x[first:last, first:last]
    power_y = power_y[first:last, first:last]
    xx, yy = np.meshgrid(angle, angle)
    total = max(float(power.sum()), np.finfo(float).eps)
    divergence_full_deg = 2.0*np.sqrt(float(np.sum((xx*xx+yy*yy)*power)/total))
    return angle, power, power_x, power_y, divergence_full_deg


def vector_far_field_complex(
    field_x: np.ndarray,
    field_y: np.ndarray,
    wavelength_nm: float,
    length_um: float,
    view_deg: float = 1.5,
    padding: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cropped complex Fx/Fy including Liang's obliquity factor."""
    n = field_x.shape[0]
    padded_n = padding*n
    spectra = []
    for field in (field_x, field_y):
        padded = np.zeros((padded_n, padded_n), dtype=np.complex128)
        start = (padded_n-n)//2
        padded[start:start+n, start:start+n] = field
        spectra.append(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded))))
    spacing_m = length_um*1e-6/n
    frequency = np.fft.fftshift(np.fft.fftfreq(padded_n, d=spacing_m))
    angle_full = np.rad2deg(np.arctan(wavelength_nm*1e-9*frequency))
    theta_x, theta_y = np.meshgrid(np.deg2rad(angle_full), np.deg2rad(angle_full))
    obliquity = np.cos(theta_x)+np.cos(theta_y)-1.0
    keep = np.flatnonzero(np.abs(angle_full) <= view_deg)
    first, last = int(keep[0]), int(keep[-1])+1
    return (
        angle_full[first:last],
        spectra[0][first:last, first:last]*obliquity[first:last, first:last],
        spectra[1][first:last, first:last]*obliquity[first:last, first:last],
    )


def radiation_proxy_field(mode: LinearMode) -> np.ndarray:
    """Construct the aperture field used when an explicit 3-D Crad is absent.

    Summing the four in-plane waves would make symmetry-protected A/B modes
    identically zero.  A real finite PCSEL radiates through the vertical
    coupling matrix and its edges.  Until a device-specific 3-D radiation
    matrix is supplied, retain the computed envelope and attach the lowest
    symmetry-allowed aperture factor.  The resulting width is useful for
    aperture/divergence studies, but its detailed lobes are not a substitute
    for RCWA/FEM/FDTD far-field reconstruction.
    """
    n = mode.fields.shape[-1]
    axis = np.linspace(-1.0, 1.0, n)
    x, y = np.meshgrid(axis, axis)
    envelope = np.sqrt(mode.intensity)
    factors = {
        "A": x + 1j*y,
        "B": x - 1j*y,
        "C": np.ones_like(x, dtype=np.complex128),
        "D": (1.0 + 0.30*x - 0.30*y).astype(np.complex128),
    }
    field = envelope*factors[mode.name]
    return field/max(float(np.max(np.abs(field))), np.finfo(float).eps)


def _draw_inclusion(
    ax: plt.Axes,
    lattice: LatticeSpec,
    center: tuple[float, float],
    inclusion: LatticeInclusionSpec,
) -> None:
    a = lattice.constant_nm
    rx = inclusion.radius_x_nm/a
    ry = inclusion.radius_y_nm/a
    cx = center[0]+inclusion.center_x_nm/a
    cy = center[1]+inclusion.center_y_nm/a
    if inclusion.shape == "circle":
        patch = Circle((cx, cy), rx, facecolor="white", edgecolor="0.25")
    elif inclusion.shape == "ellipse":
        patch = EllipsePatch((cx, cy), 2*rx, 2*ry, angle=inclusion.rotation_deg,
                             facecolor="white", edgecolor="0.25")
    elif inclusion.shape == "triangle":
        vertices = np.array([[-rx, -ry], [rx, 0.0], [-rx, ry]])
        theta = np.deg2rad(inclusion.rotation_deg)
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        vertices = vertices @ rotation.T + np.array([cx, cy])
        patch = Polygon(vertices, facecolor="white", edgecolor="0.25", joinstyle="round")
    elif inclusion.shape == "rit":
        vertices = np.array([[-rx, -ry], [rx, -ry], [rx, ry]])
        theta = np.deg2rad(inclusion.rotation_deg)
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        vertices = vertices @ rotation.T + np.array([cx, cy])
        patch = Polygon(vertices, facecolor="white", edgecolor="0.25", joinstyle="round")
    else:
        raise ValueError(f"Unsupported inclusion shape: {inclusion.shape}")
    ax.add_patch(patch)


def _draw_hole(ax: plt.Axes, lattice: LatticeSpec, center: tuple[float, float]) -> None:
    inclusions = lattice.inclusions or (
        LatticeInclusionSpec(
            shape=lattice.hole_shape,
            radius_x_nm=lattice.hole_radius_x_nm,
            radius_y_nm=lattice.hole_radius_y_nm,
            rotation_deg=lattice.hole_rotation_deg,
        ),
    )
    for inclusion in inclusions:
        _draw_inclusion(ax, lattice, center, inclusion)


def plot_lattice(lattice: LatticeSpec, path: Path, cells: int = 9) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    background = Rectangle((-0.8, -0.8), cells+0.6, cells+0.6,
                           facecolor="#6baed6", edgecolor="none")
    ax.add_patch(background)
    for row in range(cells):
        for col in range(cells):
            x = float(col)
            y = float(row)
            if lattice.lattice_type == "triangular":
                x += 0.5*(row % 2)
                y *= np.sqrt(3)/2
            _draw_hole(ax, lattice, (x, y))
    ax.set_aspect("equal")
    ax.set_title(
        f"{lattice.lattice_type} lattice, {lattice.hole_shape} holes\n"
        f"a={lattice.constant_nm:.1f} nm, fill={lattice.fill_fraction:.3f}"
    )
    ax.set_xlabel("x/a")
    ax.set_ylabel("y/a")
    ax.autoscale_view()
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_k_space(lattice: LatticeSpec, path: Path, order: int = 3) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    ax = axes[0]
    if lattice.lattice_type == "square":
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.0, 1.0])
        b1, b2 = 2*np.pi*a1, 2*np.pi*a2
    else:
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.5, np.sqrt(3)/2])
        reciprocal = 2*np.pi*np.linalg.inv(np.column_stack((a1, a2))).T
        b1, b2 = reciprocal[:, 0], reciprocal[:, 1]
    points = []
    labels = []
    for m in range(-order, order+1):
        for n in range(-order, order+1):
            points.append(m*b1+n*b2)
            labels.append((m, n))
    points = np.asarray(points)
    ax.scatter(points[:, 0], points[:, 1], c=np.hypot(points[:, 0], points[:, 1]), cmap="viridis")
    for (x, y), label in zip(points, labels, strict=True):
        if abs(label[0])+abs(label[1]) <= 1:
            ax.text(x, y, str(label), fontsize=8)
    ax.set(title="reciprocal lattice / Bloch harmonics", xlabel=r"$k_x a$", ylabel=r"$k_y a$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    ax = axes[1]
    arrows = ((1, 0, "Rx"), (-1, 0, "Sx"), (0, 1, "Ry"), (0, -1, "Sy"))
    for u, v, label in arrows:
        ax.arrow(0, 0, u, v, width=0.025, length_includes_head=True)
        ax.text(1.12*u, 1.12*v, label, ha="center", va="center")
    ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), title="retained four-wave basis")
    ax.set_aspect("equal")
    ax.axis("off")
    if lattice.lattice_type != "square":
        ax.text(0, -1.28, "WARNING: triangular lattice needs a six-wave solver",
                ha="center", color="tab:red", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_layer_stack(layers: tuple[LayerSpec, ...], path: Path) -> None:
    thicknesses = np.array([max(layer.thickness_nm, 1.0) for layer in layers])
    visual = np.log10(thicknesses+10.0)
    visual /= visual.sum()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    y = 0.0
    for layer, height in zip(layers, visual, strict=True):
        ax.add_patch(Rectangle((0.1, y), 0.8, height*0.92, fc=layer.color, ec="0.25"))
        ax.text(0.5, y+height*0.46,
                f"{layer.name}: {layer.thickness_nm:g} nm, n={layer.refractive_index:.4g}",
                ha="center", va="center", fontsize=8)
        y += height
    ax.set(xlim=(0, 1), ylim=(0, 1), title="vertical refractive-index stack")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_vertical_mode(vertical_mode, pc_layer_name: str, path: Path) -> None:
    fig, ax_index = plt.subplots(figsize=(8.0, 4.8))
    ax_field = ax_index.twinx()
    ax_index.plot(vertical_mode.z_um, vertical_mode.index, color="0.25", label="n(z)")
    normalized = np.abs(vertical_mode.field)/max(
        float(np.max(np.abs(vertical_mode.field))), np.finfo(float).eps
    )
    ax_field.plot(vertical_mode.z_um, normalized, color="tab:red", label="|Theta0|")
    ax_index.set(xlabel="z (um)", ylabel="refractive index", title="solved TE0 vertical mode")
    ax_field.set_ylabel("normalized field amplitude")
    ax_index.grid(alpha=0.2)
    lines = ax_index.lines+ax_field.lines
    ax_index.legend(lines, [line.get_label() for line in lines], loc="upper right")
    ax_index.text(
        0.02, 0.04,
        f"neff={vertical_mode.effective_index:.6f}, "
        f"Gamma_PC={vertical_mode.confinement[pc_layer_name]:.5f}",
        transform=ax_index.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_mode_atlas(lattice: LatticeSpec, spec: FourWaveOpticalSpec,
                    modes: dict[str, LinearMode], path: Path,
                    unit_cell_builder=None) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.0))
    for row, name in enumerate(MODE_NAMES):
        mode = modes[name]
        axes[row, 0].imshow(mode.intensity, origin="lower", cmap="turbo",
                            extent=(0, spec.domain_um, 0, spec.domain_um))
        axes[row, 0].set(title=f"{name}: whole-device envelope", xlabel="x (um)", ylabel="y (um)")
        center = mode.fields.shape[-1]//2
        coefficients = mode.fields[:, center, center]
        if unit_cell_builder is None:
            axis, cell_intensity, phase = unit_cell_field_from_coefficients(coefficients)
        else:
            axis, cell_intensity, phase = unit_cell_builder(coefficients)
        axes[row, 1].imshow(cell_intensity, origin="lower", cmap="turbo",
                            extent=(-0.5, 0.5, -0.5, 0.5))
        _draw_hole(axes[row, 1], lattice, (0.0, 0.0))
        axes[row, 1].set(title=f"{name}: one-cell intensity", xlabel="x/a", ylabel="y/a")
        if mode.radiation_x is not None and mode.radiation_y is not None:
            angle, ffp, _, _, divergence = vector_far_field(
                mode.radiation_x, mode.radiation_y,
                spec.wavelength_nm, spec.domain_um, view_deg=1.2,
            )
            far_field_label = "3-D CWT FFP"
        else:
            coherent = radiation_proxy_field(mode)
            angle, ffp, divergence = far_field(
                coherent, spec.wavelength_nm, spec.domain_um, view_deg=1.2
            )
            far_field_label = "FFP proxy"
        axes[row, 2].imshow(ffp, origin="lower", cmap="hot",
                            norm=PowerNorm(gamma=0.65, vmin=0, vmax=1),
                            extent=(angle[0], angle[-1], angle[0], angle[-1]))
        axes[row, 2].set(title=f"{name}: {far_field_label}, full RMS={divergence:.2f} deg",
                         xlabel="theta_x (deg)", ylabel="theta_y (deg)")
    fig.suptitle("A/B/C/D finite-area modes: envelope, unit cell and far field")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_vector_far_field_diagnostics(
    spec: FourWaveOpticalSpec, mode: LinearMode, path: Path
) -> None:
    """Plot total/vector FFP, circular-component phase, and polarization axes."""
    if mode.radiation_x is None or mode.radiation_y is None:
        raise ValueError("A geometry-derived radiation field is required")
    angle, fx, fy = vector_far_field_complex(
        mode.radiation_x, mode.radiation_y,
        spec.wavelength_nm, spec.domain_um,
    )
    ix, iy = np.abs(fx)**2, np.abs(fy)**2
    total = ix+iy
    scale = max(float(total.max()), np.finfo(float).eps)
    ix, iy, total = ix/scale, iy/scale, total/scale
    circular = fx+1j*fy
    phase = np.angle(circular)
    s1 = ix-iy
    s2 = 2.0*np.real(fx*np.conj(fy))/scale
    orientation = 0.5*np.arctan2(s2, s1)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0))
    extent = (angle[0], angle[-1], angle[0], angle[-1])
    for ax, values, title in (
        (axes[0, 0], total, "total |Fx|^2+|Fy|^2"),
        (axes[0, 1], ix, "x-polarized |Fx|^2"),
        (axes[1, 0], iy, "y-polarized |Fy|^2"),
    ):
        ax.imshow(values, origin="lower", extent=extent, cmap="hot",
                  norm=PowerNorm(gamma=0.65, vmin=0, vmax=1))
        ax.set_title(title)
    axes[1, 1].imshow(phase, origin="lower", extent=extent, cmap="twilight",
                      vmin=-np.pi, vmax=np.pi)
    stride = max(1, len(angle)//13)
    xgrid, ygrid = np.meshgrid(angle, angle)
    amplitude = np.sqrt(total)
    axes[1, 1].quiver(
        xgrid[::stride, ::stride], ygrid[::stride, ::stride],
        amplitude[::stride, ::stride]*np.cos(orientation[::stride, ::stride]),
        amplitude[::stride, ::stride]*np.sin(orientation[::stride, ::stride]),
        color="white", pivot="middle", scale=10.0, width=0.004,
    )
    axes[1, 1].set_title("phase of Fx+iFy and polarization axes")
    for ax in axes.flat:
        ax.set(xlabel="theta_x (deg)", ylabel="theta_y (deg)")
    center = len(angle)//2
    center_ratio = float(total[center, center])
    fig.suptitle(
        f"Mode {mode.name} vector far field; center/max={center_ratio:.2e}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_threshold_summary(spec: FourWaveOpticalSpec, modes: dict[str, LinearMode], path: Path) -> None:
    names = list(MODE_NAMES)
    delta_l = [modes[name].delta_per_m*spec.domain_um*1e-6 for name in names]
    alpha_l = [modes[name].alpha_per_m*spec.domain_um*1e-6 for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].bar(names, delta_l, color="#3182bd")
    axes[0].set(ylabel="delta L", title="normalized mode detuning")
    axes[1].bar(names, alpha_l, color="#e6550d")
    axes[1].set(ylabel="alpha L", title="normalized finite-area threshold")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_length_sweep(spec: FourWaveOpticalSpec, path: Path) -> None:
    lengths = np.linspace(max(20.0, 0.15*spec.domain_um), 1.5*spec.domain_um, 90)
    radiation = np.asarray(spec.modal_radiation_loss_cm)*100.0
    reference_l = spec.domain_um*1e-6
    edge_reference = np.array([0.35, 0.50, 1.1, 1.1])/reference_l
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    for index, name in enumerate(MODE_NAMES):
        length_m = lengths*1e-6
        alpha = radiation[index] + edge_reference[index]*(spec.domain_um/lengths)**1.7
        ax.plot(lengths, alpha*length_m, label=name)
    ax.set(xlabel="device length L (um)", ylabel="normalized threshold alpha L",
           title="finite-size threshold versus device length")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_length_sweep_from_solver(
    spec: FourWaveOpticalSpec,
    coupling: np.ndarray,
    band_basis: np.ndarray,
    radiation_builder,
    path: Path,
    lengths_um: np.ndarray | None = None,
    grid_points: int | tuple[int, ...] = 13,
) -> dict[str, np.ndarray]:
    """Re-solve Eq. (4.21) at every length, optionally extrapolating N -> infinity."""
    if lengths_um is None:
        lengths_um = np.linspace(40.0, 400.0, 10)
    values = {name: [] for name in MODE_NAMES}
    for length_um in lengths_um:
        if isinstance(grid_points, tuple):
            local_spec = replace(
                spec, domain_um=float(length_um), grid_points=grid_points[-1]
            )
            modes, _ = solve_finite_modes_converged(
                local_spec, grid_points, coupling, band_basis, radiation_builder
            )
        else:
            local_spec = replace(spec, domain_um=float(length_um), grid_points=grid_points)
            modes = solve_finite_modes(
                local_spec, coupling, band_basis, radiation_builder
            )
        for name in MODE_NAMES:
            values[name].append(modes[name].alpha_per_m*length_um*1e-6)
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    for name in MODE_NAMES:
        ax.plot(lengths_um, values[name], "o-", label=name)
    grid_label = f"N={grid_points}" if isinstance(grid_points, int) else f"N={grid_points}, extrapolated"
    ax.set(xlabel="device length L (um)", ylabel="normalized threshold alpha L",
           title=f"Eq. (4.21) length sweep, {grid_label}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return {"length_um": np.asarray(lengths_um), **{
        name: np.asarray(series) for name, series in values.items()
    }}


def write_parameter_report(output: Path, lattice: LatticeSpec, spec: FourWaveOpticalSpec,
                           layers: tuple[LayerSpec, ...], extra: dict[str, object]) -> Path:
    report = {
        "lattice": asdict(lattice),
        "derived_fill_fraction": lattice.fill_fraction,
        "optical": asdict(spec),
        "layers": [asdict(layer) for layer in layers],
        "extra": extra,
        "model_scope": "square-lattice four-wave coupled-wave theory",
    }
    path = output/"parameters.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_mode_table(output: Path, spec: FourWaveOpticalSpec,
                     modes: dict[str, LinearMode]) -> Path:
    path = output/"mode_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "delta_per_m", "alpha_per_m", "delta_L", "alpha_L",
                         "band_overlap"))
        for name in MODE_NAMES:
            mode = modes[name]
            writer.writerow((name, mode.delta_per_m, mode.alpha_per_m,
                             mode.delta_per_m*spec.domain_um*1e-6,
                             mode.alpha_per_m*spec.domain_um*1e-6,
                             mode.band_overlap))
    return path
