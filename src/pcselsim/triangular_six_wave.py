"""Geometry-derived six-wave 3-D CWT for triangular-lattice semiconductor PCSELs.

This module follows Liang thesis Sec. 5.2 and Appendix B (equivalently Liang
et al., Optics Express 21, 565-580, 2013).  The retained TE basic waves are

``(R1,S1,R2,S2,R3,S3) = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1))``.

The coupling matrix is derived as ``C = Cb + Cr + Ch``: direct basic-wave
coupling, radiative coupling through the zero-order wave, and indirect coupling
through evanescent high-order waves.  It is deliberately separate from the
square-lattice four-wave implementation; no 4x4 matrix is padded or fitted.

中文：本模块按 Liang 论文第 5.2 节及附录 B 从头构造三角晶格六波模型。
六个基本波的传播方向、TE 偏振、直接耦合、零级辐射耦合和高阶波间接耦合均由
晶格和纵向 TE0 模计算。它不是把现有四波矩阵扩展成 6x6 的经验近似。

Scope / 适用范围：本模块覆盖无限周期结构的 Gamma 带边态、M-Gamma-X
局域能带、辐射常数和晶胞场。有限圆形器件、整体包络和远场由配套的
``triangular_finite.py`` 实现；六波半导体载流子时域方程尚未实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse as EllipsePatch, Polygon
from scipy.special import j1

from .geometry import polygon_fourier_integral
from .band_structure import (
    BandDiagram,
    detuning_to_normalized_frequency,
    track_complex_bands,
)
from .constants import pi
from .three_d_cwt import (
    _double_green_integral,
    _pc_layer_coordinates,
    _project_radiation_passive,
    _surface_green_integral,
    _trapezoid_weights,
)
from .vertical import LayerStack, VerticalMode


TRIANGULAR_BASIC_ORDERS: tuple[tuple[int, int], ...] = (
    (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)
)
TRIANGULAR_MODE_NAMES = ("A", "B1", "B2", "C", "D1", "D2")


def reciprocal_components(order: tuple[int, int]) -> tuple[float, float]:
    """Return ``(m0x,n0y)`` in units of ``beta0=4*pi/(sqrt(3)*a)``.

    Liang Eq. (5.2): ``m0x=(m+n)/2`` and ``n0y=sqrt(3)(n-m)/2``.
    """
    m, n = order
    return 0.5 * (m + n), 0.5 * np.sqrt(3.0) * (n - m)


def reciprocal_metric_squared(m: int, n: int) -> int:
    """Return ``|G_mn|^2/beta0^2 = m^2-m*n+n^2``."""
    return int(m * m - m * n + n * n)


def te_polarization(order: tuple[int, int]) -> tuple[float, float]:
    """Return the in-plane TE vector ``(rho,eta)=(ny,-mx)/|G|``."""
    mx, ny = reciprocal_components(order)
    norm = float(np.hypot(mx, ny))
    if norm == 0.0:
        raise ValueError("The radiative (0,0) order has no in-plane TE direction")
    return ny / norm, -mx / norm


TRIANGULAR_DIRECTIONS = np.asarray(
    [reciprocal_components(order) for order in TRIANGULAR_BASIC_ORDERS], dtype=float
)
TRIANGULAR_POLARIZATIONS = np.asarray(
    [te_polarization(order) for order in TRIANGULAR_BASIC_ORDERS], dtype=float
)


@dataclass(frozen=True)
class TriangularEllipse:
    """Ellipse in a triangular primitive cell.

    ``center_fractional`` is expressed in the direct basis ``u*a1+v*a2``.
    ``radii_over_a`` and ``angle_deg`` describe the physical Cartesian ellipse.
    """

    center_fractional: tuple[float, float] = (0.0, 0.0)
    radii_over_a: tuple[float, float] = (0.2, 0.2)
    angle_deg: float = 0.0
    epsilon: float = 1.0

    @property
    def fill_fraction(self) -> float:
        primitive_area_over_a2 = np.sqrt(3.0) / 2.0
        return float(np.pi * self.radii_over_a[0] * self.radii_over_a[1]
                     / primitive_area_over_a2)


@dataclass(frozen=True)
class TriangularPolygon:
    """Polygon whose vertices are expressed in the ``(a1,a2)`` basis.

    Use :meth:`regular` to specify a regular triangle or other polygon by a
    physical Cartesian circumradius and rotation.  The Fourier transform is
    evaluated analytically from its boundary, without rasterization.
    """

    vertices_fractional: tuple[tuple[float, float], ...]
    epsilon: float = 1.0

    def __post_init__(self) -> None:
        if len(self.vertices_fractional) < 3:
            raise ValueError("a polygon inclusion needs at least three vertices")

    @property
    def fill_fraction(self) -> float:
        points = np.asarray(self.vertices_fractional, dtype=float)
        area = 0.5 * np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
        return float(abs(area))

    @classmethod
    def regular(
        cls,
        center_fractional: tuple[float, float],
        circumradius_over_a: float,
        sides: int = 3,
        angle_deg: float = 0.0,
        epsilon: float = 1.0,
    ) -> "TriangularPolygon":
        """Construct a regular polygon using physical Cartesian geometry."""
        if sides < 3:
            raise ValueError("sides must be at least three")
        if circumradius_over_a <= 0.0:
            raise ValueError("circumradius_over_a must be positive")
        direct = np.asarray(
            ((np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0), (-0.5, 0.5))
        )
        center_cartesian = direct @ np.asarray(center_fractional, dtype=float)
        angles = np.deg2rad(angle_deg) + 2.0 * np.pi * np.arange(sides) / sides
        vertices_cartesian = center_cartesian[:, None] + circumradius_over_a * np.vstack(
            (np.cos(angles), np.sin(angles))
        )
        vertices_fractional = np.linalg.solve(direct, vertices_cartesian).T
        return cls(
            vertices_fractional=tuple(
                (float(vertex[0]), float(vertex[1]))
                for vertex in vertices_fractional
            ),
            epsilon=epsilon,
        )

    @classmethod
    def rounded_regular(
        cls,
        center_fractional: tuple[float, float],
        circumradius_over_a: float,
        sides: int = 3,
        angle_deg: float = 0.0,
        corner_fraction: float = 0.10,
        samples_per_corner: int = 5,
        epsilon: float = 1.0,
    ) -> "TriangularPolygon":
        """Construct a convex polygon with fabrication-like rounded corners.

        Each ideal vertex is replaced by a quadratic Bezier arc.  The
        dimensionless ``corner_fraction`` gives the setback along each adjacent
        edge (zero recovers :meth:`regular`).  The returned polygon uses the
        same analytic Fourier boundary integral as every other inclusion.

        中文：用二次 Bezier 圆滑每个尖角；``corner_fraction`` 是沿相邻边的
        退让比例。返回多边形近似，仍直接进入解析 Fourier 边界积分。
        """
        if corner_fraction < 0.0 or corner_fraction >= 0.5:
            raise ValueError("corner_fraction must satisfy 0 <= value < 0.5")
        if samples_per_corner < 2:
            raise ValueError("samples_per_corner must be at least two")
        if np.isclose(corner_fraction, 0.0):
            return cls.regular(
                center_fractional,
                circumradius_over_a,
                sides,
                angle_deg,
                epsilon,
            )
        if sides < 3:
            raise ValueError("sides must be at least three")
        if circumradius_over_a <= 0.0:
            raise ValueError("circumradius_over_a must be positive")

        direct = np.asarray(
            ((np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0), (-0.5, 0.5))
        )
        center_cartesian = direct @ np.asarray(center_fractional, dtype=float)
        angles = np.deg2rad(angle_deg) + 2.0 * np.pi * np.arange(sides) / sides
        ideal = center_cartesian + circumradius_over_a * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        rounded: list[np.ndarray] = []
        for index, vertex in enumerate(ideal):
            previous = ideal[(index - 1) % sides]
            following = ideal[(index + 1) % sides]
            incoming = vertex + corner_fraction * (previous - vertex)
            outgoing = vertex + corner_fraction * (following - vertex)
            for parameter in np.linspace(
                0.0, 1.0, samples_per_corner, endpoint=False
            ):
                point = (
                    (1.0 - parameter) ** 2 * incoming
                    + 2.0 * (1.0 - parameter) * parameter * vertex
                    + parameter**2 * outgoing
                )
                rounded.append(point)
        vertices_fractional = np.linalg.solve(direct, np.asarray(rounded).T).T
        return cls(
            vertices_fractional=tuple(
                (float(vertex[0]), float(vertex[1]))
                for vertex in vertices_fractional
            ),
            epsilon=epsilon,
        )


@dataclass(frozen=True)
class TriangularLatticeCell:
    """Triangular Bravais cell with analytic elliptical Fourier coefficients."""

    background_epsilon: float
    inclusions: tuple[TriangularEllipse | TriangularPolygon, ...]

    @property
    def fill_fraction(self) -> float:
        return float(sum(item.fill_fraction for item in self.inclusions))

    def fourier_epsilon(self, m: int, n: int) -> complex:
        """Return ``xi_mn`` using the sign convention of Liang Eq. (5.6)."""
        value = self.background_epsilon if (m, n) == (0, 0) else 0.0j
        beta0_a = 4.0 * np.pi / np.sqrt(3.0)
        direction = np.asarray(reciprocal_components((m, n)), dtype=float)
        wave_a = beta0_a * direction
        for inclusion in self.inclusions:
            if isinstance(inclusion, TriangularEllipse):
                angle = np.deg2rad(inclusion.angle_deg)
                rotation = np.asarray(
                    ((np.cos(angle), -np.sin(angle)),
                     (np.sin(angle), np.cos(angle)))
                )
                local_wave_a = rotation.T @ wave_a
                argument = float(np.hypot(
                    inclusion.radii_over_a[0] * local_wave_a[0],
                    inclusion.radii_over_a[1] * local_wave_a[1],
                ))
                form = 1.0 if argument == 0.0 else 2.0 * j1(argument) / argument
                phase = np.exp(2j * np.pi * (
                    m * inclusion.center_fractional[0]
                    + n * inclusion.center_fractional[1]
                ))
                transform = inclusion.fill_fraction * form * phase
            else:
                transform = polygon_fourier_integral(
                    inclusion.vertices_fractional, m, n
                )
            value += (inclusion.epsilon - self.background_epsilon) * transform
        return complex(value)


@dataclass(frozen=True)
class TriangularCWTSettings:
    """Accuracy controls for the triangular high-order and vertical sums.

    High orders are retained in the C6-symmetric reciprocal shell
    ``m^2-m*n+n^2 <= truncation_order^2``.  A rectangular index cutoff would
    spuriously split the B1/B2 and D1/D2 doublets of a circular unit cell.
    """

    truncation_order: int = 10
    vertical_step_nm: float = 2.0
    bragg_tolerance_nm: float = 1e-4
    bragg_max_iterations: int = 30
    bragg_relaxation: float = 0.65


@dataclass
class TriangularCouplingResult:
    """Geometry-derived six-wave matrices and reconstructed field responses."""

    coupling_m: np.ndarray
    cb_m: np.ndarray
    cr_m: np.ndarray
    ch_m: np.ndarray
    eigenvalues_m: np.ndarray
    eigenvectors: np.ndarray
    bragg_wavelength_nm: float
    effective_index: float
    group_index: float
    pc_confinement: float
    vertical_mode: VerticalMode
    lattice_constant_nm: float
    average_pc_epsilon: float
    passivity_correction_m: float
    fourier: Callable[[int, int], complex]
    unit_cell_response_x: dict[tuple[int, int], np.ndarray]
    unit_cell_response_y: dict[tuple[int, int], np.ndarray]
    radiation_surface_factor: complex

    def radiation_amplitudes(self, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the two normal-radiation polarizations from Liang Eq. (B1)."""
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if coefficients.shape[0] != 6:
            raise ValueError("The leading coefficient dimension must contain six waves")
        weights_x = np.asarray([
            self.fourier(-m, -n) * rho
            for (m, n), (rho, _) in zip(
                TRIANGULAR_BASIC_ORDERS, TRIANGULAR_POLARIZATIONS, strict=True
            )
        ])
        weights_y = np.asarray([
            self.fourier(-m, -n) * eta
            for (m, n), (_, eta) in zip(
                TRIANGULAR_BASIC_ORDERS, TRIANGULAR_POLARIZATIONS, strict=True
            )
        ])
        return (
            self.radiation_surface_factor * np.einsum("a,a...->...", weights_x, coefficients),
            self.radiation_surface_factor * np.einsum("a,a...->...", weights_y, coefficients),
        )

    def unit_cell_fields(
        self, coefficients: np.ndarray, points: int = 121
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct Ex/Ey at the PC center from basic, radiative and high orders."""
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if coefficients.shape != (6,):
            raise ValueError("coefficients must have shape (6,)")
        axis = np.linspace(-0.5, 0.5, int(points))
        u, v = np.meshgrid(axis, axis)
        x = 0.5 * np.sqrt(3.0) * (u + v)
        y = 0.5 * (v - u)
        ex = np.zeros_like(u, dtype=np.complex128)
        ey = np.zeros_like(u, dtype=np.complex128)
        orders = set(self.unit_cell_response_x) | set(self.unit_cell_response_y)
        for m, n in orders:
            phase = np.exp(-2j * np.pi * (m * u + n * v))
            if (m, n) in self.unit_cell_response_x:
                ex += (self.unit_cell_response_x[(m, n)] @ coefficients) * phase
            if (m, n) in self.unit_cell_response_y:
                ey += (self.unit_cell_response_y[(m, n)] @ coefficients) * phase
        intensity = np.abs(ex) ** 2 + np.abs(ey) ** 2
        intensity /= max(float(intensity.max()), np.finfo(float).eps)
        return x, y, intensity, ex, ey


def solve_triangular_bragg_vertical_mode(
    stack: LayerStack,
    lattice_constant_nm: float,
    wavelength_guess_nm: float,
    settings: TriangularCWTSettings,
) -> tuple[float, VerticalMode]:
    """Solve ``beta=4*pi/(sqrt(3)*a)`` or ``lambda=sqrt(3)*a*n_eff/2``."""
    wavelength_nm = float(wavelength_guess_nm)
    factor = np.sqrt(3.0) / 2.0
    for _ in range(settings.bragg_max_iterations):
        mode = stack.solve_te0(wavelength_nm, dz_nm=settings.vertical_step_nm)
        target_nm = factor * lattice_constant_nm * mode.effective_index
        updated_nm = (
            settings.bragg_relaxation * target_nm
            + (1.0 - settings.bragg_relaxation) * wavelength_nm
        )
        if abs(updated_nm - wavelength_nm) <= settings.bragg_tolerance_nm:
            wavelength_nm = updated_nm
            return wavelength_nm, stack.solve_te0(
                wavelength_nm, dz_nm=settings.vertical_step_nm
            )
        wavelength_nm = updated_nm
    raise RuntimeError("Triangular Bragg wavelength iteration did not converge")


def _validate_six_wave_passivity(matrix: np.ndarray, atol: float) -> None:
    if matrix.shape != (6, 6):
        raise ValueError("The triangular six-wave coupling matrix must be 6x6")
    radiation = (matrix - matrix.conj().T) / (2j)
    if float(np.min(np.linalg.eigvalsh(radiation))) < -atol:
        raise ValueError("The six-wave radiation matrix is not passive")


def build_triangular_coupling(
    cell: TriangularLatticeCell,
    stack: LayerStack,
    pc_layer_name: str,
    lattice_constant_nm: float,
    wavelength_guess_nm: float,
    settings: TriangularCWTSettings = TriangularCWTSettings(),
) -> TriangularCouplingResult:
    """Build Liang Appendix-B ``C=Cb+Cr+Ch`` from geometry and the TE0 mode."""
    if settings.truncation_order < 2:
        raise ValueError("truncation_order must be >= 2")
    average_epsilon = float(np.real(cell.fourier_epsilon(0, 0)))
    if average_epsilon <= 0.0:
        raise ValueError("The triangular-cell average epsilon must be positive")

    wavelength_nm, vertical_mode = solve_triangular_bragg_vertical_mode(
        stack, lattice_constant_nm, wavelength_guess_nm, settings
    )
    z_m, theta, weights_m = _pc_layer_coordinates(stack, vertical_mode, pc_layer_name)
    pc_confinement = float(np.sum(np.abs(theta) ** 2 * weights_m))
    inverse_epsilon_integral = pc_confinement / average_epsilon
    lattice_m = lattice_constant_nm * 1e-9
    k0 = 2.0 * pi / (wavelength_nm * 1e-9)
    beta0 = 4.0 * pi / (np.sqrt(3.0) * lattice_m)
    pc_index = float(np.sqrt(average_epsilon))
    all_z_m = vertical_mode.z_um * 1e-6
    all_weights_m = _trapezoid_weights(all_z_m)
    group_index = float(
        np.sum(vertical_mode.index ** 2 * np.abs(vertical_mode.field) ** 2 * all_weights_m)
        / vertical_mode.effective_index
    )

    cache: dict[tuple[int, int], complex] = {}

    def xi(m: int, n: int) -> complex:
        if (m, n) == (0, 0):
            return 0.0j
        key = (int(m), int(n))
        if key not in cache:
            cache[key] = cell.fourier_epsilon(*key)
        return cache[key]

    polar = TRIANGULAR_POLARIZATIONS
    cb = np.zeros((6, 6), dtype=np.complex128)
    prefactor_basic = -k0 ** 2 * pc_confinement / (2.0 * beta0)
    for row, (p, q) in enumerate(TRIANGULAR_BASIC_ORDERS):
        for column, (r, s) in enumerate(TRIANGULAR_BASIC_ORDERS):
            if row == column:
                continue
            polarization_overlap = float(np.dot(polar[row], polar[column]))
            cb[row, column] = (
                prefactor_basic * xi(p - r, q - s) * polarization_overlap
            )
    cb = 0.5 * (cb + cb.conj().T)

    beta_radiative = complex(k0 * pc_index)
    radiation_integral = _double_green_integral(
        z_m, theta, weights_m, beta_radiative, radiative=True
    )
    surface_integral = _surface_green_integral(
        z_m, theta, weights_m, beta_radiative
    )
    cr = np.zeros((6, 6), dtype=np.complex128)
    prefactor_radiative = -k0 ** 4 / (2.0 * beta0)
    for row, (p, q) in enumerate(TRIANGULAR_BASIC_ORDERS):
        for column, (r, s) in enumerate(TRIANGULAR_BASIC_ORDERS):
            cr[row, column] = (
                prefactor_radiative
                * xi(p, q)
                * xi(-r, -s)
                * float(np.dot(polar[row], polar[column]))
                * radiation_integral
            )
    cr, passivity_correction = _project_radiation_passive(cr)

    probe = z_m.size // 2
    z_probe = float(z_m[probe])
    theta_probe = complex(theta[probe])
    radiative_kernel = (
        -1j / (2.0 * beta_radiative)
        * np.exp(-1j * beta_radiative * np.abs(z_probe - z_m))
    )
    radiative_probe = complex(np.sum(radiative_kernel * theta * weights_m))
    response_x: dict[tuple[int, int], np.ndarray] = {}
    response_y: dict[tuple[int, int], np.ndarray] = {}
    for column, order in enumerate(TRIANGULAR_BASIC_ORDERS):
        vector_x = np.zeros(6, dtype=np.complex128)
        vector_y = np.zeros(6, dtype=np.complex128)
        vector_x[column] = polar[column, 0] * theta_probe
        vector_y[column] = polar[column, 1] * theta_probe
        response_x[order] = vector_x
        response_y[order] = vector_y
    response_x[(0, 0)] = k0 ** 2 * radiative_probe * np.asarray([
        xi(-m, -n) * polar[index, 0]
        for index, (m, n) in enumerate(TRIANGULAR_BASIC_ORDERS)
    ])
    response_y[(0, 0)] = k0 ** 2 * radiative_probe * np.asarray([
        xi(-m, -n) * polar[index, 1]
        for index, (m, n) in enumerate(TRIANGULAR_BASIC_ORDERS)
    ])

    ch = np.zeros((6, 6), dtype=np.complex128)
    prefactor_high = -k0 ** 2 / (2.0 * beta0)
    order_limit = settings.truncation_order
    # The metric circle reaches |m| or |n| = 2D/sqrt(3), not merely D.
    # Covering that full bounding box is required to preserve all C6 rotations.
    index_limit = int(np.ceil(2.0 * order_limit / np.sqrt(3.0)))
    for m in range(-index_limit, index_limit + 1):
        for n in range(-index_limit, index_limit + 1):
            metric_squared = reciprocal_metric_squared(m, n)
            if metric_squared <= 1 or metric_squared > order_limit ** 2:
                continue
            mx, ny = reciprocal_components((m, n))
            beta_high = complex(np.sqrt(
                metric_squared * beta0 ** 2 - k0 ** 2 * average_epsilon + 0j
            ))
            if beta_high.real <= 0.0:
                raise ValueError(
                    f"High-order triangular wave ({m},{n}) is not evanescent"
                )
            green_integral = _double_green_integral(
                z_m, theta, weights_m, beta_high, radiative=False
            )
            probe_kernel = (
                1.0 / (2.0 * beta_high)
                * np.exp(-beta_high * np.abs(z_probe - z_m))
            )
            probe_green = complex(np.sum(probe_kernel * theta * weights_m))

            zeta_x = np.empty(6, dtype=np.complex128)
            zeta_y = np.empty(6, dtype=np.complex128)
            zeta_x_probe = np.empty(6, dtype=np.complex128)
            zeta_y_probe = np.empty(6, dtype=np.complex128)
            for column, (r, s) in enumerate(TRIANGULAR_BASIC_ORDERS):
                coefficient = xi(m - r, n - s)
                rho, eta = polar[column]
                transverse = ny * rho - mx * eta
                longitudinal = mx * rho + ny * eta
                mu = k0 ** 2 * coefficient * transverse * green_integral
                nu = -coefficient * longitudinal * inverse_epsilon_integral
                mu_probe = k0 ** 2 * coefficient * transverse * probe_green
                nu_probe = (
                    -coefficient * longitudinal * theta_probe / average_epsilon
                )
                zeta_x[column] = (ny * mu + mx * nu) / metric_squared
                zeta_y[column] = (-mx * mu + ny * nu) / metric_squared
                zeta_x_probe[column] = (
                    ny * mu_probe + mx * nu_probe
                ) / metric_squared
                zeta_y_probe[column] = (
                    -mx * mu_probe + ny * nu_probe
                ) / metric_squared
            response_x[(m, n)] = zeta_x_probe
            response_y[(m, n)] = zeta_y_probe
            for row, (p, q) in enumerate(TRIANGULAR_BASIC_ORDERS):
                ch[row] += (
                    prefactor_high
                    * xi(p - m, q - n)
                    * (polar[row, 0] * zeta_x + polar[row, 1] * zeta_y)
                )
    ch = 0.5 * (ch + ch.conj().T)

    coupling = cb + cr + ch
    _validate_six_wave_passivity(
        coupling, atol=max(1e-8, 1e-10 * float(np.linalg.norm(coupling)))
    )
    eigenvalues, eigenvectors = np.linalg.eig(coupling)
    ordering = np.lexsort((eigenvalues.imag, eigenvalues.real))
    eigenvalues = eigenvalues[ordering]
    eigenvectors = eigenvectors[:, ordering]
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0, keepdims=True)

    return TriangularCouplingResult(
        coupling_m=coupling,
        cb_m=cb,
        cr_m=cr,
        ch_m=ch,
        eigenvalues_m=eigenvalues,
        eigenvectors=eigenvectors,
        bragg_wavelength_nm=wavelength_nm,
        effective_index=vertical_mode.effective_index,
        group_index=group_index,
        pc_confinement=pc_confinement,
        vertical_mode=vertical_mode,
        lattice_constant_nm=lattice_constant_nm,
        average_pc_epsilon=average_epsilon,
        passivity_correction_m=passivity_correction,
        fourier=xi,
        unit_cell_response_x=response_x,
        unit_cell_response_y=response_y,
        radiation_surface_factor=k0 ** 2 * surface_integral,
    )


def triangular_band_diagram(
    result: TriangularCouplingResult,
    q_max: float = 0.05,
    points: int = 241,
) -> BandDiagram:
    """Return the local ``M <- Gamma -> X`` six-wave band diagram.

    The two physical directions are Cartesian 0 degrees and 30 degrees.  Liang
    Fig. 5.4 calls them Gamma-X and Gamma-J; the output uses the user-requested
    M/Gamma/X naming for the same two triangular-lattice symmetry lines.
    Couplings are frozen at Gamma while the exact radial mismatch from Liang
    Eq. (B12) is retained.
    """
    q = np.linspace(-float(q_max), float(q_max), int(points))
    lattice_m = result.lattice_constant_nm * 1e-9
    beta0 = 4.0 * np.pi / (np.sqrt(3.0) * lattice_m)

    def matrix(value: float) -> np.ndarray:
        magnitude = abs(value) * 2.0 * np.pi / lattice_m
        if value < 0.0:  # Gamma-M in the requested naming (Liang: Gamma-X)
            kx, ky = magnitude, 0.0
        else:  # Gamma-X in the requested naming (Liang: Gamma-J), 30 degrees
            kx, ky = magnitude * np.cos(np.pi / 6.0), magnitude * np.sin(np.pi / 6.0)
        delta_x, delta_y = kx / beta0, ky / beta0
        mismatch = []
        for mx0, ny0 in TRIANGULAR_DIRECTIONS:
            mismatch.append(
                (np.hypot(mx0 + delta_x, ny0 + delta_y) - 1.0) * beta0
            )
        return result.coupling_m + np.diag(mismatch)

    eigenvalues, _ = track_complex_bands(matrix, q)
    frequency = detuning_to_normalized_frequency(
        eigenvalues,
        result.lattice_constant_nm,
        result.bragg_wavelength_nm,
        result.effective_index,
    )
    return BandDiagram(
        wave_number_2pi_over_a=q,
        normalized_frequency=frequency,
        radiation_constant_cm=2.0 * np.maximum(eigenvalues.imag, 0.0) / 100.0,
        labels=TRIANGULAR_MODE_NAMES,
        left_endpoint="M",
        right_endpoint="X",
        approximation=(
            "triangular six-wave 3-D CWT; M-Gamma-X display path; "
            "Cb+Cr+Ch frozen at Gamma; "
            "Liang Eq. B12 wavevector mismatch"
        ),
    )


def plot_triangular_lattice(
    cell: TriangularLatticeCell, path: Path, cells: int = 7
) -> None:
    """Plot the real-space triangular lattice and its primitive cell."""
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    a1 = np.asarray((np.sqrt(3.0) / 2.0, -0.5))
    a2 = np.asarray((np.sqrt(3.0) / 2.0, 0.5))
    half = cells // 2
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            origin = i * a1 + j * a2
            for inclusion in cell.inclusions:
                if isinstance(inclusion, TriangularEllipse):
                    center = origin + inclusion.center_fractional[0] * a1 \
                        + inclusion.center_fractional[1] * a2
                    rx, ry = inclusion.radii_over_a
                    if np.isclose(rx, ry):
                        patch = Circle(
                            center, rx, facecolor="#f7f7f7", edgecolor="black"
                        )
                    else:
                        patch = EllipsePatch(
                            center,
                            2.0 * rx,
                            2.0 * ry,
                            angle=inclusion.angle_deg,
                            facecolor="#f7f7f7",
                            edgecolor="black",
                        )
                else:
                    vertices = np.asarray([
                        origin + vertex[0] * a1 + vertex[1] * a2
                        for vertex in inclusion.vertices_fractional
                    ])
                    patch = Polygon(
                        vertices,
                        closed=True,
                        facecolor="#f7f7f7",
                        edgecolor="black",
                    )
                ax.add_patch(patch)
    primitive = np.asarray(((-0.5 * (a1 + a2)),
                            (0.5 * (a1 - a2)),
                            (0.5 * (a1 + a2)),
                            (0.5 * (-a1 + a2))))
    ax.add_patch(Polygon(primitive, closed=True, fill=False, edgecolor="#d62728", lw=2.0))
    ax.arrow(0.0, 0.0, *a1, color="#1f77b4", width=0.008, length_includes_head=True)
    ax.arrow(0.0, 0.0, *a2, color="#1f77b4", width=0.008, length_includes_head=True)
    ax.text(a1[0], a1[1], r"  $\mathbf{a}_1$")
    ax.text(a2[0], a2[1], r"  $\mathbf{a}_2$")
    ax.set(aspect="equal", xlabel="x/a", ylabel="y/a", title="Triangular-lattice unit cells")
    ax.autoscale_view()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_triangular_reciprocal_space(path: Path, order: int = 2) -> None:
    """Plot reciprocal points and the six retained second-order-Gamma waves."""
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    points = []
    for m in range(-order, order + 1):
        for n in range(-order, order + 1):
            points.append(reciprocal_components((m, n)))
    points_array = np.asarray(points)
    ax.scatter(points_array[:, 0], points_array[:, 1], s=18, color="0.15")
    colors = ("#d62728", "#d62728", "#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c")
    for name, direction, color in zip(
        ("R1", "S1", "R2", "S2", "R3", "S3"),
        TRIANGULAR_DIRECTIONS,
        colors,
        strict=True,
    ):
        ax.arrow(0.0, 0.0, *direction, color=color, width=0.018,
                 length_includes_head=True, zorder=3)
        ax.text(1.08 * direction[0], 1.08 * direction[1], name, color=color,
                ha="center", va="center")
    angles = np.linspace(0.0, 2.0 * np.pi, 7) + np.pi / 6.0
    bz = np.column_stack((np.cos(angles), np.sin(angles))) / np.sqrt(3.0)
    ax.plot(bz[:, 0], bz[:, 1], "--", color="0.55", lw=1.0, label="first BZ (scaled)")
    ax.set(
        aspect="equal", xlabel=r"$G_x/\beta_0$", ylabel=r"$G_y/\beta_0$",
        title="Six retained waves at the second-order Gamma point",
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_six_band_edge_states(
    result: TriangularCouplingResult, path: Path, points: int = 101
) -> None:
    """Plot the six geometry-derived Gamma eigenstates inside one primitive cell."""
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8), constrained_layout=True)
    for index, (name, ax) in enumerate(zip(TRIANGULAR_MODE_NAMES, axes.flat, strict=True)):
        coefficients = result.eigenvectors[:, index]
        x, y, intensity, ex, ey = result.unit_cell_fields(coefficients, points=points)
        phase = np.angle((ex + ey).flat[np.argmax(intensity)])
        ex_plot = np.real(ex * np.exp(-1j * phase))
        ey_plot = np.real(ey * np.exp(-1j * phase))
        mesh = ax.pcolormesh(x, y, intensity, shading="auto", cmap="magma", vmin=0.0, vmax=1.0)
        stride = max(1, points // 13)
        magnitude = np.hypot(ex_plot, ey_plot)
        scale = max(float(magnitude.max()), np.finfo(float).eps)
        ax.quiver(
            x[::stride, ::stride], y[::stride, ::stride],
            ex_plot[::stride, ::stride] / scale,
            ey_plot[::stride, ::stride] / scale,
            color="white", pivot="mid", scale=17.0, width=0.004,
        )
        radiation_cm = 2.0 * max(result.eigenvalues_m[index].imag, 0.0) / 100.0
        ax.set(
            aspect="equal", xlabel="x/a", ylabel="y/a",
            title=f"{name}: alpha_r={radiation_cm:.2f} cm$^{{-1}}$",
        )
    fig.colorbar(mesh, ax=axes, shrink=0.80, label="normalized |E|^2")
    fig.suptitle("Triangular six-wave Gamma states (basic + radiative + high orders)")
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_radiation_constants(diagram: BandDiagram, path: Path) -> None:
    """Plot the six branch radiation constants beside the band diagram."""
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for index, name in enumerate(diagram.labels):
        ax.plot(
            diagram.wave_number_2pi_over_a,
            diagram.radiation_constant_cm[index],
            lw=1.5,
            label=name,
        )
    ax.set(
        xlabel=r"Wavenumber ($2\pi/a$)",
        ylabel=r"Radiation constant $\alpha_r=2\,\mathrm{Im}(\delta)$ (cm$^{-1}$)",
        title="Triangular-lattice radiation constants near Gamma",
    )
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

