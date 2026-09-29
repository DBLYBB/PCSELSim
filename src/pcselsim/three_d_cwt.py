"""Geometry-derived three-dimensional coupled-wave coefficients for PCSELs.

This module implements Liang thesis Eqs. (3.14), (3.20), (3.27)-(3.35).
The four retained basic waves are ordered as (Rx, Sx, Ry, Sy).  In contrast
to the calibrated model, every matrix element is generated from the unit-cell
Fourier coefficients and the solved vertical TE0 field.

The Green functions are the interface-reflection-free approximations used in
Liang Eq. (3.19) and Eq. (3.23).  This is the paper's documented baseline 3-D
CWT, not the later generalized Green function for tilted walls/back reflectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .constants import pi
from .coupling import validate_passive_coupling
from .geometry import SquareLatticeCell
from .vertical import LayerStack, VerticalMode


BASIC_ORDERS: tuple[tuple[int, int], ...] = (
    (1, 0), (-1, 0), (0, 1), (0, -1)
)


@dataclass(frozen=True)
class ThreeDCWTSettings:
    truncation_order: int = 10
    vertical_step_nm: float = 2.0
    bragg_tolerance_nm: float = 1e-4
    bragg_max_iterations: int = 30
    bragg_relaxation: float = 0.65


@dataclass
class GeometryCouplingResult:
    coupling_m: np.ndarray
    c1d_m: np.ndarray
    crad_m: np.ndarray
    c2d_m: np.ndarray
    eigenvalues_m: np.ndarray
    eigenvectors: np.ndarray
    bragg_wavelength_nm: float
    effective_index: float
    group_index: float
    pc_confinement: float
    vertical_mode: VerticalMode
    pc_z_m: np.ndarray
    pc_field: np.ndarray
    pc_weights_m: np.ndarray
    radiation_surface_factor: complex
    lattice_constant_nm: float
    average_pc_epsilon: float
    passivity_correction_m: float
    fourier: Callable[[int, int], complex]
    unit_cell_response_x: dict[tuple[int, int], np.ndarray]
    unit_cell_response_y: dict[tuple[int, int], np.ndarray]

    def radiation_fields(self, fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return Delta Ex and Delta Ey just above the PC, Liang Eq. (3.20)."""
        rx, sx, ry, sy = fields
        xi = self.fourier
        factor = self.radiation_surface_factor
        delta_ex = factor*(xi(0, -1)*ry + xi(0, 1)*sy)
        delta_ey = factor*(xi(-1, 0)*rx + xi(1, 0)*sx)
        return delta_ex, delta_ey

    def unit_cell_fields(
        self, coefficients: np.ndarray, points: int = 161
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct the local Bloch field including radiative/high orders.

        The supplied four coefficients are the slowly varying basic-wave
        amplitudes at one device location.  Responses stored from Liang
        Eqs. (3.20) and (3.25)-(3.26) rebuild Ex/Ey at the PC-layer centre.
        """
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if coefficients.shape != (4,):
            raise ValueError("coefficients must have shape (4,)")
        axis = np.linspace(-0.5, 0.5, points)
        x, y = np.meshgrid(axis, axis)
        ex = np.zeros_like(x, dtype=np.complex128)
        ey = np.zeros_like(x, dtype=np.complex128)
        for order, response in self.unit_cell_response_x.items():
            m, n = order
            ex += (response@coefficients)*np.exp(-2j*np.pi*(m*x+n*y))
        for order, response in self.unit_cell_response_y.items():
            m, n = order
            ey += (response@coefficients)*np.exp(-2j*np.pi*(m*x+n*y))
        intensity = np.abs(ex)**2+np.abs(ey)**2
        scale = max(float(intensity.max()), np.finfo(float).eps)
        return axis, intensity/scale, np.angle(ex+ey)


def _trapezoid_weights(axis: np.ndarray) -> np.ndarray:
    if axis.size < 2:
        raise ValueError("At least two vertical points are required in the PC layer")
    weights = np.empty_like(axis, dtype=float)
    weights[0] = 0.5*(axis[1]-axis[0])
    weights[-1] = 0.5*(axis[-1]-axis[-2])
    weights[1:-1] = 0.5*(axis[2:]-axis[:-2])
    return weights


def _pc_layer_coordinates(
    stack: LayerStack, mode: VerticalMode, pc_layer_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_nm = 2.0*stack.padding_um*1e3 + sum(x.thickness_nm for x in stack.layers)
    z_nm = mode.z_um*1e3 + 0.5*total_nm
    cursor_nm = stack.padding_um*1e3
    ranges: list[tuple[float, float]] = []
    for layer in stack.layers:
        stop_nm = cursor_nm+layer.thickness_nm
        if layer.name == pc_layer_name:
            ranges.append((cursor_nm, stop_nm))
        cursor_nm = stop_nm
    if not ranges:
        raise ValueError(f"PC layer {pc_layer_name!r} was not found in the vertical stack")
    mask = np.zeros(z_nm.shape, dtype=bool)
    for start_nm, stop_nm in ranges:
        mask |= (z_nm >= start_nm) & (z_nm <= stop_nm)
    z_m = z_nm[mask]*1e-9
    field = np.asarray(mode.field[mask], dtype=np.complex128)
    return z_m, field, _trapezoid_weights(z_m)


def solve_bragg_vertical_mode(
    stack: LayerStack,
    lattice_constant_nm: float,
    wavelength_guess_nm: float,
    settings: ThreeDCWTSettings,
) -> tuple[float, VerticalMode]:
    """Solve beta(lambda)=2*pi/a by fixed-point iteration lambda=a*neff."""
    wavelength_nm = float(wavelength_guess_nm)
    for _ in range(settings.bragg_max_iterations):
        mode = stack.solve_te0(wavelength_nm, dz_nm=settings.vertical_step_nm)
        target_nm = lattice_constant_nm*mode.effective_index
        updated_nm = (
            settings.bragg_relaxation*target_nm
            + (1.0-settings.bragg_relaxation)*wavelength_nm
        )
        if abs(updated_nm-wavelength_nm) <= settings.bragg_tolerance_nm:
            wavelength_nm = updated_nm
            mode = stack.solve_te0(wavelength_nm, dz_nm=settings.vertical_step_nm)
            return wavelength_nm, mode
        wavelength_nm = updated_nm
    raise RuntimeError("Bragg wavelength/effective-index iteration did not converge")


def _double_green_integral(
    z_m: np.ndarray,
    field: np.ndarray,
    weights_m: np.ndarray,
    vertical_wavenumber_m: complex,
    radiative: bool,
) -> complex:
    distance = np.abs(z_m[:, None]-z_m[None, :])
    if radiative:
        kernel = (
            -1j/(2.0*vertical_wavenumber_m)
            * np.exp(-1j*vertical_wavenumber_m*distance)
        )
    else:
        kernel = (
            1.0/(2.0*vertical_wavenumber_m)
            * np.exp(-vertical_wavenumber_m*distance)
        )
    left = np.conj(field)*weights_m
    right = field*weights_m
    return complex(left @ kernel @ right)


def _surface_green_integral(
    z_m: np.ndarray,
    field: np.ndarray,
    weights_m: np.ndarray,
    vertical_wavenumber_m: complex,
) -> complex:
    z_surface = float(z_m[-1])
    kernel = (
        -1j/(2.0*vertical_wavenumber_m)
        * np.exp(-1j*vertical_wavenumber_m*np.abs(z_surface-z_m))
    )
    return complex(np.sum(kernel*field*weights_m))


def _project_radiation_passive(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove only non-physical negative radiation eigenvalues from roundoff/approximation."""
    hermitian = 0.5*(matrix+matrix.conj().T)
    radiation = (matrix-matrix.conj().T)/(2j)
    values, vectors = np.linalg.eigh(radiation)
    clipped = np.maximum(values, 0.0)
    correction = float(np.max(np.abs(clipped-values)))
    radiation_psd = (vectors*clipped) @ vectors.conj().T
    return hermitian+1j*radiation_psd, correction


def build_geometry_coupling(
    cell: SquareLatticeCell,
    stack: LayerStack,
    pc_layer_name: str,
    lattice_constant_nm: float,
    wavelength_guess_nm: float,
    settings: ThreeDCWTSettings = ThreeDCWTSettings(),
) -> GeometryCouplingResult:
    """Build C=C1D+Crad+C2D directly from geometry and the vertical mode."""
    if settings.truncation_order < 2:
        raise ValueError("truncation_order must be >= 2")
    average_epsilon = float(np.real(cell.fourier_epsilon(0, 0)))
    if average_epsilon <= 0.0:
        raise ValueError("The unit-cell average dielectric constant must be positive")

    wavelength_nm, vertical_mode = solve_bragg_vertical_mode(
        stack, lattice_constant_nm, wavelength_guess_nm, settings
    )
    z_m, theta, weights_m = _pc_layer_coordinates(stack, vertical_mode, pc_layer_name)
    pc_confinement = float(np.sum(np.abs(theta)**2*weights_m))
    integral_inverse_epsilon = pc_confinement/average_epsilon
    lattice_m = lattice_constant_nm*1e-9
    k0 = 2.0*pi/(wavelength_nm*1e-9)
    beta0 = 2.0*pi/lattice_m
    pc_index = np.sqrt(average_epsilon)
    z_all_m = vertical_mode.z_um*1e-6
    all_weights_m = _trapezoid_weights(z_all_m)
    group_index = float(
        np.sum(vertical_mode.index**2*np.abs(vertical_mode.field)**2*all_weights_m)
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

    c1d = np.zeros((4, 4), dtype=np.complex128)
    prefactor_1d = -k0**2*pc_confinement/(2.0*beta0)
    c1d[0, 1] = prefactor_1d*xi(2, 0)
    c1d[1, 0] = prefactor_1d*xi(-2, 0)
    c1d[2, 3] = prefactor_1d*xi(0, 2)
    c1d[3, 2] = prefactor_1d*xi(0, -2)
    c1d = 0.5*(c1d+c1d.conj().T)

    beta_radiative = complex(k0*pc_index)
    radiation_integral = _double_green_integral(
        z_m, theta, weights_m, beta_radiative, radiative=True
    )
    surface_integral = _surface_green_integral(
        z_m, theta, weights_m, beta_radiative
    )
    crad = np.zeros((4, 4), dtype=np.complex128)
    prefactor_rad = -k0**4/(2.0*beta0)
    for row in (0, 1):
        p, q = BASIC_ORDERS[row]
        for column in (0, 1):
            r, s = BASIC_ORDERS[column]
            crad[row, column] = (
                prefactor_rad*xi(p, q)*xi(-r, -s)*radiation_integral
            )
    for row in (2, 3):
        p, q = BASIC_ORDERS[row]
        for column in (2, 3):
            r, s = BASIC_ORDERS[column]
            crad[row, column] = (
                prefactor_rad*xi(p, q)*xi(-r, -s)*radiation_integral
            )
    crad, passivity_correction = _project_radiation_passive(crad)

    c2d = np.zeros((4, 4), dtype=np.complex128)
    # Response matrices at the PC-layer centre are retained for a genuine
    # high-order reconstruction of the one-cell near field (Liang Fig. 3.3).
    probe = z_m.size//2
    z_probe = float(z_m[probe])
    theta_probe = complex(theta[probe])
    radiative_kernel = (
        -1j/(2.0*beta_radiative)
        * np.exp(-1j*beta_radiative*np.abs(z_probe-z_m))
    )
    radiative_probe = complex(np.sum(radiative_kernel*theta*weights_m))
    response_x: dict[tuple[int, int], np.ndarray] = {
        (0, 1): theta_probe*np.asarray((0.0, 0.0, 1.0, 0.0)),
        (0, -1): theta_probe*np.asarray((0.0, 0.0, 0.0, 1.0)),
        (0, 0): k0**2*radiative_probe*np.asarray(
            (0.0, 0.0, xi(0, -1), xi(0, 1)), dtype=np.complex128
        ),
    }
    response_y: dict[tuple[int, int], np.ndarray] = {
        (1, 0): theta_probe*np.asarray((1.0, 0.0, 0.0, 0.0)),
        (-1, 0): theta_probe*np.asarray((0.0, 1.0, 0.0, 0.0)),
        (0, 0): k0**2*radiative_probe*np.asarray(
            (xi(-1, 0), xi(1, 0), 0.0, 0.0), dtype=np.complex128
        ),
    }
    order = settings.truncation_order
    for m in range(-order, order+1):
        for n in range(-order, order+1):
            squared_order = m*m+n*n
            if squared_order <= 1:
                continue
            beta_high = complex(np.sqrt(squared_order*beta0**2-k0**2*average_epsilon+0j))
            if beta_high.real <= 0.0:
                raise ValueError(
                    f"High-order wave ({m},{n}) is not evanescent; increase model scope"
                )
            green_integral = _double_green_integral(
                z_m, theta, weights_m, beta_high, radiative=False
            )
            probe_kernel = (
                1.0/(2.0*beta_high)
                * np.exp(-beta_high*np.abs(z_probe-z_m))
            )
            probe_green = complex(np.sum(probe_kernel*theta*weights_m))
            mu = np.empty(4, dtype=np.complex128)
            nu = np.empty(4, dtype=np.complex128)
            mu_probe = np.empty(4, dtype=np.complex128)
            nu_probe = np.empty(4, dtype=np.complex128)
            for column, (r, s) in enumerate(BASIC_ORDERS):
                coefficient = xi(m-r, n-s)
                mu[column] = k0**2*coefficient*green_integral
                nu[column] = -coefficient*integral_inverse_epsilon
                mu_probe[column] = k0**2*coefficient*probe_green
                nu_probe[column] = -coefficient*theta_probe/average_epsilon

            e_minus = np.asarray((-m*mu[0], -m*mu[1], n*mu[2], n*mu[3]))
            e_plus = np.asarray((n*nu[0], n*nu[1], m*nu[2], m*nu[3]))
            denominator = float(squared_order)
            sigma_x = (n*e_minus+m*e_plus)/denominator
            sigma_y = (-m*e_minus+n*e_plus)/denominator

            e_minus_probe = np.asarray((
                -m*mu_probe[0], -m*mu_probe[1],
                n*mu_probe[2], n*mu_probe[3],
            ))
            e_plus_probe = np.asarray((
                n*nu_probe[0], n*nu_probe[1],
                m*nu_probe[2], m*nu_probe[3],
            ))
            response_x[(m, n)] = (
                n*e_minus_probe+m*e_plus_probe
            )/denominator
            response_y[(m, n)] = (
                -m*e_minus_probe+n*e_plus_probe
            )/denominator

            for row, (p, q) in enumerate(BASIC_ORDERS):
                sigma = sigma_y if row < 2 else sigma_x
                c2d[row] += (
                    -k0**2/(2.0*beta0)*xi(p-m, q-n)*sigma
                )
    c2d = 0.5*(c2d+c2d.conj().T)

    coupling = c1d+crad+c2d
    validate_passive_coupling(coupling, atol=max(1e-8, 1e-10*np.linalg.norm(coupling)))
    eigenvalues, eigenvectors = np.linalg.eig(coupling)
    ordering = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[ordering]
    eigenvectors = eigenvectors[:, ordering]
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0, keepdims=True)

    return GeometryCouplingResult(
        coupling_m=coupling,
        c1d_m=c1d,
        crad_m=crad,
        c2d_m=c2d,
        eigenvalues_m=eigenvalues,
        eigenvectors=eigenvectors,
        bragg_wavelength_nm=wavelength_nm,
        effective_index=vertical_mode.effective_index,
        group_index=group_index,
        pc_confinement=pc_confinement,
        vertical_mode=vertical_mode,
        pc_z_m=z_m,
        pc_field=theta,
        pc_weights_m=weights_m,
        radiation_surface_factor=k0**2*surface_integral,
        lattice_constant_nm=lattice_constant_nm,
        average_pc_epsilon=average_epsilon,
        passivity_correction_m=passivity_correction,
        fourier=xi,
        unit_cell_response_x=response_x,
        unit_cell_response_y=response_y,
    )
