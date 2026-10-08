"""Independent square four-wave CWT with layered outgoing Green kernels.

Liang's retained four waves and longitudinal high-order algebra are unchanged;
the transverse Green functions now include interfaces and the finite DBR.
This is still a TE/weak-modulation CWT approximation, not full-vector FEM/FDTD.
中文：半导体原模块保持不变；上下辐射损耗与同一个 Crad 的反厄米部分必须守恒。
"""

from dataclasses import dataclass, replace
import numpy as np
from .three_d_cwt import BASIC_ORDERS, build_geometry_coupling
from .stratified_optics import local_green_kernel
from .coupling import validate_passive_coupling


@dataclass
class ReflectorCoupling:
    result: object
    upward_amplitude_loss_m: np.ndarray
    downward_amplitude_loss_m: np.ndarray
    downward_surface_factor: complex
    optical_theorem_residual: float

    def side_power_losses(self, fields):
        """2*<F|L_side|F>/<F|F>; inverse metres, POWER not amplitude loss."""
        flat = np.asarray(fields).reshape(4, -1)
        norm = np.sum(abs(flat) ** 2)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("A nonzero finite four-wave field is required")
        return tuple(
            float(2 * np.einsum("ik,ij,jk->", flat.conj(), op, flat).real / norm)
            for op in (self.upward_amplitude_loss_m, self.downward_amplitude_loss_m)
        )


def build_reflector_coupling(
    cell,
    guided_stack,
    radiation_stack,
    pc_layer_name,
    lattice_constant_nm,
    wavelength_guess_nm,
    settings,
):
    base = build_geometry_coupling(
        cell,
        guided_stack,
        pc_layer_name,
        lattice_constant_nm,
        wavelength_guess_nm,
        settings,
    )
    hosts = [i for i, l in enumerate(radiation_stack.layers) if l.name == pc_layer_name]
    if (
        len(hosts) != 1
        or sum(l.name == pc_layer_name for l in guided_stack.layers) != 1
    ):
        raise ValueError("Exactly one matching PC host required in each stack")
    host = hosts[0]
    if not np.isclose(
        complex(radiation_stack.layers[host].refractive_index) ** 2,
        base.average_pc_epsilon,
        rtol=1e-10,
    ):
        raise ValueError(
            "Radiation host index must equal the geometry-averaged PC index"
        )
    # Translate the original padded/centred vertical coordinates to physical
    # finite-radiation-stack coordinates. PC centre must match in both stacks.
    guided_start = guided_stack.padding_um * 1e-6
    for layer in guided_stack.layers:
        if layer.name == pc_layer_name:
            break
        guided_start += layer.thickness_nm * 1e-9
    z = base.pc_z_m - guided_start + radiation_stack.boundaries_m[host]
    theta, weights, xi = base.pc_field, base.pc_weights_m, base.fourier
    weighted = theta * weights
    k0 = 2 * np.pi / (base.bragg_wavelength_nm * 1e-9)
    beta = 2 * np.pi / (lattice_constant_nm * 1e-9)
    kernel = local_green_kernel(radiation_stack, host, base.bragg_wavelength_nm)
    matrix = kernel.matrix(z)
    integral = weighted.conj() @ matrix @ weighted
    outgoing = kernel.outgoing_amplitudes(z)
    top = complex(outgoing.top @ weighted)
    bottom = complex(outgoing.bottom @ weighted)
    # F maps four basic waves to two polarization sources. Real dielectric
    # Fourier coefficients obey xi(-G)=xi(G)*, ensuring reciprocal PSD flux.
    source = np.array(
        [[xi(-1, 0), xi(1, 0), 0, 0], [0, 0, xi(0, -1), xi(0, 1)]], complex
    )
    source_gram = source.conj().T @ source
    prefactor = k0**4 / (2 * beta)
    crad = -prefactor * integral * source_gram
    up = (
        prefactor * max(outgoing.top_admittance_m.real, 0) * abs(top) ** 2 * source_gram
    )
    down = (
        prefactor
        * max(outgoing.bottom_admittance_m.real, 0)
        * abs(bottom) ** 2
        * source_gram
    )
    rad_loss = (crad - crad.conj().T) / (2j)
    residual = float(
        np.linalg.norm(rad_loss - up - down) / max(np.linalg.norm(rad_loss), 1e-30)
    )
    if not kernel.lossless:
        raise ValueError(
            "Absorbing films require a separate absorption-loss operator before rate coupling"
        )
    if residual > 1e-8:
        raise ValueError(f"Radiation optical theorem failed: {residual}")
    # Rebuild transverse high-order responses with reflected evanescent Green.
    # Longitudinal nu term is the original local epsilon approximation.
    c2d = np.zeros((4, 4), complex)
    response_x = {
        key: value.copy()
        for key, value in base.unit_cell_response_x.items()
        if sum(x * x for x in key) <= 1
    }
    response_y = {
        key: value.copy()
        for key, value in base.unit_cell_response_y.items()
        if sum(x * x for x in key) <= 1
    }
    probe = len(z) // 2
    probe_rad = matrix[probe] @ weighted
    response_x[(0, 0)] = k0**2 * probe_rad * source[1]
    response_y[(0, 0)] = k0**2 * probe_rad * source[0]
    for m in range(-settings.truncation_order, settings.truncation_order + 1):
        for n in range(-settings.truncation_order, settings.truncation_order + 1):
            order2 = m * m + n * n
            if order2 <= 1:
                continue
            high = local_green_kernel(
                radiation_stack,
                host,
                base.bragg_wavelength_nm,
                q_parallel_per_m=np.sqrt(order2) * beta,
            )
            if (
                high.top_admittance_m.real > 1e-6
                or high.bottom_admittance_m.real > 1e-6
            ):
                raise ValueError(
                    "Open high-order exterior channels require an extended lossy C2D model"
                )
            high_matrix = high.matrix(z)
            integ = weighted.conj() @ high_matrix @ weighted
            probe_integ = high_matrix[probe] @ weighted
            coeff = np.array([xi(m - r, n - s) for r, s in BASIC_ORDERS])
            mu, mu_probe = k0**2 * coeff * integ, k0**2 * coeff * probe_integ
            nu = -coeff * base.pc_confinement / base.average_pc_epsilon
            nu_probe = -coeff * theta[probe] / base.average_pc_epsilon
            minus = np.array([-m * mu[0], -m * mu[1], n * mu[2], n * mu[3]])
            plus = np.array([n * nu[0], n * nu[1], m * nu[2], m * nu[3]])
            sx, sy = (n * minus + m * plus) / order2, (-m * minus + n * plus) / order2
            mp = np.array(
                [-m * mu_probe[0], -m * mu_probe[1], n * mu_probe[2], n * mu_probe[3]]
            )
            pp = np.array(
                [n * nu_probe[0], n * nu_probe[1], m * nu_probe[2], m * nu_probe[3]]
            )
            response_x[(m, n)], response_y[(m, n)] = (n * mp + m * pp) / order2, (
                -m * mp + n * pp
            ) / order2
            for row, (p, q) in enumerate(BASIC_ORDERS):
                c2d[row] += (
                    -(k0**2) / (2 * beta) * xi(p - m, q - n) * (sy if row < 2 else sx)
                )
    c2d = (c2d + c2d.conj().T) / 2
    coupling = base.c1d_m + crad + c2d
    validate_passive_coupling(coupling, atol=1e-7)
    values, vectors = np.linalg.eig(coupling)
    order = np.argsort(values.real)
    result = replace(
        base,
        coupling_m=coupling,
        crad_m=crad,
        c2d_m=c2d,
        eigenvalues_m=values[order],
        eigenvectors=vectors[:, order],
        radiation_surface_factor=k0**2 * top,
        passivity_correction_m=0,
        unit_cell_response_x=response_x,
        unit_cell_response_y=response_y,
    )
    return ReflectorCoupling(result, up, down, k0**2 * bottom, residual)
