"""Reflection-aware six-wave Yb:YAG interfaces for matched-area comparisons.

The triangular Bravais lattice retains all six basic TE waves. The same
stratified outgoing Green function supplies radiation damping and emitted flux;
closed higher orders use its evanescent response. No four-wave padding, fitted
mirror multiplier, or material-gain substitution is used here.

中文：保留三角晶格的六个基本波；辐射阻尼与上下出射功率来自同一个
分层出射 Green 函数。这里返回冷腔光学量，材料增益和泵浦阈值另行计算。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .stratified_optics import local_green_kernel
from .three_d_cwt import _pc_layer_coordinates, _roundoff_hermitian
from .triangular_six_wave import (
    TRIANGULAR_BASIC_ORDERS, TRIANGULAR_POLARIZATIONS,
    build_triangular_coupling, reciprocal_components,
    reciprocal_metric_squared, _validate_six_wave_passivity,
)


@dataclass
class TriangularReflectorCoupling:
    result: object
    upward_amplitude_loss_m: np.ndarray
    downward_amplitude_loss_m: np.ndarray
    downward_surface_factor: complex
    optical_theorem_residual: float

    def side_power_losses(self, fields):
        """Return upper/lower intensity loss in m^-1 / 返回上下功率损耗系数。"""
        flat = np.asarray(fields).reshape(6, -1)
        norm = np.sum(abs(flat)**2)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Nonzero finite six-wave fields are required")
        # The operators act on amplitudes; power loss is twice their expectation.
        # 算子对应场振幅衰减，因此转成功率损耗时需乘 2，不能重复乘入速率方程。
        return tuple(float(2*np.einsum("ik,ij,jk->", flat.conj(), op, flat).real/norm)
                     for op in (self.upward_amplitude_loss_m, self.downward_amplitude_loss_m))


def build_triangular_reflector_coupling(
    cell, guided_stack, radiation_stack, pc_layer_name, lattice_constant_nm,
    wavelength_guess_nm, settings, *, fixed_wavelength_nm=None,
):
    """Six-wave counterpart to ``build_reflector_coupling``.

    The existing triangular free-host derivation supplies the basic-wave
    coupling and vertical TE field. Its normal and high-order Green responses
    are rebuilt using the actual dielectric stack, preserving the triangular
    reciprocal metric and TE projectors. Absorption and open high-order exterior
    channels are rejected rather than silently omitted.
    """
    base = build_triangular_coupling(
        cell, guided_stack, pc_layer_name, lattice_constant_nm,
        wavelength_guess_nm, settings, fixed_wavelength_nm=fixed_wavelength_nm,
    )
    hosts = [i for i,l in enumerate(radiation_stack.layers) if l.name == pc_layer_name]
    if len(hosts) != 1 or sum(l.name == pc_layer_name for l in guided_stack.layers) != 1:
        raise ValueError("One matching PC host is required in each stack")
    host = hosts[0]
    if not np.isclose(complex(radiation_stack.layers[host].refractive_index)**2,
                      base.average_pc_epsilon, rtol=1e-10):
        raise ValueError("Radiation host index must match geometry-averaged PC index")
    z, theta, weights = _pc_layer_coordinates(guided_stack, base.vertical_mode, pc_layer_name)
    guided_start = guided_stack.padding_um*1e-6
    for layer in guided_stack.layers:
        if layer.name == pc_layer_name:
            break
        guided_start += layer.thickness_nm*1e-9
    z = z - guided_start + radiation_stack.boundaries_m[host]
    # Match physical PC coordinates before applying the reflected Green kernel.
    # 先对齐两个纵向模型的物理 PC 层位置；所有 Green 积分坐标均以米计。
    weighted = theta*weights
    xi = base.fourier
    k0 = 2*np.pi/(base.bragg_wavelength_nm*1e-9)
    beta0 = 4*np.pi/(np.sqrt(3)*lattice_constant_nm*1e-9)
    kernel = local_green_kernel(radiation_stack, host, base.bragg_wavelength_nm)
    if not kernel.lossless:
        raise ValueError("Absorbing films require a separate absorption operator")
    matrix = kernel.matrix(z)
    integral = weighted.conj() @ matrix @ weighted
    outgoing = kernel.outgoing_amplitudes(z)
    top, bottom = complex(outgoing.top @ weighted), complex(outgoing.bottom @ weighted)
    source = np.array([[xi(-m,-n)*polar[j]
                        for (m,n),polar in zip(TRIANGULAR_BASIC_ORDERS,
                                               TRIANGULAR_POLARIZATIONS)]
                       for j in range(2)], complex)
    gram = source.conj().T @ source
    prefactor = k0**4/(2*beta0)
    cr = -prefactor*integral*gram
    up = prefactor*max(outgoing.top_admittance_m.real,0)*abs(top)**2*gram
    down = prefactor*max(outgoing.bottom_admittance_m.real,0)*abs(bottom)**2*gram
    # A lossless stack must send exactly the anti-Hermitian loss into open ports.
    # 无吸收膜层时，反厄米辐射损耗应等于上下出射通量；超差直接报错。
    damping = (cr-cr.conj().T)/(2j)
    residual = float(np.linalg.norm(damping-up-down)/max(np.linalg.norm(damping),1e-30))
    if residual > 1e-8:
        raise ValueError(f"Six-wave radiation optical theorem failed: {residual}")
    response_x = {k:v.copy() for k,v in base.unit_cell_response_x.items()
                  if reciprocal_metric_squared(*k) == 1}
    response_y = {k:v.copy() for k,v in base.unit_cell_response_y.items()
                  if reciprocal_metric_squared(*k) == 1}
    probe = len(z)//2
    response_x[(0,0)] = k0**2*(matrix[probe] @ weighted)*source[0]
    response_y[(0,0)] = k0**2*(matrix[probe] @ weighted)*source[1]
    ch = np.zeros((6,6), complex)
    radial_green = {}
    limit = int(np.ceil(2*settings.truncation_order/np.sqrt(3)))
    for m in range(-limit,limit+1):
        for n in range(-limit,limit+1):
            metric = reciprocal_metric_squared(m,n)
            if metric <= 1 or metric > settings.truncation_order**2:
                continue
            mx,ny = reciprocal_components((m,n))
            if metric not in radial_green:
                high = local_green_kernel(radiation_stack,host,base.bragg_wavelength_nm,
                                          q_parallel_per_m=np.sqrt(metric)*beta0)
                if high.top_admittance_m.real > 1e-6 or high.bottom_admittance_m.real > 1e-6:
                    raise ValueError("Open high-order channels require an extended lossy model")
                high_matrix = high.matrix(z)
                integ = weighted.conj() @ high_matrix @ weighted
                probe_integ = high_matrix[probe] @ weighted
                radial_green[metric] = integ, probe_integ
            integ, probe_integ = radial_green[metric]
            zx,zy,zxp,zyp = (np.empty(6,complex) for _ in range(4))
            for col,(r,s) in enumerate(TRIANGULAR_BASIC_ORDERS):
                coefficient = xi(m-r,n-s)
                rho,eta = TRIANGULAR_POLARIZATIONS[col]
                transverse, longitudinal = ny*rho-mx*eta, mx*rho+ny*eta
                mu = k0**2*coefficient*transverse*integ
                nu = -coefficient*longitudinal*base.pc_confinement/base.average_pc_epsilon
                mup = k0**2*coefficient*transverse*probe_integ
                nup = -coefficient*longitudinal*theta[probe]/base.average_pc_epsilon
                zx[col],zy[col] = (ny*mu+mx*nu)/metric,(-mx*mu+ny*nu)/metric
                zxp[col],zyp[col] = (ny*mup+mx*nup)/metric,(-mx*mup+ny*nup)/metric
            response_x[(m,n)],response_y[(m,n)] = zxp,zyp
            for row,(p,q) in enumerate(TRIANGULAR_BASIC_ORDERS):
                rho,eta = TRIANGULAR_POLARIZATIONS[row]
                ch[row] += -k0**2/(2*beta0)*xi(p-m,q-n)*(rho*zx+eta*zy)
    ch = _roundoff_hermitian(ch, label="reflected Ch")
    coupling = base.cb_m+cr+ch
    _validate_six_wave_passivity(coupling,max(1e-8,1e-10*np.linalg.norm(coupling)))
    values,vectors = np.linalg.eig(coupling)
    ordering = np.lexsort((values.imag,values.real))
    vectors = vectors[:,ordering]
    vectors /= np.linalg.norm(vectors,axis=0,keepdims=True)
    result = replace(base,coupling_m=coupling,cr_m=cr,ch_m=ch,
        eigenvalues_m=values[ordering],eigenvectors=vectors,
        unit_cell_response_x=response_x,unit_cell_response_y=response_y,
        radiation_surface_factor=k0**2*top,passivity_correction_m=0.)
    return TriangularReflectorCoupling(result,up,down,k0**2*bottom,residual)


def binned_lateral_weights(fields, x_um, side_um=300., cells=31):
    """Conserve the mode's full energy while projecting a triangular mesh to x.

    On the uniform axial mesh every node has the same area. First integrate
    over each physical x column, then conservatively distribute each column's
    control interval into the reservoir bins; empty histogram-bin artefacts
    do not introduce artificial spatial hole burning. Normalization is over
    the full finite field.
    This is an x-z, y-averaged inversion model, not full x-y-z hole burning.

    中文：先沿 y 累加能量，再按控制区间重叠守恒投影到 x 分箱；纵向 z
    权重在增益适配器中另行加入。这不会恢复被平均掉的 y 向反转分布。
    """
    density = np.sum(abs(np.asarray(fields))**2,axis=0).ravel()
    x = np.asarray(x_um,float).ravel()
    if density.shape != x.shape or density.sum() <= 0 or not np.isfinite(density).all():
        raise ValueError("Finite fields and matching physical x coordinates are required")
    edges = np.linspace(-side_um/2,side_um/2,cells+1)
    if np.any(x < edges[0]-1e-8) or np.any(x > edges[-1]+1e-8):
        raise ValueError("Physical nodes fall outside the matched square aperture")
    unique,inverse=np.unique(np.round(x,10),return_inverse=True)
    column=np.bincount(inverse,weights=density)/density.sum()
    if len(unique)<2:raise ValueError("At least two physical x columns are required")
    old_edges=np.r_[-side_um/2,(unique[:-1]+unique[1:])/2,side_um/2]
    overlap=np.maximum(0,np.minimum(edges[1:,None],old_edges[None,1:])-
                          np.maximum(edges[:-1,None],old_edges[None,:-1]))
    return overlap @ (column/np.diff(old_edges))

