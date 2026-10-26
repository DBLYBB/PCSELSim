"""Auditable pulsed, uniform-inversion reduction of a supplied Yb:YAG mode.

The cold PCSEL mode, its group velocity, full-mode gain overlap and loss ports
come from ``SpatialYbYAGModel``. No cavity or loss is invented here. Inversion
is spatially uniform: this is a thin-film top-pump storage/start-up estimate,
not the x-z shared-inversion CW solver or a multimode/dynamical stability proof.
Pump powers refer to first entry, with an explicit external coupling factor.

中文：给定同一冷腔模式后求矩形顶泵脉冲响应；保留准三能级再吸收、泵受激
发射、泵耗尽及自发种子。均匀反转是明确的降阶近似，不能代替空间烧孔/多模
计算。光子数和激发态离子数用相同总离子数归一化，并积分完整有效能量账本。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import c, h
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .ybyag_spatial_rates import SpatialYbYAGModel, _constants


@dataclass(frozen=True)
class RectangularPumpPulse:
    external_energy_J: float
    duration_s: float
    coupling_efficiency: float = 1.0
    top_passes: int = 1
    return_efficiency: float = 0.98
    spontaneous_beta: float = 1e-8
    after_pulse_s: float | None = None

    def __post_init__(self):
        if (not np.isfinite([self.external_energy_J, self.duration_s,
                             self.coupling_efficiency, self.return_efficiency,
                             self.spontaneous_beta]).all()
                or self.external_energy_J < 0 or self.duration_s <= 0
                or not 0 < self.coupling_efficiency <= 1
                or not 0 <= self.return_efficiency <= 1
                or not 0 <= self.spontaneous_beta <= 1
                or not isinstance(self.top_passes, (int, np.integer))
                or self.top_passes < 1):
            raise ValueError("Finite nonnegative energy, positive duration/coupling and valid pass/beta values required")
        if self.after_pulse_s is not None and (not np.isfinite(self.after_pulse_s) or self.after_pulse_s < 0):
            raise ValueError("after_pulse_s must be finite and nonnegative")


@dataclass(frozen=True)
class PulsedYbYAGResult:
    time_s: np.ndarray
    excited_fraction: np.ndarray
    photon_number: np.ndarray
    upward_power_W: np.ndarray
    metrics: dict


def _uniform_constants(model):
    k = _constants(model)
    k["ions"] = k["nt"] * model.gain_volume_m3
    k["gain_overlap"] = float(model.optical_gain_weights.sum())
    k["energy_scale"] = k["ions"] * k["laser_energy"]
    return k


def uniform_top_pump_flow(model: SpatialYbYAGModel, fraction: float, internal_W: float,
                          passes: int = 1, return_efficiency: float = 0.98):
    """Same first-entry photons are propagated through every pass exactly once.

    Net absorption may be negative for an externally prepared over-inverted
    medium; pump stimulated emission must not be clipped away. All returned
    powers are signed physical boundary flows, not additional pump sources.
    中文：多程只回收前一程幸存泵光；不会把每一程再次计为外部输入。
    """
    k = _uniform_constants(model)
    alpha = k["nt"] * (k["ap"] * (1 - fraction) - k["ep"] * fraction)
    transmission = float(np.exp(-alpha * model.gain_thickness_m))
    dtransmission = transmission*k["nt"]*(k["ap"]+k["ep"])*model.gain_thickness_m
    entrance, dentrance = float(internal_W), 0.0
    absorbed = relay = dabsorbed = drelay = 0.0
    for i in range(passes):
        exiting = entrance * transmission
        dexiting = dentrance*transmission+entrance*dtransmission
        absorbed += entrance - exiting
        dabsorbed += dentrance-dexiting
        if i + 1 < passes:
            relay += exiting * (1 - return_efficiency)
            drelay += dexiting*(1-return_efficiency)
            entrance = exiting * return_efficiency
            dentrance = dexiting*return_efficiency
    return dict(absorbed_W=absorbed, escaped_W=exiting, relay_loss_W=relay,
                boundary_residual_W=internal_W-absorbed-exiting-relay,
                absorbed_derivative_W=dabsorbed,escaped_derivative_W=dexiting,
                relay_derivative_W=drelay)


def uniform_top_pump_threshold(model: SpatialYbYAGModel, pulse: RectangularPumpPulse):
    """Unlased CW threshold of this uniform single-mode reduction, not E_th.

    A finite-duration pulse has start-up delay and no universal energy-only
    threshold. This power is useful solely as the long-pulse reference.
    """
    k = _uniform_constants(model)
    fth = ((model.total_power_loss_per_m / (k["gain_overlap"] * k["nt"]) + k["al"])
           / (k["el"] + k["al"]))
    ceiling = k["ap"] / (k["ap"] + k["ep"])
    flow = uniform_top_pump_flow(model, fth, 1.0, pulse.top_passes, pulse.return_efficiency)
    reachable = bool(0 < fth < ceiling and flow["absorbed_W"] > 0)
    internal = k["ions"] * fth * k["pump_energy"] / k["tau"] / flow["absorbed_W"] if reachable else None
    return dict(threshold_inversion_fraction=float(fth), pump_limited_fraction=float(ceiling),
                reachable=reachable, internal_CW_threshold_W=internal,
                external_CW_threshold_W=None if internal is None else internal/pulse.coupling_efficiency,
                small_signal_single_pass_optical_depth=float(k["nt"]*k["ap"]*model.gain_thickness_m),
                interpretation="Uniform-inversion one-mode CW reference; finite-pulse energy threshold is duration/seed dependent")


def uniform_top_cw_solution(model: SpatialYbYAGModel, pulse: RectangularPumpPulse,
                            external_power_W: float):
    """Exact beta=0 uniform-inversion CW limit used as an independent check."""
    k = _uniform_constants(model)
    internal = external_power_W * pulse.coupling_efficiency
    threshold = uniform_top_pump_threshold(model, pulse)
    ceiling = threshold["pump_limited_fraction"]

    def gap(f):
        absorbed = uniform_top_pump_flow(model, f, internal, pulse.top_passes, pulse.return_efficiency)["absorbed_W"]
        return absorbed/(k["ions"]*k["pump_energy"]) - f/k["tau"]

    unlased = 0.0 if internal == 0 else float(brentq(gap, 0.0, ceiling, xtol=1e-14))
    f = unlased
    photon_fraction = 0.0
    if threshold["reachable"] and unlased > threshold["threshold_inversion_fraction"]:
        f = threshold["threshold_inversion_fraction"]
        photon_fraction = gap(f)/(model.group_velocity_m_s*model.total_power_loss_per_m)
    flow = uniform_top_pump_flow(model, f, internal, pulse.top_passes, pulse.return_efficiency)
    upwards = k["energy_scale"]*model.group_velocity_m_s*model.upward_power_loss_per_m*photon_fraction
    return dict(excited_fraction=float(f), photon_fraction=float(photon_fraction),
                upward_power_W=float(upwards), **flow)


def simulate_pulsed_ybyag(model: SpatialYbYAGModel, pulse: RectangularPumpPulse, *,
                          rtol: float = 2e-7, samples: int = 1601,
                          initial_excited_fraction: float = 0.0,
                          initial_photons: float = 0.0, method: str = "LSODA"):
    """Integrate quasi-three-level populations, photons and energy flows.

    f=N2/Nions and s=Nph/Nions. The stimulated term occurs with opposite sign
    in df and ds; q_gain is included ONCE. Fluorescence into the mode transfers
    beta*f/tau photons, while the entire f/tau leaves the excited reservoir.
    E_stored=(N2+Nph)*h*nu_l is a common effective excitation-energy convention;
    the quantum defect is explicit. Real fluorescence spectral heat exchange,
    thermal refractive changes and a coherent pump standing wave are omitted.

    中文：外部输入、入口耦合损失、吸收/逃逸/回程泵光、量子亏损、腔外荧光、
    上下输出、其他腔损耗与剩余储能全部独立积分。能量守恒不等于温度模型已验证。
    """
    if samples < 3 or not 0 <= initial_excited_fraction <= 1 or initial_photons < 0:
        raise ValueError("Physical initial populations and samples>=3 required")
    if method not in ("LSODA","Radau","BDF"):
        raise ValueError("An explicitly stiff integrator LSODA, Radau or BDF is required")
    k = _uniform_constants(model)
    beta = pulse.spontaneous_beta
    tail = 2*k["tau"] if pulse.after_pulse_s is None else pulse.after_pulse_s
    end = pulse.duration_s + tail
    # Dimensionless accumulated energies, each divided by Nions*h*nu_l.
    names = ("external_pump", "before_gain_loss", "absorbed_pump", "escaped_pump",
             "relay_loss", "quantum_defect", "fluorescence", "upward_output",
             "downward_output", "other_cavity_loss")
    y0 = np.zeros(2+len(names))
    y0[:2] = initial_excited_fraction, initial_photons/k["ions"]
    initial_storage = (y0[0]+y0[1])*k["energy_scale"]
    power = pulse.external_energy_J/pulse.duration_s
    vg = model.group_velocity_m_s
    loss, up, down = (model.total_power_loss_per_m, model.upward_power_loss_per_m,
                      model.downward_power_loss_per_m)

    def rhs(t, state, on):
        f, s = state[:2]
        external = power if on else 0.0
        internal = external*pulse.coupling_efficiency
        flow = uniform_top_pump_flow(model, f, internal, pulse.top_passes, pulse.return_efficiency)
        gain = k["nt"]*((k["el"]+k["al"])*f-k["al"])
        stimulated = vg*k["gain_overlap"]*gain*s
        df = flow["absorbed_W"]/(k["ions"]*k["pump_energy"])-f/k["tau"]-stimulated
        ds = stimulated-vg*loss*s+beta*f/k["tau"]
        scale = k["energy_scale"]
        rates = (external/scale, (external-internal)/scale,
                 flow["absorbed_W"]/scale, flow["escaped_W"]/scale,
                 flow["relay_loss_W"]/scale,
                 flow["absorbed_W"]*(1-k["laser_energy"]/k["pump_energy"])/scale,
                 (1-beta)*f/k["tau"], vg*up*s, vg*down*s, vg*(loss-up-down)*s)
        return np.array((df, ds, *rates))

    def jacobian(t, state, on):
        # Analytic derivatives also keep finite-difference Jacobian machinery
        # from perturbing ten passive accumulators by overflowing factors.
        f,s=state[:2]
        internal = power*pulse.coupling_efficiency if on else 0.0
        flow=uniform_top_pump_flow(model,f,internal,pulse.top_passes,pulse.return_efficiency)
        gain=k["nt"]*((k["el"]+k["al"])*f-k["al"])
        derivative_gain=k["nt"]*(k["el"]+k["al"])
        j=np.zeros((len(y0),len(y0)))
        j[0,0]=flow["absorbed_derivative_W"]/(k["ions"]*k["pump_energy"])-1/k["tau"]-vg*k["gain_overlap"]*derivative_gain*s
        j[0,1]=-vg*k["gain_overlap"]*gain
        j[1,0]=vg*k["gain_overlap"]*derivative_gain*s+beta/k["tau"]
        j[1,1]=vg*(k["gain_overlap"]*gain-loss)
        scale=k["energy_scale"]
        j[4,0]=flow["absorbed_derivative_W"]/scale
        j[5,0]=flow["escaped_derivative_W"]/scale
        j[6,0]=flow["relay_derivative_W"]/scale
        j[7,0]=flow["absorbed_derivative_W"]*(1-k["laser_energy"]/k["pump_energy"])/scale
        j[8,0]=(1-beta)/k["tau"]
        j[9,1],j[10,1],j[11,1]=vg*up,vg*down,vg*(loss-up-down)
        return j

    # Split at the exact pump edge; adaptive integration never straddles an
    # unannounced discontinuity. No state clipping/threshold switch is applied.
    atol = np.array([2e-11, 2e-23]+[2e-11]*len(names))
    first = solve_ivp(lambda t,y: rhs(t,y,True), (0.0,pulse.duration_s), y0,
                      method=method, rtol=rtol, atol=atol, dense_output=True,
                      jac=lambda t,y: jacobian(t,y,True))
    if not first.success:
        raise RuntimeError(f"Pulse integration failed: {first.message}")
    second = None
    if tail > 0:
        second = solve_ivp(lambda t,y: rhs(t,y,False), (pulse.duration_s,end), first.y[:,-1],
                           method=method, rtol=rtol, atol=atol, dense_output=True,
                           jac=lambda t,y: jacobian(t,y,False))
        if not second.success:
            raise RuntimeError(f"After-pulse integration failed: {second.message}")
    knots = first.y if second is None else np.column_stack((first.y,second.y))
    if knots[0].min() < -1e-8 or knots[0].max() > 1+1e-8 or knots[1].min() < -1e-18:
        raise RuntimeError("Accepted integration left physical population domain")
    on_time = np.linspace(0,pulse.duration_s,max(3,int(samples*.75)))
    off_time = np.linspace(pulse.duration_s,end,max(2,int(samples*.25)))[1:]
    if second is None:
        time, state = on_time, first.sol(on_time)
    else:
        time = np.r_[on_time,off_time]
        state = np.column_stack((first.sol(on_time),second.sol(off_time)))
    last = knots[:,-1]
    energies = {n+"_J":float(v*k["energy_scale"]) for n,v in zip(names,last[2:])}
    storage = float((last[0]+last[1])*k["energy_scale"])
    residual = (energies["absorbed_pump_J"]-energies["quantum_defect_J"]
                -sum(energies[n+"_J"] for n in ("fluorescence","upward_output","downward_output","other_cavity_loss"))
                -(storage-initial_storage))
    boundary = energies["external_pump_J"]-sum(energies[n+"_J"] for n in ("before_gain_loss","absorbed_pump","escaped_pump","relay_loss"))
    threshold = uniform_top_pump_threshold(model,pulse)
    cw = uniform_top_cw_solution(model,pulse,power)
    last_on_fraction = float((first.sol(pulse.duration_s)[7+2]-first.sol(.9*pulse.duration_s)[7+2])
                             *k["energy_scale"]/(.1*pulse.duration_s))
    # The output accumulator has name index7. Integrated averages resolve fast
    # relaxation spikes more reliably than a sparse displayed time array.
    metrics = dict(**energies, stored_effective_energy_final_J=storage,
        stored_effective_energy_initial_J=initial_storage,
        energy_closure_residual_J=float(residual), pump_boundary_residual_J=float(boundary),
        relative_energy_residual=float(abs(residual)/max(pulse.external_energy_J,1e-30)),
        peak_upward_power_W=float(k["energy_scale"]*vg*up*knots[1].max()),
        pump_peak_external_W=float(power), last_tenth_on_average_upward_W=last_on_fraction,
        uniform_beta0_CW_upward_W=cw["upward_power_W"],
        final_inversion_fraction=float(last[0]), maximum_inversion_fraction=float(knots[0].max()),
        beta_spontaneous=beta, uniform_CW_threshold=threshold,
        rtol=rtol, integrator=method, accepted_step_count=int(knots.shape[1]),
        scope="Uniform-inversion, one fixed cold mode, rectangular top pump, effective excitation-energy audit; not spatial/multimode/thermal validation")
    return PulsedYbYAGResult(time,state[0],state[1]*k["ions"],
        state[1]*k["energy_scale"]*vg*up,metrics)
