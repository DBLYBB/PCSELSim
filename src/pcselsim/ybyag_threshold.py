"""Small-signal pump thresholds of modes sharing one zero-photon Yb reservoir.

These are *unlased* gain crossings. The next branch crossing is not the pump at
which that branch turns on after the first laser clamps and burns the inversion.
Saturated turn-on requires the separate shared-inversion CW solver.

中文：所有候选模式共享同一个零光子反转分布，分别求小信号增益交点。
第二个交点不是激光起振后受增益钳位影响的第二模实际开启功率。
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import brentq
from .ybyag_multimode import _inversion, _arrays, multimode_spatial_rhs


@dataclass(frozen=True)
class SharedInversionThreshold:
    first_threshold_internal_W: float | None
    first_modes: tuple[str, ...]
    branch_thresholds_internal_W: dict[str, float | None]
    next_small_signal_threshold_internal_W: float | None
    next_to_first_ratio: float | None
    gain_to_loss_at_first: dict[str, float]
    threshold_residual: float | None
    searched_power_limit_W: float
    evaluations: int
    interpretation: str = "zero-photon shared inversion; later crossings are not saturated modal turn-on powers"


def shared_inversion_pump_threshold(system, pump_geometry, *, max_power_W=1e5,
                                     relative_tolerance=2e-9):
    """Root of max_j G_j[P,f_0(P)]/loss_j=1 at fixed geometry and ray layout.

    The pump argument must be a PumpGeometryConfig; its input power is replaced,
    while architecture, number of passes and relay efficiency remain fixed.
    Pump power counts all first-entry beams once. Nonbracketed branches return
    None and remain explicitly bounded by ``searched_power_limit_W``.

    功率单位为 W，计入进入增益区的全部首入射光束；它既不是吸收功率，
    也不是考虑外部耦合效率之前的实验泵浦源功率。未包根分支保留 None。
    """
    from .ybyag_pump_geometry import PumpGeometryConfig
    if not isinstance(pump_geometry, PumpGeometryConfig):
        raise TypeError("A PumpGeometryConfig with total first-entry power is required")
    if not np.isfinite(max_power_W) or max_power_W <= 0:
        raise ValueError("A positive finite search power limit is required")
    _, _, _, _, loss = _arrays(system, pump_geometry)
    zeros = np.zeros(len(system.modes))
    # Cold material-gain rank need not equal this spatially pumped first mode.
    # 冷腔材料增益排序不一定等于实际泵浦排序，必须逐模比较共享反转下的增益。
    cache = {}

    def ratios(power):
        key = float(power)
        if key not in cache:
            pump = replace(pump_geometry, internal_incident_power_W=key)
            fraction, _ = _inversion(system, pump, zeros)
            gain = multimode_spatial_rhs(system, pump, fraction, zeros)[2]
            cache[key] = gain/loss
        return cache[key]

    upper = min(1., max_power_W)
    while np.max(ratios(upper)) < 1 and upper < max_power_W:
        upper = min(upper*4, max_power_W)
    if np.max(ratios(upper)) < 1:
        first = None
    else:
        first = float(brentq(lambda p: np.max(ratios(p))-1, 0, upper,
                             xtol=1e-11, rtol=relative_tolerance))
    thresholds = {}
    for j, name in enumerate(system.names):
        ceiling = upper
        while ratios(ceiling)[j] < 1 and ceiling < max_power_W:
            ceiling = min(ceiling*4, max_power_W)
        thresholds[name] = (None if ratios(ceiling)[j] < 1 else
                            float(brentq(lambda p: ratios(p)[j]-1, 0, ceiling,
                                         xtol=1e-11, rtol=relative_tolerance)))
    finite = sorted(v for v in thresholds.values() if v is not None)
    # Retain the zero-photon meaning when reporting a next/first threshold gap.
    # 此处间隔仅描述零光子分支；饱和后模式竞争由独立的静态 CW 求解器处理。
    next_threshold = finite[1] if len(finite) > 1 else None
    at_first = ratios(first) if first is not None else None
    first_modes = (() if at_first is None else tuple(
        n for n, value in zip(system.names, at_first) if abs(value-1) <= 1e-6))
    return SharedInversionThreshold(first, first_modes, thresholds, next_threshold,
        next_threshold/first if first is not None and next_threshold is not None else None,
        {} if at_first is None else dict(zip(system.names, map(float, at_first))),
        None if at_first is None else float(np.max(at_first)-1), float(max_power_W), len(cache))
