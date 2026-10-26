"""Window-qualified polarization of the actual complex vector far field.

Intensity circularity is independent of polarization. The Stokes convention is
S3 = +2 Im(Ex Ey*); no right/left handedness label is attached. A single Jones
sample is coherent, whereas integration over angle can reduce the spatially
averaged degree of polarization without implying temporal depolarization.

中文：远场强度是否圆形与偏振是否圆偏振是两件事。统一采用 S3=+2 Im(Ex Ey*)，
不附会左右旋命名；角窗积分得到的是空间平均相干矩阵，不表示时间退偏振。
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _fields(field_x, field_y):
    fx, fy = np.asarray(field_x, dtype=complex), np.asarray(field_y, dtype=complex)
    if fx.ndim != 2 or fx.shape != fy.shape:
        raise ValueError("Two matching two-dimensional Jones fields are required")
    if not np.isfinite(fx).all() or not np.isfinite(fy).all():
        raise ValueError("Jones fields must be finite")
    scale = max(float(abs(fx).max()), float(abs(fy).max()))
    if scale <= 0:
        raise ValueError("A zero field has undefined polarization")
    return fx / scale, fy / scale


def _stokes(fx, fy):
    product = fx * fy.conj()
    return np.stack((abs(fx)**2 + abs(fy)**2,
                     abs(fx)**2 - abs(fy)**2,
                     2 * product.real, 2 * product.imag))


def _axis(axis):
    value = np.asarray(axis, dtype=float)
    if value.ndim != 1 or len(value) < 3 or not np.isfinite(value).all():
        raise ValueError("Angular axes require at least three finite samples")
    widths = np.diff(value)
    if np.any(widths <= 0) or not value[0] < 0 < value[-1]:
        raise ValueError("Angular axes must increase and contain the surface normal")
    return value, np.r_[widths[0]/2, (widths[:-1]+widths[1:])/2, widths[-1]/2]


def _summary(stokes):
    s0 = float(stokes[0])
    if s0 <= 1e-300:
        return None
    normalized = np.asarray(stokes[1:], dtype=float) / s0
    degree = float(np.linalg.norm(normalized))
    if degree > 1+1e-10:
        raise ValueError("Stokes vector violates positivity of the coherency matrix")
    linear = float(np.hypot(*normalized[:2]))
    circular = float(normalized[2])
    direction = None if linear < 1e-10 else float(
        np.rad2deg(.5*np.arctan2(normalized[1], normalized[0])))
    # Ellipticity angle describes only the polarized part of an angular mixture.
    # 此椭圆率角只描述角度混合中可偏振部分，不能与光斑二阶矩轴比混淆。
    ellipticity = None if degree < 1e-10 else float(
        np.rad2deg(.5*np.arcsin(np.clip(circular/degree, -1., 1.))))
    return {
        "normalized_stokes_s1_s2_s3": normalized.tolist(),
        "degree_of_polarization": min(degree, 1.),
        "degree_of_linear_polarization": min(linear, 1.),
        "signed_circular_fraction_s3_over_s0": circular,
        "absolute_circular_fraction": abs(circular),
        "linear_axis_angle_deg": direction,
        "polarized_ellipticity_angle_deg": ellipticity,
        "x_polarized_power_fraction": float((1+normalized[0])/2),
        "coherency_eigenvalue_fractions": [(1-min(degree, 1.))/2,
                                            (1+min(degree, 1.))/2],
    }


def polarization_metrics(angle_x_deg, field_x, field_y, angle_y_deg=None, *,
                         core_radius_deg=None):
    """Integrate one common Jones normalization with angular trapezoid weights.

    The normal direction is evaluated by interpolating Stokes intensities, not
    by coherently averaging fields at different observation directions. If zero
    is explicitly sampled, its exact Jones sample is used. Dark normal emission
    yields an undefined (None) on-axis polarization. Fractions refer only to the
    supplied window, using d(theta_x) d(theta_y) in the paraxial Cartesian basis.

    中文：两个 Jones 分量共用一次归一化，先形成 Stokes 强度再按角度梯形积分。
    轴上暗点的偏振记为 None；轴上、核心圆盘与整个角窗需分别报告。所有功率
    份额仅相对输入角窗，不应解释为完整半球的辐射功率占比。
    """
    x, wx = _axis(angle_x_deg)
    y, wy = _axis(x if angle_y_deg is None else angle_y_deg)
    fx, fy = _fields(field_x, field_y)
    if fx.shape != (len(y), len(x)):
        raise ValueError("Jones fields must match angular axes in (y,x) order")
    stokes = _stokes(fx, fy)
    weights = wy[:, None] * wx[None, :]
    total = np.sum(stokes * weights[None], axis=(1, 2))
    normal = np.asarray([RegularGridInterpolator((y, x), s)([[0., 0.]])[0]
                         for s in stokes])
    normal_relative_intensity = float(normal[0] / stokes[0].max())
    center = _summary(normal) if normal_relative_intensity > 1e-12 else None
    result = {
        "integrated": _summary(total),
        "on_axis": center,
        "on_axis_relative_to_peak_intensity": normal_relative_intensity,
        "angular_domain_x_deg": [float(x[0]), float(x[-1])],
        "angular_domain_y_deg": [float(y[0]), float(y[-1])],
        "samples_xy": [len(x), len(y)],
        "convention": "S0=|Ex|^2+|Ey|^2; S1=|Ex|^2-|Ey|^2; S2=2 Re(Ex Ey*); S3=+2 Im(Ex Ey*)",
        "integration_measure": "angular trapezoid d(theta_x)d(theta_y), paraxial Cartesian Jones basis",
        "scope": "evaluated angular window; spatially integrated coherency, not temporal or ensemble depolarization; no handedness label",
    }
    if core_radius_deg is not None:
        radius = float(core_radius_deg)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Core radius must be finite and positive")
        if radius > min(-x[0], x[-1], -y[0], y[-1])*(1+1e-12):
            raise ValueError("The polarization core must fit inside the evaluated window")
        xx, yy = np.meshgrid(x, y)
        core = xx**2+yy**2 <= radius**2
        value = np.sum(stokes * (weights*core)[None], axis=(1, 2))
        result["core"] = _summary(value)
        result["core_radius_deg"] = radius
        result["core_power_fraction_of_window"] = float(value[0] / total[0])
    return result


def stokes_maps(field_x, field_y, *, intensity_floor=.005):
    """Return finite maps plus the explicit mask of meaningful polarization.

    Values outside the relative-intensity mask are zero placeholders and must
    be masked in a figure. This avoids reporting a polarization axis in dark
    regions or storing NaN values in final scientific arrays.
    """
    if not 0 < intensity_floor < 1:
        raise ValueError("Relative intensity floor must lie strictly between zero and one")
    fx, fy = _fields(field_x, field_y)
    s = _stokes(fx, fy)
    intensity = s[0] / s[0].max()
    valid = intensity >= intensity_floor
    normalized = np.divide(s[1:], s[:1], out=np.zeros_like(s[1:]), where=valid[None])
    direction_valid = valid & (np.hypot(normalized[0], normalized[1]) > 1e-10)
    axis = np.where(direction_valid, .5*np.arctan2(normalized[1], normalized[0]), 0.)
    return {"relative_intensity": intensity, "valid_mask": valid,
            "s1_over_s0": normalized[0], "s2_over_s0": normalized[1],
            "s3_over_s0": normalized[2], "axis_angle_rad": axis,
            "axis_valid_mask": direction_valid}
