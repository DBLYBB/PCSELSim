"""Shared, window-qualified objectives for PCSEL design exploration.

Circular intensity is distinct from circular polarization. A second-moment
ellipse alone misses four/six lobes and a rotationally symmetric doughnut, so
we also retain center brightness, half-maximum connectivity and polar residuals.
All fractions refer only to the evaluated angular window, never a hemisphere.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label


def _weights(axis: np.ndarray) -> np.ndarray:
    widths = np.diff(axis)
    return np.r_[widths[0] / 2, (widths[:-1] + widths[1:]) / 2, widths[-1] / 2]


def beam_metrics(angle_x_deg, field_x, field_y, angle_y_deg=None, *,
                 requested_view_deg=1.0, source_nyquist_deg=None,
                 radial_samples=80, azimuth_samples=180) -> dict:
    """Evaluate a Cartesian complex vector far field with angular quadrature.

    Fields have shape (len(y),len(x)); axes must increase. ``source_nyquist_deg``
    describes the original *envelope* mesh, not zero padding or interpolation.
    Passing None explicitly leaves that sampling check unverified.
    ``radial_anisotropy_l2`` is the relative polar intensity residual inside
    the inscribed observation circle. A circular Gaussian gives zero, whereas
    a four/six-lobed pattern need not, even if its moment ellipse is unity.
    """
    x = np.asarray(angle_x_deg, dtype=float)
    y = x if angle_y_deg is None else np.asarray(angle_y_deg, dtype=float)
    fx, fy = np.asarray(field_x, complex), np.asarray(field_y, complex)
    for axis in (x, y):
        if axis.ndim != 1 or len(axis) < 5 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError("Far-field axes require >=5 finite, increasing samples")
        if not axis[0] < 0 < axis[-1]:
            raise ValueError("Observation window must include the surface normal")
    if not np.isfinite(requested_view_deg) or requested_view_deg <= 0:
        raise ValueError("Requested angular half-window must be finite and positive")
    if fx.shape != (len(y), len(x)) or fy.shape != fx.shape:
        raise ValueError("Complex vector fields must match the two angular axes")
    if not np.isfinite(fx).all() or not np.isfinite(fy).all():
        raise ValueError("Far fields must be finite")
    scale = max(float(np.abs(fx).max()), float(np.abs(fy).max()))
    if scale <= 0:
        raise ValueError("A zero-radiation mode has no defined normalized beam")
    intensity = abs(fx / scale)**2 + abs(fy / scale)**2
    intensity /= intensity.max()
    xx, yy = np.meshgrid(x, y)
    weights = _weights(y)[:, None] * _weights(x)[None, :]
    mass = weights * intensity
    total = float(mass.sum())
    cx, cy = float((xx * mass).sum() / total), float((yy * mass).sum() / total)
    dx, dy = xx - cx, yy - cy
    covariance = np.array([[(dx*dx*mass).sum(), (dx*dy*mass).sum()],
                           [(dx*dy*mass).sum(), (dy*dy*mass).sum()]]) / total
    variances = np.maximum(np.linalg.eigvalsh(covariance), 0.)
    ellipse = float(np.sqrt(variances[-1] / max(variances[0], np.finfo(float).tiny)))
    radius = np.hypot(xx, yy)
    order = np.argsort(radius.ravel())
    cumulative = np.cumsum(mass.ravel()[order]) / total
    radii = radius.ravel()[order]
    r_quantile = lambda fraction: float(radii[min(np.searchsorted(cumulative, fraction), len(radii)-1)])
    interpolator = RegularGridInterpolator((y, x), intensity, bounds_error=True)
    center = float(interpolator([[0., 0.]])[0])
    peak_row, peak_col = np.unravel_index(np.argmax(intensity), intensity.shape)
    labels, components = label(intensity >= .5, structure=np.ones((3, 3)))
    origin_row, origin_col = int(np.argmin(abs(y))), int(np.argmin(abs(x)))
    central_label = int(labels[origin_row, origin_col])
    circular_radius = float(min(-x[0], x[-1], -y[0], y[-1]))
    radial = np.linspace(0., circular_radius * (1 - 1e-12), radial_samples)
    phi = np.linspace(0., 2*np.pi, azimuth_samples, endpoint=False)
    polar = interpolator(np.stack((radial[:, None]*np.sin(phi),
                                   radial[:, None]*np.cos(phi)), axis=-1))
    radial_mean = polar.mean(axis=1, keepdims=True)
    polar_denominator = float(np.trapz(np.mean(polar**2, axis=1)*radial, radial))
    anisotropy = float(np.sqrt(np.trapz(np.mean((polar-radial_mean)**2, axis=1)*radial, radial)
                              / max(polar_denominator, 1e-300)))
    harmonic = lambda m: float(np.sqrt(np.trapz(abs(np.mean(polar*np.exp(-1j*m*phi), axis=1))**2 * radial,
                                               radial) / max(polar_denominator, 1e-300)))
    edge = ((xx < x[0]*.9) | (xx > x[-1]*.9) | (yy < y[0]*.9) | (yy > y[-1]*.9))
    evaluated = float(max(abs(x[0]), abs(x[-1]), abs(y[0]), abs(y[-1])))
    if source_nyquist_deg is not None and (not np.isfinite(source_nyquist_deg) or source_nyquist_deg <= 0):
        raise ValueError("Source Nyquist angle must be finite and positive")
    edge_spacing = max(float(np.diff(axis)[[0, -1]].max()) for axis in (x, y))
    window_covered = bool(circular_radius + edge_spacing >= requested_view_deg*(1-1e-10))
    sampling_checked = None if source_nyquist_deg is None else bool(
        evaluated <= source_nyquist_deg*(1+1e-10)
        and requested_view_deg <= source_nyquist_deg*(1+1e-10) and window_covered)
    return {
        "ellipticity": ellipse, "center_to_peak": center,
        "centroid_offset_deg": float(np.hypot(cx, cy)),
        "peak_offset_deg": float(radius[peak_row, peak_col]),
        "full_rms_diameter_deg": float(2*np.sqrt((radius**2*mass).sum()/total)),
        "r50_deg": r_quantile(.5), "r80_deg": r_quantile(.8), "r90_deg": r_quantile(.9),
        "encircled_0p2deg": float(mass[radius <= .2].sum()/total),
        "encircled_0p5deg": float(mass[radius <= .5].sum()/total),
        "radial_anisotropy_l2": anisotropy,
        "fourfold_harmonic_l2": harmonic(4), "sixfold_harmonic_l2": harmonic(6),
        "half_max_components": int(components),
        "surface_normal_in_half_max_component": bool(central_label > 0),
        "edge_window_power_fraction": float(mass[edge].sum()/total),
        "requested_half_window_deg": float(requested_view_deg),
        "evaluated_x_deg": [float(x[0]), float(x[-1])],
        "evaluated_y_deg": [float(y[0]), float(y[-1])],
        "samples_xy": [len(x), len(y)], "inscribed_polar_radius_deg": circular_radius,
        "requested_window_covered_by_samples": window_covered,
        "source_nyquist_deg": None if source_nyquist_deg is None else float(source_nyquist_deg),
        "window_resolved_by_envelope_sampling": sampling_checked,
        "normalization_domain": "evaluated Cartesian angular window; intensity |Ex|^2+|Ey|^2",
        "circularity_definition": "centered intensity, not circular polarization; polar relative L2 residual",
    }


def beam_acceptance(metrics: dict, *, maximum_ellipticity=1.10,
                    minimum_center_to_peak=.90, maximum_radial_anisotropy=.15,
                    require_sampling=False) -> dict:
    """Explicit engineering targets, not experimental beam certification."""
    conditions = {
        "moment_round": metrics["ellipticity"] <= maximum_ellipticity,
        "bright_center": metrics["center_to_peak"] >= minimum_center_to_peak,
        "radially_round": metrics["radial_anisotropy_l2"] <= maximum_radial_anisotropy,
        "one_half_max_component": metrics["half_max_components"] == 1,
        "center_in_main_component": metrics["surface_normal_in_half_max_component"],
        "requested_window_covered": metrics["requested_window_covered_by_samples"],
    }
    if require_sampling:
        conditions["sampling_verified"] = metrics["window_resolved_by_envelope_sampling"] is True
    return {"accepted": bool(all(conditions.values())), "conditions": conditions,
            "targets": {"maximum_ellipticity": maximum_ellipticity,
                        "minimum_center_to_peak": minimum_center_to_peak,
                        "maximum_radial_anisotropy": maximum_radial_anisotropy}}
