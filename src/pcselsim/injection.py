"""Normalized electrical current-injection profiles.

The returned profile integrates to one, so multiplying it by total current
produces current density in A/m^2.  Error-function spreading is an empirical
lateral smoothing model, not a semiconductor drift--diffusion calculation.

返回的二维分布积分恒为 1，乘以总电流后就是 A/m^2 的电流密度。误差函数仅
表示经验性的横向电流扩展，并不等价于完整的电学漂移—扩散求解。
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from .config import DeviceConfig


def square_electrode_profile(x_m: np.ndarray, y_m: np.ndarray, device: DeviceConfig) -> np.ndarray:
    """Gaussian-smoothed square electrode, normalized to unit area integral."""
    half = 0.5 * device.electrode_um * 1e-6
    sigma = device.current_spread_um * 1e-6
    root2sigma = np.sqrt(2.0) * sigma

    def blurred_interval(values: np.ndarray) -> np.ndarray:
        if sigma <= 0.0:
            return (np.abs(values) <= half).astype(float)
        return 0.5 * (erf((values + half) / root2sigma) - erf((values - half) / root2sigma))

    profile = blurred_interval(x_m)[None, :] * blurred_interval(y_m)[:, None]
    dx = float(x_m[1] - x_m[0])
    dy = float(y_m[1] - y_m[0])
    integral = float(np.sum(profile) * dx * dy)
    if integral <= 0.0:
        raise ValueError("Current profile has zero integral")
    return profile / integral


def circle_electrode_profile(x_m: np.ndarray, y_m: np.ndarray, device: DeviceConfig) -> np.ndarray:
    """Radially smoothed circular contact, normalized to unit area integral.

    ``electrode_um`` is the contact diameter.  A positive ``current_spread_um``
    applies an error-function transition at the nominal contact radius.  This
    is a compact radial approximation to lateral current spreading; it is not
    an electrical drift-diffusion solution.
    """
    radius = 0.5 * device.electrode_um * 1e-6
    sigma = device.current_spread_um * 1e-6
    xx, yy = np.meshgrid(x_m, y_m)
    radial = np.hypot(xx, yy)
    if sigma <= 0.0:
        profile = (radial <= radius).astype(float)
    else:
        profile = 0.5 * (1.0 - erf((radial - radius) / (np.sqrt(2.0) * sigma)))
    dx = float(x_m[1] - x_m[0])
    dy = float(y_m[1] - y_m[0])
    integral = float(np.sum(profile) * dx * dy)
    if integral <= 0.0:
        raise ValueError("Current profile has zero integral")
    return profile / integral


def electrode_area_m2(device: DeviceConfig) -> float:
    """Return the nominal electrical contact area in square metres."""
    width_m = device.electrode_um * 1e-6
    if device.electrode_shape == "square":
        return width_m**2
    if device.electrode_shape == "circle":
        return np.pi * (0.5 * width_m)**2
    raise ValueError(f"Unsupported electrode shape: {device.electrode_shape}")


def electrode_profile(x_m: np.ndarray, y_m: np.ndarray, device: DeviceConfig) -> np.ndarray:
    """Dispatch to the configured normalized electrical contact profile."""
    if device.electrode_shape == "square":
        return square_electrode_profile(x_m, y_m, device)
    if device.electrode_shape == "circle":
        return circle_electrode_profile(x_m, y_m, device)
    raise ValueError(f"Unsupported electrode shape: {device.electrode_shape}")

