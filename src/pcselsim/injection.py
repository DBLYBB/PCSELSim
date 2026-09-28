"""Electrical current-injection profiles."""

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

