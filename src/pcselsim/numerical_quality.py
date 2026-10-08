"""Conservative diagnostics for first-order finite-device grid extrapolation.

These checks estimate sensitivity to the chosen grids, not a statistical
confidence interval.  An extrapolation never proves that the eigenfield or
its far-field moments have converged. / 检查拟合网格敏感性，不伪称统计置信区间。
"""

from __future__ import annotations

import numpy as np


def extrapolate_grid_loss(
    inverse_grid: np.ndarray, losses: np.ndarray
) -> tuple[float, float, float, str]:
    """Return accepted loss, raw intercept, empirical sensitivity and status.

    A materially negative intercept is rejected and the finest passive-grid
    value is returned, rather than silently reporting a zero-loss cavity.
    """
    inverse_grid = np.asarray(inverse_grid, dtype=float)
    losses = np.asarray(losses, dtype=float)
    fit = np.polyfit(inverse_grid, losses, 1)
    raw = float(fit[1])
    last_pair = float(np.polyfit(inverse_grid[-2:], losses[-2:], 1)[1])
    residual = float(np.max(np.abs(np.polyval(fit, inverse_grid) - losses)))
    sensitivity = max(abs(raw - last_pair), residual)
    if len(losses) > 3:
        last_three = float(np.polyfit(inverse_grid[-3:], losses[-3:], 1)[1])
        sensitivity = max(sensitivity, abs(raw - last_three))
    numerical_floor = 1e-10 * max(float(np.max(np.abs(losses))), 1.0)
    if raw < -numerical_floor:
        return float(losses[-1]), raw, sensitivity, "rejected_negative_intercept"
    accepted = max(raw, 0.0)  # Only a roundoff-scale negative number reaches here.
    scale = max(accepted, numerical_floor)
    status = "provisional" if sensitivity <= 0.25 * scale else "grid_sensitive"
    return accepted, raw, sensitivity, status
