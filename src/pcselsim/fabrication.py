"""Periodic geometry checks and process perturbations, in physical nanometres.

These are explicit assumed design rules, not a certificate from a foundry.
The caliper width measures the whole hole; sharp-tip fidelity is evaluated
separately by corner geometry. Systematic unit-cell errors are periodic and
must not be interpreted as random disorder across a million-cell device.
中文：用实际nm单位检查周期邻孔，不把光学加权评分当作加工合格证明。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from matplotlib.path import Path as PolygonPath


TRIANGULAR_DIRECT = np.asarray(((np.sqrt(3) / 2, np.sqrt(3) / 2), (-0.5, 0.5)))


@dataclass(frozen=True)
class FabricationRules:
    name: str = "research_assumption"
    minimum_gap_nm: float = 60.0
    minimum_caliper_nm: float = 65.0
    edge_bias_budget_nm: float = 3.0
    placement_budget_nm: float = 3.0
    minimum_corner_radius_nm: float = 0.0

    def __post_init__(self):
        if min(self.minimum_gap_nm, self.minimum_caliper_nm,
               self.edge_bias_budget_nm, self.placement_budget_nm, self.minimum_corner_radius_nm) < 0:
            raise ValueError("fabrication dimensions/budgets cannot be negative")


def inclusion_boundary(inclusion, direct: np.ndarray, samples: int = 96) -> np.ndarray:
    """Return Cartesian boundary in units of a, for square or triangular holes."""
    if hasattr(inclusion, "vertices_fractional"):
        return np.asarray(inclusion.vertices_fractional) @ direct.T
    if hasattr(inclusion, "vertices"):
        return np.asarray(inclusion.vertices)
    triangular = hasattr(inclusion, "radii_over_a")
    center = direct @ np.asarray(inclusion.center_fractional) if triangular else np.asarray(inclusion.center)
    radii = np.asarray(inclusion.radii_over_a if triangular else inclusion.radii)
    phase = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    theta = np.deg2rad(inclusion.angle_deg)
    rotation = np.asarray(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    return np.column_stack((np.cos(phase), np.sin(phase))) * radii @ rotation.T + center


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def polygon_gap(first: np.ndarray, second: np.ndarray) -> float:
    """Exact segment distance for sampled boundaries; zero includes overlap.

    Ellipses are inscribed polygons. Increase boundary samples to check the
    sub-nm chord error rather than treating polygon accuracy as process data.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if PolygonPath(first).contains_points(second).any() or PolygonPath(second).contains_points(first).any():
        return 0.0
    u, v = np.roll(first, -1, axis=0) - first, np.roll(second, -1, axis=0) - second
    w = second[None, :, :] - first[:, None, :]
    denominator = _cross(u[:, None, :], v[None, :, :])
    nonparallel = np.abs(denominator) > 1e-14
    safe = np.where(nonparallel, denominator, 1)
    t, s = _cross(w, v[None, :, :]) / safe, _cross(w, u[:, None, :]) / safe
    if np.any(nonparallel & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)):
        return 0.0

    def point_edges(points, starts, edges):
        delta = points[:, None, :] - starts[None, :, :]
        length2 = np.sum(edges**2, axis=1)
        if np.any(length2 <= 0):
            raise ValueError("boundary contains a zero-length edge")
        fraction = np.clip(np.sum(delta * edges[None, :, :], axis=2) / length2, 0, 1)
        residual = delta - fraction[..., None] * edges[None, :, :]
        return float(np.sqrt(np.min(np.sum(residual**2, axis=2))))

    return min(point_edges(first, second, v), point_edges(second, first, u))


def minimum_caliper(boundary: np.ndarray) -> float:
    """Minimum whole-hole width from edge normals of a convex boundary."""
    edges = np.roll(boundary, -1, axis=0) - boundary
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    projections = boundary @ normals.T
    return float(np.min(np.ptp(projections, axis=0)))


def sampled_corner_radius(boundary: np.ndarray) -> float:
    """Approximate minimum curvature radius; a three-vertex triangle is sharp.

    Polygon chord endpoints do not have true smooth curvature. This diagnostic
    is only meaningful for a sufficiently sampled smooth/rounded contour.
    """
    if len(boundary) <= 6:
        return 0.0
    prev, nxt = np.roll(boundary, 1, axis=0), np.roll(boundary, -1, axis=0)
    a = np.linalg.norm(boundary - prev, axis=1)
    b = np.linalg.norm(nxt - boundary, axis=1)
    c = np.linalg.norm(nxt - prev, axis=1)
    twice_area = np.abs(_cross(boundary - prev, nxt - prev))
    curved = twice_area > 1e-14
    return float(np.min((a * b * c)[curved] / (2 * twice_area[curved]))) if curved.any() else 0.0


def evaluate_fabrication(inclusions, lattice_nm: float, direct: np.ndarray | None = None,
                         rules: FabricationRules | None = None) -> dict:
    """Check all motifs and their nearest periodic neighbours, with margins."""
    if lattice_nm <= 0 or not inclusions:
        raise ValueError("positive lattice period and at least one inclusion required")
    direct = np.eye(2) if direct is None else np.asarray(direct, dtype=float)
    rules = rules or FabricationRules()
    boundaries = [inclusion_boundary(hole, direct) for hole in inclusions]
    # +/-2 covers the unit-cell motifs used here, including holes across edges.
    # Reject off-cell/unbounded motifs instead of claiming arbitrary image coverage.
    for boundary in boundaries:
        fractional = np.linalg.solve(direct, boundary.T).T
        if np.ptp(fractional, axis=0).max() >= 2 or np.abs(fractional).max() >= 2:
            raise ValueError("motif extends beyond supported periodic image search")
    gap = np.inf
    for i, first in enumerate(boundaries):
        for j in range(i, len(boundaries)):
            second = boundaries[j]
            for m in range(-2, 3):
                for n in range(-2, 3):
                    if i == j and m == n == 0:
                        continue
                    shifted = second + direct @ np.asarray((m, n))
                    # Bounding boxes give a rigorous cheap lower bound.
                    separation = np.maximum(np.maximum(first.min(0) - shifted.max(0),
                                                       shifted.min(0) - first.max(0)), 0)
                    if np.linalg.norm(separation) < gap:
                        gap = min(gap, polygon_gap(first, shifted))
    gap_nm = float(gap * lattice_nm)
    caliper_nm = min(minimum_caliper(p) for p in boundaries) * lattice_nm
    corner_nm = min(sampled_corner_radius(p) for p in boundaries) * lattice_nm
    # Two neighbouring edges and two independent placement errors can close a gap.
    gap_margin = gap_nm - 2 * rules.edge_bias_budget_nm - 2 * rules.placement_budget_nm
    width_margin = caliper_nm - 2 * rules.edge_bias_budget_nm
    return {
        "minimum_periodic_gap_nm": gap_nm,
        "minimum_caliper_nm": float(caliper_nm),
        "minimum_sampled_corner_radius_nm": corner_nm,
        "worst_case_gap_margin_nm": float(gap_margin),
        "worst_case_caliper_margin_nm": float(width_margin),
        "nominal_pass": bool(gap_nm >= rules.minimum_gap_nm and caliper_nm >= rules.minimum_caliper_nm
                             and corner_nm >= rules.minimum_corner_radius_nm),
        "margin_pass": bool(gap_margin >= rules.minimum_gap_nm and width_margin >= rules.minimum_caliper_nm
                            and corner_nm >= rules.minimum_corner_radius_nm),
        "rules": asdict(rules),
    }


def perturb_triangular_inclusions(inclusions, lattice_nm: float, rng: np.random.Generator,
                                  etch_bias_nm: float = 0, placement_sigma_nm: float = 0,
                                  scale_sigma_nm: float = 0, angle_sigma_deg: float = 0):
    """Perturb physical geometry without keeping fill artificially constant.

    Polygon etch offset is implemented by intersecting parallel offset edges
    of the convex sampled contour. Cell-wide identical perturbations model
    systematic errors, not spatially random disorder or scattering loss.
    """
    from .triangular_six_wave import TriangularEllipse, TriangularPolygon

    result = []
    for inclusion in inclusions:
        shift = rng.normal(0, placement_sigma_nm, 2) / lattice_nm
        angle = np.deg2rad(rng.normal(0, angle_sigma_deg))
        rotation = np.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
        radial_bias = (etch_bias_nm + rng.normal(0, scale_sigma_nm)) / lattice_nm
        if isinstance(inclusion, TriangularEllipse):
            radii = np.asarray(inclusion.radii_over_a) + radial_bias
            if radii.min() <= 0:
                raise ValueError("perturbed ellipse has nonpositive radius")
            center = TRIANGULAR_DIRECT @ np.asarray(inclusion.center_fractional) + shift
            result.append(replace(inclusion, center_fractional=tuple(np.linalg.solve(TRIANGULAR_DIRECT, center)),
                                  radii_over_a=tuple(radii), angle_deg=inclusion.angle_deg + np.rad2deg(angle)))
        elif isinstance(inclusion, TriangularPolygon):
            points = inclusion_boundary(inclusion, TRIANGULAR_DIRECT)
            center = points.mean(0)
            points = (points - center) @ rotation.T + center
            if np.sum(_cross(points, np.roll(points, -1, axis=0))) < 0:
                points = points[::-1]
            edge = np.roll(points, -1, axis=0) - points
            normal = np.column_stack((edge[:, 1], -edge[:, 0]))
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            offset = np.sum(normal * points, axis=1) + radial_bias
            # Half-plane clipping permits short edges to disappear during
            # shrink; intersecting adjacent offset lines alone was incorrect
            # when that legitimate change of contour topology occurs.
            extent = np.max(np.abs(points)) + abs(radial_bias) + 1
            shifted = np.asarray(((-extent, -extent), (extent, -extent),
                                  (extent, extent), (-extent, extent)))
            for axis, limit in zip(normal, offset, strict=True):
                clipped = []
                for start, stop in zip(shifted, np.roll(shifted, -1, axis=0), strict=True):
                    ds, dt = float(axis @ start-limit), float(axis @ stop-limit)
                    inside_start, inside_stop = ds <= 1e-12, dt <= 1e-12
                    if inside_start != inside_stop:
                        clipped.append(start + ds/(ds-dt)*(stop-start))
                    if inside_stop:
                        clipped.append(stop)
                shifted = np.asarray(clipped)
                if len(shifted) < 3:
                    raise ValueError("etch shrink eliminates polygon hole")
            # Avoid duplicate vertices created by a clipping line at a vertex.
            keep = np.linalg.norm(shifted - np.roll(shifted, 1, axis=0), axis=1) > 1e-10
            shifted = shifted[keep]
            fractional = np.linalg.solve(TRIANGULAR_DIRECT, (shifted + shift).T).T
            result.append(TriangularPolygon(tuple(map(tuple, fractional)), epsilon=inclusion.epsilon))
        else:
            raise TypeError("unsupported triangular inclusion")
    return tuple(result)


def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """Keep nondominated rows for objectives that are all minimized."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Pareto objectives must be a finite 2D array")
    return np.asarray([not np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1))
                       for row in values], dtype=bool)
