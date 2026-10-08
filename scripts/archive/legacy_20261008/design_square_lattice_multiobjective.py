"""Multi-objective geometry search for the square-lattice four-wave PCSEL.

This script independently rebuilds the square-cell Fourier coefficients,
vertical TE0 mode, C1D+Crad+C2D matrix, finite four-wave modes, and vector far
field for each candidate.  It does not reuse the triangular six-wave matrix.

中文：对方形 Bravais 晶格分别扫描单孔、双孔、三孔、十字多孔以及三角/方形混合孔形，
同时评价最低阈值、模式间隔、法线偏移、圆度和 0.5/1 度包围能量。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse as EllipsePatch, Polygon as PolygonPatch

if __package__:
    from .run_custom_semiconductor_pcsel import (
        BOTTOM_CLADDING_INDEX,
        DEVICE_SIZE_UM,
        INTERNAL_LOSS_CM,
        LATTICE_CONSTANT_NM,
        LAYERS,
        PC_LAYER_NAME,
        TOP_CLADDING_INDEX,
        VERTICAL_PADDING_UM,
        WAVELENGTH_GUESS_NM,
    )
else:
    from run_custom_semiconductor_pcsel import (
        BOTTOM_CLADDING_INDEX,
        DEVICE_SIZE_UM,
        INTERNAL_LOSS_CM,
        LATTICE_CONSTANT_NM,
        LAYERS,
        PC_LAYER_NAME,
        TOP_CLADDING_INDEX,
        VERTICAL_PADDING_UM,
        WAVELENGTH_GUESS_NM,
    )

from pcselsim.custom_analysis import (
    MODE_NAMES,
    FourWaveOpticalSpec,
    plot_grid_convergence,
    plot_threshold_summary,
    plot_vector_far_field_diagnostics,
    solve_finite_modes,
    solve_finite_modes_converged,
    vector_far_field_metrics,
    write_mode_table,
)
from pcselsim.geometry import Ellipse, PolygonInclusion, SquareLatticeCell
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from pcselsim.vertical import Layer, LayerStack


OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "results" / "custom_semiconductor_square_lattice_optimized"
)
BACKGROUND_EPSILON = 3.554**2
TARGET_FILL = 0.05
COARSE_SETTINGS = (4, 6.0, 9)
MEDIUM_SETTINGS = (7, 4.0, 13)
FINAL_TRUNCATION = 10
FINAL_VERTICAL_STEP_NM = 3.0
FINAL_GRIDS = (13, 17, 21)


@dataclass(frozen=True)
class SquareCandidate:
    name: str
    family: str
    explanation: str
    inclusions: tuple[Ellipse | PolygonInclusion, ...]


def ellipse_for_fill(
    center: tuple[float, float],
    fill: float,
    aspect: float = 1.0,
    angle_deg: float = 0.0,
) -> Ellipse:
    radius = np.sqrt(fill / np.pi)
    root = np.sqrt(aspect)
    return Ellipse(
        center=center,
        radii=(float(radius * root), float(radius / root)),
        angle_deg=angle_deg,
    )


def regular_polygon(
    center: tuple[float, float],
    fill: float,
    sides: int,
    angle_deg: float,
) -> PolygonInclusion:
    unit_area = 0.5 * sides * np.sin(2.0 * np.pi / sides)
    radius = np.sqrt(fill / unit_area)
    angles = np.deg2rad(angle_deg) + 2.0 * np.pi * np.arange(sides) / sides
    vertices = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    vertices += np.asarray(center)
    return PolygonInclusion(tuple((float(x), float(y)) for x, y in vertices))


def build_candidates() -> tuple[SquareCandidate, ...]:
    candidates: list[SquareCandidate] = [
        SquareCandidate(
            "single_circle",
            "single",
            "centered circular reference",
            (ellipse_for_fill((0.0, 0.0), TARGET_FILL),),
        )
    ]
    for aspect, angle in ((1.3, 0.0), (1.6, 0.0), (1.3, 45.0), (1.6, 45.0)):
        candidates.append(SquareCandidate(
            f"single_ellipse_a{int(aspect*10):02d}_r{int(angle):02d}",
            "single",
            "centered elliptical perturbation",
            (ellipse_for_fill((0.0, 0.0), TARGET_FILL, aspect, angle),),
        ))

    for axis_name, direction in (
        ("x", np.asarray((1.0, 0.0))),
        ("diag", np.asarray((1.0, 1.0)) / np.sqrt(2.0)),
    ):
        for separation in (0.22, 0.30, 0.38):
            centers = (-0.5 * separation * direction, 0.5 * separation * direction)
            inclusions = tuple(
                ellipse_for_fill(tuple(center), TARGET_FILL / 2.0, 1.5, 20.0 * sign)
                for center, sign in zip(centers, (1.0, -1.0), strict=True)
            )
            candidates.append(SquareCandidate(
                f"balanced_dimer_{axis_name}_d{int(separation*100):02d}",
                "dimer",
                "balanced counter-rotated equal-area ellipse dimer",
                inclusions,
            ))

    candidates.append(SquareCandidate(
        "inoue_like_asymmetric_dimer",
        "dimer",
        "idealized unequal diagonal double lattice",
        (
            Ellipse((-0.125, -0.125), (0.12, 0.075), -35.0),
            Ellipse((0.125, 0.125), (0.095, 0.060), -35.0),
        ),
    ))

    for distance, satellite_fill, aspect in (
        (0.20, 0.008, 1.3),
        (0.26, 0.010, 1.5),
        (0.32, 0.012, 1.7),
        (0.36, 0.010, 1.5),
    ):
        main_fill = TARGET_FILL - 2.0 * satellite_fill
        candidates.append(SquareCandidate(
            f"triple_xy_d{int(distance*100):02d}_sf{int(satellite_fill*1000):02d}",
            "triple",
            "centered main ellipse plus +x and +y radial satellites",
            (
                ellipse_for_fill((0.0, 0.0), main_fill, 1.15, 45.0),
                ellipse_for_fill((distance, 0.0), satellite_fill, aspect, 0.0),
                ellipse_for_fill((0.0, distance), satellite_fill, aspect, 90.0),
            ),
        ))

    for distance in (0.22, 0.30):
        satellite_fill = 0.006
        candidates.append(SquareCandidate(
            f"five_hole_cross_d{int(distance*100):02d}",
            "cross",
            "centered hole plus four balanced axial satellites",
            (
                ellipse_for_fill((0.0, 0.0), TARGET_FILL - 4.0 * satellite_fill),
                ellipse_for_fill((distance, 0.0), satellite_fill, 1.3, 0.0),
                ellipse_for_fill((-distance, 0.0), satellite_fill, 1.3, 0.0),
                ellipse_for_fill((0.0, distance), satellite_fill, 1.3, 90.0),
                ellipse_for_fill((0.0, -distance), satellite_fill, 1.3, 90.0),
            ),
        ))

    for angle in (0.0, 45.0, 90.0):
        candidates.append(SquareCandidate(
            f"triangle_main_a{int(angle):03d}",
            "triangle_main",
            "triangular main hole with two radial ellipse satellites",
            (
                regular_polygon((0.0, 0.0), 0.024, 3, angle),
                ellipse_for_fill((0.27, 0.0), 0.013, 1.4, 0.0),
                ellipse_for_fill((0.0, 0.27), 0.013, 1.4, 90.0),
            ),
        ))

    candidates.append(SquareCandidate(
        "ellipse_main_triangle_satellites",
        "triangle_satellites",
        "elliptical main hole plus +x/+y triangular satellites",
        (
            ellipse_for_fill((0.0, 0.0), 0.026, 1.15, 45.0),
            regular_polygon((0.29, 0.0), 0.012, 3, 0.0),
            regular_polygon((0.0, 0.29), 0.012, 3, 90.0),
        ),
    ))
    candidates.append(SquareCandidate(
        "square_main_ellipse_satellites",
        "square_main",
        "diamond main hole plus +x/+y ellipse satellites",
        (
            regular_polygon((0.0, 0.0), 0.024, 4, 45.0),
            ellipse_for_fill((0.28, 0.0), 0.013, 1.5, 0.0),
            ellipse_for_fill((0.0, 0.28), 0.013, 1.5, 90.0),
        ),
    ))
    candidates.append(SquareCandidate(
        "four_triangle_pinwheel",
        "polygon_cluster",
        "four rotated triangular holes around the cell center",
        tuple(
            regular_polygon(
                (0.18 * np.cos(angle), 0.18 * np.sin(angle)),
                TARGET_FILL / 4.0,
                3,
                np.rad2deg(angle) + 30.0,
            )
            for angle in np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
        ),
    ))
    return tuple(candidates)


CANDIDATES = build_candidates()


def boundary(inclusion, samples: int = 48) -> np.ndarray:
    if isinstance(inclusion, PolygonInclusion):
        return np.asarray(inclusion.vertices)
    phase = 2.0 * np.pi * np.arange(samples) / samples
    local = np.column_stack((
        inclusion.radii[0] * np.cos(phase),
        inclusion.radii[1] * np.sin(phase),
    ))
    angle = np.deg2rad(inclusion.angle_deg)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    return local @ rotation.T + np.asarray(inclusion.center)


def convex_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    for points in (first, second):
        edges = np.roll(points, -1, axis=0) - points
        axes = np.column_stack((-edges[:, 1], edges[:, 0]))
        for axis in axes:
            norm = float(np.linalg.norm(axis))
            if norm <= np.finfo(float).eps:
                continue
            unit = axis / norm
            p1, p2 = first @ unit, second @ unit
            if p1.max() <= p2.min() + 1e-5 or p2.max() <= p1.min() + 1e-5:
                return False
    return True


def validate_nonoverlap(candidate: SquareCandidate) -> None:
    boundaries = [boundary(item) for item in candidate.inclusions]
    for left, first in enumerate(boundaries):
        for right, second in enumerate(boundaries):
            for shift_x in (-1, 0, 1):
                for shift_y in (-1, 0, 1):
                    if left == right and shift_x == 0 and shift_y == 0:
                        continue
                    if convex_overlap(first, second + (shift_x, shift_y)):
                        raise ValueError(f"{candidate.name}: periodic inclusions overlap")


def build_model(candidate: SquareCandidate, truncation: int, vertical_step_nm: float):
    cell = SquareLatticeCell(BACKGROUND_EPSILON, candidate.inclusions)
    average_index = float(np.sqrt(cell.fourier_epsilon(0, 0).real))
    stack = LayerStack(
        layers=tuple(
            Layer(
                layer.name,
                layer.thickness_nm,
                average_index if layer.name == PC_LAYER_NAME else layer.refractive_index,
            )
            for layer in LAYERS
        ),
        top_index=TOP_CLADDING_INDEX,
        bottom_index=BOTTOM_CLADDING_INDEX,
        padding_um=VERTICAL_PADDING_UM,
    )
    result = build_geometry_coupling(
        cell,
        stack,
        PC_LAYER_NAME,
        LATTICE_CONSTANT_NM,
        WAVELENGTH_GUESS_NM,
        ThreeDCWTSettings(
            truncation_order=truncation,
            vertical_step_nm=vertical_step_nm,
        ),
    )
    eigen_cm = result.eigenvalues_m / 100.0
    spec = FourWaveOpticalSpec(
        wavelength_nm=result.bragg_wavelength_nm,
        effective_index=result.effective_index,
        group_index=result.group_index,
        domain_um=DEVICE_SIZE_UM,
        internal_loss_cm=INTERNAL_LOSS_CM,
        modal_detuning_cm=tuple(eigen_cm.real),
        modal_radiation_loss_cm=tuple(eigen_cm.imag),
        grid_points=FINAL_GRIDS[-1],
    )
    return cell, result, spec


def summarize(spec: FourWaveOpticalSpec, modes) -> tuple[list[dict], dict, float, float]:
    records = []
    for name in MODE_NAMES:
        mode = modes[name]
        far = vector_far_field_metrics(spec, mode)
        records.append({
            "mode": name,
            "threshold_cm-1": mode.alpha_per_m / 100.0,
            "alpha_L": mode.alpha_per_m * spec.domain_um * 1e-6,
            "center_to_peak": far.center_to_peak,
            "peak_offset_deg": far.peak_offset_deg,
            "centroid_offset_deg": far.centroid_offset_deg,
            "ellipticity": far.ellipticity,
            "encircled_power_0p5deg": far.encircled_power_0p5deg,
            "encircled_power_1deg": far.encircled_power_1deg,
            "full_rms_deg": far.full_rms_divergence_deg,
        })
    actual = min(records, key=lambda item: item["threshold_cm-1"])
    competitors = [item for item in records if item["mode"] != actual["mode"]]
    gap = min(item["threshold_cm-1"] for item in competitors) - actual["threshold_cm-1"]
    beam_score = (
        55.0 * actual["center_to_peak"]
        + 55.0 * actual["encircled_power_0p5deg"]
        + 15.0 * actual["encircled_power_1deg"]
        - 80.0 * actual["peak_offset_deg"]
        - 35.0 * actual["centroid_offset_deg"]
        - 15.0 * abs(np.log(max(actual["ellipticity"], 1e-12)))
        - 3.0 * actual["full_rms_deg"]
        + 0.25 * min(max(gap, -50.0), 50.0)
    )
    balanced_score = beam_score - 0.5 * actual["threshold_cm-1"]
    actual["mode_gap_cm-1"] = gap
    actual["beam_score"] = beam_score
    actual["balanced_score"] = balanced_score
    actual["acceptable"] = (
        actual["center_to_peak"] >= 0.75
        and actual["peak_offset_deg"] <= 0.25
        and actual["encircled_power_0p5deg"] >= 0.55
        and gap >= 1.0
    )
    return records, actual, beam_score, balanced_score


def include_internal_loss(spec: FourWaveOpticalSpec, modes):
    """Convert the shared solver's modal-loss convention to total loss.

    The common four-wave ``LinearMode.alpha`` intentionally contains vertical
    radiation plus lateral escape only because the time-domain threshold audit
    adds material loss separately.  This optimizer compares against the
    triangular total-loss tables, so add the uniform term exactly once here.
    """
    # alpha_in is a power attenuation coefficient; eigenvalue alpha is the
    # field-amplitude attenuation. The factor 1/2 follows Inoue Eq. (8).
    shift_per_m = 50.0 * spec.internal_loss_cm
    return {
        name: replace(mode, alpha_per_m=mode.alpha_per_m + shift_per_m)
        for name, mode in modes.items()
    }


def evaluate(candidate: SquareCandidate, settings: tuple[int, float, int]):
    cell, result, spec = build_model(candidate, settings[0], settings[1])
    modes = solve_finite_modes(
        FourWaveOpticalSpec(**{**spec.__dict__, "grid_points": settings[2]}),
        result.coupling_m,
        result.eigenvectors,
        result.radiation_fields,
    )
    modes = include_internal_loss(spec, modes)
    return cell, result, spec, modes, summarize(spec, modes)


def row(candidate: SquareCandidate, evaluation: tuple) -> dict:
    cell, _, _, _, summary = evaluation
    _, actual, beam_score, balanced_score = summary
    fill = 0.0
    for inclusion in candidate.inclusions:
        if isinstance(inclusion, Ellipse):
            fill += float(np.pi * inclusion.radii[0] * inclusion.radii[1])
        else:
            vertices = np.asarray(inclusion.vertices)
            fill += float(abs(0.5 * np.sum(
                vertices[:, 0] * np.roll(vertices[:, 1], -1)
                - np.roll(vertices[:, 0], -1) * vertices[:, 1]
            )))
    return {
        "candidate": candidate.name,
        "family": candidate.family,
        "fill_fraction": fill,
        **actual,
        "beam_score": beam_score,
        "balanced_score": balanced_score,
    }


def write_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    """Reload a completed search stage so an interrupted long run can resume."""
    numeric = {
        "fill_fraction", "threshold_cm-1", "alpha_L", "center_to_peak",
        "peak_offset_deg", "centroid_offset_deg", "ellipticity",
        "encircled_power_0p5deg", "encircled_power_1deg", "full_rms_deg",
        "mode_gap_cm-1", "beam_score", "balanced_score",
    }
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for item in rows:
        for key in numeric:
            item[key] = float(item[key])
        item["acceptable"] = item["acceptable"].strip().lower() == "true"
    return rows


def plot_tradeoff(rows: list[dict], path: Path, title: str) -> None:
    colors = {
        family: color for family, color in zip(
            sorted({item["family"] for item in rows}),
            plt.cm.tab10.colors,
            strict=False,
        )
    }
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    for item in rows:
        ax.scatter(
            item["threshold_cm-1"],
            item["beam_score"],
            color=colors[item["family"]],
            s=40 + 100 * item["encircled_power_0p5deg"],
            alpha=0.8,
        )
    for item in sorted(rows, key=lambda value: value["balanced_score"], reverse=True)[:5]:
        ax.annotate(
            item["candidate"],
            (item["threshold_cm-1"], item["beam_score"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set(
        xlabel="lowest finite threshold (cm$^{-1}$)",
        ylabel="beam-quality score",
        title=title,
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_cell(candidate: SquareCandidate, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            for inclusion in candidate.inclusions:
                if isinstance(inclusion, Ellipse):
                    ax.add_patch(EllipsePatch(
                        (ix + inclusion.center[0], iy + inclusion.center[1]),
                        2.0 * inclusion.radii[0],
                        2.0 * inclusion.radii[1],
                        angle=inclusion.angle_deg,
                        fc="white",
                        ec="black",
                    ))
                else:
                    vertices = np.asarray(inclusion.vertices) + (ix, iy)
                    ax.add_patch(PolygonPatch(vertices, closed=True, fc="white", ec="black"))
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False, ec="red", lw=2.0))
    ax.set(
        aspect="equal",
        xlim=(-2.5, 2.5),
        ylim=(-2.5, 2.5),
        xlabel="x/a",
        ylabel="y/a",
        title=f"Square-lattice candidate: {candidate.name}",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_device_envelope(spec: FourWaveOpticalSpec, mode, path: Path) -> None:
    """Plot the actual finite-area eigenmode envelope used for the far field."""
    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    image = ax.imshow(
        mode.intensity,
        origin="lower",
        cmap="turbo",
        extent=(0.0, spec.domain_um, 0.0, spec.domain_um),
    )
    fig.colorbar(image, ax=ax, label="normalized optical intensity")
    ax.set(
        xlabel="x (um)",
        ylabel="y (um)",
        title=f"Mode {mode.name}: whole-device finite-area envelope",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    output = OUTPUT_DIRECTORY.resolve()
    output.mkdir(parents=True, exist_ok=True)
    force_recompute = os.environ.get("PCSELSIM_FORCE_RECOMPUTE", "0") == "1"
    valid = []
    for candidate in CANDIDATES:
        try:
            validate_nonoverlap(candidate)
        except ValueError as error:
            print(f"  rejected {error}")
        else:
            valid.append(candidate)
    print(f"Square-lattice coarse screening: {len(valid)} candidates")
    coarse_path = output / "01_coarse.csv"
    if coarse_path.exists() and not force_recompute:
        print("  reusing completed coarse stage", flush=True)
        coarse_rows = read_rows(coarse_path)
    else:
        coarse_rows = []
        for index, candidate in enumerate(valid, start=1):
            print(f"  [{index}/{len(valid)}] {candidate.name} ...", flush=True)
            evaluation = evaluate(candidate, COARSE_SETTINGS)
            item = row(candidate, evaluation)
            coarse_rows.append(item)
            print(
                f"    {item['mode']}: alpha={item['threshold_cm-1']:.2f}, "
                f"beam={item['beam_score']:.1f}, center={item['center_to_peak']:.2f}"
            )
        write_rows(coarse_rows, coarse_path)
        plot_tradeoff(coarse_rows, output / "01_coarse_tradeoff.png", "Square-lattice coarse search")

    names = [
        item["candidate"]
        for item in sorted(coarse_rows, key=lambda value: value["balanced_score"], reverse=True)[:5]
    ]
    for family in sorted({item["family"] for item in coarse_rows}):
        best = max(
            (item for item in coarse_rows if item["family"] == family),
            key=lambda value: value["balanced_score"],
        )
        if best["candidate"] not in names:
            names.append(best["candidate"])
    lookup = {candidate.name: candidate for candidate in valid}
    medium_path = output / "02_medium.csv"
    if medium_path.exists() and not force_recompute:
        print("  reusing completed medium stage", flush=True)
        medium_rows = read_rows(medium_path)
    else:
        medium_rows = []
        print(f"Square-lattice medium reranking: {len(names)} candidates")
        for name in names:
            print(f"  refining {name} ...", flush=True)
            evaluation = evaluate(lookup[name], MEDIUM_SETTINGS)
            medium_rows.append(row(lookup[name], evaluation))
        write_rows(medium_rows, medium_path)
        plot_tradeoff(medium_rows, output / "02_medium_tradeoff.png", "Square-lattice medium search")

    acceptable = [item for item in medium_rows if item["acceptable"]]
    category_seed = {
        "beam_quality": max(medium_rows, key=lambda value: value["beam_score"])["candidate"],
        "balanced": max(medium_rows, key=lambda value: value["balanced_score"])["candidate"],
        "low_threshold": min(acceptable, key=lambda value: value["threshold_cm-1"])["candidate"],
    }
    final_names = list(dict.fromkeys(category_seed.values()))
    final_rows = []
    final_data = {}
    print(f"Square-lattice converged comparison: {len(final_names)} candidates")
    for name in final_names:
        candidate = lookup[name]
        print(f"  converging {name} ...", flush=True)
        cell, result, spec = build_model(
            candidate, FINAL_TRUNCATION, FINAL_VERTICAL_STEP_NM
        )
        modes, convergence = solve_finite_modes_converged(
            spec,
            FINAL_GRIDS,
            result.coupling_m,
            result.eigenvectors,
            result.radiation_fields,
        )
        modes = include_internal_loss(spec, modes)
        for mode_name in MODE_NAMES:
            convergence[mode_name] = (
                convergence[mode_name] + 0.5j * 100.0 * spec.internal_loss_cm
            )
        summary = summarize(spec, modes)
        evaluation = (cell, result, spec, modes, summary)
        item = row(candidate, evaluation)
        final_rows.append(item)
        final_data[name] = evaluation
        candidate_output = output / "finalists" / name
        candidate_output.mkdir(parents=True, exist_ok=True)
        plot_cell(candidate, candidate_output / "01_lattice.png")
        write_mode_table(candidate_output, spec, modes)
        plot_grid_convergence(spec, convergence, candidate_output / "03_convergence.png")
        plot_threshold_summary(spec, modes, candidate_output / "04_thresholds.png")
        plot_vector_far_field_diagnostics(
            spec, modes[item["mode"]], candidate_output / "05_far_field.png"
        )
        plot_device_envelope(
            spec, modes[item["mode"]], candidate_output / "06_device_envelope.png"
        )
        print(
            f"    mode={item['mode']}, alpha={item['threshold_cm-1']:.3f}, "
            f"beam={item['beam_score']:.1f}, gap={item['mode_gap_cm-1']:.3f}"
        )
    write_rows(final_rows, output / "03_final.csv")
    plot_tradeoff(final_rows, output / "03_final_tradeoff.png", "Square-lattice final search")

    eligible = [item for item in final_rows if item["acceptable"]]
    categories = {
        "beam_quality": max(eligible, key=lambda value: value["beam_score"]),
        "balanced": max(eligible, key=lambda value: value["balanced_score"]),
        "low_threshold": min(eligible, key=lambda value: value["threshold_cm-1"]),
        # Retain the mathematical lower bound, but do not present a nearly
        # degenerate design as the engineering recommendation.
        "numerical_lowest_including_unstable": min(
            final_rows, key=lambda value: value["threshold_cm-1"]
        ),
    }
    report = {
        "recommendations": categories,
        "finalists": final_rows,
        "search_counts": {
            "defined": len(CANDIDATES),
            "valid": len(valid),
            "medium": len(names),
            "final": len(final_names),
        },
        "warning": (
            "Square four-wave reduced-order search. Verify larger finite grids, "
            "fabrication tolerances, and RCWA/FEM/FDTD before fabrication."
        ),
    }
    (output / "04_recommendations.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print({key: value["candidate"] for key, value in categories.items()})


if __name__ == "__main__":
    main()
