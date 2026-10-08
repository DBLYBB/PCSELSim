"""Jointly optimize hole shape, rotation, size, and position in a three-hole cell.

All candidates retain one main hole near the primitive-cell origin and one
satellite on each of the +a1 and +a2 axes.  Ellipses and exact-boundary regular
polygons (triangles or squares) share the same geometry-derived six-wave CWT.
The search has coarse, medium, and converged-final stages so a coarse-grid
ranking is not reported as a production result.

中文：同时扫描主孔/副孔的圆、椭圆、三角形、正方形，孔的位置、尺寸和旋转角。
多边形 Fourier 系数由边界解析积分，不使用像素化掩模，也不人为指定远场光斑。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .design_triangular_three_hole_pcsel import (
        DEVICE_RADIUS_UM,
        FINAL_RADIUS_CELLS,
        build_model,
        summarize_modes,
    )
else:
    from design_triangular_three_hole_pcsel import (
        DEVICE_RADIUS_UM,
        FINAL_RADIUS_CELLS,
        build_model,
        summarize_modes,
    )

from pcselsim.triangular_finite import (
    TriangularFiniteSpec,
    plot_triangular_far_field_diagnostics,
    plot_triangular_grid_convergence,
    plot_triangular_mode_atlas,
    plot_triangular_thresholds,
    solve_triangular_finite_modes,
    solve_triangular_finite_modes_converged,
    write_triangular_finite_table,
)
from pcselsim.triangular_six_wave import (
    TriangularCWTSettings,
    TriangularEllipse,
    TriangularLatticeCell,
    TriangularPolygon,
    plot_six_band_edge_states,
    plot_triangular_lattice,
)


OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "custom_semiconductor_triangular_three_hole_shape_position_optimized"
)
TARGET_FILL = 0.15
COARSE_TRUNCATION = 4
COARSE_VERTICAL_STEP_NM = 6.0
COARSE_RADIUS_CELLS = 4
MEDIUM_TRUNCATION = 7
MEDIUM_VERTICAL_STEP_NM = 4.0
MEDIUM_RADIUS_CELLS = 7
MEDIUM_OVERALL_FINALISTS = 4
FINAL_TRUNCATION = 10
FINAL_VERTICAL_STEP_NM = 3.0
CONVERGED_OVERALL_FINALISTS = 2


@dataclass(frozen=True)
class ShapeCandidate:
    name: str
    family: str
    explanation: str
    inclusions: tuple[TriangularEllipse | TriangularPolygon, ...]


def ellipse(
    center: tuple[float, float],
    area_radius: float,
    aspect: float = 1.0,
    angle_deg: float = 0.0,
) -> TriangularEllipse:
    root = np.sqrt(aspect)
    return TriangularEllipse(
        center_fractional=center,
        radii_over_a=(float(area_radius * root), float(area_radius / root)),
        angle_deg=angle_deg,
    )


def polygon(
    center: tuple[float, float],
    radius: float,
    sides: int,
    angle_deg: float,
) -> TriangularPolygon:
    return TriangularPolygon.regular(center, radius, sides, angle_deg)


def ellipse_radius_for_fill(fill: float) -> float:
    return float(np.sqrt(fill * np.sqrt(3.0) / (2.0 * np.pi)))


def ellipse_main_with_polygon_satellites(
    name: str,
    family: str,
    distance: float,
    satellite_radius: float,
    sides: int,
    angles: tuple[float, float],
    main_aspect: float = 1.15,
) -> ShapeCandidate:
    satellite_1 = polygon((distance, 0.0), satellite_radius, sides, angles[0])
    satellite_2 = polygon((0.0, distance), satellite_radius, sides, angles[1])
    main_fill = TARGET_FILL - satellite_1.fill_fraction - satellite_2.fill_fraction
    main_radius = ellipse_radius_for_fill(main_fill)
    return ShapeCandidate(
        name,
        family,
        "elliptical main hole with exact-boundary polygon satellites",
        (
            ellipse((0.0, 0.0), main_radius, main_aspect, 90.0),
            satellite_1,
            satellite_2,
        ),
    )


def polygon_main_with_ellipse_satellites(
    name: str,
    family: str,
    distance: float,
    main_radius: float,
    main_sides: int,
    main_angle: float,
    satellite_aspect: float = 1.4,
) -> ShapeCandidate:
    main = polygon((0.0, 0.0), main_radius, main_sides, main_angle)
    satellite_fill = 0.5 * (TARGET_FILL - main.fill_fraction)
    satellite_radius = ellipse_radius_for_fill(satellite_fill)
    return ShapeCandidate(
        name,
        family,
        "exact-boundary polygon main hole with radial elliptical satellites",
        (
            main,
            ellipse((distance, 0.0), satellite_radius, satellite_aspect, -30.0),
            ellipse((0.0, distance), satellite_radius, satellite_aspect, 30.0),
        ),
    )


def all_polygon_candidate(
    name: str,
    family: str,
    distance: float,
    satellite_radius: float,
    sides: int,
    main_angle: float,
    satellite_angles: tuple[float, float],
) -> ShapeCandidate:
    satellite_1 = polygon(
        (distance, 0.0), satellite_radius, sides, satellite_angles[0]
    )
    satellite_2 = polygon(
        (0.0, distance), satellite_radius, sides, satellite_angles[1]
    )
    main_fill = TARGET_FILL - satellite_1.fill_fraction - satellite_2.fill_fraction
    unit = polygon((0.0, 0.0), 1.0, sides, main_angle).fill_fraction
    main_radius = float(np.sqrt(main_fill / unit))
    return ShapeCandidate(
        name,
        family,
        "same regular polygon used for main and satellite holes",
        (
            polygon((0.0, 0.0), main_radius, sides, main_angle),
            satellite_1,
            satellite_2,
        ),
    )


def ellipse_candidate(
    name: str,
    distance_1: float,
    distance_2: float,
    satellite_radius: float,
    satellite_aspect: float,
    main_aspect: float,
) -> ShapeCandidate:
    satellite_fill = 2.0 * np.pi * satellite_radius**2 / (np.sqrt(3.0) / 2.0)
    main_radius = ellipse_radius_for_fill(TARGET_FILL - satellite_fill)
    return ShapeCandidate(
        name,
        "ellipse",
        "radial ellipses with jointly scanned positions and aspect ratios",
        (
            ellipse((0.0, 0.0), main_radius, main_aspect, 90.0),
            ellipse(
                (distance_1, 0.0), satellite_radius, satellite_aspect, -30.0
            ),
            ellipse(
                (0.0, distance_2), satellite_radius, satellite_aspect, 30.0
            ),
        ),
    )


def build_candidates() -> tuple[ShapeCandidate, ...]:
    candidates: list[ShapeCandidate] = []
    for distance, satellite_radius, satellite_aspect, main_aspect in (
        (0.30, 0.075, 1.3, 1.10),
        (0.32, 0.075, 1.4, 1.15),
        (0.32, 0.080, 1.2, 1.15),
        (0.32, 0.080, 1.4, 1.15),
        (0.32, 0.080, 1.7, 1.15),
        (0.34, 0.080, 1.4, 1.15),
        (0.36, 0.085, 1.4, 1.15),
        (0.34, 0.080, 1.4, 1.30),
    ):
        candidates.append(ellipse_candidate(
            f"ellipse_d{int(distance*100):02d}_r{int(satellite_radius*1000):03d}"
            f"_sa{int(satellite_aspect*10):02d}_ma{int(main_aspect*10):02d}",
            distance,
            distance,
            satellite_radius,
            satellite_aspect,
            main_aspect,
        ))
    candidates.append(ellipse_candidate(
        "ellipse_staggered_d030_d034", 0.30, 0.34, 0.08, 1.4, 1.15
    ))

    for distance, radius, orientation_name, angles in (
        (0.34, 0.095, "out", (-30.0, 30.0)),
        (0.34, 0.110, "out", (-30.0, 30.0)),
        (0.36, 0.120, "out", (-30.0, 30.0)),
        (0.38, 0.130, "out", (-30.0, 30.0)),
        (0.36, 0.110, "in", (150.0, -150.0)),
        (0.38, 0.120, "in", (150.0, -150.0)),
        (0.36, 0.110, "parallel", (90.0, 90.0)),
    ):
        candidates.append(ellipse_main_with_polygon_satellites(
            f"ellipse_main_triangle_sat_d{int(distance*100):02d}"
            f"_r{int(radius*1000):03d}_{orientation_name}",
            "triangle_satellites",
            distance,
            radius,
            3,
            angles,
        ))

    for distance, radius, angle in (
        (0.36, 0.18, 0.0),
        (0.38, 0.20, 0.0),
        (0.40, 0.22, 0.0),
        (0.38, 0.20, 30.0),
        (0.40, 0.22, 30.0),
        (0.40, 0.22, 90.0),
    ):
        candidates.append(polygon_main_with_ellipse_satellites(
            f"triangle_main_d{int(distance*100):02d}_r{int(radius*100):02d}"
            f"_a{int(angle):03d}",
            "triangle_main",
            distance,
            radius,
            3,
            angle,
        ))

    for distance, radius, main_angle, orientation_name, angles in (
        (0.42, 0.10, 0.0, "out", (-30.0, 30.0)),
        (0.43, 0.11, 30.0, "out", (-30.0, 30.0)),
        (0.45, 0.12, 90.0, "out", (-30.0, 30.0)),
        (0.43, 0.11, 30.0, "in", (150.0, -150.0)),
    ):
        candidates.append(all_polygon_candidate(
            f"all_triangle_d{int(distance*100):02d}_r{int(radius*100):02d}"
            f"_{orientation_name}_ma{int(main_angle):03d}",
            "all_triangle",
            distance,
            radius,
            3,
            main_angle,
            angles,
        ))

    for distance, radius, orientation_name, angles in (
        (0.34, 0.075, "radial", (-30.0, 30.0)),
        (0.36, 0.085, "radial", (-30.0, 30.0)),
        (0.38, 0.095, "diamond", (15.0, -15.0)),
    ):
        candidates.append(ellipse_main_with_polygon_satellites(
            f"ellipse_main_square_sat_d{int(distance*100):02d}"
            f"_r{int(radius*1000):03d}_{orientation_name}",
            "square_satellites",
            distance,
            radius,
            4,
            angles,
        ))

    candidates.append(polygon_main_with_ellipse_satellites(
        "square_main_d038_r018_a045",
        "square_main",
        0.38,
        0.18,
        4,
        45.0,
    ))
    return tuple(candidates)


CANDIDATES = build_candidates()


def _boundary(inclusion, samples: int = 48) -> np.ndarray:
    direct = np.asarray(
        ((np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0), (-0.5, 0.5))
    )
    if isinstance(inclusion, TriangularPolygon):
        return (direct @ np.asarray(inclusion.vertices_fractional).T).T
    center = direct @ np.asarray(inclusion.center_fractional)
    angle = np.deg2rad(inclusion.angle_deg)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    phase = 2.0 * np.pi * np.arange(samples) / samples
    local = np.vstack((
        inclusion.radii_over_a[0] * np.cos(phase),
        inclusion.radii_over_a[1] * np.sin(phase),
    ))
    return (center[:, None] + rotation @ local).T


def _convex_polygons_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    for polygon_points in (first, second):
        edges = np.roll(polygon_points, -1, axis=0) - polygon_points
        axes = np.column_stack((-edges[:, 1], edges[:, 0]))
        for axis in axes:
            norm = float(np.linalg.norm(axis))
            if norm <= np.finfo(float).eps:
                continue
            unit = axis / norm
            projection_1 = first @ unit
            projection_2 = second @ unit
            if projection_1.max() <= projection_2.min() + 1e-5 \
                    or projection_2.max() <= projection_1.min() + 1e-5:
                return False
    return True


def validate_nonoverlap(candidate: ShapeCandidate) -> None:
    """Reject overlaps inside the cell and with nearest periodic images."""
    a1 = np.asarray((np.sqrt(3.0) / 2.0, -0.5))
    a2 = np.asarray((np.sqrt(3.0) / 2.0, 0.5))
    boundaries = [_boundary(item) for item in candidate.inclusions]
    for left, first in enumerate(boundaries):
        for right, second in enumerate(boundaries):
            for shift_1 in (-1, 0, 1):
                for shift_2 in (-1, 0, 1):
                    if left == right and shift_1 == 0 and shift_2 == 0:
                        continue
                    shifted = second + shift_1 * a1 + shift_2 * a2
                    if _convex_polygons_overlap(first, shifted):
                        raise ValueError(
                            f"{candidate.name}: inclusions overlap across periodic cell"
                        )


def evaluate(
    candidate: ShapeCandidate,
    truncation: int,
    vertical_step_nm: float,
    radius_cells: int,
):
    cell, result = build_model(
        candidate,
        TriangularCWTSettings(
            truncation_order=truncation,
            vertical_step_nm=vertical_step_nm,
        ),
    )
    modes = solve_triangular_finite_modes(
        result,
        TriangularFiniteSpec(
            radius_um=DEVICE_RADIUS_UM,
            radius_cells=radius_cells,
            eigensolutions_per_target=7,
        ),
    )
    records, actual, dark_floor, threshold_gap, score = summarize_modes(result, modes)
    return cell, result, modes, records, actual, dark_floor, threshold_gap, score


def summary_row(candidate: ShapeCandidate, evaluation: tuple) -> dict:
    cell, _, _, _, actual, dark_floor, threshold_gap, score = evaluation
    return {
        "candidate": candidate.name,
        "family": candidate.family,
        "fill_fraction": cell.fill_fraction,
        "actual_lasing_mode": actual["mode"],
        "actual_threshold_cm-1": actual["threshold_cm-1"],
        "center_to_peak": actual["center_to_peak"],
        "peak_offset_deg": actual["peak_offset_deg"],
        "centroid_offset_deg": actual["centroid_offset_deg"],
        "ellipticity": actual["ellipticity"],
        "encircled_power_1deg": actual["encircled_power_1deg"],
        "full_rms_deg": actual["full_rms_deg"],
        "dark_competitor_floor_cm-1": dark_floor,
        "nearest_mode_gap_cm-1": threshold_gap,
        "score": score,
    }


def write_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    """Reload a completed stage so an interrupted long search can resume."""
    text_fields = {"candidate", "family", "actual_lasing_mode"}
    nullable_fields = {"dark_competitor_floor_cm-1"}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in tuple(row.items()):
            if key in text_fields:
                continue
            if key in nullable_fields and value == "":
                row[key] = None
            else:
                row[key] = float(value)
    return rows


def plot_stage(rows: list[dict], path: Path, title: str) -> None:
    ordered = sorted(rows, key=lambda row: row["score"], reverse=True)
    labels = [row["candidate"] for row in ordered]
    y = np.arange(len(ordered))
    height = max(6.0, 0.36 * len(ordered) + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, height), sharey=True)
    colors = {
        "ellipse": "#4c78a8",
        "triangle_satellites": "#f28e2b",
        "triangle_main": "#e15759",
        "all_triangle": "#b07aa1",
        "square_satellites": "#59a14f",
        "square_main": "#76b7b2",
    }
    axes[0].barh(
        y,
        [row["score"] for row in ordered],
        color=[colors[row["family"]] for row in ordered],
    )
    axes[0].set(
        xlabel="beam-quality design score",
        yticks=y,
        yticklabels=labels,
    )
    axes[0].invert_yaxis()
    axes[1].plot(
        [row["center_to_peak"] for row in ordered],
        y,
        "o-",
        label="center/max",
    )
    axes[1].plot(
        [row["encircled_power_1deg"] for row in ordered],
        y,
        "s-",
        label="power within 1 deg",
    )
    axes[1].set_xlabel("normalized beam metric")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def inclusion_record(inclusion) -> dict:
    if isinstance(inclusion, TriangularEllipse):
        return {
            "shape": "ellipse",
            "center_fractional": inclusion.center_fractional,
            "radii_over_a": inclusion.radii_over_a,
            "angle_deg": inclusion.angle_deg,
            "fill_fraction": inclusion.fill_fraction,
        }
    return {
        "shape": "polygon",
        "sides": len(inclusion.vertices_fractional),
        "vertices_fractional": inclusion.vertices_fractional,
        "fill_fraction": inclusion.fill_fraction,
    }


def main() -> None:
    output = OUTPUT_DIRECTORY.resolve()
    output.mkdir(parents=True, exist_ok=True)
    valid_candidates: list[ShapeCandidate] = []
    for candidate in CANDIDATES:
        try:
            validate_nonoverlap(candidate)
        except ValueError as error:
            print(f"  rejected geometry: {error}")
        else:
            valid_candidates.append(candidate)
    coarse_path = output / "01_coarse_screening.csv"
    if coarse_path.is_file():
        coarse_rows = read_rows(coarse_path)
    else:
        coarse_rows = []
    if len(coarse_rows) != len(valid_candidates):
        print(f"Coarse shape-position screening: {len(valid_candidates)} valid candidates")
        coarse_rows = []
        for index, candidate in enumerate(valid_candidates, start=1):
            print(f"  [{index}/{len(valid_candidates)}] {candidate.name} ...", flush=True)
            evaluation = evaluate(
                candidate,
                COARSE_TRUNCATION,
                COARSE_VERTICAL_STEP_NM,
                COARSE_RADIUS_CELLS,
            )
            row = summary_row(candidate, evaluation)
            coarse_rows.append(row)
            print(
                f"    {row['actual_lasing_mode']}: score={row['score']:.1f}, "
                f"center={row['center_to_peak']:.2f}, "
                f"offset={row['peak_offset_deg']:.2f} deg, "
                f"E1={row['encircled_power_1deg']:.2f}"
            )
        write_rows(coarse_rows, coarse_path)
        plot_stage(
            coarse_rows,
            output / "01_coarse_screening.png",
            "Coarse shape-position scan",
        )
    else:
        print(f"Reusing {len(coarse_rows)} completed coarse candidates")
    plot_stage(
        coarse_rows,
        output / "01_coarse_screening.png",
        "Coarse shape-position scan",
    )

    coarse_names = [
        row["candidate"]
        for row in sorted(coarse_rows, key=lambda item: item["score"], reverse=True)
        [:MEDIUM_OVERALL_FINALISTS]
    ]
    for family in sorted({row["family"] for row in coarse_rows}):
        family_best = max(
            (row for row in coarse_rows if row["family"] == family),
            key=lambda item: item["score"],
        )
        if family_best["candidate"] not in coarse_names:
            coarse_names.append(family_best["candidate"])
    medium_candidates = [
        candidate for name in coarse_names for candidate in valid_candidates
        if candidate.name == name
    ]
    medium_path = output / "02_medium_reranking.csv"
    medium_rows = read_rows(medium_path) if medium_path.is_file() else []
    if len(medium_rows) != len(medium_candidates):
        medium_rows = []
        print("Medium-grid reranking")
        for candidate in medium_candidates:
            print(f"  refining {candidate.name} ...", flush=True)
            evaluation = evaluate(
                candidate,
                MEDIUM_TRUNCATION,
                MEDIUM_VERTICAL_STEP_NM,
                MEDIUM_RADIUS_CELLS,
            )
            row = summary_row(candidate, evaluation)
            medium_rows.append(row)
            print(f"    score={row['score']:.1f}, mode={row['actual_lasing_mode']}")
        write_rows(medium_rows, medium_path)
    else:
        print(f"Reusing {len(medium_rows)} completed medium candidates")
    plot_stage(medium_rows, output / "02_medium_reranking.png", "Medium-grid reranking")

    final_names = [
        row["candidate"]
        for row in sorted(medium_rows, key=lambda item: item["score"], reverse=True)
        [:CONVERGED_OVERALL_FINALISTS]
    ]
    best_polygon = max(
        (row for row in medium_rows if row["family"] != "ellipse"),
        key=lambda item: item["score"],
    )
    if best_polygon["candidate"] not in final_names:
        final_names.append(best_polygon["candidate"])
    final_candidates = [
        candidate for name in final_names for candidate in valid_candidates
        if candidate.name == name
    ]
    final_rows: list[dict] = []
    final_data = {}
    print("Converged final comparison")
    for candidate in final_candidates:
        print(f"  converging {candidate.name} ...", flush=True)
        cell, result = build_model(
            candidate,
            TriangularCWTSettings(
                truncation_order=FINAL_TRUNCATION,
                vertical_step_nm=FINAL_VERTICAL_STEP_NM,
            ),
        )
        modes, convergence = solve_triangular_finite_modes_converged(
            result,
            TriangularFiniteSpec(
                radius_um=DEVICE_RADIUS_UM,
                radius_cells=FINAL_RADIUS_CELLS[-1],
                eigensolutions_per_target=8,
            ),
            FINAL_RADIUS_CELLS,
        )
        records, actual, dark_floor, threshold_gap, score = summarize_modes(result, modes)
        evaluation = (
            cell, result, modes, records, actual, dark_floor, threshold_gap, score
        )
        row = summary_row(candidate, evaluation)
        final_rows.append(row)
        final_data[candidate.name] = (candidate, evaluation, convergence)
        candidate_output = output / "finalists" / candidate.name
        candidate_output.mkdir(parents=True, exist_ok=True)
        plot_triangular_lattice(cell, candidate_output / "01_lattice.png")
        write_triangular_finite_table(
            modes, result, candidate_output / "02_modes.csv"
        )
        plot_triangular_grid_convergence(
            convergence,
            DEVICE_RADIUS_UM,
            candidate_output / "03_grid_convergence.png",
        )
        plot_triangular_far_field_diagnostics(
            modes[actual["mode"]],
            result,
            candidate_output / "04_lasing_mode_far_field.png",
        )
        print(
            f"    {actual['mode']}: alpha={actual['threshold_cm-1']:.2f}, "
            f"score={score:.1f}, center={actual['center_to_peak']:.2f}, "
            f"E1={actual['encircled_power_1deg']:.2f}"
        )
    write_rows(final_rows, output / "03_converged_finalists.csv")
    plot_stage(final_rows, output / "03_converged_finalists.png", "Converged finalists")

    winner_row = max(final_rows, key=lambda item: item["score"])
    winner, evaluation, convergence = final_data[winner_row["candidate"]]
    cell, result, modes, records, actual, dark_floor, threshold_gap, score = evaluation
    report = {
        "winner": winner.name,
        "family": winner.family,
        "explanation": winner.explanation,
        "fill_fraction": cell.fill_fraction,
        "inclusions": [inclusion_record(item) for item in winner.inclusions],
        "actual_lowest_threshold_mode": actual,
        "dark_competitor_floor_cm-1": dark_floor,
        "nearest_mode_gap_cm-1": threshold_gap,
        "final_score": score,
        "all_modes": records,
        "search_counts": {
            "defined": len(CANDIDATES),
            "valid_nonoverlapping": len(valid_candidates),
            "medium_finalists": len(medium_candidates),
            "converged_finalists": len(final_candidates),
        },
        "accuracy": {
            "coarse": [COARSE_TRUNCATION, COARSE_VERTICAL_STEP_NM, COARSE_RADIUS_CELLS],
            "medium": [MEDIUM_TRUNCATION, MEDIUM_VERTICAL_STEP_NM, MEDIUM_RADIUS_CELLS],
            "final_truncation": FINAL_TRUNCATION,
            "final_vertical_step_nm": FINAL_VERTICAL_STEP_NM,
            "final_radius_cells": FINAL_RADIUS_CELLS,
        },
        "warning": (
            "Reduced-order design search. Verify fabrication tolerances and "
            "full-wave RCWA/FEM/FDTD before fabrication."
        ),
    }
    (output / "04_winner.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    plot_triangular_lattice(cell, output / "04_winner_lattice.png")
    write_triangular_finite_table(modes, result, output / "04_winner_modes.csv")
    plot_triangular_thresholds(modes, output / "05_winner_thresholds.png")
    plot_triangular_grid_convergence(
        convergence, DEVICE_RADIUS_UM, output / "05_winner_grid_convergence.png"
    )
    plot_six_band_edge_states(result, output / "06_winner_unit_cell_states.png", points=101)
    plot_triangular_mode_atlas(modes, result, output / "07_winner_mode_atlas.png")
    plot_triangular_far_field_diagnostics(
        modes[actual["mode"]], result, output / "08_winner_lasing_mode_far_field.png"
    )
    print(
        f"winner={winner.name}, family={winner.family}, mode={actual['mode']}, "
        f"score={score:.2f}"
    )


if __name__ == "__main__":
    main()
