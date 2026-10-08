"""Refine the low-threshold triangular-main-hole six-wave design.

The search varies main-triangle size/rotation, satellite distance and angular
spread, plus elliptical or triangular satellite shape/orientation.  Candidates
are ranked by both beam quality and cold-cavity threshold; final output keeps
separate beam, balanced, and low-threshold recommendations.

中文：围绕低阈值三角主孔结构，联合扫描主孔、两个副孔的位置角、形状和方向。
最终分别报告光束质量、低阈值和折中最优，避免单一评分权重掩盖物理权衡。
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
    from .design_triangular_three_hole_shape_position import (
        ShapeCandidate,
        inclusion_record,
        validate_nonoverlap,
    )
else:
    from design_triangular_three_hole_pcsel import (
        DEVICE_RADIUS_UM,
        FINAL_RADIUS_CELLS,
        build_model,
        summarize_modes,
    )
    from design_triangular_three_hole_shape_position import (
        ShapeCandidate,
        inclusion_record,
        validate_nonoverlap,
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
    TriangularPolygon,
    plot_triangular_lattice,
)


OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "custom_semiconductor_triangular_triangle_main_refined"
)
TARGET_FILL = 0.15
COARSE_SETTINGS = (3, 8.0, 4)
MEDIUM_SETTINGS = (7, 4.0, 7)
FINAL_SETTINGS = (10, 3.0, FINAL_RADIUS_CELLS)
DIRECT = np.asarray(
    ((np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0), (-0.5, 0.5))
)


@dataclass(frozen=True)
class DesignParameters:
    name: str
    satellite_shape: str
    main_radius: float
    main_angle_deg: float
    distance: float
    spread_deg: float
    satellite_aspect: float = 1.4
    orientation_offset_deg: float = 0.0
    triangle_orientation: str = "outward"
    main_corner_fraction: float = 0.0
    satellite_corner_fraction: float = 0.0


def fractional_center(distance: float, physical_angle_deg: float) -> tuple[float, float]:
    angle = np.deg2rad(physical_angle_deg)
    cartesian = distance * np.asarray((np.cos(angle), np.sin(angle)))
    fractional = np.linalg.solve(DIRECT, cartesian)
    return float(fractional[0]), float(fractional[1])


def candidate_from_parameters(parameters: DesignParameters) -> ShapeCandidate:
    main = TriangularPolygon.rounded_regular(
        (0.0, 0.0),
        parameters.main_radius,
        sides=3,
        angle_deg=parameters.main_angle_deg,
        corner_fraction=parameters.main_corner_fraction,
    )
    remaining_fill = TARGET_FILL - main.fill_fraction
    if remaining_fill <= 0.0:
        raise ValueError("main triangle exceeds target fill")
    angles = (
        -30.0 - parameters.spread_deg,
        30.0 + parameters.spread_deg,
    )
    centers = tuple(
        fractional_center(parameters.distance, angle) for angle in angles
    )
    satellites: list[TriangularEllipse | TriangularPolygon] = []
    if parameters.satellite_shape == "ellipse":
        area_radius = float(np.sqrt(
            0.5 * remaining_fill * np.sqrt(3.0) / (2.0 * np.pi)
        ))
        root = np.sqrt(parameters.satellite_aspect)
        radii = (float(area_radius * root), float(area_radius / root))
        for center, radial_angle in zip(centers, angles, strict=True):
            satellites.append(TriangularEllipse(
                center_fractional=center,
                radii_over_a=radii,
                angle_deg=radial_angle + parameters.orientation_offset_deg,
            ))
    elif parameters.satellite_shape == "triangle":
        unit_satellite = TriangularPolygon.rounded_regular(
            (0.0, 0.0),
            1.0,
            sides=3,
            corner_fraction=parameters.satellite_corner_fraction,
        )
        satellite_radius = float(np.sqrt(
            0.5 * remaining_fill / unit_satellite.fill_fraction
        ))
        for center, radial_angle in zip(centers, angles, strict=True):
            if parameters.triangle_orientation == "outward":
                triangle_angle = radial_angle
            elif parameters.triangle_orientation == "inward":
                triangle_angle = radial_angle + 180.0
            elif parameters.triangle_orientation == "tangential":
                triangle_angle = radial_angle + 90.0
            else:
                raise ValueError("unknown triangle orientation")
            satellites.append(TriangularPolygon.rounded_regular(
                center,
                satellite_radius,
                sides=3,
                angle_deg=triangle_angle,
                corner_fraction=parameters.satellite_corner_fraction,
            ))
    else:
        raise ValueError("unknown satellite shape")
    return ShapeCandidate(
        parameters.name,
        f"triangle_main_{parameters.satellite_shape}_satellites",
        "local multi-objective refinement around the low-threshold triangle main hole",
        (main, satellites[0], satellites[1]),
    )


def parameter_panel() -> tuple[DesignParameters, ...]:
    baseline = dict(
        satellite_shape="ellipse",
        main_radius=0.20,
        main_angle_deg=0.0,
        distance=0.38,
        spread_deg=0.0,
        satellite_aspect=1.4,
        orientation_offset_deg=0.0,
    )
    parameters = [DesignParameters("baseline", **baseline)]

    def add_variation(field: str, values: tuple[float, ...], prefix: str) -> None:
        for value in values:
            if np.isclose(value, baseline[field]):
                continue
            modified = dict(baseline)
            modified[field] = value
            parameters.append(DesignParameters(f"{prefix}_{value:+.3f}", **modified))

    add_variation("main_radius", (0.18, 0.19, 0.21, 0.22), "main_r")
    add_variation("main_angle_deg", (-30.0, -15.0, 15.0, 30.0), "main_angle")
    add_variation("distance", (0.34, 0.36, 0.40, 0.42), "distance")
    add_variation("spread_deg", (-8.0, -4.0, 4.0, 8.0), "spread")
    add_variation("satellite_aspect", (1.1, 1.25, 1.6, 1.8), "sat_aspect")
    add_variation(
        "orientation_offset_deg", (-20.0, -10.0, 10.0, 20.0), "sat_angle"
    )
    combined = (
        (0.19, 15.0, 0.38, 0.0, 1.6, 0.0),
        (0.20, 15.0, 0.40, 4.0, 1.4, 0.0),
        (0.21, 0.0, 0.38, 0.0, 1.25, 10.0),
        (0.20, -15.0, 0.36, -4.0, 1.6, -10.0),
        (0.21, 30.0, 0.40, 4.0, 1.8, 10.0),
    )
    for index, values in enumerate(combined, start=1):
        parameters.append(DesignParameters(
            f"combined_{index:02d}",
            "ellipse",
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
        ))
    for index, values in enumerate((
        (0.18, 0.40, 0.0, 0.0, "outward"),
        (0.20, 0.42, 0.0, 0.0, "outward"),
        (0.22, 0.45, 0.0, 0.0, "outward"),
        (0.20, 0.42, 30.0, 0.0, "outward"),
        (0.20, 0.42, 0.0, 0.0, "inward"),
        (0.20, 0.42, 0.0, 0.0, "tangential"),
        (0.20, 0.44, 0.0, 6.0, "outward"),
        (0.21, 0.44, 15.0, -6.0, "inward"),
    ), start=1):
        parameters.append(DesignParameters(
            f"triangle_sat_{index:02d}",
            "triangle",
            values[0],
            values[2],
            values[1],
            values[3],
            triangle_orientation=values[4],
        ))
    return tuple(parameters)


PARAMETERS = parameter_panel()


def evaluate(candidate: ShapeCandidate, settings: tuple[int, float, int]):
    truncation, vertical_step_nm, radius_cells = settings
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
    summary = summarize_modes(result, modes)
    return cell, result, modes, summary


def row_from_evaluation(parameters: DesignParameters, evaluation: tuple) -> dict:
    cell, _, _, summary = evaluation
    _, actual, dark_floor, threshold_gap, beam_score = summary
    balanced_score = (
        beam_score
        - 0.40 * actual["threshold_cm-1"]
        + 0.15 * min(max(threshold_gap, -50.0), 50.0)
    )
    acceptable = (
        actual["center_to_peak"] >= 0.85
        and actual["peak_offset_deg"] <= 0.40
        and actual["encircled_power_1deg"] >= 0.65
        and threshold_gap >= 2.0
    )
    return {
        "candidate": parameters.name,
        "satellite_shape": parameters.satellite_shape,
        "main_radius": parameters.main_radius,
        "main_angle_deg": parameters.main_angle_deg,
        "distance": parameters.distance,
        "spread_deg": parameters.spread_deg,
        "satellite_aspect": parameters.satellite_aspect,
        "orientation_offset_deg": parameters.orientation_offset_deg,
        "triangle_orientation": parameters.triangle_orientation,
        "main_corner_fraction": parameters.main_corner_fraction,
        "satellite_corner_fraction": parameters.satellite_corner_fraction,
        "fill_fraction": cell.fill_fraction,
        "mode": actual["mode"],
        "threshold_cm-1": actual["threshold_cm-1"],
        "center_to_peak": actual["center_to_peak"],
        "peak_offset_deg": actual["peak_offset_deg"],
        "centroid_offset_deg": actual["centroid_offset_deg"],
        "ellipticity": actual["ellipticity"],
        "encircled_power_1deg": actual["encircled_power_1deg"],
        "full_rms_deg": actual["full_rms_deg"],
        "dark_floor_cm-1": dark_floor,
        "mode_gap_cm-1": threshold_gap,
        "beam_score": beam_score,
        "balanced_score": balanced_score,
        "acceptable": acceptable,
    }


def write_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    text_fields = {"candidate", "satellite_shape", "mode", "triangle_orientation"}
    bool_fields = {"acceptable"}
    nullable = {"dark_floor_cm-1"}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in tuple(row.items()):
            if key in text_fields:
                continue
            if key in bool_fields:
                row[key] = value.lower() == "true"
            elif key in nullable and value == "":
                row[key] = None
            else:
                row[key] = float(value)
    return rows


def unique_names(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def select_medium(rows: list[dict]) -> list[str]:
    acceptable = [row for row in rows if row["acceptable"]]
    names = [
        row["candidate"]
        for row in sorted(rows, key=lambda item: item["beam_score"], reverse=True)[:3]
    ]
    names += [
        row["candidate"]
        for row in sorted(rows, key=lambda item: item["balanced_score"], reverse=True)[:3]
    ]
    names += [
        row["candidate"]
        for row in sorted(acceptable, key=lambda item: item["threshold_cm-1"])[:2]
    ]
    triangle_rows = [row for row in rows if row["satellite_shape"] == "triangle"]
    if triangle_rows:
        names.append(max(triangle_rows, key=lambda item: item["balanced_score"])["candidate"])
    ellipse_rows = [row for row in rows if row["satellite_shape"] == "ellipse"]
    if ellipse_rows:
        names.append(max(ellipse_rows, key=lambda item: item["beam_score"])["candidate"])
        names.append(max(ellipse_rows, key=lambda item: item["balanced_score"])["candidate"])
    return unique_names(names)


def select_final(rows: list[dict]) -> tuple[list[str], dict[str, str]]:
    acceptable = [row for row in rows if row["acceptable"]]
    beam = max(rows, key=lambda item: item["beam_score"])
    balanced = max(rows, key=lambda item: item["balanced_score"])
    low = min(acceptable, key=lambda item: item["threshold_cm-1"])
    categories = {
        "beam_quality": beam["candidate"],
        "balanced": balanced["candidate"],
        "low_threshold": low["candidate"],
    }
    triangle_rows = [row for row in rows if row["satellite_shape"] == "triangle"]
    if triangle_rows:
        categories["triangle_satellite"] = max(
            triangle_rows, key=lambda item: item["balanced_score"]
        )["candidate"]
    ellipse_rows = [row for row in rows if row["satellite_shape"] == "ellipse"]
    if ellipse_rows:
        categories["ellipse_satellite"] = max(
            ellipse_rows, key=lambda item: item["balanced_score"]
        )["candidate"]
    return unique_names(list(categories.values())), categories


def plot_tradeoff(rows: list[dict], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    colors = {"ellipse": "#4c78a8", "triangle": "#e15759"}
    for row in rows:
        ax.scatter(
            row["threshold_cm-1"],
            row["beam_score"],
            s=35 + 90 * row["encircled_power_1deg"],
            color=colors[row["satellite_shape"]],
            alpha=0.8,
        )
    best = sorted(rows, key=lambda item: item["balanced_score"], reverse=True)[:5]
    for row in best:
        ax.annotate(
            row["candidate"],
            (row["threshold_cm-1"], row["beam_score"]),
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


def main() -> None:
    output = OUTPUT_DIRECTORY.resolve()
    output.mkdir(parents=True, exist_ok=True)
    parameter_lookup = {item.name: item for item in PARAMETERS}
    valid: list[tuple[DesignParameters, ShapeCandidate]] = []
    for parameters in PARAMETERS:
        candidate = candidate_from_parameters(parameters)
        try:
            validate_nonoverlap(candidate)
        except ValueError as error:
            print(f"  rejected {parameters.name}: {error}")
        else:
            valid.append((parameters, candidate))

    coarse_path = output / "01_coarse.csv"
    coarse_rows = read_rows(coarse_path) if coarse_path.is_file() else []
    if len(coarse_rows) != len(valid):
        coarse_rows = []
        print(f"Coarse triangular-main refinement: {len(valid)} candidates")
        for index, (parameters, candidate) in enumerate(valid, start=1):
            print(f"  [{index}/{len(valid)}] {parameters.name} ...", flush=True)
            evaluation = evaluate(candidate, COARSE_SETTINGS)
            row = row_from_evaluation(parameters, evaluation)
            coarse_rows.append(row)
            print(
                f"    {row['mode']}: alpha={row['threshold_cm-1']:.1f}, "
                f"beam={row['beam_score']:.1f}, balanced={row['balanced_score']:.1f}"
            )
        write_rows(coarse_rows, coarse_path)
    else:
        print(f"Reusing {len(coarse_rows)} coarse candidates")
    plot_tradeoff(coarse_rows, output / "01_coarse_tradeoff.png", "Coarse triangle-main tradeoff")

    medium_names = select_medium(coarse_rows)
    medium_path = output / "02_medium.csv"
    medium_rows = read_rows(medium_path) if medium_path.is_file() else []
    if {row["candidate"] for row in medium_rows} != set(medium_names):
        medium_rows = []
        print(f"Medium reranking: {len(medium_names)} candidates")
        for name in medium_names:
            parameters = parameter_lookup[name]
            candidate = candidate_from_parameters(parameters)
            print(f"  refining {name} ...", flush=True)
            evaluation = evaluate(candidate, MEDIUM_SETTINGS)
            medium_rows.append(row_from_evaluation(parameters, evaluation))
        write_rows(medium_rows, medium_path)
    else:
        print(f"Reusing {len(medium_rows)} medium candidates")
    plot_tradeoff(medium_rows, output / "02_medium_tradeoff.png", "Medium triangle-main tradeoff")

    final_names, category_seed = select_final(medium_rows)
    final_rows: list[dict] = []
    final_data = {}
    print(f"Converged final comparison: {len(final_names)} candidates")
    for name in final_names:
        parameters = parameter_lookup[name]
        candidate = candidate_from_parameters(parameters)
        print(f"  converging {name} ...", flush=True)
        cell, result = build_model(
            candidate,
            TriangularCWTSettings(
                truncation_order=FINAL_SETTINGS[0],
                vertical_step_nm=FINAL_SETTINGS[1],
            ),
        )
        modes, convergence = solve_triangular_finite_modes_converged(
            result,
            TriangularFiniteSpec(
                radius_um=DEVICE_RADIUS_UM,
                radius_cells=FINAL_SETTINGS[2][-1],
                eigensolutions_per_target=8,
            ),
            FINAL_SETTINGS[2],
        )
        summary = summarize_modes(result, modes)
        row = row_from_evaluation(parameters, (cell, result, modes, summary))
        final_rows.append(row)
        final_data[name] = (candidate, cell, result, modes, convergence, summary)
        candidate_output = output / "finalists" / name
        candidate_output.mkdir(parents=True, exist_ok=True)
        plot_triangular_lattice(cell, candidate_output / "01_lattice.png")
        write_triangular_finite_table(modes, result, candidate_output / "02_modes.csv")
        plot_triangular_grid_convergence(
            convergence, DEVICE_RADIUS_UM, candidate_output / "03_convergence.png"
        )
        plot_triangular_thresholds(modes, candidate_output / "04_thresholds.png")
        plot_triangular_far_field_diagnostics(
            modes[row["mode"]], result, candidate_output / "05_far_field.png"
        )
        print(
            f"    mode={row['mode']}, alpha={row['threshold_cm-1']:.2f}, "
            f"beam={row['beam_score']:.1f}, gap={row['mode_gap_cm-1']:.2f}"
        )
    write_rows(final_rows, output / "03_final.csv")
    plot_tradeoff(final_rows, output / "03_final_tradeoff.png", "Converged triangle-main tradeoff")

    final_by_name = {row["candidate"]: row for row in final_rows}
    categories = {
        "beam_quality": max(final_rows, key=lambda item: item["beam_score"])["candidate"],
        "balanced": max(final_rows, key=lambda item: item["balanced_score"])["candidate"],
        "low_threshold": min(
            (row for row in final_rows if row["acceptable"]),
            key=lambda item: item["threshold_cm-1"],
        )["candidate"],
    }
    if "triangle_satellite" in category_seed:
        categories["triangle_satellite"] = category_seed["triangle_satellite"]
    if "ellipse_satellite" in category_seed:
        categories["ellipse_satellite"] = category_seed["ellipse_satellite"]
    report = {
        "recommendations": {
            category: final_by_name.get(name, {"candidate": name})
            for category, name in categories.items()
        },
        "finalists": final_rows,
        "geometries": {
            name: [inclusion_record(item) for item in final_data[name][0].inclusions]
            for name in final_data
        },
        "search_counts": {
            "defined": len(PARAMETERS),
            "valid": len(valid),
            "medium": len(medium_names),
            "final": len(final_names),
        },
        "warning": (
            "Reduced-order multi-objective search. Verify larger finite grids, rounded "
            "triangle corners, tolerances, and RCWA/FEM/FDTD before fabrication."
        ),
    }
    (output / "04_recommendations.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    balanced_name = categories["balanced"]
    _, _, balanced_result, balanced_modes, _, balanced_summary = final_data[balanced_name]
    plot_triangular_mode_atlas(
        balanced_modes,
        balanced_result,
        output / "05_balanced_mode_atlas.png",
    )
    print(f"recommendations={categories}")


if __name__ == "__main__":
    main()
