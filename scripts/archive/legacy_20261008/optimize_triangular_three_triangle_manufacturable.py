"""Optimize the 300-um three-triangle PCSEL with fabrication-aware metrics.

The scan is deliberately performed on the final-size square device rather
than reusing the old 30-um ranking.  Every geometry rebuilds the six-wave
Cb+Cr+Ch matrix.  Optical ranking uses the actual lowest-threshold mode;
fabrication ranking reports minimum periodic gap, minimum hole span, corner
rounding, and alignment with a lattice mirror axis.

中文：在 300 um 方形器件上重新扫描主三角孔角度、尺寸、孔距、双副孔夹角和
主/副孔圆角。每个候选都重新生成 Fourier 系数与六波矩阵，并把最小周期孔间距和
最小孔宽纳入综合评分。结果仍需用全波方法和实际工艺规则复核。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np

from pcselsim.triangular_finite import (
    TriangularFiniteSpec,
    plot_triangular_far_field_diagnostics,
    plot_triangular_grid_convergence,
    plot_triangular_mode_atlas,
    plot_triangular_thresholds,
    solve_triangular_finite_modes,
    solve_triangular_finite_modes_converged,
    triangular_vector_far_field,
    write_triangular_finite_table,
)
from pcselsim.triangular_six_wave import (
    TRIANGULAR_MODE_NAMES,
    TriangularCWTSettings,
    plot_triangular_lattice,
)

if __package__:
    from .design_triangular_three_hole_shape_position import validate_nonoverlap
    from .design_triangular_triangle_main_refinement import (
        DesignParameters,
        candidate_from_parameters,
    )
    from .run_best_triangular_three_triangle_pcsel import (
        DEVICE_HALF_SIZE_UM,
        LATTICE_CONSTANT_NM,
        build_model_for_candidate,
    )
else:
    from design_triangular_three_hole_shape_position import validate_nonoverlap
    from design_triangular_triangle_main_refinement import (
        DesignParameters,
        candidate_from_parameters,
    )
    from run_best_triangular_three_triangle_pcsel import (
        DEVICE_HALF_SIZE_UM,
        LATTICE_CONSTANT_NM,
        build_model_for_candidate,
    )


OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "results" / "triangular_three_triangle_manufacturable_300um"
)
COARSE_SETTINGS = (4, 7.0, 5)
MEDIUM_SETTINGS = (7, 4.0, 7)
FINAL_SETTINGS = (10, 3.0, (7, 9, 11))
DIRECT = np.asarray(
    ((np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0), (-0.5, 0.5))
)
A1 = np.asarray((np.sqrt(3.0) / 2.0, -0.5))
A2 = np.asarray((np.sqrt(3.0) / 2.0, 0.5))


def parameter_panel() -> tuple[DesignParameters, ...]:
    """Return a compact local design-of-experiments around triangle_sat_08."""
    values: dict[tuple[float, ...], DesignParameters] = {}

    def add(
        main_radius: float = 0.21,
        angle: float = 15.0,
        distance: float = 0.44,
        spread: float = -6.0,
        main_round: float = 0.0,
        satellite_round: float | None = None,
    ) -> None:
        satellite_round = main_round if satellite_round is None else satellite_round
        key = (
            main_radius, angle, distance, spread, main_round, satellite_round
        )
        name = (
            f"r{main_radius:.3f}_a{angle:04.1f}_d{distance:.3f}_"
            f"s{spread:+04.1f}_q{main_round:.2f}_{satellite_round:.2f}"
        ).replace(".", "p").replace("+", "P").replace("-", "M")
        values[key] = DesignParameters(
            name=name,
            satellite_shape="triangle",
            main_radius=main_radius,
            main_angle_deg=angle,
            distance=distance,
            spread_deg=spread,
            triangle_orientation="inward",
            main_corner_fraction=main_round,
            satellite_corner_fraction=satellite_round,
        )

    # Direct test of the questioned 15-degree rotation, both sharp and rounded.
    for angle in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        add(angle=angle, main_round=0.0)
        add(angle=angle, main_round=0.10)

    # One-factor and selected interaction scans around the previous optimum.
    for angle in (0.0, 10.0, 15.0):
        for distance in (0.40, 0.42, 0.44, 0.46):
            add(angle=angle, distance=distance, main_round=0.10)
        for spread in (-8.0, -6.0, -4.0, -2.0, 0.0):
            add(angle=angle, spread=spread, main_round=0.10)
        for radius in (0.19, 0.20, 0.21, 0.22):
            add(main_radius=radius, angle=angle, main_round=0.10)
        for rounding in (0.05, 0.10, 0.15, 0.20):
            add(angle=angle, main_round=rounding)

    # Mixed rounding tests identify whether only the small satellites need relief.
    for angle in (0.0, 10.0, 15.0):
        add(angle=angle, main_round=0.05, satellite_round=0.15)
        add(angle=angle, main_round=0.10, satellite_round=0.20)
    return tuple(values.values())


PARAMETERS = parameter_panel()


def _boundary(inclusion) -> np.ndarray:
    return (DIRECT @ np.asarray(inclusion.vertices_fractional).T).T


def _point_segment_distance(point, first, second) -> float:
    edge = second - first
    denominator = float(edge @ edge)
    if denominator <= np.finfo(float).eps:
        return float(np.linalg.norm(point - first))
    parameter = float(np.clip(((point - first) @ edge) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (first + parameter * edge)))


def _polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    result = np.inf
    for point in first:
        for index in range(len(second)):
            result = min(result, _point_segment_distance(
                point, second[index], second[(index + 1) % len(second)]
            ))
    for point in second:
        for index in range(len(first)):
            result = min(result, _point_segment_distance(
                point, first[index], first[(index + 1) % len(first)]
            ))
    return float(result)


def fabrication_metrics(candidate, parameters: DesignParameters) -> dict[str, float]:
    boundaries = [_boundary(inclusion) for inclusion in candidate.inclusions]
    minimum_gap = np.inf
    for left, first in enumerate(boundaries):
        for right, second in enumerate(boundaries):
            for shift_1 in (-1, 0, 1):
                for shift_2 in (-1, 0, 1):
                    if left == right and shift_1 == 0 and shift_2 == 0:
                        continue
                    shifted = second + shift_1 * A1 + shift_2 * A2
                    minimum_gap = min(
                        minimum_gap, _polygon_distance(first, shifted)
                    )
    spans = []
    for boundary in boundaries:
        centered = boundary - boundary.mean(axis=0)
        angles = np.linspace(0.0, np.pi, 181)
        widths = []
        for angle in angles:
            axis = np.asarray((np.cos(angle), np.sin(angle)))
            projection = centered @ axis
            widths.append(float(projection.max() - projection.min()))
        spans.append(min(widths))
    lattice_nm = LATTICE_CONSTANT_NM
    minimum_gap_nm = float(minimum_gap * lattice_nm)
    minimum_span_nm = float(min(spans) * lattice_nm)
    rounding_relief = min(
        parameters.main_corner_fraction,
        parameters.satellite_corner_fraction,
    )
    mirror_axis_distance_deg = float(
        min(abs(parameters.main_angle_deg - axis) for axis in (0.0, 30.0))
    )
    score = (
        4.0 * min(minimum_gap_nm / 45.0, 1.5)
        + 3.0 * min(minimum_span_nm / 75.0, 1.5)
        + 2.0 * min(rounding_relief / 0.10, 1.5)
        + 1.0 * (1.0 - mirror_axis_distance_deg / 15.0)
    )
    return {
        "minimum_periodic_gap_nm": minimum_gap_nm,
        "minimum_hole_span_nm": minimum_span_nm,
        "rounding_relief": rounding_relief,
        "mirror_axis_distance_deg": mirror_axis_distance_deg,
        "fabrication_score": float(score),
    }


def summarize(model, modes) -> dict[str, float | str]:
    records = []
    for name in TRIANGULAR_MODE_NAMES:
        mode = modes[name]
        far = triangular_vector_far_field(
            mode, model.bragg_wavelength_nm, view_deg=1.5, samples=81
        )
        records.append({
            "mode": name,
            "threshold_cm-1": mode.alpha_per_m / 100.0,
            "center_to_peak": far.center_to_peak,
            "peak_offset_deg": far.peak_offset_deg,
            "centroid_offset_deg": far.centroid_offset_deg,
            "ellipticity": far.ellipticity,
            "encircled_power_0p5deg": far.encircled_power_0p5deg,
            "encircled_power_1deg": far.encircled_power_1deg,
            "full_rms_deg": far.full_rms_divergence_deg,
        })
    actual = min(records, key=lambda row: row["threshold_cm-1"])
    gap = min(
        row["threshold_cm-1"] for row in records if row["mode"] != actual["mode"]
    ) - actual["threshold_cm-1"]
    optical_score = (
        40.0 * actual["center_to_peak"]
        + 80.0 * actual["encircled_power_0p5deg"]
        + 20.0 * actual["encircled_power_1deg"]
        - 0.8 * actual["threshold_cm-1"]
        + 0.8 * min(gap, 10.0)
        - 25.0 * actual["full_rms_deg"]
        - 12.0 * abs(np.log(max(actual["ellipticity"], 1e-12)))
        - 25.0 * actual["peak_offset_deg"]
        - 15.0 * actual["centroid_offset_deg"]
    )
    return {**actual, "mode_gap_cm-1": float(gap), "optical_score": float(optical_score)}


def evaluate(parameters: DesignParameters, settings: tuple[int, float, int]):
    candidate = candidate_from_parameters(parameters)
    validate_nonoverlap(candidate)
    cell, _, model = build_model_for_candidate(
        candidate,
        TriangularCWTSettings(
            truncation_order=settings[0], vertical_step_nm=settings[1]
        ),
    )
    modes = solve_triangular_finite_modes(
        model,
        TriangularFiniteSpec(
            radius_um=DEVICE_HALF_SIZE_UM,
            radius_cells=settings[2],
            internal_loss_cm=0.0,
            aperture_shape="square",
            eigensolutions_per_target=7,
        ),
    )
    return candidate, cell, model, modes, summarize(model, modes)


def result_row(parameters: DesignParameters, evaluation) -> dict:
    candidate, cell, model, _, optical = evaluation
    fabrication = fabrication_metrics(candidate, parameters)
    balanced = optical["optical_score"] + 0.65 * fabrication["fabrication_score"]
    return {
        "candidate": parameters.name,
        "main_radius": parameters.main_radius,
        "main_angle_deg": parameters.main_angle_deg,
        "distance": parameters.distance,
        "spread_deg": parameters.spread_deg,
        "main_corner_fraction": parameters.main_corner_fraction,
        "satellite_corner_fraction": parameters.satellite_corner_fraction,
        "fill_fraction": cell.fill_fraction,
        "bragg_wavelength_nm": model.bragg_wavelength_nm,
        **optical,
        **fabrication,
        "balanced_score": float(balanced),
    }


def write_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_angle_scan(rows: list[dict], path: Path) -> None:
    selected = [
        row for row in rows
        if np.isclose(row["main_radius"], 0.21)
        and np.isclose(row["distance"], 0.44)
        and np.isclose(row["spread_deg"], -6.0)
        and np.isclose(row["main_corner_fraction"], row["satellite_corner_fraction"])
        and row["main_corner_fraction"] in (0.0, 0.1)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5), sharex=True)
    for rounding, label in ((0.0, "sharp"), (0.1, "rounded q=0.10")):
        current = sorted(
            (row for row in selected if np.isclose(row["main_corner_fraction"], rounding)),
            key=lambda row: row["main_angle_deg"],
        )
        x = [row["main_angle_deg"] for row in current]
        axes[0, 0].plot(x, [row["threshold_cm-1"] for row in current], "o-", label=label)
        axes[0, 1].plot(x, [row["mode_gap_cm-1"] for row in current], "o-", label=label)
        axes[1, 0].plot(x, [row["full_rms_deg"] for row in current], "o-", label=label)
        axes[1, 1].plot(x, [row["encircled_power_0p5deg"] for row in current], "o-", label=label)
    labels = (
        ("cold-cavity loss (cm$^{-1}$)", "nearest-mode gap (cm$^{-1}$)"),
        ("full RMS divergence (deg)", "encircled power within 0.5 deg"),
    )
    for row_axes, row_labels in zip(axes, labels, strict=True):
        for ax, ylabel in zip(row_axes, row_labels, strict=True):
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel("main-triangle rotation (deg)")
    axes[0, 0].legend()
    fig.suptitle("Why 15 degrees? Controlled 300-um angle scan")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tradeoff(rows: list[dict], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    scatter = ax.scatter(
        [row["threshold_cm-1"] for row in rows],
        [row["full_rms_deg"] for row in rows],
        c=[row["fabrication_score"] for row in rows],
        s=[35.0 + 100.0 * row["encircled_power_0p5deg"] for row in rows],
        cmap="viridis", alpha=0.85,
    )
    for row in sorted(rows, key=lambda item: item["balanced_score"], reverse=True)[:6]:
        ax.annotate(row["candidate"], (row["threshold_cm-1"], row["full_rms_deg"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=6.5)
    ax.set(
        xlabel="lowest cold-cavity loss (cm$^{-1}$)",
        ylabel="full RMS divergence (deg)", title=title,
    )
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="fabrication score")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def unique(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def main() -> None:
    output = OUTPUT_DIRECTORY.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lookup = {item.name: item for item in PARAMETERS}
    rows = []
    print(f"300-um fabrication-aware screen: {len(PARAMETERS)} candidates")
    for index, parameters in enumerate(PARAMETERS, start=1):
        print(f"  [{index}/{len(PARAMETERS)}] {parameters.name}", flush=True)
        try:
            evaluation = evaluate(parameters, COARSE_SETTINGS)
        except ValueError as error:
            print(f"    rejected: {error}")
            continue
        row = result_row(parameters, evaluation)
        rows.append(row)
        print(
            f"    {row['mode']}: alpha={row['threshold_cm-1']:.3f}, "
            f"gap={row['mode_gap_cm-1']:.3f}, rms={row['full_rms_deg']:.3f}, "
            f"gap_nm={row['minimum_periodic_gap_nm']:.1f}"
        )
    write_rows(rows, output / "01_coarse.csv")
    plot_angle_scan(rows, output / "01_angle_scan.png")
    plot_tradeoff(rows, output / "01_coarse_tradeoff.png", "300-um coarse screen")

    names = unique(
        [row["candidate"] for row in sorted(rows, key=lambda row: row["balanced_score"], reverse=True)[:8]]
        + [row["candidate"] for row in sorted(rows, key=lambda row: row["optical_score"], reverse=True)[:3]]
        + [min(rows, key=lambda row: row["threshold_cm-1"])["candidate"]]
        + [max(
            (row for row in rows if row["main_corner_fraction"] >= 0.1
             and np.isclose(row["mirror_axis_distance_deg"], 0.0)),
            key=lambda row: row["balanced_score"],
        )["candidate"]]
    )
    medium_rows = []
    print(f"Medium reranking: {len(names)} candidates")
    for name in names:
        evaluation = evaluate(lookup[name], MEDIUM_SETTINGS)
        medium_rows.append(result_row(lookup[name], evaluation))
    write_rows(medium_rows, output / "02_medium.csv")
    plot_tradeoff(medium_rows, output / "02_medium_tradeoff.png", "300-um medium reranking")

    categories = {
        "balanced": max(medium_rows, key=lambda row: row["balanced_score"])["candidate"],
        "optical": max(medium_rows, key=lambda row: row["optical_score"])["candidate"],
        "lowest_loss": min(medium_rows, key=lambda row: row["threshold_cm-1"])["candidate"],
        "aligned_rounded": max(
            (row for row in medium_rows if row["main_corner_fraction"] >= 0.1
             and np.isclose(row["mirror_axis_distance_deg"], 0.0)),
            key=lambda row: row["balanced_score"],
        )["candidate"],
    }
    final_names = unique(list(categories.values()))
    final_rows = []
    final_data = {}
    print(f"Production multi-grid finals: {len(final_names)} candidates")
    for name in final_names:
        parameters = lookup[name]
        candidate = candidate_from_parameters(parameters)
        validate_nonoverlap(candidate)
        cell, _, model = build_model_for_candidate(candidate)
        modes, convergence = solve_triangular_finite_modes_converged(
            model,
            TriangularFiniteSpec(
                radius_um=DEVICE_HALF_SIZE_UM,
                radius_cells=FINAL_SETTINGS[2][-1],
                internal_loss_cm=0.0,
                aperture_shape="square",
                eigensolutions_per_target=8,
            ),
            FINAL_SETTINGS[2],
        )
        evaluation = (candidate, cell, model, modes, summarize(model, modes))
        row = result_row(parameters, evaluation)
        final_rows.append(row)
        final_data[name] = (candidate, cell, model, modes, convergence)
        folder = output / "finalists" / name
        folder.mkdir(parents=True, exist_ok=True)
        plot_triangular_lattice(cell, folder / "01_lattice.png")
        write_triangular_finite_table(modes, model, folder / "02_modes.csv")
        plot_triangular_grid_convergence(
            convergence, DEVICE_HALF_SIZE_UM, folder / "03_convergence.png"
        )
        plot_triangular_thresholds(modes, folder / "04_thresholds.png")
        plot_triangular_far_field_diagnostics(
            modes[row["mode"]], model, folder / "05_far_field.png", view_deg=1.5
        )
        print(
            f"  {name}: {row['mode']} alpha={row['threshold_cm-1']:.3f}, "
            f"gap={row['mode_gap_cm-1']:.3f}, rms={row['full_rms_deg']:.3f}"
        )
    write_rows(final_rows, output / "03_final.csv")
    plot_tradeoff(final_rows, output / "03_final_tradeoff.png", "Production multi-grid finalists")

    winner = max(final_rows, key=lambda row: row["balanced_score"])
    winner_name = winner["candidate"]
    candidate, cell, model, modes, _ = final_data[winner_name]
    plot_triangular_mode_atlas(
        modes, model, output / "04_winner_mode_atlas.png", view_deg=1.5
    )
    report = {
        "categories": categories,
        "winner": winner,
        "winner_parameters": asdict(lookup[winner_name]),
        "finalists": final_rows,
        "candidate_counts": {
            "defined": len(PARAMETERS), "valid": len(rows),
            "medium": len(medium_rows), "final": len(final_rows),
        },
        "interpretation": (
            "Linear six-wave cold-cavity optimization on a 300-um square device. "
            "Fabrication score is a geometric proxy, not a foundry design-rule check."
        ),
    }
    (output / "04_recommendation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"winner={winner_name}")


if __name__ == "__main__":
    main()
