"""Optimize a main-hole plus two-axis-satellite triangular PCSEL cell.

The primitive cell always contains one main air hole at the origin and two
satellite holes on ``d1*a1`` and ``d2*a2``.  Every candidate is converted to
Fourier coefficients before the independent six-wave 3-D CWT and finite-device
solvers are run.  The selected design therefore follows from its geometry; no
far-field spot is imposed or fitted.

中文：本脚本固定“三孔原胞”的拓扑，只扫描孔径、间距、椭圆率和方向。评分对象是
实际最低阈值模式，并同时检查中心亮度、峰值偏移、质心偏移、椭圆率、1 度内能量和
竞争模式阈值。筛选结果仍须用 RCWA/FEM/FDTD 和制造公差分析复核。
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
    TriangularEllipse,
    TriangularLatticeCell,
    build_triangular_coupling,
    plot_six_band_edge_states,
    plot_triangular_lattice,
)
from pcselsim.vertical import Layer, LayerStack


LATTICE_CONSTANT_NM = 341.0
BACKGROUND_EPSILON = 12.7449
HOLE_EPSILON = 1.0
WAVELENGTH_GUESS_NM = 995.0
DEVICE_RADIUS_UM = 30.0
TARGET_FILL = 0.15
SCREEN_TRUNCATION = 4
SCREEN_VERTICAL_STEP_NM = 6.0
SCREEN_RADIUS_CELLS = 5
FINAL_TRUNCATION = 10
FINAL_VERTICAL_STEP_NM = 3.0
FINAL_RADIUS_CELLS = (7, 9, 11)
OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "results" / "custom_semiconductor_triangular_three_hole_optimized"
)

PC_LAYER_NAME = "PC"
VERTICAL_LAYERS_FIXED = (
    ("p-clad AlGaAs", 1500.0, np.sqrt(11.0224)),
    ("GaAs", 59.0, np.sqrt(12.7449)),
    (PC_LAYER_NAME, 118.0, 0.0),
    ("active InGaAs/GaAs", 88.5, np.sqrt(12.8603)),
    ("n-clad AlGaAs", 1500.0, np.sqrt(11.0224)),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    explanation: str
    inclusions: tuple[TriangularEllipse, ...]


def _area_equivalent_axes(radius: float, aspect: float) -> tuple[float, float]:
    """Return ellipse axes with the same area as a circle of ``radius``."""
    root = np.sqrt(aspect)
    return float(radius * root), float(radius / root)


def _main_radius(satellite_radius: float, target_fill: float = TARGET_FILL) -> float:
    area_sum = target_fill * np.sqrt(3.0) / (2.0 * np.pi)
    remaining = area_sum - 2.0 * satellite_radius**2
    if remaining <= 0.0:
        raise ValueError("satellite holes exceed target fill fraction")
    return float(np.sqrt(remaining))


def three_hole_candidate(
    name: str,
    explanation: str,
    distance_1: float,
    distance_2: float,
    satellite_radius: float,
    satellite_aspect: float = 1.0,
    orientation: str = "circular",
    main_aspect: float = 1.0,
    target_fill: float = TARGET_FILL,
) -> Candidate:
    """Build one centered main hole and satellites on +a1 and +a2."""
    main_radius = _main_radius(satellite_radius, target_fill)
    main_axes = _area_equivalent_axes(main_radius, main_aspect)
    satellite_axes = _area_equivalent_axes(satellite_radius, satellite_aspect)
    if orientation == "radial":
        satellite_angles = (-30.0, 30.0)
    elif orientation == "tangential":
        satellite_angles = (60.0, -60.0)
    elif orientation == "parallel_x":
        satellite_angles = (0.0, 0.0)
    elif orientation == "counter_rotated":
        satellite_angles = (15.0, -15.0)
    elif orientation == "circular":
        satellite_angles = (0.0, 0.0)
    else:
        raise ValueError(f"unknown orientation: {orientation}")
    return Candidate(
        name,
        explanation,
        (
            TriangularEllipse(
                center_fractional=(0.0, 0.0),
                radii_over_a=main_axes,
                angle_deg=90.0 if main_aspect != 1.0 else 0.0,
                epsilon=HOLE_EPSILON,
            ),
            TriangularEllipse(
                center_fractional=(distance_1, 0.0),
                radii_over_a=satellite_axes,
                angle_deg=satellite_angles[0],
                epsilon=HOLE_EPSILON,
            ),
            TriangularEllipse(
                center_fractional=(0.0, distance_2),
                radii_over_a=satellite_axes,
                angle_deg=satellite_angles[1],
                epsilon=HOLE_EPSILON,
            ),
        ),
    )


CANDIDATES = (
    three_hole_candidate(
        "circle_d026_r007", "close, small circular satellites", 0.26, 0.26, 0.07
    ),
    three_hole_candidate(
        "circle_d030_r007", "moderate distance, small satellites", 0.30, 0.30, 0.07
    ),
    three_hole_candidate(
        "circle_d030_r008", "balanced circular reference", 0.30, 0.30, 0.08
    ),
    three_hole_candidate(
        "circle_d034_r008", "larger Fourier phase delay", 0.34, 0.34, 0.08
    ),
    three_hole_candidate(
        "circle_d036_r009", "strong, well-separated satellites", 0.36, 0.36, 0.09
    ),
    three_hole_candidate(
        "radial_d030_a14", "elliptical satellites point along a1 and a2",
        0.30, 0.30, 0.08, 1.4, "radial"
    ),
    three_hole_candidate(
        "radial_d034_a17", "stronger radial anisotropy and phase delay",
        0.34, 0.34, 0.08, 1.7, "radial"
    ),
    three_hole_candidate(
        "tangent_d030_a14", "satellite major axes transverse to a1 and a2",
        0.30, 0.30, 0.08, 1.4, "tangential"
    ),
    three_hole_candidate(
        "tangent_d034_a17", "strong tangential anisotropy",
        0.34, 0.34, 0.08, 1.7, "tangential"
    ),
    three_hole_candidate(
        "parallel_d032_a15", "Noda-like co-oriented elliptical satellites",
        0.32, 0.32, 0.08, 1.5, "parallel_x"
    ),
    three_hole_candidate(
        "counter_d032_a15", "counter-rotated satellite ellipses",
        0.32, 0.32, 0.08, 1.5, "counter_rotated"
    ),
    three_hole_candidate(
        "staggered_d028_d034", "unequal axial phases test beam steering",
        0.28, 0.34, 0.08, 1.4, "radial"
    ),
    three_hole_candidate(
        "radial_main_y_d032", "radial satellites plus a weakly elliptical main hole",
        0.32, 0.32, 0.08, 1.4, "radial", 1.15
    ),
    three_hole_candidate(
        "radial_lowfill", "lower 0.13 fill tests radiation-loss trade-off",
        0.32, 0.32, 0.075, 1.5, "radial", target_fill=0.13
    ),
    three_hole_candidate(
        "radial_highfill", "higher 0.17 fill tests stronger coupling",
        0.32, 0.32, 0.085, 1.5, "radial", target_fill=0.17
    ),
)


def build_model(candidate: Candidate, settings: TriangularCWTSettings):
    """Rebuild vertical mode and all six-wave couplings from cell geometry."""
    cell = TriangularLatticeCell(BACKGROUND_EPSILON, candidate.inclusions)
    average_index = float(np.sqrt(cell.fourier_epsilon(0, 0).real))
    stack = LayerStack(
        layers=tuple(
            Layer(name, thickness, average_index if name == PC_LAYER_NAME else index)
            for name, thickness, index in VERTICAL_LAYERS_FIXED
        ),
        top_index=float(np.sqrt(11.0224)),
        bottom_index=float(np.sqrt(11.0224)),
        padding_um=1.0,
    )
    result = build_triangular_coupling(
        cell,
        stack,
        PC_LAYER_NAME,
        LATTICE_CONSTANT_NM,
        WAVELENGTH_GUESS_NM,
        settings,
    )
    return cell, result


def summarize_modes(result, modes):
    """Score the actual lowest-threshold state rather than a hand-picked mode."""
    records = []
    for name in TRIANGULAR_MODE_NAMES:
        mode = modes[name]
        far = triangular_vector_far_field(
            mode, result.bragg_wavelength_nm, view_deg=3.0, samples=81
        )
        records.append({
            "mode": name,
            "threshold_cm-1": mode.alpha_per_m / 100.0,
            "alpha_L": mode.alpha_l,
            "center_to_peak": far.center_to_peak,
            "full_rms_deg": far.full_rms_divergence_deg,
            "peak_offset_deg": far.peak_offset_deg,
            "centroid_offset_deg": far.centroid_offset_deg,
            "ellipticity": far.ellipticity,
            "encircled_power_1deg": far.encircled_power_1deg,
        })
    actual = min(records, key=lambda item: item["threshold_cm-1"])
    competitors = [item for item in records if item["mode"] != actual["mode"]]
    threshold_gap = min(item["threshold_cm-1"] for item in competitors) \
        - actual["threshold_cm-1"]
    dark = [item for item in competitors if item["center_to_peak"] < 0.20]
    dark_floor = min(
        (item["threshold_cm-1"] for item in dark),
        default=None,
    )
    discrimination_reference = (
        dark_floor
        if dark_floor is not None
        else min(item["threshold_cm-1"] for item in competitors)
    )
    score = (
        55.0 * actual["center_to_peak"]
        + 55.0 * actual["encircled_power_1deg"]
        - 18.0 * actual["peak_offset_deg"]
        - 12.0 * actual["centroid_offset_deg"]
        - 14.0 * abs(np.log(max(actual["ellipticity"], 1e-12)))
        - 2.0 * actual["full_rms_deg"]
        + 0.20 * min(
            max(discrimination_reference - actual["threshold_cm-1"], -100.0), 100.0
        )
        + 0.10 * min(max(threshold_gap, -100.0), 100.0)
        - 0.03 * actual["threshold_cm-1"]
    )
    if actual["center_to_peak"] < 0.45:
        score -= 150.0
    if actual["peak_offset_deg"] > 0.50:
        score -= 40.0
    return records, actual, dark_floor, float(threshold_gap), float(score)


def evaluate(candidate: Candidate):
    cell, result = build_model(
        candidate,
        TriangularCWTSettings(
            truncation_order=SCREEN_TRUNCATION,
            vertical_step_nm=SCREEN_VERTICAL_STEP_NM,
        ),
    )
    modes = solve_triangular_finite_modes(
        result,
        TriangularFiniteSpec(
            radius_um=DEVICE_RADIUS_UM,
            radius_cells=SCREEN_RADIUS_CELLS,
            eigensolutions_per_target=7,
        ),
    )
    return cell, result, modes, summarize_modes(result, modes)


def plot_screening(rows: list[dict], path: Path) -> None:
    """Compare threshold and directly calculated beam-quality metrics."""
    x = np.arange(len(rows))
    labels = [row["candidate"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 11.0), sharex=True)
    axes[0].bar(x, [row["actual_threshold_cm-1"] for row in rows])
    axes[0].set_ylabel("lowest threshold (cm$^{-1}$)")
    axes[1].plot(x, [row["center_to_peak"] for row in rows], "o-", label="center/max")
    axes[1].plot(
        x,
        [row["encircled_power_1deg"] for row in rows],
        "s-",
        label="power within 1 deg",
    )
    axes[1].set_ylabel("normalized beam metric")
    axes[1].legend()
    axes[2].bar(x, [row["score"] for row in rows], color="#59a14f")
    axes[2].set(ylabel="design score", xticks=x, xticklabels=labels)
    axes[2].tick_params(axis="x", rotation=32)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Three-hole triangular-cell screening (higher score is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    output = OUTPUT_DIRECTORY.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("Screening centered-main + a1/a2-satellite three-hole cells")
    summary_rows: list[dict] = []
    for candidate in CANDIDATES:
        print(f"  screening {candidate.name} ...", flush=True)
        cell, _, _, summary = evaluate(candidate)
        _, actual, dark_floor, threshold_gap, score = summary
        row = {
            "candidate": candidate.name,
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
        summary_rows.append(row)
        print(
            f"    {actual['mode']}: alpha={actual['threshold_cm-1']:.2f}, "
            f"center={actual['center_to_peak']:.2f}, peak={actual['peak_offset_deg']:.2f} deg, "
            f"E1={actual['encircled_power_1deg']:.2f}, score={score:.1f}"
        )
    winner_row = max(summary_rows, key=lambda item: item["score"])
    winner = next(item for item in CANDIDATES if item.name == winner_row["candidate"])
    with (output / "01_screening.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_screening(summary_rows, output / "01_screening.png")

    print(f"Rebuilding winner with production settings: {winner.name}")
    cell, result = build_model(
        winner,
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
    records, actual, dark_floor, threshold_gap, final_score = summarize_modes(result, modes)
    final_report = {
        "winner": winner.name,
        "explanation": winner.explanation,
        "fill_fraction": cell.fill_fraction,
        "inclusions": [item.__dict__ for item in winner.inclusions],
        "actual_lowest_threshold_mode": actual,
        "dark_competitor_floor_cm-1": dark_floor,
        "nearest_mode_gap_cm-1": threshold_gap,
        "final_score": final_score,
        "all_modes": records,
        "screening_settings": {
            "truncation": SCREEN_TRUNCATION,
            "vertical_step_nm": SCREEN_VERTICAL_STEP_NM,
            "radius_cells": SCREEN_RADIUS_CELLS,
        },
        "final_settings": {
            "truncation": FINAL_TRUNCATION,
            "vertical_step_nm": FINAL_VERTICAL_STEP_NM,
            "radius_cells": list(FINAL_RADIUS_CELLS),
            "reported_eigenvalues": "linear 1/N extrapolation to zero grid spacing",
        },
        "warning": (
            "Reduced-order design exploration. Verify convergence, fabrication tolerances "
            "and full-wave RCWA/FEM/FDTD before fabrication."
        ),
    }
    (output / "02_winner.json").write_text(
        json.dumps(final_report, indent=2), encoding="utf-8"
    )
    write_triangular_finite_table(modes, result, output / "02_winner_modes.csv")
    plot_triangular_lattice(cell, output / "02_winner_lattice.png")
    plot_triangular_thresholds(modes, output / "03_winner_thresholds.png")
    plot_triangular_grid_convergence(
        convergence, DEVICE_RADIUS_UM, output / "03_winner_grid_convergence.png"
    )
    plot_six_band_edge_states(result, output / "04_winner_unit_cell_states.png", points=101)
    plot_triangular_mode_atlas(modes, result, output / "05_winner_mode_atlas.png")
    selected = modes[actual["mode"]]
    plot_triangular_far_field_diagnostics(
        selected, result, output / "06_winner_lasing_mode_far_field.png"
    )
    print(
        f"winner={winner.name}, lasing mode={selected.name}, "
        f"alpha={selected.alpha_per_m / 100.0:.3f} cm^-1, "
        f"center/max={actual['center_to_peak']:.3f}, "
        f"peak offset={actual['peak_offset_deg']:.3f} deg"
    )


if __name__ == "__main__":
    main()
