"""Search triangular-cell motifs for a low-threshold center-bright finite mode.

The screening metric is calculated, not drawn: every candidate rebuilds the
Fourier coefficients and six-wave C matrix, solves the finite circular device,
and evaluates the complex vector far field.  Coarse settings rank candidates;
the winner is rebuilt with the production settings below and saved separately.

中文：本脚本尝试提高中心暗斑/奇点模式的阈值，同时寻找中心亮、近单瓣且阈值较低的模式。
筛选阶段使用较低截断阶数和较粗有限网格，最终候选会用生产参数重算。它是设计探索，不能替代
制造公差扫描或 RCWA/FEM/FDTD 复核。
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


# Fast ranking and final verification controls.
LATTICE_CONSTANT_NM = 341.0
BACKGROUND_EPSILON = 12.7449
HOLE_EPSILON = 1.0
WAVELENGTH_GUESS_NM = 995.0
DEVICE_RADIUS_UM = 30.0
SCREEN_TRUNCATION = 4
SCREEN_VERTICAL_STEP_NM = 6.0
SCREEN_RADIUS_CELLS = 5
FINAL_TRUNCATION = 10
FINAL_VERTICAL_STEP_NM = 3.0
FINAL_RADIUS_CELLS = (7, 9, 11)
BRIGHT_CENTER_MINIMUM = 0.45
SINGULAR_CENTER_MAXIMUM = 0.12
OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "results" / "custom_semiconductor_triangular_designed_bright_mode"
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


def circle(radius: float, center=(0.0, 0.0)) -> TriangularEllipse:
    return TriangularEllipse(center_fractional=center, radii_over_a=(radius, radius))


CANDIDATES = (
    Candidate("reference_circle", "Liang f approximately 0.15 reference", (circle(0.20),)),
    Candidate(
        "centered_ellipse_x",
        "C2 ellipse opens formerly protected channels",
        (TriangularEllipse(radii_over_a=(0.245, 0.170), angle_deg=0.0),),
    ),
    Candidate(
        "centered_ellipse_30deg",
        "same C2 perturbation rotated relative to the six waves",
        (TriangularEllipse(radii_over_a=(0.245, 0.170), angle_deg=30.0),),
    ),
    Candidate(
        "balanced_elliptic_dimer",
        "balanced counter-rotated double ellipse with destructive radiation paths",
        (
            TriangularEllipse(
                center_fractional=(-0.19, 0.0), radii_over_a=(0.19, 0.11), angle_deg=15.0
            ),
            TriangularEllipse(
                center_fractional=(0.19, 0.0), radii_over_a=(0.19, 0.11), angle_deg=-15.0
            ),
        ),
    ),
    Candidate(
        "asymmetric_circle_dimer",
        "unequal dimer breaks C6 and inversion symmetry",
        (
            circle(0.180, (-0.12, -0.02)),
            circle(0.095, (0.25, 0.07)),
        ),
    ),
    Candidate(
        "asymmetric_ellipse_dimer",
        "unequal rotated ellipses provide amplitude and phase control",
        (
            TriangularEllipse(
                center_fractional=(-0.12, -0.03),
                radii_over_a=(0.190, 0.135),
                angle_deg=20.0,
            ),
            TriangularEllipse(
                center_fractional=(0.25, 0.08),
                radii_over_a=(0.105, 0.075),
                angle_deg=-25.0,
            ),
        ),
    ),
)


def build_model(candidate: Candidate, settings: TriangularCWTSettings):
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
    """Return per-mode metrics, the best center-bright state, and design score."""
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
        })
    bright = [item for item in records if item["center_to_peak"] >= BRIGHT_CENTER_MINIMUM]
    singular = [item for item in records if item["center_to_peak"] <= SINGULAR_CENTER_MAXIMUM]
    if not bright:
        score = -1e9
        best_bright = min(records, key=lambda item: item["threshold_cm-1"])
    else:
        best_bright = min(bright, key=lambda item: item["threshold_cm-1"])
        singular_floor = min(
            (item["threshold_cm-1"] for item in singular), default=best_bright["threshold_cm-1"]
        )
        # Positive threshold discrimination is the primary goal; a strong
        # on-axis field and compact second moment break near ties.
        score = (
            singular_floor - best_bright["threshold_cm-1"]
            + 35.0 * best_bright["center_to_peak"]
            - 2.0 * best_bright["full_rms_deg"]
        )
    return records, best_bright, float(score)


def evaluate(candidate: Candidate, truncation: int, vertical_step: float, cells: int):
    cell, result = build_model(
        candidate,
        TriangularCWTSettings(
            truncation_order=truncation,
            vertical_step_nm=vertical_step,
        ),
    )
    modes = solve_triangular_finite_modes(
        result,
        TriangularFiniteSpec(
            radius_um=DEVICE_RADIUS_UM,
            radius_cells=cells,
            eigensolutions_per_target=7,
        ),
    )
    records, best_bright, score = summarize_modes(result, modes)
    return cell, result, modes, records, best_bright, float(score)


def plot_screening(rows: list[dict], path: Path) -> None:
    names = [row["candidate"] for row in rows]
    scores = [row["score"] for row in rows]
    bright_losses = [row["best_bright_threshold_cm-1"] for row in rows]
    singular_losses = [row["lowest_singular_threshold_cm-1"] for row in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=True)
    axes[0].bar(x - 0.18, bright_losses, width=0.36, label="best center-bright")
    axes[0].bar(x + 0.18, singular_losses, width=0.36, label="lowest center-dark")
    axes[0].set_ylabel("finite threshold (cm^-1)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, scores, color="#59a14f")
    axes[1].set(ylabel="design score", xticks=x, xticklabels=names)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Coarse six-wave motif screening (higher score is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    output = OUTPUT_DIRECTORY.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("Coarse geometry-derived six-wave design screening")
    summary_rows: list[dict] = []
    evaluations = {}
    for candidate in CANDIDATES:
        print(f"  screening {candidate.name} ...", flush=True)
        evaluation = evaluate(
            candidate, SCREEN_TRUNCATION, SCREEN_VERTICAL_STEP_NM, SCREEN_RADIUS_CELLS
        )
        evaluations[candidate.name] = evaluation
        cell, _, _, records, best_bright, score = evaluation
        singular = [item for item in records if item["center_to_peak"] <= SINGULAR_CENTER_MAXIMUM]
        singular_floor = min(
            (item["threshold_cm-1"] for item in singular),
            default=best_bright["threshold_cm-1"],
        )
        summary_rows.append({
            "candidate": candidate.name,
            "fill_fraction": cell.fill_fraction,
            "best_bright_mode": best_bright["mode"],
            "best_bright_threshold_cm-1": best_bright["threshold_cm-1"],
            "best_bright_center_to_peak": best_bright["center_to_peak"],
            "best_bright_full_rms_deg": best_bright["full_rms_deg"],
            "lowest_singular_threshold_cm-1": singular_floor,
            "score": score,
        })
        print(
            f"    bright={best_bright['mode']}, alpha={best_bright['threshold_cm-1']:.2f}, "
            f"center={best_bright['center_to_peak']:.2f}, singular floor={singular_floor:.2f}, "
            f"score={score:.2f}"
        )
    winner_row = max(summary_rows, key=lambda item: item["score"])
    winner = next(item for item in CANDIDATES if item.name == winner_row["candidate"])
    with (output / "01_screening.csv").open("w", newline="", encoding="utf-8-sig") as stream:
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
    records, best_bright, final_score = summarize_modes(result, modes)
    final_report = {
        "winner": winner.name,
        "explanation": winner.explanation,
        "fill_fraction": cell.fill_fraction,
        "inclusions": [item.__dict__ for item in winner.inclusions],
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
        "final_score": final_score,
        "best_center_bright_mode": best_bright,
        "all_modes": records,
        "warning": (
            "Coarse discrete design search; verify truncation, finite-grid and fabrication "
            "tolerance convergence plus RCWA/FEM/FDTD before fabrication."
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
    selected = modes[best_bright["mode"]]
    plot_triangular_far_field_diagnostics(
        selected, result, output / "06_winner_bright_mode_far_field.png"
    )
    print(
        f"winner={winner.name}, bright mode={selected.name}, "
        f"alpha={selected.alpha_per_m/100.0:.3f} cm^-1"
    )


if __name__ == "__main__":
    main()

