"""Equal-size cold-cavity comparison of square four-wave and triangular six-wave PCSELs.

The two Bravais lattices use the same Inoue semiconductor layer stack, material
loss, 300-um transverse bounding box, and carrier/gain parameters.  The script
compares a square aperture against circular, regular-hexagonal, and square
apertures for the triangular six-wave design.  It reports both the passive
amplitude loss and the modal gain/current required by the Inoue convention
``g_modal = alpha_internal + 2*alpha_mode``.

中文：本脚本用于公平比较，而不是再次把 30-um 三角器件与 300-um 方形器件直接相比。
三角晶格仍使用独立六波矩阵；没有把方形四波时域求解器套到六波场上。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e

from pcselsim.custom_analysis import (
    MODE_NAMES,
    plot_grid_convergence,
    plot_threshold_summary,
    plot_vector_far_field_diagnostics,
    solve_finite_modes_converged,
    vector_far_field_metrics,
    write_mode_table,
)
from pcselsim.triangular_finite import (
    TriangularFiniteSpec,
    plot_triangular_far_field_diagnostics,
    plot_triangular_grid_convergence,
    plot_triangular_mode_atlas,
    plot_triangular_thresholds,
    solve_triangular_finite_modes_converged,
    triangular_vector_far_field,
    write_triangular_finite_table,
)
from pcselsim.triangular_six_wave import (
    TRIANGULAR_MODE_NAMES,
    TriangularCWTSettings,
    TriangularLatticeCell,
    build_triangular_coupling,
    plot_triangular_lattice,
)
from pcselsim.vertical import Layer, LayerStack

if __package__:
    from .design_square_lattice_multiobjective import (
        CANDIDATES as SQUARE_CANDIDATES,
        build_model as build_square_model,
    )
    from .design_triangular_triangle_main_refinement import (
        DesignParameters,
        candidate_from_parameters,
    )
    from .run_custom_semiconductor_pcsel import (
        ACTIVE_INDEX,
        ACTIVE_LAYER_NAME,
        ACTIVE_THICKNESS_NM,
        BACKGROUND_INDEX,
        BOTTOM_CLADDING_INDEX,
        CARRIER_LIFETIME_NS,
        INTERNAL_LOSS_CM,
        LATTICE_CONSTANT_NM as SQUARE_LATTICE_CONSTANT_NM,
        LAYERS,
        MAXIMUM_GAIN_CM,
        PC_LAYER_NAME,
        TOP_CLADDING_INDEX,
        TRANSPARENCY_DENSITY_CM3,
        VERTICAL_PADDING_UM,
        WAVELENGTH_GUESS_NM,
        ZERO_CARRIER_GAIN_CM,
    )
else:
    from design_square_lattice_multiobjective import (
        CANDIDATES as SQUARE_CANDIDATES,
        build_model as build_square_model,
    )
    from design_triangular_triangle_main_refinement import (
        DesignParameters,
        candidate_from_parameters,
    )
    from run_custom_semiconductor_pcsel import (
        ACTIVE_INDEX,
        ACTIVE_LAYER_NAME,
        ACTIVE_THICKNESS_NM,
        BACKGROUND_INDEX,
        BOTTOM_CLADDING_INDEX,
        CARRIER_LIFETIME_NS,
        INTERNAL_LOSS_CM,
        LATTICE_CONSTANT_NM as SQUARE_LATTICE_CONSTANT_NM,
        LAYERS,
        MAXIMUM_GAIN_CM,
        PC_LAYER_NAME,
        TOP_CLADDING_INDEX,
        TRANSPARENCY_DENSITY_CM3,
        VERTICAL_PADDING_UM,
        WAVELENGTH_GUESS_NM,
        ZERO_CARRIER_GAIN_CM,
    )


# ===========================================================================
# COMPARISON PANEL / 公平对比参数区
# ===========================================================================
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "equal_size_square_triangular_comparison"
BOUNDING_SIZE_UM = 300.0
HALF_SIZE_UM = BOUNDING_SIZE_UM / 2.0
TRIANGULAR_APERTURES = ("circle", "hexagon", "square")
TRIANGULAR_GRIDS = (7, 9, 11)
SQUARE_GRIDS = (13, 17, 21)
TRUNCATION_ORDER = 10
VERTICAL_STEP_NM = 3.0
FAR_FIELD_VIEW_DEG = 1.5
FAR_FIELD_SAMPLES = 161
HEXAGON_SIZE_SWEEP_UM = (60.0, 100.0, 150.0)  # circumradius; full width is 2R
HEXAGON_SWEEP_GRIDS = (7, 9, 11)

# Match the magnitude of the first reciprocal vectors:
# 2*pi/a_square = 4*pi/(sqrt(3)*a_triangular).
TRIANGULAR_LATTICE_CONSTANT_NM = (
    2.0 * SQUARE_LATTICE_CONSTANT_NM / np.sqrt(3.0)
)

TRIANGULAR_PARAMETERS = DesignParameters(
    name="triangle_sat_08_equal_size",
    satellite_shape="triangle",
    main_radius=0.21,
    main_angle_deg=15.0,
    distance=0.44,
    spread_deg=-6.0,
    triangle_orientation="inward",
)
SQUARE_CANDIDATE_NAME = "triangle_main_a045"


def aperture_area_m2(shape: str, half_size_um: float) -> float:
    radius_m = half_size_um * 1e-6
    if shape == "circle":
        return float(np.pi * radius_m**2)
    if shape == "hexagon":
        return float(1.5 * np.sqrt(3.0) * radius_m**2)
    if shape == "square":
        return float((2.0 * radius_m) ** 2)
    raise ValueError(f"unknown aperture shape: {shape}")


def threshold_audit(
    alpha_mode_per_m: float,
    area_m2: float,
    confinement: float,
    effective_index: float,
) -> dict[str, float]:
    """Apply the same Inoue amplitude/gain convention to both lattices."""
    overlap = confinement * ACTIVE_INDEX / effective_index
    required_modal_gain_m = INTERNAL_LOSS_CM * 100.0 + 2.0 * alpha_mode_per_m
    required_material_gain_m = required_modal_gain_m / overlap
    maximum_gain_m = MAXIMUM_GAIN_CM * 100.0
    if required_material_gain_m >= maximum_gain_m:
        return {
            "required_modal_gain_cm-1": required_modal_gain_m / 100.0,
            "required_material_gain_cm-1": required_material_gain_m / 100.0,
            "threshold_density_cm-3": float("nan"),
            "threshold_current_A": float("nan"),
            "threshold_current_density_A_cm-2": float("nan"),
        }
    ratio = required_material_gain_m / maximum_gain_m
    gmax_over_minus_g0 = MAXIMUM_GAIN_CM / (-ZERO_CARRIER_GAIN_CM)
    density_cm3 = TRANSPARENCY_DENSITY_CM3 * (
        1.0 + ratio * gmax_over_minus_g0
    ) / (1.0 - ratio)
    current_a = (
        e * ACTIVE_THICKNESS_NM * 1e-9 * area_m2
        * density_cm3 * 1e6 / (CARRIER_LIFETIME_NS * 1e-9)
    )
    return {
        "required_modal_gain_cm-1": required_modal_gain_m / 100.0,
        "required_material_gain_cm-1": required_material_gain_m / 100.0,
        "threshold_density_cm-3": density_cm3,
        "threshold_current_A": float(current_a),
        "threshold_current_density_A_cm-2": float(current_a / (area_m2 * 1e4)),
    }


def cold_cavity_q(
    wavelength_nm: float,
    group_index: float,
    alpha_mode_per_m: float,
) -> float:
    total_amplitude_loss = alpha_mode_per_m + 0.5 * INTERNAL_LOSS_CM * 100.0
    return float(
        np.pi * group_index / (wavelength_nm * 1e-9 * total_amplitude_loss)
    )


def build_common_triangular_model():
    candidate = candidate_from_parameters(TRIANGULAR_PARAMETERS)
    cell = TriangularLatticeCell(
        background_epsilon=BACKGROUND_INDEX**2,
        inclusions=candidate.inclusions,
    )
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
    model = build_triangular_coupling(
        cell,
        stack,
        PC_LAYER_NAME,
        TRIANGULAR_LATTICE_CONSTANT_NM,
        WAVELENGTH_GUESS_NM,
        TriangularCWTSettings(
            truncation_order=TRUNCATION_ORDER,
            vertical_step_nm=VERTICAL_STEP_NM,
        ),
    )
    return candidate, cell, model


def summarize_triangular(
    shape: str,
    modes,
    model,
    half_size_um: float = HALF_SIZE_UM,
) -> tuple[dict, list[dict]]:
    records = []
    for name in TRIANGULAR_MODE_NAMES:
        mode = modes[name]
        far = triangular_vector_far_field(
            mode,
            model.bragg_wavelength_nm,
            view_deg=FAR_FIELD_VIEW_DEG,
            samples=FAR_FIELD_SAMPLES,
        )
        records.append({
            "mode": name,
            "alpha_mode_cm-1": mode.alpha_per_m / 100.0,
            "center_to_peak": far.center_to_peak,
            "peak_offset_deg": far.peak_offset_deg,
            "centroid_offset_deg": far.centroid_offset_deg,
            "ellipticity": far.ellipticity,
            "encircled_power_0p5deg": far.encircled_power_0p5deg,
            "encircled_power_1deg": far.encircled_power_1deg,
            "full_rms_deg": far.full_rms_divergence_deg,
        })
    actual = min(records, key=lambda item: item["alpha_mode_cm-1"])
    gap = min(
        item["alpha_mode_cm-1"] for item in records if item["mode"] != actual["mode"]
    ) - actual["alpha_mode_cm-1"]
    confinement = model.vertical_mode.confinement[ACTIVE_LAYER_NAME]
    audit = threshold_audit(
        actual["alpha_mode_cm-1"] * 100.0,
        aperture_area_m2(shape, half_size_um),
        confinement,
        model.effective_index,
    )
    row = {
        "lattice": "triangular_six_wave",
        "aperture": shape,
        "bounding_size_um": 2.0 * half_size_um,
        "fill_fraction": 0.15,
        "wavelength_nm": model.bragg_wavelength_nm,
        "effective_index": model.effective_index,
        "group_index": model.group_index,
        "active_confinement": confinement,
        **actual,
        "mode_gap_cm-1": gap,
        "cold_cavity_Q": cold_cavity_q(
            model.bragg_wavelength_nm,
            model.group_index,
            actual["alpha_mode_cm-1"] * 100.0,
        ),
        **audit,
    }
    row["acceptable"] = bool(
        row["center_to_peak"] >= 0.75
        and row["peak_offset_deg"] <= 0.25
        and row["encircled_power_0p5deg"] >= 0.70
        and row["mode_gap_cm-1"] >= 1.0
    )
    return row, records


def summarize_square(modes, result, spec) -> tuple[dict, list[dict]]:
    records = []
    for name in MODE_NAMES:
        mode = modes[name]
        far = vector_far_field_metrics(spec, mode, view_deg=FAR_FIELD_VIEW_DEG)
        records.append({
            "mode": name,
            "alpha_mode_cm-1": mode.alpha_per_m / 100.0,
            "center_to_peak": far.center_to_peak,
            "peak_offset_deg": far.peak_offset_deg,
            "centroid_offset_deg": far.centroid_offset_deg,
            "ellipticity": far.ellipticity,
            "encircled_power_0p5deg": far.encircled_power_0p5deg,
            "encircled_power_1deg": far.encircled_power_1deg,
            "full_rms_deg": far.full_rms_divergence_deg,
        })
    actual = min(records, key=lambda item: item["alpha_mode_cm-1"])
    gap = min(
        item["alpha_mode_cm-1"] for item in records if item["mode"] != actual["mode"]
    ) - actual["alpha_mode_cm-1"]
    confinement = result.vertical_mode.confinement[ACTIVE_LAYER_NAME]
    audit = threshold_audit(
        actual["alpha_mode_cm-1"] * 100.0,
        aperture_area_m2("square", HALF_SIZE_UM),
        confinement,
        result.effective_index,
    )
    row = {
        "lattice": "square_four_wave",
        "aperture": "square",
        "bounding_size_um": BOUNDING_SIZE_UM,
        "fill_fraction": 0.05,
        "wavelength_nm": result.bragg_wavelength_nm,
        "effective_index": result.effective_index,
        "group_index": result.group_index,
        "active_confinement": confinement,
        **actual,
        "mode_gap_cm-1": gap,
        "cold_cavity_Q": cold_cavity_q(
            result.bragg_wavelength_nm,
            result.group_index,
            actual["alpha_mode_cm-1"] * 100.0,
        ),
        **audit,
    }
    row["acceptable"] = bool(
        row["center_to_peak"] >= 0.75
        and row["peak_offset_deg"] <= 0.25
        and row["encircled_power_0p5deg"] >= 0.70
        and row["mode_gap_cm-1"] >= 1.0
    )
    return row, records


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(rows: list[dict], path: Path) -> None:
    labels = [
        "square\nsquare" if row["lattice"].startswith("square")
        else f"triangular\n{row['aperture']}"
        for row in rows
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    panels = (
        ("required_modal_gain_cm-1", "required modal gain (cm$^{-1}$)"),
        ("mode_gap_cm-1", "nearest-mode gap (cm$^{-1}$)"),
        ("encircled_power_0p5deg", "encircled power within 0.5 deg"),
        ("full_rms_deg", "full RMS divergence (deg)"),
    )
    colors = ["#4c78a8"] + ["#f58518", "#54a24b", "#e45756"]
    for ax, (key, ylabel) in zip(axes.flat, panels, strict=True):
        values = [row[key] for row in rows]
        bars = ax.bar(labels, values, color=colors)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3g}",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Equal 300-um bounding box; common semiconductor stack")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_size_sweep(rows: list[dict], path: Path) -> None:
    width = np.asarray([item["bounding_size_um"] for item in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for ax, key, ylabel in (
        (axes[0], "required_modal_gain_cm-1", "required modal gain (cm$^{-1}$)"),
        (axes[1], "full_rms_deg", "full RMS divergence (deg)"),
        (axes[2], "mode_gap_cm-1", "mode gap (cm$^{-1}$)"),
    ):
        ax.plot(width, [item[key] for item in rows], "o-")
        ax.set(xlabel="hexagon vertex-to-vertex size (um)", ylabel=ylabel)
        ax.grid(alpha=0.2)
    fig.suptitle("Triangular six-wave hexagonal-aperture size sweep")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    output = OUTPUT_DIRECTORY.resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("Building common-stack square four-wave reference ...", flush=True)
    square_candidate = next(
        item for item in SQUARE_CANDIDATES if item.name == SQUARE_CANDIDATE_NAME
    )
    _, square_result, square_spec = build_square_model(
        square_candidate, TRUNCATION_ORDER, VERTICAL_STEP_NM
    )
    square_modes, square_convergence = solve_finite_modes_converged(
        square_spec,
        SQUARE_GRIDS,
        square_result.coupling_m,
        square_result.eigenvectors,
        square_result.radiation_fields,
    )
    square_row, square_records = summarize_square(
        square_modes, square_result, square_spec
    )
    square_output = output / "square_reference"
    square_output.mkdir(exist_ok=True)
    write_mode_table(square_output, square_spec, square_modes)
    plot_grid_convergence(
        square_spec, square_convergence, square_output / "01_convergence.png"
    )
    plot_threshold_summary(
        square_spec, square_modes, square_output / "02_thresholds.png"
    )
    plot_vector_far_field_diagnostics(
        square_spec,
        square_modes[square_row["mode"]],
        square_output / "03_far_field.png",
    )

    print("Building common-stack triangular six-wave model ...", flush=True)
    triangular_candidate, triangular_cell, triangular_model = (
        build_common_triangular_model()
    )
    plot_triangular_lattice(
        triangular_cell, output / "01_triangular_common_stack_lattice.png"
    )

    comparison_rows = [square_row]
    triangular_modes_by_shape = {}
    for shape in TRIANGULAR_APERTURES:
        print(f"Solving 300-um triangular {shape} aperture ...", flush=True)
        spec = TriangularFiniteSpec(
            radius_um=HALF_SIZE_UM,
            radius_cells=TRIANGULAR_GRIDS[-1],
            internal_loss_cm=0.0,
            aperture_shape=shape,
        )
        modes, convergence = solve_triangular_finite_modes_converged(
            triangular_model, spec, TRIANGULAR_GRIDS
        )
        row, records = summarize_triangular(shape, modes, triangular_model)
        comparison_rows.append(row)
        triangular_modes_by_shape[shape] = modes
        shape_output = output / f"triangular_{shape}"
        shape_output.mkdir(exist_ok=True)
        write_triangular_finite_table(
            modes, triangular_model, shape_output / "01_modes.csv"
        )
        plot_triangular_grid_convergence(
            convergence, HALF_SIZE_UM, shape_output / "02_convergence.png"
        )
        plot_triangular_thresholds(modes, shape_output / "03_thresholds.png")
        plot_triangular_far_field_diagnostics(
            modes[row["mode"]],
            triangular_model,
            shape_output / "04_far_field.png",
            view_deg=FAR_FIELD_VIEW_DEG,
        )
        print(
            f"  {shape}: mode={row['mode']}, gain={row['required_modal_gain_cm-1']:.3f}, "
            f"gap={row['mode_gap_cm-1']:.3f}, E0.5={row['encircled_power_0p5deg']:.3f}"
        )

    acceptable_triangular = [
        row for row in comparison_rows[1:] if row["acceptable"]
    ]
    selection_pool = acceptable_triangular or comparison_rows[1:]
    recommended_triangular = max(
        selection_pool,
        key=lambda row: (
            80.0 * row["center_to_peak"]
            + 60.0 * row["encircled_power_0p5deg"]
            - 20.0 * row["full_rms_deg"]
            + 0.5 * min(row["mode_gap_cm-1"], 30.0)
            - 0.5 * row["required_modal_gain_cm-1"]
        ),
    )
    plot_triangular_mode_atlas(
        triangular_modes_by_shape[recommended_triangular["aperture"]],
        triangular_model,
        output / "02_recommended_triangular_mode_atlas.png",
        view_deg=FAR_FIELD_VIEW_DEG,
    )

    print("Running triangular hexagonal-aperture size sweep ...", flush=True)
    size_rows = []
    for half_size in HEXAGON_SIZE_SWEEP_UM:
        modes, _ = solve_triangular_finite_modes_converged(
            triangular_model,
            TriangularFiniteSpec(
                radius_um=half_size,
                radius_cells=HEXAGON_SWEEP_GRIDS[-1],
                internal_loss_cm=0.0,
                aperture_shape="hexagon",
            ),
            HEXAGON_SWEEP_GRIDS,
        )
        row, _ = summarize_triangular(
            "hexagon", modes, triangular_model, half_size_um=half_size
        )
        size_rows.append(row)

    write_rows(output / "03_equal_size_comparison.csv", comparison_rows)
    write_rows(output / "04_hexagon_size_sweep.csv", size_rows)
    plot_comparison(comparison_rows, output / "03_equal_size_comparison.png")
    plot_size_sweep(size_rows, output / "04_hexagon_size_sweep.png")

    report = {
        "comparison_basis": {
            "bounding_size_um": BOUNDING_SIZE_UM,
            "common_vertical_stack": "Inoue 2019 idealized semiconductor stack",
            "internal_loss_cm-1": INTERNAL_LOSS_CM,
            "square_lattice_constant_nm": SQUARE_LATTICE_CONSTANT_NM,
            "triangular_lattice_constant_nm": TRIANGULAR_LATTICE_CONSTANT_NM,
            "gain_condition": "g_modal = alpha_internal + 2 alpha_mode",
            "note": (
                "Threshold-current audit assumes uniform carrier density and the same "
                "material gain law. It is not a six-wave nonlinear time-domain result."
            ),
        },
        "square_reference": square_row,
        "recommended_triangular": recommended_triangular,
        "all_equal_size_cases": comparison_rows,
        "triangular_geometry": {
            "candidate": triangular_candidate.name,
            "main_radius_over_a": TRIANGULAR_PARAMETERS.main_radius,
            "main_angle_deg": TRIANGULAR_PARAMETERS.main_angle_deg,
            "satellite_distance_over_a": TRIANGULAR_PARAMETERS.distance,
            "satellite_spread_deg": TRIANGULAR_PARAMETERS.spread_deg,
            "satellite_orientation": TRIANGULAR_PARAMETERS.triangle_orientation,
        },
        "remaining_limitations": [
            "No six-wave carrier-field nonlinear time-domain solver yet.",
            "Ideal sharp triangular holes; fabrication corner rounding is not included.",
            "Threshold current uses uniform injection and omits thermal/spatial-hole burning.",
            "Final fabrication candidates still require RCWA/FEM/FDTD validation.",
        ],
    }
    (output / "05_comparison_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "Recommended triangular aperture:",
        recommended_triangular["aperture"],
        "mode", recommended_triangular["mode"],
    )
    print(f"Finished. Results: {output}")


if __name__ == "__main__":
    main()
