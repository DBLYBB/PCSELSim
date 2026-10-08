"""Complete cold-cavity workflow for the historical three-triangle candidate.

2026-10-08 audit / 本轮审计：keep this geometry as a regression baseline,
not a certified manufacturing optimum. Low-grid extrapolated thresholds are
grid-sensitive; see docs/research_design_audit_20261008_zh.md before quoting.

Geometry: one central triangular hole plus two smaller inward-pointing
triangular satellite holes in a triangular Bravais lattice.  The device uses
the common Inoue semiconductor stack and a 300-um square aperture selected by
the equal-size comparison.

This is a true six-wave optical calculation.  The pump sweep is a uniform,
small-signal threshold audit; it is not presented as a nonlinear six-wave
carrier-field time-domain simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from scipy.constants import e

from pcselsim.band_structure import plot_band_diagram
from pcselsim.custom_analysis import plot_layer_stack, plot_vertical_mode
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
    TRIANGULAR_BASIC_ORDERS,
    TRIANGULAR_MODE_NAMES,
    TriangularCWTSettings,
    TriangularLatticeCell,
    build_triangular_coupling,
    plot_radiation_constants,
    plot_six_band_edge_states,
    plot_triangular_lattice,
    plot_triangular_reciprocal_space,
    triangular_band_diagram,
)
from pcselsim.vertical import Layer, LayerStack
from pcselsim.fabrication import inclusion_boundary, TRIANGULAR_DIRECT

if __package__:
    from .compare_square_triangular_equal_size import (
        aperture_area_m2,
        cold_cavity_q,
        threshold_audit,
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
        DN_DN_CM3,
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
    from compare_square_triangular_equal_size import (
        aperture_area_m2,
        cold_cavity_q,
        threshold_audit,
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
        DN_DN_CM3,
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
# PARAMETER PANEL A - optimized three-triangle unit cell
# ===========================================================================
RUN_TITLE = "BEST TRIANGULAR PCSEL: TRIANGLE MAIN + TWO TRIANGLE SATELLITES"
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "best_triangular_three_triangle_full"

LATTICE_CONSTANT_NM = 2.0 * SQUARE_LATTICE_CONSTANT_NM / np.sqrt(3.0)
HOLE_EPSILON = 1.0
DESIGN = DesignParameters(
    name="triangle_sat_manufacturable_300um",
    satellite_shape="triangle",
    main_radius=0.22,
    main_angle_deg=0.0,
    distance=0.44,
    spread_deg=-6.0,
    triangle_orientation="inward",
    main_corner_fraction=0.10,
    satellite_corner_fraction=0.10,
)


# ===========================================================================
# PARAMETER PANEL B - finite device and numerical convergence
# ===========================================================================
DEVICE_SHAPE = "square"
DEVICE_HALF_SIZE_UM = 150.0       # 300-um square side
FINITE_GRIDS = (7, 9, 11)
CWT_TRUNCATION_ORDER = 10
VERTICAL_STEP_NM = 3.0
BAND_Q_MAX_2PI_OVER_A = 0.01
BAND_POINTS = 241
UNIT_CELL_FIELD_POINTS = 121
FAR_FIELD_VIEW_DEG = 1.5
FAR_FIELD_SAMPLES = 161
PUMP_RATIOS = tuple(np.linspace(0.5, 1.5, 41))


# ===========================================================================
# STEP SWITCHES
# ===========================================================================
STEPS = {
    "01_parameters": True,
    "02_lattice": True,
    "03_k_space": True,
    "04_vertical_mode": True,
    "05_band_structure": True,
    "06_radiation_constants": True,
    "07_band_edge_states": True,
    "08_finite_modes": True,
    "09_mode_atlas": True,
    "10_best_far_field": True,
    "11_threshold_audit": True,
    "12_small_signal_pump": True,
}


def build_model_for_candidate(candidate, settings: TriangularCWTSettings | None = None):
    """Build the common Inoue-stack six-wave model for any triangular candidate."""
    cell = TriangularLatticeCell(
        background_epsilon=BACKGROUND_INDEX**2,
        inclusions=candidate.inclusions,
    )
    average_index = float(np.sqrt(cell.fourier_epsilon(0, 0).real))
    display_layers = tuple(
        replace(layer, refractive_index=average_index)
        if layer.name == PC_LAYER_NAME else layer
        for layer in LAYERS
    )
    stack = LayerStack(
        layers=tuple(
            Layer(layer.name, layer.thickness_nm, layer.refractive_index)
            for layer in display_layers
        ),
        top_index=TOP_CLADDING_INDEX,
        bottom_index=BOTTOM_CLADDING_INDEX,
        padding_um=VERTICAL_PADDING_UM,
    )
    model = build_triangular_coupling(
        cell,
        stack,
        PC_LAYER_NAME,
        LATTICE_CONSTANT_NM,
        WAVELENGTH_GUESS_NM,
        settings or TriangularCWTSettings(
            truncation_order=CWT_TRUNCATION_ORDER,
            vertical_step_nm=VERTICAL_STEP_NM,
        ),
    )
    return cell, display_layers, model


def build_model():
    candidate = candidate_from_parameters(DESIGN)
    cell, display_layers, model = build_model_for_candidate(candidate)
    return candidate, cell, display_layers, model


def _cartesian_vertices(inclusion, origin=(0.0, 0.0)) -> np.ndarray:
    vertices = inclusion_boundary(inclusion, TRIANGULAR_DIRECT)
    return vertices + np.asarray(origin, dtype=float)


def plot_device_overview(candidate, cell, path: Path) -> None:
    """Show the complete 300-um device and two successive lattice zoom levels."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    half = DEVICE_HALF_SIZE_UM
    side = 2.0 * half
    primitive_area_um2 = (
        np.sqrt(3.0) / 2.0 * (LATTICE_CONSTANT_NM * 1e-3) ** 2
    )
    cell_count = side**2 / primitive_area_um2

    ax = axes[0]
    ax.add_patch(Rectangle(
        (-half, -half), side, side,
        facecolor="#d9ecf7", edgecolor="#174a7e", lw=2.2,
        label="finite PC / uniform pumped area",
    ))
    zoom_half = 3.0
    ax.add_patch(Rectangle(
        (-zoom_half, -zoom_half), 2 * zoom_half, 2 * zoom_half,
        fill=False, edgecolor="#d62728", lw=1.5,
    ))
    ax.annotate("lattice zoom", (zoom_half, zoom_half), (55.0, 85.0),
                arrowprops={"arrowstyle": "->", "color": "#d62728"},
                color="#d62728")
    ax.set(
        aspect="equal", xlim=(-170, 170), ylim=(-170, 170),
        xlabel="x (um)", ylabel="y (um)",
        title="complete finite device (top view)",
    )
    ax.text(
        0.0, -118.0,
        f"{side:.0f} um x {side:.0f} um\n~{cell_count:,.0f} primitive cells",
        ha="center", va="center", fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)

    a1 = np.asarray((np.sqrt(3.0) / 2.0, -0.5))
    a2 = np.asarray((np.sqrt(3.0) / 2.0, 0.5))
    ax = axes[1]
    for i in range(-3, 4):
        for j in range(-3, 4):
            origin = i * a1 + j * a2
            for inclusion in cell.inclusions:
                ax.add_patch(Polygon(
                    _cartesian_vertices(inclusion, origin), closed=True,
                    facecolor="white", edgecolor="black", lw=0.65,
                ))
    ax.set(
        aspect="equal", xlim=(-4.2, 4.2), ylim=(-3.7, 3.7),
        xlabel="x/a", ylabel="y/a", title="lattice-scale zoom",
    )

    ax = axes[2]
    for index, inclusion in enumerate(candidate.inclusions):
        vertices = _cartesian_vertices(inclusion)
        ax.add_patch(Polygon(
            vertices, closed=True, facecolor="#f7f7f7", edgecolor="black", lw=1.5,
        ))
        center = vertices.mean(axis=0)
        ax.text(
            center[0], center[1], "main" if index == 0 else f"sat {index}",
            ha="center", va="center", fontsize=8,
        )
    primitive = np.asarray((
        -0.5 * (a1 + a2), 0.5 * (a1 - a2),
        0.5 * (a1 + a2), 0.5 * (-a1 + a2),
    ))
    ax.add_patch(Polygon(
        primitive, closed=True, fill=False, edgecolor="#d62728", lw=2.0,
    ))
    ax.arrow(0.0, 0.0, *a1, color="#1f77b4", width=0.006,
             length_includes_head=True)
    ax.arrow(0.0, 0.0, *a2, color="#1f77b4", width=0.006,
             length_includes_head=True)
    ax.set(
        aspect="equal", xlim=(-0.75, 0.95), ylim=(-0.72, 0.72),
        xlabel="x/a", ylabel="y/a", title="one optimized primitive cell",
    )
    ax.text(
        0.02, -0.69,
        f"a={LATTICE_CONSTANT_NM:.3f} nm, fill={cell.fill_fraction:.3f}",
        ha="center", va="bottom", fontsize=9,
    )
    fig.suptitle("Three-hole PCSEL: device-to-unit-cell overview")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=210)
    plt.close(fig)


def encode_matrix(matrix: np.ndarray) -> list[list[list[float]]]:
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in np.asarray(matrix)
    ]


def geometry_report(candidate, cell) -> list[dict]:
    entries = []
    for inclusion in candidate.inclusions:
        vertices = np.asarray(inclusion.vertices_fractional, dtype=float)
        entries.append({
            "shape": "triangle",
            "vertices_fractional": vertices.tolist(),
            "fill_fraction": inclusion.fill_fraction,
        })
    return entries


def write_parameter_report(output: Path, candidate, cell, layers, model) -> None:
    report = {
        "title": RUN_TITLE,
        "model": "triangular Bravais lattice, six-wave 3-D coupled-wave theory",
        "geometry": {
            "lattice_constant_nm": LATTICE_CONSTANT_NM,
            "background_index": BACKGROUND_INDEX,
            "hole_index": np.sqrt(HOLE_EPSILON),
            "total_fill_fraction": cell.fill_fraction,
            "design_parameters": DESIGN.__dict__,
            "inclusions": geometry_report(candidate, cell),
        },
        "device": {
            "shape": DEVICE_SHAPE,
            "half_size_um": DEVICE_HALF_SIZE_UM,
            "full_bounding_size_um": 2.0 * DEVICE_HALF_SIZE_UM,
            "finite_grids": FINITE_GRIDS,
            "internal_loss_cm-1": INTERNAL_LOSS_CM,
        },
        "vertical": {
            "layers": [layer.__dict__ for layer in layers],
            "bragg_wavelength_nm": model.bragg_wavelength_nm,
            "effective_index": model.effective_index,
            "group_index": model.group_index,
            "pc_confinement": model.pc_confinement,
            "active_confinement": model.vertical_mode.confinement[ACTIVE_LAYER_NAME],
        },
        "numerics": {
            "cwt_truncation_order": CWT_TRUNCATION_ORDER,
            "vertical_step_nm": VERTICAL_STEP_NM,
            "band_q_max_2pi_over_a": BAND_Q_MAX_2PI_OVER_A,
        },
        "basic_orders": TRIANGULAR_BASIC_ORDERS,
        "coupling_matrix_units": "m^-1; each complex entry is [real, imaginary]",
        "Cb": encode_matrix(model.cb_m),
        "Cr": encode_matrix(model.cr_m),
        "Ch": encode_matrix(model.ch_m),
        "C_total": encode_matrix(model.coupling_m),
        "limitations": [
            "six-wave carrier-field nonlinear time domain not implemented",
            "rounded contours are sampled polygons; no measured etch morphology",
            "scalar TE0 vertical approximation",
        ],
    }
    (output / "01_parameters_and_coupling.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def analyze_lasing_mode(modes, model) -> tuple[object, object, float, dict]:
    lasing_mode = min(modes.values(), key=lambda item: item.alpha_per_m)
    competitors = [
        item.alpha_per_m for item in modes.values() if item.name != lasing_mode.name
    ]
    gap_cm = (min(competitors) - lasing_mode.alpha_per_m) / 100.0
    far = triangular_vector_far_field(
        lasing_mode,
        model.bragg_wavelength_nm,
        view_deg=FAR_FIELD_VIEW_DEG,
        samples=FAR_FIELD_SAMPLES,
    )
    audit = threshold_audit(
        lasing_mode.alpha_per_m,
        aperture_area_m2(DEVICE_SHAPE, DEVICE_HALF_SIZE_UM),
        model.vertical_mode.confinement[ACTIVE_LAYER_NAME],
        model.effective_index,
    )
    return lasing_mode, far, gap_cm, audit


def write_threshold_audit(
    output: Path, lasing_mode, far, gap_cm: float, audit: dict, model
) -> dict:
    summary = {
        "lasing_mode": lasing_mode.name,
        "alpha_mode_cm-1": lasing_mode.alpha_per_m / 100.0,
        "internal_loss_cm-1": INTERNAL_LOSS_CM,
        "gain_condition": "g_modal = alpha_internal + 2 alpha_mode",
        **audit,
        "mode_gap_cm-1": gap_cm,
        "cold_cavity_Q": cold_cavity_q(
            model.bragg_wavelength_nm,
            model.group_index,
            lasing_mode.alpha_per_m,
        ),
        "far_field": {
            "center_to_peak": far.center_to_peak,
            "peak_offset_deg": far.peak_offset_deg,
            "centroid_offset_deg": far.centroid_offset_deg,
            "ellipticity": far.ellipticity,
            "encircled_power_0p5deg": far.encircled_power_0p5deg,
            "encircled_power_1deg": far.encircled_power_1deg,
            "full_rms_divergence_deg": far.full_rms_divergence_deg,
        },
        "interpretation": (
            "Uniform-injection small-signal threshold audit; not a nonlinear "
            "six-wave time-domain L-I prediction."
        ),
    }
    (output / "11_threshold_current_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def small_signal_pump_sweep(output: Path, audit: dict, lasing_mode, model) -> None:
    area_m2 = aperture_area_m2(DEVICE_SHAPE, DEVICE_HALF_SIZE_UM)
    threshold_current = audit["threshold_current_A"]
    overlap = (
        model.vertical_mode.confinement[ACTIVE_LAYER_NAME]
        * ACTIVE_INDEX / model.effective_index
    )
    rows = []
    for ratio in PUMP_RATIOS:
        current = ratio * threshold_current
        density_m3 = (
            current * CARRIER_LIFETIME_NS * 1e-9
            / (e * ACTIVE_THICKNESS_NM * 1e-9 * area_m2)
        )
        density_cm3 = density_m3 * 1e-6
        denominator = density_cm3 + (
            MAXIMUM_GAIN_CM / (-ZERO_CARRIER_GAIN_CM)
        ) * TRANSPARENCY_DENSITY_CM3
        material_gain_cm = MAXIMUM_GAIN_CM * (
            density_cm3 - TRANSPARENCY_DENSITY_CM3
        ) / denominator
        modal_gain_cm = overlap * material_gain_cm
        net_amplitude_cm = (
            0.5 * (modal_gain_cm - INTERNAL_LOSS_CM)
            - lasing_mode.alpha_per_m / 100.0
        )
        dn_eff = overlap * DN_DN_CM3 * (
            density_cm3 - audit["threshold_density_cm-3"]
        )
        wavelength_shift_nm = model.bragg_wavelength_nm * dn_eff / model.effective_index
        rows.append({
            "current_ratio_to_small_signal_threshold": float(ratio),
            "current_A": float(current),
            "uniform_carrier_density_cm-3": float(density_cm3),
            "material_gain_cm-1": float(material_gain_cm),
            "modal_gain_cm-1": float(modal_gain_cm),
            "net_field_amplitude_rate_cm-1": float(net_amplitude_cm),
            "unsaturated_index_shift": float(dn_eff),
            "unsaturated_wavelength_shift_nm": float(wavelength_shift_nm),
        })
    with (output / "12_small_signal_pump_sweep.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    ratios = np.asarray([row["current_ratio_to_small_signal_threshold"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    axes[0].plot(ratios, [row["net_field_amplitude_rate_cm-1"] for row in rows])
    axes[0].axhline(0.0, color="black", lw=1.0)
    axes[0].axvline(1.0, color="tab:red", ls="--", lw=1.0)
    axes[0].set(ylabel="small-signal net amplitude rate (cm$^{-1}$)")
    axes[1].plot(ratios, np.asarray([
        row["uniform_carrier_density_cm-3"] for row in rows
    ]) / 1e18)
    axes[1].set(ylabel=r"uniform carrier density ($10^{18}$ cm$^{-3}$)")
    axes[2].plot(ratios, [row["unsaturated_wavelength_shift_nm"] for row in rows])
    axes[2].set(ylabel="unsaturated carrier wavelength shift (nm)")
    for ax in axes:
        ax.set_xlabel("I / I_th,small-signal")
        ax.grid(alpha=0.2)
    fig.suptitle("Uniform-pump small-signal audit (not an L-I curve)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output / "12_small_signal_pump_sweep.png", dpi=200)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=RUN_TITLE)
    parser.add_argument("--only", nargs="*", choices=tuple(STEPS))
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    switches = STEPS.copy()
    if args.only:
        switches = {name: name in args.only for name in switches}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print(RUN_TITLE)
    print(f"output={output}")
    print("building optimized three-triangle six-wave Cb+Cr+Ch ...", flush=True)
    candidate, cell, display_layers, model = build_model()
    print(
        f"a={LATTICE_CONSTANT_NM:.4f} nm, fill={cell.fill_fraction:.5f}, "
        f"lambda_B={model.bragg_wavelength_nm:.4f} nm"
    )
    print(
        f"device={DEVICE_SHAPE}, bounding size={2*DEVICE_HALF_SIZE_UM:g} um, "
        f"finite grids={FINITE_GRIDS}"
    )
    print("=" * 82)

    band = None
    modes = None
    convergence = None
    analysis = None

    def need_band():
        nonlocal band
        if band is None:
            band = triangular_band_diagram(
                model, q_max=BAND_Q_MAX_2PI_OVER_A, points=BAND_POINTS
            )
        return band

    def need_modes():
        nonlocal modes, convergence, analysis
        if modes is None:
            print("solving six-wave finite device with multi-grid extrapolation ...", flush=True)
            modes, convergence = solve_triangular_finite_modes_converged(
                model,
                TriangularFiniteSpec(
                    radius_um=DEVICE_HALF_SIZE_UM,
                    radius_cells=FINITE_GRIDS[-1],
                    internal_loss_cm=0.0,
                    aperture_shape=DEVICE_SHAPE,
                ),
                FINITE_GRIDS,
            )
            analysis = analyze_lasing_mode(modes, model)
        return modes

    if switches["01_parameters"]:
        print("[01] parameters and six-wave coupling matrices")
        write_parameter_report(output, candidate, cell, display_layers, model)
    if switches["02_lattice"]:
        print("[02] optimized real-space lattice")
        plot_device_overview(candidate, cell, output / "02_device_overview.png")
        plot_triangular_lattice(cell, output / "02_three_triangle_lattice.png")
    if switches["03_k_space"]:
        print("[03] six retained reciprocal-space waves")
        plot_triangular_reciprocal_space(output / "03_six_wave_k_space.png")
    if switches["04_vertical_mode"]:
        print("[04] common Inoue layer stack and solved TE0")
        plot_layer_stack(display_layers, output / "04_layer_stack.png")
        plot_vertical_mode(
            model.vertical_mode, PC_LAYER_NAME, output / "04_vertical_TE0.png"
        )
    if switches["05_band_structure"]:
        print("[05] M <- Gamma -> X six-wave band structure")
        plot_band_diagram(
            need_band(), output / "05_six_wave_band_structure.png",
            "Optimized three-triangle six-wave bands: M - Gamma - X",
        )
    if switches["06_radiation_constants"]:
        print("[06] radiation constants")
        plot_radiation_constants(
            need_band(), output / "06_radiation_constants.png"
        )
    if switches["07_band_edge_states"]:
        print("[07] six Gamma band-edge unit-cell states")
        plot_six_band_edge_states(
            model, output / "07_six_band_edge_states.png",
            points=UNIT_CELL_FIELD_POINTS,
        )
    if switches["08_finite_modes"]:
        print("[08] finite-device thresholds, table and convergence")
        current = need_modes()
        write_triangular_finite_table(
            current, model, output / "08_finite_modes.csv"
        )
        plot_triangular_thresholds(current, output / "08_finite_thresholds.png")
        plot_triangular_grid_convergence(
            convergence, DEVICE_HALF_SIZE_UM, output / "08_grid_convergence.png"
        )
    if switches["09_mode_atlas"]:
        print("[09] six-mode device/cell/far-field atlas")
        plot_triangular_mode_atlas(
            need_modes(), model, output / "09_six_mode_atlas.png",
            view_deg=FAR_FIELD_VIEW_DEG,
        )
    if switches["10_best_far_field"]:
        need_modes()
        lasing_mode = analysis[0]
        print(f"[10] vector far field for lowest-threshold mode {lasing_mode.name}")
        plot_triangular_far_field_diagnostics(
            lasing_mode, model, output / "10_best_mode_far_field.png",
            view_deg=FAR_FIELD_VIEW_DEG,
        )
    if switches["11_threshold_audit"]:
        need_modes()
        print("[11] Q and uniform-injection threshold audit")
        write_threshold_audit(output, *analysis, model)
    if switches["12_small_signal_pump"]:
        need_modes()
        print("[12] uniform-pump small-signal gain/index sweep")
        lasing_mode, _, _, audit = analysis
        small_signal_pump_sweep(output, audit, lasing_mode, model)

    if modes is not None:
        lasing_mode, far, gap_cm, audit = analysis
        summary = {
            "mode": lasing_mode.name,
            "alpha_mode_cm-1": lasing_mode.alpha_per_m / 100.0,
            "required_modal_gain_cm-1": audit["required_modal_gain_cm-1"],
            "threshold_current_A": audit["threshold_current_A"],
            "mode_gap_cm-1": gap_cm,
            "center_to_peak": far.center_to_peak,
            "encircled_power_0p5deg": far.encircled_power_0p5deg,
            "encircled_power_1deg": far.encircled_power_1deg,
            "full_rms_divergence_deg": far.full_rms_divergence_deg,
            "ellipticity": far.ellipticity,
        }
        (output / "00_run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
    print("Finished. Nonlinear six-wave time-domain remains a separate future model.")


if __name__ == "__main__":
    main()
