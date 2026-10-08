"""Curated six-wave flow; no changes to optical equations / 显式输入的六波流程。"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np
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
    TriangularLatticeCell, TriangularEllipse, TriangularPolygon,
    build_triangular_coupling,
    plot_radiation_constants,
    plot_six_band_edge_states,
    plot_triangular_lattice,
    plot_triangular_reciprocal_space,
    triangular_band_diagram,
)
from pcselsim.vertical import Layer, LayerStack
from pcselsim.fabrication import TRIANGULAR_DIRECT
from pcselsim.fabrication import FabricationRules, evaluate_fabrication
from scripts._presentation import plot_device_overview as plot_overview, plot_far_field_zoom



@dataclass(frozen=True)
class TriangularSettings:
    ACTIVE_INDEX: object
    ACTIVE_LAYER_NAME: object
    ACTIVE_THICKNESS_NM: object
    BACKGROUND_INDEX: object
    BAND_POINTS: object
    BAND_Q_MAX_2PI_OVER_A: object
    BOTTOM_CLADDING_INDEX: object
    CARRIER_LIFETIME_NS: object
    CWT_TRUNCATION_ORDER: object
    DESIGN: object
    DEVICE_HALF_SIZE_UM: object
    DEVICE_SHAPE: object
    DN_DN_CM3: object
    FAR_FIELD_SAMPLES: object
    FAR_FIELD_VIEW_DEG: object
    FAR_FIELD_ZOOM_DEG: float
    FINITE_GRIDS: object
    HOLE_EPSILON: object
    INTERNAL_LOSS_CM: object
    LATTICE_CONSTANT_NM: object
    LAYERS: object
    MAXIMUM_GAIN_CM: object
    PC_LAYER_NAME: object
    PUMP_RATIOS: object
    RUN_TITLE: object
    TOP_CLADDING_INDEX: object
    TRANSPARENCY_DENSITY_CM3: object
    UNIT_CELL_FIELD_POINTS: object
    VERTICAL_PADDING_UM: object
    VERTICAL_STEP_NM: object
    WAVELENGTH_GUESS_NM: object
    ZERO_CARRIER_GAIN_CM: object
    FABRICATION_RULES: tuple[FabricationRules, ...]


@dataclass(frozen=True)
class TriangularCandidate:
    name: str
    inclusions: tuple


@dataclass(frozen=True)
class Design:
    name: str
    main_shape: str
    satellite_shape: str
    fill: float
    main_area_share: float
    distance: float
    spread_deg: float
    rounding: float = 0.3
    main_angle_deg: float = 0.0
    ellipse_aspect: float = 1.3


def build_candidate(p: Design, epsilon: float = 1.0):
    area = np.sqrt(3) / 2
    angles = (-30 - p.spread_deg, 30 + p.spread_deg)
    centers = ((0, 0),) + tuple(tuple(np.linalg.solve(TRIANGULAR_DIRECT,
        p.distance * np.asarray((np.cos(np.deg2rad(t)), np.sin(np.deg2rad(t)))))) for t in angles)
    shapes = (p.main_shape, p.satellite_shape, p.satellite_shape)
    shares = (p.main_area_share, (1-p.main_area_share)/2, (1-p.main_area_share)/2)
    inclusions = []
    for i, (shape, center, share) in enumerate(zip(shapes, centers, shares, strict=True)):
        angle = p.main_angle_deg if i == 0 else angles[i-1] + (180 if shape == "triangle" else 0)
        if shape == "triangle":
            unit = TriangularPolygon.rounded_regular((0, 0), 1, corner_fraction=p.rounding,
                                                    samples_per_corner=9)
            radius = np.sqrt(p.fill * share / unit.fill_fraction)
            hole = TriangularPolygon.rounded_regular(center, radius, angle_deg=angle,
                                                     corner_fraction=p.rounding, samples_per_corner=9)
        else:
            radius = np.sqrt(p.fill * share * area / np.pi)
            aspect = p.ellipse_aspect if shape == "ellipse" else 1
            hole = TriangularEllipse(center, (radius*np.sqrt(aspect), radius/np.sqrt(aspect)), angle)
        inclusions.append(hole)
    return TriangularCandidate(p.name, tuple(replace(hole, epsilon=epsilon) for hole in inclusions))


def aperture_area_m2(shape: str, half_size_um: float) -> float:
    """Physical gain-aperture area; hexagon size is circumradius / 实际面积。"""
    radius_m = half_size_um*1e-6
    if shape == "circle":
        return float(np.pi*radius_m**2)
    if shape == "hexagon":
        return float(1.5*np.sqrt(3.0)*radius_m**2)
    if shape == "square":
        return float((2.0*radius_m)**2)
    raise ValueError(f"unknown aperture shape: {shape}")


def threshold_audit(p, 
    alpha_mode_per_m: float,
    area_m2: float,
    confinement: float,
    effective_index: float,
) -> dict[str, float]:
    """Apply the same Inoue amplitude/gain convention to both lattices."""
    overlap = confinement * p.ACTIVE_INDEX / effective_index
    required_modal_gain_m = p.INTERNAL_LOSS_CM * 100.0 + 2.0 * alpha_mode_per_m
    required_material_gain_m = required_modal_gain_m / overlap
    maximum_gain_m = p.MAXIMUM_GAIN_CM * 100.0
    if required_material_gain_m >= maximum_gain_m:
        return {
            "required_modal_gain_cm-1": required_modal_gain_m / 100.0,
            "required_material_gain_cm-1": required_material_gain_m / 100.0,
            "threshold_density_cm-3": float("nan"),
            "threshold_current_A": float("nan"),
            "threshold_current_density_A_cm-2": float("nan"),
        }
    ratio = required_material_gain_m / maximum_gain_m
    gmax_over_minus_g0 = p.MAXIMUM_GAIN_CM / (-p.ZERO_CARRIER_GAIN_CM)
    density_cm3 = p.TRANSPARENCY_DENSITY_CM3 * (
        1.0 + ratio * gmax_over_minus_g0
    ) / (1.0 - ratio)
    current_a = (
        e * p.ACTIVE_THICKNESS_NM * 1e-9 * area_m2
        * density_cm3 * 1e6 / (p.CARRIER_LIFETIME_NS * 1e-9)
    )
    return {
        "required_modal_gain_cm-1": required_modal_gain_m / 100.0,
        "required_material_gain_cm-1": required_material_gain_m / 100.0,
        "threshold_density_cm-3": density_cm3,
        "threshold_current_A": float(current_a),
        "threshold_current_density_A_cm-2": float(current_a / (area_m2 * 1e4)),
    }


def cold_cavity_q(p, 
    wavelength_nm: float,
    group_index: float,
    alpha_mode_per_m: float,
) -> float:
    total_amplitude_loss = alpha_mode_per_m + 0.5 * p.INTERNAL_LOSS_CM * 100.0
    return float(
        np.pi * group_index / (wavelength_nm * 1e-9 * total_amplitude_loss)
    )


def build_model_for_candidate(p, candidate, settings: TriangularCWTSettings | None = None):
    """Build the common Inoue-stack six-wave model for any triangular candidate."""
    cell = TriangularLatticeCell(
        background_epsilon=p.BACKGROUND_INDEX**2,
        inclusions=candidate.inclusions,
    )
    average_index = float(np.sqrt(cell.fourier_epsilon(0, 0).real))
    display_layers = tuple(
        replace(layer, refractive_index=average_index)
        if layer.name == p.PC_LAYER_NAME else layer
        for layer in p.LAYERS
    )
    stack = LayerStack(
        layers=tuple(
            Layer(layer.name, layer.thickness_nm, layer.refractive_index)
            for layer in display_layers
        ),
        top_index=p.TOP_CLADDING_INDEX,
        bottom_index=p.BOTTOM_CLADDING_INDEX,
        padding_um=p.VERTICAL_PADDING_UM,
    )
    model = build_triangular_coupling(
        cell,
        stack,
        p.PC_LAYER_NAME,
        p.LATTICE_CONSTANT_NM,
        p.WAVELENGTH_GUESS_NM,
        settings or TriangularCWTSettings(
            truncation_order=p.CWT_TRUNCATION_ORDER,
            vertical_step_nm=p.VERTICAL_STEP_NM,
        ),
    )
    return cell, display_layers, model


def build_model(p, ):
    candidate = build_candidate(p.DESIGN, p.HOLE_EPSILON)
    cell, display_layers, model = build_model_for_candidate(p, candidate)
    return candidate, cell, display_layers, model


def encode_matrix(matrix: np.ndarray) -> list[list[list[float]]]:
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in np.asarray(matrix)
    ]


def geometry_report(candidate, cell) -> list[dict]:
    """Preserve actual circles/ellipses/polygons, rather than assuming triangles."""
    return [dict(shape=type(hole).__name__, **asdict(hole)) for hole in candidate.inclusions]



def write_parameter_report(p, output: Path, candidate, cell, layers, model) -> None:
    report = {
        "title": p.RUN_TITLE,
        "model": "triangular Bravais lattice, six-wave 3-D coupled-wave theory",
        "geometry": {
            "lattice_constant_nm": p.LATTICE_CONSTANT_NM,
            "background_index": p.BACKGROUND_INDEX,
            "hole_index": np.sqrt(p.HOLE_EPSILON),
            "total_fill_fraction": cell.fill_fraction,
            "design_parameters": p.DESIGN.__dict__,
            "inclusions": geometry_report(candidate, cell),
        },
        "device": {
            "shape": p.DEVICE_SHAPE,
            "half_size_um": p.DEVICE_HALF_SIZE_UM,
            "full_bounding_size_um": 2.0 * p.DEVICE_HALF_SIZE_UM,
            "finite_grids": p.FINITE_GRIDS,
            "internal_loss_cm-1": p.INTERNAL_LOSS_CM,
        },
        "vertical": {
            "layers": [layer.__dict__ for layer in layers],
            "bragg_wavelength_nm": model.bragg_wavelength_nm,
            "effective_index": model.effective_index,
            "group_index": model.group_index,
            "pc_confinement": model.pc_confinement,
            "active_confinement": model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME],
        },
        "numerics": {
            "cwt_truncation_order": p.CWT_TRUNCATION_ORDER,
            "vertical_step_nm": p.VERTICAL_STEP_NM,
            "band_q_max_2pi_over_a": p.BAND_Q_MAX_2PI_OVER_A,
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


def analyze_lasing_mode(p, modes, model) -> tuple[object, object, float, dict]:
    lasing_mode = min(modes.values(), key=lambda item: item.alpha_per_m)
    competitors = [
        item.alpha_per_m for item in modes.values() if item.name != lasing_mode.name
    ]
    gap_cm = (min(competitors) - lasing_mode.alpha_per_m) / 100.0
    far = triangular_vector_far_field(
        lasing_mode,
        model.bragg_wavelength_nm,
        view_deg=p.FAR_FIELD_VIEW_DEG,
        samples=p.FAR_FIELD_SAMPLES,
    )
    audit = threshold_audit(p, 
        lasing_mode.alpha_per_m,
        aperture_area_m2(p.DEVICE_SHAPE, p.DEVICE_HALF_SIZE_UM),
        model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME],
        model.effective_index,
    )
    return lasing_mode, far, gap_cm, audit


def write_threshold_audit(p, 
    output: Path, lasing_mode, far, gap_cm: float, audit: dict, model
) -> dict:
    summary = {
        "lasing_mode": lasing_mode.name,
        "alpha_mode_cm-1": lasing_mode.alpha_per_m / 100.0,
        "internal_loss_cm-1": p.INTERNAL_LOSS_CM,
        "gain_condition": "g_modal = alpha_internal + 2 alpha_mode",
        **audit,
        "mode_gap_cm-1": gap_cm,
        "cold_cavity_Q": cold_cavity_q(p, 
            model.bragg_wavelength_nm,
            model.group_index,
            lasing_mode.alpha_per_m,
        ),
        "far_field": {
            "center_to_peak": far.center_to_peak,
            "peak_offset_deg": far.peak_offset_deg,
            "centroid_offset_deg": far.centroid_offset_deg,
            "ellipticity": far.ellipticity,
            "selected_family_extrapolation_status": lasing_mode.extrapolation_status,
            "selection_interpretation": "Lowest extrapolated loss among six retained families, not proven lasing mode",
            "far_field_requested_view_deg": far.requested_view_deg,
            "far_field_evaluated_view_deg": far.evaluated_view_deg,
            "energy_normalization": far.energy_normalization,
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


def small_signal_pump_sweep(p, output: Path, audit: dict, lasing_mode, model) -> None:
    area_m2 = aperture_area_m2(p.DEVICE_SHAPE, p.DEVICE_HALF_SIZE_UM)
    threshold_current = audit["threshold_current_A"]
    overlap = (
        model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME]
        * p.ACTIVE_INDEX / model.effective_index
    )
    rows = []
    for ratio in p.PUMP_RATIOS:
        current = ratio * threshold_current
        density_m3 = (
            current * p.CARRIER_LIFETIME_NS * 1e-9
            / (e * p.ACTIVE_THICKNESS_NM * 1e-9 * area_m2)
        )
        density_cm3 = density_m3 * 1e-6
        denominator = density_cm3 + (
            p.MAXIMUM_GAIN_CM / (-p.ZERO_CARRIER_GAIN_CM)
        ) * p.TRANSPARENCY_DENSITY_CM3
        material_gain_cm = p.MAXIMUM_GAIN_CM * (
            density_cm3 - p.TRANSPARENCY_DENSITY_CM3
        ) / denominator
        modal_gain_cm = overlap * material_gain_cm
        net_amplitude_cm = (
            0.5 * (modal_gain_cm - p.INTERNAL_LOSS_CM)
            - lasing_mode.alpha_per_m / 100.0
        )
        dn_eff = overlap * p.DN_DN_CM3 * (
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


def run_triangular(p: TriangularSettings, switches: dict[str, bool], output: Path) -> None:
    """Six-wave cold-cavity flow with explicit immutable parameters."""
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print(p.RUN_TITLE)
    print(f"output={output}")
    print("building three-hole candidate six-wave Cb+Cr+Ch ...", flush=True)
    candidate, cell, display_layers, model = build_model(p, )
    if switches.get("00_fabrication", False):
        print("[00] periodic geometry and assumed fabrication budgets")
        fabrication = [evaluate_fabrication(candidate.inclusions, p.LATTICE_CONSTANT_NM,
                                           TRIANGULAR_DIRECT, rule) for rule in p.FABRICATION_RULES]
        (output/"00_fabrication.json").write_text(json.dumps(fabrication, indent=2), encoding="utf-8")
    print(
        f"a={p.LATTICE_CONSTANT_NM:.4f} nm, fill={cell.fill_fraction:.5f}, "
        f"lambda_B={model.bragg_wavelength_nm:.4f} nm"
    )
    print(
        f"device={p.DEVICE_SHAPE}, bounding size={2*p.DEVICE_HALF_SIZE_UM:g} um, "
        f"finite grids={p.FINITE_GRIDS}"
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
                model, q_max=p.BAND_Q_MAX_2PI_OVER_A, points=p.BAND_POINTS
            )
        return band

    def need_modes():
        nonlocal modes, convergence, analysis
        if modes is None:
            print("solving six-wave finite device with multi-grid extrapolation ...", flush=True)
            modes, convergence = solve_triangular_finite_modes_converged(
                model,
                TriangularFiniteSpec(
                    radius_um=p.DEVICE_HALF_SIZE_UM,
                    radius_cells=p.FINITE_GRIDS[-1],
                    internal_loss_cm=0.0,
                    aperture_shape=p.DEVICE_SHAPE,
                ),
                p.FINITE_GRIDS,
            )
            quality = {name: {
                "extrapolation_status": mode.extrapolation_status,
                "reported_amplitude_loss_cm-1": mode.alpha_per_m/100.0,
                "finest_grid_amplitude_loss_cm-1": mode.grid_alpha_per_m/100.0,
                "empirical_sensitivity_cm-1": mode.extrapolation_uncertainty_per_m/100.0,
            } for name, mode in modes.items()}
            (output/"08_numerical_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
            np.savez_compressed(output/"08_cold_cavity_arrays.npz", C_m=model.coupling_m,
                **{f"fields_{name}": mode.fields for name, mode in modes.items()},
                **{f"radiation_x_{name}": mode.radiation_x for name, mode in modes.items()},
                **{f"radiation_y_{name}": mode.radiation_y for name, mode in modes.items()},
                **{f"convergence_{name}": convergence[name] for name in modes})
            analysis = analyze_lasing_mode(p, modes, model)
        return modes

    if switches["01_parameters"]:
        print("[01] parameters and six-wave coupling matrices")
        write_parameter_report(p, output, candidate, cell, display_layers, model)
    if switches["02_lattice"]:
        print("[02] candidate real-space lattice")
        plot_triangular_lattice(cell, output / "02_three_hole_lattice.png")
    if switches.get("02_device_overview", False):
        print("[02-overview] device boundary, sampled centers and actual-scale hole array")
        plot_overview(cell, p.LATTICE_CONSTANT_NM, p.DEVICE_HALF_SIZE_UM,
                      p.DEVICE_SHAPE, output/"02_device_overview.png", TRIANGULAR_DIRECT)
    if switches["03_k_space"]:
        print("[03] six retained reciprocal-space waves")
        plot_triangular_reciprocal_space(output / "03_six_wave_k_space.png")
    if switches["04_vertical_mode"]:
        print("[04] common Inoue layer stack and solved TE0")
        plot_layer_stack(display_layers, output / "04_layer_stack.png")
        plot_vertical_mode(
            model.vertical_mode, p.PC_LAYER_NAME, output / "04_vertical_TE0.png"
        )
    if switches["05_band_structure"]:
        print("[05] M <- Gamma -> X six-wave band structure")
        plot_band_diagram(
            need_band(), output / "05_six_wave_band_structure.png",
            "Three-hole six-wave bands: M - Gamma - X (local near-Gamma)",
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
            points=p.UNIT_CELL_FIELD_POINTS,
        )
    if switches["08_finite_modes"]:
        print("[08] finite-device thresholds, table and convergence")
        current = need_modes()
        write_triangular_finite_table(
            current, model, output / "08_finite_modes.csv"
        )
        plot_triangular_thresholds(current, output / "08_finite_thresholds.png")
        plot_triangular_grid_convergence(
            convergence, p.DEVICE_HALF_SIZE_UM, output / "08_grid_convergence.png"
        )
    if switches["09_mode_atlas"]:
        print("[09] six-mode device/cell/far-field atlas")
        plot_triangular_mode_atlas(
            need_modes(), model, output / "09_six_mode_atlas.png",
            view_deg=p.FAR_FIELD_VIEW_DEG,
        )
    if switches["10_best_far_field"]:
        need_modes()
        lasing_mode = analysis[0]
        print(f"[10] vector far field for lowest-threshold mode {lasing_mode.name}")
        plot_triangular_far_field_diagnostics(
            lasing_mode, model, output / "10_best_mode_far_field.png",
            view_deg=p.FAR_FIELD_VIEW_DEG,
        )
    if switches.get("10_far_field_zoom", False):
        need_modes()
        selected, far, _, _ = analysis
        plot_far_field_zoom(far.angle_deg, far.power, p.FAR_FIELD_ZOOM_DEG,
                            output/"10_selected_far_field_zoom.png", selected.name)
    if switches["11_threshold_audit"]:
        need_modes()
        print("[11] Q and uniform-injection threshold audit")
        write_threshold_audit(p, output, *analysis, model)
    if switches["12_small_signal_pump"]:
        need_modes()
        print("[12] uniform-pump small-signal gain/index sweep")
        lasing_mode, _, _, audit = analysis
        small_signal_pump_sweep(p, output, audit, lasing_mode, model)

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
            "selected_family_extrapolation_status": lasing_mode.extrapolation_status,
            "lowest_finest_grid_family": min(modes.values(), key=lambda item: item.grid_alpha_per_m).name,
            "selection_interpretation": "Lowest extrapolated loss among six retained families, not proven lasing mode",
            "far_field_requested_view_deg": far.requested_view_deg,
            "far_field_evaluated_view_deg": far.evaluated_view_deg,
            "energy_normalization": far.energy_normalization,
        }
        (output / "00_run_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
    print("Finished. Nonlinear six-wave time-domain remains a separate future model.")


