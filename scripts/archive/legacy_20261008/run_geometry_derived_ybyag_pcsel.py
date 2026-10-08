"""Geometry-derived exploratory thin-film Yb:YAG PCSEL workflow.

Unlike the legacy Yb:YAG example, this script derives the square-lattice
four-wave C1D+Crad+C2D matrix from an explicit air-hole cell and a guided
Yb:YAG/YAG vertical stack.  The proposed stack is a *design hypothesis*, not a
published fabricated PCSEL.  The quasi-three-level reservoir remains spatially
averaged and temperature independent.

中文：本入口不再手填 A/B/C/D 光学损耗，而是从空气孔、薄膜波导 TE0 和 Liang
三维耦合波理论推导四波矩阵。纵向结构是用于验证迁移路线的设计假设，不代表已经
制备或实验标定的 Yb:YAG PCSEL；速率方程仍未包含热效应和三维泵浦传播。
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

from pcselsim.band_structure import plot_band_diagram, square_four_wave_band_diagram
from pcselsim.custom_analysis import (
    FourWaveOpticalSpec,
    LatticeSpec,
    LayerSpec,
    plot_grid_convergence,
    plot_k_space,
    plot_lattice,
    plot_layer_stack,
    plot_mode_atlas,
    plot_threshold_summary,
    plot_vector_far_field_diagnostics,
    plot_vertical_mode,
    solve_finite_modes_converged,
    vector_far_field_metrics,
    write_mode_table,
)
from pcselsim.geometry import Ellipse, SquareLatticeCell
from pcselsim.solid_state import (
    YbYAGMediumConfig,
    gain_per_m,
    plot_gain_curve,
    plot_pump_scan,
    plot_rate_dynamics,
    pump_limited_excited_fraction,
    solve_ybyag_rates,
    solve_unlased_pump_propagation,
    threshold_excited_fraction,
    threshold_pump_intensity_W_cm2,
    ybyag_energy_budget,
)
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from pcselsim.vertical import Layer, LayerStack


# ---------------------------------------------------------------------------
# A. Hypothetical but physically guided Yb:YAG thin-film structure
# ---------------------------------------------------------------------------
LASER_WAVELENGTH_GUESS_NM = 1030.0
PUMP_WAVELENGTH_NM = 940.0
LATTICE_CONSTANT_NM = 566.0
HOLE_RADIUS_NM = 68.0
HOLE_INDEX = 1.0
YBYAG_CORE_INDEX = 1.8220
YAG_CLADDING_INDEX = 1.8148
AIR_INDEX = 1.0
PC_LAYER_NAME = "patterned Yb:YAG PC"
GAIN_LAYER_NAME = "Yb:YAG gain film"
PC_DEPTH_NM = 250.0
GAIN_FILM_THICKNESS_NM = 4000.0
YAG_BUFFER_NM = 2000.0
LAYERS = (
    # LayerStack order is substrate -> air, i.e. bottom -> top.
    # 中文：自基底向空气铺层，图形层必须在最后，才能成为表面PC。
    LayerSpec("undoped YAG buffer", YAG_BUFFER_NM, YAG_CLADDING_INDEX, "#bebada"),
    LayerSpec(GAIN_LAYER_NAME, GAIN_FILM_THICKNESS_NM, YBYAG_CORE_INDEX, "#80b1d3"),
    LayerSpec(PC_LAYER_NAME, PC_DEPTH_NM, 0.0, "#fdb462"),
)


# ---------------------------------------------------------------------------
# B. Optical finite device and numerical accuracy
# ---------------------------------------------------------------------------
DEVICE_SIZE_UM = 500.0
INTERNAL_POWER_LOSS_CM = 0.05
# Three finer grids for a first-order intercept; still inspect status and
# sensitivity rather than treating an extrapolation as a full-wave validation.
FINITE_GRIDS = (25, 33, 41)
# Large sizes are only a separate inexpensive screen. Their outputs explicitly
# record this coarser grid; they are not the default-device convergence study.
SIZE_SWEEP_GRIDS = (9, 13, 17)
CWT_TRUNCATION = 6
VERTICAL_STEP_NM = 5.0
VERTICAL_PADDING_UM = 3.0


# ---------------------------------------------------------------------------
# C. Replaceable room-temperature Yb:YAG spectroscopy
# ---------------------------------------------------------------------------
YB_DOPANT_DENSITY_CM3 = 1.38e20
UPPER_STATE_LIFETIME_MS = 0.95
PUMP_ABSORPTION_CROSS_SECTION_CM2 = 0.70e-20
PUMP_EMISSION_CROSS_SECTION_CM2 = 0.10e-20
LASER_ABSORPTION_CROSS_SECTION_CM2 = 0.126e-20
LASER_EMISSION_CROSS_SECTION_CM2 = 2.00e-20
PUMP_INTENSITY_W_CM2 = 2.5e4
SPONTANEOUS_EMISSION_FACTOR = 1.0e-8
RATE_END_TIME_MS = 5.0
RATE_SAMPLES = 1000


OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "geometry_derived_ybyag_audit_20261008"
STEPS = {
    "01_parameters": True,
    "02_lattice": True,
    "03_k_space": True,
    "04_vertical_mode": True,
    "05_band_structure": True,
    "06_finite_modes": True,
    "07_mode_atlas": True,
    "08_far_field": True,
    "09_gain_threshold": True,
    "10_pump_absorption": True,
    "11_rate_dynamics": True,
    "12_pump_scan": True,
    "13_size_feasibility": True,
}

SIZE_SWEEP_UM = (500.0, 1000.0, 2000.0, 5000.0, 10000.0)


def build_model():
    radius_over_a = HOLE_RADIUS_NM / LATTICE_CONSTANT_NM
    cell = SquareLatticeCell(
        background_epsilon=YBYAG_CORE_INDEX**2,
        inclusions=(Ellipse(
            center=(0.0, 0.0), radii=(radius_over_a, radius_over_a),
            epsilon=HOLE_INDEX**2,
        ),),
    )
    average_pc_index = float(np.sqrt(cell.fourier_epsilon(0, 0).real))
    display_layers = tuple(
        replace(layer, refractive_index=average_pc_index)
        if layer.name == PC_LAYER_NAME else layer
        for layer in LAYERS
    )
    stack = LayerStack(
        layers=tuple(
            Layer(layer.name, layer.thickness_nm, layer.refractive_index)
            for layer in display_layers
        ),
        top_index=AIR_INDEX,
        bottom_index=YAG_CLADDING_INDEX,
        padding_um=VERTICAL_PADDING_UM,
    )
    coupling = build_geometry_coupling(
        cell,
        stack,
        PC_LAYER_NAME,
        LATTICE_CONSTANT_NM,
        LASER_WAVELENGTH_GUESS_NM,
        ThreeDCWTSettings(
            truncation_order=CWT_TRUNCATION,
            vertical_step_nm=VERTICAL_STEP_NM,
        ),
    )
    lattice = LatticeSpec(
        lattice_type="square",
        constant_nm=LATTICE_CONSTANT_NM,
        hole_shape="circle",
        hole_radius_x_nm=HOLE_RADIUS_NM,
        hole_radius_y_nm=HOLE_RADIUS_NM,
        hole_rotation_deg=0.0,
        background_index=YBYAG_CORE_INDEX,
        hole_index=HOLE_INDEX,
    )
    eigen_cm = coupling.eigenvalues_m / 100.0
    optical = FourWaveOpticalSpec(
        wavelength_nm=coupling.bragg_wavelength_nm,
        effective_index=coupling.effective_index,
        group_index=coupling.group_index,
        domain_um=DEVICE_SIZE_UM,
        internal_loss_cm=INTERNAL_POWER_LOSS_CM,
        modal_detuning_cm=tuple(eigen_cm.real),
        modal_radiation_loss_cm=tuple(eigen_cm.imag),
        grid_points=FINITE_GRIDS[-1],
    )
    confinement = coupling.vertical_mode.confinement[GAIN_LAYER_NAME]
    medium = YbYAGMediumConfig(
        pump_wavelength_nm=PUMP_WAVELENGTH_NM,
        laser_wavelength_nm=coupling.bragg_wavelength_nm,
        refractive_index=YBYAG_CORE_INDEX,
        upper_state_lifetime_ms=UPPER_STATE_LIFETIME_MS,
        dopant_density_cm3=YB_DOPANT_DENSITY_CM3,
        pump_absorption_cross_section_cm2=PUMP_ABSORPTION_CROSS_SECTION_CM2,
        pump_emission_cross_section_cm2=PUMP_EMISSION_CROSS_SECTION_CM2,
        laser_absorption_cross_section_cm2=LASER_ABSORPTION_CROSS_SECTION_CM2,
        laser_emission_cross_section_cm2=LASER_EMISSION_CROSS_SECTION_CM2,
        confinement_factor=confinement,
        spontaneous_emission_factor=SPONTANEOUS_EMISSION_FACTOR,
        pump_intensity_W_cm2=PUMP_INTENSITY_W_CM2,
        pumped_area_um2=DEVICE_SIZE_UM**2,
        gain_thickness_um=GAIN_FILM_THICKNESS_NM * 1e-3,
        end_time_ms=RATE_END_TIME_MS,
        samples=RATE_SAMPLES,
    )
    return cell, lattice, display_layers, coupling, optical, medium


def encode_matrix(matrix: np.ndarray) -> list[list[list[float]]]:
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in np.asarray(matrix)
    ]


def write_json_atomic(path: Path, payload: dict) -> None:
    """Readers see either the previous complete report or the new one."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def pump_absorption_audit(medium: YbYAGMediumConfig, path: Path) -> dict:
    thickness_cm = medium.gain_thickness_um * 1e-4
    alpha_unpumped_cm = (
        medium.dopant_density_cm3 * medium.pump_absorption_cross_section_cm2
    )
    z_um = np.linspace(0.0, medium.gain_thickness_um, 400)
    intensity = medium.pump_intensity_W_cm2 * np.exp(
        -alpha_unpumped_cm * z_um * 1e-4
    )
    single_pass_absorption = float(1.0 - np.exp(-alpha_unpumped_cm * thickness_cm))
    fraction_unlased = float(pump_limited_excited_fraction(medium))
    alpha_pumped_cm = medium.dopant_density_cm3 * (
        medium.pump_absorption_cross_section_cm2 * (1.0 - fraction_unlased)
        - medium.pump_emission_cross_section_cm2 * fraction_unlased
    )
    pumped_absorption = float(-np.expm1(-alpha_pumped_cm * thickness_cm))
    profile = solve_unlased_pump_propagation(medium, medium.gain_thickness_um)
    incident_power = medium.pump_intensity_W_cm2 * medium.pumped_area_um2 * 1e-8
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(z_um, intensity / medium.pump_intensity_W_cm2)
    ax.plot(z_um, np.exp(-alpha_pumped_cm * z_um * 1e-4), ls="--",
            label="uniform unlased inversion: saturated net absorption")
    ax.plot(profile.distance_um, profile.intensity_W_cm2 / medium.pump_intensity_W_cm2,
            ls=":", label="1-D local saturated pump propagation")
    ax.set(
        xlabel="depth in gain film (um)", ylabel="I_p(z) / I_p(0)",
        title="Small-signal 940-nm pump absorption (single pass)",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return {
        "unpumped_absorption_coefficient_cm-1": alpha_unpumped_cm,
        "gain_thickness_um": medium.gain_thickness_um,
        "single_pass_absorbed_fraction": single_pass_absorption,
        "unlased_local_excited_fraction": fraction_unlased,
        "saturated_net_absorption_coefficient_cm-1": alpha_pumped_cm,
        "saturated_single_pass_absorbed_fraction": pumped_absorption,
        "one_dimensional_saturated_absorbed_fraction": profile.net_absorbed_fraction,
        "incident_pump_power_W": incident_power,
        "unpumped_absorbed_power_upper_bound_W": incident_power * single_pass_absorption,
        "saturated_unlased_absorbed_power_estimate_W": incident_power * pumped_absorption,
        "warning": "Single pass with 1-D unlased pump saturation; no lateral pump, reflections or laser depletion propagation.",
    }


def size_feasibility_sweep(coupling, optical, medium, output: Path) -> list[dict]:
    """Find whether lateral scaling alone can bring loss below Yb:YAG gain."""
    rows = []
    for size_um in SIZE_SWEEP_UM:
        spec = replace(optical, domain_um=size_um)
        modes, _ = solve_finite_modes_converged(
            spec,
            SIZE_SWEEP_GRIDS,
            coupling.coupling_m,
            coupling.eigenvectors,
            coupling.radiation_fields,
        )
        lasing = min(modes.values(), key=lambda mode: mode.alpha_per_m)
        total_power_gain_cm = (
            INTERNAL_POWER_LOSS_CM + 2.0 * lasing.alpha_per_m / 100.0
        )
        threshold = float(threshold_excited_fraction(
            np.asarray([total_power_gain_cm * 100.0]), medium
        )[0])
        local_threshold = float(threshold_pump_intensity_W_cm2(
            np.asarray([total_power_gain_cm * 100.0]), medium
        )[0])
        pump_ceiling = float(pump_limited_excited_fraction(medium))
        rows.append({
            "grid_points": "/".join(map(str, SIZE_SWEEP_GRIDS)),
            "source": "geometry-derived surface-PC stack; independent coarse size screen",
            "scope": "local gain feasibility only; not converged device or thermal/pump design",
            "device_size_um": size_um,
            "mode": lasing.name,
            "field_loss_cm-1": lasing.alpha_per_m / 100.0,
            "required_power_gain_cm-1": total_power_gain_cm,
            "required_material_gain_cm-1": total_power_gain_cm / medium.confinement_factor,
            "threshold_excited_fraction": threshold,
            "numerical_loss_status": getattr(lasing, "extrapolation_status", "unassessed"),
            "grid_sensitivity_field_loss_cm-1": getattr(lasing, "extrapolation_uncertainty_per_m", float("nan")) / 100.0,
            "reachable_at_full_inversion": threshold <= 1.0,
            "reachable_at_configured_local_pump": threshold < pump_ceiling,
            "threshold_local_pump_intensity_W_cm2": local_threshold if np.isfinite(local_threshold) else None,
            "threshold_incident_uniform_pump_power_W": (
                local_threshold * size_um**2 * 1e-8 if np.isfinite(local_threshold) else None
            ),
        })
    with (output / "13_size_feasibility.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    size_mm = np.asarray([row["device_size_um"] for row in rows]) / 1000.0
    axes[0].plot(size_mm, [row["required_power_gain_cm-1"] for row in rows], "o-")
    axes[0].axhline(
        float(medium.confinement_factor * gain_per_m(1.0, medium) / 100.0), color="tab:red", ls="--",
        label="maximum MODAL gain (full inversion)",
    )
    axes[0].set(xlabel="square device side (mm)", ylabel="required power gain (cm$^{-1}$)")
    axes[0].legend()
    axes[0].axhline(
        float(medium.confinement_factor * gain_per_m(pump_limited_excited_fraction(medium), medium) / 100.0),
        color="tab:orange", ls=":", label="modal gain at configured local pump",
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(size_mm, [row["threshold_excited_fraction"] for row in rows], "o-")
    axes[1].axhline(1.0, color="tab:red", ls="--", label="full inversion")
    axes[1].axhline(float(pump_limited_excited_fraction(medium)), color="tab:orange", ls=":",
                    label="configured-pump inversion ceiling")
    axes[1].set(xlabel="square device side (mm)", ylabel="threshold excited fraction")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Local gain feasibility only: pump absorption and thermal design still required")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(output / "13_size_feasibility.png", dpi=200)
    plt.close(fig)
    return rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    print("Building geometry-derived thin-film Yb:YAG four-wave model ...", flush=True)
    cell, lattice, layers, coupling, optical, medium = build_model()
    modes = None
    convergence = None

    def need_modes():
        nonlocal modes, convergence
        if modes is None:
            modes, convergence = solve_finite_modes_converged(
                optical,
                FINITE_GRIDS,
                coupling.coupling_m,
                coupling.eigenvectors,
                coupling.radiation_fields,
            )
        return modes

    if switches["01_parameters"]:
        report = {
            "status": "exploratory geometry-derived thin-film design",
            "geometry": {
                "lattice_constant_nm": LATTICE_CONSTANT_NM,
                "hole_radius_nm": HOLE_RADIUS_NM,
                "fill_fraction": lattice.fill_fraction,
                "device_size_um": DEVICE_SIZE_UM,
            },
            "layers": [layer.__dict__ for layer in layers],
            "vertical_solution": {
                "bragg_wavelength_nm": coupling.bragg_wavelength_nm,
                "effective_index": coupling.effective_index,
                "group_index": coupling.group_index,
                "pc_confinement": coupling.pc_confinement,
                "gain_confinement": medium.confinement_factor,
            },
            "spectroscopy": medium.__dict__,
            "C1D_m-1": encode_matrix(coupling.c1d_m),
            "Crad_m-1": encode_matrix(coupling.crad_m),
            "C2D_m-1": encode_matrix(coupling.c2d_m),
            "Ctotal_m-1": encode_matrix(coupling.coupling_m),
            "limitations": [
                "hypothetical unvalidated thin-film stack",
                "room-temperature constant cross sections",
                "mean-field inversion without spatial hole burning",
                "no thermal/stress model",
            ],
        }
        report["default_device_grid_points"] = list(FINITE_GRIDS)
        report["size_screen_grid_points"] = list(SIZE_SWEEP_GRIDS)
        write_json_atomic(output / "01_parameters_and_coupling.json", report)
    if switches["02_lattice"]:
        plot_lattice(lattice, output / "02_lattice.png")
    if switches["03_k_space"]:
        plot_k_space(lattice, output / "03_k_space.png")
    if switches["04_vertical_mode"]:
        plot_layer_stack(layers, output / "04_layer_stack.png")
        plot_vertical_mode(
            coupling.vertical_mode, PC_LAYER_NAME, output / "04_vertical_TE0.png"
        )
    if switches["05_band_structure"]:
        diagram = square_four_wave_band_diagram(
            coupling.coupling_m,
            LATTICE_CONSTANT_NM,
            coupling.bragg_wavelength_nm,
            coupling.effective_index,
        )
        plot_band_diagram(
            diagram, output / "05_band_structure.png",
            "Geometry-derived thin-film Yb:YAG: M - Gamma - X",
        )
    if switches["06_finite_modes"]:
        current = need_modes()
        plot_threshold_summary(optical, current, output / "06_thresholds.png")
        plot_grid_convergence(optical, convergence, output / "06_grid_convergence.png")
        write_mode_table(output, optical, current)
    if switches["07_mode_atlas"]:
        plot_mode_atlas(
            lattice, optical, need_modes(), output / "07_mode_atlas.png",
            unit_cell_builder=coupling.unit_cell_fields,
        )

    current = need_modes()
    lasing = min(current.values(), key=lambda mode: mode.alpha_per_m)
    if switches["08_far_field"]:
        plot_vector_far_field_diagnostics(
            optical, lasing, output / "08_best_far_field.png"
        )

    mode_names = tuple(current)
    # Finite eigenvalue alpha is a field-amplitude loss.  Photon-density rate
    # equations therefore use twice alpha, plus the explicit power loss.
    total_power_loss_m = np.asarray([
        INTERNAL_POWER_LOSS_CM * 100.0 + 2.0 * current[name].alpha_per_m
        for name in mode_names
    ])
    # Finite envelopes contain mixtures of band-edge basis states; assigning
    # an infinite-band eigenvalue by its A/B/C/D index is not an output budget.
    # Project the local radiative-loss operator onto each finest-grid envelope.
    # 中文：按有限包络投影辐射算子，避免将无限晶格标签损耗直接当成输出损耗。
    radiation_loss_operator = (coupling.crad_m - coupling.crad_m.conj().T) / (2.0j)
    projected_radiation_m = np.asarray([
        2.0 * max(float(np.einsum(
            "axy,ab,bxy->", current[name].fields.conj(), radiation_loss_operator,
            current[name].fields,
        ).real / np.sum(np.abs(current[name].fields)**2)), 0.0)
        for name in mode_names
    ])
    output_power_loss_m = np.minimum(projected_radiation_m, total_power_loss_m)
    radiation_budget_clipped = projected_radiation_m > total_power_loss_m
    thresholds = threshold_excited_fraction(total_power_loss_m, medium)
    if switches["09_gain_threshold"]:
        plot_gain_curve(
            medium, total_power_loss_m, mode_names, output / "09_ybyag_gain.png"
        )
    absorption = pump_absorption_audit(
        medium, output / "10_pump_absorption.png"
    ) if switches["10_pump_absorption"] else {}
    energy_budget = {}
    if switches["11_rate_dynamics"]:
        result = solve_ybyag_rates(
            medium, mode_names, total_power_loss_m, output_power_loss_m
        )
        plot_rate_dynamics(result, output / "11_rate_dynamics.png")
        np.savez_compressed(
            output / "11_rate_dynamics.npz",
            time_ms=result.time_ms,
            excited_fraction=result.excited_fraction,
            photon_density_m3=result.photon_density_m3,
            output_power_W=result.output_power_W,
            threshold_fractions=result.threshold_fractions,
        )
        energy_budget = ybyag_energy_budget(
            np.concatenate(([result.excited_fraction[-1]], result.photon_density_m3[:, -1])),
            medium, total_power_loss_m, output_power_loss_m,
        )
    if switches["12_pump_scan"]:
        plot_pump_scan(
            medium, total_power_loss_m, output_power_loss_m,
            mode_names, output / "12_pump_scan.png",
        )
    size_rows = size_feasibility_sweep(
        coupling, optical, medium, output
    ) if switches["13_size_feasibility"] else []

    far = vector_far_field_metrics(optical, lasing, view_deg=1.5, padding=8)
    cavity_power_loss_m = float(total_power_loss_m[mode_names.index(lasing.name)])
    maximum_modal_gain_m = float(medium.confinement_factor * gain_per_m(1.0, medium))
    configured_modal_gain_m = float(medium.confinement_factor * gain_per_m(
        pump_limited_excited_fraction(medium), medium
    ))
    q_prefactor = 2.0 * np.pi * coupling.group_index / (coupling.bragg_wavelength_nm * 1e-9)
    summary = {
        "default_device_grid_points": list(FINITE_GRIDS),
        "size_screen_grid_points": list(SIZE_SWEEP_GRIDS),
        "lowest_cold_cavity_mode": lasing.name,
        "bragg_wavelength_nm": coupling.bragg_wavelength_nm,
        "effective_index": coupling.effective_index,
        "gain_confinement": medium.confinement_factor,
        "cold_cavity_field_loss_cm-1": lasing.alpha_per_m / 100.0,
        "numerical_loss_status": getattr(lasing, "extrapolation_status", "unassessed"),
        "grid_sensitivity_field_loss_cm-1": getattr(lasing, "extrapolation_uncertainty_per_m", float("nan")) / 100.0,
        "required_power_gain_cm-1": float(
            total_power_loss_m[mode_names.index(lasing.name)] / 100.0
        ),
        "required_material_gain_cm-1": float(
            total_power_loss_m[mode_names.index(lasing.name)] / 100.0 / medium.confinement_factor
        ),
        "threshold_excited_fraction": float(
            thresholds[mode_names.index(lasing.name)]
        ),
        "threshold_reachable_at_full_inversion": bool(
            thresholds[mode_names.index(lasing.name)] <= 1.0
        ),
        "maximum_material_gain_cm-1": float(gain_per_m(1.0, medium) / 100.0),
        "maximum_modal_gain_cm-1": float(medium.confinement_factor * gain_per_m(1.0, medium) / 100.0),
        "cold_cavity_Q_with_internal_power_loss": q_prefactor / cavity_power_loss_m,
        "minimum_required_Q_at_full_inversion": q_prefactor / maximum_modal_gain_m,
        "minimum_required_Q_at_configured_local_pump": (
            q_prefactor / configured_modal_gain_m if configured_modal_gain_m > 0.0 else None
        ),
        "unlased_excited_fraction_at_configured_local_pump": float(pump_limited_excited_fraction(medium)),
        "threshold_reachable_at_configured_local_pump": bool(
            thresholds[mode_names.index(lasing.name)] < pump_limited_excited_fraction(medium)
        ),
        "pump_line_infinite_intensity_inversion_limit": (
            medium.pump_absorption_cross_section_cm2
            / (medium.pump_absorption_cross_section_cm2 + medium.pump_emission_cross_section_cm2)
        ),
        "total_power_loss_per_m": total_power_loss_m.tolist(),
        "projected_finite_envelope_radiation_power_loss_per_m": projected_radiation_m.tolist(),
        "radiation_budget_clipped_by_extrapolated_loss": radiation_budget_clipped.tolist(),
        "output_note": "Projection includes both radiation directions; not calibrated collected upward power.",
        "photon_normalization": "S_m=N_photon,m/V_gain; Gamma in both stimulated population and photon terms",
        "energy_budget_at_final_time": energy_budget,
        "far_field": far.__dict__,
        "pump_absorption": absorption,
        "size_feasibility": size_rows,
        "scope": "exploratory; local mean-field gain test, not a fabricated-device or collected-power prediction",
    }
    write_json_atomic(output / "00_feasibility_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
