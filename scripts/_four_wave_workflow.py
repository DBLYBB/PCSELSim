"""Curated four-wave workflow: explicit parameters, unchanged physical backend.

中文：只组织输入/输出与步骤；不改 src 中的四波/载流子/辐射方程。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import numpy as np

from pcselsim.band_structure import plot_band_diagram, square_four_wave_band_diagram
from pcselsim.config import (
    CarrierConfig, DeviceConfig, NumericsConfig, OpticalConfig,
    ReproductionConfig, SimulationConfig, validate_config,
)
from pcselsim.custom_analysis import (
    MODE_NAMES, FourWaveOpticalSpec, LatticeInclusionSpec, LatticeSpec, LayerSpec,
    ensure_square_four_wave,
    plot_grid_convergence, plot_k_space, plot_lattice, plot_layer_stack,
    plot_length_sweep_from_solver, plot_mode_atlas, plot_threshold_summary,
    plot_vector_far_field_diagnostics, plot_vertical_mode,
    solve_finite_modes_converged,
    write_mode_table, write_parameter_report,
    vector_far_field,
)
from pcselsim.constants import c, e, pi
from pcselsim.geometry import Ellipse, PolygonInclusion, SquareLatticeCell
from pcselsim.io import save_result
from pcselsim.injection import electrode_area_m2
from pcselsim.plotting import plot_spatial, plot_spectra, plot_transient
from pcselsim.solver import TimeDomainSolver
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from pcselsim.vertical import Layer, LayerStack
from pcselsim.fabrication import FabricationRules, evaluate_fabrication
from scripts._presentation import plot_device_overview, plot_far_field_zoom



@dataclass(frozen=True)
class FourWaveSettings:
    """All device, carrier and numerical inputs are supplied by the main script."""
    ACTIVE_CONFINEMENT_FACTOR: object
    ACTIVE_CONFINEMENT_SOURCE: object
    ACTIVE_INDEX: object
    ACTIVE_LAYER_NAME: object
    ACTIVE_THICKNESS_NM: object
    BACKGROUND_INDEX: object
    BOTTOM_CLADDING_INDEX: object
    CARRIER_LIFETIME_NS: object
    CFL: object
    CURRENT_RATIOS: object
    CURRENT_SPREAD_UM: object
    CWT_TRUNCATION_ORDER: object
    DEVICE_PRESET: object
    DEVICE_SIZE_UM: object
    DIFFUSION_CM2_S: object
    DN_DN_CM3: object
    ELECTRODE_SHAPE: object
    ELECTRODE_SIZE_UM: object
    ENABLE_NOISE: object
    END_TIME_NS: object
    FINITE_EIGEN_GRIDS: object
    GEOMETRY_FIDELITY: object
    HOLE_INDEX: object
    HOLE_RADIUS_X_NM: object
    HOLE_RADIUS_Y_NM: object
    HOLE_ROTATION_DEG: object
    HOLE_SHAPE: object
    INTERNAL_LOSS_CM: object
    LATTICE_CONSTANT_NM: object
    LATTICE_TYPE: object
    LAYERS: object
    LENGTH_SWEEP_GRID_POINTS: object
    LENGTH_SWEEP_UM: object
    MAXIMUM_GAIN_CM: object
    PAPER_THRESHOLD_CURRENT_A: object
    PC_LAYER_NAME: object
    RANDOM_SEED: object
    RUN_TITLE: object
    SAMPLE_INTERVAL_PS: object
    SPECTRUM_WINDOW_NS: object
    SPONTANEOUS_EMISSION_FACTOR: object
    THRESHOLD_CURRENT_SOURCE: object
    TIME_DOMAIN_SIZE_UM: object
    TIME_GRID_POINTS: object
    TOP_CLADDING_INDEX: object
    TRANSPARENCY_DENSITY_CM3: object
    UNIT_CELL_INCLUSIONS: object
    VERTICAL_PADDING_UM: object
    VERTICAL_STEP_NM: object
    WAVELENGTH_GUESS_NM: object
    ZERO_CARRIER_GAIN_CM: object
    FABRICATION_RULES: tuple[FabricationRules, ...]
    FAR_FIELD_ZOOM_DEG: float


def build_unit_cell(lattice: LatticeSpec) -> SquareLatticeCell:
    """Convert editable holes into exact Fourier geometry / 构建解析 Fourier 晶胞。"""
    epsilon_hole = lattice.hole_index**2
    source = lattice.inclusions or (
        LatticeInclusionSpec(
            lattice.hole_shape, lattice.hole_radius_x_nm, lattice.hole_radius_y_nm,
            rotation_deg=lattice.hole_rotation_deg,
        ),
    )
    inclusions = []
    for item in source:
        rx = item.radius_x_nm/lattice.constant_nm
        ry = item.radius_y_nm/lattice.constant_nm
        center = (
            item.center_x_nm/lattice.constant_nm,
            item.center_y_nm/lattice.constant_nm,
        )
        if item.shape in {"circle", "ellipse"}:
            inclusions.append(Ellipse(
                center=center, radii=(rx, ry),
                angle_deg=item.rotation_deg, epsilon=epsilon_hole,
            ))
            continue
        if item.shape == "triangle":
            vertices = ((-rx, -ry), (rx, 0.0), (-rx, ry))
        elif item.shape == "rit":
            vertices = ((-rx, -ry), (rx, -ry), (rx, ry))
        else:
            raise ValueError(f"Unsupported hole shape: {item.shape}")
        angle = np.deg2rad(item.rotation_deg)
        rotation = np.asarray(((np.cos(angle), -np.sin(angle)),
                               (np.sin(angle), np.cos(angle))))
        rotated = np.asarray(vertices)@rotation.T+np.asarray(center)
        inclusions.append(PolygonInclusion(
            vertices=tuple(map(tuple, rotated)), epsilon=epsilon_hole
        ))
    return SquareLatticeCell(
        background_epsilon=lattice.background_index**2,
        inclusions=tuple(inclusions),
    )


def build_specs(p: FourWaveSettings):
    lattice = LatticeSpec(
        lattice_type=p.LATTICE_TYPE,
        constant_nm=p.LATTICE_CONSTANT_NM,
        hole_shape=p.HOLE_SHAPE,
        hole_radius_x_nm=p.HOLE_RADIUS_X_NM,
        hole_radius_y_nm=p.HOLE_RADIUS_Y_NM,
        hole_rotation_deg=p.HOLE_ROTATION_DEG,
        background_index=p.BACKGROUND_INDEX,
        hole_index=p.HOLE_INDEX,
        inclusions=p.UNIT_CELL_INCLUSIONS,
    )
    ensure_square_four_wave(lattice)
    cell = build_unit_cell(lattice)
    average_pc_index = float(np.sqrt(cell.fourier_epsilon(0, 0).real))
    display_layers = tuple(
        replace(layer, refractive_index=average_pc_index)
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
    geometry_model = build_geometry_coupling(
        cell=cell,
        stack=stack,
        pc_layer_name=p.PC_LAYER_NAME,
        lattice_constant_nm=p.LATTICE_CONSTANT_NM,
        wavelength_guess_nm=p.WAVELENGTH_GUESS_NM,
        settings=ThreeDCWTSettings(
            truncation_order=p.CWT_TRUNCATION_ORDER,
            vertical_step_nm=p.VERTICAL_STEP_NM,
        ),
    )
    modal_values_cm = geometry_model.eigenvalues_m/100.0
    solved_active_confinement = geometry_model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME]
    if p.ACTIVE_CONFINEMENT_SOURCE == "paper":
        active_confinement = p.ACTIVE_CONFINEMENT_FACTOR
    elif p.ACTIVE_CONFINEMENT_SOURCE == "solved_vertical_TE0":
        active_confinement = solved_active_confinement
    else:
        raise ValueError(
            "p.ACTIVE_CONFINEMENT_SOURCE must be 'paper' or 'solved_vertical_TE0'"
        )
    linear = FourWaveOpticalSpec(
        wavelength_nm=geometry_model.bragg_wavelength_nm,
        effective_index=geometry_model.effective_index,
        group_index=geometry_model.group_index,
        domain_um=p.DEVICE_SIZE_UM,
        internal_loss_cm=p.INTERNAL_LOSS_CM,
        modal_detuning_cm=tuple(modal_values_cm.real),
        modal_radiation_loss_cm=tuple(modal_values_cm.imag),
        grid_points=p.FINITE_EIGEN_GRIDS[-1],
    )
    simulation = SimulationConfig(
        optical=OpticalConfig(
            lattice_constant_nm=p.LATTICE_CONSTANT_NM,
            wavelength_nm=geometry_model.bragg_wavelength_nm,
            group_index=geometry_model.group_index,
            effective_index=geometry_model.effective_index,
            active_index=p.ACTIVE_INDEX,
            internal_loss_cm=p.INTERNAL_LOSS_CM,
            confinement_factor=active_confinement,
            dn_dN_cm3=p.DN_DN_CM3,
            modal_detuning_cm=tuple(modal_values_cm.real),
            modal_radiation_loss_cm=tuple(modal_values_cm.imag),
        ),
        carrier=CarrierConfig(
            maximum_gain_cm=p.MAXIMUM_GAIN_CM,
            zero_carrier_gain_cm=p.ZERO_CARRIER_GAIN_CM,
            transparency_density_cm3=p.TRANSPARENCY_DENSITY_CM3,
            lifetime_ns=p.CARRIER_LIFETIME_NS,
            diffusion_cm2_s=p.DIFFUSION_CM2_S,
            active_thickness_nm=p.ACTIVE_THICKNESS_NM,
            spontaneous_emission_factor=p.SPONTANEOUS_EMISSION_FACTOR,
        ),
        device=DeviceConfig(
            domain_um=p.TIME_DOMAIN_SIZE_UM,
            electrode_um=p.ELECTRODE_SIZE_UM,
            current_spread_um=p.CURRENT_SPREAD_UM,
            threshold_current_A=p.PAPER_THRESHOLD_CURRENT_A,
            electrode_shape=p.ELECTRODE_SHAPE,
        ),
        numerics=NumericsConfig(
            points=p.TIME_GRID_POINTS,
            end_time_ns=p.END_TIME_NS,
            cfl=p.CFL,
            sample_interval_ps=p.SAMPLE_INTERVAL_PS,
            seed=p.RANDOM_SEED,
            noise=p.ENABLE_NOISE,
        ),
        reproduction=ReproductionConfig(
            current_ratios=p.CURRENT_RATIOS,
            spectrum_window_ns=p.SPECTRUM_WINDOW_NS,
        ),
    )
    validate_config(simulation)
    return lattice, linear, simulation, display_layers, geometry_model


def threshold_current_audit(p: FourWaveSettings, simulation: SimulationConfig, mode) -> dict[str, float | str]:
    """Invert Inoue Eq. (11) for the cold-cavity modal threshold.

    mode.alpha_per_m is an amplitude loss in the time-domain generator,
    whereas material/internal gain appears with a factor 1/2. The threshold
    condition used by the solver is g_modal=alpha_i+2*alpha_mode.
    """
    optical = simulation.optical
    carrier = simulation.carrier
    device = simulation.device
    overlap = optical.confinement_factor*optical.active_index/optical.effective_index
    required_modal_gain_m = optical.internal_loss_cm*100.0+2.0*mode.alpha_per_m
    required_material_gain_m = required_modal_gain_m/overlap
    maximum_gain_m = carrier.maximum_gain_cm*100.0
    if required_material_gain_m >= maximum_gain_m:
        raise ValueError(
            "The computed cold-cavity loss requires material gain above gmax; "
            "change the geometry/layers or use p.THRESHOLD_CURRENT_SOURCE='paper' only "
            "for a clearly labelled qualitative transient."
        )
    ratio = required_material_gain_m/maximum_gain_m
    gmax_over_minus_g0 = carrier.maximum_gain_cm/(-carrier.zero_carrier_gain_cm)
    threshold_density_cm3 = carrier.transparency_density_cm3*(
        1.0+ratio*gmax_over_minus_g0
    )/(1.0-ratio)
    area_m2 = electrode_area_m2(device)
    threshold_current_a = (
        e*carrier.active_thickness_nm*1e-9*area_m2
        * threshold_density_cm3*1e6/(carrier.lifetime_ns*1e-9)
    )
    return {
        "lasing_mode": mode.name,
        "mode_amplitude_loss_per_m": float(mode.alpha_per_m),
        "required_modal_gain_per_m": float(required_modal_gain_m),
        "required_material_gain_per_m": float(required_material_gain_m),
        "threshold_density_cm3": float(threshold_density_cm3),
        "derived_uniform_injection_threshold_A": float(threshold_current_a),
        "paper_reference_threshold_A": p.PAPER_THRESHOLD_CURRENT_A,
        "threshold_condition": "g_modal = alpha_internal + 2 alpha_mode",
    }


def run_four_wave(p: FourWaveSettings, switches: dict[str, bool], output: Path) -> None:
    """Explicit-input workflow; no mutation of archived module globals."""
    output.mkdir(parents=True, exist_ok=True)
    lattice, linear, simulation, display_layers, geometry_model = build_specs(p)
    modes = None
    convergence = None
    threshold_audit = None

    if switches.get("00_fabrication", False):
        print("[00] periodic hole width/gap and assumed fabrication budgets")
        cell = build_unit_cell(lattice)
        metrics = [evaluate_fabrication(cell.inclusions, p.LATTICE_CONSTANT_NM, rules=rule)
                   for rule in p.FABRICATION_RULES]
        (output/"00_fabrication.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("="*78)
    print(p.RUN_TITLE)
    print(f"output={output}")
    print(
        f"Bragg lambda={geometry_model.bragg_wavelength_nm:.4f} nm, "
        f"neff={geometry_model.effective_index:.6f}, ng={geometry_model.group_index:.6f}"
    )
    print(f"lattice={p.LATTICE_TYPE}, hole={p.HOLE_SHAPE}, a={p.LATTICE_CONSTANT_NM} nm")
    print(
        f"finite PC={p.DEVICE_SIZE_UM} um, time domain={p.TIME_DOMAIN_SIZE_UM} um, "
        f"electrode={p.ELECTRODE_SHAPE} {p.ELECTRODE_SIZE_UM} um, "
        f"grid={p.TIME_GRID_POINTS}x{p.TIME_GRID_POINTS}"
    )
    print("="*78)

    def need_modes():
        nonlocal modes, convergence, simulation, threshold_audit, linear
        ensure_square_four_wave(lattice)
        if modes is None:
            print(f"    finite-area convergence grids={p.FINITE_EIGEN_GRIDS} ...")
            modes, convergence = solve_finite_modes_converged(
                linear,
                p.FINITE_EIGEN_GRIDS,
                geometry_model.coupling_m,
                geometry_model.eigenvectors,
                geometry_model.radiation_fields,
            )
            # FFP sampling metadata must use the actual retained field grid.
            linear = replace(linear, grid_points=next(iter(modes.values())).fields.shape[-1])
            np.savez_compressed(output/"05_cold_cavity_arrays.npz",
                C_m=geometry_model.coupling_m,
                **{f"fields_{name}": mode.fields for name, mode in modes.items()},
                **{f"convergence_{key}": value for key, value in convergence.items()})
            quality = {name: {
                "extrapolation_status": mode.extrapolation_status,
                "reported_amplitude_loss_cm-1": mode.alpha_per_m/100.0,
                "finest_grid_amplitude_loss_cm-1": mode.grid_alpha_per_m/100.0,
                "empirical_sensitivity_cm-1": mode.extrapolation_uncertainty_per_m/100.0,
            } for name, mode in modes.items()}
            (output/"05_numerical_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
            lasing_name = min(MODE_NAMES, key=lambda name: modes[name].alpha_per_m)
            threshold_audit = threshold_current_audit(p, simulation, modes[lasing_name])
            selected_threshold = (
                threshold_audit["derived_uniform_injection_threshold_A"]
                if p.THRESHOLD_CURRENT_SOURCE == "derived"
                else p.PAPER_THRESHOLD_CURRENT_A
            )
            if p.THRESHOLD_CURRENT_SOURCE not in {"derived", "paper"}:
                raise ValueError("p.THRESHOLD_CURRENT_SOURCE must be 'derived' or 'paper'")
            simulation = replace(
                simulation,
                device=replace(
                    simulation.device, threshold_current_A=float(selected_threshold)
                ),
            )
            threshold_audit["selected_threshold_source"] = p.THRESHOLD_CURRENT_SOURCE
            threshold_audit["selected_threshold_A"] = float(selected_threshold)
            threshold_audit["selected_family_extrapolation_status"] = modes[lasing_name].extrapolation_status
            threshold_audit["lowest_finest_grid_family"] = min(
                MODE_NAMES, key=lambda name: modes[name].grid_alpha_per_m)
            threshold_audit["selection_interpretation"] = (
                "Lowest reported loss among four retained band-connected families; "
                "not a complete transverse spectrum or proof of the lasing mode.")
            (output/"05_threshold_current_audit.json").write_text(
                json.dumps(threshold_audit, indent=2), encoding="utf-8"
            )
        return modes

    if switches["01_parameters"]:
        print("[01] parameter report")
        need_modes()  # makes the saved threshold current consistent with the solved cavity
        write_parameter_report(output, lattice, linear, display_layers, {
            "device_preset": p.DEVICE_PRESET,
            "geometry_fidelity": p.GEOMETRY_FIDELITY,
            "carrier": simulation.carrier.__dict__,
            "device": simulation.device.__dict__,
            "numerics": simulation.numerics.__dict__,
            "current_ratios": p.CURRENT_RATIOS,
            "cwt_truncation_order": p.CWT_TRUNCATION_ORDER,
            "vertical_step_nm": p.VERTICAL_STEP_NM,
            "finite_eigen_grids": p.FINITE_EIGEN_GRIDS,
            "bragg_wavelength_nm": geometry_model.bragg_wavelength_nm,
            "derived_effective_index": geometry_model.effective_index,
            "derived_group_index": geometry_model.group_index,
            "used_active_confinement": simulation.optical.confinement_factor,
            "solved_vertical_TE0_active_confinement": (
                geometry_model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME]
            ),
            "active_confinement_source": p.ACTIVE_CONFINEMENT_SOURCE,
            "threshold_current_source": p.THRESHOLD_CURRENT_SOURCE,
            "paper_threshold_current_A": p.PAPER_THRESHOLD_CURRENT_A,
        })
        def encode_matrix(matrix):
            return [[[float(value.real), float(value.imag)] for value in row] for row in matrix]
        coupling_report = {
            "units": "m^-1; each element is [real, imaginary]",
            "C1D": encode_matrix(geometry_model.c1d_m),
            "Crad": encode_matrix(geometry_model.crad_m),
            "C2D": encode_matrix(geometry_model.c2d_m),
            "C_total": encode_matrix(geometry_model.coupling_m),
            "band_eigenvectors": encode_matrix(geometry_model.eigenvectors),
            "infinite_eigenvalues_cm^-1": [
                [float(value.real/100.0), float(value.imag/100.0)]
                for value in geometry_model.eigenvalues_m
            ],
            "fourier_coefficients": {
                f"{m},{n}": [
                    float(geometry_model.fourier(m, n).real),
                    float(geometry_model.fourier(m, n).imag),
                ]
                for m in range(-p.CWT_TRUNCATION_ORDER-1, p.CWT_TRUNCATION_ORDER+2)
                for n in range(-p.CWT_TRUNCATION_ORDER-1, p.CWT_TRUNCATION_ORDER+2)
                if (m, n) != (0, 0)
            },
            "average_pc_epsilon": geometry_model.average_pc_epsilon,
            "pc_confinement": geometry_model.pc_confinement,
            "passivity_correction_m^-1": geometry_model.passivity_correction_m,
        }
        (output/"01_geometry_coupling.json").write_text(
            json.dumps(coupling_report, indent=2), encoding="utf-8"
        )
    if switches["02_lattice"]:
        print("[02] real-space lattice")
        plot_lattice(lattice, output/"02_lattice.png")
    if switches.get("02_device_overview", False):
        print("[02-overview] device boundary, sampled centers and actual-scale hole array")
        plot_device_overview(build_unit_cell(lattice), p.LATTICE_CONSTANT_NM,
                             p.DEVICE_SIZE_UM/2, "square", output/"02_device_overview.png")
    if switches["03_k_space"]:
        print("[03] reciprocal lattice and retained Bloch waves")
        plot_k_space(lattice, output/"03_k_space.png")
    if switches["04_layer_stack"]:
        print("[04] vertical index stack")
        plot_layer_stack(display_layers, output/"04_layer_stack.png")
        plot_vertical_mode(
            geometry_model.vertical_mode, p.PC_LAYER_NAME,
            output/"04_vertical_TE0.png",
        )
    if switches["05_band_structure"]:
        print("[05-band] four-wave M <- Gamma -> X normalized-frequency bands")
        band_diagram = square_four_wave_band_diagram(
            geometry_model.coupling_m,
            lattice_constant_nm=p.LATTICE_CONSTANT_NM,
            bragg_wavelength_nm=geometry_model.bragg_wavelength_nm,
            effective_index=geometry_model.effective_index,
            q_max=0.01,
            points=201,
        )
        plot_band_diagram(
            band_diagram,
            output/"05_four_wave_band_structure.png",
            "Square four-wave bands: M - Gamma - X",
        )
    if switches["05_linear_modes"]:
        print("[05] finite-area thresholds and mode table")
        current_modes = need_modes()
        plot_threshold_summary(linear, current_modes, output/"05_thresholds.png")
        write_mode_table(output, linear, current_modes)
        plot_grid_convergence(linear, convergence, output/"05_grid_convergence.png")
        if (
            lattice.hole_shape == "circle"
            and abs(lattice.fill_fraction-0.16) < 0.01
            and abs(linear.domain_um-70.0) < 1.0
        ):
            # Liang Table 4.2 is a validation target, never an input to C.
            reference_alpha_l = {"A": 0.23, "B": 0.52, "C": 2.18, "D": 2.18}
            with (output/"05_liang_table4_2_comparison.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(("mode", "computed_alpha_L", "paper_alpha_L", "relative_error_percent"))
                for name in ("A", "B", "C", "D"):
                    computed = current_modes[name].alpha_per_m*linear.domain_um*1e-6
                    reference = reference_alpha_l[name]
                    writer.writerow((name, computed, reference, 100.0*(computed/reference-1.0)))
    if switches["06_mode_atlas"]:
        print("[06] whole-device, unit-cell and far-field mode atlas")
        current_modes = need_modes()
        plot_mode_atlas(
            lattice, linear, current_modes, output/"06_mode_atlas.png",
            unit_cell_builder=geometry_model.unit_cell_fields,
        )
        plot_vector_far_field_diagnostics(
            linear, current_modes["A"], output/"06_A_vector_far_field.png"
        )
    if switches.get("06_far_field_zoom", False):
        current_modes = need_modes()
        selected = current_modes[str(threshold_audit["lasing_mode"])]
        angle, power, _, _, _ = vector_far_field(
            selected.radiation_x, selected.radiation_y,
            linear.wavelength_nm, linear.domain_um, view_deg=1.0, padding=16)
        plot_far_field_zoom(angle, power, p.FAR_FIELD_ZOOM_DEG,
                            output/"06_selected_far_field_zoom.png", selected.name)
    if switches["07_length_sweep"]:
        print("[07] finite-size threshold sweep")
        ensure_square_four_wave(lattice)
        sweep = plot_length_sweep_from_solver(
            linear,
            geometry_model.coupling_m,
            geometry_model.eigenvectors,
            geometry_model.radiation_fields,
            output/"07_length_sweep.png",
            lengths_um=np.asarray(p.LENGTH_SWEEP_UM),
            grid_points=p.LENGTH_SWEEP_GRID_POINTS,
        )
        with (output/"07_length_sweep.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(("length_um", *MODE_NAMES))
            for index, length_um in enumerate(sweep["length_um"]):
                writer.writerow((length_um, *(sweep[name][index] for name in MODE_NAMES)))
    if switches["08_time_domain"]:
        print("[08] Inoue semiconductor carrier-field time-domain calculation")
        ensure_square_four_wave(lattice)
        current_modes = need_modes()
        lasing_name = str(threshold_audit["lasing_mode"])
        mode_index = MODE_NAMES.index(lasing_name)
        frame_shift_m = current_modes[lasing_name].delta_per_m
        rotating_coupling = (
            geometry_model.coupling_m
            - frame_shift_m*np.identity(4, dtype=np.complex128)
        )
        reference_frequency_hz = (
            c/(geometry_model.bragg_wavelength_nm*1e-9)
            + (c/geometry_model.group_index)*frame_shift_m/(2.0*pi)
        )
        reference_wavelength_nm = c/reference_frequency_hz*1e9
        threshold_audit["rotating_frame_shift_per_m"] = float(frame_shift_m)
        threshold_audit["spectrum_reference_wavelength_nm"] = float(reference_wavelength_nm)
        (output/"05_threshold_current_audit.json").write_text(
            json.dumps(threshold_audit, indent=2), encoding="utf-8"
        )
        solver = TimeDomainSolver(
            simulation,
            coupling_m=rotating_coupling,
            signal_projection=geometry_model.eigenvectors[:, mode_index],
        )
        results = []
        for ratio in p.CURRENT_RATIOS:
            print(f"    I/Ith={ratio:.2f}")
            result = solver.run(ratio)
            results.append(result)
            tag = f"I_{ratio:.2f}".replace(".", "p")
            save_result(result, output/f"08_{tag}.npz")
            plot_transient(result, output/f"08_{tag}_transient.png")
            plot_spatial(result, output/f"08_{tag}_spatial.png")
        with (output/"08_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "current_ratio", "actual_tail_window_ns", "mean_power_W",
                "maximum_power_W", "mean_center_carrier_cm-3",
            ))
            for result in results:
                steady = result.time_ns >= result.time_ns[-1]-1.0
                writer.writerow((
                    result.current_ratio,
                    float(result.time_ns[steady][-1]-result.time_ns[steady][0]),
                    float(np.mean(result.power_W[steady])),
                    float(np.max(result.power_W)),
                    float(np.mean(result.center_carrier_cm3[steady])),
                ))
        plot_spectra(
            results, output/"08_spectra.png", p.SPECTRUM_WINDOW_NS,
            center_wavelength_nm=reference_wavelength_nm,
        )
    print("Finished. Edit the parameter panel at the top of this script for your device.")


