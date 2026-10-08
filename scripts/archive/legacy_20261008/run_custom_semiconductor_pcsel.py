"""Editable semiconductor PCSEL workflow built on the Inoue time-domain model.

PyCharm users: edit only the PARAMETER PANEL and STEP SWITCHES, then click Run.
Every dimensional name carries its unit.  Results are written to a new folder.

中文操作：初学者只需修改下方三个 ``PARAMETER PANEL`` 和 ``STEP SWITCHES``，
然后在 PyCharm 中右键本文件选择 Run。程序按“晶胞 Fourier 系数 -> 纵向 TE0 ->
Liang 3-D CWT 耦合矩阵 -> 有限区域冷腔模 -> Inoue 载流子—光场时域方程”执行。
线性与时域步骤使用同一个几何推导 C 矩阵；不要用数值网格或绘图范围去拟合实验。
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
)
from pcselsim.constants import c, e, pi
from pcselsim.geometry import Ellipse, PolygonInclusion, SquareLatticeCell
from pcselsim.io import save_result
from pcselsim.injection import electrode_area_m2
from pcselsim.plotting import plot_spatial, plot_spectra, plot_transient
from pcselsim.solver import TimeDomainSolver
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from pcselsim.vertical import Layer, LayerStack


# ===========================================================================
# PARAMETER PANEL A - wavelength, lattice and hole geometry
# 参数区 A：波长、晶格与孔形。孔尺寸/位置改变会重新计算全部 Fourier 系数。
# ===========================================================================
DEVICE_PRESET = "inoue2019_idealized_double_lattice"
RUN_TITLE = "CUSTOM SEMICONDUCTOR PCSEL - Inoue idealized double lattice"
GEOMETRY_FIDELITY = (
    "Inoue Table I/II plus idealized Fig. 2(a) twin ellipses; "
    "the unpublished SEM contour is not reconstructed"
)
WAVELENGTH_GUESS_NM = 950.65  # Bragg wavelength is solved from lambda=a*n_eff(lambda).
LATTICE_TYPE = "square"       # For triangular lattices use the independent six-wave script.
LATTICE_CONSTANT_NM = 277.0
HOLE_SHAPE = "inoue_double_ellipse"  # descriptive label for this two-hole motif
HOLE_RADIUS_X_NM = 0.0                # unused while UNIT_CELL_INCLUSIONS is non-empty
HOLE_RADIUS_Y_NM = 0.0
HOLE_ROTATION_DEG = 0.0
BACKGROUND_INDEX = 3.554
HOLE_INDEX = 1.0

# Inoue Fig. 2(a) publishes the relative center displacement but not the SEM
# hole contour dimensions.  These two ellipses are therefore an explicit,
# editable idealization; they must not be described as the unpublished SEM mesh.
UNIT_CELL_INCLUSIONS = (
    LatticeInclusionSpec(
        "ellipse", 0.120*LATTICE_CONSTANT_NM, 0.075*LATTICE_CONSTANT_NM,
        -0.125*LATTICE_CONSTANT_NM, -0.125*LATTICE_CONSTANT_NM, -35.0,
    ),
    LatticeInclusionSpec(
        "ellipse", 0.095*LATTICE_CONSTANT_NM, 0.060*LATTICE_CONSTANT_NM,
        +0.125*LATTICE_CONSTANT_NM, +0.125*LATTICE_CONSTANT_NM, -35.0,
    ),
)

# Inoue Table I stack, listed from bottom to top.  The PC index written here
# is replaced automatically by sqrt(<epsilon>) from the selected unit cell.
LAYERS = (
    LayerSpec("p-clad AlGaAs", 2000.0, 3.297, "#7fc97f"),
    LayerSpec("photonic crystal", 190.0, 0.0, "#fdb462"),
    LayerSpec("GaAs", 110.0, 3.554, "#ffffb3"),
    LayerSpec("AlGaAs spacer", 25.0, 3.269, "#80b1d3"),
    LayerSpec("active InGaAs QWs", 10.0, 3.584, "#f0027f"),
    LayerSpec("active AlGaAs barrier 1", 20.0, 3.445, "#ccebc5"),
    LayerSpec("active InGaAs QWs", 10.0, 3.584, "#f0027f"),
    LayerSpec("active AlGaAs barrier 2", 20.0, 3.445, "#ccebc5"),
    LayerSpec("active InGaAs QWs", 10.0, 3.584, "#f0027f"),
    LayerSpec("AlGaAs", 80.0, 3.445, "#ccebc5"),
    LayerSpec("n-clad AlGaAs", 1000.0, 3.122, "#bebada"),
)
PC_LAYER_NAME = "photonic crystal"
ACTIVE_LAYER_NAME = "active InGaAs QWs"
TOP_CLADDING_INDEX = 3.122
BOTTOM_CLADDING_INDEX = 3.297
VERTICAL_PADDING_UM = 1.0


# ===========================================================================
# PARAMETER PANEL B - four-wave optical model and finite device
# 参数区 B：纵向光学、有限器件尺寸与数值收敛设置。
# ===========================================================================
ACTIVE_INDEX = 3.584
ACTIVE_CONFINEMENT_FACTOR = 0.044  # Inoue Table II
ACTIVE_CONFINEMENT_SOURCE = "paper"  # "paper" or "solved_vertical_TE0"
INTERNAL_LOSS_CM = 5.0
DEVICE_SIZE_UM = 300.0
TIME_DOMAIN_SIZE_UM = 400.0       # includes the 25 um current-spreading exterior
ELECTRODE_SIZE_UM = 300.0
ELECTRODE_SHAPE = "square"
CURRENT_SPREAD_UM = 25.0
PAPER_THRESHOLD_CURRENT_A = 0.7
THRESHOLD_CURRENT_SOURCE = "paper"  # paper=0.7 A; derived value is still audited

# Accuracy controls for geometry -> C1D+Crad+C2D and finite Eq. (4.21).
# D=10 is the convergence setting stated in Liang Chapter 3.
# 这些是精度参数而不是器件参数；正式结果至少比较两档 D 和多档有限网格。
CWT_TRUNCATION_ORDER = 10
VERTICAL_STEP_NM = 3.0
FINITE_EIGEN_GRIDS = (13, 17, 21, 25)  # extrapolated to zero grid spacing
LENGTH_SWEEP_GRID_POINTS = (11, 15, 19)
LENGTH_SWEEP_UM = (70.0, 150.0, 300.0, 500.0)


# ===========================================================================
# PARAMETER PANEL C - semiconductor carrier and time-domain model
# 参数区 C：Inoue 式 (10)-(11) 的半导体载流子与时域设置。
# ===========================================================================
MAXIMUM_GAIN_CM = 2000.0
ZERO_CARRIER_GAIN_CM = -5000.0
TRANSPARENCY_DENSITY_CM3 = 1.5e18
CARRIER_LIFETIME_NS = 1.5
DIFFUSION_CM2_S = 100.0
ACTIVE_THICKNESS_NM = 30.0
SPONTANEOUS_EMISSION_FACTOR = 1.0e-4
DN_DN_CM3 = -5.0e-21

TIME_GRID_POINTS = 31             # odd integer >= 9
END_TIME_NS = 10.0                # publication-style window; use 3.0 for a quick check
CFL = 0.8
SAMPLE_INTERVAL_PS = 0.1          # resolves the geometry-derived optical detuning
RANDOM_SEED = 19
ENABLE_NOISE = True
CURRENT_RATIOS = (1.05, 1.40, 2.80, 4.20)
SPECTRUM_WINDOW_NS = 2.0


# ===========================================================================
# STEP SWITCHES / 步骤开关：False 只跳过输出，不会更改其他步骤的物理参数。
# ===========================================================================
STEPS = {
    "01_parameters": True,    # save inputs and C1D/Crad/C2D / 保存参数与耦合分量
    "02_lattice": True,       # real-space unit cell / 实空间晶胞
    "03_k_space": True,       # retained Rx,Sx,Ry,Sy waves / 四个基本 Bloch 波
    "04_layer_stack": True,   # scalar TE0 and overlap / 纵向 TE0 与限制因子
    "05_band_structure": True, # M-Gamma-X normalized-frequency bands / M-Gamma-X 能带
    "05_linear_modes": True,  # Liang Eqs. 4.21-4.23 / 有限区域冷腔模与阈值
    "06_mode_atlas": True,    # device, unit cell, vector FFP / 包络、晶胞场与矢量远场
    "07_length_sweep": True,  # finite-size scaling / 器件尺寸扫描
    "08_time_domain": True,   # Inoue Eqs. 8-11; slowest / 最慢的非线性时域步骤
}

OUTPUT_DIRECTORY = PROJECT_ROOT/"results"/"custom_semiconductor_inoue2019"


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


def build_specs():
    lattice = LatticeSpec(
        lattice_type=LATTICE_TYPE,
        constant_nm=LATTICE_CONSTANT_NM,
        hole_shape=HOLE_SHAPE,
        hole_radius_x_nm=HOLE_RADIUS_X_NM,
        hole_radius_y_nm=HOLE_RADIUS_Y_NM,
        hole_rotation_deg=HOLE_ROTATION_DEG,
        background_index=BACKGROUND_INDEX,
        hole_index=HOLE_INDEX,
        inclusions=UNIT_CELL_INCLUSIONS,
    )
    ensure_square_four_wave(lattice)
    cell = build_unit_cell(lattice)
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
        top_index=TOP_CLADDING_INDEX,
        bottom_index=BOTTOM_CLADDING_INDEX,
        padding_um=VERTICAL_PADDING_UM,
    )
    geometry_model = build_geometry_coupling(
        cell=cell,
        stack=stack,
        pc_layer_name=PC_LAYER_NAME,
        lattice_constant_nm=LATTICE_CONSTANT_NM,
        wavelength_guess_nm=WAVELENGTH_GUESS_NM,
        settings=ThreeDCWTSettings(
            truncation_order=CWT_TRUNCATION_ORDER,
            vertical_step_nm=VERTICAL_STEP_NM,
        ),
    )
    modal_values_cm = geometry_model.eigenvalues_m/100.0
    solved_active_confinement = geometry_model.vertical_mode.confinement[ACTIVE_LAYER_NAME]
    if ACTIVE_CONFINEMENT_SOURCE == "paper":
        active_confinement = ACTIVE_CONFINEMENT_FACTOR
    elif ACTIVE_CONFINEMENT_SOURCE == "solved_vertical_TE0":
        active_confinement = solved_active_confinement
    else:
        raise ValueError(
            "ACTIVE_CONFINEMENT_SOURCE must be 'paper' or 'solved_vertical_TE0'"
        )
    linear = FourWaveOpticalSpec(
        wavelength_nm=geometry_model.bragg_wavelength_nm,
        effective_index=geometry_model.effective_index,
        group_index=geometry_model.group_index,
        domain_um=DEVICE_SIZE_UM,
        internal_loss_cm=INTERNAL_LOSS_CM,
        modal_detuning_cm=tuple(modal_values_cm.real),
        modal_radiation_loss_cm=tuple(modal_values_cm.imag),
        grid_points=FINITE_EIGEN_GRIDS[-1],
    )
    simulation = SimulationConfig(
        optical=OpticalConfig(
            lattice_constant_nm=LATTICE_CONSTANT_NM,
            wavelength_nm=geometry_model.bragg_wavelength_nm,
            group_index=geometry_model.group_index,
            effective_index=geometry_model.effective_index,
            active_index=ACTIVE_INDEX,
            internal_loss_cm=INTERNAL_LOSS_CM,
            confinement_factor=active_confinement,
            dn_dN_cm3=DN_DN_CM3,
            modal_detuning_cm=tuple(modal_values_cm.real),
            modal_radiation_loss_cm=tuple(modal_values_cm.imag),
        ),
        carrier=CarrierConfig(
            maximum_gain_cm=MAXIMUM_GAIN_CM,
            zero_carrier_gain_cm=ZERO_CARRIER_GAIN_CM,
            transparency_density_cm3=TRANSPARENCY_DENSITY_CM3,
            lifetime_ns=CARRIER_LIFETIME_NS,
            diffusion_cm2_s=DIFFUSION_CM2_S,
            active_thickness_nm=ACTIVE_THICKNESS_NM,
            spontaneous_emission_factor=SPONTANEOUS_EMISSION_FACTOR,
        ),
        device=DeviceConfig(
            domain_um=TIME_DOMAIN_SIZE_UM,
            electrode_um=ELECTRODE_SIZE_UM,
            current_spread_um=CURRENT_SPREAD_UM,
            threshold_current_A=PAPER_THRESHOLD_CURRENT_A,
            electrode_shape=ELECTRODE_SHAPE,
        ),
        numerics=NumericsConfig(
            points=TIME_GRID_POINTS,
            end_time_ns=END_TIME_NS,
            cfl=CFL,
            sample_interval_ps=SAMPLE_INTERVAL_PS,
            seed=RANDOM_SEED,
            noise=ENABLE_NOISE,
        ),
        reproduction=ReproductionConfig(
            current_ratios=CURRENT_RATIOS,
            spectrum_window_ns=SPECTRUM_WINDOW_NS,
        ),
    )
    validate_config(simulation)
    return lattice, linear, simulation, display_layers, geometry_model


def threshold_current_audit(simulation: SimulationConfig, mode) -> dict[str, float | str]:
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
            "change the geometry/layers or use THRESHOLD_CURRENT_SOURCE='paper' only "
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
        "paper_reference_threshold_A": PAPER_THRESHOLD_CURRENT_A,
        "threshold_condition": "g_modal = alpha_internal + 2 alpha_mode",
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an editable semiconductor PCSEL workflow")
    parser.add_argument("--only", nargs="*", choices=tuple(STEPS), help="run selected steps only")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    switches = STEPS.copy()
    if args.only:
        switches = {name: name in args.only for name in switches}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lattice, linear, simulation, display_layers, geometry_model = build_specs()
    modes = None
    convergence = None
    threshold_audit = None

    print("="*78)
    print(RUN_TITLE)
    print(f"output={output}")
    print(
        f"Bragg lambda={geometry_model.bragg_wavelength_nm:.4f} nm, "
        f"neff={geometry_model.effective_index:.6f}, ng={geometry_model.group_index:.6f}"
    )
    print(f"lattice={LATTICE_TYPE}, hole={HOLE_SHAPE}, a={LATTICE_CONSTANT_NM} nm")
    print(
        f"finite PC={DEVICE_SIZE_UM} um, time domain={TIME_DOMAIN_SIZE_UM} um, "
        f"electrode={ELECTRODE_SHAPE} {ELECTRODE_SIZE_UM} um, "
        f"grid={TIME_GRID_POINTS}x{TIME_GRID_POINTS}"
    )
    print("="*78)

    def need_modes():
        nonlocal modes, convergence, simulation, threshold_audit
        ensure_square_four_wave(lattice)
        if modes is None:
            print(f"    finite-area convergence grids={FINITE_EIGEN_GRIDS} ...")
            modes, convergence = solve_finite_modes_converged(
                linear,
                FINITE_EIGEN_GRIDS,
                geometry_model.coupling_m,
                geometry_model.eigenvectors,
                geometry_model.radiation_fields,
            )
            lasing_name = min(MODE_NAMES, key=lambda name: modes[name].alpha_per_m)
            threshold_audit = threshold_current_audit(simulation, modes[lasing_name])
            selected_threshold = (
                threshold_audit["derived_uniform_injection_threshold_A"]
                if THRESHOLD_CURRENT_SOURCE == "derived"
                else PAPER_THRESHOLD_CURRENT_A
            )
            if THRESHOLD_CURRENT_SOURCE not in {"derived", "paper"}:
                raise ValueError("THRESHOLD_CURRENT_SOURCE must be 'derived' or 'paper'")
            simulation = replace(
                simulation,
                device=replace(
                    simulation.device, threshold_current_A=float(selected_threshold)
                ),
            )
            threshold_audit["selected_threshold_source"] = THRESHOLD_CURRENT_SOURCE
            threshold_audit["selected_threshold_A"] = float(selected_threshold)
            (output/"05_threshold_current_audit.json").write_text(
                json.dumps(threshold_audit, indent=2), encoding="utf-8"
            )
        return modes

    if switches["01_parameters"]:
        print("[01] parameter report")
        need_modes()  # makes the saved threshold current consistent with the solved cavity
        write_parameter_report(output, lattice, linear, display_layers, {
            "device_preset": DEVICE_PRESET,
            "geometry_fidelity": GEOMETRY_FIDELITY,
            "carrier": simulation.carrier.__dict__,
            "device": simulation.device.__dict__,
            "numerics": simulation.numerics.__dict__,
            "current_ratios": CURRENT_RATIOS,
            "cwt_truncation_order": CWT_TRUNCATION_ORDER,
            "vertical_step_nm": VERTICAL_STEP_NM,
            "finite_eigen_grids": FINITE_EIGEN_GRIDS,
            "bragg_wavelength_nm": geometry_model.bragg_wavelength_nm,
            "derived_effective_index": geometry_model.effective_index,
            "derived_group_index": geometry_model.group_index,
            "used_active_confinement": simulation.optical.confinement_factor,
            "solved_vertical_TE0_active_confinement": (
                geometry_model.vertical_mode.confinement[ACTIVE_LAYER_NAME]
            ),
            "active_confinement_source": ACTIVE_CONFINEMENT_SOURCE,
            "threshold_current_source": THRESHOLD_CURRENT_SOURCE,
            "paper_threshold_current_A": PAPER_THRESHOLD_CURRENT_A,
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
                for m in range(-CWT_TRUNCATION_ORDER-1, CWT_TRUNCATION_ORDER+2)
                for n in range(-CWT_TRUNCATION_ORDER-1, CWT_TRUNCATION_ORDER+2)
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
    if switches["03_k_space"]:
        print("[03] reciprocal lattice and retained Bloch waves")
        plot_k_space(lattice, output/"03_k_space.png")
    if switches["04_layer_stack"]:
        print("[04] vertical index stack")
        plot_layer_stack(display_layers, output/"04_layer_stack.png")
        plot_vertical_mode(
            geometry_model.vertical_mode, PC_LAYER_NAME,
            output/"04_vertical_TE0.png",
        )
    if switches["05_band_structure"]:
        print("[05-band] four-wave M <- Gamma -> X normalized-frequency bands")
        band_diagram = square_four_wave_band_diagram(
            geometry_model.coupling_m,
            lattice_constant_nm=LATTICE_CONSTANT_NM,
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
    if switches["07_length_sweep"]:
        print("[07] finite-size threshold sweep")
        ensure_square_four_wave(lattice)
        sweep = plot_length_sweep_from_solver(
            linear,
            geometry_model.coupling_m,
            geometry_model.eigenvectors,
            geometry_model.radiation_fields,
            output/"07_length_sweep.png",
            lengths_um=np.asarray(LENGTH_SWEEP_UM),
            grid_points=LENGTH_SWEEP_GRID_POINTS,
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
        for ratio in CURRENT_RATIOS:
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
                "current_ratio", "steady_window_ns", "mean_power_W",
                "maximum_power_W", "mean_center_carrier_cm-3",
            ))
            for result in results:
                steady = result.time_ns >= result.time_ns[-1]-1.0
                writer.writerow((
                    result.current_ratio,
                    1.0,
                    float(np.mean(result.power_W[steady])),
                    float(np.max(result.power_W)),
                    float(np.mean(result.center_carrier_cm3[steady])),
                ))
        plot_spectra(
            results, output/"08_spectra.png", SPECTRUM_WINDOW_NS,
            center_wavelength_nm=reference_wavelength_nm,
        )
    print("Finished. Edit the parameter panel at the top of this script for your device.")


if __name__ == "__main__":
    main()
