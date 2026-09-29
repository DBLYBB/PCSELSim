"""Editable Yb:YAG photonic-crystal laser workflow.

The optical part uses the same square-lattice four-wave coupled-wave basis as
the semiconductor platform.  Semiconductor carriers are replaced by a
quasi-three-level Yb:YAG population reservoir with ground-state reabsorption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import numpy as np

from pcselsim.custom_analysis import (
    FourWaveOpticalSpec, LatticeSpec, LayerSpec, ensure_square_four_wave,
    plot_k_space, plot_lattice, plot_layer_stack, plot_length_sweep,
    plot_mode_atlas, plot_threshold_summary, solve_finite_modes,
    write_mode_table, write_parameter_report,
)
from pcselsim.solid_state import (
    YbYAGMediumConfig, plot_gain_curve, plot_pump_scan, plot_rate_dynamics,
    solve_ybyag_rates,
)


# ===========================================================================
# PARAMETER PANEL A - PC geometry at the Yb:YAG laser wavelength
# ===========================================================================
LASER_WAVELENGTH_NM = 1030.0
PUMP_WAVELENGTH_NM = 940.0
LATTICE_TYPE = "square"       # current optical solver is square-lattice only
LATTICE_CONSTANT_NM = 565.0   # approximately lambda/n_eff at second-order Gamma
HOLE_SHAPE = "circle"         # circle / ellipse / triangle / rit
HOLE_RADIUS_X_NM = 125.0
HOLE_RADIUS_Y_NM = 125.0
HOLE_ROTATION_DEG = 0.0
YAG_REFRACTIVE_INDEX = 1.82
HOLE_INDEX = 1.0

LAYERS = (
    LayerSpec("YAG substrate", 500000.0, 1.82, "#bebada"),
    LayerSpec("Yb:YAG gain layer", 200000.0, 1.82, "#80b1d3"),
    LayerSpec("patterned Yb:YAG PC", 3000.0, 1.72, "#fdb462"),
    LayerSpec("air superstrate", 10000.0, 1.0, "#ffffff"),
)


# ===========================================================================
# PARAMETER PANEL B - optical cavity and four-wave coupling
# ===========================================================================
EFFECTIVE_INDEX = 1.78
GROUP_INDEX = 1.83
DEVICE_SIZE_UM = 200.0
INTERNAL_LOSS_CM = 0.05
CONFINEMENT_FACTOR = 0.80
MODAL_DETUNING_CM = (-80.0, -25.0, 35.0, 90.0)
MODAL_RADIATION_LOSS_CM = (0.40, 0.80, 2.00, 2.40)
EDGE_LOSS_CM = (0.25, 0.32, 0.55, 0.55)
FINITE_EIGEN_GRID_POINTS = 17


# ===========================================================================
# PARAMETER PANEL C - Yb:YAG quasi-three-level medium
# Replace cross sections with measurements for your temperature/doping batch.
# ===========================================================================
YB_DOPANT_DENSITY_CM3 = 1.38e20       # approximately 1 at.% Yb on Y sites
UPPER_STATE_LIFETIME_MS = 0.95
PUMP_ABSORPTION_CROSS_SECTION_CM2 = 0.70e-20
PUMP_EMISSION_CROSS_SECTION_CM2 = 0.10e-20
LASER_ABSORPTION_CROSS_SECTION_CM2 = 0.126e-20
LASER_EMISSION_CROSS_SECTION_CM2 = 2.00e-20
PUMP_INTENSITY_W_CM2 = 2.50e4       # above the default A-mode threshold
PUMPED_AREA_UM2 = 200.0**2
GAIN_THICKNESS_UM = 200.0
SPONTANEOUS_EMISSION_FACTOR = 1.0e-8
RATE_END_TIME_MS = 5.0
RATE_SAMPLES = 1000


# ===========================================================================
# STEP SWITCHES
# ===========================================================================
STEPS = {
    "01_parameters": True,
    "02_lattice": True,
    "03_k_space": True,
    "04_layer_stack": True,
    "05_linear_modes": True,
    "06_mode_atlas": True,
    "07_length_sweep": True,
    "08_ybyag_gain": True,
    "09_rate_dynamics": True,
    "10_pump_scan": True,
}

OUTPUT_DIRECTORY = PROJECT_ROOT/"results"/"custom_ybyag_pcsel"


def build_specs():
    lattice = LatticeSpec(
        lattice_type=LATTICE_TYPE,
        constant_nm=LATTICE_CONSTANT_NM,
        hole_shape=HOLE_SHAPE,
        hole_radius_x_nm=HOLE_RADIUS_X_NM,
        hole_radius_y_nm=HOLE_RADIUS_Y_NM,
        hole_rotation_deg=HOLE_ROTATION_DEG,
        background_index=YAG_REFRACTIVE_INDEX,
        hole_index=HOLE_INDEX,
    )
    optical = FourWaveOpticalSpec(
        wavelength_nm=LASER_WAVELENGTH_NM,
        effective_index=EFFECTIVE_INDEX,
        group_index=GROUP_INDEX,
        domain_um=DEVICE_SIZE_UM,
        internal_loss_cm=INTERNAL_LOSS_CM,
        modal_detuning_cm=MODAL_DETUNING_CM,
        modal_radiation_loss_cm=MODAL_RADIATION_LOSS_CM,
        grid_points=FINITE_EIGEN_GRID_POINTS,
    )
    medium = YbYAGMediumConfig(
        pump_wavelength_nm=PUMP_WAVELENGTH_NM,
        laser_wavelength_nm=LASER_WAVELENGTH_NM,
        refractive_index=YAG_REFRACTIVE_INDEX,
        upper_state_lifetime_ms=UPPER_STATE_LIFETIME_MS,
        dopant_density_cm3=YB_DOPANT_DENSITY_CM3,
        pump_absorption_cross_section_cm2=PUMP_ABSORPTION_CROSS_SECTION_CM2,
        pump_emission_cross_section_cm2=PUMP_EMISSION_CROSS_SECTION_CM2,
        laser_absorption_cross_section_cm2=LASER_ABSORPTION_CROSS_SECTION_CM2,
        laser_emission_cross_section_cm2=LASER_EMISSION_CROSS_SECTION_CM2,
        confinement_factor=CONFINEMENT_FACTOR,
        spontaneous_emission_factor=SPONTANEOUS_EMISSION_FACTOR,
        pump_intensity_W_cm2=PUMP_INTENSITY_W_CM2,
        pumped_area_um2=PUMPED_AREA_UM2,
        gain_thickness_um=GAIN_THICKNESS_UM,
        end_time_ms=RATE_END_TIME_MS,
        samples=RATE_SAMPLES,
    )
    return lattice, optical, medium


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an editable Yb:YAG PCSEL workflow")
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
    lattice, optical, medium = build_specs()
    modes = None

    print("="*78)
    print("CUSTOM Yb:YAG PHOTONIC-CRYSTAL LASER")
    print(f"output={output}")
    print(f"pump={PUMP_WAVELENGTH_NM} nm, laser={LASER_WAVELENGTH_NM} nm")
    print(f"a={LATTICE_CONSTANT_NM} nm, device={DEVICE_SIZE_UM} um")
    print("model=quasi-three-level Yb:YAG reservoir + square-lattice four-wave CWT")
    print("="*78)

    def need_modes():
        nonlocal modes
        ensure_square_four_wave(lattice)
        if modes is None:
            print("    solving finite-area A/B/C/D modes ...")
            modes = solve_finite_modes(optical)
        return modes

    # Rate-equation losses are explicit and editable.  This avoids confusing
    # finite-difference numerical outflow with measured crystal/cavity loss.
    radiation_loss_m = 100.0*np.asarray(MODAL_RADIATION_LOSS_CM)
    total_loss_m = 100.0*(
        INTERNAL_LOSS_CM+np.asarray(MODAL_RADIATION_LOSS_CM)+np.asarray(EDGE_LOSS_CM)
    )

    if switches["01_parameters"]:
        print("[01] parameter report")
        write_parameter_report(output, lattice, optical, LAYERS, {
            "ybyag_medium": medium.__dict__,
            "edge_loss_cm": EDGE_LOSS_CM,
            "rate_model": "spatially averaged quasi-three-level reservoir",
        })
    if switches["02_lattice"]:
        print("[02] real-space lattice")
        plot_lattice(lattice, output/"02_lattice.png")
    if switches["03_k_space"]:
        print("[03] reciprocal lattice and retained Bloch waves")
        plot_k_space(lattice, output/"03_k_space.png")
    if switches["04_layer_stack"]:
        print("[04] vertical index stack")
        plot_layer_stack(LAYERS, output/"04_layer_stack.png")
    if switches["05_linear_modes"]:
        print("[05] finite-area optical thresholds")
        current_modes = need_modes()
        plot_threshold_summary(optical, current_modes, output/"05_thresholds.png")
        write_mode_table(output, optical, current_modes)
    if switches["06_mode_atlas"]:
        print("[06] whole-device, unit-cell and far-field mode atlas")
        plot_mode_atlas(lattice, optical, need_modes(), output/"06_mode_atlas.png")
    if switches["07_length_sweep"]:
        print("[07] finite-size threshold sweep")
        ensure_square_four_wave(lattice)
        plot_length_sweep(optical, output/"07_length_sweep.png")
    if switches["08_ybyag_gain"]:
        print("[08] Yb:YAG gain, reabsorption and threshold fractions")
        plot_gain_curve(medium, total_loss_m, ("A", "B", "C", "D"), output/"08_ybyag_gain.png")
    if switches["09_rate_dynamics"]:
        print("[09] quasi-three-level rate-equation dynamics")
        result = solve_ybyag_rates(
            medium, ("A", "B", "C", "D"), total_loss_m, radiation_loss_m
        )
        plot_rate_dynamics(result, output/"09_rate_dynamics.png")
        np.savez_compressed(
            output/"09_rate_dynamics.npz",
            time_ms=result.time_ms,
            excited_fraction=result.excited_fraction,
            photon_density_m3=result.photon_density_m3,
            output_power_W=result.output_power_W,
            threshold_fractions=result.threshold_fractions,
        )
    if switches["10_pump_scan"]:
        print("[10] steady pump scan")
        plot_pump_scan(
            medium, total_loss_m, radiation_loss_m,
            ("A", "B", "C", "D"), output/"10_pump_scan.png",
        )
    print("Finished. Treat default Yb:YAG cross sections as replaceable room-temperature examples.")


if __name__ == "__main__":
    main()
