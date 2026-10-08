"""Editable triangular-lattice semiconductor PCSEL six-wave workflow.

PyCharm: edit the PARAMETER PANEL and STEP SWITCHES, then run this file.
The implementation follows Liang thesis Sec. 5.2 and Appendix B.  It builds a
new 6x6 matrix from the triangular unit cell and the solved vertical TE0 mode;
the square-lattice four-wave solver is neither called nor modified.

中文操作：修改下方参数区和步骤开关，然后在 PyCharm 中右键运行。本脚本按
“三角晶胞 Fourier 系数 -> 纵向 TE0 -> 六波 Cb+Cr+Ch -> Gamma 带边态 ->
M-Gamma-X 能带 -> 有限器件阈值/包络 -> 矢量远场”执行。有限器件步骤使用
六方向六角特征网格和零入射开放边界；它不调用四波求解器。载流子时域尚未实现。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import numpy as np

from pcselsim.band_structure import plot_band_diagram
from pcselsim.custom_analysis import plot_vertical_mode
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
    TRIANGULAR_DIRECTIONS,
    TRIANGULAR_MODE_NAMES,
    TRIANGULAR_POLARIZATIONS,
    TriangularCWTSettings,
    TriangularEllipse,
    TriangularLatticeCell,
    build_triangular_coupling,
    plot_radiation_constants,
    plot_six_band_edge_states,
    plot_triangular_lattice,
    plot_triangular_reciprocal_space,
    triangular_band_diagram,
)
from pcselsim.vertical import Layer, LayerStack


# ===========================================================================
# PARAMETER PANEL A - triangular lattice and semiconductor material
# 参数区 A：三角晶格与材料。圆孔半径改变会重新计算全部 Fourier 系数和 6x6 C。
# ===========================================================================
RUN_TITLE = "CUSTOM TRIANGULAR-LATTICE SEMICONDUCTOR PCSEL - SIX-WAVE 3D-CWT"
LATTICE_CONSTANT_NM = 341.0       # Liang Ch. 5 default
HOLE_RADIUS_OVER_A = 0.20         # paper: f about 0.15 for r/a about 0.20
BACKGROUND_EPSILON = 12.7449      # GaAs in Liang Table 3.1
HOLE_EPSILON = 1.0                # air
WAVELENGTH_GUESS_NM = 995.0

# One centered circular air hole.  Add more TriangularEllipse entries for a
# multi-atom triangular motif; centers use fractional coordinates u*a1+v*a2.
UNIT_CELL_INCLUSIONS = (
    TriangularEllipse(
        center_fractional=(0.0, 0.0),
        radii_over_a=(HOLE_RADIUS_OVER_A, HOLE_RADIUS_OVER_A),
        angle_deg=0.0,
        epsilon=HOLE_EPSILON,
    ),
)


# ===========================================================================
# PARAMETER PANEL B - Liang Table 3.1 vertical structure, bottom to top
# 参数区 B：Liang 表 3.1 的 GaAs/AlGaAs 标量 TE0 纵向结构。
# ===========================================================================
PC_LAYER_NAME = "PC"
VERTICAL_LAYERS_FIXED = (
    ("p-clad AlGaAs", 1500.0, np.sqrt(11.0224)),
    ("GaAs", 59.0, np.sqrt(12.7449)),
    # The PC-layer index is replaced by sqrt(<epsilon>) below.
    (PC_LAYER_NAME, 118.0, 0.0),
    ("active InGaAs/GaAs", 88.5, np.sqrt(12.8603)),
    ("n-clad AlGaAs", 1500.0, np.sqrt(11.0224)),
)
TOP_CLADDING_INDEX = np.sqrt(11.0224)
BOTTOM_CLADDING_INDEX = np.sqrt(11.0224)
VERTICAL_PADDING_UM = 1.0


# ===========================================================================
# PARAMETER PANEL C - numerical convergence and band window
# 参数区 C：高阶波、纵向网格和 Gamma 点附近能带范围。
# ===========================================================================
CWT_TRUNCATION_ORDER = 10         # C6-symmetric shell m^2-mn+n^2 <= D^2
VERTICAL_STEP_NM = 3.0
BAND_Q_MAX_2PI_OVER_A = 0.05      # use 0.01 for the narrow style in the example image
BAND_POINTS = 241
UNIT_CELL_FIELD_POINTS = 101


# ===========================================================================
# PARAMETER PANEL D - finite device and physical far field
# 参数区 D：圆形/正六边形/方形器件。网格越密越准确，但稀疏本征求解耗时也越长。
# ===========================================================================
DEVICE_RADIUS_UM = 30.0           # Liang Ch. 5 finite-device radius L
DEVICE_SHAPE = "circle"           # circle / hexagon / square
FINITE_RADIUS_CELLS = (7, 9, 11)  # increasing N; eigenvalues are extrapolated to 1/N=0
INTERNAL_LOSS_CM = 0.0            # add measured material/internal amplitude loss here
FAR_FIELD_VIEW_DEG = 3.0


# ===========================================================================
# STEP SWITCHES / 步骤开关
# ===========================================================================
STEPS = {
    "01_parameters": True,       # input, six-wave matrices and Gamma table
    "02_lattice": True,          # real-space triangular lattice
    "03_k_space": True,          # six retained reciprocal-space waves
    "04_vertical_mode": True,    # scalar TE0 and PC overlap
    "05_band_structure": True,   # M <- Gamma -> X normalized-frequency plot
    "06_radiation_constants": True,
    "07_band_edge_states": True, # six unit-cell E-field states
    "08_finite_modes": True,      # six-wave open-boundary thresholds and envelopes
    "09_finite_mode_atlas": True, # device envelope + unit cell + vector far field
    "10_best_mode_far_field": True,
}

OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "custom_semiconductor_triangular_six_wave"


def build_model():
    """Build the triangular cell, vertical stack and geometry-derived six-wave C."""
    cell = TriangularLatticeCell(
        background_epsilon=BACKGROUND_EPSILON,
        inclusions=UNIT_CELL_INCLUSIONS,
    )
    average_pc_index = float(np.sqrt(cell.fourier_epsilon(0, 0).real))
    layers = tuple(
        Layer(name, thickness_nm, average_pc_index if name == PC_LAYER_NAME else index)
        for name, thickness_nm, index in VERTICAL_LAYERS_FIXED
    )
    stack = LayerStack(
        layers=layers,
        top_index=float(TOP_CLADDING_INDEX),
        bottom_index=float(BOTTOM_CLADDING_INDEX),
        padding_um=VERTICAL_PADDING_UM,
    )
    model = build_triangular_coupling(
        cell=cell,
        stack=stack,
        pc_layer_name=PC_LAYER_NAME,
        lattice_constant_nm=LATTICE_CONSTANT_NM,
        wavelength_guess_nm=WAVELENGTH_GUESS_NM,
        settings=TriangularCWTSettings(
            truncation_order=CWT_TRUNCATION_ORDER,
            vertical_step_nm=VERTICAL_STEP_NM,
        ),
    )
    return cell, stack, model


def _encode_matrix(matrix: np.ndarray) -> list[list[list[float]]]:
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in np.asarray(matrix)
    ]


def write_parameter_and_mode_reports(output: Path, cell, stack, model) -> None:
    """Write enough raw data to reproduce every plotted Gamma-state number."""
    report = {
        "model": "Liang triangular-lattice TE six-wave 3-D CWT",
        "references": {
            "six_wave_equation": "Liang thesis Eq. (5.15)-(5.17)",
            "coupling_matrix": "Liang thesis Appendix B, Eqs. (B6)-(B14)",
            "journal": "Optics Express 21, 565-580 (2013), DOI 10.1364/OE.21.000565",
        },
        "implemented_scope": [
            "infinite-periodic Gamma eigenstates",
            "near-Gamma M-Gamma-X band structure",
            "radiation constants",
            "unit-cell fields including retained high-order responses",
            "finite circle/hexagon/square six-wave envelopes on a characteristic grid",
            "vector far fields from the geometry-derived radiative aperture",
        ],
        "not_implemented_scope": [
            "the exact generalized staggered-mass matrix form of Liang Eq. (5.18)",
            "six-wave semiconductor carrier-field time domain",
            "full-vector vertical mode and interface-reflected Green function",
        ],
        "lattice_constant_nm": LATTICE_CONSTANT_NM,
        "hole_radius_over_a": HOLE_RADIUS_OVER_A,
        "computed_fill_fraction": cell.fill_fraction,
        "background_epsilon": BACKGROUND_EPSILON,
        "hole_epsilon": HOLE_EPSILON,
        "average_pc_epsilon": model.average_pc_epsilon,
        "bragg_wavelength_nm": model.bragg_wavelength_nm,
        "effective_index": model.effective_index,
        "group_index": model.group_index,
        "pc_confinement": model.pc_confinement,
        "truncation_order": CWT_TRUNCATION_ORDER,
        "truncation_rule": "m^2-m*n+n^2 <= D^2 (C6-symmetric shell)",
        "vertical_step_nm": VERTICAL_STEP_NM,
        "finite_device_radius_um": DEVICE_RADIUS_UM,
        "finite_device_shape": DEVICE_SHAPE,
        "finite_radius_cells": list(FINITE_RADIUS_CELLS),
        "internal_loss_cm-1": INTERNAL_LOSS_CM,
        "vertical_layers": [layer.__dict__ for layer in stack.layers],
        "basic_orders": TRIANGULAR_BASIC_ORDERS,
        "directions_beta0": TRIANGULAR_DIRECTIONS.tolist(),
        "te_polarizations": TRIANGULAR_POLARIZATIONS.tolist(),
        "matrix_units": "m^-1; complex entries stored as [real, imaginary]",
        "Cb": _encode_matrix(model.cb_m),
        "Cr": _encode_matrix(model.cr_m),
        "Ch": _encode_matrix(model.ch_m),
        "C_total": _encode_matrix(model.coupling_m),
        "passivity_projection_correction_m^-1": model.passivity_correction_m,
    }
    (output / "01_parameters_and_coupling.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    with (output / "01_gamma_modes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "mode", "detuning_cm-1", "amplitude_loss_cm-1",
            "radiation_constant_cm-1", "normalized_frequency_a_over_lambda",
            "radiation_Ex_abs", "radiation_Ey_abs",
        ))
        for index, name in enumerate(TRIANGULAR_MODE_NAMES):
            value = model.eigenvalues_m[index]
            ex, ey = model.radiation_amplitudes(model.eigenvectors[:, index])
            normalized_frequency = (
                LATTICE_CONSTANT_NM / model.bragg_wavelength_nm
                + value.real * LATTICE_CONSTANT_NM * 1e-9
                / (2.0 * np.pi * model.effective_index)
            )
            writer.writerow((
                name, value.real / 100.0, max(value.imag, 0.0) / 100.0,
                2.0 * max(value.imag, 0.0) / 100.0,
                normalized_frequency, abs(ex), abs(ey),
            ))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the triangular-lattice semiconductor PCSEL six-wave workflow"
    )
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

    print("=" * 82)
    print(RUN_TITLE)
    print(f"output={output}")
    print("building geometry-derived Cb + Cr + Ch ...")
    cell, stack, model = build_model()
    print(
        f"a={LATTICE_CONSTANT_NM:.3f} nm, r/a={HOLE_RADIUS_OVER_A:.4f}, "
        f"fill={cell.fill_fraction:.5f}"
    )
    print(
        f"Bragg lambda={model.bragg_wavelength_nm:.4f} nm, "
        f"neff={model.effective_index:.6f}, ng={model.group_index:.6f}"
    )
    for name, value in zip(TRIANGULAR_MODE_NAMES, model.eigenvalues_m / 100.0, strict=True):
        print(
            f"{name:>2s}: detuning={value.real:+10.3f} cm^-1, "
            f"alpha_r={2.0 * max(value.imag, 0.0):9.3f} cm^-1"
        )
    print("=" * 82)

    band = None
    finite_modes = None
    finite_convergence = None

    def need_band():
        nonlocal band
        if band is None:
            band = triangular_band_diagram(
                model, q_max=BAND_Q_MAX_2PI_OVER_A, points=BAND_POINTS
            )
        return band

    def need_finite_modes():
        nonlocal finite_modes, finite_convergence
        if finite_modes is None:
            print(
                "solving six-wave finite device: "
                f"shape={DEVICE_SHAPE}, size={DEVICE_RADIUS_UM:g} um, "
                f"N={FINITE_RADIUS_CELLS} ..."
            )
            finite_modes, finite_convergence = solve_triangular_finite_modes_converged(
                model,
                TriangularFiniteSpec(
                    radius_um=DEVICE_RADIUS_UM,
                    radius_cells=FINITE_RADIUS_CELLS[-1],
                    internal_loss_cm=INTERNAL_LOSS_CM,
                    aperture_shape=DEVICE_SHAPE,
                ),
                FINITE_RADIUS_CELLS,
            )
        return finite_modes

    if switches["01_parameters"]:
        print("[01] parameters, matrices and Gamma-state table")
        write_parameter_and_mode_reports(output, cell, stack, model)
    if switches["02_lattice"]:
        print("[02] triangular real-space lattice")
        plot_triangular_lattice(cell, output / "02_triangular_lattice.png")
    if switches["03_k_space"]:
        print("[03] reciprocal lattice and six retained waves")
        plot_triangular_reciprocal_space(output / "03_six_wave_k_space.png")
    if switches["04_vertical_mode"]:
        print("[04] scalar vertical TE0")
        plot_vertical_mode(model.vertical_mode, PC_LAYER_NAME, output / "04_vertical_TE0.png")
    if switches["05_band_structure"]:
        print("[05] six-wave M <- Gamma -> X band structure")
        plot_band_diagram(
            need_band(), output / "05_six_wave_band_structure.png",
            "Triangular six-wave bands: M - Gamma - X",
        )
    if switches["06_radiation_constants"]:
        print("[06] radiation constants along the band path")
        plot_radiation_constants(need_band(), output / "06_radiation_constants.png")
    if switches["07_band_edge_states"]:
        print("[07] six Gamma-point unit-cell states")
        plot_six_band_edge_states(
            model, output / "07_six_band_edge_states.png", points=UNIT_CELL_FIELD_POINTS
        )
    if switches["08_finite_modes"]:
        print("[08] finite-device thresholds and whole-device envelopes")
        modes = need_finite_modes()
        write_triangular_finite_table(modes, model, output / "08_finite_modes.csv")
        plot_triangular_thresholds(modes, output / "08_finite_thresholds.png")
        plot_triangular_grid_convergence(
            finite_convergence,
            DEVICE_RADIUS_UM,
            output / "08_grid_convergence.png",
        )
        for name, mode in modes.items():
            far = triangular_vector_far_field(
                mode, model.bragg_wavelength_nm, view_deg=FAR_FIELD_VIEW_DEG
            )
            print(
                f"  {name:>2s}: alpha={mode.alpha_per_m/100.0:8.3f} cm^-1, "
                f"alpha*L={mode.alpha_l:.4f}, center/max={far.center_to_peak:.3f}"
            )
    if switches["09_finite_mode_atlas"]:
        print("[09] six-wave envelope / unit-cell / vector-far-field atlas")
        plot_triangular_mode_atlas(
            need_finite_modes(), model, output / "09_finite_mode_atlas.png",
            view_deg=FAR_FIELD_VIEW_DEG,
        )
    if switches["10_best_mode_far_field"]:
        modes = need_finite_modes()
        best = min(modes.values(), key=lambda item: item.alpha_per_m)
        print(f"[10] vector far-field diagnostics for lowest-threshold mode {best.name}")
        plot_triangular_far_field_diagnostics(
            best, model, output / "10_best_mode_vector_far_field.png",
            view_deg=FAR_FIELD_VIEW_DEG,
        )
    print("Finished. Six-wave finite envelopes and vector far fields are included.")


if __name__ == "__main__":
    main()

