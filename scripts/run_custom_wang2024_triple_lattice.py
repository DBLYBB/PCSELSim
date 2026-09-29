"""Wang 2024 triple-lattice semiconductor PCSEL preset.

Run this file directly in PyCharm.  It reuses the same geometry-derived
3D-CWT and Inoue carrier-field solver as ``run_custom_semiconductor_pcsel``
but writes to an independent result directory.

The main paper specifies the in-plane geometry and measured device sizes but
puts the complete epitaxial recipe in Supplementary Table S1.  The layer stack
below is therefore an explicitly labelled effective Fig. 3(d) reconstruction;
it must not be presented as the authors' exact epitaxy.

中文：该预设复用自定义半导体主程序的同一套几何 3-D CWT 与 Inoue 时域内核。
主文公开的三圆孔晶胞可直接重建，但完整外延表和 InAlGaAs 动力学参数不足，故默认
只运行冷腔步骤。``08_time_domain`` 在获得可靠材料参数前保持 False；当前阈值电流
仅为审计估算，不能替代论文实验值。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
    from . import run_custom_semiconductor_pcsel as base
else:
    from _project_bootstrap import bootstrap_project
    import run_custom_semiconductor_pcsel as base

PROJECT_ROOT = bootstrap_project()


# ===========================================================================
# WANG 2024 PARAMETER PANEL - edit here for a derivative device
# Wang 2024 参数区：派生器件从这里改，结果写入独立目录，不覆盖 Inoue 输出。
# ===========================================================================
LATTICE_CONSTANT_NM = 474.0
HOLE_DIAMETER_NM = 90.0
HOLE_RADIUS_NM = 0.5 * HOLE_DIAMETER_NM
BACKGROUND_INDEX = 3.366       # effective PC background; tuned to Fig. 3(d)/1551 nm Bragg point
INP_HOLE_INDEX = 3.170         # InP-filled holes, not voids
PC_THICKNESS_NM = 430.0

PC_SIZE_UM = 300.0
TIME_DOMAIN_SIZE_UM = 300.0
P_MESA_DIAMETER_UM = 220.0
CONTACT_WINDOW_DIAMETER_UM = 200.0
N_RING_INNER_DIAMETER_UM = 240.0
CURRENT_SPREAD_UM = 8.0        # assumed radial smoothing; not specified in the main paper

REFERENCE_WAVELENGTH_NM = 1551.0
REFERENCE_PC_CONFINEMENT = 0.52
REFERENCE_MQW_CONFINEMENT = 0.06
REFERENCE_THRESHOLD_CURRENT_A = 0.52
REFERENCE_LOWEST_MODE_LOSS_CM = 28.0
REFERENCE_FAR_FIELD_DEG = (3.0, 17.0)  # experimental 1/e^2 widths (x, y)

# The main paper does not provide an InAlGaAs gain/recombination parameter set.
# These are editable placeholders only; step 08 is intentionally OFF by default.
MAXIMUM_GAIN_CM = 5000.0
ZERO_CARRIER_GAIN_CM = -6000.0
TRANSPARENCY_DENSITY_CM3 = 1.2e18
CARRIER_LIFETIME_NS = 1.0
DIFFUSION_CM2_S = 20.0
ACTIVE_THICKNESS_NM = 45.0
OPTICAL_MQW_REGION_THICKNESS_NM = 68.0  # QWs plus barriers in the effective index stack
SPONTANEOUS_EMISSION_FACTOR = 1.0e-4
DN_DN_CM3 = -2.0e-20

# The paper gives 0.4/0.5/0.6 A CW spectra and 1 A CW output.  These ratios use
# the measured 0.52 A threshold.  They are used only when step 08 is enabled.
CURRENT_RATIOS = (0.40/0.52, 0.50/0.52, 0.60/0.52, 1.00/0.52)

STEPS = {
    "01_parameters": True,   # also writes paper-vs-model audit / 同时写论文对照
    "02_lattice": True,      # published three-circle motif / 主文三圆孔晶胞
    "03_k_space": True,      # square-lattice four-wave basis / 方形四波基底
    "04_layer_stack": True,  # effective reconstruction, not exact epitaxy / 有效层结构
    "05_linear_modes": True, # cold-cavity modes and convergence / 冷腔模与收敛
    "06_mode_atlas": True,   # ideal single-mode fields / 理想单模场
    "07_length_sweep": True, # finite-area trend / 有限尺寸趋势
    "08_time_domain": False,  # enable only after supplying a validated InAlGaAs carrier model
}

OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "custom_semiconductor_wang2024_triple_lattice"


def configure_base_workflow() -> None:
    """Apply this preset to the shared semiconductor workflow.

    只替换器件参数和步骤开关；求解方程、单位约定及输出流程仍与半导体主程序相同。
    """
    a = LATTICE_CONSTANT_NM
    r = HOLE_RADIUS_NM
    base.DEVICE_PRESET = "wang2024_structure1_triple_lattice"
    base.RUN_TITLE = "CUSTOM SEMICONDUCTOR PCSEL - Wang 2024 triple lattice"
    base.GEOMETRY_FIDELITY = (
        "Wang 2024 main-text structure-1 geometry and device sizes; circular "
        "vertical hole walls; effective Fig. 3(d) layer stack because the exact "
        "epitaxy is only in Supplementary Table S1"
    )
    base.WAVELENGTH_GUESS_NM = REFERENCE_WAVELENGTH_NM
    base.LATTICE_TYPE = "square"
    base.LATTICE_CONSTANT_NM = a
    base.HOLE_SHAPE = "wang_triple_circle_structure_1"
    base.HOLE_RADIUS_X_NM = 0.0
    base.HOLE_RADIUS_Y_NM = 0.0
    base.HOLE_ROTATION_DEG = 0.0
    base.BACKGROUND_INDEX = BACKGROUND_INDEX
    base.HOLE_INDEX = INP_HOLE_INDEX
    base.UNIT_CELL_INCLUSIONS = (
        base.LatticeInclusionSpec("circle", r, r, -0.25*a, -0.25*a, 0.0),
        base.LatticeInclusionSpec("circle", r, r, +0.25*a, -0.25*a, 0.0),
        base.LatticeInclusionSpec("circle", r, r, +0.25*a, +0.25*a, 0.0),
    )

    # Effective bottom-to-top stack reconstructed from Fig. 3(c,d).  The PC
    # average index is replaced automatically by sqrt(<epsilon>) from the cell.
    base.LAYERS = (
        base.LayerSpec("n-InP cladding", 1500.0, 3.170, "#80b1d3"),
        base.LayerSpec("lower InAlGaAs SCH", 120.0, 3.285, "#b3de69"),
        base.LayerSpec(
            "active InAlGaAs MQWs", OPTICAL_MQW_REGION_THICKNESS_NM, 3.470, "#f0027f"
        ),
        base.LayerSpec("upper InAlGaAs SCH", 120.0, 3.285, "#b3de69"),
        base.LayerSpec("InP spacer", 80.0, 3.170, "#ccebc5"),
        base.LayerSpec("photonic crystal", PC_THICKNESS_NM, 0.0, "#fdb462"),
        base.LayerSpec("p-InP regrowth", 1500.0, 3.170, "#bebada"),
    )
    base.PC_LAYER_NAME = "photonic crystal"
    base.ACTIVE_LAYER_NAME = "active InAlGaAs MQWs"
    base.TOP_CLADDING_INDEX = 3.170
    base.BOTTOM_CLADDING_INDEX = 3.170
    base.VERTICAL_PADDING_UM = 1.5

    base.ACTIVE_INDEX = 3.470
    base.ACTIVE_CONFINEMENT_FACTOR = REFERENCE_MQW_CONFINEMENT
    base.ACTIVE_CONFINEMENT_SOURCE = "paper"
    base.INTERNAL_LOSS_CM = 5.0
    base.DEVICE_SIZE_UM = PC_SIZE_UM
    base.TIME_DOMAIN_SIZE_UM = TIME_DOMAIN_SIZE_UM
    base.ELECTRODE_SIZE_UM = CONTACT_WINDOW_DIAMETER_UM
    base.ELECTRODE_SHAPE = "circle"
    base.CURRENT_SPREAD_UM = CURRENT_SPREAD_UM
    base.PAPER_THRESHOLD_CURRENT_A = REFERENCE_THRESHOLD_CURRENT_A
    base.THRESHOLD_CURRENT_SOURCE = "paper"

    base.CWT_TRUNCATION_ORDER = 10
    base.VERTICAL_STEP_NM = 3.0
    base.FINITE_EIGEN_GRIDS = (13, 17, 21, 25)
    base.LENGTH_SWEEP_GRID_POINTS = (11, 15, 19)
    base.LENGTH_SWEEP_UM = (100.0, 200.0, 300.0, 400.0)

    base.MAXIMUM_GAIN_CM = MAXIMUM_GAIN_CM
    base.ZERO_CARRIER_GAIN_CM = ZERO_CARRIER_GAIN_CM
    base.TRANSPARENCY_DENSITY_CM3 = TRANSPARENCY_DENSITY_CM3
    base.CARRIER_LIFETIME_NS = CARRIER_LIFETIME_NS
    base.DIFFUSION_CM2_S = DIFFUSION_CM2_S
    base.ACTIVE_THICKNESS_NM = ACTIVE_THICKNESS_NM
    base.SPONTANEOUS_EMISSION_FACTOR = SPONTANEOUS_EMISSION_FACTOR
    base.DN_DN_CM3 = DN_DN_CM3
    base.TIME_GRID_POINTS = 31
    base.END_TIME_NS = 10.0
    base.SAMPLE_INTERVAL_PS = 0.2
    base.CURRENT_RATIOS = CURRENT_RATIOS
    base.SPECTRUM_WINDOW_NS = 2.0
    base.STEPS = STEPS
    base.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY


def selected_output_directory() -> Path:
    """Read the shared workflow's optional --output argument without consuming it."""
    if "--output" in sys.argv:
        index = sys.argv.index("--output")
        if index + 1 >= len(sys.argv):
            raise ValueError("--output requires a directory")
        return Path(sys.argv[index + 1]).resolve()
    return OUTPUT_DIRECTORY.resolve()


def write_wang_comparison(output: Path) -> None:
    """Compare only like-for-like quantities disclosed in the main paper."""
    mode_path = output / "mode_summary.csv"
    parameter_path = output / "parameters.json"
    coupling_path = output / "01_geometry_coupling.json"
    audit_path = output / "05_threshold_current_audit.json"
    if not (
        mode_path.exists() and parameter_path.exists()
        and coupling_path.exists() and audit_path.exists()
    ):
        return

    with mode_path.open(encoding="utf-8-sig") as stream:
        modes = list(csv.DictReader(stream))
    lowest = min(modes, key=lambda row: float(row["alpha_per_m"]))
    parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
    coupling = json.loads(coupling_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    def complex_entry(matrix_name: str, row: int, column: int) -> complex:
        real, imag = coupling[matrix_name][row][column]
        return complex(real, imag)

    c180_values_cm = (
        abs(complex_entry("C_total", 0, 1)) / 100.0,
        abs(complex_entry("C_total", 2, 3)) / 100.0,
    )
    threshold_area_cm2 = 3.141592653589793 * (0.5*CONTACT_WINDOW_DIAMETER_UM*1e-4)**2
    report = {
        "interpretation": (
            "Main-text geometry test. Exact epitaxy and InAlGaAs carrier parameters "
            "remain unavailable without Supplementary Table S1 and independent gain data."
        ),
        "published": {
            "lattice_constant_nm": LATTICE_CONSTANT_NM,
            "hole_diameter_nm": HOLE_DIAMETER_NM,
            "fill_fraction_percent": 8.5,
            "wavelength_nm_approximately": REFERENCE_WAVELENGTH_NM,
            "pc_confinement_percent": 100.0*REFERENCE_PC_CONFINEMENT,
            "mqw_confinement_percent": 100.0*REFERENCE_MQW_CONFINEMENT,
            "lowest_mode_loss_cm^-1_approximately": REFERENCE_LOWEST_MODE_LOSS_CM,
            "maximum_180_degree_coupling_cm^-1_approximately": 253.0,
            "threshold_current_A": REFERENCE_THRESHOLD_CURRENT_A,
            "threshold_current_density_kA_cm^-2": 1.66,
            "cw_output_power_mW_at_1A": 2.1,
            "cw_slope_efficiency_W_A_approximately": 0.02,
            "experimental_far_field_1e2_width_deg_xy": REFERENCE_FAR_FIELD_DEG,
        },
        "computed": {
            "fill_fraction_percent": 100.0*parameters["derived_fill_fraction"],
            "bragg_wavelength_nm": parameters["extra"]["bragg_wavelength_nm"],
            "effective_index": parameters["extra"]["derived_effective_index"],
            "pc_confinement_percent": 100.0*coupling["pc_confinement"],
            "solved_mqw_confinement_percent": (
                100.0*parameters["extra"]["solved_vertical_TE0_active_confinement"]
            ),
            "used_mqw_confinement_percent": (
                100.0*parameters["extra"]["used_active_confinement"]
            ),
            "triple_lattice_A20_structure_factor": 3.0,
            "C_total_180_degree_magnitudes_cm^-1": c180_values_cm,
            "lowest_mode": lowest["mode"],
            "lowest_mode_amplitude_loss_cm^-1": float(lowest["alpha_per_m"])/100.0,
            "twice_amplitude_loss_cm^-1": 2.0*float(lowest["alpha_per_m"])/100.0,
            "internal_plus_twice_amplitude_loss_cm^-1": (
                base.INTERNAL_LOSS_CM+2.0*float(lowest["alpha_per_m"])/100.0
            ),
            "placeholder_carrier_model_derived_threshold_A": (
                audit["derived_uniform_injection_threshold_A"]
            ),
            "threshold_current_density_kA_cm^-2_from_200um_circle": (
                REFERENCE_THRESHOLD_CURRENT_A/threshold_area_cm2/1e3
            ),
        },
        "comparison_limits": {
            "mode_loss_convention": (
                "The code records envelope-amplitude loss alpha and also reports 2*alpha; "
                "confirm the paper/implementation convention before fitting 28 cm^-1."
            ),
            "far_field": (
                "The paper's 3 x 17 degree pattern is experimental, multimode and affected "
                "by nonuniform near field and cleaved-facet feedback. The current ideal open-"
                "boundary single-mode model should not be fitted to that ellipse."
            ),
            "time_domain": (
                "Disabled by default because the main paper does not disclose a complete "
                "gain, recombination and electrical spreading model. The derived threshold "
                "from placeholder carrier parameters is diagnostic, not a prediction."
            ),
        },
    }
    (output / "00_wang2024_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Wang comparison written to 00_wang2024_comparison.json")


def main() -> None:
    configure_base_workflow()
    output = selected_output_directory()
    base.main()
    write_wang_comparison(output)


if __name__ == "__main__":
    main()
