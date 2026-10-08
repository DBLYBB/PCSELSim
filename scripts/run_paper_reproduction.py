"""Paper-reference semiconductor entry / 论文对照主程序。

Default: geometry-derived idealized Inoue motif, not certified full reproduction.
改下方参数和开关，右键 Run；先用 --dry-run 看计划，--quick 只做流程检查。
"""
from __future__ import annotations

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project
PROJECT_ROOT = bootstrap_project()

from pcselsim.custom_analysis import LatticeInclusionSpec, LayerSpec
from scripts._four_wave_workflow import FourWaveSettings, run_four_wave
from scripts._run_controls import execute_run
from pcselsim.fabrication import FabricationRules

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
FAR_FIELD_ZOOM_DEG = 0.2          # display-only crop / 只放大显示，不改变远场统计


# ===========================================================================
# STEP SWITCHES / 步骤开关：False 只跳过输出，不会更改其他步骤的物理参数。
# ===========================================================================
STEPS = {
    "00_fabrication": True,   # assumed process rules, not foundry approval
    "01_parameters": True,    # save inputs and C1D/Crad/C2D / 保存参数与耦合分量
    "02_lattice": True,       # real-space unit cell / 实空间晶胞
    "02_device_overview": True, # device boundary + array + physical scale
    "03_k_space": True,       # retained Rx,Sx,Ry,Sy waves / 四个基本 Bloch 波
    "04_layer_stack": True,   # scalar TE0 and overlap / 纵向 TE0 与限制因子
    "05_band_structure": True, # M-Gamma-X normalized-frequency bands / M-Gamma-X 能带
    "05_linear_modes": True,  # Liang Eqs. 4.21-4.23 / 有限区域冷腔模与阈值
    "06_mode_atlas": True,    # device, unit cell, vector FFP / 包络、晶胞场与矢量远场
    "06_far_field_zoom": True, # display-only selected-family close-up
    "07_length_sweep": True,  # finite-size scaling / 器件尺寸扫描
    "08_time_domain": True,   # Inoue Eqs. 8-11; slowest / 最慢的非线性时域步骤
}

FABRICATION_RULES = (FabricationRules("research_assumption", 40, 50, 3, 3),)



def build_parameters() -> FourWaveSettings:
    return FourWaveSettings(
        FABRICATION_RULES=FABRICATION_RULES,
        FAR_FIELD_ZOOM_DEG=FAR_FIELD_ZOOM_DEG,
        ACTIVE_CONFINEMENT_FACTOR=ACTIVE_CONFINEMENT_FACTOR,
        ACTIVE_CONFINEMENT_SOURCE=ACTIVE_CONFINEMENT_SOURCE,
        ACTIVE_INDEX=ACTIVE_INDEX,
        ACTIVE_LAYER_NAME=ACTIVE_LAYER_NAME,
        ACTIVE_THICKNESS_NM=ACTIVE_THICKNESS_NM,
        BACKGROUND_INDEX=BACKGROUND_INDEX,
        BOTTOM_CLADDING_INDEX=BOTTOM_CLADDING_INDEX,
        CARRIER_LIFETIME_NS=CARRIER_LIFETIME_NS,
        CFL=CFL,
        CURRENT_RATIOS=CURRENT_RATIOS,
        CURRENT_SPREAD_UM=CURRENT_SPREAD_UM,
        CWT_TRUNCATION_ORDER=CWT_TRUNCATION_ORDER,
        DEVICE_PRESET=DEVICE_PRESET,
        DEVICE_SIZE_UM=DEVICE_SIZE_UM,
        DIFFUSION_CM2_S=DIFFUSION_CM2_S,
        DN_DN_CM3=DN_DN_CM3,
        ELECTRODE_SHAPE=ELECTRODE_SHAPE,
        ELECTRODE_SIZE_UM=ELECTRODE_SIZE_UM,
        ENABLE_NOISE=ENABLE_NOISE,
        END_TIME_NS=END_TIME_NS,
        FINITE_EIGEN_GRIDS=FINITE_EIGEN_GRIDS,
        GEOMETRY_FIDELITY=GEOMETRY_FIDELITY,
        HOLE_INDEX=HOLE_INDEX,
        HOLE_RADIUS_X_NM=HOLE_RADIUS_X_NM,
        HOLE_RADIUS_Y_NM=HOLE_RADIUS_Y_NM,
        HOLE_ROTATION_DEG=HOLE_ROTATION_DEG,
        HOLE_SHAPE=HOLE_SHAPE,
        INTERNAL_LOSS_CM=INTERNAL_LOSS_CM,
        LATTICE_CONSTANT_NM=LATTICE_CONSTANT_NM,
        LATTICE_TYPE=LATTICE_TYPE,
        LAYERS=LAYERS,
        LENGTH_SWEEP_GRID_POINTS=LENGTH_SWEEP_GRID_POINTS,
        LENGTH_SWEEP_UM=LENGTH_SWEEP_UM,
        MAXIMUM_GAIN_CM=MAXIMUM_GAIN_CM,
        PAPER_THRESHOLD_CURRENT_A=PAPER_THRESHOLD_CURRENT_A,
        PC_LAYER_NAME=PC_LAYER_NAME,
        RANDOM_SEED=RANDOM_SEED,
        RUN_TITLE=RUN_TITLE,
        SAMPLE_INTERVAL_PS=SAMPLE_INTERVAL_PS,
        SPECTRUM_WINDOW_NS=SPECTRUM_WINDOW_NS,
        SPONTANEOUS_EMISSION_FACTOR=SPONTANEOUS_EMISSION_FACTOR,
        THRESHOLD_CURRENT_SOURCE=THRESHOLD_CURRENT_SOURCE,
        TIME_DOMAIN_SIZE_UM=TIME_DOMAIN_SIZE_UM,
        TIME_GRID_POINTS=TIME_GRID_POINTS,
        TOP_CLADDING_INDEX=TOP_CLADDING_INDEX,
        TRANSPARENCY_DENSITY_CM3=TRANSPARENCY_DENSITY_CM3,
        UNIT_CELL_INCLUSIONS=UNIT_CELL_INCLUSIONS,
        VERTICAL_PADDING_UM=VERTICAL_PADDING_UM,
        VERTICAL_STEP_NM=VERTICAL_STEP_NM,
        WAVELENGTH_GUESS_NM=WAVELENGTH_GUESS_NM,
        ZERO_CARRIER_GAIN_CM=ZERO_CARRIER_GAIN_CM,
    )


def main(argv=None) -> None:
    execute_run(build_parameters(), STEPS, "paper_reference", run_four_wave, argv)


if __name__ == "__main__":
    main()

