"""Triangular three-hole entry / 三角晶格三孔主程序。
Rounded triangle main hole + two circle satellites; research-process candidate.
不存在“已证明全局最优/单模”结论；六波时域未实现，小信号泵扫描不是 L-I 曲线。
"""
from __future__ import annotations

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project
PROJECT_ROOT = bootstrap_project()

import numpy as np
from pcselsim.custom_analysis import LayerSpec
from scripts._triangular_workflow import Design, TriangularSettings, run_triangular
from scripts._run_controls import execute_run
from pcselsim.fabrication import FabricationRules

# A: lattice + all motif parameters / 参数都是初筛工程设计而非已测加工膜
RUN_TITLE = "TRIANGULAR PCSEL - rounded triangle main + two circles"
LATTICE_CONSTANT_NM = 2.0*277.0/np.sqrt(3.0)
WAVELENGTH_GUESS_NM = 950.65
BACKGROUND_INDEX = 3.554
HOLE_EPSILON = 1.0
DESIGN = Design(
    name="triangle_circle_f0.15_m0.40_d0.50_s0",
    main_shape="triangle",       # triangle/circle/ellipse
    satellite_shape="circle",   # circle/ellipse/triangle
    fill=0.15,                  # all holes' total area / primitive-cell area
    main_area_share=0.40,       # each satellite gets (1-main_area_share)/2
    distance=0.50,              # radial Cartesian distance / a
    spread_deg=0.0,             # satellites sit at -30-spread, +30+spread
    rounding=0.30,              # triangle corner rounding fraction
    main_angle_deg=0.0,
    ellipse_aspect=1.30,        # major/minor radius if ellipse is selected
)

# B: actual vertical layers (bottom -> top), common semiconductor rates
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


ACTIVE_INDEX = 3.584
ACTIVE_LAYER_NAME = "active InGaAs QWs"
ACTIVE_THICKNESS_NM = 30.0
INTERNAL_LOSS_CM = 5.0
MAXIMUM_GAIN_CM = 2000.0
ZERO_CARRIER_GAIN_CM = -5000.0
TRANSPARENCY_DENSITY_CM3 = 1.5e18
CARRIER_LIFETIME_NS = 1.5
DN_DN_CM3 = -5.0e-21

# C: device / numerical convergence; grids are accuracy controls, not fitted physics.
DEVICE_SHAPE = "square"        # square/circle/hexagon
DEVICE_HALF_SIZE_UM = 150.0
FINITE_GRIDS = (11, 15, 19)
CWT_TRUNCATION_ORDER = 10
VERTICAL_STEP_NM = 3.0
BAND_Q_MAX_2PI_OVER_A = 0.01    # LOCAL near-Gamma segment toward M and X(K)
BAND_POINTS = 241
UNIT_CELL_FIELD_POINTS = 121
FAR_FIELD_VIEW_DEG = 1.0       # fixed angular observation window for comparisons
FAR_FIELD_ZOOM_DEG = 0.2       # display crop only; original RMS/energy unchanged
FAR_FIELD_SAMPLES = 161
PUMP_RATIOS = tuple(np.linspace(0.5, 1.5, 41))

# D: step switches / 求解依赖仍会执行，关闭的是指定输出。
STEPS = {
    "00_fabrication": True,
    "01_parameters": True,
    "02_lattice": True,
    "02_device_overview": True,
    "03_k_space": True,
    "04_vertical_mode": True,
    "05_band_structure": True,
    "06_radiation_constants": True,
    "07_band_edge_states": True,
    "08_finite_modes": True,
    "09_mode_atlas": True,
    "10_best_far_field": True,
    "10_far_field_zoom": True,
    "11_threshold_audit": True,
    "12_small_signal_pump": True,
}

FABRICATION_RULES = (
    FabricationRules("fine_ebeam_assumption", 40, 50, 3, 3, 5),
    FabricationRules("research_ebeam_assumption", 60, 65, 3, 3, 10),
)

def build_parameters() -> TriangularSettings:
    return TriangularSettings(
        FABRICATION_RULES=FABRICATION_RULES,
        ACTIVE_INDEX=ACTIVE_INDEX,
        ACTIVE_LAYER_NAME=ACTIVE_LAYER_NAME,
        ACTIVE_THICKNESS_NM=ACTIVE_THICKNESS_NM,
        BACKGROUND_INDEX=BACKGROUND_INDEX,
        BAND_POINTS=BAND_POINTS,
        BAND_Q_MAX_2PI_OVER_A=BAND_Q_MAX_2PI_OVER_A,
        BOTTOM_CLADDING_INDEX=BOTTOM_CLADDING_INDEX,
        CARRIER_LIFETIME_NS=CARRIER_LIFETIME_NS,
        CWT_TRUNCATION_ORDER=CWT_TRUNCATION_ORDER,
        DESIGN=DESIGN,
        DEVICE_HALF_SIZE_UM=DEVICE_HALF_SIZE_UM,
        DEVICE_SHAPE=DEVICE_SHAPE,
        DN_DN_CM3=DN_DN_CM3,
        FAR_FIELD_SAMPLES=FAR_FIELD_SAMPLES,
        FAR_FIELD_VIEW_DEG=FAR_FIELD_VIEW_DEG,
        FAR_FIELD_ZOOM_DEG=FAR_FIELD_ZOOM_DEG,
        FINITE_GRIDS=FINITE_GRIDS,
        HOLE_EPSILON=HOLE_EPSILON,
        INTERNAL_LOSS_CM=INTERNAL_LOSS_CM,
        LATTICE_CONSTANT_NM=LATTICE_CONSTANT_NM,
        LAYERS=LAYERS,
        MAXIMUM_GAIN_CM=MAXIMUM_GAIN_CM,
        PC_LAYER_NAME=PC_LAYER_NAME,
        PUMP_RATIOS=PUMP_RATIOS,
        RUN_TITLE=RUN_TITLE,
        TOP_CLADDING_INDEX=TOP_CLADDING_INDEX,
        TRANSPARENCY_DENSITY_CM3=TRANSPARENCY_DENSITY_CM3,
        UNIT_CELL_FIELD_POINTS=UNIT_CELL_FIELD_POINTS,
        VERTICAL_PADDING_UM=VERTICAL_PADDING_UM,
        VERTICAL_STEP_NM=VERTICAL_STEP_NM,
        WAVELENGTH_GUESS_NM=WAVELENGTH_GUESS_NM,
        ZERO_CARRIER_GAIN_CM=ZERO_CARRIER_GAIN_CM,
    )


def main(argv=None) -> None:
    execute_run(build_parameters(), STEPS, "triangular_three_hole", run_triangular, argv)


if __name__ == "__main__":
    main()

