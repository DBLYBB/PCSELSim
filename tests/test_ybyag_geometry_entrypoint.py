import numpy as np

from scripts.archive.legacy_20261008 import run_geometry_derived_ybyag_pcsel as ybyag


def test_ybyag_geometry_preset_targets_1030_nm_and_has_explicit_waveguide() -> None:
    assert np.isclose(ybyag.LATTICE_CONSTANT_NM, 566.0)
    assert ybyag.PC_DEPTH_NM > 0.0
    assert ybyag.GAIN_FILM_THICKNESS_NM > ybyag.PC_DEPTH_NM
    assert ybyag.YBYAG_CORE_INDEX > ybyag.YAG_CLADDING_INDEX
    assert ybyag.FINITE_GRIDS == tuple(sorted(ybyag.FINITE_GRIDS))
    assert ybyag.SIZE_SWEEP_UM[0] == ybyag.DEVICE_SIZE_UM
    assert ybyag.SIZE_SWEEP_UM[-1] >= 10_000.0


def test_ybyag_default_single_pass_absorption_is_small() -> None:
    alpha_cm = (
        ybyag.YB_DOPANT_DENSITY_CM3
        * ybyag.PUMP_ABSORPTION_CROSS_SECTION_CM2
    )
    thickness_cm = ybyag.GAIN_FILM_THICKNESS_NM * 1e-7
    absorbed = 1.0 - np.exp(-alpha_cm * thickness_cm)
    assert 0.0 < absorbed < 1e-3


def test_ybyag_patterned_layer_is_at_air_side() -> None:
    # vertical.LayerStack fills from bottom toward top: PC is the last layer.
    assert ybyag.LAYERS[-1].name == ybyag.PC_LAYER_NAME
    assert ybyag.LAYERS[0].name == "undoped YAG buffer"
