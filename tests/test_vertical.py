import numpy as np
import pytest
from scipy.optimize import brentq

from pcselsim.vertical import Layer, LayerStack, inoue2019_stack, sample_vertical_interval


def test_symmetric_slab_te0_is_guided() -> None:
    stack = LayerStack(
        layers=(Layer("core", 500.0, 3.5),),
        top_index=3.0,
        bottom_index=3.0,
        padding_um=1.5,
    )
    mode = stack.solve_te0(wavelength_nm=1000.0, dz_nm=10.0)
    assert 3.0 < mode.effective_index < 3.5
    assert 0.0 < mode.confinement["core"] < 1.0


@pytest.mark.parametrize("start_nm", [21.1, 22.4, 25.9])
def test_subgrid_layer_integral_keeps_both_interfaces(start_nm):
    axis = np.arange(0.0, 61.0, 3.0)*1e-9
    layer_z, density = sample_vertical_interval(
        axis, np.full(axis.shape, 2e6), start_nm*1e-9, (start_nm+10.0)*1e-9
    )
    assert layer_z[0] == pytest.approx(start_nm*1e-9)
    assert layer_z[-1] == pytest.approx((start_nm+10.0)*1e-9)
    assert np.trapz(density, layer_z) == pytest.approx(0.02, rel=1e-13)


@pytest.mark.parametrize("dz_nm", [3.0, 4.7, 7.0])
def test_thin_layer_partition_preserves_total_waveguide_overlap(dz_nm):
    # Splitting an otherwise identical uniform core must not delete the
    # intervals between the last mesh node of one layer and the next layer.
    full = LayerStack((Layer("core", 500.0, 3.5),), 3.0, 3.0, 1.0).solve_te0(1000.0, dz_nm)
    partitioned = LayerStack(
        (Layer("lower", 241.3, 3.5), Layer("QW", 1.1, 3.5), Layer("upper", 257.6, 3.5)),
        3.0, 3.0, 1.0,
    ).solve_te0(1000.0, dz_nm)
    assert partitioned.effective_index == pytest.approx(full.effective_index, rel=1e-12)
    assert sum(partitioned.confinement.values()) == pytest.approx(full.confinement["core"], rel=1e-10)
    assert partitioned.confinement["QW"] > 0.0


def test_te0_mesh_refinement_matches_analytic_symmetric_slab():
    core_n, clad_n, thickness_m, wavelength_m = 3.5, 3.0, 500e-9, 1000e-9
    k0 = 2*np.pi/wavelength_m
    v = k0*thickness_m/2*np.sqrt(core_n**2-clad_n**2)
    u = brentq(lambda x: x*np.tan(x)-np.sqrt(v*v-x*x), 1e-10, min(v, np.pi/2)-1e-10)
    analytic_neff = np.sqrt(core_n**2-(2*u/(thickness_m*k0))**2)
    stack = LayerStack((Layer("core", 500.0, core_n),), clad_n, clad_n, 1.5)
    errors = [abs(stack.solve_te0(1000.0, dz).effective_index-analytic_neff) for dz in (10.0, 5.0, 2.5)]
    assert errors[1] < 0.3*errors[0]
    assert errors[2] < 0.3*errors[1]


def test_inoue_table_i_contains_three_well_barrier_periods():
    layers = inoue2019_stack().layers
    assert sum(layer.thickness_nm for layer in layers if layer.name == "active-InGaAs") == 30.0
    assert sum(layer.thickness_nm for layer in layers if layer.name.startswith("active-AlGaAs")) == 60.0


def test_all_current_semiconductor_presets_contain_full_inoue_active_stack():
    from scripts import run_paper_reproduction, run_square_double_hole, run_triangular_three_hole
    for entry in (run_paper_reproduction, run_square_double_hole, run_triangular_three_hole):
        layers = entry.LAYERS
        assert sum(layer.thickness_nm for layer in layers if "InGaAs QWs" in layer.name) == 30.0
        assert sum(layer.thickness_nm for layer in layers if "AlGaAs barrier" in layer.name) == 60.0

