"""Geometry invariants for the independently scanned three-hole motif."""
from dataclasses import replace
import numpy as np
import pytest
from pcselsim.fabrication import TRIANGULAR_DIRECT
from scripts.optimization.matched300_triangular_search import ThreeHole, candidate, base_spec, tune_period
from scripts.optimization.triangular_semiconductor_search import parameters


@pytest.mark.parametrize("shapes",[("circle","circle","circle"),("ellipse","triangle","ellipse"),("triangle","ellipse","triangle")])
def test_three_exact_nonzero_hole_areas(shapes):
    s=ThreeHole("test",fill=.137,main_share=.57,satellite_balance=.37,
        main_shape=shapes[0],satellite1_shape=shapes[1],satellite2_shape=shapes[2],
        main_aspect=1.31,satellite1_aspect=1.21,satellite2_aspect=1.43)
    holes=candidate(s).inclusions
    assert len(holes)==3
    shares=np.array((.57,.43*.37,.43*.63))
    np.testing.assert_allclose([h.fill_fraction for h in holes],s.fill*shares,atol=1e-14)


def test_satellite_positions_are_independent_cartesian_directions():
    s=ThreeHole("test",distance1=.33,distance2=.46,angle1_deg=-24,angle2_deg=37)
    holes=candidate(s).inclusions
    for h,d,a in zip(holes[1:],(.33,.46),(-24,37),strict=True):
        np.testing.assert_allclose(TRIANGULAR_DIRECT@np.asarray(h.center_fractional),
            d*np.array((np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a)))),atol=1e-14)
    np.testing.assert_array_equal(holes[0].center_fractional,(0,0))


def test_inoue_stack_and_aperture_are_fixed():
    p=parameters(base_spec(ThreeHole("test")))
    assert p.DEVICE_SHAPE=="square" and p.DEVICE_HALF_SIZE_UM==150
    assert p.INTERNAL_LOSS_CM==5
    assert next(l for l in p.LAYERS if l.name==p.PC_LAYER_NAME).thickness_nm==190
    assert sum(l.thickness_nm for l in p.LAYERS if l.name==p.ACTIVE_LAYER_NAME)==30


def test_period_tuning_uses_target_vertical_mode_and_not_hole_orientation():
    first=tune_period(ThreeHole("one",fill=.14),dz=4)
    second=tune_period(ThreeHole("two",fill=.14,main_orientation_deg=37,angle1_deg=-21),dz=4)
    assert 310<first.lattice_nm<340
    assert first.lattice_nm==pytest.approx(second.lattice_nm,abs=1e-10)


def test_cannot_remove_satellite_through_zero_area():
    with pytest.raises(ValueError): candidate(ThreeHole("invalid",satellite_balance=0))


def test_rounded_contour_resolution_is_explicit_and_area_preserving():
    original=ThreeHole("rounded",fill=.138150834674,main_share=.419191203,main_shape="triangle",rounding=.394332883)
    coarse=candidate(original).inclusions[0]
    fine=candidate(replace(original,triangle_samples_per_corner=27)).inclusions[0]
    assert len(fine.vertices_fractional)>len(coarse.vertices_fractional)
    assert coarse.fill_fraction==pytest.approx(original.fill*original.main_share,abs=1e-14)
    assert fine.fill_fraction==pytest.approx(coarse.fill_fraction,abs=1e-14)
    assert original.triangle_samples_per_corner==9
