"""Geometry/protocol checks for the new center-and-satellite search."""
from dataclasses import replace,FrozenInstanceError
import numpy as np
import pytest
from scripts.optimization.matched300_square_search import Design, build_cell, hole, CachedSquareCell
from pcselsim.geometry import SquareLatticeCell


@pytest.mark.parametrize("kind",("circle","ellipse","superellipse","rounded_triangle"))
def test_area_and_real_dielectric_reciprocity(kind):
    h=hole((0.,0.),.075,kind,1.3,37.,3.2,64)
    cell=SquareLatticeCell(3.554**2,(h,))
    assert cell.fourier_epsilon(0,0).real == pytest.approx(3.554**2+(1-3.554**2)*.075,abs=1e-12)
    for m,n in ((1,0),(2,-1),(-3,4)):
        assert cell.fourier_epsilon(m,n) == pytest.approx(cell.fourier_epsilon(-m,-n).conjugate(),abs=1e-12)


def test_main_is_central_and_area_share_is_physical():
    d=Design(fill=.12,main_share=.7,distance=.5,position_angle_deg=40.)
    cell=build_cell(d)
    assert cell.inclusions[0].center==(0.,0.)
    assert np.linalg.norm(cell.inclusions[1].center)==pytest.approx(.5)
    assert np.pi*np.prod(cell.inclusions[0].radii)==pytest.approx(.084)
    assert np.pi*np.prod(cell.inclusions[1].radii)==pytest.approx(.036)


def test_translation_changes_phase_but_not_fourier_magnitude():
    d=Design(distance=.55,position_angle_deg=49.)
    cell=build_cell(d);shift=np.array([.193,-.219])
    moved=SquareLatticeCell(cell.background_epsilon,tuple(replace(h,center=tuple(np.array(h.center)+shift)) for h in cell.inclusions))
    for m,n in ((1,0),(1,1),(2,-3)):
        expected=cell.fourier_epsilon(m,n)*np.exp(2j*np.pi*np.array([m,n])@shift)
        assert moved.fourier_epsilon(m,n)==pytest.approx(expected,abs=1e-12)


def test_matched_protocol_prevents_silent_aperture_or_stack_changes():
    with pytest.raises(ValueError):Design(length_um=200.)
    with pytest.raises(ValueError):Design(pc_nm=120.)
    with pytest.raises(ValueError):Design(main_share=1.)


@pytest.mark.parametrize("vertices",(48,64,96))
def test_rounded_triangle_physical_area_centroid_is_the_requested_position(vertices):
    center=np.array([.123,-.217])
    polygon=hole(center,.065,"rounded_triangle",1.38,51.,vertices=vertices)
    p=np.asarray(polygon.vertices);q=np.roll(p,-1,axis=0)
    cross=p[:,0]*q[:,1]-p[:,1]*q[:,0]
    centroid=np.sum((p+q)*cross[:,None],axis=0)/(3*cross.sum())
    np.testing.assert_allclose(centroid,center,rtol=0,atol=2e-16)


def test_exact_geometry_fourier_cache_reuses_only_identical_frozen_geometry():
    CachedSquareCell.fourier_epsilon.cache_clear()
    d=Design(main_shape="rounded_triangle",satellite_shape="ellipse")
    cell=build_cell(d,64);value=cell.fourier_epsilon(2,-3)
    baseline=SquareLatticeCell(cell.background_epsilon,cell.inclusions)
    assert value==baseline.fourier_epsilon(2,-3)
    before=CachedSquareCell.fourier_epsilon.cache_info()
    other_period=build_cell(replace(d,lattice_nm=286.),64)
    assert other_period.fourier_epsilon(2,-3)==value
    assert CachedSquareCell.fourier_epsilon.cache_info().hits==before.hits+1
    other_geometry=build_cell(replace(d,main_angle_deg=3.),64)
    assert other_geometry.fourier_epsilon(2,-3)!=value
    other_material=build_cell(d,64,background_index=1.82)
    assert other_material.fourier_epsilon(2,-3)!=value
    with pytest.raises(FrozenInstanceError):cell.inclusions=()
    with pytest.raises(FrozenInstanceError):cell.extra_state=1


def test_mesh_seed_interpolation_preserves_complex_component_phase_ratios():
    from scripts.optimization.matched300_square_mesh import interpolate_field
    axis=(np.arange(17)+.5)/17
    yy,xx=np.meshgrid(axis,axis,indexing="ij")
    envelope=np.exp(-((xx-.5)**2+(yy-.5)**2)*8)*np.exp(1j*(.2*xx-.3*yy))
    ratios=np.array([1.,.4*np.exp(.3j),-.2j,.7*np.exp(-.7j)])
    field=ratios[:,None,None]*envelope
    result=interpolate_field(field,25)
    assert np.linalg.norm(result)==pytest.approx(1.,abs=1e-14)
    for i in range(4):np.testing.assert_allclose(result[i],result[0]*ratios[i],atol=1e-15)
