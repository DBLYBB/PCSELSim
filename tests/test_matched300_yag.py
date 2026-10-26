"""Independent Green-function limits and flux accounting for matched YAG."""
import numpy as np
import pytest
from dataclasses import replace
from pcselsim.hybrid_vertical import HybridLayerStack
from pcselsim.vertical import Layer
from pcselsim.stratified_optics import StratifiedStack,StratifiedLayer,quarter_wave_dbr
from pcselsim.triangular_six_wave import (TriangularLatticeCell,TriangularEllipse,
    TriangularCWTSettings,build_triangular_coupling,TRIANGULAR_BASIC_ORDERS)
from pcselsim.matched300_yag import (build_triangular_reflector_coupling,
    binned_lateral_weights)
from pcselsim.polarization import polarization_metrics


def geometry():
    cell=TriangularLatticeCell(1.82**2,(TriangularEllipse((.13,-.07),(.10,.075),21.,1.),))
    n=np.sqrt(cell.fourier_epsilon(0,0).real)
    guide=HybridLayerStack((Layer("gain",500.,1.82),Layer("PC",120.,n)),
        top_index=1.45,bottom_index=1.45,padding_um=2.)
    return cell,guide,TriangularCWTSettings(2,10.),n


def test_uniform_triangular_reflector_recovers_six_wave_host_limit():
    cell,guide,settings,n=geometry()
    radiation=StratifiedStack(tuple(StratifiedLayer(l.name,l.thickness_nm,n)
                                    for l in guide.layers),n,n)
    free=build_triangular_coupling(cell,guide,"PC",690.,1030.,settings,fixed_wavelength_nm=1030.)
    layered=build_triangular_reflector_coupling(cell,guide,radiation,"PC",690.,1030.,settings,fixed_wavelength_nm=1030.)
    for key in ("cb_m","cr_m","ch_m","coupling_m"):
        np.testing.assert_allclose(getattr(free,key),getattr(layered.result,key),rtol=3e-10,atol=2e-8)
    assert layered.result.radiation_surface_factor==pytest.approx(free.radiation_surface_factor,rel=3e-10)
    for attr in ("unit_cell_response_x","unit_cell_response_y"):
        for order,value in getattr(free,attr).items():
            np.testing.assert_allclose(value,getattr(layered.result,attr)[order],rtol=3e-10,atol=2e-10)
    np.testing.assert_allclose(layered.upward_amplitude_loss_m,layered.downward_amplitude_loss_m,rtol=2e-11,atol=1e-10)


def test_triangular_dbr_flux_optical_theorem_and_six_wave_shape():
    cell,guide,settings,n=geometry()
    radiation=StratifiedStack((*quarter_wave_dbr(1030.,2.1,1.45,8,first_layer="low"),
        StratifiedLayer("spacer",2000.,1.45),
        *(StratifiedLayer(l.name,l.thickness_nm,l.refractive_index) for l in guide.layers)),1.8154,1.45)
    out=build_triangular_reflector_coupling(cell,guide,radiation,"PC",690.,1030.,settings,fixed_wavelength_nm=1030.)
    assert out.result.coupling_m.shape==(6,6)
    damping=(out.result.coupling_m-out.result.coupling_m.conj().T)/(2j)
    np.testing.assert_allclose(damping,out.upward_amplitude_loss_m+out.downward_amplitude_loss_m,rtol=1e-10,atol=2e-8)
    assert out.optical_theorem_residual<1e-10
    assert np.linalg.eigvalsh(damping).min()>-1e-8
    assert np.trace(out.downward_amplitude_loss_m).real>0
    fields=np.random.default_rng(863).normal(size=(6,13))+1j*np.random.default_rng(983).normal(size=(6,13))
    up,down=out.side_power_losses(fields)
    expected=2*np.einsum("ik,ij,jk->",fields.conj(),damping,fields).real/np.sum(abs(fields)**2)
    assert up+down==pytest.approx(expected,rel=1e-10)


def test_triangular_projection_preserves_mode_energy_and_asymmetry():
    fields=np.zeros((6,5),complex);fields[0]=[1,2,3,4,5]
    weights=binned_lateral_weights(fields,[-150,-75,0,75,150],300,5)
    assert weights.sum()==pytest.approx(1.)
    assert weights[-1]>weights[0]
    # A flat field sampled on few columns must not create empty reservoir
    # bins when the physical pump grid is finer than the optical grid.
    fine=binned_lateral_weights(np.ones((6,5)),[-150,-75,0,75,150],300,31)
    assert fine.sum()==pytest.approx(1.)
    assert np.all(fine>0)


def test_polarization_convention_and_dark_center():
    fx=np.ones((5,5),complex);fy=1j*fx
    report=polarization_metrics(np.linspace(-1,1,5),fx,fy)
    np.testing.assert_allclose(report["integrated"]["normalized_stokes_s1_s2_s3"],[0,0,-1])
    assert report["integrated"]["degree_of_linear_polarization"]==pytest.approx(0)
    fx[2,2]=fy[2,2]=0
    assert polarization_metrics(np.linspace(-1,1,5),fx,fy)["on_axis"] is None


def test_triangular_same_stack_dbr_preserves_circular_cell_doublets():
    # A planar mirror cannot break in-plane C6. A rectangular truncation or
    # wrong polarization projector spuriously splits these two doublets.
    cell=TriangularLatticeCell(1.82**2,(TriangularEllipse((0.,0.),(.16,.16)),))
    n=np.sqrt(cell.fourier_epsilon(0,0).real)
    layers=(Layer("gain",500.,1.82),Layer("PC",120.,n))
    guide=HybridLayerStack(layers,1.,1.45,2.)
    radiation=StratifiedStack((*quarter_wave_dbr(1030.,2.1,1.45,8,first_layer="low"),
        StratifiedLayer("spacer",2000.,1.45),
        *(StratifiedLayer(l.name,l.thickness_nm,l.refractive_index) for l in layers)),1.8154,1.)
    out=build_triangular_reflector_coupling(cell,guide,radiation,"PC",690.,1030.,TriangularCWTSettings(3,10.),fixed_wavelength_nm=1030.)
    values=out.result.eigenvalues_m
    distances=np.sort(abs(values[:,None]-values[None,:]),axis=1)[:,1]
    assert np.sum(distances<1e-7)>=4


def test_global_hole_translation_is_unitary_gauge_and_preserves_outgoing_jones():
    cell,guide,settings,n=geometry()
    radiation=StratifiedStack((*quarter_wave_dbr(1030.,2.1,1.45,8,first_layer="low"),
        StratifiedLayer("spacer",2000.,1.45),
        *(StratifiedLayer(l.name,l.thickness_nm,l.refractive_index) for l in guide.layers)),1.8154,1.45)
    displacement=np.array([.173,-.219])
    shifted=replace(cell,inclusions=tuple(replace(h,
        center_fractional=tuple(np.array(h.center_fractional)+displacement)) for h in cell.inclusions))
    a=build_triangular_reflector_coupling(cell,guide,radiation,"PC",690.,1030.,settings,fixed_wavelength_nm=1030.)
    b=build_triangular_reflector_coupling(shifted,guide,radiation,"PC",690.,1030.,settings,fixed_wavelength_nm=1030.)
    phase=np.exp(2j*np.pi*(np.asarray(TRIANGULAR_BASIC_ORDERS)@displacement))
    transform=lambda matrix:phase[:,None]*matrix*phase.conj()[None,:]
    for key in ("coupling_m","cb_m","cr_m","ch_m"):
        np.testing.assert_allclose(getattr(b.result,key),transform(getattr(a.result,key)),rtol=4e-11,atol=2e-8)
    for key in ("upward_amplitude_loss_m","downward_amplitude_loss_m"):
        np.testing.assert_allclose(getattr(b,key),transform(getattr(a,key)),rtol=4e-11,atol=2e-8)
    fields=np.random.default_rng(427).normal(size=(6,9))+1j*np.random.default_rng(926).normal(size=(6,9))
    ax,ay=a.result.radiation_amplitudes(fields)
    bx,by=b.result.radiation_amplitudes(phase[:,None]*fields)
    np.testing.assert_allclose(bx,ax,rtol=4e-11,atol=2e-8)
    np.testing.assert_allclose(by,ay,rtol=4e-11,atol=2e-8)
    np.testing.assert_allclose(b.side_power_losses(phase[:,None]*fields),a.side_power_losses(fields),rtol=4e-11)


def test_triangular_reflector_rejects_absorption_and_open_omitted_channels():
    cell,guide,settings,n=geometry()
    layers=tuple(StratifiedLayer(l.name,l.thickness_nm,l.refractive_index) for l in guide.layers)
    absorbing=StratifiedStack((StratifiedLayer("absorber",200.,1.5-1e-4j),*layers),1.45,1.45)
    with pytest.raises(ValueError,match="Absorbing"):
        build_triangular_reflector_coupling(cell,guide,absorbing,"PC",690.,1030.,settings,fixed_wavelength_nm=1030.)
    open_high_order=StratifiedStack(layers,4.,1.45)
    with pytest.raises(ValueError,match="Open high-order"):
        build_triangular_reflector_coupling(cell,guide,open_high_order,"PC",690.,1030.,settings,fixed_wavelength_nm=1030.)


def test_yag_private_polygon_aspect_preserves_area_centroid_and_legacy_semantics():
    # The YAG-only extension must not reinterpret any old unflagged design.
    import importlib.util,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root),str(root/"scripts"/"optimization")]
    from scripts.optimization.matched300_yag_search import geometry,ThreeHole,asdict
    from pcselsim.fabrication import inclusion_boundary,TRIANGULAR_DIRECT
    d=asdict(ThreeHole("aspect_test",main_shape="triangle",main_aspect=1.45,
        satellite1_shape="triangle",satellite1_aspect=1.2,distance1=.46,distance2=.46))
    unflagged=geometry("triangular",d)
    false_flag=geometry("triangular",{**d,"_yag_triangle_aspect_enabled":False})
    stretched=geometry("triangular",{**d,"_yag_triangle_aspect_enabled":True})
    def measure(h):
        p=inclusion_boundary(h,TRIANGULAR_DIRECT);x,y=p.T
        cross=x*np.roll(y,-1)-np.roll(x,-1)*y
        centroid=np.array([np.sum((x+np.roll(x,-1))*cross),np.sum((y+np.roll(y,-1))*cross)])/(3*np.sum(cross))
        return abs(cross.sum())/2,centroid
    for a,b,c in zip(unflagged.inclusions,false_flag.inclusions,stretched.inclusions):
        np.testing.assert_array_equal(inclusion_boundary(a,TRIANGULAR_DIRECT),inclusion_boundary(b,TRIANGULAR_DIRECT))
        aa,ca=measure(a);ac,cc=measure(c)
        assert ac==pytest.approx(aa,rel=1e-12)
        np.testing.assert_allclose(cc,ca,atol=2e-15)
    assert not np.allclose(inclusion_boundary(unflagged.inclusions[0],TRIANGULAR_DIRECT),
                           inclusion_boundary(stretched.inclusions[0],TRIANGULAR_DIRECT))
    again=geometry("triangular",{**d,"_yag_triangle_aspect_enabled":True})
    np.testing.assert_array_equal(inclusion_boundary(again.inclusions[0],TRIANGULAR_DIRECT),
                                 inclusion_boundary(stretched.inclusions[0],TRIANGULAR_DIRECT))


def test_explicit_yag_geometry_json_roundtrip_preserves_complex_fourier():
    import json,sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root),str(root/"scripts"/"optimization")]
    from scripts.optimization.matched300_yag_search import geometry,ThreeHole,asdict
    d=asdict(ThreeHole("json_test",main_shape="triangle",satellite1_shape="ellipse",distance1=.46,distance2=.46))
    cell=geometry("triangular",d)
    explicit=[dict(type="triangular_polygon" if hasattr(h,"vertices_fractional") else "triangular_ellipse",parameters=asdict(h)) for h in cell.inclusions]
    clone=geometry("triangular",json.loads(json.dumps({**d,"_yag_explicit_holes":explicit})))
    for m,n in ((0,0),(1,0),(0,-1),(3,5),(-7,4)):
        assert clone.fourier_epsilon(m,n)==cell.fourier_epsilon(m,n)


def test_yag_contour_sampling_is_explicit_and_preserves_area_and_centers():
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root),str(root/"scripts"/"optimization")]
    from scripts.optimization.matched300_yag_search import geometry,ThreeHole,asdict
    d=asdict(ThreeHole("contour_test",main_shape="triangle",satellite1_shape="triangle",distance1=.46,distance2=.46))
    old=geometry("triangular",d)
    for samples in (9,18,27):
        new=geometry("triangular",{**d,"_yag_triangle_samples_per_corner":samples})
        assert new.fourier_epsilon(0,0)==pytest.approx(old.fourier_epsilon(0,0),rel=2e-15)
        for a,b in zip(old.inclusions,new.inclusions):
            if hasattr(a,"vertices_fractional"):
                assert len(b.vertices_fractional)==3*samples
                assert b.fill_fraction==pytest.approx(a.fill_fraction,rel=2e-14)
                np.testing.assert_allclose(np.mean(a.vertices_fractional,axis=0),np.mean(b.vertices_fractional,axis=0),atol=1e-15)
                if samples==9:np.testing.assert_array_equal(a.vertices_fractional,b.vertices_fractional)


def test_actual_pole_pool_final_audit_keeps_target_completion_metadata(monkeypatch):
    import sys,copy
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root),str(root/"scripts"/"optimization")]
    from scripts.optimization import matched300_yag_search as workflow
    d=workflow.asdict(workflow.SquareDesign(fill=.25,main_share=.75,distance=.636396))
    model,_,_=workflow.optical("square",d,2,10.)
    spec,_,problem,mesh=workflow.finite("square",model,9,4,False)
    captured=[]
    monkeypatch.setattr(workflow,"save",lambda path,value:captured.append(copy.deepcopy(value)))
    pool=workflow.seed_pool("square",model,spec,problem,mesh,4,0,False,"memory_audit")
    assert pool
    final=captured[-1]
    assert final["planned_targets"]==final["completed_targets"]==len(final["targets"])==4
    assert final["total_independent_returned"]==len(pool)
