"""Independent units and angular-transform checks for the search entry."""
from types import SimpleNamespace

import numpy as np
import pytest

from pcselsim.config import CarrierConfig
from pcselsim.materials import active_gain_m
from scripts.optimization.square_semiconductor_search import (
    BASE, Design, broad_designs, build_cell, build_stack,
    direct_vector_far_field, minimum_grid, threshold,
)
from scripts.optimization.square_semiconductor_validate import independent_pole, perturbed_designs


@pytest.mark.parametrize("exponent", [2., 3.2, 4.])
def test_search_shape_fill_is_the_actual_fourier_average(exponent):
    design=Design("double_rounded",.14,exponent=exponent,aspect=1.3,angle=17.)
    cell=build_cell(design)
    inferred=(cell.background_epsilon-cell.fourier_epsilon(0,0).real)/(cell.background_epsilon-1)
    assert inferred==pytest.approx(design.fill,abs=2e-15)
    assert cell.fourier_epsilon(3,-2)==pytest.approx(np.conj(cell.fourier_epsilon(-3,2)))


def test_threshold_inverts_the_material_gain_law_and_physical_current_units():
    answer=threshold(500.,.08,3.4,300.)
    carrier=CarrierConfig(maximum_gain_cm=BASE.MAXIMUM_GAIN_CM,
        zero_carrier_gain_cm=BASE.ZERO_CARRIER_GAIN_CM,
        transparency_density_cm3=BASE.TRANSPARENCY_DENSITY_CM3)
    material=active_gain_m(np.asarray(answer["threshold_carrier_cm3"]*1e6),carrier)/100
    assert material==pytest.approx(answer["g_th_cm1"],rel=1e-13)
    assert answer["g_th_cm1"]*.08*BASE.ACTIVE_INDEX/3.4==pytest.approx(15.)
    assert answer["I_th_A"]==pytest.approx(answer["J_th_A_cm2"]*.03**2)


def test_direct_angular_transform_matches_an_independent_uniform_aperture_sum():
    count=9;length_um=200.;wavelength_nm=950.
    mode=SimpleNamespace(fields=np.ones((4,count,count)),radiation_x=np.ones((count,count)),
                         radiation_y=np.zeros((count,count)))
    axis,fx,fy,metrics=direct_vector_far_field(mode,wavelength_nm,length_um,samples=129)
    frequency=np.tan(np.deg2rad(axis))/(wavelength_nm*1e-9)
    phase=2*np.pi*frequency*length_um*1e-6/count
    # Closed-form geometric sum on symmetrically centered cell coefficients.
    amplitude=np.divide(np.sin(count*phase/2),np.sin(phase/2),
        out=np.full_like(phase,float(count)),where=abs(phase)>1e-12)
    xx,yy=np.meshgrid(np.deg2rad(axis),np.deg2rad(axis))
    exact=amplitude[:,None]*amplitude[None,:]*(np.cos(xx)+np.cos(yy)-1)
    assert np.allclose(fx,exact,rtol=2e-11,atol=2e-12)
    assert np.max(abs(fy))==0
    assert metrics["window_resolved_by_envelope_sampling"]
    assert metrics["center_to_peak"]==pytest.approx(1.)
    assert metrics["ellipticity"]==pytest.approx(1.)


def test_search_keeps_all_three_barriers_and_uses_a_sampling_safe_grid():
    design=Design("double",.14,pc_nm=210.,gaas_nm=90.)
    stack=build_stack(design,build_cell(design))
    assert sum(layer.thickness_nm for layer in stack.layers if "barrier" in layer.name)==60
    for length in (150.,200.,300.,400.,500.):
        n=minimum_grid(length,900.)
        nyquist=np.rad2deg(np.arctan(900e-9/(2*length*1e-6/n)))
        assert nyquist>1.


def test_design_proposals_replay_with_the_same_seed_and_count():
    first,rejected=broad_designs(4,183)
    second,_=broad_designs(4,183)
    assert [d.identifier for d in first]==[d.identifier for d in second]
    assert len(first)==6
    assert all(d.tag.startswith("broad_") for d in first[2:])


def test_lateral_cluster_dedup_keeps_an_independent_degenerate_partner():
    first=np.array([1.,0.],complex);second=np.array([0.,1.],complex)
    assert not independent_pole(100+3j,first*1j,[100+3j],[first])
    assert independent_pole(100+3j,second,[100+3j],[first])
    assert independent_pole(100.1+3j,first,[100+3j],[first])


def test_process_etch_bias_changes_actual_fill_and_never_preserves_it_artificially():
    design=Design("double",.14)
    perturbations=perturbed_designs(design)
    etched=[d for d in perturbations if d.tag.startswith("etch_bias")]
    assert len(perturbations)==8
    assert etched[0].fill<design.fill<etched[1].fill
    original=build_cell(design)
    for changed,bias in zip(etched,(-3.,3.)):
        cell=build_cell(changed)
        for before,after in zip(original.inclusions,cell.inclusions):
            assert np.asarray(after.radii)-before.radii==pytest.approx(np.full(2,bias/design.lattice_nm))
