"""Degenerate span counting and resolved nonnormal-mode identity regressions."""
from types import SimpleNamespace

import numpy as np
import pytest

from pcselsim.mode_identity import independent_eigenmode, independent_mode_indices
from scripts.optimization.square_semiconductor_validate import independent_pole
from scripts.optimization.matched300_yag_search import distinct_closed_modes, seed_pool


def test_rotated_basis_does_not_create_a_third_degenerate_mode():
    e1,e2,e3=np.eye(3,dtype=complex)
    fields=[e1,e2,(e1+1j*e2)/np.sqrt(2),e3]
    assert independent_mode_indices([100+2j]*4,fields)==[0,1,3]
    assert not independent_pole(100+2j,fields[2],[100+2j]*2,fields[:2])


def test_resolved_nearby_nonnormal_modes_are_kept_despite_high_overlap():
    # Distinct eigenvalues of [[v0,1],[0,v1]] have nearly parallel right
    # eigenvectors. Overlap alone would incorrectly delete the second mode.
    v0=100+2j;v1=v0+2e-5+1e-5j
    first=np.array([1.,0.],complex)
    second=np.array([1.,v1-v0],complex)
    assert abs(np.vdot(first,second)/np.linalg.norm(second))**2>.999999
    assert independent_eigenmode(v1,second,[v0],[first])
    assert independent_pole(v1,second,[v0],[first])


def test_mode_identity_is_invariant_to_independent_complex_scale_and_phase():
    e1,e2=np.eye(2,dtype=complex)
    candidate=(e1+e2)*(-3e-120+2e-120j)
    previous=[e1*(2e120-1e120j),e2*3j]
    assert not independent_eigenmode(100+2j,candidate,[100+2j]*2,previous)
    assert not independent_eigenmode(100+2j,e1*1e-200,[100+2j],[e1*1e200])


def test_eigenvalue_clustering_uses_imaginary_as_well_as_real_part():
    field=np.ones(3,complex)
    assert independent_eigenmode(100+2.001j,field,[100+2j],[field])
    assert not independent_eigenmode(100+2j+1e-9,field,[100+2j],[field])


def test_closed_yag_dedup_respects_family_and_evaluation_frequency():
    e1,e2=np.eye(2,dtype=complex)
    fields=[e1,e2,(e1+e2)/np.sqrt(2),e1,e1,e1]
    rows=[dict(name=f"m{i}",delta_per_m=100.,alpha_field_per_m=2.,
        wavelength_nm=1030.,closed_TE_family=0) for i in range(6)]
    rows[3]["closed_TE_family"]=1
    rows[4]["wavelength_nm"]=1030.001
    rows[5]["alpha_field_per_m"]=2.001
    stored=[(SimpleNamespace(fields=field),) for field in fields]
    kept,audit=distinct_closed_modes(rows,stored)
    assert kept==[0,1,3,4,5]
    assert audit["repeated_records"]==[dict(name="m2",compared_with=["m0","m1"])]


def test_yag_seed_census_counts_a_repeated_rotated_subspace_once():
    e1,e2,_,_=np.eye(4,dtype=complex)
    value=100+2j
    basis=np.column_stack((e1,e2,(e1+e2)/np.sqrt(2)))
    problem=SimpleNamespace(stiffness=np.zeros((4,4)),
        near=lambda *args,**kwargs:(np.full(3,value),basis),
        center_operator=lambda:np.eye(4)*value)
    model=SimpleNamespace(result=SimpleNamespace(eigenvalues_m=np.array([value])))
    modes=seed_pool("square",model,SimpleNamespace(grid_points=1),problem,None,
                    side_roots=3,extra_targets=1)
    assert len(modes)==2
    assert np.linalg.matrix_rank(np.column_stack([field.ravel() for _,field in modes]))==2


def test_invalid_identity_inputs_raise_instead_of_silently_changing_count():
    with pytest.raises(ValueError):independent_eigenmode(1.,np.zeros(2),[],[])
    with pytest.raises(ValueError):independent_eigenmode(1.,np.ones(2),[1.],[])
    with pytest.raises(ValueError):independent_eigenmode(np.nan,np.ones(2),[],[])
    with pytest.raises(ValueError):independent_eigenmode(1.,np.ones(2),[1.],[np.ones(3)])
