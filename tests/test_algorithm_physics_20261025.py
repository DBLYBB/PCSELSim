"""Physical counterexamples for nonnormal mode identity and legacy closure."""
from types import SimpleNamespace
import numpy as np
from scipy import sparse


def passive_nearly_parallel_poles(dimension=6):
    """Analytic upper-triangular passive matrix, not a mocked eigenvalue list."""
    first=1e5+60j
    second=first+.01
    matrix=np.diag([first]+[second]+[2e5+60j]*(dimension-2)).astype(complex)
    matrix[0,1]=100.
    field0=np.eye(dimension,dtype=complex)[:,0]
    field1=field0+(.01/100)*np.eye(dimension,dtype=complex)[:,1]
    field1/=np.linalg.norm(field1)
    # Im(C) has eigenvalues 10,110,60,...: this is a passive nonnormal example.
    assert np.linalg.eigvalsh((matrix-matrix.conj().T)/(2j)).min()>0
    for value,field in ((first,field0),(second,field1)):
        assert np.linalg.norm(matrix@field-value*field)<1e-10
    assert abs(np.vdot(field0,field1))**2>.99999
    return matrix,np.array([first,second]),np.column_stack((field0,field1))


def test_triangular_pool_retains_distinct_passive_nearly_parallel_poles(monkeypatch):
    from scripts.optimization import triangular_semiconductor_search as search
    matrix,values,fields=passive_nearly_parallel_poles()
    class Problem:
        stiffness=sparse.csc_matrix(matrix)
        def center_operator(self):return matrix
        def near(self,*args,**kwargs):return values.copy(),fields.copy()
        def loss_budget(self,field):
            local=float(np.vdot(field.ravel(),matrix@field.ravel()).imag)
            return {'local_amplitude_loss_per_m':local,'transport_amplitude_loss_per_m':0.}
    grid=SimpleNamespace(points=1)
    monkeypatch.setattr(search,'triangular_finite_eigenproblem',lambda *args:(Problem(),grid))
    monkeypatch.setattr(search,'_transform_six_wave_fields',lambda *args:None)
    model=SimpleNamespace(eigenvalues_m=values[:1],eigenvectors=np.eye(6,dtype=complex))
    spec=SimpleNamespace(radius_um=1.,aperture='square')
    records,retained,_=search.collect_spectrum(model,spec,3,count=2)
    assert len(records)==2
    assert abs(records[1]['delta_per_m']-records[0]['delta_per_m']-.01)<1e-10
    assert retained.shape==(2,6,1)


def test_legacy_closed_mode_identity_requires_complex_pole_and_same_operator(tmp_path):
    from scripts.optimization.close_ybyag_lateral_challengers import closed_duplicate_name
    matrix,values,fields=passive_nearly_parallel_poles(4)
    reference=fields[:,0].reshape(4,1,1)
    candidate=fields[:,1].reshape(4,1,1)
    np.savez(tmp_path/'old_fields.npz',field=reference,C=matrix)
    rows=[{'name':'old','te_family':0,'wavelength_nm':1030.,
           'delta_per_m':float(values[0].real),'amplitude_loss_per_m':float(values[0].imag)}]
    second=SimpleNamespace(delta_per_m=values[1].real,alpha_per_m=values[1].imag,fields=candidate)
    assert closed_duplicate_name(second,1030.,matrix,rows,tmp_path) is None
    first=SimpleNamespace(delta_per_m=values[0].real,alpha_per_m=values[0].imag,fields=reference)
    assert closed_duplicate_name(first,1030.,matrix,rows,tmp_path)=='old'
    assert closed_duplicate_name(first,1030.001,matrix,rows,tmp_path) is None
    assert closed_duplicate_name(first,1030.,matrix+np.eye(4),rows,tmp_path) is None
    rows[0]['te_family']=1
    assert closed_duplicate_name(first,1030.,matrix,rows,tmp_path) is None
    assert closed_duplicate_name(first,1030.,matrix,rows,tmp_path,family=1)=='old'


def test_refined_yag_modes_use_frequency_family_operator_and_complex_identity(tmp_path):
    from scripts.optimization.refine_ybyag_saved_poles import closed_duplicate_name
    matrix,values,fields=passive_nearly_parallel_poles(4)
    np.savez(tmp_path/'retained_fields.npz',field=fields[:,0].reshape(4,1,1),C=matrix)
    rows=[{'name':'retained','te_family':2,'wavelength_nm':1030.,
           'amplitude_loss_per_m':float(values[0].imag),
           'frequency_evaluations':[{'finite_detuning_per_m':float(values[0].real)}]}]
    candidate=SimpleNamespace(delta_per_m=values[1].real,alpha_per_m=values[1].imag,
                              fields=fields[:,1].reshape(4,1,1))
    assert closed_duplicate_name(candidate,1030.,matrix,rows,tmp_path,family=2) is None
    candidate.delta_per_m=values[0].real
    candidate.fields=fields[:,0].reshape(4,1,1)
    assert closed_duplicate_name(candidate,1030.,matrix,rows,tmp_path,family=2)=='retained'


def test_small_signal_index_shift_matches_independent_dispersive_resonance(tmp_path):
    import csv
    from scipy.constants import e
    from scipy.optimize import brentq
    from scripts._triangular_workflow import small_signal_pump_sweep
    p=SimpleNamespace(DEVICE_SHAPE='square',DEVICE_HALF_SIZE_UM=150.,
        CARRIER_LIFETIME_NS=1.5,ACTIVE_THICKNESS_NM=30.,ACTIVE_LAYER_NAME='gain',
        ACTIVE_INDEX=3.584,INTERNAL_LOSS_CM=5.,PUMP_RATIOS=(.8,1.,1.2),
        MAXIMUM_GAIN_CM=2000.,ZERO_CARRIER_GAIN_CM=-5000.,
        TRANSPARENCY_DENSITY_CM3=1.5e18,DN_DN_CM3=-5e-21)
    reference_density=3e18
    area=(300e-6)**2
    current=e*30e-9*area*reference_density*1e6/(1.5e-9)
    reference_wave,neff,ng=950.,3.4,3.8
    model=SimpleNamespace(bragg_wavelength_nm=reference_wave,effective_index=neff,
        group_index=ng,vertical_mode=SimpleNamespace(confinement={'gain':.05}))
    small_signal_pump_sweep(p,tmp_path,
        {'threshold_current_A':current,'threshold_density_cm-3':reference_density},
        SimpleNamespace(alpha_per_m=100.),model)
    with (tmp_path/'12_small_signal_pump_sweep.csv').open(encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    slope=(neff-ng)/reference_wave
    for row in rows:
        dn=float(row['unsaturated_index_shift'])
        # Independent condition n_eff(lambda)/lambda=the same fixed beta/k0.
        wave=brentq(lambda w:(neff+slope*(w-reference_wave)+dn)/w-neff/reference_wave,
                    reference_wave-5.,reference_wave+5.,xtol=1e-11)
        assert abs((wave-reference_wave)-float(row['unsaturated_wavelength_shift_nm']))<2e-10
