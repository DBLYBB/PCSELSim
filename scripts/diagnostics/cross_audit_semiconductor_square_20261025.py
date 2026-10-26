"""Independent read-only checks of the square agent's selected PCSEL.

中文：不改正方晶格已冻结结果。重建各自真实波长的D64耦合与原K/M，
并独立读回增益单位、能流收支、Jones变换和轴上Stokes。
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from pcselsim.custom_analysis import FourWaveOpticalSpec,finite_area_eigenproblem
from pcselsim.finite_discretization import relative_residual
from pcselsim.design_objectives import beam_metrics
from scripts.optimization.matched300_square_search import Design,build_model,BASE

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    path=ROOT/'results/current/strict_rerun_20261025/semiconductor_square/selected_case.json'
    s=json.loads(path.read_text(encoding='utf-8'));fields_path=Path(s['fields_path'])
    before={str(p):digest(p) for p in (path,Path(s['summary_path']),fields_path)}
    d=Design(**s['design']);records=[];checks=[]
    checks.append(dict(name='fixed_size_period_and_discretization',passed=bool(d.length_um==300. and
        d.lattice_nm==s['actual_period_nm'] and s['grid']==65 and s['D']==64 and s['physical_aperture_um']==[300.,300.])))
    with np.load(fields_path,allow_pickle=False) as a:
        checks.append(dict(name='manifest_fields_sha256',passed=digest(fields_path)==s['fields']['sha256']))
        axis=a['axis_deg'];n=s['grid'];positions=((np.arange(n)+.5)/n-.5)*300e-6
        for row in s['frequency_closed_modes'][:2]:
            name=row['mode'];wave=row['wavelength_nm'];field=a[name+'_field']
            _,_,model=build_model(d,s['D'],s['dz_nm'],wave,False,96)
            spec=FourWaveOpticalSpec(wave,model.effective_index,model.group_index,300.,5.,
                (0.,)*4,(0.,)*4,n,'staggered',24,0,1600)
            problem=finite_area_eigenproblem(spec,model.coupling_m)
            alpha_per_m=100*row['alpha_field_cm1'];value=complex(row['delta_per_m'],alpha_per_m)
            residual=relative_residual(problem.center_operator(),field,value)
            c_error=float(np.linalg.norm(model.coupling_m-a[name+'_coupling_m'])/np.linalg.norm(model.coupling_m))
            beta0=2*np.pi/(d.lattice_nm*1e-9);beta=2*np.pi*model.effective_index/(wave*1e-9)
            frequency_residual=float((beta*beta-beta0*beta0)/(2*beta0)-value.real)
            budget=problem.loss_budget(field)
            budget_error=float(budget['local_amplitude_loss_per_m']+budget['transport_amplitude_loss_per_m']-alpha_per_m)
            gamma=model.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME]
            qgain=gamma*BASE.ACTIVE_INDEX/model.effective_index
            material_gain=(2*alpha_per_m+100*BASE.INTERNAL_LOSS_CM)/qgain/100
            rx,ry=model.radiation_fields(field)
            radiation_error=max(float(np.linalg.norm(actual-a[name+key])/np.linalg.norm(actual))
                for actual,key in ((rx,'_radiation_x'),(ry,'_radiation_y')))
            # Direct physical-coordinate Fourier sum; Jones components retain a common scale.
            # 中文：直接由新重建辐射包络求和，分量间不各自归一化。
            transform=np.exp(-2j*np.pi*np.tan(np.deg2rad(axis))[:,None]/(wave*1e-9)*positions[None,:])
            theta_x,theta_y=np.meshgrid(np.deg2rad(axis),np.deg2rad(axis))
            obliquity=np.cos(theta_x)+np.cos(theta_y)-1
            fx=(transform@rx@transform.T)*obliquity;fy=(transform@ry@transform.T)*obliquity
            far_error=float(np.sqrt(np.linalg.norm(fx-a[name+'_far_x'])**2+np.linalg.norm(fy-a[name+'_far_y'])**2)/
                np.sqrt(np.linalg.norm(fx)**2+np.linalg.norm(fy)**2))
            nyquist=float(np.rad2deg(np.arctan(wave*1e-9/(2*300e-6/n))))
            beam=beam_metrics(axis,fx,fy,requested_view_deg=1.,source_nyquist_deg=nyquist)
            beam_errors={key:float(abs(beam[key]-row['beam'][key])) for key in
                ('ellipticity','center_to_peak','radial_anisotropy_l2','r80_deg')}
            mid=int(np.argmin(abs(axis)));ex,ey=fx[mid,mid],fy[mid,mid]
            stokes=np.array([abs(ex)**2-abs(ey)**2,2*(ex*ey.conjugate()).real,2*(ex*ey.conjugate()).imag])/(abs(ex)**2+abs(ey)**2)
            stokes_error=float(np.max(abs(stokes-np.array(row['polarization']['on_axis']['normalized_stokes_s1_s2_s3']))))
            weights=np.ones(len(axis));weights[[0,-1]]=.5;weights=weights[:,None]*weights[None,:]
            s0=float(np.sum((abs(fx)**2+abs(fy)**2)*weights))
            integrated=np.array([np.sum((abs(fx)**2-abs(fy)**2)*weights),
                np.sum(2*(fx*fy.conj()).real*weights),np.sum(2*(fx*fy.conj()).imag*weights)])/s0
            integrated_error=float(np.max(abs(integrated-np.array(row['polarization']['integrated']['normalized_stokes_s1_s2_s3']))))
            passed=bool(c_error<2e-10 and residual<2e-7 and abs(frequency_residual)<.05 and abs(budget_error)<1e-4 and
                abs(material_gain-row['g_th_cm1'])<1e-8 and far_error<1e-10 and radiation_error<1e-10 and
                max(beam_errors.values())<1e-9 and stokes_error<1e-10 and integrated_error<1e-10)
            records.append(dict(mode=name,passed=passed,wavelength_nm=wave,fixed_period_nm=d.lattice_nm,
                D=s['D'],dz_nm=s['dz_nm'],N=n,original_KM_residual=residual,coupling_relative_rebuild_error=c_error,
                frequency_residual_per_m=frequency_residual,loss_budget=budget,alpha_field_per_m=alpha_per_m,
                loss_budget_error_per_m=budget_error,internal_intensity_loss_cm1=BASE.INTERNAL_LOSS_CM,
                Gamma=gamma,active_index=BASE.ACTIVE_INDEX,effective_index=model.effective_index,gain_weight=qgain,
                recomputed_material_gain_cm1=material_gain,gain_error_cm1=material_gain-row['g_th_cm1'],
                radiation_error=radiation_error,Jones_relative_error=far_error,beam_errors=beam_errors,
                on_axis_normalized_stokes=stokes.tolist(),on_axis_stokes_error=stokes_error,
                integrated_stokes_error=integrated_error,on_axis_stokes_length=float(np.linalg.norm(stokes)),
                near_Bragg_beta_ref_over_beta0=float(beta/beta0)))
        rows=s['frequency_closed_modes'][:2]
        separation=abs(rows[0]['wavelength_nm']-rows[1]['wavelength_nm'])
        overlap=float(abs(np.vdot(a[rows[0]['mode']+'_field'].ravel(),a[rows[1]['mode']+'_field'].ravel()))/
            (np.linalg.norm(a[rows[0]['mode']+'_field'])*np.linalg.norm(a[rows[1]['mode']+'_field'])))
        checks.append(dict(name='first_two_distinct_closed_frequencies',passed=separation>2e-7,
            wavelength_separation_nm=separation,cross_frequency_field_overlap_diagnostic=overlap,
            identity_rule='Distinct closed frequencies of the same fixed geometry are separate modes; cross-frequency field overlap is not a duplicate test.'))
    after={str(p):digest(p) for p in (path,Path(s['summary_path']),fields_path)}
    checks.append(dict(name='frozen_square_artifacts_unchanged',passed=before==after))
    result=dict(status='passed' if all(r['passed'] for r in records+checks) else 'failed',checks=checks,modes=records,
        input_sha256=before,review_driver_sha256=digest(Path(__file__)),
        threshold_formula='[(2*alpha_field_per_m)+(internal_intensity_loss_cm1*100)]/gain_weight/100',
        polarization_convention='S3=+2Im(Ex Ey*), axial coherency is pure; angular integration can lower spatially averaged DOP.',
        scope='Independent read-only reconstruction of two selected modes within the declared scalar TE0 near-Bragg CWT. Does not repeat or certify complete fine-grid spectral search, Maxwell physics, fabrication, heat or nonlinear single-mode stability.')
    output=ROOT/'results/current/strict_rerun_20261025/cross_audit_semiconductor_square.json'
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(result['status'],[(r['mode'],r['original_KM_residual'],r['recomputed_material_gain_cm1'],r['Jones_relative_error']) for r in records])
    if result['status']!='passed':raise RuntimeError('Cross audit detected a mismatch')

if __name__=='__main__':main()
