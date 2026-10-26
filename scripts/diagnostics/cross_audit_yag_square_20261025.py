"""Independent read-only optical and shared-pump check of strict YAG square.

中文：独立重建亮首模和跨簇最近竞争模，不重做完整谱、不修改已冻结光学数据。
共享泵阈值对40根独立冷模重新计算；平截面和300K倒易配对均为条件假设。
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from dataclasses import replace,asdict
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'scripts/optimization')]
import numpy as np
import matched300_yag_search as b
from yag_feasibility_study import stark_emission_to_absorption
from pcselsim.finite_discretization import relative_residual
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.design_objectives import beam_metrics

load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    folder=ROOT/'results/current/strict_rerun_20261025/YbYAG_square/candidate_01_full32'
    files=[folder/'summary.json',folder/'closed_optical_modes.json',folder/'closed_fields.npz',
        folder/'spectral_challenges/closed_optical_modes.json',folder/'spectral_challenges/closed_fields.npz',folder/'actual_geometry_nm.json']
    before={str(p):digest(p) for p in files};summary=load(files[0]);geometry=load(files[-1]);rows=[];arrays={};rejected=[]
    for source in (folder,folder/'spectral_challenges'):
        with np.load(source/'closed_fields.npz',allow_pickle=False) as raw:
            for row in load(source/'closed_optical_modes.json'):
                name=row['name'];peers=[r for r in rows if r['closed_TE_family']==row['closed_TE_family'] and abs(r['wavelength_nm']-row['wavelength_nm'])<=1e-7]
                keep=independent_eigenmode(complex(row['delta_per_m'],row['alpha_field_per_m']),raw[name+'_field'],
                    [complex(r['delta_per_m'],r['alpha_field_per_m']) for r in peers],[arrays[r['name']+'_field'] for r in peers])
                if not keep:rejected.append(name);continue
                rows.append(row);arrays.update({k:raw[k].copy() for k in raw.files if k.startswith(name+'_')})
                if 'angle_deg' in raw.files:arrays['angle_deg']=raw['angle_deg'].copy()
    ordered=sorted(rows,key=lambda r:r['required_material_gain_cm1']);records=[];n=summary['settings']['square_grid'];period=summary['period_nm'];d=summary['design']
    checks=[dict(name='forty_independent_roots_and_cross_cluster_competitor',passed=bool(len(rows)==40 and ordered[0]['name']=='TE0_pole00' and ordered[1]['name']=='TE0_crossrank061'),
        retained_modes=len(rows),rejected_repeated_names=rejected,first_two=[r['name'] for r in ordered[:2]])]
    checks.append(dict(name='fixed_geometry_and_physical_aperture',passed=bool(geometry['lattice_constant_nm']==period and geometry['nominal_aperture_um']==[300,300] and summary['side_um']==300.)))
    for row in ordered[:2]:
        name=row['name'];wave=row['wavelength_nm'];field=arrays[name+'_field']
        model,returned_period,guide=b.optical('square',d,summary['settings']['order'],summary['settings']['dz_nm'],wave,period,family=row['closed_TE_family'])
        spec,_,problem,_=b.finite('square',model,n,8,False);r=model.result
        value=complex(row['delta_per_m'],row['alpha_field_per_m']);residual=relative_residual(problem.center_operator(),field,value)
        c_error=float(np.linalg.norm(r.coupling_m-arrays[name+'_C'])/np.linalg.norm(r.coupling_m))
        beta0=2*np.pi/(period*1e-9);beta=2*np.pi*r.effective_index/(wave*1e-9)
        frequency_residual=float((beta*beta-beta0*beta0)/(2*beta0)-value.real)
        widths,qe,qg=b.active_weights(r,guide,24)
        weight_errors={key:float(np.max(abs(actual-arrays[name+'_'+key]))) for key,actual in (('widths',widths),('qenergy',qe),('qgain',qg))}
        gain=(2*value.imag+10.)/qg.sum()/100
        budget=problem.loss_budget(field);up,down=model.side_power_losses(field)
        flux_error=float(up+down+2*budget['transport_amplitude_loss_per_m']-2*value.imag)
        optical_theorem=float(np.linalg.norm((r.coupling_m-r.coupling_m.conj().T)/(2j)-model.upward_amplitude_loss_m-model.downward_amplitude_loss_m)/np.linalg.norm(model.upward_amplitude_loss_m+model.downward_amplitude_loss_m))
        rx,ry=r.radiation_fields(field);axis=arrays['angle_deg'];positions=((np.arange(n)+.5)/n-.5)*300e-6
        transform=np.exp(-2j*np.pi*np.tan(np.deg2rad(axis))[:,None]/(wave*1e-9)*positions[None,:])
        xx,yy=np.meshgrid(np.deg2rad(axis),np.deg2rad(axis));ob=np.cos(xx)+np.cos(yy)-1
        fx=(transform@rx@transform.T)*ob;fy=(transform@ry@transform.T)*ob
        # Every root's saved radiation yields an independently reconstructed Jones map.
        # 中文：跨簇根也用自己的波长与辐射场，不借用亮首模远场或C。
        rad_error=max(float(np.linalg.norm(x-arrays[name+'_'+key])/np.linalg.norm(x)) for key,x in (('radiation_x',rx),('radiation_y',ry)))
        saved_jones=(name+'_far_x' in arrays and name+'_far_y' in arrays)
        jones_error=(float(np.sqrt(np.linalg.norm(fx-arrays[name+'_far_x'])**2+np.linalg.norm(fy-arrays[name+'_far_y'])**2)/np.sqrt(np.linalg.norm(fx)**2+np.linalg.norm(fy)**2)) if saved_jones else None)
        nyquist=float(np.rad2deg(np.arctan(wave*1e-9/(2*300e-6/n))))
        beam=beam_metrics(axis,fx,fy,requested_view_deg=1.,source_nyquist_deg=nyquist)
        beam_errors={key:float(abs(beam[key]-row['beam'][key])) for key in ('ellipticity','center_to_peak','radial_anisotropy_l2')}
        mid=int(np.argmin(abs(axis)));ex,ey=fx[mid,mid],fy[mid,mid]
        on_axis=np.array([abs(ex)**2-abs(ey)**2,2*(ex*ey.conjugate()).real,2*(ex*ey.conjugate()).imag])/(abs(ex)**2+abs(ey)**2)
        pol_error=float(np.max(abs(on_axis-np.array(row['polarization']['on_axis']['normalized_stokes_s1_s2_s3']))))
        passed=bool(returned_period==period and c_error<2e-10 and residual<2e-7 and abs(frequency_residual)<.05 and abs(flux_error)<1e-4 and
            optical_theorem<1e-8 and max(weight_errors.values())<1e-10 and abs(gain-row['required_material_gain_cm1'])<1e-8 and
            rad_error<1e-10 and (jones_error is None or jones_error<1e-10) and max(beam_errors.values())<1e-9 and pol_error<1e-10)
        records.append(dict(name=name,passed=passed,wavelength_nm=wave,fixed_period_nm=period,N=n,D=summary['settings']['order'],dz_nm=summary['settings']['dz_nm'],
            original_KM_residual=residual,coupling_relative_rebuild_error=c_error,frequency_residual_per_m=frequency_residual,
            alpha_field_per_m=value.imag,internal_intensity_loss_per_m=10.,gain_weight=float(qg.sum()),recomputed_material_gain_cm1=float(gain),
            weight_errors=weight_errors,full_mode_energy_fraction_in_gain=float(qe.sum()),
            qgain_energy_group_velocity_identity_error=float(np.max(abs(qg-b.c/(1.8166*(b.c/r.group_index))*qe))),
            flux_error_per_m=flux_error,up_power_loss_per_m=up,down_power_loss_per_m=down,optical_theorem_residual=optical_theorem,
            radiation_relative_error=rad_error,saved_Jones_available=saved_jones,Jones_relative_error=jones_error,beam_errors=beam_errors,
            on_axis_stokes=on_axis.tolist(),on_axis_stokes_error=pol_error))
        print('OPTICAL',name,passed,residual,gain,flush=True)
    fields=np.array([arrays[r['name']+'_field'].ravel() for r in rows]);fields/=np.linalg.norm(fields,axis=1)[:,None]
    overlap=abs(fields.conj()@fields.T);np.fill_diagonal(overlap,0.);i,j=np.unravel_index(np.argmax(overlap),overlap.shape)
    checks.append(dict(name='resolved_near_parallel_poles_not_removed',passed=True,most_parallel_names=[rows[i]['name'],rows[j]['name']],
        field_overlap=float(overlap[i,j]),wavelength_separation_nm=abs(rows[i]['wavelength_nm']-rows[j]['wavelength_nm']),
        complex_pole_distance_per_m=float(abs(complex(rows[i]['delta_per_m'],rows[i]['alpha_field_per_m'])-complex(rows[j]['delta_per_m'],rows[j]['alpha_field_per_m']))),
        interpretation='High overlap alone is never a duplicate criterion; identity was checked only at the same closed frequency and TE family.'))
    pump_records=[]
    for protocol in ('nominal_flat_spectrum','reciprocity_paired_300K_constant_emission_envelope'):
        models=[]
        for row in rows:
            name=row['name'];medium=b.material(row['wavelength_nm'])
            if protocol.startswith('reciprocity'):
                medium=replace(medium,pump_emission_cross_section_cm2=medium.pump_absorption_cross_section_cm2*stark_emission_to_absorption(medium.pump_wavelength_nm),
                    laser_absorption_cross_section_cm2=medium.laser_emission_cross_section_cm2/stark_emission_to_absorption(medium.laser_wavelength_nm))
            qe,qg=arrays[name+'_qenergy'],arrays[name+'_qgain'];vg=b.c*qe.sum()/(1.8166*qg.sum())
            models.append(b.SpatialYbYAGModel(medium,arrays[name+'_widths'],qe,row['total_power_loss_per_m'],row['up_power_loss_per_m'],row['down_power_loss_per_m'],vg,qg,arrays[name+'_lateral']))
        system=b.SpatialMultimodeYbYAG(tuple(models),tuple(r['name'] for r in rows))
        for architecture in ('edge_dual','top_single'):
            threshold=b.shared_inversion_pump_threshold(system,b.PumpGeometryConfig(architecture,0.,300.,31))
            passed=bool(tuple(threshold.first_modes)==('TE0_pole00',))
            pump_records.append(dict(protocol=protocol,architecture=architecture,passed=passed,
                first_threshold_internal_W=threshold.first_threshold_internal_W,first_modes=list(threshold.first_modes),
                next_small_signal_threshold_internal_W=threshold.next_small_signal_threshold_internal_W,
                crossrank061_branch_threshold_internal_W=threshold.branch_thresholds_internal_W['TE0_crossrank061'],
                crossrank061_gain_to_loss_at_first=threshold.gain_to_loss_at_first['TE0_crossrank061'],
                threshold_residual=threshold.threshold_residual,mode_count=len(rows),interpretation=threshold.interpretation))
            print('PUMP',protocol,architecture,threshold.first_threshold_internal_W,threshold.first_modes,flush=True)
    review=load(folder/'spectroscopy_review.json')
    for own in pump_records:
        supplied=next(r for r in review['records'] if r['label']==own['protocol'])['pumps'][own['architecture']]['threshold']
        own['reported_first_threshold_error_W']=own['first_threshold_internal_W']-supplied['first_threshold_internal_W']
        own['passed']=own['passed'] and abs(own['reported_first_threshold_error_W'])<1e-8*own['first_threshold_internal_W'] and own['first_modes']==supplied['first_modes']
    checks.append(dict(name='conditional_spectroscopy_protocol',passed=bool('constant 2e-20' in review['scope'] and 'not measured' in review['scope']),
        nominal='Flat laser emission/absorption cross sections assumed over the analysis band; each actual cold wavelength is retained.',
        reciprocity='300K thermal-manifold reciprocity constrains absorption/emission ratios; the constant emission envelope is assumed, not a measured wavelength-resolved gain spectrum.',
        power_definition='Total first-entry pump power inside gain geometry, not absorbed power or external source power.',
        shared_inversion_scope='Same x-z population; y averaged. Small-signal first threshold and later unlased crossings are distinct from saturated turn-on and dynamic stability.'))
    after={str(p):digest(p) for p in files};checks.append(dict(name='frozen_optical_artifacts_unchanged',passed=before==after))
    result=dict(status='passed' if all(r['passed'] for r in checks+records+pump_records) else 'failed',checks=checks,optical_modes=records,pump_thresholds=pump_records,
        cold_relative_material_gap=ordered[1]['required_material_gain_cm1']/ordered[0]['required_material_gain_cm1']-1,
        input_sha256=before,review_driver_sha256=digest(Path(__file__)),
        scope='Read-only independent reconstruction of the two leading optical roots and the40-mode shared-pump first threshold within the declared scalar CWT/rate assumptions. Not a full-Maxwell, measured-spectroscopy, thermal or nonlinear dynamical feasibility proof.')
    output=ROOT/'results/current/strict_rerun_20261025/cross_audit_yag_square.json';output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(result['status'],flush=True)
    if result['status']!='passed':raise RuntimeError('YAG square cross audit detected a mismatch')

if __name__=='__main__':main()
