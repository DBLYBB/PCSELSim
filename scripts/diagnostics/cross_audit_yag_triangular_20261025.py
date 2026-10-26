"""Read-only independent audit of the final triangular Yb:YAG merged case.

中文：重建最低两根自身C/KM/频率/辐射/Jones，并对所有根重算两套共享泵首阈。
Uses actual coefficient coordinates and a direct point-sum transform, no FFT interpolation.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from dataclasses import replace,asdict
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'scripts/optimization')]
import numpy as np
import matched300_yag_search as b
import strict_yag_feasibility_20261025 as files
from yag_feasibility_study import stark_emission_to_absorption
from pcselsim.finite_discretization import relative_residual
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.design_objectives import beam_metrics
from pcselsim.polarization import polarization_metrics

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    selected,sp,rp,fp,cp=files.files_for('triangular')
    summary=read(sp);rows=read(rp);folder=sp.parent
    gp=Path(selected.get('geometry_path',folder/'actual_geometry_nm.json'))
    vp=Path(selected.get('spectroscopy_review_path',folder/'spectroscopy_review.json'))
    review=read(vp);geometry=read(gp)
    sources=[sp,rp,fp,cp,gp,vp];before={str(p):sha(p) for p in sources}
    with np.load(fp,allow_pickle=False) as raw:arrays={k:raw[k].copy() for k in raw.files}
    period=summary['period_nm'];design=summary['design'];settings=summary['settings']
    D,dz,N=settings['order'],settings['dz_nm'],settings['tri_radius_cells']
    accepted=[];duplicates=[]
    for row in rows:
        peers=[r for r in accepted if r['closed_TE_family']==row['closed_TE_family']
            and abs(r['wavelength_nm']-row['wavelength_nm'])<=1e-7]
        if not independent_eigenmode(complex(row['delta_per_m'],row['alpha_field_per_m']),arrays[row['name']+'_field'],
            [complex(r['delta_per_m'],r['alpha_field_per_m']) for r in peers],
            [arrays[r['name']+'_field'] for r in peers]):duplicates.append(row['name'])
        else:accepted.append(row)
    ordered=sorted(rows,key=lambda r:r['required_material_gain_cm1'])
    checks=[dict(name='closed_identity_and_root_count',passed=bool(not duplicates and len(rows)>=32),
        retained_modes=len(rows),duplicate_names=duplicates,lowest_two=[r['name'] for r in ordered[:2]])]
    boundaries=[b.inclusion_boundary(h,b.TRIANGULAR_DIRECT,384)*period for h in b.geometry('triangular',design).inclusions]
    geometry_error=max(float(np.max(abs(actual-np.array(saved['boundary_cartesian_nm']))))
        for actual,saved in zip(boundaries,geometry['holes']))
    checks.append(dict(name='physical_geometry_and_aperture',passed=bool(len(geometry['holes'])==3
        and geometry['nominal_aperture_um']==[300,300] and summary['side_um']==300.
        and geometry['lattice_constant_nm']==period and geometry_error<1e-8),
        boundary_max_difference_nm=geometry_error,period_nm=period))
    records=[];axis=arrays[selected.get('axis_key','angle_deg')]
    for row in ordered[:2]:
        name=row['name'];wave=row['wavelength_nm'];field=arrays[name+'_field']
        model,returned_period,guide=b.optical('triangular',design,D,dz,wave,period,row['closed_TE_family'])
        spec,_,problem,mesh=b.finite('triangular',model,N,8,False);r=model.result
        value=complex(row['delta_per_m'],row['alpha_field_per_m'])
        eigen=relative_residual(problem.center_operator(),field,value)
        cerror=float(np.linalg.norm(r.coupling_m-arrays[name+'_C'])/np.linalg.norm(r.coupling_m))
        beta0=4*np.pi/np.sqrt(3)/(period*1e-9);beta=2*np.pi*r.effective_index/(wave*1e-9)
        frequency=float((beta*beta-beta0*beta0)/(2*beta0)-value.real)
        widths,qe,qg=b.active_weights(r,guide,24)
        weight_errors={key:float(np.max(abs(actual-arrays[name+'_'+key])))
            for key,actual in (('widths',widths),('qenergy',qe),('qgain',qg))}
        gain=(2*value.imag+10.)/qg.sum()/100
        up,down=model.side_power_losses(field)
        flux=float(up+down+2*problem.loss_budget(field)['transport_amplitude_loss_per_m']-2*value.imag)
        optical=float(np.linalg.norm((r.coupling_m-r.coupling_m.conj().T)/(2j)
            -model.upward_amplitude_loss_m-model.downward_amplitude_loss_m)
            /max(np.linalg.norm(model.upward_amplitude_loss_m+model.downward_amplitude_loss_m),1e-30))
        coordinate_error=max(float(np.max(abs(mesh.x_um-arrays['x_um']))),float(np.max(abs(mesh.y_um-arrays['y_um']))))
        rx,ry=r.radiation_amplitudes(field)
        rad_error=max(float(np.linalg.norm(actual-arrays[name+'_'+key])/max(np.linalg.norm(actual),1e-30))
            for key,actual in (('radiation_x',rx),('radiation_y',ry)))
        k=2*np.pi/(wave*1e-9)*np.tan(np.deg2rad(axis))
        tx=np.exp(-1j*k[:,None]*(mesh.x_um*1e-6)[None,:])
        ty=np.exp(-1j*k[:,None]*(mesh.y_um*1e-6)[None,:])
        ax,ay=np.meshgrid(np.deg2rad(axis),np.deg2rad(axis));ob=np.cos(ax)+np.cos(ay)-1
        fx=((ty*rx[None,:])@tx.T)*ob;fy=((ty*ry[None,:])@tx.T)*ob
        saved_jones=name+'_far_x' in arrays and name+'_far_y' in arrays
        jerror=(float(np.sqrt(np.linalg.norm(fx-arrays[name+'_far_x'])**2+np.linalg.norm(fy-arrays[name+'_far_y'])**2)
            /max(np.sqrt(np.linalg.norm(fx)**2+np.linalg.norm(fy)**2),1e-30)) if saved_jones else None)
        safe=float(np.rad2deg(np.arctan(2*wave*1e-9/(3*mesh.spacing_m*(1+1/np.sqrt(3))))))
        beam=beam_metrics(axis,fx,fy,requested_view_deg=1.,source_nyquist_deg=safe)
        beam_errors={key:float(abs(beam[key]-row['beam'][key]))
            for key in ('ellipticity','center_to_peak','radial_anisotropy_l2')}
        pol=polarization_metrics(axis,fx,fy,core_radius_deg=.2)
        pol_errors=[]
        for region in ('on_axis','core','integrated'):
            own,supplied=pol[region],row['polarization'][region]
            if own is None or supplied is None:
                pol_errors.append(0. if own is None and supplied is None else np.inf)
            else:pol_errors.append(float(np.max(abs(np.array(own['normalized_stokes_s1_s2_s3'])
                -np.array(supplied['normalized_stokes_s1_s2_s3'])))))
        pol_error=max(pol_errors)
        passed=bool(returned_period==period and cerror<2e-9 and eigen<2e-7 and abs(frequency)<.05
            and abs(flux)<1e-4 and optical<1e-8 and max(weight_errors.values())<1e-10
            and abs(gain-row['required_material_gain_cm1'])<1e-8 and coordinate_error<1e-12
            and rad_error<1e-10 and (jerror is None or jerror<1e-9)
            and max(beam_errors.values())<1e-8 and pol_error<1e-8)
        records.append(dict(name=name,passed=passed,wavelength_nm=wave,D=D,dz_nm=dz,N=N,
            fixed_period_nm=period,original_KM_residual=eigen,C_relative_rebuild_error=cerror,
            frequency_residual_per_m=frequency,material_gain_cm1=float(gain),gain_weight=float(qg.sum()),
            weight_errors=weight_errors,passive_flux_error_per_m=flux,optical_theorem_residual=optical,
            coefficient_coordinate_max_error_um=coordinate_error,radiation_relative_error=rad_error,
            saved_Jones_available=saved_jones,Jones_relative_error=jerror,beam_errors=beam_errors,
            polarization_stokes_max_absolute_error=pol_error,on_axis_polarization=pol['on_axis']))
        print('INDEPENDENT_TRI_OPTICAL',name,passed,gain,frequency,flush=True)
    pumps=[]
    for protocol in ('nominal_flat_spectrum','reciprocity_paired_300K_constant_emission_envelope'):
        models=[]
        for row in rows:
            name=row['name'];medium=b.material(row['wavelength_nm'])
            if protocol.startswith('reciprocity'):
                medium=replace(medium,pump_emission_cross_section_cm2=medium.pump_absorption_cross_section_cm2*stark_emission_to_absorption(medium.pump_wavelength_nm),
                    laser_absorption_cross_section_cm2=medium.laser_emission_cross_section_cm2/stark_emission_to_absorption(medium.laser_wavelength_nm))
            qe,qg=arrays[name+'_qenergy'],arrays[name+'_qgain'];vg=b.c*qe.sum()/(1.8166*qg.sum())
            if not np.allclose(qg,b.c/(1.8166*vg)*qe,rtol=2e-12,atol=1e-16):raise ValueError('Mode gain-energy identity failed')
            models.append(b.SpatialYbYAGModel(medium,arrays[name+'_widths'],qe,row['total_power_loss_per_m'],
                row['up_power_loss_per_m'],row['down_power_loss_per_m'],vg,qg,arrays[name+'_lateral']))
        system=b.SpatialMultimodeYbYAG(tuple(models),tuple(r['name'] for r in rows))
        expected=next(r for r in review['records'] if r['label']==protocol)
        for architecture in ('edge_dual','top_single'):
            threshold=b.shared_inversion_pump_threshold(system,b.PumpGeometryConfig(architecture,0.,300.,31))
            supplied=expected['pumps'][architecture]['threshold']
            difference=abs(threshold.first_threshold_internal_W-supplied['first_threshold_internal_W'])
            passed=bool(list(threshold.first_modes)==supplied['first_modes'] and difference<1e-8*threshold.first_threshold_internal_W)
            pumps.append(dict(protocol=protocol,architecture=architecture,passed=passed,
                threshold=asdict(threshold),reported_first_threshold_difference_W=difference,
                actual_first_wavelengths_nm=[next(r['wavelength_nm'] for r in rows if r['name']==n) for n in threshold.first_modes]))
            print('INDEPENDENT_TRI_PUMP',protocol,architecture,threshold.first_threshold_internal_W,flush=True)
    nominal_edge=next(p for p in pumps if p['protocol']=='nominal_flat_spectrum' and p['architecture']=='edge_dual')
    checks.append(dict(name='actual_edge_first_emission_target',passed=bool(nominal_edge['actual_first_wavelengths_nm']
        and all(abs(w-1030.)<=.05 for w in nominal_edge['actual_first_wavelengths_nm']))))
    checks.append(dict(name='frozen_sources_unchanged',passed=before=={str(p):sha(p) for p in sources}))
    result=dict(status='passed' if all(r['passed'] for r in checks+records+pumps) else 'failed',
        checks=checks,optical_modes=records,pump_thresholds=pumps,input_sha256=before,
        review_driver_sha256=sha(__file__),cold_relative_material_gap=ordered[1]['required_material_gain_cm1']/ordered[0]['required_material_gain_cm1']-1,
        spectroscopy_scope='Nominal flat scalar cross sections and300K reciprocity-paired constant-emission-envelope sensitivity; not measured broadband gain.',
        scope='Independent first-two own-C/KM/frequency/flux/Jones/polarization rebuild and all-retained-mode first shared-pump thresholds. Does not certify exhaustive Maxwell, thermal or dynamical stability.')
    output=ROOT/'results/current/strict_rerun_20261025/cross_audit_yag_triangular.json'
    b.save(output,result);print(result['status'],flush=True)
    if result['status']!='passed':raise RuntimeError('Independent triangular YAG audit mismatch')


if __name__=='__main__':main()
