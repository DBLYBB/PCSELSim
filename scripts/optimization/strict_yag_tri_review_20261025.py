"""Fresh physical readback and shared-pump review of triangular Yb:YAG roots.

中文：保存的每根复场均重建自身波长的C、原K/M、通量和Jones。
The optional 300K reciprocity pairing constrains absorption/emission ratios;
the emission envelope remains nominal, never an invented measured spectrum.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,hashlib
from pathlib import Path
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(Path(__file__).parent)]
import numpy as np
import matched300_yag_search as b
from yag_feasibility_study import stark_emission_to_absorption
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.polarization import stokes_maps
OUT=ROOT/'results/current/strict_rerun_20261025/YbYAG_triangular'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def controls_readback(folder):
    summary=read(folder/'summary.json');source=read(summary['input']);design=source['design']
    period_path=next(Path(p) for p in summary['source_sha256'] if p.endswith('outer_period_design.json'))
    period=read(period_path)['frozen_period_nm']
    checks=[]
    for result in summary['results']:
        order,grid,dz=result['D'],result['N'],result['dz_nm']
        if result['status']!='completed' or result['independent_branch_count']!=2:raise RuntimeError('Two independent controls required; merged branches have no valid gap')
        with np.load(folder/f'D{order}_N{grid}_dz{dz:g}.npz',allow_pickle=False) as z:
            for row in result['closed_modes']:
                name=row['name'];wave=row['wavelength_nm'];model,_,guide=b.optical('triangular',design,order,dz,wave,period,0)
                spec,_,p,mesh=b.finite('triangular',model,grid,8,False);field=z[name+'_field'];value=complex(row['delta_per_m'],row['alpha_field_per_m'])
                ce=float(np.linalg.norm(model.result.coupling_m-z[name+'_C'])/np.linalg.norm(model.result.coupling_m));er=b.relative_residual(p.center_operator(),field,value)
                beta=2*np.pi*model.result.effective_index/(wave*1e-9);beta0=4*np.pi/np.sqrt(3)/(period*1e-9);fr=float((beta**2-beta0**2)/(2*beta0)-value.real)
                up,down=model.side_power_losses(field);flux=abs(up+down+2*p.loss_budget(field)['transport_amplitude_loss_per_m']-2*value.imag)
                mode=b.as_mode('triangular',field,value,model.result,mesh,name);mode=replace(mode,eigen_residual=er)
                fresh,parts=b.mode_row('triangular',model,guide,spec,mode,wave,201);axis,fx,fy,*_=parts
                je=float(max(np.linalg.norm(fx-z[name+'_far_x'])/max(np.linalg.norm(fx),1e-30),np.linalg.norm(fy-z[name+'_far_y'])/max(np.linalg.norm(fy),1e-30)))
                gain_error=abs(fresh['required_material_gain_cm1']-row['required_material_gain_cm1'])
                passed=bool(ce<2e-9 and er<2e-7 and abs(fr)<.05 and flux<1e-4 and je<1e-9 and gain_error<1e-10)
                checks.append(dict(D=order,N=grid,dz_nm=dz,name=name,passed=passed,C_relative_error=ce,original_pencil_residual=er,true_frequency_residual_per_m=fr,
                    passive_flux_closure_per_m=flux,Jones_readback_relative_error=je,gain_readback_absolute_error_cm1=gain_error,
                    coefficient_points=mesh.points,x_node_extent_um=[float(mesh.x_um.min()),float(mesh.x_um.max())],y_node_extent_um=[float(mesh.y_um.min()),float(mesh.y_um.max())]))
                print('CONTROL_READBACK',order,grid,dz,name,passed,flush=True)
    b.save(folder/'readback_qa.json',dict(status='passed' if all(r['passed'] for r in checks) else 'failed',checks=checks,period_nm=period,
        nominal_aperture_um=[300.,300.],area_correction_applied=False,scope='Actual complex fields on every controlled mesh; no scalar area/gain correction and no claim of a complete new-grid spectrum.'))
    if not all(r['passed'] for r in checks):raise RuntimeError('Control original-equation readback failed')


def review(folder):
    plan=read(folder/'frozen_census.json');rows=read(folder/'closed_optical_modes.json')
    design,period,grid=plan['design'],plan['period_nm'],plan['grid']
    with np.load(folder/'closed_fields.npz',allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
    retained=[];duplicates=[];checks=[];models=[]
    for row in rows:
        if row['closed_TE_family']!=0:raise RuntimeError('TE family0 required')
        name=row['name'];wave=row['wavelength_nm'];model,_,guide=b.optical('triangular',design,32,1.,wave,period,0)
        spec,_,p,mesh=b.finite('triangular',model,grid,8,False)
        C=raw[name+'_C'];field=raw[name+'_field'];value=complex(row['delta_per_m'],row['alpha_field_per_m'])
        ce=float(np.linalg.norm(C-model.result.coupling_m)/np.linalg.norm(C))
        residual=b.relative_residual(p.center_operator(),field,value);budget=p.loss_budget(field)
        beta=2*np.pi*model.result.effective_index/(wave*1e-9);beta0=4*np.pi/np.sqrt(3)/(period*1e-9)
        frequency_residual=float((beta**2-beta0**2)/(2*beta0)-value.real)
        flat=field.reshape(6,-1);norm=np.sum(abs(flat)**2);Lup,Ldown=raw[name+'_Lup'],raw[name+'_Ldown']
        up,down=[float(2*np.einsum('ik,ij,jk->',flat.conj(),L,flat).real/norm) for L in (Lup,Ldown)]
        flux=abs(up+down+2*budget['transport_amplitude_loss_per_m']-2*value.imag)
        optical=float(np.linalg.norm((C-C.conj().T)/(2j)-Lup-Ldown)/max(np.linalg.norm(Lup+Ldown),1e-30))
        mode=b.as_mode('triangular',field,value,model.result,mesh,name);mode=replace(mode,eigen_residual=residual)
        fresh,parts=b.mode_row('triangular',model,guide,spec,mode,wave,201)
        axis,fx,fy,widths,qe,qg=parts
        je=float(max(np.linalg.norm(fx-raw[name+'_far_x'])/max(np.linalg.norm(fx),1e-30),np.linalg.norm(fy-raw[name+'_far_y'])/max(np.linalg.norm(fy),1e-30)))
        gain_error=abs(fresh['required_material_gain_cm1']-row['required_material_gain_cm1'])
        qe_old,qg_old=raw[name+'_qenergy'],raw[name+'_qgain']
        vg=b.c*qe_old.sum()/(1.8166*qg_old.sum())
        weight_error=float(np.max(abs(qg_old-b.c/(1.8166*vg)*qe_old)))
        passed=bool(ce<2e-9 and residual<2e-7 and abs(frequency_residual)<.05 and flux<1e-4 and optical<1e-8 and je<1e-9 and gain_error<1e-10 and weight_error<1e-13)
        checks.append(dict(name=name,passed=passed,C_relative_error=ce,original_pencil_residual=residual,
            true_frequency_residual_per_m=frequency_residual,passive_flux_closure_per_m=flux,optical_theorem_residual=optical,
            Jones_relative_readback_error=je,material_gain_readback_error_cm1=gain_error,gain_energy_identity_error=weight_error))
        if not passed:raise RuntimeError(f'Original physical readback failed: {checks[-1]}')
        peers=[old for old in retained if abs(old['wavelength_nm']-wave)<=1e-7]
        for old in peers:
            if np.linalg.norm(C-raw[old['name']+'_C'])>2e-9*max(np.linalg.norm(C),1.):raise RuntimeError('Mode identity requires matching same-frequency C')
        if not independent_eigenmode(value,field,[complex(x['delta_per_m'],x['alpha_field_per_m']) for x in peers],[raw[x['name']+'_field'] for x in peers]):
            duplicates.append(dict(name=name,peers=[x['name'] for x in peers]));continue
        retained.append(row)
        models.append(b.SpatialYbYAGModel(b.material(wave),raw[name+'_widths'],qe_old,row['total_power_loss_per_m'],up,down,vg,qg_old,raw[name+'_lateral']))
        print('TRI_READBACK',folder.name,name,row['required_material_gain_cm1'],flush=True)
    b.save(folder/'readback_qa.json',dict(status='passed',checks=checks,independent_retained_modes=len(retained),repeated_roots=duplicates,
        scope='Fresh own-wavelength C, original K/M, real frequency, port/transport flux, qg/qE identity and Jones readback.'))
    records=[]
    for label in ('nominal_flat_spectrum','reciprocity_paired_300K_constant_emission_envelope'):
        varied=[];materials={}
        for row,original in zip(retained,models):
            med=original.medium
            if label.startswith('reciprocity'):
                med=replace(med,pump_emission_cross_section_cm2=med.pump_absorption_cross_section_cm2*stark_emission_to_absorption(med.pump_wavelength_nm),
                    laser_absorption_cross_section_cm2=med.laser_emission_cross_section_cm2/stark_emission_to_absorption(med.laser_wavelength_nm))
            varied.append(replace(original,medium=med));fmax=med.pump_absorption_cross_section_cm2/(med.pump_absorption_cross_section_cm2+med.pump_emission_cross_section_cm2)
            materials[row['name']]=dict(medium=asdict(med),pump_only_inversion_ceiling=fmax,
                pump_limited_material_gain_ceiling_cm1=med.dopant_density_cm3*((med.laser_emission_cross_section_cm2+med.laser_absorption_cross_section_cm2)*fmax-med.laser_absorption_cross_section_cm2))
        system=b.SpatialMultimodeYbYAG(tuple(varied),tuple(r['name'] for r in retained));pumps={}
        for architecture in ('edge_dual','top_single'):
            threshold=b.shared_inversion_pump_threshold(system,b.PumpGeometryConfig(architecture,0.,300.,31))
            pumps[architecture]=dict(threshold=asdict(threshold),actual_winners=[next(r for r in retained if r['name']==n) for n in threshold.first_modes])
            print('TRI_PUMP',folder.name,label,architecture,threshold.first_threshold_internal_W,threshold.first_modes,flush=True)
        records.append(dict(label=label,materials=materials,pumps=pumps));b.save(folder/'spectroscopy_review.json',dict(status='running',records=records))
    b.save(folder/'spectroscopy_review.json',dict(status='completed',retained_independent_modes=len(retained),records=records,
        scope='Mode-specific wavelengths; 300K Stark reciprocity paired sensitivity with constant nominal emission envelope, not a measured broadband spectrum. Internal first-entry pump, x-z inversion with y averaged.'))
    ordered=sorted(retained,key=lambda r:r['required_material_gain_cm1']);b.save(folder/'merged_closed_optical_modes.json',ordered)
    nominal=records[0]['pumps'];winners={a:p['actual_winners'] for a,p in nominal.items()}
    fabrication=b.evaluate_fabrication(b.geometry('triangular',design).inclusions,period,b.TRIANGULAR_DIRECT,b.RULE)
    b.export_actual_geometry('triangular',design,period,fabrication,folder/'actual_geometry_nm.json')
    longitudinal=b.longitudinal_audit('triangular',design,period,1.);b.save(folder/'longitudinal_applicability.json',longitudinal)
    summary=dict(lattice='triangular',design=design,period_nm=period,side_um=300.,isolated_TE_count=1,
        included_closed_modes=len(retained),settings=dict(order=32,dz_nm=1.,tri_radius_cells=grid,tri_grid=grid),
        fields_file='closed_fields.npz',modes_file='merged_closed_optical_modes.json',actual_geometry_file='actual_geometry_nm.json',axis_key='angle_deg',
        pump_results=[dict(architecture=a,threshold=p['threshold'],cw=[]) for a,p in nominal.items()],actual_winners=winners,
        recommended_pump_architecture='edge_dual',additional_reciprocity_review=records[1],
        fabrication=fabrication,longitudinal_applicability=longitudinal,closed_mode_identity_audit=dict(independent=len(retained),duplicates=duplicates),
        internal_intensity_loss_cm1=.1,common_analysis_band_nm=[1000,1060],assumptions='same500nm Yb:YAG /120nm passive air PC /2umSiO2 /16pairDBR; nominal10at% constant sigma, conditional feasibility only',
        scope='Own-wavelength frequency-closed scalar TE0 CWT; complete coarse QR plus bounded staged fine spectrum, lowest-rank and delta-cluster/Gamma closure; shared x-z inversion with y averaged. No full-wave, thermal, fabrication or nonlinear stability certification.')
    b.save(folder/'merged_summary.json',summary);b.save(folder/'summary.json',summary)
    first=nominal['edge_dual']['actual_winners'][0] if nominal['edge_dual']['actual_winners'] else None
    wide=None
    if first:
        other=min((r for r in ordered if r['name']!=first['name']),key=lambda r:r['required_material_gain_cm1']);gap=other['required_material_gain_cm1']/first['required_material_gain_cm1']-1
        beam=first['beam'];passes=dict(bright_center=beam['center_to_peak']>=.98,axis_ratio=beam['ellipticity']<=1.05,radial_engineering_line=beam['radial_anisotropy_l2']<=.10,
            material_gap=gap>=.1,wavelength=abs(first['wavelength_nm']-1030.)<=.05)
        # Verify that a bright central plot is not merely an angular crop of a
        # stronger remote lobe, within the larger resolved paraxial window.
        # 检查更大且已解析角窗；不把局部亮斑误称整个半球输出结论。
        opt,_,_=b.optical('triangular',design,32,1.,first['wavelength_nm'],period,0)
        _,_,_,mesh=b.finite('triangular',opt,grid,8,False)
        mode=b.as_mode('triangular',raw[first['name']+'_field'],complex(first['delta_per_m'],first['alpha_field_per_m']),opt.result,mesh,first['name'])
        ff=b.triangular_vector_far_field(mode,first['wavelength_nm'],3.,301);power=abs(ff.field_x)**2+abs(ff.field_y)**2
        axis=ff.angle_deg;weights=np.ones(len(axis));weights[[0,-1]]=.5;measure=weights[:,None]*weights[None,:]
        core=(abs(axis[:,None])<=1.)&(abs(axis[None,:])<=1.)
        wide=dict(mode=first['name'],wavelength_nm=first['wavelength_nm'],evaluated_window_deg=[-3.,3.],source_alias_free_view_deg=ff.alias_free_square_view_deg,
            center_to_peak_in_larger_window=float(power[len(axis)//2,len(axis)//2]/power.max()),fraction_in_1deg_square_of_3deg_window=float(np.sum(power*measure*core)/np.sum(power*measure)),
            scope='Resolved paraxial angular-sheet transform; window comparison only, not full-hemisphere energy or full-vector Maxwell far field.')
        if ff.alias_free_square_view_deg<3.:raise RuntimeError('Wide-window audit is not sampling resolved')
        passes['bright_center_in_resolved_3deg_window']=wide['center_to_peak_in_larger_window']>=.98
        np.savez_compressed(folder/'wide_window_Jones.npz',angle_deg=axis,far_x=ff.field_x,far_y=ff.field_y)
        b.save(folder/'wide_window_audit.json',wide)
    else:other=None;gap=None;passes={}
    selected=dict(status='completed',id='YbYAG_triangular',lattice='triangular',TE_family=0,design=design,period_nm=period,D=32,dz_nm=1.,N=grid,
        nominal_aperture_um=[300.,300.],first_mode=first['name'] if first else None,first=first,lowest_cold_root=ordered[0],closest_competitor=other,
        relative_material_gap=gap,first_mode_wavelength_nm=first['wavelength_nm'] if first else None,
        g1_cm1=first['required_material_gain_cm1'] if first else None,g2_cm1=other['required_material_gain_cm1'] if other else None,
        beam=first['beam'] if first else None,polarization=first['polarization'] if first else None,
        wider_angular_window_audit=wide,
        engineering_target_pass=bool(passes and all(passes.values())),engineering_checks=passes,
        engineering_line_note='radialL2<=.10 and axis<=1.05 are this-run design objectives, not physical feasibility boundaries; retain continuous Pareto tradeoffs.',
        independent_closed_modes=len(retained),summary_path=str(folder/'merged_summary.json'),merged_summary=str(folder/'merged_summary.json'),
        closed_modes_path=str(folder/'merged_closed_optical_modes.json'),fields_path=str(folder/'closed_fields.npz'),geometry_path=str(folder/'actual_geometry_nm.json'),
        configuration_path=str(folder/'selected_configuration.json'),spectroscopy_review_path=str(folder/'spectroscopy_review.json'),readback_path=str(folder/'readback_qa.json'),
        mode_keys=[r['name'] for r in ordered],axis_key='angle_deg',field_key_pattern='{mode}_field',Jones_x_pattern='{mode}_far_x',Jones_y_pattern='{mode}_far_y',C_key_pattern='{mode}_C',
        field_mapping=dict(near_field='{name}_field',coupling='{name}_C',radiation_x='{name}_radiation_x',radiation_y='{name}_radiation_y',
            gain_weights='{name}_qgain',energy_weights='{name}_qenergy',z_widths='{name}_widths',lateral_x_weights='{name}_lateral',
            far_x='{name}_far_x',far_y='{name}_far_y',angular_axis='angle_deg',coefficient_x_um='x_um',coefficient_y_um='y_um',
            description='Each root keeps its own actual wavelength/C and qgain/qenergy; all saved Jones components share one common scale.'),
        fields_sha256=sha(folder/'closed_fields.npz'),closed_modes_sha256=sha(folder/'merged_closed_optical_modes.json'),
        driver_path=str(Path(__file__)),driver_sha256=sha(__file__),scope=summary['scope'])
    config=dict(design=design,period_nm=period,aperture_um=[300.,300.],gain_thickness_nm=500.,passive_pc_thickness_nm=120.,silica_spacer_nm=2000.,DBR_pairs=16,
        optical_settings=summary['settings'],material_protocols=[dict(label=r['label'],materials=r['materials']) for r in records],
        target_first_emission_nm=1030.,target_tolerance_nm=.05,assumptions=summary['scope'],material_note='Reciprocity sensitivity keeps nominal sigma_emission=2e-20cm2 constant; not a measured full gain spectrum.')
    b.save(folder/'selected_configuration.json',config);b.save(folder/'selected_case.json',selected)
    print('TRI_CASE',folder.name,selected['first_mode'],selected['g1_cm1'],selected['relative_material_gap'],selected['engineering_target_pass'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',type=Path);p.add_argument('--controls',action='store_true');args=p.parse_args()
    (controls_readback if args.controls else review)(args.case.resolve())
