"""Read-only independent review of the new YAG feasibility study.

中文：不修改仿真、配置和原始结果；只写审计JSON。独立检查保存样点的功率代数、
原300um锚点、速率方程粒子账，以及尺寸模式的原K/M残差。共享生产模型的回放
属于重建验证，不替代实测光谱、全矢量电磁或实验验证。
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/optimization')]
import numpy as np
from scipy.constants import c,h
from scipy.sparse.linalg import splu
import yag_feasibility_study as study
from pcselsim.ybyag_multimode import solve_multimode_spatial_ybyag
from pcselsim.ybyag_pump_geometry import propagate_geometry_pump
from pcselsim.custom_analysis import finite_area_eigenproblem
from pcselsim.triangular_finite import triangular_finite_eigenproblem
from pcselsim.finite_discretization import relative_residual

DATA=ROOT/'results/current/yag_feasibility_20261025'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/current/physics_audit_20261025/new_yag_study_review.json')
    p.add_argument('--skip-replay',action='store_true')
    p.add_argument('--include-pulse',action='store_true')
    args=p.parse_args();started=time.monotonic();checks=[];details={};snapshots={}
    def record(name,actual,limit):
        checks.append(dict(name=name,actual=float(actual),limit=float(limit),passed=bool(actual<=limit)))
    def read(path):
        snapshots[str(path.relative_to(ROOT))]=digest(path)
        return json.loads(path.read_text(encoding='utf-8'))
    manifest=read(study.MANIFEST)
    selected={x['lattice']:x for x in manifest['cases'] if x['id'].startswith('YbYAG_')}
    system_by_lattice={};anchor={};cw_info=[];cw_count=0;max_res={k:0. for k in ['energy_W','pump_W','population_per_s','complementarity']}
    for lattice in ('square','triangular'):
        system,summary,rows=study.load_saved_system(lattice);system_by_lattice[lattice]=system
        directory=(ROOT/selected[lattice]['summary_path']).parent
        read(directory/'summary.json');read(directory/'closed_optical_modes.json');snapshots[str((directory/'closed_fields.npz').relative_to(ROOT))]=digest(directory/'closed_fields.npz')
        record(lattice+' supplies 32 distinct CW modes',abs(len(system.modes)-32),0)
        for model,row in zip(system.modes,rows):
            qe,qg=model.optical_energy_fractions,model.optical_gain_weights
            vg=c*np.sum(qe)/(model.medium.refractive_index*np.sum(qg))
            record(lattice+' '+row['name']+' own vg from qE/qg',abs(vg/model.group_velocity_m_s-1),2e-13)
            record(lattice+' '+row['name']+' cellwise energy/gain identity',np.max(abs(qg-c/(model.medium.refractive_index*vg)*qe)),2e-13)
            record(lattice+' '+row['name']+' physical gain volume',abs(model.gain_volume_m3/(300e-6*300e-6*.5e-6)-1),2e-13)
        anchor[lattice]={r['architecture']:r for r in summary['pump_results']}
        for architecture in ('edge_dual','top1','top20'):
            path=DATA/'cw'/f'{lattice}_{architecture}.json';x=read(path)
            old_arch='top_single' if architecture=='top1' else architecture
            old=anchor[lattice].get(old_arch)
            if old is None and architecture=='top20':
                old=next((r for r in summary['pump_results'] if r['architecture'] in ('top20','top_20pass')),None)
            if old is None:
                old=next((r for r in summary['pump_results'] if ('top' in r['architecture'] and (r.get('top_passes')==20))),None) if architecture=='top20' else None
            th=x['threshold']['first_threshold_internal_W']
            if old is not None:
                record(lattice+' '+architecture+' 300um threshold anchor',abs(th/old['threshold']['first_threshold_internal_W']-1),2e-11)
            if architecture=='top20' and old is None:
                details.setdefault('anchor_notes',[]).append(lattice+' top20 has no direct old named architecture; threshold is fresh ideal relay extension')
            record(lattice+' '+architecture+' all 13 CW samples complete',abs(len([v for v in x['points'] if v['status']=='completed'])-13),0)
            for point in x['points']:
                if point['status']!='completed':continue
                cw_count+=1;budget=point['energy_budget_W'];pump=point['internal_pump_W'];up=point['upward_output_W']
                record(f'{path.name} R{point["ratio"]} pump definition',abs(pump-point['ratio']*th),1e-11*max(th,1.))
                record(f'{path.name} R{point["ratio"]} modal power sum',abs(sum(point['mode_upward_W'].values())-up),2e-12)
                pump_error=budget['incident_internal_pump_W']-budget['absorbed_pump_W']-budget['pump_escape_W']-budget['pump_relay_loss_W']
                energy_error=(budget['absorbed_pump_W']-budget['pump_quantum_defect_reference_W']-
                    budget['intermode_stimulated_thermalization_W']-budget['fluorescence_outside_modes_W']-
                    budget['upward_output_W']-budget['downward_output_W']-budget['other_cavity_loss_W']-
                    budget['stored_excitation_plus_photon_energy_rate_W'])
                record(f'{path.name} R{point["ratio"]} independent pump boundary sum',abs(pump_error),1e-10*max(pump,1e-3))
                record(f'{path.name} R{point["ratio"]} independent energy sum',abs(energy_error),1e-10*max(budget['absorbed_pump_W'],1e-3))
                record(f'{path.name} R{point["ratio"]} nonnegative laser output',max(0.,-up,-min(point['mode_upward_W'].values())),1e-15)
                record(f'{path.name} R{point["ratio"]} total power bound',max(0.,up-budget['absorbed_pump_W']*940./1030.),2e-8)
                if point['ratio']<1.:record(f'{path.name} R{point["ratio"]} beta0 subthreshold',up,1e-14)
                max_res['energy_W']=max(max_res['energy_W'],abs(energy_error));max_res['pump_W']=max(max_res['pump_W'],abs(pump_error));max_res['population_per_s']=max(max_res['population_per_s'],point['population_residual_per_s']);max_res['complementarity']=max(max_res['complementarity'],point['complementarity_residual'])
                if old is not None:
                    peers=[q for q in old['cw'] if q['status']=='completed' and abs(q['ratio']-point['ratio'])<1e-12]
                    if peers:
                        record(f'{path.name} R{point["ratio"]} old CW power anchor',abs(sum(peers[0]['upward_W'].values())-up),2e-10)
            cw_info.append({'lattice':lattice,'architecture':architecture,'threshold_internal_W':th,'points':len(x['points']),'wavelength_range_nm':[min(r['wavelength_nm'] for r in rows),max(r['wavelength_nm'] for r in rows)]})
    details['CW']={'points_checked':cw_count,'cases':cw_info,'maximum_stored_residuals':max_res}
    # Fresh six-case replay; derive stimulated populations and photon balances
    # independently from returned f/S without calling the production RHS helper.
    replay=[]
    if not args.skip_replay:
        for lattice,system in system_by_lattice.items():
            for architecture in ('edge_dual','top1','top20'):
                x=read(DATA/'cw'/f'{lattice}_{architecture}.json');point=next(q for q in x['points'] if q['ratio']==2.)
                pump=study.pump_config(architecture,300.,point['internal_pump_W'])
                state=solve_multimode_spatial_ybyag(system,pump)
                record(lattice+' '+architecture+' fresh CW output replay',abs(state.upward_output_W.sum()-point['upward_output_W']),2e-9)
                f=state.excited_fraction;photons=state.photon_density_m3;first=system.modes[0];medium=first.medium
                widths=np.asarray(first.cell_widths_m);volume=np.broadcast_to(widths/widths.sum()/f.shape[0],f.shape)
                weights=np.array([np.asarray(m.lateral_energy_fractions)[:,None]*np.asarray(m.optical_gain_weights)[None,:] for m in system.modes])
                nt=medium.dopant_density_cm3*1e6;tau=medium.upper_state_lifetime_ms*1e-3
                ap=medium.pump_absorption_cross_section_cm2*1e-4;ep=medium.pump_emission_cross_section_cm2*1e-4
                el=np.array([m.medium.laser_emission_cross_section_cm2*1e-4 for m in system.modes])[:,None,None]
                al=np.array([m.medium.laser_absorption_cross_section_cm2*1e-4 for m in system.modes])[:,None,None]
                velocity=np.array([m.group_velocity_m_s for m in system.modes]);loss=np.array([m.total_power_loss_per_m for m in system.modes])
                transport=propagate_geometry_pump(first,pump,f);pump_energy=h*c/(medium.pump_wavelength_nm*1e-9)
                flux=transport.intensity_W_cm2*1e4/pump_energy;local_gain=nt*((el+al)*f[None,:,:]-al)
                gain=np.sum(weights*local_gain,axis=(1,2))
                df=ap*flux*(1-f)-ep*flux*f-f/tau-np.sum(velocity[:,None,None]*photons[:,None,None]*weights*local_gain/volume,axis=0)/nt
                ds=velocity*(gain-loss)*photons
                volume_m3=first.medium.pumped_area_um2*1e-12*widths.sum()
                particle_lhs=volume_m3*(nt*np.sum(volume*df)+np.sum(ds))
                particle_rhs=transport.absorbed_power_W/pump_energy-volume_m3*nt*np.sum(volume*f)/tau-volume_m3*np.sum(velocity*loss*photons)
                particle_scale=max(transport.absorbed_power_W/pump_energy,1.)
                record(lattice+' '+architecture+' independent excitation balance',abs(particle_lhs-particle_rhs)/particle_scale,5e-11)
                record(lattice+' '+architecture+' independent population residual',np.max(abs(df)),2e-5)
                energies=h*c/(np.array([m.medium.laser_wavelength_nm for m in system.modes])*1e-9)
                independently_up=energies*volume_m3*velocity*photons*np.array([m.upward_power_loss_per_m for m in system.modes])
                record(lattice+' '+architecture+' photon-count output conversion',np.max(abs(independently_up-state.upward_output_W)),2e-13)
                replay.append({'lattice':lattice,'architecture':architecture,'power_W':point['internal_pump_W'],'particle_relative_residual':float(abs(particle_lhs-particle_rhs)/particle_scale),'population_residual_per_s':float(np.max(abs(df))),'active_modes':list(state.active_modes)})
    details['fresh_CW_replay']=replay
    # Reciprocity calculations and a strict analytic lifetime scaling test.
    sensitivity=[]
    lower=np.array([0.,581.,619.,786.]);upper=np.array([10327.,10634.,10927.]);kT=.6950348004861274*300.
    def reciprocal(wavelength):return np.exp(-lower/kT).sum()/np.exp(-(upper-upper.min())/kT).sum()*np.exp((upper.min()-1e7/wavelength)/kT)
    for lattice in ('square','triangular'):
        x=read(DATA/'sensitivity'/f'{lattice}.json');byname={r['label']:r for r in x['records']}
        record(lattice+' 13 material cases',abs(len(byname)-13),0)
        for row in x['records']:
            med=row['medium']
            if row['label']!='nominal':
                record(lattice+' '+row['label']+' pump reciprocal cross-section pair',abs(med['pump_emission_cross_section_cm2']/med['pump_absorption_cross_section_cm2']/reciprocal(med['pump_wavelength_nm'])-1),3e-13)
                record(lattice+' '+row['label']+' laser reciprocal cross-section pair',abs(med['laser_emission_cross_section_cm2']/med['laser_absorption_cross_section_cm2']/reciprocal(med['laser_wavelength_nm'])-1),3e-13)
            for architecture,threshold in row['pumps'].items():
                value=threshold['first_threshold_internal_W']
                record(lattice+' '+row['label']+' '+architecture+' nonnegative threshold',0. if value is None or value>0 else 1.,0.)
        for architecture in ('edge_dual','top1','top20'):
            nominal=next(v['threshold_internal_W'] for v in cw_info if v['lattice']==lattice and v['architecture']==architecture)
            record(lattice+' '+architecture+' nominal sensitivity anchor',abs(byname['nominal']['pumps'][architecture]['first_threshold_internal_W']/nominal-1),2e-11)
            reference=byname['reciprocity_300K']['pumps'][architecture]['first_threshold_internal_W']
            for label,tau in [('reciprocity_tau_0p90ms',.9),('reciprocity_tau_1p00ms',1.)]:
                value=byname[label]['pumps'][architecture]['first_threshold_internal_W']
                record(lattice+' '+architecture+' '+label+' inverse lifetime scaling',abs(value/reference/(.95/tau)-1),2e-7)
        sensitivity.append({'lattice':lattice,'cases_checked':len(x['records']),'reciprocity_scope':'ratios only, not a measured line shape; all nonnominal cases use the reciprocal-pair baseline'})
    details['material_sensitivity']=sensitivity
    # Snapshot completed sizes only; other jobs may continue to create files.
    size_files=[]
    for path in sorted((DATA/'sizes').glob('*.json')):
        try:x=read(path)
        except json.JSONDecodeError:continue
        if x.get('status')!='completed' or not x.get('modes'):continue
        lattice=x['lattice'];nominal=read(ROOT/selected[lattice]['summary_path'])
        record(path.name+' fixed physical period',abs(x['period_nm']/nominal['period_nm']-1),1e-14)
        record(path.name+' unchanged design dictionary',0 if x['design']==nominal['design'] else 1,0)
        record(path.name+' residual-qualified saved roots',max(r['eigen_residual'] for r in x['modes']),2e-7)
        record(path.name+' real-frequency closure',max(abs(r['frequency_residual_per_m']) for r in x['modes']),.05)
        for row in x['modes']:
            record(path.name+' '+row['name']+' material-loss conversion',abs(row['material_gain_cm1']*100*row['gain_weight']-row['total_power_loss_per_m']),2e-8)
            record(path.name+' '+row['name']+' radiative loss below total',max(0,row['up_power_loss_per_m']+row['down_power_loss_per_m']-row['total_power_loss_per_m']),2e-8)
        if 'target_band_mode_names' in x:
            names=[r['name'] for r in x['modes'] if 1029<=r['wavelength_nm']<=1031]
            record(path.name+' explicit target-band membership',0 if names==x['target_band_mode_names'] else 1,0)
        size_files.append((path,x))
    details['size_snapshot']={'completed_files_checked':len(size_files),'sampled_modes_checked':sum(len(x['modes']) for _,x in size_files),'status_note':'Jobs can add new files; this audit covers only the listed snapshot. Completed means the job produced modes, not complete fine-spectrum enumeration.','files':[str(p.relative_to(ROOT)) for p,_ in size_files]}
    # Choose small/coarse, near-limit/fine and triangular files for direct K/M.
    wanted=['square_L30_D12_dz2_N25_K4.json','square_L225_D24_dz1_N41_K4.json','triangular_L175_D24_dz1_N26_K4.json','triangular_L300_D12_dz2_N15.json']
    spot=[]
    for filename in wanted:
        matching=[(p,x) for p,x in size_files if p.name==filename]
        if not matching:continue
        path,x=matching[0];raw_path=path.with_suffix('.npz');snapshots[str(raw_path.relative_to(ROOT))]=digest(raw_path)
        with np.load(raw_path,allow_pickle=False) as raw:
            for row in [x['modes'][0],x['modes'][-1]]:
                model,_,guide=study.optical_backend.optical(x['lattice'],x['design'],x['order'],x['dz_nm'],row['wavelength_nm'],x['period_nm'])
                spec,problem,mesh=study.make_problem(x['lattice'],model,x['side_um'],x['grid'])
                actual_length=spec.domain_um if x['lattice']=='square' else 2*spec.radius_um
                record(path.name+' actual aperture in finite operator',abs(actual_length-x['side_um']),1e-12)
                field=raw[row['name']+'_field'];value=complex(row['delta_per_m'],row['alpha_field_per_m'])
                links=splu(problem.mass.tocsc()).solve(field.ravel())
                lhs=problem.stiffness@links;rhs=value*(problem.mass@links)
                residual=np.linalg.norm(lhs-rhs)/max(np.linalg.norm(lhs),np.linalg.norm(rhs),1e-30)
                record(path.name+' '+row['name']+' original K/M residual',residual,2e-7)
                beta=2*np.pi*model.result.effective_index/(row['wavelength_nm']*1e-9);beta0=(2*np.pi if x['lattice']=='square' else 4*np.pi/np.sqrt(3))/(x['period_nm']*1e-9)
                frequency_residual=(beta*beta-beta0*beta0)/(2*beta0)-value.real
                record(path.name+' '+row['name']+' independent physical frequency',abs(frequency_residual),.05)
                widths,qe,qg=study.optical_backend.active_weights(model.result,guide,24)
                record(path.name+' '+row['name']+' freshly rebuilt gain weight',abs(qg.sum()/row['gain_weight']-1),2e-11)
                spot.append({'file':str(path.relative_to(ROOT)),'mode':row['name'],'actual_aperture_um':actual_length,'original_K_M_relative_residual':float(residual),'frequency_residual_per_m':float(frequency_residual)})
    details['original_pencil_spot_checks']=spot
    # AST, not filename grep: ensure no direct fixed-300 finite/seed factory.
    source_path=ROOT/'scripts/optimization/yag_feasibility_study.py';source=source_path.read_text(encoding='utf-8');tree=ast.parse(source)
    calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))]
    forbidden=[s for s in calls if s in ('seed_pool','finite')]
    record('new size driver avoids fixed-300 seed/finite helper',len(forbidden),0)
    details['algorithm_scope']={'forbidden_calls':forbidden,'inverse_iteration':'Accepts only original physical-center eigenpair residual; interpolation supplies seeds only. Full small-grid spectrum does not prove complete fine-grid or continuous spectrum.','spectral_scope':'Flat nominal cross sections over 1000-1060nm challenge. Target-band curves separately restrict 1029-1031nm. Neither is a measured Yb:YAG gain spectrum.','discretization_scope':'Size points retain their D,dz,N metadata; rank selection is not proof of convergence, particularly near inversion ceiling.'}
    pulse=DATA/'pulsed/pulse_energy_scan.json'
    if args.include_pulse and pulse.exists():
        x=read(pulse);extension=read(DATA/'pulsed/short_pulse_extension.json');qa=read(DATA/'pulsed/pulse_limit_audit.json')
        pulse_summary=read(DATA/'pulsed/summary.json');physical=read(DATA/'pulsed/physical_mode_inputs.json')
        responses=x['cases']+extension['cases'];validation=qa['cases']
        record('124 pulsed response cases',abs(len(responses)-124),0)
        record('16 pulsed validation records',abs(len(validation)-16),0)
        record('pulse response IDs unique',len(responses)-len(set(r['id'] for r in responses)),0)
        details['pulse_scope']=x.get('scope');details['pulse_response_cases_checked']=len(responses);details['pulse_validation_cases_checked']=len(validation)
        for lattice,system in system_by_lattice.items():
            model=system.modes[system.names.index('TE0_pole00')];saved=physical['models'][lattice]
            record(lattice+' pulsed input gain volume',abs(saved['gain_volume_m3']/model.gain_volume_m3-1),2e-13)
            record(lattice+' pulsed input qgain',np.max(abs(np.array(saved['qgain'])-model.optical_gain_weights)),2e-13)
        maximum_absorbed_relative_error=0.;long_limit=[]
        for row in responses+validation:
            met=row['metrics'];pin=met['external_pump_J'];pulse_config=row['pulse']
            boundary_error=pin-met['before_gain_loss_J']-met['absorbed_pump_J']-met['escaped_pump_J']-met['relay_loss_J']
            stored=met['stored_effective_energy_final_J']-met['stored_effective_energy_initial_J']
            energy_error=(met['absorbed_pump_J']-met['quantum_defect_J']-met['fluorescence_J']-
                met['upward_output_J']-met['downward_output_J']-met['other_cavity_loss_J']-stored)
            record(row['id']+' pulsed first-entry boundary energy',abs(boundary_error),1e-10*max(pin,1e-9))
            record(row['id']+' pulsed independent stored-energy sum',abs(energy_error),1e-8*max(met['absorbed_pump_J'],1e-14))
            record(row['id']+' external-coupling loss',abs(met['before_gain_loss_J']-pin*(1-pulse_config['coupling_efficiency'])),1e-11*max(pin,1e-9))
            record(row['id']+' energy versus duration/peak-power',abs(met['pump_peak_external_W']*pulse_config['duration_s']-pin),1e-11*max(pin,1e-9))
            model=system_by_lattice[row['lattice']].modes[0];med=model.medium;nt=med.dopant_density_cm3*1e6
            sigma_a=med.laser_absorption_cross_section_cm2*1e-4;sigma_e=med.laser_emission_cross_section_cm2*1e-4
            fth=(model.total_power_loss_per_m/(model.optical_gain_weights.sum()*nt)+sigma_a)/(sigma_e+sigma_a)
            ceiling=med.pump_absorption_cross_section_cm2/(med.pump_absorption_cross_section_cm2+med.pump_emission_cross_section_cm2)
            transmission=np.exp(-nt*(med.pump_absorption_cross_section_cm2*(1-fth)-med.pump_emission_cross_section_cm2*fth)*1e-4*model.cell_widths_m.sum())
            absorbed_fraction=pulse_config['coupling_efficiency']*(1-transmission)*sum((pulse_config['return_efficiency']*transmission)**i for i in range(pulse_config['top_passes']))
            external_threshold=nt*model.gain_volume_m3*fth*h*c/(med.pump_wavelength_nm*1e-9)/(med.upper_state_lifetime_ms*1e-3)/absorbed_fraction
            record(row['id']+' independent uniform-CW threshold',abs(external_threshold/met['uniform_CW_threshold']['external_CW_threshold_W']-1),2e-11)
            record(row['id']+' physical inversion upper bound',max(0,met['maximum_inversion_fraction']-ceiling),1e-8)
            record(row['id']+' nonnegative upward energy',max(0,-met['upward_output_J']),1e-15)
            record(row['id']+' pulsed energy extraction bound',max(0,met['upward_output_J']-met['absorbed_pump_J']*med.pump_wavelength_nm/med.laser_wavelength_nm-met['stored_effective_energy_initial_J']),1e-10)
            maximum_absorbed_relative_error=max(maximum_absorbed_relative_error,abs(energy_error)/max(abs(met['absorbed_pump_J']),1e-30))
            if row in responses and row['duration_s']>=.005 and row['power_to_uniform_CW_threshold']>=2:
                error=abs(met['last_tenth_on_average_upward_W']/met['uniform_beta0_CW_upward_W']-1)
                record(row['id']+' long-pulse late-CW limit',error,2e-5);long_limit.append(error)
        byid={r['id']:r for r in validation}
        integrator_difference=abs(byid['integrator_LSODA']['metrics']['upward_output_J']/byid['integrator_Radau']['metrics']['upward_output_J']-1)
        record('independent stiff integrator output comparison',integrator_difference,1e-3)
        details['pulse_validation']={'maximum_relative_energy_error_against_absorbed_energy':maximum_absorbed_relative_error,
            'integrator_relative_output_difference':integrator_difference,'long_pulse_CW_limit_cases':len(long_limit),
            'scope_note':'Uniform single saved mode; finite-beta seed, finite pulse duration and assumed eta=.7. Not the 32-mode spatial CW model, dynamical stability, damage limit or thermal validation. Sparse plotted traces cannot determine MHz frequencies or output energy by trapezoid integration.'}
    else:details['pulse_scope']='Not finalized/reviewed in this snapshot; pulse agent still owns its single-mode limitation and energy audit.'
    mask_review=ROOT/'results/current/physics_audit_20261025/triangular_square_mask_review.json'
    if mask_review.exists():
        mask=read(mask_review);details['triangular_boundary_mask']={'source':str(mask_review.relative_to(ROOT)),
            'gain_relative_change_N26_vs_N25':mask.get('N26_gain_relative_to_N25'),'scope':'Boundary alignment diagnostic, not a threshold rescaling or convergence certificate.'}
    grid_challenge=ROOT/'results/current/physics_audit_20261025/triangular_nonaligned_grid_challenge/summary.json'
    if grid_challenge.exists():
        ch=read(grid_challenge);details['nonaligned_grid_challenge']={'source':str(grid_challenge.relative_to(ROOT)),
            'results':[{k:r[k] for k in ('N','name','frozen_C_material_gain_estimate_cm1','gain_relative_difference','original_K_M_relative_residual')} for r in ch['results'] if r['status']=='completed'],
            'scope':'Frozen own D32 C and reference wavelength, no frequency closure. No complete fine-spectrum or beam robustness claim.'}
    closed_challenge=grid_challenge.with_name('summary_closed.json')
    if closed_challenge.exists():
        closed=read(closed_challenge);records=closed['results']
        record('N64 two tracked modes completed real-frequency closure',abs(len([r for r in records if r['status']=='completed'])-2),0)
        for r in records:
            if r['status']!='completed':continue
            record('N64 '+r['name']+' closed original K/M residual',r['original_K_M_relative_residual'],2e-7)
            record('N64 '+r['name']+' closed physical frequency residual',abs(r['frequency_residual_per_m']),.05)
            record('N64 '+r['name']+' closed fixed period',abs(r['period_nm']/selected['triangular']['case_inputs']['actual_period_nm']-1) if 'case_inputs' in selected['triangular'] else abs(r['period_nm']/json.loads((ROOT/selected['triangular']['summary_path']).read_text())['period_nm']-1),1e-14)
            record('N64 '+r['name']+' closed gain conversion',abs(r['material_gain_cm1']*100*r['gain_weight']-(2*r['alpha_field_per_m']+10)),2e-8)
        details['closed_nonaligned_grid_challenge']={'source':str(closed_challenge.relative_to(ROOT)),
            'tracked_two_mode_material_gap':closed.get('tracked_two_mode_material_gap'),
            'results':[{k:r[k] for k in ('name','material_gain_cm1','gain_relative_to_N52','wavelength_nm')} for r in records if r['status']=='completed'],
            'scope':'Two individually reclosed branches at fixed period, D32 dz1 N64. Not a complete fine spectrum or beam robustness certification.'}
    quality=DATA/'necessary_Q.json'
    if quality.exists():
        q=read(quality);cap=(.7e-20/(.7e-20+.1e-20))
        for row in q['records']:
            ceiling=row['dopant_density_cm3']*((2e-20+1.26e-21)*cap-1.26e-21)
            required=2*np.pi*q['group_index']/(q['wavelength_nm']*1e-9*q['full_mode_gain_weight']*100*ceiling)
            record('necessary Q nt='+str(row['dopant_density_cm3'])+' independent gain ceiling',abs(ceiling/row['pump_limited_gain_ceiling_cm1']-1),2e-13)
            record('necessary Q nt='+str(row['dopant_density_cm3'])+' omega-tau loss relation',abs(required/row['necessary_loaded_Q']-1),2e-13)
        details['necessary_Q_scope']='Analytic necessary passive-loss Q bound at assumed ng/qg and nominal scalar spectroscopy; excludes Yb absorption already present in net-gain term, pump coupling, actual mode volume/design and thermal/damage validation.'
    changed=[s for s,d in snapshots.items() if not (ROOT/s).exists() or digest(ROOT/s)!=d]
    record('snapshot input files unchanged during audit',len(changed),0)
    result={'status':'passed_for_reviewed_snapshot_with_model_limits' if all(q['passed'] for q in checks) else 'findings_require_review','checks_passed':sum(q['passed'] for q in checks),'checks_total':len(checks),'checks':checks,'details':details,'input_sha256':snapshots,'inputs_changed_during_audit':changed,'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),source_path,ROOT/'scripts/diagnostics/plot_yag_feasibility.py']},'elapsed_seconds':time.monotonic()-started,'claim_boundary':'Data arithmetic, source scope, original-pencil spot checks and same-model replay only. No independent 3D Maxwell, measured spectroscopy or microcavity experiment validation.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    failures=[q for q in checks if not q['passed']]
    print(json.dumps({'status':result['status'],'checks':f"{result['checks_passed']}/{len(checks)}",'CW_points':cw_count,'size_files':len(size_files),'K_M_spot_checks':len(spot),'failed_checks':failures,'elapsed_seconds':result['elapsed_seconds']},ensure_ascii=False))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
