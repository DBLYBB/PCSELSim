"""Read-only optical review and conditional spectroscopy sensitivity.

中文：只回读新跑出的复场，不修改光学本征场；材料倒易配对不是实测完整谱。
The mode-specific wavelength is retained in every gain/pump calculation.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts'), str(Path(__file__).parent)]
import numpy as np
import matched300_yag_search as backend
from yag_feasibility_study import stark_emission_to_absorption


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def saved_system(folder):
    """Recover vg with the same nondispersive gain-core energy identity.

    中文：逐 z 单元核对 qg=(c/n_core/vg)qE，不独立调整群折射率/增益权重。
    """
    rows = read(folder/'closed_optical_modes.json')
    with np.load(folder/'closed_fields.npz', allow_pickle=False) as raw:
        arrays = {key:raw[key].copy() for key in raw.files}
    added, rejected = [], []
    extra=folder/'spectral_challenges'
    if (extra/'closed_optical_modes.json').exists():
        with np.load(extra/'closed_fields.npz', allow_pickle=False) as raw:
            for row in read(extra/'closed_optical_modes.json'):
                name=row['name']; peers=[old for old in rows
                    if old['closed_TE_family']==row['closed_TE_family']
                    and abs(old['wavelength_nm']-row['wavelength_nm'])<=1e-7]
                independent=backend.independent_eigenmode(complex(row['delta_per_m'],row['alpha_field_per_m']),
                    raw[name+'_field'],[complex(p['delta_per_m'],p['alpha_field_per_m']) for p in peers],
                    [arrays[p['name']+'_field'] for p in peers])
                if not independent:
                    rejected.append(dict(name=name,peers=[p['name'] for p in peers],
                        reason='same closed complex root and no additional envelope-span dimension'))
                    continue
                rows.append(row);added.append(name)
                arrays.update({key:raw[key].copy() for key in raw.files if key.startswith(name+'_')})
    models, audits = [], []
    summary=read(folder/'summary.json')
    opt,_,_=backend.optical('square',summary['design'],summary['settings']['order'],
        summary['settings']['dz_nm'],rows[0]['wavelength_nm'],summary['period_nm'])
    spec,_,_,_=backend.finite('square',opt,summary['settings']['square_grid'],8,False)
    physical_checks=[]
    raw=arrays
    for row in rows:
        name = row['name']
        qe, qg = raw[name+'_qenergy'], raw[name+'_qgain']
        vg = backend.c*qe.sum()/(1.8166*qg.sum())
        error = np.max(abs(qg-backend.c/(1.8166*vg)*qe))
        if not np.allclose(qg, backend.c/(1.8166*vg)*qe, rtol=2e-12, atol=1e-16):
            raise ValueError(f'Inconsistent saved energy/gain weights: {name}')
        models.append(backend.SpatialYbYAGModel(backend.material(row['wavelength_nm']),
            raw[name+'_widths'], qe, row['total_power_loss_per_m'],
            row['up_power_loss_per_m'], row['down_power_loss_per_m'], vg, qg,
            raw[name+'_lateral']))
        audits.append(dict(name=name, wavelength_nm=row['wavelength_nm'],
            group_velocity_m_s=vg, group_index=backend.c/vg,
            gain_energy_identity_max_absolute_error=error,
            required_material_gain_cm1=row['required_material_gain_cm1']))
        field=raw[name+'_field']; flat=field.reshape(4,-1); norm=np.sum(abs(flat)**2)
        C,Lup,Ldown=raw[name+'_C'],raw[name+'_Lup'],raw[name+'_Ldown']
        problem=backend.finite_area_eigenproblem(spec,C)
        value=complex(row['delta_per_m'],row['alpha_field_per_m'])
        residual=backend.relative_residual(problem.center_operator(),field,value)
        flux=problem.loss_budget(field)
        up,down=[float(2*np.einsum('ik,ij,jk->',flat.conj(),L,flat).real/norm) for L in (Lup,Ldown)]
        fluxres=abs(up+down+2*flux['transport_amplitude_loss_per_m']-2*value.imag)
        opticalres=float(np.linalg.norm((C-C.conj().T)/(2j)-Lup-Ldown)/max(np.linalg.norm(Lup+Ldown),1e-30))
        passed=(residual<2e-7 and fluxres<1e-4 and opticalres<1e-8
            and abs(row['frequency_residual_per_m'])<.05
            and abs(up-row['up_power_loss_per_m'])<1e-7
            and abs(down-row['down_power_loss_per_m'])<1e-7)
        physical_checks.append(dict(name=name,passed=bool(passed),eigen_residual=residual,
            passive_loss_closure_per_m=fluxres,optical_theorem_residual=opticalres,
            stored_frequency_residual_per_m=row['frequency_residual_per_m']))
    backend.save(folder/'augmented_mode_identity_and_flux_audit.json',dict(
        status='passed' if all(r['passed'] for r in physical_checks) else 'failed',
        added_independent_challenge_modes=added,rejected_challenge_modes=rejected,
        retained_modes=len(rows),physical_checks=physical_checks,
        scope='Stored original pencils and flux checked for all retained roots. Supplementary closed roots are compared at equal wavelength/complex eigenvalue using independent envelope-span rank.'))
    if not all(r['passed'] for r in physical_checks):raise ValueError('Augmented optical readback failed')
    return backend.SpatialMultimodeYbYAG(tuple(models), tuple(r['name'] for r in rows)), rows, audits


def review(folder, cw=False):
    system, rows, weights = saved_system(folder)
    summary = read(folder/'summary.json')
    ordered = sorted(rows, key=lambda r: r['required_material_gain_cm1'])
    result = dict(status='running', source_case=str(folder), retained_independent_modes=len(rows),
        source_summary_sha256=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest(),
        source_fields_sha256=hashlib.sha256((folder/'closed_fields.npz').read_bytes()).hexdigest(),
        weight_identity=weights, records=[],
        cold_first_two_names=[r['name'] for r in ordered[:2]],
        cold_relative_material_gain_gap=ordered[1]['required_material_gain_cm1']/ordered[0]['required_material_gain_cm1']-1,
        scope='Fixed cold fields and mode-specific wavelengths; 300 K thermal-manifold reciprocity constrains absorption/emission pairs only. The assumed emission envelope is a constant 2e-20 cm2, not measured gain at each wavelength. Internal first-entry pump power; shared x-z inversion with y averaged; no external coupling/thermal/full-wave/dynamic stability certification.')
    destination=folder/'spectroscopy_review.json'
    for label in ('nominal_flat_spectrum', 'reciprocity_paired_300K_constant_emission_envelope'):
        models, materials = [], []
        for old in system.modes:
            med = old.medium
            if label.startswith('reciprocity'):
                med = replace(med,
                    pump_emission_cross_section_cm2=med.pump_absorption_cross_section_cm2*stark_emission_to_absorption(med.pump_wavelength_nm),
                    laser_absorption_cross_section_cm2=med.laser_emission_cross_section_cm2/stark_emission_to_absorption(med.laser_wavelength_nm))
            models.append(replace(old, medium=med))
            fmax = med.pump_absorption_cross_section_cm2/(med.pump_absorption_cross_section_cm2+med.pump_emission_cross_section_cm2)
            ceiling = med.dopant_density_cm3*((med.laser_emission_cross_section_cm2+med.laser_absorption_cross_section_cm2)*fmax-med.laser_absorption_cross_section_cm2)
            materials.append(dict(medium=asdict(med), pump_only_inversion_ceiling=fmax,
                pump_limited_material_gain_ceiling_cm1=ceiling))
        varied = backend.SpatialMultimodeYbYAG(tuple(models), system.names)
        record = dict(label=label, materials=dict(zip(system.names, materials)), pumps={})
        for architecture in ('edge_dual', 'top_single'):
            pump = backend.PumpGeometryConfig(architecture, 0., summary['side_um'], 31)
            threshold = backend.shared_inversion_pump_threshold(varied, pump)
            item = dict(threshold=asdict(threshold), cw=[])
            item['actual_winners'] = [next(r for r in rows if r['name']==name) for name in threshold.first_modes]
            if cw and threshold.first_threshold_internal_W is not None:
                for ratio in (1.05, 1.5, 2., 4.):
                    started = time.monotonic()
                    try:
                        state = backend.solve_multimode_spatial_ybyag(varied,
                            replace(pump, internal_incident_power_W=ratio*threshold.first_threshold_internal_W))
                        point = dict(status='completed', ratio=ratio,
                            internal_pump_W=ratio*threshold.first_threshold_internal_W,
                            active_modes=state.active_modes,
                            upward_W=dict(zip(system.names, map(float,state.upward_output_W))),
                            energy_budget_W=state.energy_budget_W,
                            ncp_residual=state.complementarity_residual,
                            population_residual_per_s=state.population_residual_per_s)
                    except Exception as error:
                        point = dict(status='failed', ratio=ratio, error=repr(error))
                    point['seconds'] = time.monotonic()-started
                    item['cw'].append(point)
            record['pumps'][architecture] = item
            print('review', folder.name, label, architecture, threshold.first_threshold_internal_W,
                  threshold.first_modes, flush=True)
        result['records'].append(record)
        backend.save(destination,result)
    result['status']='completed'
    backend.save(destination,result)


def control_readback(folder):
    """Check stored higher-D and higher-N poles without choosing new modes."""
    summary=read(folder/'summary.json');checks=[]
    for control in read(folder/'accuracy_controls/summary.json')['results']:
        source=folder/'accuracy_controls'/f"D{control['order']}_N{control['grid']}"
        first=control['closed_modes'][0]
        opt,_,_=backend.optical('square',summary['design'],control['order'],control['dz_nm'],
            first['wavelength_nm'],summary['period_nm'])
        spec,_,_,_=backend.finite('square',opt,control['grid'],8,False)
        with np.load(source.with_suffix('.npz'),allow_pickle=False) as raw:
            for row in control['closed_modes']:
                name=row['name'];field=raw[name+'_field'];flat=field.reshape(4,-1)
                C,Lup,Ldown=raw[name+'_C'],raw[name+'_Lup'],raw[name+'_Ldown']
                norm=np.sum(abs(flat)**2);value=complex(row['delta_per_m'],row['alpha_field_per_m'])
                problem=backend.finite_area_eigenproblem(spec,C)
                residual=backend.relative_residual(problem.center_operator(),field,value)
                up,down=[float(2*np.einsum('ik,ij,jk->',flat.conj(),L,flat).real/norm) for L in (Lup,Ldown)]
                fluxres=abs(up+down+2*problem.loss_budget(field)['transport_amplitude_loss_per_m']-2*value.imag)
                gain=(2*value.imag+10.)/raw[name+'_qgain'].sum()/100
                gain_difference=abs(gain-row['required_material_gain_cm1'])
                passed=(residual<2e-7 and fluxres<1e-4 and gain_difference<1e-10
                    and abs(row['frequency_residual_per_m'])<.05)
                checks.append(dict(order=control['order'],grid=control['grid'],name=name,passed=bool(passed),
                    original_eigen_residual=residual,passive_loss_closure_per_m=fluxres,
                    material_gain_readback_absolute_difference_cm1=gain_difference,
                    frequency_residual_per_m=row['frequency_residual_per_m']))
    result=dict(status='passed' if all(r['passed'] for r in checks) else 'failed',checks=checks,
        scope='Supplied low-threshold branches on higher D or N; not a new complete fine-grid spectrum or a joint D48/N65 run.')
    backend.save(folder/'accuracy_controls/readback_qa.json',result)
    if result['status']!='passed':raise ValueError('Control pencil/flux/frequency readback failed')


def export_case(folder):
    """Freeze a single merged source for later comparisons and reporting.

    中文：40 根合并后提供统一场文件，避免读到原 32 根里的错误次模。
    This copies accepted optical arrays; it does not recompute or relabel roots.
    """
    review=read(folder/'spectroscopy_review.json')
    identity=read(folder/'augmented_mode_identity_and_flux_audit.json')
    controls=read(folder/'accuracy_controls/summary.json')
    readback=read(folder/'readback_qa.json')
    if review['status']!='completed' or identity['status']!='passed':
        raise ValueError('Complete material review and passed merged optical audit required')
    if readback['status']!='passed':raise ValueError('Original field/Jones/CW readback failed')
    cw_checks=[]
    for record in review['records']:
        for architecture,pump in record['pumps'].items():
            if len(pump['cw'])!=4 or any(p['status']!='completed' for p in pump['cw']):
                raise ValueError('Four successful augmented shared-CW points required per protocol/architecture')
            for point in pump['cw']:
                budget=point['energy_budget_W']
                passed=(abs(budget['energy_closure_residual_W'])<1e-8
                    and abs(budget['pump_flux_residual_W'])<1e-8 and point['ncp_residual']<2e-7)
                cw_checks.append(dict(label=record['label'],architecture=architecture,
                    ratio=point['ratio'],passed=bool(passed),energy_closure_residual_W=budget['energy_closure_residual_W'],
                    pump_flux_residual_W=budget['pump_flux_residual_W'],ncp_residual=point['ncp_residual']))
    backend.save(folder/'augmented_shared_cw_readback.json',dict(
        status='passed' if all(r['passed'] for r in cw_checks) else 'failed',checks=cw_checks))
    if not all(r['passed'] for r in cw_checks):raise ValueError('Augmented shared-CW budget failed')
    if any(r['status']!='completed' for r in controls['results']):
        raise ValueError('Precision controls incomplete')
    control_readback(folder)
    accepted={r['name'] for r in identity['physical_checks'] if r['passed']}
    rows=[];arrays={};sources={}
    for source in (folder,folder/'spectral_challenges'):
        rows += [r for r in read(source/'closed_optical_modes.json') if r['name'] in accepted]
        with np.load(source/'closed_fields.npz',allow_pickle=False) as raw:
            for key in raw.files:
                if key=='angle_deg':
                    if key in arrays and not np.array_equal(arrays[key],raw[key]):raise ValueError('Angular axes differ')
                    arrays[key]=raw[key].copy()
                elif any(key.startswith(name+'_') for name in accepted):arrays[key]=raw[key].copy()
        for file in ('closed_optical_modes.json','closed_fields.npz'):
            p=source/file;sources[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows.sort(key=lambda r:r['required_material_gain_cm1'])
    backend.save(folder/'merged_closed_optical_modes.json',rows)
    np.savez_compressed(folder/'merged_closed_fields.npz',**arrays)
    original=read(folder/'summary.json');nominal=review['records'][0];paired=review['records'][1]
    summary=dict(original)
    summary.update(included_closed_modes=len(rows),fields_file='merged_closed_fields.npz',
        modes_file='merged_closed_optical_modes.json',source_original_summary='summary.json',
        source_spectroscopy_review='spectroscopy_review.json',source_optical_sha256=sources,
        actual_winners={a:p['actual_winners'] for a,p in nominal['pumps'].items()},
        pump_results=[dict(architecture=a,threshold=p['threshold'],cw=p['cw']) for a,p in nominal['pumps'].items()],
        additional_reciprocity_review=paired,
        closed_mode_identity_audit=identity,
        scope=original['scope']+'; augmented with explicit cross-cluster/Gamma representatives and a fresh all-retained-mode shared-pump/CW review')
    backend.save(folder/'merged_summary.json',summary)
    configuration=dict(design=original['design'],period_nm=original['period_nm'],
        aperture_um=[300.,300.],gain_thickness_nm=500.,passive_pc_thickness_nm=120.,
        silica_spacer_nm=2000.,DBR_pairs=16,
        optical_settings=original['settings'],material_protocols=[dict(label=r['label'],
            materials=r['materials']) for r in review['records']],
        assumptions=review['scope'],target_first_emission_nm=1030.,target_tolerance_nm=.05)
    backend.save(folder/'selected_configuration.json',configuration)
    first=rows[0];second=rows[1]
    def full(p):return str(p.resolve())
    selected=dict(status='completed',id='YbYAG_square',role='isolation_primary',lattice='square',
        case_directory=full(folder),design=original['design'],period_nm=original['period_nm'],
        D=original['settings']['order'],dz_nm=original['settings']['dz_nm'],N=original['settings']['square_grid'],
        aperture_um=[300.,300.],actual_first_wavelength_nm=nominal['pumps']['edge_dual']['actual_winners'][0]['wavelength_nm'],
        first_mode_wavelength_nm=nominal['pumps']['edge_dual']['actual_winners'][0]['wavelength_nm'],
        lowest_cold_root=first['name'],g1_cm1=first['required_material_gain_cm1'],
        closest_competitor_root=second['name'],g2_cm1=second['required_material_gain_cm1'],
        closest_competitor_wavelength_nm=second['wavelength_nm'],
        relative_material_gap=second['required_material_gain_cm1']/first['required_material_gain_cm1']-1,
        beam=first['beam'],polarization=first['polarization'],independent_closed_modes=len(rows),
        summary_path=full(folder/'merged_summary.json'),closed_modes_path=full(folder/'merged_closed_optical_modes.json'),
        fields_path=full(folder/'merged_closed_fields.npz'),merged_cross_cluster_json=full(folder/'merged_closed_optical_modes.json'),
        geometry_path=full(folder/'actual_geometry_nm.json'),config_path=full(folder/'selected_configuration.json'),
        nominal=nominal['pumps'],reciprocity_300K=paired['pumps'],
        nominal_pump_results=nominal['pumps'],reciprocity_pump_results=paired['pumps'],
        verification=dict(original32_readback_path=full(folder/'readback_qa.json'),
            augmented_identity_flux_path=full(folder/'augmented_mode_identity_and_flux_audit.json'),
            augmented_shared_cw_path=full(folder/'augmented_shared_cw_readback.json'),
            control_readback_path=full(folder/'accuracy_controls/readback_qa.json'),
            status='passed',scope='Original-pencil/flux/frequency/identity plus stored Jones and shared-CW conservation; independent two-root rebuild is recorded separately by the cross-review agent.'),
        controls_path=full(folder/'accuracy_controls/summary.json'),controls=controls,
        source_optical_sha256=sources,
        scientific_scope=review['scope'],
        spectral_scope='40 independent actual-frequency roots: lowest32 frozen-rank census plus8 explicit cross-cluster/Gamma representatives. Bounded sparse fine spectrum, not exhaustive Maxwell certification.',
        recommendation='Selected for the larger confirmed cold material-gain isolation at nearly unchanged first threshold. Candidate00 remains a lower-threshold Pareto alternative.')
    selected['source_sha256']={str((folder/name).resolve()):hashlib.sha256((folder/name).read_bytes()).hexdigest()
        for name in ('merged_summary.json','merged_closed_optical_modes.json','merged_closed_fields.npz',
            'actual_geometry_nm.json','selected_configuration.json','spectroscopy_review.json')}
    backend.save(folder/'selected_case.json',selected)
    backend.save(folder.parent/'selected_case.json',selected)
    print('exported_merged',folder,len(rows),selected['relative_material_gap'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--cw', action='store_true')
    parser.add_argument('--export-only',action='store_true')
    args=parser.parse_args()
    if args.export_only:export_case(args.case)
    else:review(args.case,args.cw)
