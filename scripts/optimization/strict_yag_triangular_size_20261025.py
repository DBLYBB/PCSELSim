"""Fixed final triangular Yb:YAG motif: finite-size poles and pump thresholds.

中文：只改变物理正方边界边长，不逐尺寸调整周期，不按面积或1/L缩放损耗。
The300um point is the final merged anchor; every other size solves new poles.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(Path(__file__).parent)]
import numpy as np
import yag_feasibility_study as study
import matched300_yag_search as backend
import strict_yag_feasibility_20261025 as rates
from pcselsim.finite_discretization import relative_residual

OUT=ROOT/'results/current/strict_rerun_20261025/yag_feasibility'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def anchor(summary_path,rows_path,fields_path):
    summary=read(summary_path);rows=read(rows_path)
    system,_,_=rates.load_system('triangular')
    target=[i for i,r in enumerate(rows) if 1029.<=r['wavelength_nm']<=1031.]
    result=dict(status='completed',lattice='triangular',side_um=300.,period_nm=summary['period_nm'],
        design=summary['design'],order=summary['settings']['order'],dz_nm=summary['settings']['dz_nm'],
        grid=summary['settings']['tri_radius_cells'],modes=[],closure_failures=[],pumps={},target_band_pumps={},
        target_band_nm=[1029.,1031.],target_band_mode_names=[rows[i]['name'] for i in target],
        anchor=True,source_summary_path=str(summary_path),source_fields_path=str(fields_path),source_modes_path=str(rows_path),
        census=dict(retained_independent_modes=len(rows),scope='Read-only final merged300um anchor; no repeated root search.'),
        scope='All retained independent actual-frequency roots; nominal flat spectroscopy; target-window thresholds are conditional subsets and do not delete physical competitors.')
    for r in rows:
        result['modes'].append(dict(name=r['name'],wavelength_nm=r['wavelength_nm'],delta_per_m=r['delta_per_m'],
            alpha_field_per_m=r['alpha_field_per_m'],material_gain_cm1=r['required_material_gain_cm1'],
            total_power_loss_per_m=r['total_power_loss_per_m'],up_power_loss_per_m=r['up_power_loss_per_m'],
            down_power_loss_per_m=r['down_power_loss_per_m'],gain_weight=r['gain_weight'],
            eigen_residual=r['eigen_residual'],frequency_residual_per_m=r['frequency_residual_per_m']))
    target_system=(study.SpatialMultimodeYbYAG(tuple(system.modes[i] for i in target),
        tuple(system.names[i] for i in target)) if target else None)
    for architecture in ('edge_dual','top1','top20'):
        pump=study.pump_config(architecture,300.)
        result['pumps'][architecture]=asdict(study.shared_inversion_pump_threshold(system,pump))
        result['target_band_pumps'][architecture]=(asdict(study.shared_inversion_pump_threshold(target_system,pump))
            if target_system is not None else dict(first_threshold_internal_W=None,status='no_documented_roots_in_window'))
    return result


def audit_fields(result,fields):
    """Save own C and radiation and test the unmodified K/M and frequency.

    中文：读回检查独立于搜模内部收敛；不通过的点不得当作物理解。
    """
    checks=[];bad=[]
    for row in result['modes']:
        name=row['name'];wave=row['wavelength_nm']
        model,period,_=backend.optical('triangular',result['design'],result['order'],result['dz_nm'],wave,result['period_nm'])
        spec,problem,mesh=study.make_problem('triangular',model,result['side_um'],result['grid'])
        field=fields[name+'_field'];value=complex(row['delta_per_m'],row['alpha_field_per_m'])
        residual=relative_residual(problem.center_operator(),field,value)
        beta0=4*np.pi/np.sqrt(3)/(period*1e-9)
        beta=2*np.pi*model.result.effective_index/(wave*1e-9)
        freqres=float((beta*beta-beta0*beta0)/(2*beta0)-value.real)
        budget=problem.loss_budget(field)
        energy_error=float(budget['local_amplitude_loss_per_m']+budget['transport_amplitude_loss_per_m']-value.imag)
        gain=(2*value.imag+10.)/fields[name+'_qgain'].sum()/100
        qg,qe=fields[name+'_qgain'],fields[name+'_qenergy']
        recovered_vg=backend.c*qe.sum()/(1.8166*qg.sum())
        weight_identity=np.allclose(qg,backend.c/(1.8166*recovered_vg)*qe,rtol=2e-12,atol=1e-16)
        passed=bool(period==result['period_nm'] and value.imag>=0 and residual<2e-7
            and abs(freqres)<.05 and abs(energy_error)<1e-4 and abs(gain-row['material_gain_cm1'])<1e-8 and weight_identity)
        checks.append(dict(name=name,passed=passed,original_KM_residual=residual,
            actual_period_nm=period,frequency_residual_per_m=freqres,energy_error_per_m=energy_error,
            recomputed_material_gain_cm1=float(gain),gain_energy_identity_pass=bool(weight_identity)))
        if not passed:bad.append(name)
        rx,ry=model.result.radiation_amplitudes(field)
        fields.update({name+'_C':model.result.coupling_m,name+'_radiation_x':rx,name+'_radiation_y':ry,
            name+'_Lup':model.upward_amplitude_loss_m,name+'_Ldown':model.downward_amplitude_loss_m,
            'x_um':mesh.x_um,'y_um':mesh.y_um})
        row.update(group_index=model.result.group_index,group_velocity_m_s=backend.c/model.result.group_index)
    result.update(strict_original_field_readback=checks,physically_invalid_rows=bad,
        missing_root_status='some_attempts_failed_and_remain_unresolved' if result['closure_failures'] else 'all_supplied_seeds_processed',
        target_window_interpretation='Conditional1029–1031nm subset only; no physical spectral filter. All accepted roots remain in total-spectrum pumps.')
    if bad:result['status']='invalid_physical_readback'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sizes',nargs='+',type=float,default=[50,100,200,300,500,750])
    p.add_argument('--order',type=int,default=12);p.add_argument('--dz',type=float,default=2.)
    p.add_argument('--grid',type=int,default=15);p.add_argument('--keep',type=int,default=8)
    a=p.parse_args();selected,summary_path,rows_path,fields_path,selected_path=rates.files_for('triangular')
    summary=read(summary_path)
    def chosen(lattice):
        if lattice!='triangular':raise ValueError('Triangular-only wrapper')
        return dict(id='YbYAG_triangular',summary_path=str(summary_path))
    study.selected=chosen;study.OUT=OUT;OUT.mkdir(parents=True,exist_ok=True)
    sources={str(q):sha(q) for q in (summary_path,rows_path,fields_path,selected_path)}
    stamp=time.time_ns();(OUT/f'executed_triangular_size_{stamp}.py').write_bytes(Path(__file__).read_bytes())
    study.save(OUT/f'runtime_triangular_size_{stamp}.json',dict(argv=sys.argv,source_input_sha256=sources,
        wrapper_sha256=sha(__file__),backend_sha256=sha(study.__file__),
        core_sha256={str(q.relative_to(ROOT)):sha(q) for q in (ROOT/'src/pcselsim').glob('*.py')}))
    for length in a.sizes:
        tag='triangular_L300_formal_merged_anchor' if length==300. else f'triangular_L{length:g}_D{a.order}_dz{a.dz:g}_N{a.grid}_K{a.keep}'
        output=OUT/'sizes'/f'{tag}.json'
        if output.exists() and read(output).get('status')=='completed':continue
        started=time.monotonic();print('STRICT_SIZE_START',tag,flush=True)
        result=None;fields=None
        previous=read(output) if output.exists() else None
        if previous and previous.get('status') not in ('completed','pending_readback'):
            failure_path=OUT/'size_execution_failures'/f'{tag}_{time.time_ns()}.json'
            study.save(failure_path,dict(source_path=str(output),record=previous,
                scope='Preserved execution/readback failure; not reclassified as physical nonlasing.'))
        try:
            if length==300.:result=anchor(summary_path,rows_path,fields_path)
            else:
                if previous and previous.get('modes') and previous.get('status') in ('pending_readback','failed_readback') and output.with_suffix('.npz').exists():
                    result=previous;result['status']=result.get('solver_status','completed')
                    with np.load(output.with_suffix('.npz'),allow_pickle=False) as raw:
                        fields={k:raw[k].copy() for k in raw.files}
                else:
                    result,fields=study.size_point('triangular',length,a.order,a.dz,a.grid,a.keep)
                    output.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(output.with_suffix('.npz'),**fields)
                    study.save(output,{**result,'solver_status':result['status'],'status':'pending_readback'})
                audit_fields(result,fields);output.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(output.with_suffix('.npz'),**fields)
            result.update(source_unit_cell_path=str(summary_path),historical_spectra_reused=False,
                fixed_period_for_all_sizes=True,period_retuned_at_each_size=False,
                threshold_curve_scope='Nominal constant cross sections; bounded TE0 modes in1000–1060nm analysis band.',
                seconds_total=time.monotonic()-started)
        except Exception as error:
            if result is not None and result.get('modes'):
                result.update(status='failed_readback',readback_error=repr(error),seconds_total=time.monotonic()-started)
                if fields is not None:np.savez_compressed(output.with_suffix('.npz'),**fields)
            else:result=dict(status='failed',lattice='triangular',side_um=length,error=repr(error),seconds_total=time.monotonic()-started)
        study.save(output,result)
        print('STRICT_SIZE_DONE',tag,result['status'],len(result.get('modes',[])),
            sorted((r['material_gain_cm1'],r['wavelength_nm']) for r in result.get('modes',[]))[:3],flush=True)
    if any(sha(q)!=value for q,value in sources.items()):raise RuntimeError('Frozen selected source changed during size study')


if __name__=='__main__':main()
