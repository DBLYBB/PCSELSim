"""Fixed new-YAG-square motif size study; no period or loss rescaling.

中文：复用已审查size_point物理流程，只重绑定选定几何与输出目录。300um显式
读取新40模合并锚点，其余尺寸重新求有限谱与实频；目标窗只是条件报告，不滤总谱。
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from dataclasses import asdict
import sys,json,hashlib,time,argparse
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(Path(__file__).parent)]
import numpy as np
import yag_feasibility_study as study
import matched300_yag_search as backend
from pcselsim.finite_discretization import relative_residual

OUT=ROOT/'results/current/strict_rerun_20261025/yag_feasibility'
SOURCE=ROOT/'results/current/strict_rerun_20261025/YbYAG_square/candidate_01_full32'
SUMMARY=SOURCE/'merged_summary.json'
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def strict_selected(lattice):
    if lattice!='square':raise ValueError('This wrapper is only the newly selected square motif')
    return dict(id='YbYAG_square',summary_path=SUMMARY.relative_to(ROOT).as_posix())

def anchor():
    """Use the audited40-root300um point / 不从旧32模加载器重新选锚点。"""
    summary=load(SUMMARY);rows=load(SOURCE/'merged_closed_optical_modes.json');models=[]
    with np.load(SOURCE/'merged_closed_fields.npz',allow_pickle=False) as arrays:
        for row in rows:
            name=row['name'];qe,qg=arrays[name+'_qenergy'],arrays[name+'_qgain']
            vg=backend.c*qe.sum()/(1.8166*qg.sum())
            models.append(study.SpatialYbYAGModel(backend.material(row['wavelength_nm']),arrays[name+'_widths'],qe,
                row['total_power_loss_per_m'],row['up_power_loss_per_m'],row['down_power_loss_per_m'],vg,qg,arrays[name+'_lateral']))
    modes=[dict(name=r['name'],wavelength_nm=r['wavelength_nm'],delta_per_m=r['delta_per_m'],alpha_field_per_m=r['alpha_field_per_m'],
        material_gain_cm1=r['required_material_gain_cm1'],total_power_loss_per_m=r['total_power_loss_per_m'],
        up_power_loss_per_m=r['up_power_loss_per_m'],down_power_loss_per_m=r['down_power_loss_per_m'],gain_weight=r['gain_weight'],
        eigen_residual=r['eigen_residual'],frequency_residual_per_m=r['frequency_residual_per_m']) for r in rows]
    system=study.SpatialMultimodeYbYAG(tuple(models),tuple(r['name'] for r in rows))
    target=[i for i,r in enumerate(rows) if 1029.<=r['wavelength_nm']<=1031.]
    target_system=study.SpatialMultimodeYbYAG(tuple(models[i] for i in target),tuple(rows[i]['name'] for i in target))
    result=dict(status='completed',lattice='square',side_um=300.,period_nm=summary['period_nm'],design=summary['design'],
        order=32,dz_nm=1.,grid=49,modes=modes,closure_failures=[],pumps={},target_band_pumps={},
        target_band_nm=[1029.,1031.],target_band_mode_names=[rows[i]['name'] for i in target],
        anchor=True,source_summary_path=str(SUMMARY),source_fields_path=str(SOURCE/'merged_closed_fields.npz'),
        source_modes_path=str(SOURCE/'merged_closed_optical_modes.json'),
        census=dict(retained_independent_modes=len(rows),scope='Previously audited40-root merged300um formal anchor, no new300um root solve.'),
        scope='New selected fixed motif; nominal flat spectroscopy. All40 roots enter total-spectrum shared inversion; the target window is a separate conditional report, not a physical spectral filter.')
    for architecture in ('edge_dual','top1','top20'):
        result['pumps'][architecture]=asdict(study.shared_inversion_pump_threshold(system,study.pump_config(architecture,300.)))
        result['target_band_pumps'][architecture]=asdict(study.shared_inversion_pump_threshold(target_system,study.pump_config(architecture,300.)))
    return result

def audit_fields(result,fields):
    """Rebuild each accepted root for readback / 补存自身C、辐射与原残差。"""
    records=[];bad=[]
    for row in result['modes']:
        name=row['name'];wave=row['wavelength_nm'];model,period,guide=backend.optical('square',result['design'],
            result['order'],result['dz_nm'],wave,result['period_nm'])
        spec,problem,_=study.make_problem('square',model,result['side_um'],result['grid'])
        field=fields[name+'_field'];value=complex(row['delta_per_m'],row['alpha_field_per_m'])
        residual=relative_residual(problem.center_operator(),field,value)
        beta0=2*np.pi/(period*1e-9);beta=2*np.pi*model.result.effective_index/(wave*1e-9)
        frequency_residual=float((beta*beta-beta0*beta0)/(2*beta0)-value.real)
        budget=problem.loss_budget(field);energy_error=float(sum(budget[k] for k in ('local_amplitude_loss_per_m','transport_amplitude_loss_per_m'))-value.imag)
        gain=(2*value.imag+10.)/fields[name+'_qgain'].sum()/100
        passed=bool(period==result['period_nm'] and value.imag>=0 and residual<2e-7 and abs(frequency_residual)<.05 and abs(energy_error)<1e-4 and abs(gain-row['material_gain_cm1'])<1e-8)
        record=dict(name=name,passed=passed,actual_period_nm=period,original_KM_residual=residual,frequency_residual_per_m=frequency_residual,
            energy_error_per_m=energy_error,recomputed_material_gain_cm1=float(gain))
        records.append(record)
        if not passed:bad.append(name)
        rx,ry=model.result.radiation_fields(field)
        fields.update({name+'_C':model.result.coupling_m,name+'_radiation_x':rx,name+'_radiation_y':ry,
            name+'_Lup':model.upward_amplitude_loss_m,name+'_Ldown':model.downward_amplitude_loss_m})
    result['strict_original_field_readback']=records;result['physically_invalid_rows']=bad
    if bad:result['status']='invalid_physical_readback'
    result['missing_root_status']='some_attempts_failed_and_remain_unresolved' if result['closure_failures'] else 'all_supplied_seeds_processed'
    result['target_window_interpretation']='Conditional threshold considering only1029–1031nm roots; this does not delete other physical roots or imply that a gain/window filter exists.'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sizes',nargs='+',type=float,default=[50,100,200,300,500,750])
    p.add_argument('--order',type=int,default=12);p.add_argument('--dz',type=float,default=2.)
    p.add_argument('--grid',type=int,default=25);p.add_argument('--keep',type=int,default=8)
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    study.OUT=OUT;study.selected=strict_selected
    source_hashes={str(p):sha(p) for p in (SUMMARY,SOURCE/'merged_closed_fields.npz',SOURCE/'merged_closed_optical_modes.json')}
    stamp=time.time_ns();(OUT/f'executed_square_size_{stamp}.py').write_bytes(Path(__file__).read_bytes())
    study.save(OUT/f'runtime_square_size_{stamp}.json',dict(argv=sys.argv,source_input_sha256=source_hashes,
        wrapper_sha256=sha(__file__),backend_sha256=sha(study.__file__),
        core_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/pcselsim').glob('*.py')}))
    for length in a.sizes:
        tag='square_L300_formal_merged40_anchor' if length==300. else f'square_L{length:g}_D{a.order}_dz{a.dz:g}_N{a.grid}_K{a.keep}'
        output=OUT/'sizes'/f'{tag}.json'
        if output.exists() and load(output).get('status')=='completed':continue
        print('STRICT_SIZE_START',tag,flush=True);started=time.monotonic()
        try:
            if length==300.:result=anchor()
            else:
                result,fields=study.size_point('square',length,a.order,a.dz,a.grid,a.keep)
                audit_fields(result,fields);output.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(output.with_suffix('.npz'),**fields)
            result.update(source_unit_cell_path=str(SUMMARY),historical_spectra_reused=False,
                fixed_period_for_all_sizes=True,period_retuned_at_each_size=False,
                threshold_curve_scope='Nominal flat cross sections only; bounded TE0 modes in the common1000–1060nm model analysis band.',
                seconds_total=time.monotonic()-started)
        except Exception as error:result=dict(status='failed',lattice='square',side_um=length,error=repr(error),seconds_total=time.monotonic()-started)
        study.save(output,result)
        print('STRICT_SIZE_DONE',tag,result['status'],len(result.get('modes',[])),
            sorted([(r['material_gain_cm1'],r['wavelength_nm']) for r in result.get('modes',[])])[:3],flush=True)
    if any(sha(p)!=v for p,v in source_hashes.items()):raise RuntimeError('Frozen source changed during size study')

if __name__=='__main__':main()
