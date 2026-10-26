"""Pump/energy/size studies on the post-audit selected Yb:YAG geometries.

中文：只读本轮最终合并复场，包含跨频段竞争模；不回写旧结果。CW共享反转，
脉冲采用均匀反转单模降阶，尺寸变化逐点重求本征场/实际频率，三者范围分开。
The assumed70% first-entry pump coupling and ideal20-pass relay are sensitivity
protocols, not measured device specifications. No dynamic plots enter the PPT.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
import argparse,hashlib,json,sys,time
from dataclasses import asdict,replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
import numpy as np
from scipy.constants import c
import yag_feasibility_study as f
from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG,solve_multimode_spatial_ybyag
from pcselsim.ybyag_spatial_rates import SpatialYbYAGModel
from pcselsim.ybyag_threshold import shared_inversion_pump_threshold
from pcselsim.ybyag_pulsed import RectangularPumpPulse,simulate_pulsed_ybyag,uniform_top_pump_threshold

BASE=ROOT/'results/current/strict_rerun_20261025'
OUT=BASE/'yag_feasibility'

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resolve(path):return Path(path) if Path(path).is_absolute() else ROOT/path

def files_for(lattice):
    selected_path=BASE/('YbYAG_'+lattice)/'selected_case.json';selected=read(selected_path)
    if selected.get('status')!='completed':raise RuntimeError('A completed selected case is required')
    summary=resolve(selected.get('merged_summary_path',selected['summary_path']))
    def supplied(keys,default):
        value=next((selected[k] for k in keys if selected.get(k)),default)
        return resolve(value)
    rows=supplied(('closed_modes_path','closed_optical_modes_path','merged_modes_path'),summary.parent/'merged_closed_optical_modes.json')
    fields=supplied(('fields_path','closed_fields_path','merged_fields_path'),summary.parent/'merged_closed_fields.npz')
    if not rows.exists() or not fields.exists():raise FileNotFoundError('Merged rows and complex fields must be explicit in selected_case')
    return selected,summary,rows,fields,selected_path

def selected_for_size(lattice):
    _,summary,_,_,_=files_for(lattice)
    return dict(id='YbYAG_'+lattice,summary_path=str(summary))

def load_system(lattice):
    selected,sp,rp,fp,_=files_for(lattice);summary=read(sp);rows=read(rp)
    models=[]
    with np.load(fp,allow_pickle=False) as raw:
        for row in rows:
            name=row['name'];qe=raw[name+'_qenergy'];qg=raw[name+'_qgain']
            vg=c*qe.sum()/(1.8166*qg.sum())
            if not np.allclose(qg,c/(1.8166*vg)*qe,rtol=2e-12,atol=1e-16):raise RuntimeError('Saved gain/energy weight inconsistency')
            models.append(SpatialYbYAGModel(f.optical_backend.material(row['wavelength_nm']),
                raw[name+'_widths'],qe,row['total_power_loss_per_m'],row['up_power_loss_per_m'],
                row['down_power_loss_per_m'],vg,qg,raw[name+'_lateral']))
    system=SpatialMultimodeYbYAG(tuple(models),tuple(r['name'] for r in rows))
    return system,summary,rows

def source_audit(lattices):
    files=[Path(__file__),ROOT/'scripts/optimization/yag_feasibility_study.py',
        ROOT/'src/pcselsim/ybyag_pulsed.py',ROOT/'src/pcselsim/ybyag_multimode.py']
    for lattice in lattices:
        _,sp,rp,fp,cp=files_for(lattice);files.extend((sp,rp,fp,cp))
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def cw(lattices):
    for lattice in lattices:
        system,_,_=load_system(lattice)
        for architecture in ('edge_dual','top1','top20'):
            path=OUT/'cw'/f'{lattice}_{architecture}.json'
            if path.exists() and read(path).get('status')=='completed':continue
            pump=f.pump_config(architecture,300.);threshold=shared_inversion_pump_threshold(system,pump)
            result=dict(status='running',lattice=lattice,architecture=architecture,retained_modes=len(system.modes),
                threshold=asdict(threshold),points=[],scope='Stationary shared x-z inversion, y-averaged cold modes; total internal first-entry pump W. Ideal20-pass relay is conditional, not manufactured optics.')
            if threshold.first_threshold_internal_W is None:result['status']='unreachable';f.save(path,result);continue
            for ratio in (0.,.5,.9,.99,1.01,1.05,1.1,1.25,1.5,2.,3.,4.,6.):
                started=time.monotonic();power=ratio*threshold.first_threshold_internal_W
                try:
                    state=solve_multimode_spatial_ybyag(system,replace(pump,internal_incident_power_W=power))
                    row=dict(status='completed',ratio=ratio,internal_pump_W=power,
                        upward_output_W=float(state.upward_output_W.sum()),active_modes=state.active_modes,
                        output_by_mode_W=dict(zip(system.names,map(float,state.upward_output_W))),
                        energy_budget_W=state.energy_budget_W,population_residual_per_s=state.population_residual_per_s,
                        complementarity_residual=state.complementarity_residual,
                        maximum_inactive_gain_excess_per_m=state.maximum_inactive_gain_excess_per_m)
                except Exception as e:row=dict(status='failed',ratio=ratio,internal_pump_W=power,error=repr(e))
                row['seconds']=time.monotonic()-started;result['points'].append(row);f.save(path,result)
                print('CW',lattice,architecture,ratio,row['status'],row.get('upward_output_W'),flush=True)
            result['status']='completed' if all(x['status']=='completed' for x in result['points']) else 'partial'
            f.save(path,result)

def pulses(lattices,rtol,qa):
    for lattice in lattices:
        system,_,rows=load_system(lattice)
        cold=min(rows,key=lambda r:r['required_material_gain_cm1']);name=cold['name']
        if abs(cold['wavelength_nm']-1030.)>.05:raise RuntimeError('Selected lowest cold mode is outside actual wavelength requirement')
        model=system.modes[system.names.index(name)]
        path=OUT/'pulsed'/f'{lattice}_energy_scan.json';records=read(path)['cases'] if path.exists() else []
        for passes in (1,20):
            for duration in (1e-5,1e-4,.001,.005):
                template=RectangularPumpPulse(0.,duration,coupling_efficiency=.7,top_passes=passes,
                    return_efficiency=.98,after_pulse_s=2*.95e-3)
                threshold=uniform_top_pump_threshold(model,template)
                for ratio in (.5,1.,1.2,2.,4.,8.):
                    key=f'{lattice}_top{passes}_T{duration:g}_R{ratio:g}'
                    if any(r['id']==key for r in records):continue
                    pulse=replace(template,external_energy_J=threshold['external_CW_threshold_W']*duration*ratio)
                    started=time.monotonic();response=simulate_pulsed_ybyag(model,pulse,rtol=rtol,samples=1601)
                    row=dict(id=key,lattice=lattice,first_cold_mode=name,wavelength_nm=cold['wavelength_nm'],
                        top_passes=passes,duration_s=duration,power_to_uniform_CW_threshold=ratio,
                        pulse=asdict(pulse),threshold=threshold,metrics=response.metrics,seconds=time.monotonic()-started)
                    if qa and passes==20 and duration==.001 and ratio==2.:
                        fine=simulate_pulsed_ybyag(model,pulse,rtol=rtol/10,samples=2401)
                        error=abs(fine.metrics['upward_output_J']-response.metrics['upward_output_J'])/max(abs(fine.metrics['upward_output_J']),1e-30)
                        row['accuracy_challenge']=dict(rtol=rtol/10,metrics=fine.metrics,relative_upward_energy_change=error,passed=bool(error<.005))
                    records.append(row)
                    if passes==20 and duration==.001 and ratio in (2.,4.):
                        path.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(path.parent/(key+'_trace.npz'),time_s=response.time_s,
                            inversion=response.excited_fraction,upward_W=response.upward_power_W)
                    f.save(path,dict(status='running',lattice=lattice,cases=records,
                        scope='Uniform inversion, one saved cold mode, fixed300um aperture. Not spatial multimode dynamics or thermal feasibility. External first-entry pulse energy includes an assumed70% entrance coupling.'))
                    print('PULSE',key,pulse.external_energy_J,response.metrics['upward_output_J'],flush=True)
        data=read(path);data['status']='completed';f.save(path,data)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('cw','pulse','size'))
    p.add_argument('--lattices',nargs='+',choices=('square','triangular'),default=['square','triangular'])
    p.add_argument('--rtol',type=float,default=2e-6);p.add_argument('--qa',action='store_true')
    p.add_argument('--sizes',nargs='+',type=float,default=[50.,100.,200.,300.,500.,750.])
    p.add_argument('--order',type=int,default=12);p.add_argument('--dz',type=float,default=2.)
    p.add_argument('--square-grid',type=int,default=25);p.add_argument('--tri-grid',type=int,default=15)
    p.add_argument('--closed-modes',type=int,default=6);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    before=source_audit(a.lattices);stamp=time.time_ns()
    f.save(OUT/f'provenance_{a.phase}_{stamp}.json',dict(argv=sys.argv,source_sha256=before))
    f.OUT=OUT;f.selected=selected_for_size
    if a.phase=='cw':cw(a.lattices)
    elif a.phase=='pulse':pulses(a.lattices,a.rtol,a.qa)
    else:f.size_phase(a)
    after=source_audit(a.lattices)
    if after!=before:raise RuntimeError('Frozen input changed during the study')
    f.save(OUT/f'completion_{a.phase}_{stamp}.json',dict(completed=True,inputs_unchanged=True,source_sha256=after))

if __name__=='__main__':main()
