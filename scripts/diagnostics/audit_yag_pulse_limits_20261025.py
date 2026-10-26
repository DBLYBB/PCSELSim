"""Pulsed energy/duration, seed sensitivity and independent integrator limits."""
from __future__ import annotations
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/optimization')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from yag_feasibility_study import load_saved_system
from pcselsim.ybyag_pulsed import RectangularPumpPulse,simulate_pulsed_ybyag,uniform_top_pump_threshold


def main():
    out=ROOT/'results/current/yag_feasibility_20261025/pulsed'
    out.mkdir(parents=True,exist_ok=True)
    file=out/'pulse_limit_audit.json'
    records=json.loads(file.read_text(encoding='utf-8'))['cases'] if file.exists() else []
    models={}
    for lattice in ('square','triangular'):
        s,_,_=load_saved_system(lattice);models[lattice]=s.modes[s.names.index('TE0_pole00')]
    def run(key,lattice,pulse,**settings):
        old=next((r for r in records if r['id']==key),None)
        if old:return old
        start=time.monotonic();r=simulate_pulsed_ybyag(models[lattice],pulse,**settings)
        record=dict(id=key,lattice=lattice,pulse=asdict(pulse),metrics=r.metrics,elapsed_s=time.monotonic()-start)
        records.append(record)
        file.write_text(json.dumps(dict(scope='Read-only cold-mode reuse, additional uniform single-mode pulse limits',cases=records),indent=2),encoding='utf-8')
        print(key,r.metrics['upward_output_J'],record['elapsed_s'],flush=True)
        return record
    # Exact same external energy for every pulse width; no P_th*T scaling here.
    for lattice in models:
        for duration in (1e-5,1e-4,.001,.005):
            pulse=RectangularPumpPulse(.005,duration,coupling_efficiency=.7,top_passes=20,after_pulse_s=.0019)
            run(f'same5mJ_{lattice}_T{duration:g}',lattice,pulse,rtol=2e-6,samples=1201)
    # Seed beta is not measured. Its role in finite-pulse start-up is exposed.
    for lattice,m in models.items():
        p0=RectangularPumpPulse(0.,.001,coupling_efficiency=.7,top_passes=20,after_pulse_s=.0019)
        power=uniform_top_pump_threshold(m,p0)['external_CW_threshold_W']*4
        for beta in (1e-10,1e-8,1e-6):
            run(f'beta_{lattice}_{beta:g}',lattice,
                replace(p0,external_energy_J=power*.001,spontaneous_beta=beta),rtol=2e-6,samples=1201)
    p0=RectangularPumpPulse(0.,.0003,coupling_efficiency=.7,top_passes=20,after_pulse_s=.0005)
    power=uniform_top_pump_threshold(models['square'],p0)['external_CW_threshold_W']*4
    p=replace(p0,external_energy_J=power*p0.duration_s)
    a=run('integrator_LSODA','square',p,method='LSODA',rtol=2e-7,samples=1201)
    b=run('integrator_Radau','square',p,method='Radau',rtol=2e-7,samples=1201)
    error=abs(a['metrics']['upward_output_J']/b['metrics']['upward_output_J']-1)
    assert error<1e-4,(a,b)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.1))
    for ax,lattice in zip(axes,models):
        chosen=[r for r in records if r['id'].startswith(f'same5mJ_{lattice}')]
        ax.semilogx([r['pulse']['duration_s']*1000 for r in chosen],
            [r['metrics']['upward_output_J']*1e6 for r in chosen],'o-')
        ax.set(xlabel='Pump pulse duration (ms)',ylabel='Upward-mode energy incl. spontaneous (uJ)',
            title=f'{lattice}: fixed external pump 5 mJ')
        ax.grid(alpha=.25)
    fig.suptitle('Uniform one-mode reduction; top20/.98 relay and eta=.7 assumed',fontsize=11)
    fig.tight_layout();fig.savefig(out/'same_energy_different_duration.png',dpi=190);plt.close(fig)
    file.write_text(json.dumps(dict(status='passed',scope='Uniform single-mode only; no thermal/spatial multimode claim',
        cases=records,integrator_relative_output_difference=error,
        maximum_relative_energy_residual=max(r['metrics']['relative_energy_residual'] for r in records)),indent=2),encoding='utf-8')


if __name__=='__main__':main()
