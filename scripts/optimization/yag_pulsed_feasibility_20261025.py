"""Pulsed-energy study of the two saved 300-um Yb:YAG PCSEL first modes.

中文：只读取正式复场的同模损耗/权重；不重新设计几何、不把均匀单模脉冲结果
冒充空间多模预测。入口耦合70%是敏感性假设，非制造器件的测量值。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/optimization')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from yag_feasibility_study import load_saved_system
from pcselsim.ybyag_pulsed import (
    RectangularPumpPulse, simulate_pulsed_ybyag, uniform_top_pump_threshold,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'results/current/yag_feasibility_20261025/pulsed')
    p.add_argument('--rtol',type=float,default=2e-6)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    result_file=a.output/'pulse_energy_scan.json'
    rows=json.loads(result_file.read_text(encoding='utf-8'))['cases'] if result_file.exists() else []
    codes=[ROOT/'src/pcselsim/ybyag_pulsed.py',Path(__file__),ROOT/'scripts/optimization/yag_feasibility_study.py']
    provenance={str(path.relative_to(ROOT)):sha(path) for path in codes}
    durations=[1e-5,1e-4,.001,.005]
    ratios=[.5,1.,1.2,2.,4.,8.]
    if a.pilot:
        durations=[.001];ratios=[2.]
    models={}
    for lattice in ('square','triangular'):
        system,summary,closed=load_saved_system(lattice)
        models[lattice]=system.modes[system.names.index('TE0_pole00')]
    def save():
        result_file.write_text(json.dumps(dict(scope='Uniform inversion, one saved cold mode, rectangular top pump; not 32-mode spatial CW; eta=.7 is assumed, not measured',
            source_sha256=provenance,cases=rows),indent=2),encoding='utf-8')
    for lattice,model in models.items():
        for passes in (1,20):
            for duration in durations:
                template=RectangularPumpPulse(0.,duration,coupling_efficiency=.7,top_passes=passes,
                    return_efficiency=.98,after_pulse_s=2*.95e-3)
                threshold=uniform_top_pump_threshold(model,template)
                pth=threshold['external_CW_threshold_W']
                for ratio in ratios:
                    key=f'{lattice}_top{passes}_T{duration:g}_R{ratio:g}'
                    if any(row['id']==key for row in rows):
                        continue
                    pulse=replace(template,external_energy_J=pth*duration*ratio)
                    started=time.monotonic()
                    response=simulate_pulsed_ybyag(model,pulse,rtol=a.rtol,samples=1601)
                    row=dict(id=key,lattice=lattice,top_passes=passes,duration_s=duration,
                        power_to_uniform_CW_threshold=ratio,pulse=asdict(pulse),
                        elapsed_s=time.monotonic()-started,metrics=response.metrics)
                    rows.append(row)
                    if passes==20 and duration==.001 and ratio in (2.,4.):
                        np.savez_compressed(a.output/(key+'_trace.npz'),time_s=response.time_s,
                            inversion=response.excited_fraction,upward_W=response.upward_power_W)
                    save()
                    print(key,'Ein_J',pulse.external_energy_J,'Eout_J',response.metrics['upward_output_J'],
                          'steps',response.metrics['accepted_step_count'],'seconds',row['elapsed_s'],flush=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    colors=plt.cm.viridis(np.linspace(.08,.9,len(durations)))
    for ax,passes in zip(axes,(1,20)):
        for lattice,style in [('square','o-'),('triangular','s--')]:
            for color,duration in zip(colors,durations):
                selected=sorted([r for r in rows if r['lattice']==lattice and r['top_passes']==passes and r['duration_s']==duration],key=lambda r:r['pulse']['external_energy_J'])
                if selected:
                    ax.plot([r['pulse']['external_energy_J']*1000 for r in selected],
                        [r['metrics']['upward_output_J']*1e6 for r in selected],style,color=color,
                        ms=4,label=f'{lattice}, {duration*1000:g} ms')
        ax.set(xlabel='External first-entry pump energy (mJ)',ylabel='Upward-mode energy incl. spontaneous (uJ)',
            title=f'Top {passes} pass(es); eta=0.7 assumed')
        ax.set_xscale('log');ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Uniform-inversion single-mode pulse study; fixed 300 um geometry, 500 nm gain film',fontsize=11)
    fig.tight_layout();fig.savefig(a.output/'pulse_energy_input_output.png',dpi=190);plt.close(fig)
    save()


if __name__=='__main__':
    main()
