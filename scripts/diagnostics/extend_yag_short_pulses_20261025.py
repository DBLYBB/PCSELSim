"""Extend short pulses across start-up, without treating tiny beta light as lasing."""
from __future__ import annotations
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/optimization')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from yag_feasibility_study import load_saved_system,selected
from pcselsim.ybyag_pulsed import RectangularPumpPulse,simulate_pulsed_ybyag,uniform_top_pump_threshold


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=ROOT/'results/current/yag_feasibility_20261025/pulsed';out.mkdir(parents=True,exist_ok=True)
    file=out/'short_pulse_extension.json'
    records=json.loads(file.read_text(encoding='utf-8'))['cases'] if file.exists() else []
    fixtures={}
    for lattice in ('square','triangular'):
        system,summary,closed=load_saved_system(lattice)
        model=system.modes[system.names.index('TE0_pole00')]
        c=selected(lattice)
        fixtures[lattice]=dict(medium=asdict(model.medium),gain_volume_m3=model.gain_volume_m3,
            cell_widths_m=model.cell_widths_m.tolist(),qenergy=model.optical_energy_fractions.tolist(),
            qgain=model.optical_gain_weights.tolist(),vg=model.group_velocity_m_s,
            loss=model.total_power_loss_per_m,up_loss=model.upward_power_loss_per_m,
            down_loss=model.downward_power_loss_per_m,
            source_fields_path=c['fields_path'],source_fields_sha256=digest(ROOT/c['fields_path']),
            source_summary_path=c['summary_path'],source_summary_sha256=digest(ROOT/c['summary_path']))
        for passes in (1,20):
            for duration,ratios in ((1e-5,(20.,50.,100.,200.)),(1e-4,(10.,20.,50.))):
                p0=RectangularPumpPulse(0.,duration,coupling_efficiency=.7,top_passes=passes,return_efficiency=.98,after_pulse_s=.0019)
                pth=uniform_top_pump_threshold(model,p0)['external_CW_threshold_W']
                for ratio in ratios:
                    key=f'{lattice}_top{passes}_T{duration:g}_R{ratio:g}'
                    if any(r['id']==key for r in records):continue
                    pulse=replace(p0,external_energy_J=pth*duration*ratio)
                    started=time.monotonic();r=simulate_pulsed_ybyag(model,pulse,rtol=2e-6,samples=601)
                    record=dict(id=key,lattice=lattice,top_passes=passes,duration_s=duration,
                        power_to_uniform_CW_threshold=ratio,pulse=asdict(pulse),metrics=r.metrics,
                        external_fluence_J_cm2=pulse.external_energy_J/(model.medium.pumped_area_um2*1e-8),
                        elapsed_s=time.monotonic()-started)
                    records.append(record)
                    file.write_text(json.dumps(dict(scope='High-power short-pulse formal kinetics; optical damage and thermal response not evaluated',cases=records),indent=2),encoding='utf-8')
                    print(key,r.metrics['upward_output_J'],record['elapsed_s'],flush=True)
    (out/'physical_mode_inputs.json').write_text(json.dumps(dict(scope='Exact scalar cold-mode/rate inputs; independent of later main-script edits',models=fixtures),indent=2),encoding='utf-8')
    original=json.loads((out/'pulse_energy_scan.json').read_text(encoding='utf-8'))['cases']
    all_cases=original+records
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    durations=[1e-5,1e-4,.001,.005];colors=plt.cm.viridis(np.linspace(.08,.9,4))
    for ax,passes in zip(axes,(1,20)):
        for lattice,style in [('square','o-'),('triangular','s--')]:
            for color,duration in zip(colors,durations):
                group=sorted([r for r in all_cases if r['lattice']==lattice and r['top_passes']==passes and r['duration_s']==duration],key=lambda r:r['pulse']['external_energy_J'])
                ax.plot([r['pulse']['external_energy_J']*1000 for r in group],
                    [r['metrics']['upward_output_J']*1e6 for r in group],style,color=color,ms=4,
                    label=f'{lattice}, {duration*1000:g} ms')
        ax.set(xlabel='External first-entry pump energy (mJ)',ylabel='Upward-mode energy incl. spontaneous (uJ)',
            title=f'Top {passes} pass(es); eta=0.7 assumed')
        ax.set_xscale('log');ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Uniform one-mode energy response; short pulses extended through start-up',fontsize=11)
    fig.tight_layout();fig.savefig(out/'pulse_energy_input_output_extended.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for row,duration in enumerate((1e-5,1e-4)):
        for col,passes in enumerate((1,20)):
            ax=axes[row,col]
            for lattice,color,marker in [('square','#0072B2','o'),('triangular','#D55E00','s')]:
                group=sorted([r for r in all_cases if r['lattice']==lattice and r['top_passes']==passes and r['duration_s']==duration],key=lambda r:r['pulse']['external_energy_J'])
                x=np.array([r['pulse']['external_energy_J']*1000 for r in group])
                y=np.array([r['metrics']['upward_output_J']*1e6 for r in group])
                crossed=np.array([r['metrics']['maximum_inversion_fraction']>r['metrics']['uniform_CW_threshold']['threshold_inversion_fraction'] for r in group])
                ax.plot(x,y,color=color,lw=1.5,label=lattice)
                ax.scatter(x[~crossed],y[~crossed],marker=marker,edgecolors=color,facecolors='white',s=30,zorder=3)
                ax.scatter(x[crossed],y[crossed],marker=marker,edgecolors=color,facecolors=color,s=30,zorder=3)
            ax.set(xlabel='External first-entry pump energy (mJ)',ylabel='Upward-mode energy (uJ)',
                title=f'{duration*1e6:g} us pump; top {passes} pass(es)')
            ax.set_xscale('log');ax.grid(alpha=.25);ax.legend(fontsize=9)
    fig.suptitle('Formal single-mode start-up: open markers never reach net-gain threshold',fontsize=12)
    fig.text(.5,.012,'All outputs include beta seeds. eta=.7 and relay=.98 assumed; thermal/damage limits are untested.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.96));fig.savefig(out/'short_pulse_startup.png',dpi=190);plt.close(fig)
    file.write_text(json.dumps(dict(status='completed',cases=records,original_case_count=len(original),
        extended_case_count=len(records),combined_case_count=len(all_cases),
        scope='Formal uniform single-mode kinetics; short-pulse fluence/peak power is not a damage-qualified operating point'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
