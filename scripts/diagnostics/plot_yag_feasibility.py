"""Plot new simulation data without changing the retained optical results.

中文：所有曲线来自保存的数值样点；未收敛或不可达值不以零阈值代替。
"""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/current/yag_feasibility_20261025'
COLORS={'square':'#0072B2','triangular':'#D55E00'}


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def cw():
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    records=[]
    for ax,architecture,title in zip(axes,('edge_dual','top1','top20'),
        ('Ideal dual-edge injection','Single-pass top pumping','Ideal 20-pass top relay, r=0.98')):
        for lattice,color in COLORS.items():
            path=DATA/'cw'/f'{lattice}_{architecture}.json'
            if not path.exists():continue
            x=read(path);points=[p for p in x['points'] if p['status']=='completed']
            scale=1000 if architecture=='edge_dual' else 1
            ax.plot([p['internal_pump_W']*scale for p in points],
                    [p['upward_output_W']*1000 for p in points],'o-',color=color,
                    label=f'{lattice.capitalize()}, Pth={x["threshold"]["first_threshold_internal_W"]*scale:.3g} '+('mW' if scale==1000 else 'W'))
            for p in points:
                if p['ratio'] in (1.5,2.,4.,6.):
                    ax.annotate(str(len(p['active_modes'])),
                        (p['internal_pump_W']*scale,p['upward_output_W']*1000),
                        xytext=(0,6),textcoords='offset points',
                        fontsize=8,ha='center',color=color)
                records.append(dict(lattice=lattice,architecture=architecture,ratio=p['ratio'],
                    internal_pump_W=p['internal_pump_W'],upward_output_W=p['upward_output_W'],
                    active_mode_count=len(p['active_modes']),
                    absorbed_pump_W=p['energy_budget_W']['absorbed_pump_W'],
                    energy_residual_W=p['energy_budget_W']['energy_closure_residual_W']))
        ax.set(title=title,xlabel='First-entry internal pump ('+('mW' if architecture=='edge_dual' else 'W')+')',
               ylabel='Total upward laser output (mW)')
        ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_ylim(bottom=-.07)
    fig.suptitle('300 x 300 um Yb:YAG PCSEL | shared-inversion stationary input-output',fontsize=14)
    fig.supxlabel('Numbers: active-mode counts in the supplied 32-mode basis. Nominal spectroscopy; x-z inversion, y averaged.\nInternal entry power excludes external coupling. Multimode stationary solutions do not establish dynamic stability.',fontsize=9)
    fig.savefig(DATA/'YAG_CW_input_output.png',dpi=210);plt.close(fig)
    if records:
        with (DATA/'CW_input_output.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def size():
    cases={}
    for p in (DATA/'sizes').glob('*.json'):
        x=read(p)
        if x.get('status')!='completed' or not x.get('modes'):continue
        key=(x['lattice'],x['side_um'])
        rank=(x['order'],x['grid'],len(x.get('target_band_mode_names',[])),len(x['modes']))
        if key not in cases or rank>cases[key][0]:cases[key]=(rank,x,p)
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained'); records=[]
    for lattice,color in COLORS.items():
        values=sorted([value for (lat,_),value in cases.items() if lat==lattice],key=lambda v:v[1]['side_um'])
        xs=[];gs=[];near_x=[];near_g=[];pump_x=[];pump_y=[];relay_y=[]
        for _,x,p in values:
            length=x['side_um'];best=min(x['modes'],key=lambda r:r['material_gain_cm1'])
            band=[r for r in x['modes'] if 1029<=r['wavelength_nm']<=1031]
            xs.append(length);gs.append(best['material_gain_cm1'])
            if band:
                b=min(band,key=lambda r:r['material_gain_cm1']);near_x.append(length);near_g.append(b['material_gain_cm1'])
            th=x.get('target_band_pumps',{}).get('top1',{}).get('first_threshold_internal_W')
            relay=x.get('target_band_pumps',{}).get('top20',{}).get('first_threshold_internal_W')
            if th is not None:pump_x.append(length);pump_y.append(th);relay_y.append(relay)
            records.append(dict(lattice=lattice,side_um=length,order=x['order'],grid=x['grid'],
                minimum_sampled_material_gain_cm1=best['material_gain_cm1'],
                minimum_mode_wavelength_nm=best['wavelength_nm'],
                near1030_material_gain_cm1=min([r['material_gain_cm1'] for r in band],default=None),
                near1030_internal_top1_threshold_W=th,near1030_internal_top20_threshold_W=relay,
                source=p.relative_to(ROOT).as_posix()))
        axes[0].loglog(xs,gs,'o-',color=color,mfc='white',label=lattice.capitalize()+' / all sampled roots')
        fine=[v[1] for v in values if v[1]['order']>=24]
        if fine:
            axes[0].scatter([v['side_um'] for v in fine],
                [min(r['material_gain_cm1'] for r in v['modes']) for v in fine],
                color=color,s=28,zorder=4)
        axes[0].loglog(near_x,near_g,'s--',color=color,mfc='white',ms=5,label=lattice.capitalize()+' / 1029-1031 nm')
        axes[1].plot(pump_x,pump_y,'o-',color=color,label=lattice.capitalize())
        axes[2].plot(pump_x,relay_y,'o-',color=color,label=lattice.capitalize())
    axes[0].axhline(23.93265,color='#333333',ls=':',label='Nominal uniform-pump gain ceiling')
    axes[0].set(xlabel='Square-aperture side L (um)',ylabel='Required material gain (cm$^{-1}$)',title='Cold-cavity requirement')
    axes[1].set(xlabel='L (um)',ylabel='First-entry internal pump threshold (W)',title='Near-1030 band | one top pass')
    axes[2].set(xlabel='L (um)',ylabel='First-entry internal pump threshold (W)',title='Near-1030 band | ideal 20 passes')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Size sensitivity | fixed optimized unit cells and vertical stack',fontsize=14)
    fig.supxlabel('Each point re-solves finite modes and real-frequency closure. Filled circles: D24 or higher controls; open circles: exploratory coarse points.\nPump curves restrict 1029-1031 nm and assume nominal constant cross sections: conditional band thresholds, not full-spectrum device predictions.',fontsize=9)
    fig.savefig(DATA/'YAG_size_threshold.png',dpi=210);plt.close(fig)
    with (DATA/'size_threshold.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (DATA/'size_selected_cases.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


def material():
    fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
    pretty={'nominal':'Nominal','reciprocity_300K':'Reciprocity, 300 K',
        'reciprocity_Nt_half':r'$N_t=0.69\times10^{21}$ cm$^{-3}$',
        'reciprocity_Nt_1p5':r'$N_t=2.07\times10^{21}$ cm$^{-3}$',
        'reciprocity_tau_0p90ms':r'$\tau=0.90$ ms','reciprocity_tau_1p00ms':r'$\tau=1.00$ ms',
        'reciprocity_sigma_ap_0p5e20':r'$\sigma_{ap}=0.50\times10^{-20}$ cm$^2$',
        'reciprocity_sigma_ap_0p82e20':r'$\sigma_{ap}=0.82\times10^{-20}$ cm$^2$',
        'reciprocity_sigma_el_1p6e20':r'$\sigma_{el}=1.60\times10^{-20}$ cm$^2$',
        'reciprocity_sigma_el_2p2e20':r'$\sigma_{el}=2.20\times10^{-20}$ cm$^2$',
        'reciprocity_internal_0p5cm':r'$\alpha_{int}=0.5$ cm$^{-1}$',
        'reciprocity_internal_1cm':r'$\alpha_{int}=1.0$ cm$^{-1}$',
        'reciprocity_internal_2cm':r'$\alpha_{int}=2.0$ cm$^{-1}$'}
    for ax,architecture in zip(axes,('top1','edge_dual')):
        for offset,(lattice,color) in zip((-.18,.18),COLORS.items()):
            path=DATA/'sensitivity'/f'{lattice}.json'
            if not path.exists():continue
            x=read(path);labels=[pretty[r['label']] for r in x['records']]
            th=[r['pumps'][architecture]['first_threshold_internal_W'] for r in x['records']]
            baseline=th[0]
            ax.barh(np.arange(len(labels))+offset,[v/baseline if v is not None else np.nan for v in th],
                    height=.34,color=color,label=lattice.capitalize())
            ax.set_yticks(np.arange(len(labels)),labels,fontsize=8)
        ax.axvline(1.,color='#444',ls=':');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.set(title='Single top pass' if architecture=='top1' else 'Ideal dual-edge injection',
               xlabel='Threshold / own nominal threshold');ax.legend(fontsize=8)
    fig.suptitle('Material and extra-loss sensitivity | fixed 300-um cold-mode basis',fontsize=13)
    fig.supxlabel('One-at-a-time assumed changes. Except nominal, cross-section pairs obey 300 K Stark-manifold reciprocity.\nThis supplies no measured spectral line shape; concentration changes hold the optical index fixed.',fontsize=9)
    fig.savefig(DATA/'YAG_material_sensitivity.png',dpi=210);plt.close(fig)


if __name__=='__main__':
    DATA.mkdir(parents=True,exist_ok=True)
    cw();size();material()
    print('saved scientific figures and selected data',flush=True)
