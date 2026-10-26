"""Collect reproducible Yb:YAG optimization evidence and make static figures."""
from __future__ import annotations
import argparse
import json
import sys
import shutil
from dataclasses import asdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
from evaluate_ybyag_saved_modes import load_closed
from optimize_ybyag_cavity import save,structure
from study_ybyag_saved_sensitivity import write_csv
from pcselsim.custom_analysis import vector_far_field_complex


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);source=Path(a.source).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    summary=read(source/'summary.json');rows=read(source/'optical_modes.json')
    design,system=load_closed(source)
    threshold=next(p['threshold'] for p in summary['pump_results'] if p['architecture']=='edge_dual')
    first=next(r for r in rows if r['name']==threshold['first_modes'][0])
    cell,guide,stack,gap=structure(design);period=summary['fixed_period_nm']
    radii=np.sqrt(design.fill*np.array([design.share,1-design.share])/np.pi)*period
    centers=[[-design.dx*period/2,-design.dy*period/2],[design.dx*period/2,design.dy*period/2]]
    material=asdict(system.modes[0].medium)
    selected=dict(design=asdict(design),physical_period_nm=period,
        air_holes=[dict(center_nm=center,radius_x_nm=float(radius*np.sqrt(design.aspect)),
                       radius_y_nm=float(radius/np.sqrt(design.aspect)),angle_deg=design.angle_deg)
                   for center,radius in zip(centers,radii)],
        periodic_conservative_minimum_edge_gap_nm=gap*period,
        layer_stack_bottom_to_top=[dict(name=x.name,thickness_nm=x.thickness_nm,index=float(np.real(x.refractive_index))) for x in stack.layers],
        exterior_indices=dict(substrate=float(np.real(stack.bottom_index)),superstrate=float(np.real(stack.top_index))),
        material=material,nominal_doping_at_percent=10.,uniform_internal_power_loss_cm_inverse=.10,
        selected_mode=first,threshold=threshold,pump_results=summary['pump_results'],numerics=summary.get('numerics'),
        source=str(source.resolve()),scope=summary['scope'],
        physical_status='Conditional scalar CWT candidate, not experimentally validated device or full Maxwell pole',
        missing_physics=['measured wavelength/temperature-dependent spectra and n/k','TM and coherent TE/TM mixing',
            'all finer-grid lateral poles beyond continued seed set','3D pump mode injection and pattern scattering',
            'thermal lens/stress, dynamic stability and manufacturing characterization'])
    save(out/'selected_full_parameters.json',selected)
    write_csv(out/'closed_modes.csv',[dict(name=r['name'],wavelength_nm=r['wavelength_nm'],
        total_power_loss_cm_inverse=r['total_power_loss_per_m']/100,
        up_loss_cm_inverse=r['up_loss_per_m']/100,gain_weight=r['gain_weight'],
        frequency_residual_per_m=r['frequency_residual_per_m'],eigen_residual=r['eigen_residual'],
        ellipticity=r['ffp']['ellipticity'],center_to_peak=r['ffp']['center_to_peak'],
        radial_anisotropy_l2=r['ffp']['radial_anisotropy_l2'],
        small_signal_threshold_internal_W=threshold['branch_thresholds_internal_W'][r['name']]) for r in rows])

    base=ROOT/'results/current/optimization_20261021/yag'
    studies=ROOT/'results/current/optimization_ybyag_20261021'
    convergence=[]
    sources=[base/'recentered_gain500_D6_N21/gain500_air120_center1030_L350']
    sources+=sorted(studies.glob('convergence_*/gain500_air120_center1030_L350'))
    sources.append(source)
    for folder in sources:
        if not (folder/'summary.json').exists():continue
        s=read(folder/'summary.json');rs=read(folder/'optical_modes.json')
        r=next(r for r in rs if r['name']=='TE0_A')
        t=next(p['threshold'] for p in s['pump_results'] if p['architecture']=='edge_dual')
        n=s.get('numerics') or dict(grid_points=21,CWT_truncation=6,vertical_step_nm=5.)
        convergence.append(dict(run=str(folder.relative_to(ROOT)),**n,wavelength_nm=r['wavelength_nm'],
            first_threshold_internal_W=t['first_threshold_internal_W'],
            total_power_loss_per_m=r['total_power_loss_per_m'],ellipticity=r['ffp']['ellipticity'],
            center_to_peak=r['ffp']['center_to_peak'],radial_anisotropy_l2=r['ffp']['radial_anisotropy_l2']))
    write_csv(out/'convergence.csv',convergence);save(out/'convergence.json',convergence)
    tolerance={}
    for directory in [base/'fixed_period_3nm_budget_D4_N13_dz1',studies/'tolerance_resume_D4_N13_dz1']:
        for path in directory.glob('*/summary.json'):
            s=read(path);rs=read(path.parent/'optical_modes.json')
            t=next(p['threshold'] for p in s['pump_results'] if p['architecture']=='edge_dual')
            r=next(r for r in rs if r['name']==t['first_modes'][0])
            tolerance[s['design']['id']]=dict(case=s['design']['id'],wavelength_nm=r['wavelength_nm'],
                internal_threshold_W=t['first_threshold_internal_W'],first_modes=';'.join(t['first_modes']),
                ellipticity=r['ffp']['ellipticity'],center_to_peak=r['ffp']['center_to_peak'],
                radial_anisotropy_l2=r['ffp']['radial_anisotropy_l2'],accepted=r['ffp']['acceptance']['accepted'],
                source=str(path.parent.relative_to(ROOT)))
    write_csv(out/'fixed_period_tolerance_3nm.csv',list(tolerance.values()))
    save(out/'fixed_period_tolerance_3nm.json',list(tolerance.values()))
    alternatives=[]
    for folder in sorted((studies/'size_finalists_D8_N17').glob('L*_19poles')):
        if not (folder/'summary.json').exists():continue
        s=read(folder/'summary.json');rs=read(folder/'optical_modes.json')
        p=next(p for p in s['pump_results'] if p['architecture']=='edge_dual');t=p['threshold']
        r=next(r for r in rs if r['name']==t['first_modes'][0]);f=r['ffp']
        alternatives.append(dict(size_um=s['design']['device_um'],internal_threshold_W=t['first_threshold_internal_W'],
            next_to_first_small_signal_ratio=t['next_to_first_ratio'],first_modes=';'.join(t['first_modes']),
            wavelength_nm=r['wavelength_nm'],ellipticity=f['ellipticity'],center_to_peak=f['center_to_peak'],
            radial_anisotropy_l2=f['radial_anisotropy_l2'],full_one_degree_qualification=f['acceptance']['accepted'],
            numerics_note='D8/N17; retained four use dz2.5nm, added lateral poles dz5nm; comparison not final refinement',
            source=str(folder.relative_to(ROOT))))
    write_csv(out/'size_alternatives.csv',alternatives);save(out/'size_alternatives.json',alternatives)
    raw=np.load(source/(first['name']+'_fields.npz'))
    angle,fx,fy=vector_far_field_complex(raw['radiation_x'],raw['radiation_y'],first['wavelength_nm'],design.device_um,
                                        view_deg=1.,padding=32)
    intensity=abs(fx)**2+abs(fy)**2;intensity/=intensity.max()
    fig,axes=plt.subplots(2,2,figsize=(11,8.4),constrained_layout=True)
    ax=axes[0,0];ax.add_patch(Rectangle((-period/2,-period/2),period,period,facecolor='#c8dded',edgecolor='#4a6380'))
    for center,radius in zip(centers,radii):ax.add_patch(Circle(center,radius,facecolor='white',edgecolor='#263e55',lw=1.8))
    ax.set(xlim=(-period*.55,period*.55),ylim=(-period*.55,period*.55),xlabel='x (nm)',ylabel='y (nm)',
        title=f'Square double-hole motif: a={period:.3f} nm\nr={radii[0]:.1f}, {radii[1]:.1f} nm; fill={design.fill:.2f}')
    ax.set_aspect('equal')
    ax=axes[0,1]
    z=0.
    colors={'DBR low':'#b3cde3','DBR high':'#8c96c6'}
    for layer in stack.layers:
        color=('#efb86b' if 'gain' in layer.name else '#709fc2' if 'photonic' in layer.name else
               '#d5e8ef' if 'spacer' in layer.name else '#92a6c0' if layer.refractive_index.real>1.8 else '#c8dce8')
        ax.add_patch(Rectangle((0,z/1000),1,layer.thickness_nm/1000,facecolor=color,edgecolor='white',lw=.3))
        z+=layer.thickness_nm
    ax.text(.5,2.4,'16-pair DBR',ha='center',va='center')
    ax.text(.5,5.8,'2.0 um SiO2 spacer',ha='center',va='center')
    ax.annotate('500 nm Yb:YAG\n10 at.% nominal',xy=(1.,(z-370)/1000),xytext=(1.12,(z-550)/1000),
        va='center',fontsize=9,arrowprops=dict(arrowstyle='-',color='#526576'))
    ax.annotate('120 nm patterned YAG\nAir superstrate',xy=(1.,(z-60)/1000),xytext=(1.12,z/1000+.65),
        va='center',fontsize=9,arrowprops=dict(arrowstyle='-',color='#526576'))
    ax.set(xlim=(-.08,2.0),ylim=(0,z/1000+1.1),ylabel='z from substrate (um)',title='Proposed epitaxial/bonded stack')
    ax.set_xticks([])
    ax=axes[1,0];im=ax.pcolormesh(angle,angle,intensity,shading='auto',cmap='magma',vmin=0,vmax=1)
    ax.contour(angle,angle,intensity,levels=[.5],colors=['white'],linewidths=.8)
    ax.set(xlim=(-.5,.5),ylim=(-.5,.5),aspect='equal',xlabel='theta x (deg)',ylabel='theta y (deg)',
        title=f"Complex vector far field |Ex|^2+|Ey|^2\nellipse={first['ffp']['ellipticity']:.4f}; radial error={first['ffp']['radial_anisotropy_l2']:.4f}")
    fig.colorbar(im,ax=ax,label='Normalized intensity',fraction=.045)
    ax=axes[1,1];labels=[];values=[]
    for pump in summary['pump_results']:
        labels.append('Dual edge' if pump['architecture']=='edge_dual' else f"Top {pump.get('top_passes',20)} pass")
        values.append(pump['threshold']['first_threshold_internal_W'])
    ax.bar(labels,values,color=['#3d779c','#c57859','#718f57'][:len(values)])
    ax.set(yscale='log',ylabel='Total internal first-entry pump threshold (W)',title='Different physical pump paths')
    for i,value in enumerate(values):ax.text(i,value*1.12,f'{value:.3g} W',ha='center')
    ax.set_ylim(min(values)*.5,max(values)*2.2);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Yb:YAG candidate: conditional low-threshold, near-circular mode',fontsize=14)
    fig.savefig(out/'ybyag_optimization_overview.png',dpi=190);plt.close(fig)
    if (source/'selected_vector_ffp.png').exists():shutil.copy2(source/'selected_vector_ffp.png',out/'selected_vector_ffp.png')
    sensitivity=(studies/'material_pump_sensitivity_D12_N37' if (studies/'material_pump_sensitivity_D12_N37/summary.json').exists()
                 else studies/'material_pump_sensitivity')
    for filename in ['pump_curve_and_sensitivity.png','cw_pump_curve.csv','material_sensitivity.csv','threshold_pump_details.json','pump_architectures.json']:
        if (sensitivity/filename).exists():shutil.copy2(sensitivity/filename,out/filename)
    save(out/'summary.json',dict(selected=selected,convergence=convergence,tolerance=list(tolerance.values()),size_alternatives=alternatives,
        evidence_original_19pole_source=str((base/'recentered_gain500_19branch_N21').resolve()),
        material_pump_sensitivity=str(sensitivity.resolve())))
    print(out.resolve(),flush=True)


if __name__=='__main__':main()
