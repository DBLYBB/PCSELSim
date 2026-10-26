"""Independent read-only QA and scientific figures for the strict YAG study.

中文：不重跑主脉冲/CW，不覆盖尺寸数据；只审查已有记录并绘制仿真科学图。
Missing/failed/unreachable thresholds remain null and are never plotted as0.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='1'
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
from datetime import datetime,timezone
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/optimization')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import strict_yag_feasibility_20261025 as driver
from pcselsim.custom_analysis import finite_area_eigenproblem
from pcselsim.triangular_finite import TriangularFiniteSpec,triangular_finite_eigenproblem
from pcselsim.finite_discretization import relative_residual
from pcselsim.mode_identity import independent_eigenmode

BASE=ROOT/'results/current/strict_rerun_20261025'
OUT=BASE/'yag_feasibility'
SOURCE_HASH={}
COLORS={'square':'#176a9c','triangular':'#c66216'}
LABELS={'square':'Square','triangular':'Triangular'}
ARCH={'edge_dual':'Dual-edge pump','top1':'Top: one pass','top20':'Top: ideal 20 passes'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.facecolor':'white'})


def read(path):
    path=Path(path)
    for attempt in range(4):
        try:
            payload=path.read_bytes();value=json.loads(payload)
            SOURCE_HASH[str(path.resolve())]=hashlib.sha256(payload).hexdigest()
            return value
        except json.JSONDecodeError:
            if attempt==3:raise
            time.sleep(.15)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(driver.f.clean(value),ensure_ascii=False,indent=2),encoding='utf8')


def finite_nonnegative(value,tol=1e-18):
    return bool(np.isfinite(value) and value>=-tol)


def pulse_qa(lattice,system,rows):
    path=OUT/'pulsed'/f'{lattice}_energy_scan.json'
    if not path.exists():return dict(status='missing',path=str(path),records=[]),None
    data=read(path);cold=min(rows,key=lambda r:r['required_material_gain_cm1'])
    checks=[];fine=[];ids=[]
    for row in data['cases']:
        p,m=row['pulse'],row['metrics'];ids.append(row['id'])
        energy=p['external_energy_J'];duration=p['duration_s'];eta=p['coupling_efficiency']
        denominator_ok=np.isfinite([energy,duration,eta]).all() and energy>0 and duration>0 and 0<eta<=1
        expected=row['threshold']['external_CW_threshold_W']*duration*row['power_to_uniform_CW_threshold']
        budget=(m['absorbed_pump_J']-m['quantum_defect_J']-m['fluorescence_J']
            -m['upward_output_J']-m['downward_output_J']-m['other_cavity_loss_J']
            -m['stored_effective_energy_final_J']+m['stored_effective_energy_initial_J'])
        boundary=energy-m['before_gain_loss_J']-m['absorbed_pump_J']-m['escaped_pump_J']-m['relay_loss_J']
        scale=max(energy,1e-30)
        nonnegative=all(finite_nonnegative(m[key],max(1e-18,scale*1e-10)) for key in (
            'external_pump_J','before_gain_loss_J','absorbed_pump_J','escaped_pump_J','relay_loss_J',
            'quantum_defect_J','fluorescence_J','upward_output_J','downward_output_J','other_cavity_loss_J',
            'stored_effective_energy_final_J','stored_effective_energy_initial_J'))
        nonnegative=nonnegative and finite_nonnegative(m['peak_upward_power_W'],1e-15)
        errors=dict(input_definition=abs(energy-expected)/scale,
            integrated_external=abs(m['external_pump_J']-energy)/scale,
            coupling_loss=abs(m['before_gain_loss_J']-(1-eta)*energy)/scale,
            effective_energy=abs(budget)/scale,pump_boundary=abs(boundary)/scale,
            reported_effective_residual_difference=abs(budget-m['energy_closure_residual_J'])/scale,
            reported_boundary_residual_difference=abs(boundary-m['pump_boundary_residual_J'])/scale)
        mode_ok=(row['first_cold_mode']==cold['name'] and abs(row['wavelength_nm']-cold['wavelength_nm'])<1e-9)
        population_ok=(-1e-8<=m['final_inversion_fraction']<=1+1e-8 and
            -1e-8<=m['maximum_inversion_fraction']<=row['threshold']['pump_limited_fraction']+1e-6)
        beta_ok=bool(0<=p['spontaneous_beta']<=1 and p['spontaneous_beta']==m['beta_spontaneous'])
        passed=bool(denominator_ok and nonnegative and mode_ok and population_ok and beta_ok
            and max(errors.values())<1e-6 and eta==.7 and abs(m['pump_peak_external_W']-energy/duration)<max(1e-9,energy/duration*1e-10))
        checks.append(dict(id=row['id'],passed=passed,mode_ok=mode_ok,denominators_positive=bool(denominator_ok),
            energy_and_power_nonnegative=nonnegative,population_bounds_pass=population_ok,relative_errors=errors,
            spontaneous_beta_consistent=beta_ok,
            input_external_J=energy,output_upward_J=m['upward_output_J'],
            output_over_external=m['upward_output_J']/energy))
        if 'accuracy_challenge' in row:
            q=row['accuracy_challenge'];actual=abs(q['metrics']['upward_output_J']-m['upward_output_J'])/max(abs(q['metrics']['upward_output_J']),1e-30)
            fine.append(dict(id=row['id'],passed=bool(actual<.005 and q['rtol']<=m['rtol']/9.9 and
                abs(actual-q['relative_upward_energy_change'])<1e-12),coarse_rtol=m['rtol'],fine_rtol=q['rtol'],relative_upward_energy_change=actual))
    expected_ids={f'{lattice}_top{passes}_T{duration:g}_R{ratio:g}' for passes in (1,20)
        for duration in (1e-5,1e-4,.001,.005) for ratio in (.5,1.,1.2,2.,4.,8.)}
    complete=data['status']=='completed' and len(ids)==48 and set(ids)==expected_ids
    trace_checks=[]
    for trace in (OUT/'pulsed').glob(f'{lattice}_*_trace.npz'):
        SOURCE_HASH[str(trace.resolve())]=hashlib.sha256(trace.read_bytes()).hexdigest()
        with np.load(trace,allow_pickle=False) as raw:
            t,f,p=raw['time_s'],raw['inversion'],raw['upward_W']
            passed=bool(np.isfinite(t).all() and np.isfinite(f).all() and np.isfinite(p).all()
                and np.all(np.diff(t)>0) and t[0]==0 and f[0]==0 and p[0]==0
                and f.min()>-1e-8 and f.max()<1+1e-8
                and p.min()>-1e-10)
            trace_checks.append(dict(path=str(trace),passed=passed,display_samples=len(t),
                minimum_inversion=float(f.min()),maximum_inversion=float(f.max()),minimum_upward_W=float(p.min())))
    status='passed' if all(r['passed'] for r in checks+fine+trace_checks) else 'failed'
    return dict(status=status,source_status=data['status'],all48_present=complete,records=checks,
        unique_id_count=len(set(ids)),accuracy_challenges=fine,trace_checks=trace_checks,
        qa_scope='One top20/1ms/2x case has10x tighter rtol. Traces are display samples, not fast-spike/FFT resolution certification.'),data


def cw_qa(lattice,system):
    reports=[];datasets=[];maxloss=max(m.total_power_loss_per_m for m in system.modes)
    for architecture in ('edge_dual','top1','top20'):
        path=OUT/'cw'/f'{lattice}_{architecture}.json'
        if not path.exists():reports.append(dict(architecture=architecture,status='missing'));continue
        data=read(path);checks=[];threshold=data['threshold']['first_threshold_internal_W']
        for row in data['points']:
            if row['status']!='completed':checks.append(dict(ratio=row['ratio'],passed=False,error=row.get('error')));continue
            b=row['energy_budget_W'];power=row['internal_pump_W'];outputs=row['output_by_mode_W']
            energy=(b['stored_excitation_plus_photon_energy_rate_W']-b['absorbed_pump_W']
                +b['pump_quantum_defect_reference_W']+b['fluorescence_outside_modes_W']
                +b['upward_output_W']+b['downward_output_W']+b['other_cavity_loss_W']
                +b['intermode_stimulated_thermalization_W'])
            boundary=power-b['absorbed_pump_W']-b['pump_escape_W']-b['pump_relay_loss_W']
            scale=max(power,1.)
            errors=dict(power_definition=abs(power-row['ratio']*threshold),
                output_sum=abs(sum(outputs.values())-row['upward_output_W']),
                energy=abs(energy),pump_boundary=abs(boundary),
                reported_energy_difference=abs(energy-b['energy_closure_residual_W']),
                reported_boundary_difference=abs(boundary-b['pump_boundary_flow_residual_W']))
            nonnegative=(all(finite_nonnegative(value,1e-14) for value in outputs.values()) and
                all(finite_nonnegative(b[k],1e-12) for k in ('incident_internal_pump_W','absorbed_pump_W',
                    'pump_quantum_defect_reference_W','fluorescence_outside_modes_W','upward_output_W',
                    'downward_output_W','other_cavity_loss_W','pump_escape_W','pump_relay_loss_W')))
            population_scaled=row['population_residual_per_s']*system.modes[0].medium.upper_state_lifetime_ms*1e-3
            passed=bool(set(outputs)==set(system.names) and set(row['active_modes'])<=set(system.names)
                and nonnegative and max(errors.values())<1e-8*scale and row['complementarity_residual']<2e-7
                and row['maximum_inactive_gain_excess_per_m']<2e-7*maxloss and population_scaled<2e-7)
            checks.append(dict(ratio=row['ratio'],passed=passed,absolute_errors_W=errors,
                energy_power_nonnegative=nonnegative,active_modes=row['active_modes'],
                population_residual_times_lifetime=population_scaled,complementarity_residual=row['complementarity_residual']))
        expected=[0.,.5,.9,.99,1.01,1.05,1.1,1.25,1.5,2.,3.,4.,6.]
        complete=(data['status']=='completed' and [r['ratio'] for r in data['points']]==expected)
        reports.append(dict(architecture=architecture,status='passed' if all(r['passed'] for r in checks) else 'failed',
            source_status=data['status'],all13_present=complete,retained_modes=data['retained_modes'],checks=checks,
            first_threshold_internal_W=threshold,first_modes=data['threshold']['first_modes']))
        datasets.append(data)
    return reports,datasets


def size_stored_pencil_qa(data,path):
    """Independent topology + saved own-C check, without new eigenvalue solves.

    中文：用实际L/N重建K/M拓扑、读回每根自身C；不重新扫描或再求本征根。
    Fresh frequency-specific C validation remains the size runner's separate record.
    """
    if data.get('anchor'):
        rawpath=Path(data['source_fields_path'])
    else:rawpath=path.with_suffix('.npz')
    if not rawpath.exists():return [dict(passed=False,error='Missing own complex-field archive',path=str(rawpath))]
    SOURCE_HASH[str(rawpath.resolve())]=hashlib.sha256(rawpath.read_bytes()).hexdigest()
    checks=[];values=[];fields=[];wavelengths=[]
    with np.load(rawpath,allow_pickle=False) as raw:
        for row in data['modes']:
            name=row['name'];C=raw[name+'_C'];field=raw[name+'_field'];value=complex(row['delta_per_m'],row['alpha_field_per_m'])
            if data['lattice']=='square':
                # Only these physical topology fields enter this function.
                spec=SimpleNamespace(grid_points=data['grid'],domain_um=data['side_um'],discretization='staggered')
                problem=finite_area_eigenproblem(spec,C)
            else:
                spec=TriangularFiniteSpec(data['side_um']/2,data['grid'],0.,8,'square','staggered')
                problem,_=triangular_finite_eigenproblem(SimpleNamespace(coupling_m=C),spec)
            residual=relative_residual(problem.center_operator(),field,value)
            budget=problem.loss_budget(field)
            flux_error=abs(budget['local_amplitude_loss_per_m']+budget['transport_amplitude_loss_per_m']-value.imag)
            qg,qe=raw[name+'_qgain'],raw[name+'_qenergy'];vg=driver.c*qe.sum()/(1.8166*qg.sum())
            gain=(2*value.imag+10.)/qg.sum()/100
            Lup,Ldown=raw[name+'_Lup'],raw[name+'_Ldown']
            optical=float(np.linalg.norm((C-C.conj().T)/(2j)-Lup-Ldown)/max(np.linalg.norm(Lup+Ldown),1e-30))
            peers=[i for i,w in enumerate(wavelengths) if abs(w-row['wavelength_nm'])<=1e-7]
            independent=independent_eigenmode(value,field,[values[i] for i in peers],[fields[i] for i in peers])
            physical_weights=bool(np.isfinite(qg).all() and np.isfinite(qe).all() and qg.min()>=0
                and qe.min()>=0 and qg.sum()>0 and 0<qe.sum()<=1+1e-9
                and (raw[name+'_widths']>0).all() and np.allclose(qg,driver.c/(1.8166*vg)*qe,rtol=2e-12,atol=1e-16))
            passed=bool(residual<2e-7 and flux_error<1e-4 and optical<1e-8 and independent
                and abs(gain-row['material_gain_cm1'])<1e-8 and abs(row['frequency_residual_per_m'])<.05 and physical_weights)
            checks.append(dict(name=name,passed=passed,original_KM_residual=residual,
                amplitude_flux_error_per_m=flux_error,optical_theorem_residual=optical,
                gain_readback_difference_cm1=abs(gain-row['material_gain_cm1']),independent=bool(independent),
                physical_weights_pass=physical_weights,stored_frequency_residual_per_m=row['frequency_residual_per_m']))
            values.append(value);fields.append(field);wavelengths.append(row['wavelength_nm'])
    return checks


def sizes_review(lattice):
    records=[];selected=[]
    for path in sorted((OUT/'sizes').glob(f'{lattice}_*.json')):
        data=read(path);modes=data.get('modes',[]);valid=data.get('status')=='completed' and bool(modes)
        if data.get('strict_original_field_readback'):
            valid=valid and all(r['passed'] for r in data['strict_original_field_readback'])
        ownchecks=size_stored_pencil_qa(data,path) if valid else []
        valid=valid and all(r['passed'] for r in ownchecks)
        records.append(dict(path=str(path),status=data.get('status'),physically_valid_record=valid,
            side_um=data.get('side_um'),mode_count=len(modes),closure_failures=data.get('closure_failures',[]),
            order=data.get('order'),grid=data.get('grid'),dz_nm=data.get('dz_nm'),
            independent_stored_pencil_checks=ownchecks,
            failed_census_targets=[r for r in data.get('census',{}).get('attempts',[]) if 'error' in r],
            readback_checks=data.get('strict_original_field_readback',[])))
        if not valid:continue
        cold=min(modes,key=lambda r:r['material_gain_cm1'])
        first_wavelengths={};target_first_wavelengths={}
        for arch,pump in data['pumps'].items():
            first_wavelengths[arch]=[next(r['wavelength_nm'] for r in modes if r['name']==name)
                for name in pump.get('first_modes',[])]
        for arch,pump in data.get('target_band_pumps',{}).items():
            target_first_wavelengths[arch]=[next(r['wavelength_nm'] for r in modes if r['name']==name)
                for name in pump.get('first_modes',[])]
        selected.append(dict(path=str(path),side_um=data['side_um'],order=data['order'],dz_nm=data['dz_nm'],
            grid=data['grid'],anchor=data.get('anchor',False),period_nm=data['period_nm'],
            cold_mode=cold['name'],cold_wavelength_nm=cold['wavelength_nm'],material_gain_cm1=cold['material_gain_cm1'],
            pumps=data['pumps'],target_band_pumps=data.get('target_band_pumps',{}),
            target_band_nm=data.get('target_band_nm'),mode_count=len(modes),
            pump_first_wavelengths_nm=first_wavelengths,
            target_band_first_wavelengths_nm=target_first_wavelengths,
            closure_failure_count=len(data.get('closure_failures',[])),scope=data.get('scope')))
    best=[]
    for length in sorted({r['side_um'] for r in selected}):
        available=[r for r in selected if r['side_um']==length]
        best.append(max(available,key=lambda r:(r['anchor'],r['order'],r['grid'],-r['dz_nm'])))
    return records,best


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def figures(cw,pulsed,sizes):
    plots=OUT/'science_plots';plots.mkdir(exist_ok=True)
    written=[]
    def finish(fig,name):
        path=plots/name;fig.savefig(path,dpi=190,bbox_inches='tight');plt.close(fig)
        written.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    def quality(row):
        return 'anchor' if row['anchor'] else ('checked' if row['order']>=32 else 'exploratory')
    quality_handles=[Line2D([],[],color='.35',marker='o',ls='',mfc='white',label='Exploratory D12'),
        Line2D([],[],color='.35',marker='o',ls='',label='Checked D32 / finer grid'),
        Line2D([],[],color='.35',marker='*',ms=10,ls='',label='Formal300 um merged anchor')]
    def quality_markers(ax,rows,ys,color):
        for row,y in zip(rows,ys):
            if not np.isfinite(y):continue
            q=quality(row)
            ax.plot(row['side_um'],y,'*' if q=='anchor' else 'o',ms=10 if q=='anchor' else 5,
                mfc='white' if q=='exploratory' else color,mec=color,ls='',zorder=4)
    if cw:
        fig,axes=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
        for lattice,datasets in cw.items():
            for ax,architecture in zip(axes,('edge_dual','top1','top20')):
                data=next((d for d in datasets if d['architecture']==architecture),None)
                if data is None:continue
                pts=[r for r in data['points'] if r['status']=='completed']
                ax.plot([r['internal_pump_W'] for r in pts],[r['upward_output_W']*1e3 for r in pts],
                    'o-',ms=3,color=COLORS[lattice],label=LABELS[lattice])
                ax.axvline(data['threshold']['first_threshold_internal_W'],color=COLORS[lattice],ls=':',alpha=.65)
                ax.set(title=ARCH[architecture],xlabel='Internal first-entry pump (W)',ylabel='Upward output (mW)')
                ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Stationary shared-inversion CW; frozen cold fields, nominal flat cross sections')
        finish(fig,'CW_input_output.png')
        fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),layout='constrained')
        architectures=('edge_dual','top1','top20');centers=np.arange(3);count=len(cw);width=.68/max(count,1)
        for index,(lattice,datasets) in enumerate(cw.items()):
            absorbed=[];upward=[]
            for architecture in architectures:
                data=next((d for d in datasets if d['architecture']==architecture),None)
                point=next((r for r in (data or {}).get('points',[]) if r.get('status')=='completed' and r['ratio']==2.),None)
                if point is None or point['internal_pump_W']<=0:
                    absorbed.append(np.nan);upward.append(np.nan);continue
                absorbed.append(point['energy_budget_W']['absorbed_pump_W']/point['internal_pump_W'])
                upward.append(point['upward_output_W']/point['internal_pump_W'])
            x=centers+(index-(count-1)/2)*width
            for ax,values in zip(axes,(absorbed,upward)):
                ax.bar(x,values,width=width*.9,color=COLORS[lattice],label=LABELS[lattice])
        for ax,label in zip(axes,('Absorbed pump / internal first entry','Upward mode output / internal first entry')):
            ax.set(yscale='log',ylabel=label,xticks=centers,xticklabels=['Dual edge','Top1 pass','Top20 ideal passes'])
            ax.grid(alpha=.2,axis='y',which='both');ax.legend()
        fig.suptitle('At twice each own threshold; absorption and first-entry conversion fractions')
        finish(fig,'CW_power_flow_fractions.png')
    for lattice,data in pulsed.items():
        fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
        for ax,passes in zip(axes,(1,20)):
            for duration,color in zip((1e-5,1e-4,.001,.005),plt.cm.viridis(np.linspace(.05,.85,4))):
                pts=sorted([r for r in data['cases'] if r['top_passes']==passes and r['duration_s']==duration],
                    key=lambda r:r['pulse']['external_energy_J'])
                if not pts:continue
                ax.loglog([r['pulse']['external_energy_J']*1e3 for r in pts],
                    [r['metrics']['upward_output_J']*1e6 if r['metrics']['upward_output_J']>0 else np.nan for r in pts],
                    'o-',color=color,ms=3,label=f'{duration*1e6:g} us')
            ax.set(title=f'Top pump: {passes} pass'+(' (ideal relay)' if passes==20 else ''),
                xlabel='External pulse energy (mJ); entry coupling=0.7',
                ylabel='Upward mode energy including\nspontaneous emission (uJ)')
            ax.grid(alpha=.2,which='both');ax.legend(title='Pump duration')
        fig.suptitle(LABELS[lattice]+' lattice: uniform-inversion single-mode pulses')
        finish(fig,f'{lattice}_pulse_energy_input_output.png')
    if pulsed:
        fig,axes=plt.subplots(1,len(pulsed),figsize=(5.7*len(pulsed),4.3),layout='constrained',squeeze=False)
        for ax,(lattice,data) in zip(axes[0],pulsed.items()):
            for duration,color in zip((1e-5,1e-4,.001,.005),plt.cm.viridis(np.linspace(.05,.85,4))):
                pts=sorted([r for r in data['cases'] if r['top_passes']==20 and r['duration_s']==duration],
                    key=lambda r:r['power_to_uniform_CW_threshold'])
                ax.plot([r['power_to_uniform_CW_threshold'] for r in pts],
                    [r['metrics']['maximum_inversion_fraction']/r['threshold']['threshold_inversion_fraction'] for r in pts],
                    'o-',color=color,ms=3,label=f'{duration*1e6:g} us')
            ax.axhline(1.,color='.35',ls='--',lw=1)
            ax.set(title=LABELS[lattice]+'; ideal20-pass top pump',
                xlabel='External power / own uniform CW reference',ylabel='Maximum inversion / threshold inversion')
            ax.grid(alpha=.2);ax.legend(title='Pump duration',fontsize=8)
        fig.suptitle('Uniform one-mode pulse inversion; crossing1 is necessary, not a full lasing certification')
        finish(fig,'pulse_inversion_startup.png')
    if any(sizes.values()):
        fig,ax=plt.subplots(figsize=(7,4.6),layout='constrained')
        for lattice,rows in sizes.items():
            if not rows:continue
            values=[r['material_gain_cm1'] for r in rows]
            ax.plot([r['side_um'] for r in rows],values,'-',color=COLORS[lattice],label=LABELS[lattice])
            quality_markers(ax,rows,values,COLORS[lattice])
        ax.axhline(23.93265,color='.4',ls='--',label='Nominal pump-limited material gain ceiling')
        ax.set(yscale='log',xlabel='Square physical aperture side (um)',ylabel='Lowest documented material gain (cm^-1)',
            title='Fixed final unit cell/period/stack; new poles at each aperture')
        ax.set_xticks([50,100,200,300,500,750]);ax.set_xlim(35,775)
        ax.grid(alpha=.2,which='both')
        handles,labels=ax.get_legend_handles_labels();ax.legend(handles=handles+quality_handles,fontsize=8)
        finish(fig,'size_material_gain.png')
        fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
        for lattice,rows in sizes.items():
            for ax,arch in zip(axes,('edge_dual','top1','top20')):
                for target,style,label in ((False,'-','all retained roots'),(True,'--','1029-1031 nm subset')):
                    values=[r['target_band_pumps' if target else 'pumps'].get(arch,{}).get('first_threshold_internal_W') for r in rows]
                    # None means unreachable/unresolved/no supplied in-window mode: never0.
                    y=[float(v) if v is not None and v>0 else np.nan for v in values]
                    ax.plot([r['side_um'] for r in rows],y,style,color=COLORS[lattice],
                        label=LABELS[lattice]+': '+label)
                    quality_markers(ax,rows,y,COLORS[lattice])
                missing=[r['side_um'] for r in rows if r['pumps'].get(arch,{}).get('first_threshold_internal_W') is None]
                ax.plot(missing,[1.02]*len(missing),'x',color=COLORS[lattice],
                    transform=ax.get_xaxis_transform(),clip_on=False,ms=5)
                ax.set(yscale='log',title=ARCH[arch],xlabel='Physical side (um)',ylabel='Internal threshold (W)')
                ax.set_xlim(35,775);ax.set_xticks([50,100,200,300,500,750]);ax.tick_params(axis='x',rotation=45)
                ax.grid(alpha=.2,which='both')
        axes[-1].legend(fontsize=8)
        fig.legend(handles=quality_handles,loc='lower center',bbox_to_anchor=(.5,-.065),ncol=3,fontsize=8)
        fig.suptitle('Top-edge x: no finite total-spectrum threshold; wavelength subset is conditional')
        finish(fig,'size_internal_pump_threshold.png')
    return written


def write_report(result,cw,pulsed,sizes):
    """Keep prose tied to the current snapshot, including unfinished work."""
    link='../results/current/strict_rerun_20261025/yag_feasibility/'
    lines=['# 严格复算后 Yb:YAG 微腔的泵浦、能量与尺寸可行性分析','',
        '本报告对应本轮重新选定的正方/三角晶格单元，不沿用上一轮旧几何的速率或尺寸曲线。'
        +('要求的两腔任务及关键尺寸精度控制均已有完成记录。' if result['all_requested_runs_complete'] else
          '当前为随计算更新的科学报告；缺失或仍在计算的三角晶格/精度控制不被记为完成。'),'',
        '低材料阈值、所给模型内泵浦可达、能量守恒三者均不等于高效率、可制造或已验证热稳定。尤其不能把约38mW的理想内部双侧泵阈值直接称为容易实现的外部毫瓦泵浦。','',
        '## 1. 本轮结果与证据边界','',
        '| 晶格 | 实际首冷模波长nm | 最低材料g/cm^-1 | 独立闭频根数 | 脉冲记录 | 共享CW点 | 记录QA |',
        '|---|---:|---:|---:|---:|---:|---|']
    for lattice,review in result['reports'].items():
        if review['status']=='awaiting_final_selected_case':
            lines.append(f'| {LABELS[lattice]} | 待最终选定 | — | — | — | — | 未完成 |');continue
        p=review['optical_reference'];n=sum(len(r['checks']) for r in review['cw'])
        lines.append(f"| {LABELS[lattice]} | {p['wavelength_nm']:.9f} | {p['material_gain_cm1']:.6f} | {review['retained_mode_count']} | {len(review['pulse']['records'])}/48 | {n}/39 | {review['status']} |")
    lines += ['', '| 晶格 | ΣqGain | 模型光子寿命ps | 模型损耗Q | 向上损耗/总功率损耗 |',
        '|---|---:|---:|---:|---:|']
    for lattice,review in result['reports'].items():
        if 'optical_reference' not in review:continue
        p=review['optical_reference']
        lines.append(f"| {LABELS[lattice]} | {p['gain_weight']:.6f} | {p['photon_lifetime_s']*1e12:.6f} | {p['cold_loss_Q']:.5g} | {p['upward_loss_fraction']:.6f} |")
    lines += ['', '此处τ_ph=1/(vgκ)、Q=2πcτ_ph/λ来自当前近Bragg速率接口，κ包含声明的附加内损耗，但不重复包含σal描述的再吸收。向上端口比例不是外部收集效率；侧向泄漏等其余端口仍消耗腔光子。']
    lines+=['','上述根数是有记录的有限候选集合。即使原K/M本征残差、固定周期频率闭合、能流及身份检查通过，也不能称为穷尽全Maxwell谱。不同频率的根保留各自C、qGain/qEnergy、群速度、辐射与激光波长；不能先用1030nm代替所有竞争模波长。','',
        '## 2. 层结构、材料和实际输入量','',
        '两腔均用300×300µm的正方形物理边界、500nm有源Yb:YAG、120nm被动PC、2µm SiO2间隔和16对DBR。尺寸扫描只改边界边长L，固定最终晶胞、周期与层结构；300µm点显式读取最终合并本征根作为锚点。','',
        '名义材料Nt=1.38×10^21cm^-3（约10at.%的输入假设），τ=0.95ms，泵波长940nm。截面为σap=.70、σep=.10、σal=.126、σel=2.0，单位均10^-20cm²。这是常数比较组，不能拼接称为当前500nm、图形化、10at.%薄膜的联合实测参数。','',
        '[Körner等，JOSA B29,2493(2012)](https://opg.optica.org/josab/abstract.cfm?uri=josab-29-9-2493)测量温度相关吸收和荧光，并结合Füchtbauer–Ladenburg与倒易关系获得截面。其表1能级和表2寿命支持本轮300K配对敏感性及0.95ms量级；没有据此生成1024nm等波长的实测发射谱。','',
        '[Li等，Optics Letters48,5359(2023)，DOI10.1364/OL.502795](https://opg.optica.org/ol/abstract.cfm?uri=ol-48-20-5359)以Yb:YAG晶膜微盘实现WGM激光，采用946nm泵并报告1.1mW输出。这支持Yb:YAG本体微腔路线，但不是本PCSEL结构、泵耦合或效率的实验验证。其27%最大斜率不可直接套用本轮结构。','',
        '## 3. 材料增益、泵浦阈值与单位','',
        r'光学本征值μ=δ+iα的虚部α是振幅损耗/m；速率模型使用功率损耗κ=2α+10/m，10/m对应0.1cm^-1附加内损耗。材料阈值为g_th=κ/(100ΣqGain)，并非只把Im(μ)换算成cm^-1。','',
        r'局部净增益g_i=Nt[(σel+σal)f_i−σal]；泵吸收αp_i=Nt[σap(1−f_i)−σep f_i]。体积/能量权重与增益权重保留全模归一化，不在有源层内重新归一成1。vg从同模保存的qGain=(c/(n_active vg))qEnergy恢复并逐z核验。','',
        r'均匀反转的阈值f_th=[g_th/Nt+σal]/(σel+σal)，泵极限f_max=σap/(σap+σep)=0.875。名义材料的最大可达净增益约23.93265cm^-1；超过该上限的已计算根不能靠无限加泵达到阈值。尺寸图中的null保留为不可达/未包根，不画成0。','',
        'ODE内部使用m/s、m^-1、m^-3和m²；Nt[cm^-3]乘10^6、σ[cm²]乘10^-4转成SI。CW功率是进入增益区的全部内部首入射总W。脉冲输入是外部矩形泵能量J，显式使用假设入口耦合η=0.7；E_internal=ηE_external。20程把同一批幸存泵光按98%回程效率交给下一程，不把每一程再计为新的外部能量。','',
        '## 4. 500nm薄膜吸收与泵强度口径','',
        '未反转时αp=Ntσap=9.66cm^-1，500nm单程光学厚度4.83×10^-4，吸收率约0.04829%。接近激光阈值时，基态耗尽与泵受激发射会进一步降低净吸收。薄膜单程吸收弱是顶泵阈值高的重要来源。','',
        '| 晶格 | 理想内部双侧泵总阈值W | 每侧入口平均强度kW/cm² | 单程顶泵阈值W | 顶部平均强度kW/cm² |',
        '|---|---:|---:|---:|---:|']
    for lattice,data in cw.items():
        edge=next((d for d in data if d['architecture']=='edge_dual'),None)
        top=next((d for d in data if d['architecture']=='top1'),None)
        if edge is None or top is None:continue
        ep=edge['threshold']['first_threshold_internal_W'];tp=top['threshold']['first_threshold_internal_W']
        lines.append(f'| {LABELS[lattice]} | {ep:.9f} | {ep/2/(150e-8)/1000:.4f} | {tp:.6f} | {tp/(90000e-8)/1000:.4f} |')
    lines+=['','单边横截面积仅500nm×300µm=150µm²，双侧总功率必须先分半再除以各自入口面积。这里的强度仍是在亚微米有源层内的理想入口平均值；尚未设计可把外部泵光有效耦入500nm膜的NA、束腰、端面/图形层散射与损伤限制。不能由总edge功率小推断实验泵耦合容易。','',
        '顶部面积300×300µm²=9×10^-4cm²。入射强度与吸收发热量也不同：多数单程泵光透射逃逸，不能把全部21W左右入射光都算成膜内发热；反过来，有效能量账本也没有给出真实温升或损伤阈值。','',
        '300K热化能级倒易配对的首阈值敏感性如下。这里仍假设σel恒定2×10^-20cm²；小幅变化不能当成完整材料不确定性的误差条。','',
        '| 晶格/协议 | 内部edge首阈W | 内部top1首阈W |',
        '|---|---:|---:|']
    for lattice,review in result['reports'].items():
        for record in review.get('spectroscopy_protocols',[]):
            edge=record['pumps']['edge_dual']['threshold']['first_threshold_internal_W']
            top=record['pumps']['top_single']['threshold']['first_threshold_internal_W']
            label='nominal常σ' if record['label'].startswith('nominal') else '300K配对/常发射包络'
            lines.append(f'| {LABELS[lattice]}/{label} | {edge:.9f} | {top:.6f} |')
    lines += ['',
        '## 5. 共享空间反转的CW输入—输出','',
        f'![CW输入输出]({link}science_plots/CW_input_output.png)','',
        '模型在x-z上共用反转并对y平均；所有提供的闭频模式共同耗尽同一反转。不是把各单模独立输出相加。每架构13个实际点包含0、0.5、0.9、0.99、1.01、1.05、1.1、1.25、1.5、2、3、4、6倍该架构首阈值；连线只连接实际记录，不代表额外扫描点或拟合定律。','',
        '| 晶格/泵型 | 首阈值内部W | 4×向上输出mW/活动模数 | 6×向上输出mW/活动模数 |',
        '|---|---:|---:|---:|']
    for lattice,data in cw.items():
        for d in data:
            four=next((p for p in d['points'] if p.get('status')=='completed' and p['ratio']==4),None)
            six=next((p for p in d['points'] if p.get('status')=='completed' and p['ratio']==6),None)
            def cell(p):return '未完成' if p is None else f"{p['upward_output_W']*1e3:.6f}/{len(p['active_modes'])}"
            lines.append(f"| {LABELS[lattice]}/{ARCH[d['architecture']]} | {d['threshold']['first_threshold_internal_W']:.9f} | {cell(four)} | {cell(six)} |")
    lines+=['','活动模数仅是已求静态互补解，不证明时域稳定。小信号第二分支交点也不是受首模增益钳位后的第二模开启功率。高泵功率下可以出现模式竞争改变；例如本轮正方晶格在已算1.05–4×点保持单活动模，而6×存在第二活动模，不能把整个泵范围称为单模。','',
        '20程结果是理想回收敏感性：98%每次返回会减少有效程数，且模型没有实现实际再入射光学、泵频DBR反射、角度/模式失配和图形层散射。它可以降低所需首入射功率，不是已制造器件方案。','',
        f'![泵吸收与首入射转换]({link}science_plots/CW_power_flow_fractions.png)','',
        '下表均在各自架构2×首阈值，属于相同归一化工作点，不是相同输入功率比较。向上/首入射只是模型内端口转换率，不是外部或电光效率；多数顶泵首入射光没有被500nm薄膜吸收。','',
        '| 晶格/泵型 | 内部首入射W | 吸收泵mW | 向上输出mW | 吸收/首入射 |',
        '|---|---:|---:|---:|---:|']
    for lattice,data in cw.items():
        for d in data:
            point=next((r for r in d['points'] if r.get('status')=='completed' and r['ratio']==2.),None)
            if point and point['internal_pump_W']>0:
                power=point['internal_pump_W'];absorbed=point['energy_budget_W']['absorbed_pump_W']
                lines.append(f"| {LABELS[lattice]}/{ARCH[d['architecture']]} | {power:.6g} | {absorbed*1e3:.6f} | {point['upward_output_W']*1e3:.6f} | {absorbed/power:.6g} |")
    lines += ['',
        '## 6. 外部脉冲能量—向上输出能量','',
        '矩形脉冲扫描两种程数(1/20)、四种脉宽(10µs/100µs/1ms/5ms)和六个各自均匀CW参考功率比(0.5/1/1.2/2/4/8)，每腔48例。输入E_ext=P_ext T；积分到关泵后2τ，保留剩余储能。初始反转和腔光子数均为0，没有注入外部激光种子；β=10^-8的自发辐射进入所选模式并提供启动源。微小正E_out可能主要是该自发辐射，不能以E_out>0自动认定已激光起振。','',
        '| 晶格 | 顶泵程数 | 1ms、2×CW参考的外部输入mJ | 向上输出µJ | 向上/外部能量 |',
        '|---|---:|---:|---:|---:|']
    for lattice,data in pulsed.items():
        for passes in (1,20):
            row=next((r for r in data['cases'] if r['top_passes']==passes and r['duration_s']==.001 and r['power_to_uniform_CW_threshold']==2),None)
            if row:
                ei=row['pulse']['external_energy_J'];eo=row['metrics']['upward_output_J']
                lines.append(f'| {LABELS[lattice]} | {passes} | {ei*1e3:.6f} | {eo*1e6:.6f} | {eo/ei:.6g} |')
    lines+=['','表中1程与20程均为各自CW参考的2倍功率，因此并不是相同外部能量对照。输出接近而输入差约有效回收程数，反映理想泵回收降低外部首入射需求。不同脉宽会改变储能建立、寿命耗散和启动延迟；不存在脱离脉宽、种子、泵型和初始反转的唯一能量阈值，P_CW,th×T只能作为参考。','',
        '脉冲采用均匀反转单模降阶，选择真实最低材料阈值冷模；共享空间多模CW则为静态模型，各腔实际根数见上表。这两种结果不可混称“空间多模动力学”。稀疏显示轨迹不用于验证快弛豫尖峰宽度、FFT或脉宽；累计能量来自刚性积分内部累计变量。','']
    lines+=['令f=N2/N_ions，s=N_ph/N_ions，qg=ΣqGain，g=Nt[(σel+σal)f−σal]。本单模降阶实际积分：','',
        r'\[\dot f=P_{abs}/(N_{ions}h\nu_p)-f/\tau-v_gq_ggs,\qquad \dot s=v_g(q_gg-\kappa)s+\beta f/\tau.\]','',
        r'\[P_{up}=N_{ions}h\nu_l v_g\kappa_{up}s,\qquad E_{stored}=N_{ions}h\nu_l(f+s).\]','',
        '受激项在反转和光子方程中等大反号，qGain只出现一次。P_abs由含泵受激发射的净Beer吸收及真实多程递推给出，并非指定的常数吸收效率。矩形泵开关处分段刚性积分，不用阈值开关强行切换激光态，也不把负人口裁剪成物理解。','',
        '图中的输出是所选腔模向上端口能量，包含进入该模的β自发辐射；不是样品全部向上荧光。模外荧光另在账本列项。关泵后2τ末仍可能有激发态储能，已显式保留，不能遗漏后再宣称输入能量都变成了光或热。','']
    for lattice in pulsed:lines.append(f'![{LABELS[lattice]}脉冲能量]({link}science_plots/{lattice}_pulse_energy_input_output.png)\n')
    lines += [f'![反转建立]({link}science_plots/pulse_inversion_startup.png)','',
        '短脉宽不能只参考长脉冲CW功率。下面读取20程、8×各自CW参考下的实际最大反转；若f_max,pulse<f_th，说明本次输入范围还没有建立起阈值反转，不代表该脉宽下无论多高能量都不可能激光。反转跨阈也只是必要条件，尚需有限建立时间和足够受激辐射。','',
        '| 晶格 | 脉宽µs | 8×CW参考的外部输入mJ | 最大反转/阈值反转 | 向上模能量µJ |',
        '|---|---:|---:|---:|---:|']
    for lattice,data in pulsed.items():
        for duration in (1e-5,1e-4):
            row=next((r for r in data['cases'] if r['top_passes']==20 and r['duration_s']==duration and r['power_to_uniform_CW_threshold']==8),None)
            if row:
                ratio=row['metrics']['maximum_inversion_fraction']/row['threshold']['threshold_inversion_fraction']
                lines.append(f"| {LABELS[lattice]} | {duration*1e6:g} | {row['pulse']['external_energy_J']*1e3:.6f} | {ratio:.6f} | {row['metrics']['upward_output_J']*1e6:.6g} |")
    lines.append('')
    lines+=['## 7. 固定晶胞的尺寸依赖','',
        f'![尺寸材料阈值]({link}science_plots/size_material_gain.png)','',
        f'![尺寸内部泵阈值]({link}science_plots/size_internal_pump_threshold.png)','',
        'L=50/100/200/300/500/750µm。每个非锚点重新求有限本征场和实际频率，不能面积缩放、1/L外推，也不逐L调整周期。粗D12/dz2nm用于趋势；100/200/500µm补高D32及更细网格，300µm保持正式合并根锚点。图中空心圆是D12探索点，实心圆为D32/更细网格核验，星形为正式300µm合并锚点。SQ的50/750为D12/N25；100/200/500为D32/N49。表内每点写明真实精度，连线只导引视线，不能把六点看成同等收敛精度或据线段推算最小可行尺寸。','',
        '| 晶格 | Lµm | D/dz_nm/N | 最低已记录g/cm^-1 | 该冷模λnm | edge阈值W | top1阈值W |',
        '|---|---:|---|---:|---:|---:|---:|']
    def threshold_cell(value):return 'null' if value is None else f'{value:.6g}'
    for lattice,data in sizes.items():
        for row in data:
            edge=row['pumps'].get('edge_dual',{}).get('first_threshold_internal_W')
            top=row['pumps'].get('top1',{}).get('first_threshold_internal_W')
            lines.append(f"| {LABELS[lattice]} | {row['side_um']:g} | {row['order']}/{row['dz_nm']:g}/{row['grid']} | {row['material_gain_cm1']:.6f} | {row['cold_wavelength_nm']:.6f} | {threshold_cell(edge)} | {threshold_cell(top)} |")
    lines+=['','null含义需要查看原JSON：可能是所供模式都超过泵极限增益、有限搜寻功率内未包根、或条件窗口没有保存根。不是零阈值，也不能把某尺寸未发现低损耗根称为连续谱数学不存在。主曲线保留所有被接受的真实频率根；1029–1031nm窗口曲线只回答“若只考察该窗口”的条件问题，不是器件已有光谱滤波器。','',
        '表中λ是最低材料g冷模的波长，不自动等同于每种空间泵型的实际赢家。各泵型首模名、首模自身λ及条件窗口首模λ另存尺寸JSON和CSV，避免把冷损耗排序误当成所有泵型的阈值排序。','',
        '窗口外的物理模式仍能先耗尽共享反转，所以条件窗口数值也不是该窗口模式实际开启的保证。新尺寸没有重新优化孔形或重新保证首模远场亮中心/近圆；尺寸曲线是固定单元的损耗和泵浦研究，不是六个同时满足原全部设计目标的新器件。','',
        '边长增大可降低所算冷腔泄漏，但泵浦总阈值还取决于体积、反转分布、吸收路程和模式重叠；因此最低g下降不要求顶泵总W单调下降。500/750µm根本征频率也可能不同于1030nm，常σ总谱比较与真实器件谱可达必须分开。','',
        '## 8. 独立记录QA与归档来源','',
        f'逐记录QA：[independent_output_qa.json]({link}independent_output_qa.json)。输入输出CSV：[CW]({link}CW_input_output.csv)、[pulse]({link}pulse_input_output.csv)、[size]({link}size_threshold.csv)。原始JSON/NPZ和source SHA保留。', '',
        '逐脉冲重新组合外部/入口损失/吸收/逃逸/回程边界账本，及吸收减量子亏损/荧光/上下输出/其他腔损耗/剩余储能账本；核对单位、正分母、人口范围、实际最低g根和η=.7能量定义。逐CW重新组合模式输出总和、热化项、泵边界、互补残差与人口残差。允许不同激光光子能的热化项有符号，不能把该项的负值误当成负输出。','',
        '尺寸记录还独立读回实际L/N拓扑与自身保存C的原K/M、增益权重、被动能流及闭根身份。该项独立读回不重新求根；尺寸执行器另有每根自身波长C和频率重建。通过已保存根的原方程不等于所有搜索目标都成功；有界Arnoldi目标失败和真正频率闭合失败分别保留，不能把失败目标当成物理不存在。','',
        '每腔只有top20/1ms/2×一例收紧10倍rtol复算；不能说48例均做了10倍收敛重算。其余记录的预算检查属于模型内一致性检查，不弥补遗漏物理。']
    if (OUT/'size_execution_failures').exists():
        lines.append('\n新增三角尺寸wrapper最初误用了正方结果对象的radiation_fields接口，已改为三角对象的radiation_amplitudes。该错误属于后处理执行错误，失败记录保留在size_execution_failures；没有改物理算符、参数或验收阈值，相应最终点已重新求解并完成严格读回，不能把先前执行失败当成物理不可达。')
    for lattice,r in result['reports'].items():
        if 'pulse' not in r:continue
        pulsemax=max((max(p['relative_errors'].values()) for p in r['pulse']['records']),default=0.)
        cwmax=max((max(p.get('absolute_errors_W',{}).values(),default=0.) for c in r['cw'] for p in c.get('checks',[])),default=0.)
        fine=r['pulse']['accuracy_challenges']
        lines.append(f"\n{LABELS[lattice]}：已核对{len(r['pulse']['records'])}个脉冲和{sum(len(c.get('checks',[])) for c in r['cw'])}个CW点；最大脉冲相对账本/定义差{pulsemax:.3g}，CW最大绝对账本/定义差{cwmax:.3g}W。代表性收紧rtol输出差：{fine[0]['relative_upward_energy_change']:.3g}。" if fine else f'\n{LABELS[lattice]}：收紧rtol记录尚未出现。')
        sizechecks=sum(len(s['independent_stored_pencil_checks']) for s in r['sizes'])
        targetfails=sum(len(s['failed_census_targets']) for s in r['sizes'])
        closurefails=sum(len(s['closure_failures']) for s in r['sizes'])
        lines.append(f'尺寸档共独立检查{sizechecks}根（含正式锚点），原记录保留{targetfails}个未成功有界搜谱目标、{closurefails}个频率闭合失败。最终300µm锚点的完整原始谱审查另见其选型源路径。')
    lines+=['','## 9. 可行性判断和仍缺失的物理','',
        '当前支持的是：在已声明的标量TE近Bragg耦合波、固定冷场和名义材料/理想泵条件下，部分300µm及更大边界存在净增益可达的闭频根；对应速率模型给出能量自洽的静态CW及均匀单模启动/储能响应。现有弱反馈面发射设计缩小到50–200µm时，部分已算根超过名义泵增益上限，不能靠功率无限补偿；需改反馈、缺陷局域化/反射结构、降低损耗或改腔型，而不能直接把大腔结果等比缩小。','',
        '尚缺当前薄膜的实际掺杂/寿命/非辐射量子产额、完整温度相关吸收发射谱、图形化传播/散射损耗、实际泵耦合/多程回收设计、热阻与温升、应力/热致折射率漂移、反转引起的场重整及相干多模动力学。低材料g和低理想内部阈值不可直接推断高外部效率或制造良率。','',
        '有限边界采用包络特征场的零入射条件，没有逐个解析最后一排孔的微观终止、三维侧壁/刻蚀深度变化和SEM重建。各网格上的原本征方程正确求解与真实器件完整边界一致是两层问题，后者还需要相应全波和实测验证。','',
        '正方和三角腔各自采用不同的优化孔形与参数，因此两组阈值/效率差不是晶格本身的普遍优劣证明。晶格研究价值应结合耦合方向、对称性、有限边界与制造可控性研究，不能仅用本轮两个不同几何的数值排序认定先天优势。','',
        '冷腔κ不重复包含已经通过σal计入速率方程的Yb再吸收。若与未泵浦实测Q比较，需先分离Yb吸收，避免双计。有效激发能守恒把荧光和不同模光子能差用指定参考记账，不是完整声子/荧光谱热模型，也不能把全部侧向出射光视作膜内热。','',
        '近Bragg接口保持群速度、qGain、损耗和场归一化一致；本轮没有只单改ng或qGain来“修正”一处公式。其离Bragg投影近似和有限基本波/TE截断仍存在；数值残差小或预算闭合不能把它升级为完全按照论文的三维实测孔形全波复现。','',
        '复算科学整理（不会重跑主仿真）：`E:/Anaconda/envs/env00/python.exe scripts/diagnostics/review_strict_yag_feasibility_20261025.py`。主任务输入SHA冻结；尺寸及速率原始输出分别保留来源与完成状态。']
    path=ROOT/'docs/strict_yag_feasibility_20261025_zh.md'
    path.write_text('\n'.join(lines)+'\n',encoding='utf8')
    return str(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lattices',nargs='+',default=['square','triangular'])
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    own_source=Path(__file__).read_bytes();own_hash=hashlib.sha256(own_source).hexdigest()
    (OUT/'executed_output_review.py').write_bytes(own_source)
    dependencies=[ROOT/'scripts/optimization/strict_yag_feasibility_20261025.py',
        ROOT/'scripts/optimization/yag_feasibility_study.py',ROOT/'scripts/optimization/matched300_yag_search.py',
        *list((ROOT/'src/pcselsim').glob('*.py'))]
    dependency_hash={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in dependencies}
    reports={};cw_data={};pulse_data={};size_data={};flat_pulse=[];flat_cw=[];flat_size=[]
    for lattice in a.lattices:
        selected=BASE/('YbYAG_'+lattice)/'selected_case.json'
        if not selected.exists() or read(selected).get('status')!='completed':
            reports[lattice]=dict(status='awaiting_final_selected_case');continue
        system,_,rows=driver.load_system(lattice)
        selected_data,summary_path,rows_path,fields_path,_=driver.files_for(lattice)
        read(summary_path);read(rows_path)
        SOURCE_HASH[str(fields_path.resolve())]=hashlib.sha256(fields_path.read_bytes()).hexdigest()
        pulse,pdata=pulse_qa(lattice,system,rows);cw,cdata=cw_qa(lattice,system);sizes,best=sizes_review(lattice)
        key_controls={str(int(length)):any(r['physically_valid_record'] and r['side_um']==length
            and (r['order'] or 0)>=32 and (r['grid'] or 0)>=(32 if lattice=='triangular' else 49) for r in sizes)
            for length in (100.,200.,500.)}
        reports[lattice]=dict(status='passed' if pulse['status']=='passed' and all(r['status']=='passed' for r in cw) else 'incomplete_or_failed',
            retained_mode_count=len(rows),pulse=pulse,cw=cw,sizes=sizes,selected_size_cases=best)
        reports[lattice]['key_size_controls_available']=key_controls
        cold=min(rows,key=lambda r:r['required_material_gain_cm1']);model=system.modes[system.names.index(cold['name'])]
        reports[lattice]['optical_reference']=dict(name=cold['name'],wavelength_nm=cold['wavelength_nm'],
            material_gain_cm1=cold['required_material_gain_cm1'],group_velocity_m_s=model.group_velocity_m_s,
            gain_weight=float(model.optical_gain_weights.sum()),upward_loss_fraction=model.upward_power_loss_per_m/model.total_power_loss_per_m,
            photon_lifetime_s=1/(model.group_velocity_m_s*model.total_power_loss_per_m),
            cold_loss_Q=2*np.pi*driver.c/(cold['wavelength_nm']*1e-9)/(model.group_velocity_m_s*model.total_power_loss_per_m))
        spectroscopy=Path(selected_data.get('spectroscopy_review_path',summary_path.parent/'spectroscopy_review.json'))
        if spectroscopy.exists():reports[lattice]['spectroscopy_protocols']=read(spectroscopy)['records']
        cw_data[lattice]=cdata;size_data[lattice]=best
        if pdata is not None:pulse_data[lattice]=pdata
        for row in (pdata or {}).get('cases',[]):
            flat_pulse.append(dict(lattice=lattice,id=row['id'],top_passes=row['top_passes'],duration_s=row['duration_s'],
                power_ratio_to_uniform_CW=row['power_to_uniform_CW_threshold'],external_input_J=row['pulse']['external_energy_J'],
                upward_output_J=row['metrics']['upward_output_J'],absorbed_pump_J=row['metrics']['absorbed_pump_J'],
                peak_external_W=row['metrics']['pump_peak_external_W'],first_cold_mode=row['first_cold_mode']))
        for data in cdata:
            for row in data['points']:
                if row['status']!='completed':continue
                flat_cw.append(dict(lattice=lattice,architecture=data['architecture'],ratio=row['ratio'],
                    internal_input_W=row['internal_pump_W'],upward_output_W=row['upward_output_W'],
                    active_modes=';'.join(row['active_modes']),absorbed_pump_W=row['energy_budget_W']['absorbed_pump_W']))
        for row in best:
            for arch in ('edge_dual','top1','top20'):
                flat_size.append(dict(lattice=lattice,side_um=row['side_um'],D=row['order'],dz_nm=row['dz_nm'],N=row['grid'],
                    architecture=arch,lowest_documented_g_cm1=row['material_gain_cm1'],cold_wavelength_nm=row['cold_wavelength_nm'],
                    total_spectrum_internal_threshold_W=row['pumps'].get(arch,{}).get('first_threshold_internal_W'),
                    conditional1029_1031_internal_threshold_W=row['target_band_pumps'].get(arch,{}).get('first_threshold_internal_W'),
                    total_first_modes=';'.join(row['pumps'].get(arch,{}).get('first_modes',[])),
                    total_first_wavelengths_nm=';'.join(f'{v:.9f}' for v in row['pump_first_wavelengths_nm'].get(arch,[])),
                    conditional_first_wavelengths_nm=';'.join(f'{v:.9f}' for v in row['target_band_first_wavelengths_nm'].get(arch,[])),
                    mode_count=row['mode_count'],closure_failure_count=row['closure_failure_count'],source_path=row['path']))
    write_csv(OUT/'pulse_input_output.csv',flat_pulse);write_csv(OUT/'CW_input_output.csv',flat_cw)
    write_csv(OUT/'size_threshold.csv',flat_size);save(OUT/'size_selected_cases.json',size_data)
    plots=figures(cw_data,pulse_data,size_data)
    complete=all(l in reports and reports[l]['status']=='passed' and reports[l]['pulse']['all48_present']
        and len(reports[l]['pulse']['accuracy_challenges'])==1 and all(r.get('all13_present') for r in reports[l]['cw'])
        and all(reports[l]['key_size_controls_available'].values())
        and {r['side_um'] for r in size_data.get(l,[])}=={50.,100.,200.,300.,500.,750.} for l in a.lattices)
    changed=[p for p,h in SOURCE_HASH.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=own_hash:changed.append(str(Path(__file__).resolve()))
    changed.extend(str(ROOT/q) for q,h in dependency_hash.items() if hashlib.sha256((ROOT/q).read_bytes()).hexdigest()!=h)
    result=dict(status='completed' if complete and not changed else 'partial',
        generated_utc=datetime.now(timezone.utc).isoformat(),all_requested_runs_complete=complete,
        source_files_changed_during_read=changed,reports=reports,plots=plots,source_sha256=SOURCE_HASH,
        own_script_sha256=own_hash,executed_driver_snapshot_path=str(OUT/'executed_output_review.py'),
        argv=sys.argv,dependency_sha256=dependency_hash,
        execution_history_files=[str(p) for p in sorted((OUT/'size_execution_failures').glob('*.json'))],
        execution_history_scope='Preserved resolved wrapper execution failures are separate from bounded spectral-target failures and physical unreachable thresholds.',
        scope='Independent stored-record QA. No primary CW/pulse/size simulations repeated; conservation does not certify omitted spectra, thermal dynamics or fabrication.')
    all_sizes_complete=all(l in size_data and {r['side_um'] for r in size_data[l]}=={50.,100.,200.,300.,500.,750.}
        and all(reports[l]['key_size_controls_available'].values()) for l in a.lattices)
    save(OUT/'all_sizes_summary.json',dict(status='completed' if all_sizes_complete else 'partial',
        by_lattice=size_data,source_sha256={p:h for p,h in SOURCE_HASH.items() if '/sizes/' in p.replace('\\','/')},
        scope='Combined read-only comparison; original square-only summaries remain untouched. Null thresholds are not0.'))
    if 'triangular' in size_data:
        tri=reports['triangular'];tri_complete=({r['side_um'] for r in size_data['triangular']}=={50.,100.,200.,300.,500.,750.}
            and all(tri['key_size_controls_available'].values()))
        save(OUT/'triangular_sizes_summary.json',dict(status='completed' if tri_complete else 'partial',
            points=size_data['triangular'],all_records=tri['sizes'],key_controls=tri['key_size_controls_available'],
            scope='Fixed final triangular motif/period/stack, new finite poles and own-frequency fields; final300um merged anchor. Conditional wavelength window is a report subset, not a physical filter.'))
        write_csv(OUT/'triangular_size_summary.csv',[r for r in flat_size if r['lattice']=='triangular'])
        text=['# 最终三角晶格Yb:YAG固定周期尺寸结果','',
            '状态：'+('完成六尺寸及关键高精度控制。' if tri_complete else '仍有尺寸/高精度控制未完成。'),'',
            '| Lµm | D/dz_nm/N | 最低已记录g/cm^-1 | 冷模λnm | 全谱edge阈值W | 全谱top1阈值W |',
            '|---|---|---:|---:|---:|---:|']
        for r in size_data['triangular']:
            def cell(arch):
                value=r['pumps'].get(arch,{}).get('first_threshold_internal_W')
                return 'null' if value is None else f'{value:.9g}'
            text.append(f"| {r['side_um']:g} | {r['order']}/{r['dz_nm']:g}/{r['grid']} | {r['material_gain_cm1']:.8f} | {r['cold_wavelength_nm']:.8f} | {cell('edge_dual')} | {cell('top1')} |")
        text+=['','精确根/功率/字段及独立保存C+K/M读回见triangular_sizes_summary.json和triangular_size_summary.csv。',
            'null不画成0；常σ总谱首模与1029–1031nm条件窗首模分别报告。不同L不重新调周期、不做面积或1/L损耗缩放。',
            'D12粗点与D32/N32或40验证必须区分；连线只导引视线，不能据离散点推导最小可行尺寸。',
            '所有阈值为内部首入射总功率；相同g下降不要求顶泵总W单调下降。理想20程与实际外部泵耦合/热可行性仍属不同问题。']
        (OUT/'triangular_size_report_zh.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    result['report_path']=write_report(result,cw_data,pulse_data,size_data)
    save(OUT/'independent_output_qa.json',result)
    print(result['status'],[(l,r['status']) for l,r in reports.items()],len(flat_pulse),len(flat_cw),len(flat_size),flush=True)


if __name__=='__main__':main()
