"""Post-audit rerun of the 300-um triangular semiconductor PCSEL.

中文：原结果只读。多参数局扫只作候选筛选，完整粗谱/细网格挑战与实频闭合
分别记录；插值仅用作初值，最终场必须满足原始K/M与物理频率方程。
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from dataclasses import asdict,replace
import hashlib,json,time,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from pcselsim.triangular_finite import TriangularFiniteSpec,TriangularFiniteMode,triangular_finite_eigenproblem
from pcselsim.finite_discretization import relative_residual
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.design_objectives import beam_metrics
from pcselsim.polarization import polarization_metrics
from scripts.optimization.matched300_triangular_search import ThreeHole,geometry
from scripts.optimization.matched300_triangular_qz_guided import model_for,scalar_record
from scripts.optimization.matched300_triangular_actual_controls import follow,separable_far_field
from scripts.optimization.matched300_triangular_close import close
from scripts.optimization.triangular_semiconductor_search import json_write,threadpool_limits

OLD=ROOT/'results/current/matched300_20261021/triangular/formal_low_gain_verified_N91'
UNION=ROOT/'results/current/matched300_20261021/triangular/low_gain_union_N91'
OUT=ROOT/'results/current/strict_rerun_20261025/semiconductor_triangular'


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    json_write(p,x)


def old_reference():
    return load(OLD/'summary.json'),np.load(UNION/'complex_fields.npz',allow_pickle=False)


def make_seeds(source,reference,folder,cells=91,order=48,dz=1.,roots=6):
    """Interpolate known branches, then solve the NEW operator / 原场只作初值。"""
    spec,p,model=model_for(source,order,dz)
    finite=TriangularFiniteSpec(150.,cells,0.,16,'square','staggered')
    problem,grid=triangular_finite_eigenproblem(model,finite)
    old_xy=np.column_stack((reference['x_um'],reference['y_um']))
    triangle=Delaunay(old_xy);points=np.column_stack((grid.x_um,grid.y_um))
    values=[];fields=[];records=[];failed=[]
    for index in range(min(roots,len(reference['eigenvalues']))):
        started=time.monotonic()
        initial=LinearNDInterpolator(triangle,reference['fields'][index].T,fill_value=0.)(points).T.ravel()
        initial/=np.linalg.norm(initial)
        target=complex(np.vdot(initial,problem.center_operator()@initial)) + 50+5j
        try:
            value,field,overlap,solver=follow(problem,target,initial,24)
            if overlap<.75:raise RuntimeError(f'Lost supplied branch overlap={overlap}')
            if not independent_eigenmode(value,field,values,fields):
                failed.append(dict(index=index,status='duplicate_same_complex_pole_and_span'))
                continue
            residual=relative_residual(problem.center_operator(),field,value)
            if residual>2e-7:raise RuntimeError(f'Original pencil residual {residual}')
            field=field.reshape(6,grid.points)
            record=scalar_record(value,field,model,p,problem,grid,model.bragg_wavelength_nm,index)
            record.update(initialization_overlap=overlap,solver=solver,elapsed_seconds=time.monotonic()-started)
            values.append(value);fields.append(field);records.append(record)
            print('SEED',cells,order,index,record['g_th_cm'],flush=True)
        except Exception as error:
            failed.append(dict(index=index,status='failed',error=repr(error)))
    order_indices=np.argsort(np.imag(values))
    fields=np.asarray(fields)[order_indices];values=np.asarray(values)[order_indices]
    records=[records[i] for i in order_indices]
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'result.json',dict(three_hole_spec=asdict(spec),N=cells,D=order,dz_nm=dz,
        reference_wavelength_nm=model.bragg_wavelength_nm,spectrum=records,failures=failed,
        scope='Known branches solved on current original pencil; seeds only, not a new complete census.'))
    np.savez_compressed(folder/'complex_fields.npz',eigenvalues=values,fields=fields,
        x_um=grid.x_um,y_um=grid.y_um,coupling_m=model.coupling_m)
    return records


def known(args):
    source,reference=old_reference()
    initialization_source=UNION
    if args.candidate:
        local=load(args.candidate/'result.json')
        source={**source,'three_hole_spec':local['three_hole_spec']}
        reference=np.load(args.candidate/'complex_fields.npz',allow_pickle=False)
        initialization_source=args.candidate
    folder=OUT/args.label
    make_seeds(source,reference,folder/'seeds',args.cells,args.order,args.dz,args.roots)
    result=close(folder/'seeds',folder/'frequency_closed',args.roots,args.order,args.dz,
                 inverse=True,separable=True,newton=True)
    result.update(nominal_size_um=[300,300],case_label=args.label,
                  initialization_source=str(initialization_source.resolve().relative_to(ROOT)),new_identity_rule='resolved complex pole plus subspace')
    save(folder/'summary.json',result)


def stripe_audit(args):
    """Analytic opposite-wave limits supply challenge shifts, not final modes."""
    from scripts.diagnostics.audit_counterpropagating_stripes import stripe_pair
    source=load((args.candidate if args.candidate else OLD)/'summary.json')
    _,p,model=model_for(source,args.order,args.dz)
    qgain=model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME]*p.ACTIVE_INDEX/model.effective_index
    pairs=[]
    for pair,angle in enumerate((-60.,60.,0.)):
        length=300/max(abs(np.sin(np.deg2rad(angle))),abs(np.cos(np.deg2rad(angle))))
        block=model.coupling_m[2*pair:2*pair+2,2*pair:2*pair+2]
        result=stripe_pair(block,length,qgain,5.,grids=(101,201))
        pole=result['independent_continuous_transfer_pole'];pole['usable']=pole['converged']
        pairs.append(dict(indices=[2*pair,2*pair+1],length_um=length,**result))
    output_name=f'{args.label}_actual_stripe_limits.json' if args.candidate else 'baseline_actual_stripe_limits.json'
    save(OUT/output_name,dict(actual_wavelength_nm=model.bragg_wavelength_nm,
        D=args.order,dz_nm=args.dz,gain_weight=qgain,pairs=pairs,
        coupling_real_per_m=model.coupling_m.real.tolist(),coupling_imag_per_m=model.coupling_m.imag.tolist(),
        scope='Independent two-wave transfer roots only provide adversarial shifts for full six-wave finite-cavity search.'))


def retune(args):
    """Tune a candidate at actual mode frequency, then freeze its period.

    中文：外层先重调实际发射周期；其后的每根闭频仍固定同一几何，不逐模调周期。
    """
    if args.candidate is None:raise ValueError('retune requires a retained candidate folder')
    old,_=old_reference();local=load(args.candidate/'result.json')
    source={**old,'three_hole_spec':local['three_hole_spec']}
    reference=np.load(args.candidate/'complex_fields.npz',allow_pickle=False)
    folder=OUT/args.label;trace=[]
    for iteration in range(4):
        seed_folder=folder/f'retune_{iteration}_seed'
        make_seeds(source,reference,seed_folder,args.cells,args.order,args.dz,3)
        result=close(seed_folder,folder/f'retune_{iteration}_closed',1,args.order,args.dz,
                     inverse=True,separable=True,newton=True)
        if result['roots_closed']!=1:raise RuntimeError('Actual finite mode failed to close during period design')
        wavelength=result['lowest']['wavelength_nm']
        trace.append(dict(iteration=iteration,period_nm=source['three_hole_spec']['lattice_nm'],
                          wavelength_nm=wavelength,g_th_cm=result['lowest']['g_th_cm']))
        save(folder/'period_retuning.json',trace)
        if abs(wavelength-950.)<.003:
            result.update(period_retuning=trace,actual_target_wavelength_nm=950.,actual_target_pass=True,
                nominal_size_um=[300,300],source_candidate=str(args.candidate.resolve().relative_to(ROOT)),
                scope='Period design of one tracked candidate; new full-coarse/fine census remains required.')
            save(folder/'summary.json',result)
            break
        spec={**source['three_hole_spec']};spec['lattice_nm']*=950./wavelength
        source={**source,'three_hole_spec':spec}
    else:raise RuntimeError('Finite-mode period retuning did not reach the actual target')


def controls(spec):
    """One-knob and simultaneous local changes; shape-inactive aspects excluded."""
    rows=[('nominal',spec)]
    knobs=[('fill',(-.004,-.002,.002,.004)),('main_share',(-.02,-.01,.01,.02)),
        ('satellite_balance',(-.02,.02)),('distance1',(-.006,-.003,.003,.006)),
        ('distance2',(-.006,-.003,.003,.006)),('angle1_deg',(-2.,-1.,1.,2.)),
        ('angle2_deg',(-2.,-1.,1.,2.)),('main_orientation_deg',(-3.,-1.5,1.5,3.)),
        ('rounding',(-.04,.04))]
    for key,values in knobs:
        for delta in values:rows.append((f'{key}_{delta:+g}',replace(spec,**{key:getattr(spec,key)+delta})))
    for index in (1,2):
        for aspect in (1.06,1.15):
            for angle in (-30.,30.):
                rows.append((f'sat{index}_ellipse_{aspect}_{angle}',replace(spec,**{
                    f'satellite{index}_shape':'ellipse',f'satellite{index}_aspect':aspect,
                    f'satellite{index}_orientation_deg':angle})))
    for shape in ('circle','ellipse'):
        rows.append((f'main_{shape}',replace(spec,main_shape=shape,main_aspect=1.1)))
    rng=np.random.default_rng(20261025)
    for i in range(28):
        rows.append((f'joint_{i:02d}',replace(spec,fill=spec.fill+rng.normal(0,.002),
            main_share=spec.main_share+rng.normal(0,.01),distance1=spec.distance1+rng.normal(0,.003),
            distance2=spec.distance2+rng.normal(0,.003),angle1_deg=spec.angle1_deg+rng.normal(0,1.),
            angle2_deg=spec.angle2_deg+rng.normal(0,1.),main_orientation_deg=spec.main_orientation_deg+rng.normal(0,1.5))))
    return rows


def stability(args):
    """Repeat resolved poles with perturbed seeds / 扰动初场与移位，检查极点稳定性。"""
    if args.candidate is None:raise ValueError('stability requires a common-frequency pole pool')
    prior=load(args.candidate/'result.json')
    source=dict(three_hole_spec=prior['three_hole_spec'],
                lowest=dict(wavelength_nm=prior['reference_wavelength_nm']))
    _,p,model=model_for(source,args.order,args.dz)
    problem,grid=triangular_finite_eigenproblem(model,
        TriangularFiniteSpec(150.,prior['N'],0.,24,'square','staggered'))
    arrays=np.load(args.candidate/'complex_fields.npz',allow_pickle=False)
    if np.linalg.norm(arrays['coupling_m']-model.coupling_m)>1e-10*np.linalg.norm(model.coupling_m):
        raise ValueError('Stability audit must use the identical physical operator')
    fields=arrays['fields'].reshape(len(arrays['fields']),-1)
    fields=fields/np.linalg.norm(fields,axis=1)[:,None]
    gram=np.abs(fields.conj()@fields.T);np.fill_diagonal(gram,0.)
    pair=np.unravel_index(np.argmax(gram),gram.shape)
    rows=[];rng=np.random.default_rng(20261025)
    for rank in range(min(args.roots,len(fields))):
        for repeat,offset in enumerate((30+7j,-40+11j,65-9j)):
            field=fields[rank].copy()
            field+=1e-3*(rng.normal(size=field.size)+1j*rng.normal(size=field.size))/np.sqrt(field.size)
            field/=np.linalg.norm(field)
            try:
                value,actual,overlap,trace=follow(problem,arrays['eigenvalues'][rank]+offset,field,24)
                delta=abs(value-arrays['eigenvalues'][rank])
                row=dict(rank=rank,repeat=repeat,status='completed',delta_per_m=value.real,alpha_per_m=value.imag,
                    pole_error_absolute_per_m=float(delta),pole_error_relative=float(delta/max(abs(value),1.)),
                    phase_invariant_field_overlap=float(abs(np.vdot(fields[rank],actual))),
                    original_pencil_relative_residual=relative_residual(problem.center_operator(),actual,value),solver=trace)
            except Exception as error:row=dict(rank=rank,repeat=repeat,status='failed',error=repr(error))
            rows.append(row);save(OUT/args.label/'progress.json',rows)
            print('STABILITY',rank,repeat,row['status'],row.get('pole_error_absolute_per_m'),flush=True)
    save(OUT/args.label/'summary.json',dict(source_pool=str(args.candidate.resolve().relative_to(ROOT)),
        N=prior['N'],D=args.order,dz_nm=args.dz,attempts=len(rows),rows=rows,
        maximum_pairwise_field_overlap=float(gram[pair]),most_parallel_pair=[int(i) for i in pair],
        resolved_complex_pole_distance_per_m=float(abs(arrays['eigenvalues'][pair[0]]-arrays['eigenvalues'][pair[1]])),
        identity_rule='Resolved complex poles are retained even with high field overlap; only same pole/span duplicates are merged',
        scope='Perturbed-repeat stability of supplied branches at one identical frozen C(lambda), not a spectral completeness proof.'))


def package(args):
    """Export solved static data and geometry / 导出实际求解的静态数据与几何。"""
    from pcselsim.triangular_six_wave import triangular_band_diagram
    from pcselsim.band_structure import plot_band_diagram
    from scripts.optimization.matched300_triangular_package import actual_geometry
    if args.candidate is None:raise ValueError('package requires a frequency-closed case folder')
    source=args.candidate.resolve();result=load(source/'summary.json')
    closed=source/'frequency_closed'
    records=load(closed/'frequency_closed_poles.json')
    records=sorted((r for r in records if r['status']=='completed'),key=lambda r:r['g_th_cm'])
    folder=OUT/args.label;folder.mkdir(parents=True,exist_ok=True)
    spec,p,model=model_for(result,args.order,args.dz)
    save(folder/'actual_geometry_nm.json',actual_geometry(spec))
    bands=triangular_band_diagram(model,q_max=.01,points=241)
    plot_band_diagram(bands,folder/'bands.png','Triangular semiconductor PCSEL: six-wave bands near Gamma')
    np.savez_compressed(folder/'bands.npz',wave_number_2pi_over_a=bands.wave_number_2pi_over_a,
        normalized_frequency=bands.normalized_frequency,radiation_constant_cm=bands.radiation_constant_cm)
    save(folder/'band_metadata.json',dict(labels=list(bands.labels),approximation=bands.approximation,
        reference_wavelength_nm=model.bragg_wavelength_nm,reference_period_nm=spec.lattice_nm,
        scope='Frozen-Gamma six-wave local dispersion; finite-cavity modal thresholds are supplied separately.'))
    compact=[{k:v for k,v in r.items() if k!='trace'} for r in records]
    save(folder/'modes.json',compact)
    save(folder/'polarization.json',[dict(rank=i,seed_rank=r['seed_rank'],wavelength_nm=r['wavelength_nm'],
        polarization=r.get('polarization')) for i,r in enumerate(records)])
    wave=np.array([r['wavelength_nm'] for r in records]);distance=np.abs(wave[:,None]-wave[None,:])
    np.fill_diagonal(distance,np.inf)
    identity=dict(roots_closed=len(records),minimum_closed_wavelength_separation_nm=float(distance.min()) if len(records)>1 else None,
        maximum_original_pencil_relative_residual=max(r['eigen_residual'] for r in records),
        maximum_real_frequency_equation_residual_per_m=max(abs(r['frequency_residual_per_m']) for r in records),
        scope='Every displayed root is individually closed at fixed geometry; distinct closed wavelengths are not merged by field overlap.')
    save(folder/'closed_identity_audit.json',identity)
    relative=lambda path:str(path.resolve().relative_to(ROOT)).replace('\\','/')
    exported=dict(result,lowest=compact[0] if compact else None,second=compact[1] if len(compact)>1 else None,
        modes=compact,nominal_size_um=[300,300],actual_target_wavelength_nm=950.,actual_target_tolerance_nm=.05,
        actual_target_pass=bool(compact and abs(compact[0]['wavelength_nm']-950)<.05),
        source_case_path=relative(source),summary_path=relative(source/'summary.json'),
        closed_fields_path=relative(closed/'frequency_closed_fields.npz'),
        overview_path=relative(closed/'closed_modes.png'),lattice_path=relative(closed/'lattice.png'),
        geometry_path=relative(folder/'actual_geometry_nm.json'),band_diagram_path=relative(folder/'bands.png'),
        band_data_path=relative(folder/'bands.npz'),polarization_path=relative(folder/'polarization.json'),
        closed_identity_audit=identity,
        threshold_definition='Material threshold gain (2 alpha_field + 5/cm)/(Gamma*n_active/n_eff); group-index normalization retained as a near-Bragg model approximation.',
        scope='Static six-wave effective-index coupled-wave cavity model; finite coarse spectrum and sampled fine-grid searches do not certify the full Maxwell spectrum or nonlinear single-mode operation.')
    save(folder/'exported_case.json',exported)
    if args.label=='selected':save(OUT/'selected_case.json',exported)


def local(args):
    source,reference=old_reference();original=ThreeHole(**source['three_hole_spec'])
    folder=OUT/args.label;folder.mkdir(parents=True,exist_ok=True)
    existing=load(folder/'candidates.json') if (folder/'candidates.json').exists() else []
    done={r['label'] for r in existing};saved=[]
    # Same physical input branch fields, solved independently for every trial.
    old_xy=np.column_stack((reference['x_um'],reference['y_um']));triangulation=Delaunay(old_xy)
    interpolation=[LinearNDInterpolator(triangulation,f.T,fill_value=0.) for f in reference['fields'][:3]]
    seeds=None
    proposals=controls(original)
    if args.candidates:
        proposals=[(row['label'],ThreeHole(**row['three_hole_spec'])) for row in load(args.candidates)]
    for label,trial in proposals:
        if label in done:continue
        started=time.monotonic();fabrication=geometry(trial)
        if not fabrication['margin_pass']:
            row=dict(label=label,status='geometry_rejected',three_hole_spec=asdict(trial),fabrication=fabrication)
        else:
            try:
                _,p,model=model_for({**source,'three_hole_spec':asdict(trial)},args.order,args.dz)
                problem,grid=triangular_finite_eigenproblem(model,TriangularFiniteSpec(150.,args.cells,0.,16,'square','staggered'))
                if seeds is None:
                    points=np.column_stack((grid.x_um,grid.y_um))
                    seeds=[fn(points).T.ravel() for fn in interpolation]
                    seeds=[v/np.linalg.norm(v) for v in seeds]
                values=[];fields=[];records=[]
                for i,seed in enumerate(seeds):
                    target=complex(np.vdot(seed,problem.center_operator()@seed))+50+5j
                    value,field,overlap,solver=follow(problem,target,seed,16)
                    if overlap<.70:raise RuntimeError(f'Branch {i} lost overlap={overlap}')
                    if not independent_eigenmode(value,field,values,fields):continue
                    field=field.reshape(6,grid.points)
                    record=scalar_record(value,field,model,p,problem,grid,model.bragg_wavelength_nm,i)
                    record.update(initial_overlap=overlap)
                    values.append(value);fields.append(field);records.append(record)
                indices=np.argsort([r['g_th_cm'] for r in records]);records=[records[i] for i in indices];fields=[fields[i] for i in indices]
                first=records[0];rx,ry=model.radiation_amplitudes(fields[0])
                mode=TriangularFiniteMode(label,first['delta_per_m'],first['alpha_per_m'],fields[0],1.,grid,rx,ry,discretization='staggered')
                ff=separable_far_field(mode,first['frozen_CWT_pole_wavelength_nm'],1.,101)
                beam=beam_metrics(ff.angle_deg,ff.field_x,ff.field_y,requested_view_deg=1.,source_nyquist_deg=ff.alias_free_square_view_deg)
                gap=records[1]['g_th_cm']/first['g_th_cm']-1 if len(records)>1 else 0.
                score=first['g_th_cm']+300*max(beam['ellipticity']-1.05,0)+500*max(.98-beam['center_to_peak'],0)+250*max(beam['radial_anisotropy_l2']-.10,0)+200*max(.10-gap,0)
                row=dict(label=label,status='completed',three_hole_spec=asdict(trial),N=args.cells,D=args.order,dz_nm=args.dz,
                    tracked_modes=records,g_th_cm=first['g_th_cm'],beam=beam,g_th_gap_fraction=gap,score=score,
                    fabrication=fabrication,scope='Three supplied branches at frozen frequency; exploratory local score, no complete-spectrum or frequency-closure claim')
                if score<90:
                    dest=folder/'retained'/label;dest.mkdir(parents=True,exist_ok=True)
                    seed_result=dict(three_hole_spec=asdict(trial),N=args.cells,D=args.order,dz_nm=args.dz,spectrum=records,
                        reference_wavelength_nm=model.bragg_wavelength_nm)
                    save(dest/'result.json',seed_result)
                    np.savez_compressed(dest/'complex_fields.npz',fields=fields,
                        eigenvalues=[complex(r['delta_per_m'],r['alpha_per_m']) for r in records],
                        coupling_m=model.coupling_m,x_um=grid.x_um,y_um=grid.y_um)
            except Exception as error:
                row=dict(label=label,status='failed',three_hole_spec=asdict(trial),error=repr(error))
        row['elapsed_seconds']=time.monotonic()-started;existing.append(row)
        save(folder/'candidates.json',existing)
        print('LOCAL',label,row['status'],row.get('g_th_cm'),row.get('beam',{}).get('ellipticity'),row.get('g_th_gap_fraction'),flush=True)
    ranked=sorted([r for r in existing if r['status']=='completed'],key=lambda r:r['score'])
    save(folder/'summary.json',dict(attempts=len(existing),completed=len(ranked),best=ranked[:10],
        scope='Fixed physical period and 3-branch exploratory local candidates; finalists require expanded census, actual target retuning and closure.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['known','local','stripe-audit','retune','stability','package'])
    p.add_argument('--label',default=None);p.add_argument('--cells',type=int,default=91)
    p.add_argument('--order',type=int,default=48);p.add_argument('--dz',type=float,default=1.)
    p.add_argument('--roots',type=int,default=6)
    p.add_argument('--candidates',type=Path,help='Optional explicit local shortlist, retaining actual parameter records')
    p.add_argument('--candidate',type=Path,help='Known-stage candidate seed folder, never an old result output')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.label is None:args.label='baseline_N91' if args.stage=='known' else 'local_scan'
    stamp=time.time_ns();(OUT/f'executed_{args.stage}_{stamp}.py').write_bytes(Path(__file__).read_bytes())
    save(OUT/f'runtime_{args.stage}_{stamp}.json',dict(argv=sys.argv,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/pcselsim').glob('*.py')},
        old_summary_sha256=sha(OLD/'summary.json'),old_union_sha256=sha(UNION/'complex_fields.npz')))
    with threadpool_limits(limits=1):
        {'known':known,'local':local,'stripe-audit':stripe_audit,'retune':retune,'stability':stability,'package':package}[args.stage](args)


if __name__=='__main__':main()
