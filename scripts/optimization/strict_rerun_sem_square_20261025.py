"""Post-audit square PCSEL search with explicit pole identity and closure.

中文：固定300微米方形器件和文献层栈，重新扫描主/副孔与竞争模式。
完整粗QZ只作筛选；正式结果使用真实细网格本征场与实频闭合，不修改旧结果。
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import splu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcselsim.custom_analysis import FourWaveOpticalSpec,LinearMode,finite_area_eigenproblem
from pcselsim.finite_discretization import relative_residual
from pcselsim.frequency_cwt import background_detuning
from pcselsim.mode_identity import independent_eigenmode,independent_mode_indices
from pcselsim.fabrication import evaluate_fabrication,inclusion_boundary
from pcselsim.polarization import polarization_metrics,stokes_maps
from pcselsim.band_structure import square_four_wave_band_diagram,plot_band_diagram
from scripts.optimization.matched300_square_search import Design,build_model,build_cell,RULES,BASE,threshold,direct_vector_far_field,acceptable
from scripts.optimization.matched300_square_mesh import interpolate_field
from scripts.diagnostics.audit_counterpropagating_stripes import stripe_pair
from scripts.optimization.matched300_square_finalize import close_saved,design_period

OUT=ROOT/'results/current/strict_rerun_20261025/semiconductor_square'
OLD=ROOT/'results/current/matched300_20261021/square'


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,complex):return [x.real,x.imag]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2),encoding='utf-8')


def old_case(role='best_isolation'):
    index=json.loads((OLD/'final_summary.json').read_text(encoding='utf-8'))
    row=next(r for r in index['candidates'] if r['role']==role)
    path=Path(row['summary_path']);summary=json.loads(path.read_text(encoding='utf-8'))
    return Design(**summary['design']),summary,path.parent/'final_fields.npz'


def make_problem(model,n):
    spec=FourWaveOpticalSpec(model.bragg_wavelength_nm,model.effective_index,model.group_index,
        300.,5.,(0.,)*4,(0.,)*4,n,'staggered',8,0,400)
    return spec,finite_area_eigenproblem(spec,model.coupling_m)


def record(value,field,model,problem,name='pole',samples=161):
    rx,ry=model.radiation_fields(field)
    mode=LinearMode(name,float(value.real),float(value.imag),field,0.,rx,ry,discretization='staggered')
    axis,fx,fy,beam=direct_vector_far_field(mode,model.bragg_wavelength_nm,300.,samples)
    budget=problem.loss_budget(field)
    row=dict(mode=name,delta_per_m=float(value.real),alpha_field_cm1=float(value.imag/100),
        **threshold(value.imag,model.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME],model.effective_index,300.),
        beam=beam,eigen_residual=relative_residual(problem.center_operator(),field,value),
        radiative_extraction_fraction=2*budget['local_amplitude_loss_per_m']/(2*value.imag+500),
        loss_budget=budget)
    return row,(axis,fx,fy,rx,ry)


def screen_one(d,D=12,dz=2.,n=9):
    trace=[];started=time.monotonic()
    for iteration in range(10):
        d,cell,m=build_model(d,D,dz,950.,False,96);spec,p=make_problem(m,n)
        p.near(0j,4,np.ones(4*n*n,complex),dense_limit=4*n*n)
        vv,ff=p._dense_spectrum
        indices=[i for i in np.argsort(vv.imag) if vv[i].imag>=-1e-7]
        val=vv[indices[0]];residual=background_detuning(950.,m.effective_index,d.lattice_nm)-val.real
        trace.append(dict(period_nm=d.lattice_nm,residual_per_m=residual))
        if abs(residual)<.05:break
        beta=2*np.pi*m.effective_index/950e-9
        beta0=-val.real+np.sqrt(val.real**2+beta**2)
        d=replace(d,lattice_nm=float(2*np.pi/beta0*1e9))
    else:raise RuntimeError('Coarse actual-frequency period design failed')
    rows=[record(vv[i],ff[:,i].reshape(4,n,n),m,p,f'coarse_{rank}',129)[0] for rank,i in enumerate(indices[:6])]
    return dict(status='completed',design=asdict(d),D=D,dz_nm=dz,grid=n,vertices=96,
        lowest=rows[0],ordered_poles=rows,relative_material_gap=rows[1]['g_th_cm1']/rows[0]['g_th_cm1']-1,
        fabrication=evaluate_fabrication(cell.inclusions,d.lattice_nm,rules=RULES),period_trace=trace,
        complete_coarse_count=len(vv),seconds=time.monotonic()-started,
        scope='Complete coarse QZ and coarse actual950 period design only. N9 far-field metrics are screening heuristics, not qualified +/-1degree beam certification.')


def proposals(count):
    parents=[old_case(role)[0] for role in ('best_isolation','best_low_threshold','best_balanced')]
    base=parents[0];items=list(parents)
    for key,values in dict(fill=[base.fill-.006,base.fill-.003,base.fill+.003,base.fill+.006],
        main_share=[.51,.525,.55],distance=[base.distance-.02,base.distance-.01,base.distance+.01,base.distance+.02],
        position_angle_deg=[base.position_angle_deg-2,base.position_angle_deg-1,base.position_angle_deg+1,base.position_angle_deg+2],
        satellite_aspect=[base.satellite_aspect-.10,base.satellite_aspect-.05,base.satellite_aspect+.05,base.satellite_aspect+.10],
        satellite_angle_deg=[base.satellite_angle_deg-4,base.satellite_angle_deg-2,base.satellite_angle_deg+2,base.satellite_angle_deg+4]).items():
        for value in values:items.append(replace(base,**{key:value},tag='axis_'+key))
    for aspect in (1.03,1.08):
        for angle in (0.,45.,90.):items.append(replace(base,main_shape='ellipse',main_aspect=aspect,main_angle_deg=angle,tag='central_ellipse'))
    for kind in ('circle','ellipse','superellipse'):
        items.append(replace(base,satellite_shape=kind,tag='satellite_'+kind))
    rng=np.random.default_rng(2026102501)
    names=['fill','main_share','distance','position_angle_deg','satellite_aspect','satellite_angle_deg']
    scales=np.array([.004,.025,.012,1.5,.07,3.])
    while len(items)<count:
        parent=parents[0 if len(items)%4 else 1]
        values=np.array([getattr(parent,k) for k in names])+rng.normal(size=6)*scales
        values[0]=np.clip(values[0],.08,.27);values[1]=np.clip(values[1],.5,.8)
        values[2]=np.clip(values[2],.40,.68);values[4]=np.clip(values[4],1.,2.)
        d=replace(parent,**dict(zip(names,map(float,values))),tag=f'joint_{len(items)}')
        if evaluate_fabrication(build_cell(d,96).inclusions,d.lattice_nm,rules=RULES)['margin_pass']:items.append(d)
    return items[:count]


def do_screen(count):
    directory=OUT/'screen';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'records.json';rows=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    designs=proposals(count);save(directory/'designs.json',[asdict(x) for x in designs])
    for i,d in enumerate(designs[len(rows):],start=len(rows)):
        try:r=screen_one(d)
        except Exception as e:r=dict(status='failed',design=asdict(d),error=repr(e))
        rows.append(r);save(path,rows)
        print('screen',i+1,len(designs),r['status'],r.get('lowest',{}).get('g_th_cm1'),r.get('relative_material_gap'),flush=True)
    return rows


def inverse_candidate(problem,target,seed):
    op=problem.center_operator();q=seed.ravel()/np.linalg.norm(seed);reference=q.copy();z=target+10+10j
    for _ in range(12):
        try:
            link=splu((problem.stiffness-z*problem.mass).tocsc()).solve(q)
            new=problem.mass@link;norm=np.linalg.norm(new)
            if not np.isfinite(norm) or norm<1e-30:break
            new/=norm;applied=op@new;value=complex(np.vdot(new,applied))
            if relative_residual(op,new,value)<1e-10 and abs(np.vdot(reference,new))**2>.2:
                return np.array([value]),new[:,None],'inverse_original_pencil'
            q=new;z=value+1e-8+2e-8j
        except (RuntimeError,ValueError):break
    vv,ff=problem.near(target+10+10j,8,seed.ravel(),maxiter=250,dense_limit=0,tol=1e-10)
    return vv,ff,'Arnoldi_original_pencil'


def survey(d,folder,n=65,D=64,dz=1.,prior=True,wide=True,coarse_grid=13):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    _,cell,m=build_model(d,D,dz,950.,False,96)
    cs,cp=make_problem(m,coarse_grid);dim=4*coarse_grid*coarse_grid
    cp.near(0j,4,np.ones(dim,complex),dense_limit=dim)
    cv,cf=cp._dense_spectrum;ci=[i for i in np.argsort(cv.imag) if cv[i].imag>=-1e-7]
    spec,p=make_problem(m,n);targets=[]
    for i in ci[:20]:targets.append((cv[i],interpolate_field(cf[:,i].reshape(4,coarse_grid,coarse_grid),n),'full_coarse_QZ_low20'))
    if prior:
        _,old,path=old_case()
        with np.load(path,allow_pickle=False) as z:
            for row in old['frequency_closed_modes'][:10]:
                field=interpolate_field(z[row['mode']+'_field'],n)
                oldm=build_model(d,D,dz,row['wavelength_nm'],False,96)[2]
                flat=field.reshape(4,-1)
                val=complex(row['delta_per_m'],100*row['alpha_field_cm1'])
                val+=np.vdot(flat,(m.coupling_m-oldm.coupling_m)@flat)/np.vdot(flat,flat)
                targets.append((val,field,'old_closed_branch_challenge'))
    rng=np.random.default_rng(2026102565);random=rng.normal(size=(4,n,n))+1j*rng.normal(size=(4,n,n));random/=np.linalg.norm(random)
    spacing=np.pi/300e-6
    for value in m.eigenvalues_m:
        for k in (-1,0,1):targets.append((value+k*spacing,random,'Gamma_neighbour'))
    if wide:
        for x in np.linspace(m.eigenvalues_m.real.min()-4*spacing,m.eigenvalues_m.real.max()+4*spacing,10):
            targets.append((complex(x,.1),random,'real_axis_challenge'))
    pair_controls=[];coordinate=((np.arange(n)+.5)/n-.5)*300e-6
    yy,xx=np.meshgrid(coordinate,coordinate,indexing='ij')
    for pair in (0,1):
        gamma=m.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME]*BASE.ACTIVE_INDEX/m.effective_index
        result=stripe_pair(m.coupling_m[2*pair:2*pair+2,2*pair:2*pair+2],300.,gamma,5.,grids=(51,101))
        pair_controls.append(result);pole=result['independent_continuous_transfer_pole']
        if not pole['converged']:continue
        along,across=(xx,yy) if pair==0 else (yy,xx)
        envelope=np.maximum(np.cos(np.pi*along/300e-6),0.)*np.exp(-(across/50e-6)**2)
        for q in (.35,.65):
            stripe=envelope*np.exp(1j*q*np.pi/(300e-6/n)*across)
            initial=np.zeros((4,n,n),complex);initial[2*pair]=stripe;initial[2*pair+1]=1j*stripe
            initial/=np.linalg.norm(initial)
            for fraction in (0.,.5,1.):targets.append((complex(pole['delta_per_m'],fraction*pole['alpha_field_per_m']),initial,'independent_stripe_limit'))
    values=[];fields=[];audit=[];op=p.center_operator()
    for index,(target,seed,kind) in enumerate(targets):
        try:
            if kind in ('full_coarse_QZ_low20','old_closed_branch_challenge'):
                vv,ff,solver=inverse_candidate(p,target,seed)
            else:
                vv,ff=p.near(target+10+10j,8,seed.ravel(),maxiter=150,dense_limit=0,tol=1e-10);solver='bounded_Arnoldi'
            before=len(values)
            for value,field in zip(vv,ff.T):
                if value.imag< -1e-7 or relative_residual(op,field,value)>2e-7:continue
                if not independent_eigenmode(value,field,values,fields):continue
                values.append(complex(value));fields.append(field)
            audit.append(dict(kind=kind,target=[target.real,target.imag],solver=solver,returned=len(vv),new=len(values)-before))
        except Exception as e:audit.append(dict(kind=kind,target=[target.real,target.imag],error=repr(e)))
        if (index+1)%8==0:print('survey',folder.name,index+1,len(targets),'distinct',len(values),flush=True)
        save(folder/'search_progress.json',dict(targets=audit,distinct=len(values)))
    rows=[];arrays={'coupling_m':m.coupling_m}
    for rank,index in enumerate(np.argsort(np.imag(values))[:24]):
        value,field=values[index],fields[index].reshape(4,n,n)
        row,data=record(value,field,m,p,f'pole_{rank}',161);rows.append(row)
        if rank<16:arrays[f'pole_{rank}_field']=field
    result=dict(status='completed',design=asdict(d),D=D,dz_nm=dz,grid=n,vertices=96,
        ordered_poles=rows,lowest=rows[0],relative_material_gap=rows[1]['g_th_cm1']/rows[0]['g_th_cm1']-1,
        distinct_poles=len(values),full_QZ=False,complete_coarse_QZ_count=len(cv),coarse_grid=coarse_grid,
        search_targets=audit,independent_pair_controls=pair_controls,seconds=time.monotonic()-started,
        scope=f'Complete N{coarse_grid} coarse spectrum, prior-mode and high-transverse stripe challenges; strict corrected complex-pole identity and original fine residual. Bounded fine spectrum, not exhaustive.')
    save(folder/'spectral_challenge.json',result);np.savez_compressed(folder/'lateral_fields.npz',**arrays)
    return result


def do_survey(args):
    d=old_case()[0] if args.design is None else Design(**json.loads(args.design.read_text(encoding='utf-8'))['design'])
    return survey(d,OUT/args.tag,args.grid,args.D,args.dz,prior=args.design is None,wide=not args.fast,coarse_grid=args.coarse_grid)


def do_close(args):
    directory=OUT/args.source
    source=json.loads((directory/'spectral_challenge.json').read_text(encoding='utf-8'))
    raw=directory/'lateral_fields.npz';output=OUT/args.tag
    output.mkdir(parents=True,exist_ok=True)
    if args.fixed_design:
        # Retain the source operator/fields as seeds while controlling the
        # exact physical geometry across D/dz studies.
        # 固定实际几何做D/dz控制；参考C与参考场仍成对保留。
        source={**source,'design':json.loads(args.fixed_design.read_text(encoding='utf-8'))['design']}
    if args.retune:
        d,trace,residual=design_period(source,raw,950.,args.D,args.dz)
        save(output/'period_design.json',dict(design=asdict(d),trace=trace,residual_per_m=residual))
        # Reference fields remain seeds of the old operator. Their old C is
        # retained for the exact C(lambda)-C_reference predictor in close_saved.
        # 原参考C和场保持配对，只用作预测；所有输出仍重新求解新几何的原方程。
        source={**source,'design':asdict(d)}
    result=close_saved(source,raw,output,args.D,args.dz,args.closed_modes)
    d=Design(**result['design']);keep=result['frequency_closed_modes'][:max(6,args.closed_modes)]
    with np.load(output/'final_fields.npz',allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    readback=[]
    for row in keep:
        name=row['mode'];field=arrays[name+'_field']
        m=build_model(d,args.D,args.dz,row['wavelength_nm'],False,96)[2]
        _,p=make_problem(m,result['grid']);value=complex(row['delta_per_m'],100*row['alpha_field_cm1'])
        residual=relative_residual(p.center_operator(),field,value)
        if residual>2e-7:raise RuntimeError('Final saved field failed its fresh original pencil')
        rx,ry=m.radiation_fields(field)
        mode=LinearMode(name,float(value.real),float(value.imag),field,0.,rx,ry,discretization='staggered')
        axis,fx,fy,beam=direct_vector_far_field(mode,row['wavelength_nm'],300.,241)
        errors={k:abs(beam[k]-row['beam'][k]) for k in ('ellipticity','center_to_peak','radial_anisotropy_l2')}
        if max(errors.values())>1e-9:raise RuntimeError('Final far field readback differs')
        arrays.update({name+'_far_x':fx,name+'_far_y':fy,name+'_radiation_x':rx,
            name+'_radiation_y':ry,name+'_coupling_m':m.coupling_m})
        # Store each closed mode on its own common Jones scale; polarization
        # maps remain diagnostics and never enter the threshold optimization.
        # 每根闭频态均保留同尺度Jones/Stokes；偏振诊断不反向修改物理场。
        arrays.update({name+'_stokes_'+key:value for key,value in stokes_maps(fx,fy).items()})
        readback.append(dict(mode=name,original_pencil_residual=residual,beam_errors=errors,
            true_frequency_residual_per_m=background_detuning(row['wavelength_nm'],m.effective_index,d.lattice_nm)-value.real))
    np.savez_compressed(output/'final_fields.npz',axis_deg=axis,**{k:v for k,v in arrays.items() if k!='axis_deg'})
    geometry=[]
    for i,hole in enumerate(build_cell(d,96).inclusions):
        points=inclusion_boundary(hole,np.eye(2))*d.lattice_nm
        geometry.append(dict(role='central_main' if i==0 else 'diagonal_satellite',boundary_nm=points))
    save(output/'actual_geometry_nm.json',dict(period_nm=d.lattice_nm,side_um=300.,design=asdict(d),holes=geometry))
    m=build_model(d,args.D,args.dz,result['lowest']['wavelength_nm'],False,96)[2]
    band=square_four_wave_band_diagram(m.coupling_m,d.lattice_nm,m.bragg_wavelength_nm,m.effective_index,group_index=m.group_index)
    plot_band_diagram(band,output/'local_bands.png','Square four-wave | actual-frequency local bands')
    save(output/'local_bands.json',asdict(band))
    result.update(saved_jones_modes=[r['mode'] for r in keep],saved_stokes_map_modes=[r['mode'] for r in keep],strict_readback=readback,
        nominal_aperture_um=[300.,300.],target_wavelength_nm=950.,actual_target_tolerance_nm=.05,
        actual_target_pass=abs(result['lowest']['wavelength_nm']-950.)<=.05,
        source_stage=str(directory.relative_to(ROOT)),current_audit_note='Declared near-Bragg scalar CWT approximations retained; no full-wave or nonlinear mode-stability proof.')
    save(output/'summary.json',result)
    modes_figure(result,arrays,axis,output)
    return result


def modes_figure(result,arrays,axis,output):
    rows=result['frequency_closed_modes'][:6]
    fig,axes=plt.subplots(2,len(rows),figsize=(3.1*len(rows),6.0),layout='constrained')
    for index,row in enumerate(rows):
        name=row['mode'];near=np.sum(abs(arrays[name+'_field'])**2,axis=0)
        far=abs(arrays[name+'_far_x'])**2+abs(arrays[name+'_far_y'])**2
        axes[0,index].imshow(near/near.max(),origin='lower',extent=(-150,150,-150,150),cmap='inferno',vmin=0,vmax=1)
        axes[0,index].set(title=f'M{index+1}: {row["g_th_cm1"]:.3f} cm$^{{-1}}$\n{row["wavelength_nm"]:.5f} nm',xlabel='x (um)',ylabel='y (um)')
        axes[1,index].imshow(far/far.max(),origin='lower',extent=(axis[0],axis[-1],axis[0],axis[-1]),cmap='inferno',vmin=0,vmax=1)
        axes[1,index].set(xlim=(-.5,.5),ylim=(-.5,.5),xlabel='theta x (deg)',ylabel='theta y (deg)')
    fig.suptitle('Post-audit square PCSEL | individually frequency-closed finite modes; 300 x 300 um')
    fig.savefig(output/'closed_modes.png',dpi=190);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('screen','survey','close'))
    p.add_argument('--count',type=int,default=96);p.add_argument('--D',type=int,default=64)
    p.add_argument('--dz',type=float,default=1.);p.add_argument('--grid',type=int,default=65)
    p.add_argument('--coarse-grid',type=int,default=13)
    p.add_argument('--design',type=Path);p.add_argument('--tag',default='baseline_N65');p.add_argument('--fast',action='store_true')
    p.add_argument('--source',default='baseline_N65');p.add_argument('--closed-modes',type=int,default=10);p.add_argument('--retune',action='store_true')
    p.add_argument('--fixed-design',type=Path,help='Closed-source physical design, used for unchanged-geometry D/dz controls')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    stamp=time.time_ns();(OUT/f'executed_{args.phase}_{stamp}.py').write_bytes(Path(__file__).read_bytes())
    save(OUT/f'runtime_{args.phase}_{stamp}.json',dict(argv=sys.argv,source_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (ROOT/'src/pcselsim').glob('*.py')},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    {'screen':lambda:do_screen(args.count),'survey':lambda:do_survey(args),'close':lambda:do_close(args)}[args.phase]()


if __name__=='__main__':main()
