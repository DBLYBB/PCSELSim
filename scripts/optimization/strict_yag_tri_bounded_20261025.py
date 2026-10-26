"""Bounded post-audit YAG triangular spectrum and actual-frequency closure.

中文：冻结正确的局扫记录/复场对；粗QR、分阶段细谱挑战均复验原K/M。
Slow supplementary shifts run in time-bounded child processes. A timeout is
unresolved evidence, never a dark-mode rejection or a physics modification.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib,sys,time,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace,asdict
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(Path(__file__).parent)]
import numpy as np
from scipy.sparse.linalg import splu
from scipy.constants import c
import matched300_yag_search as b
from strict_yag_qr_close_20261025 import qr_spectrum
from pcselsim.finite_discretization import relative_residual
from pcselsim.triangular_finite import triangular_finite_eigenproblem,TriangularFiniteSpec
from pcselsim.mode_identity import independent_eigenmode
OUT=ROOT/'results/current/strict_rerun_20261025/YbYAG_triangular'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inverse(problem,target,seed):
    """Change only numerical shifts; validate the unchanged physical pencil."""
    operator=problem.center_operator();q=seed.ravel()/np.linalg.norm(seed);ref=q.copy();z=target+complex(.5,.7)
    history=[]
    for i in range(16):
        links=splu((problem.stiffness-z*problem.mass).tocsc()).solve(q)
        q=problem.mass@links;q/=np.linalg.norm(q);a=operator@q;v=complex(np.vdot(q,a))
        residual=relative_residual(operator,q,v);history.append(dict(iteration=i+1,eigen_residual=residual))
        if residual<1e-11:return v,q,dict(iterations=i+1,eigen_residual=residual,overlap=float(abs(np.vdot(ref,q))**2),history=history)
        z=v+complex(1e-8,2e-8)
    raise RuntimeError(f'Inverse iteration original-pencil residual unresolved: {history}')


def freeze_outer(args):
    source=OUT/'local_screen/best_record.json';fields=OUT/'local_screen/best_fields.npz'
    r=read(source)
    if r['index']!=56:raise RuntimeError('Current archived best source must still be index56')
    folder=OUT/f'candidate{args.index}_frozen';folder.mkdir(parents=True,exist_ok=True)
    if args.index!=56:
        if not (folder/'input.json').exists():prepare_candidate(args.index,folder)
        source=folder/'input.json';fields=folder/'source_fields.npz';r=read(source)
    else:
        (folder/'input.json').write_bytes(source.read_bytes());(folder/'source_fields.npz').write_bytes(fields.read_bytes())
    references=[];checks=[]
    with np.load(folder/'source_fields.npz',allow_pickle=False) as raw:
        for row in r['closed_modes']:
            if row['closed_TE_family']!=0:raise RuntimeError('TE0 source required')
            model,_,_=b.optical('triangular',r['design'],16,2.,row['wavelength_nm'],r['period_nm'],0)
            spec,_,p,mesh=b.finite('triangular',model,26,8,False)
            f=raw[row['name']+'_field'];C=raw[row['name']+'_C']
            ce=float(np.linalg.norm(C-model.result.coupling_m)/np.linalg.norm(C));v=complex(row['delta_per_m'],row['alpha_field_per_m'])
            residual=relative_residual(p.center_operator(),f,v)
            if ce>2e-9 or residual>2e-7:raise RuntimeError('Frozen source C/pencil mismatch')
            checks.append(dict(name=row['name'],coupling_relative_error=ce,original_pencil_residual=residual))
            references.append((row,model,spec,mesh,f.copy(),v))
    b.save(folder/'source_readback.json',dict(status='passed',record_index=args.index,input_sha256=sha(source),fields_sha256=sha(fields),checks=checks,
        note='Frozen record and fields copied as a pair before any new solve; earlier mislabeled candidate43_seed_fields not used.'))
    row,model,spec,mesh,field,value=references[0];period=r['period_nm'];trace=[]
    for i in range(6):
        new,_,_=b.optical('triangular',r['design'],32,1.,1030.,period,0)
        ns,_,p,nmesh=b.finite('triangular',new,52,8,False)
        f=b.interpolate_envelope('triangular',field,spec,mesh,ns,nmesh);flat=f.reshape(6,-1)
        change=new.result.coupling_m-model.result.coupling_m
        predictor=value+np.einsum('ik,ij,jk->',flat.conj(),change,flat)/np.sum(abs(flat)**2)
        val,f,init=b.follow_reference_pole(p,f,predictor)
        wave,opt,g,local_spec,mode,history=b.close_seed('triangular',r['design'],period,new,(val,f),ns,nmesh,32,1.,0)
        trace.append(dict(iteration=i,period_nm=period,wavelength_nm=wave,initialization=init,frequency_trace=history))
        b.save(folder/'outer_period_progress.json',dict(trace=trace))
        print('OUTER',args.index,i,period,wave,mode.alpha_per_m,flush=True)
        if abs(wave-1030.)<=.01:
            b.save(folder/'outer_period_design.json',dict(target_nm=1030.,tolerance_nm=.01,frozen_period_nm=period,trace=trace))
            return
        period*=1030./wave;model,spec,mesh,field,value=opt,local_spec,nmesh,mode.fields,complex(mode.delta_per_m,mode.alpha_per_m)
    raise RuntimeError('Outer target unresolved')


def prepare_candidate(index,folder):
    """Create a genuine record/field pair for a backup, never relabel best fields."""
    from strict_rerun_yag_tri_20261025 import geometric_continuation
    base=read(OUT/'candidate56_frozen/input.json');target=next(r for r in read(OUT/'local_screen/records.json') if r['index']==index)
    arrays={};closed=[]
    with np.load(OUT/'candidate56_frozen/source_fields.npz',allow_pickle=False) as raw:
        for row in base['closed_modes']:
            reference,_,_=b.optical('triangular',base['design'],16,2.,row['wavelength_nm'],base['period_nm'],0)
            spec,_,p,mesh=b.finite('triangular',reference,26,8,False)
            seed=(complex(row['delta_per_m'],row['alpha_field_per_m']),raw[row['name']+'_field'])
            model,continued,geom=geometric_continuation(base['design'],target['design'],base['period_nm'],reference,spec,mesh,seed,16,2.)
            wave,opt,g,ls,mode,trace=b.close_seed('triangular',target['design'],base['period_nm'],model,continued,spec,mesh,16,2.,0)
            mode=replace(mode,name=row['name']);out,data=b.mode_row('triangular',opt,g,ls,mode,wave,81);axis,fx,fy,widths,qe,qg=data
            out.update(wavelength_nm=wave,closed_TE_family=0,frequency_residual_per_m=trace[-1]['frequency_residual_per_m'],frequency_trace=trace,geometry_trace=geom)
            closed.append(out);name=mode.name
            arrays.update({name+'_field':mode.fields,name+'_C':opt.result.coupling_m,name+'_far_x':fx,name+'_far_y':fy,name+'_widths':widths,name+'_qenergy':qe,name+'_qgain':qg,
                'angle_deg':axis,'x_um':mode.grid.x_um,'y_um':mode.grid.y_um})
            print('PREPARED_BACKUP',index,name,wave,out['required_material_gain_cm1'],flush=True)
    record={**target,'closed_modes':closed,'lowest':closed[0],'second':closed[1],'wavelength_nm':closed[0]['wavelength_nm'],
        'source_note':'Fresh geometric continuation from verified56 fields and actual frequency closure; not a renamed field file.'}
    b.save(folder/'input.json',record);np.savez_compressed(folder/'source_fields.npz',**arrays)


def worker(args):
    request=read(args.request);folder=args.request.parent
    with np.load(folder/request['input_npz'],allow_pickle=False) as z:C=z['C'];seed=z['seed']
    spec=TriangularFiniteSpec(150.,request['grid'],0.,8,'square','staggered')
    p,mesh=triangular_finite_eigenproblem(SimpleNamespace(coupling_m=C),spec)
    target=complex(*request['target']);started=time.monotonic()
    try:
        if request['method']=='inverse':
            v,f,info=inverse(p,target,seed);vv=np.array([v]);ff=f[:,None]
        else:
            vv,ff=p.near(target+complex(.5,.7),request.get('count',6),seed.ravel(),tol=1e-10,maxiter=request.get('maxiter',60),dense_limit=0)
            info=dict(solver='unchanged_pencil_bounded_Arnoldi')
        original=p.center_operator();valid=[i for i,(v,f) in enumerate(zip(vv,ff.T)) if v.imag>=-1e-7 and relative_residual(original,f,v)<=2e-7]
        np.savez_compressed(folder/request['output_npz'],values=vv[valid],fields=ff[:,valid])
        b.save(folder/request['output_json'],dict(status='completed',returned=len(valid),seconds=time.monotonic()-started,driver_sha256=sha(__file__),**info))
    except Exception as error:b.save(folder/request['output_json'],dict(status='unresolved',error=repr(error),seconds=time.monotonic()-started,driver_sha256=sha(__file__)))


def freeze_baseline(args):
    source=ROOT/'results/current/matched300_20261021/yag/final/triangular_bright_round_v5'
    summary=read(source/'summary.json');rows=read(source/'closed_optical_modes.json');period=summary['period_nm']
    folder=OUT/'baseline_frozen';folder.mkdir(parents=True,exist_ok=True);checks=[]
    with np.load(source/'closed_fields.npz',allow_pickle=False) as z:
        for row in rows:
            if row['closed_TE_family']!=0:raise RuntimeError('Baseline source includes unsupported TE family for this explicit TE0 comparison')
            model,_,_=b.optical('triangular',summary['design'],32,1.,row['wavelength_nm'],period,0)
            spec,_,p,mesh=b.finite('triangular',model,52,8,False)
            C=z[row['name']+'_C'];f=z[row['name']+'_field'];value=complex(row['delta_per_m'],row['alpha_field_per_m'])
            ce=float(np.linalg.norm(model.result.coupling_m-C)/np.linalg.norm(C));er=relative_residual(p.center_operator(),f,value)
            beta=2*np.pi*model.result.effective_index/(row['wavelength_nm']*1e-9);beta0=4*np.pi/np.sqrt(3)/(period*1e-9)
            fr=float((beta**2-beta0**2)/(2*beta0)-value.real)
            if ce>2e-9 or er>2e-7 or abs(fr)>.05:raise RuntimeError('Baseline saved root failed its fresh physical readback')
            checks.append(dict(name=row['name'],coupling_relative_error=ce,original_pencil_residual=er,frequency_residual_per_m=fr))
            print('BASELINE_READBACK',len(checks),len(rows),er,fr,flush=True)
    record=dict(lattice='triangular',design=summary['design'],period_nm=period,settings=dict(order=32,dz_nm=1.,grid=52),
        lowest=rows[0],closed_modes=rows,wavelength_nm=rows[0]['wavelength_nm'])
    b.save(folder/'input.json',record);b.save(folder/'outer_period_design.json',dict(frozen_period_nm=period,scope='Same fixed old physical period; no retuning for baseline comparison'))
    b.save(folder/'source_readback.json',dict(status='passed',source_summary=str(source/'summary.json'),source_fields=str(source/'closed_fields.npz'),
        source_sha256={str(p):sha(p) for p in (source/'summary.json',source/'closed_optical_modes.json',source/'closed_fields.npz')},checks=checks,
        scope='Every saved TE0 root rebuilt at its own actual wavelength; this is not a fresh spectrum census.'))


def bounded_query(folder,index,C,grid,target,seed,kind,method='inverse',timeout=45.):
    work=folder/'queries';work.mkdir(parents=True,exist_ok=True);stem=f'{index:03d}'
    ip=stem+'_in.npz';op=stem+'_out.npz';jp=stem+'_out.json'
    req=dict(grid=grid,target=[target.real,target.imag],method=method,count=6,maxiter=60,input_npz=ip,output_npz=op,output_json=jp)
    req.update(coupling_sha256=hashlib.sha256(np.asarray(C).tobytes()).hexdigest(),seed_sha256=hashlib.sha256(np.asarray(seed).tobytes()).hexdigest())
    request=work/(stem+'.json')
    if request.exists() and read(request)!=req:raise RuntimeError('Cached query input/operator changed; use a new output tag')
    b.save(request,req)
    if not (work/jp).exists():
        np.savez_compressed(work/ip,C=C,seed=seed)
        start=time.monotonic()
        try:
            proc=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__)),'worker','--request',str(request)],capture_output=True,text=True,timeout=timeout)
            if not (work/jp).exists():b.save(work/jp,dict(status='worker_failed',returncode=proc.returncode,stderr=proc.stderr[-2000:]))
        except subprocess.TimeoutExpired:b.save(work/jp,dict(status='bounded_unresolved',seconds=time.monotonic()-start,timeout_s=timeout))
    audit=read(work/jp);audit.update(index=index,kind=kind,target=req['target'],method=method,timeout_s=timeout)
    if audit['status']=='completed' and not (work/op).exists():
        raise RuntimeError('Completed query arrays were compacted: use a fresh output tag; never interpret removed arrays as zero physical modes')
    if audit['status']=='completed' and (work/op).exists():
        with np.load(work/op,allow_pickle=False) as z:return z['values'].copy(),z['fields'].copy(),audit
    return np.array([],complex),np.empty((seed.size,0),complex),audit


def survey(args):
    frozen=OUT/args.frozen;r=read(frozen/'input.json');period=read(frozen/'outer_period_design.json')['frozen_period_nm']
    folder=OUT/args.tag;folder.mkdir(parents=True,exist_ok=True)
    model,_,_=b.optical('triangular',r['design'],32,1.,1030.,period,0)
    spec,_,problem,mesh=b.finite('triangular',model,args.grid,8,False)
    cs,_,cp,cm=b.finite('triangular',model,9,8,False)
    coarsefile=frozen/'coarse_QR.npz';auditfile=frozen/'coarse_QR_audit.json'
    if coarsefile.exists():
        with np.load(coarsefile,allow_pickle=False) as z:cv=z['values'];cf=z['fields']
        qr=read(auditfile)
    else:
        cv,cf,qr=qr_spectrum(cp);np.savez_compressed(coarsefile,values=cv,fields=cf);b.save(auditfile,qr)
    valid=[i for i in np.argsort(cv.imag) if cv[i].imag>=-1e-7];targets=[]
    for i in valid[:40]:
        field=b.interpolate_envelope('triangular',cf[:,i].reshape(6,cm.points),cs,cm,spec,mesh)
        targets.append((complex(cv[i]),field,'complete_coarse_QR_low40','inverse',45.))
    if args.previous:
        prev=OUT/args.previous;pr=read(prev/'frozen_census.json')
        ps,_,_,pm=b.finite('triangular',model,pr['grid'],8,False)
        with np.load(prev/pr.get('fields_file','frozen_fields.npz'),allow_pickle=False) as z:
            # Low40 plus each δ cluster/Gamma representative preserve wider bands.
            selected=set(range(min(40,len(pr['poles']))));selected.update(cluster_ranks(pr['poles'],model.result.eigenvalues_m)[0])
            for i in sorted(selected):
                field=b.interpolate_envelope('triangular',z[f'pole_{i}_field'],ps,pm,spec,mesh)
                row=pr['poles'][i];targets.append((complex(row['delta_per_m'],row['alpha_field_per_m']),field,'stage_N26_actual_pole','inverse',45.))
    random=np.random.default_rng(103056+args.grid).normal(size=(6,mesh.points))+1j*np.random.default_rng(560103+args.grid).normal(size=(6,mesh.points));random/=np.linalg.norm(random)
    for value in model.result.eigenvalues_m:targets.append((complex(value),random,'Gamma_supplement','near',30.))
    if not args.previous:
        from scripts.diagnostics.audit_counterpropagating_stripes import stripe_pair
        xx,yy=mesh.x_um*1e-6,mesh.y_um*1e-6;pair_audits=[]
        for pair,(dx,dy) in enumerate(((.5,-np.sqrt(3)/2),(.5,np.sqrt(3)/2),(1.,0.))):
            length=300. if pair==2 else 300/np.sin(np.pi/3)
            gain=model.result.vertical_mode.confinement['Yb:YAG gain']*1.8166/model.result.effective_index
            pair_result=stripe_pair(model.result.coupling_m[2*pair:2*pair+2,2*pair:2*pair+2],length,gain,.1,grids=(51,101))
            pair_audits.append(pair_result);pole=pair_result['independent_continuous_transfer_pole']
            if not pole['converged']:continue
            along,across=dx*xx+dy*yy,-dy*xx+dx*yy
            env=np.maximum(np.cos(np.pi*along/(length*1e-6)),0)*np.exp(-(across/40e-6)**2)
            for q in (.35,.65):
                stripe=env*np.exp(1j*q*np.pi/mesh.spacing_m*across);f=np.zeros((6,mesh.points),complex);f[2*pair]=stripe;f[2*pair+1]=1j*stripe;f/=np.linalg.norm(f)
                for fraction in (1.,.5,0.):targets.append((complex(pole['delta_per_m'],fraction*pole['alpha_field_per_m']),f,'analytic_stripe_N26','near',25.))
        b.save(folder/'independent_pair_controls.json',pair_audits)
    for value in np.linspace(-6e5,6e5,13):targets.append((complex(value,max(10.,min(model.result.eigenvalues_m.imag))),random,'wide_real_supplement','near',20.))
    values=[];fields=[];audits=[]
    for i,(target,seed,kind,method,limit) in enumerate(targets):
        vv,ff,audit=bounded_query(folder,i,model.result.coupling_m,args.grid,target,seed,kind,method,limit)
        before=len(values)
        for v,f in zip(vv,ff.T):
            if independent_eigenmode(v,f,values,fields):values.append(complex(v));fields.append(f.copy())
        audit['new_independent']=len(values)-before;audits.append(audit)
        b.save(folder/'search_progress.json',dict(completed_targets=i+1,planned_targets=len(targets),distinct_poles=len(values),targets=audits))
        print('CENSUS',args.tag,i+1,len(targets),len(values),audit['status'],flush=True)
    arrays={'coupling_m':model.result.coupling_m};rows=[]
    for rank,index in enumerate(np.argsort(np.imag(values))):
        v=values[index];field=fields[index].reshape(6,mesh.points)
        rows.append(dict(delta_per_m=v.real,alpha_field_per_m=v.imag,original_pencil_residual=relative_residual(problem.center_operator(),field,v)))
        arrays[f'pole_{rank}_field']=field
    np.savez_compressed(folder/'frozen_fields.npz',**arrays)
    b.save(folder/'frozen_census.json',dict(status='bounded_completed',design=r['design'],period_nm=period,order=32,dz_nm=1.,grid=args.grid,
        reference_wavelength_nm=1030.,coarse_QR_audit=str(auditfile),full_coarse_count=len(cv),poles=rows,targets=audits,
        failed_targets=[x for x in audits if x['status']!='completed'],scope='Complete coarse QR plus bounded fine queries. No full fine spectrum guarantee; timeouts remain unresolved.'))


def cluster_ranks(poles,gamma):
    groups=[]
    for rank,row in sorted(enumerate(poles),key=lambda t:t[1]['delta_per_m']):
        if not groups or row['delta_per_m']-groups[-1][-1][1]['delta_per_m']>500.:groups.append([])
        groups[-1].append((rank,row))
    selected=set();audit=[];cutoff=4*poles[0]['alpha_field_per_m']
    for group in groups:
        rank,row=min(group,key=lambda t:t[1]['alpha_field_per_m']);take=row['alpha_field_per_m']<=cutoff
        if take:selected.add(rank)
        audit.append(dict(min_delta=group[0][1]['delta_per_m'],max_delta=group[-1][1]['delta_per_m'],minimum_alpha_rank=rank,minimum_alpha_per_m=row['alpha_field_per_m'],selected=take))
    v=np.array([complex(r['delta_per_m'],r['alpha_field_per_m']) for r in poles])
    for g in gamma:selected.add(int(np.argmin(abs(v-g))))
    return selected,audit


def fine_stripes(args):
    """Also challenge newly resolved transverse scales on the fine grid.

    中文：小网格条纹延续之外，直接挑战细网格q=.35/.65，两种复移位。
    These are original-pencil inverse iterations with a hard wall-time bound;
    slow/failed seeds are retained, not interpreted as missing physical modes.
    """
    folder=OUT/args.tag;plan=read(folder/'frozen_census.json')
    if plan.get('direct_fine_stripe_challenge_completed'):return
    model,_,_=b.optical('triangular',plan['design'],32,1.,1030.,plan['period_nm'],0)
    spec,_,p,mesh=b.finite('triangular',model,plan['grid'],8,False)
    with np.load(folder/plan.get('fields_file','frozen_fields.npz'),allow_pickle=False) as z:
        fields=[z[f'pole_{i}_field'].ravel().copy() for i in range(len(plan['poles']))]
    values=[complex(r['delta_per_m'],r['alpha_field_per_m']) for r in plan['poles']]
    from scripts.diagnostics.audit_counterpropagating_stripes import stripe_pair
    targets=[];xx,yy=mesh.x_um*1e-6,mesh.y_um*1e-6;pair_audits=[]
    gain=model.result.vertical_mode.confinement['Yb:YAG gain']*1.8166/model.result.effective_index
    for pair,(dx,dy) in enumerate(((.5,-np.sqrt(3)/2),(.5,np.sqrt(3)/2),(1.,0.))):
        length=300. if pair==2 else 300/np.sin(np.pi/3)
        check=stripe_pair(model.result.coupling_m[2*pair:2*pair+2,2*pair:2*pair+2],length,gain,.1,grids=(51,101));pair_audits.append(check)
        pole=check['independent_continuous_transfer_pole']
        if not pole['converged']:continue
        along,across=dx*xx+dy*yy,-dy*xx+dx*yy
        envelope=np.maximum(np.cos(np.pi*along/(length*1e-6)),0)*np.exp(-(across/40e-6)**2)
        for q in (.35,.65):
            sheet=envelope*np.exp(1j*q*np.pi/mesh.spacing_m*across);seed=np.zeros((6,mesh.points),complex)
            seed[2*pair]=sheet;seed[2*pair+1]=1j*sheet;seed/=np.linalg.norm(seed)
            for fraction in (1.,0.):targets.append((complex(pole['delta_per_m'],fraction*pole['alpha_field_per_m']),seed,pair,q,fraction))
    audits=[];offset=len(plan['targets']);pool=ThreadPoolExecutor(max_workers=min(args.parallel_roots,2))
    futures=[pool.submit(bounded_query,folder,offset+i,model.result.coupling_m,plan['grid'],target,seed,'analytic_stripe_direct_fine','inverse',45.)
        for i,(target,seed,pair,q,fraction) in enumerate(targets)]
    for i,(target,seed,pair,q,fraction) in enumerate(targets):
        vv,ff,audit=futures[i].result()
        before=len(values)
        for value,field in zip(vv,ff.T):
            if independent_eigenmode(value,field,values,fields):values.append(complex(value));fields.append(field.copy())
        audit.update(pair=pair,normalized_transverse_q=q,imaginary_shift_fraction=fraction,new_independent=len(values)-before);audits.append(audit)
        b.save(folder/'direct_stripe_progress.json',dict(completed=i+1,planned=len(targets),targets=audits,distinct=len(values)))
        print('DIRECT_FINE_STRIPE',args.tag,i+1,len(targets),len(values),audit['status'],flush=True)
    pool.shutdown(wait=True)
    arrays={'coupling_m':model.result.coupling_m};rows=[];op=p.center_operator()
    for rank,index in enumerate(np.argsort(np.imag(values))):
        value=values[index];field=fields[index].reshape(6,mesh.points)
        rows.append(dict(delta_per_m=value.real,alpha_field_per_m=value.imag,original_pencil_residual=relative_residual(op,field,value)))
        arrays[f'pole_{rank}_field']=field
    np.savez_compressed(folder/'direct_frozen_fields.npz',**arrays)
    b.save(folder/'frozen_census_before_direct_stripes.json',plan)
    plan.update(poles=rows,fields_file='direct_frozen_fields.npz',targets=plan['targets']+audits,
        direct_fine_stripe_challenge_completed=True,direct_fine_pair_controls=pair_audits,
        failed_targets=plan['failed_targets']+[x for x in audits if x['status']!='completed'])
    # Publish the new JSON pointer last: the original JSON/NPZ pair remains
    # valid until the new complete field file exists.
    b.save(folder/'frozen_census.json',plan)


def closure_worker(args):
    """A bounded parent process can stop a difficult continuation cleanly."""
    request=read(args.request);folder=Path(request['case']);plan=read(folder/'frozen_census.json');rank=request['rank']
    progress=args.request.with_name(args.request.stem+'_progress.json');trace_calls=[];original=b.optical
    def logged_optical(*values,**kw):
        trace_calls.append(dict(wavelength_nm=values[4] if len(values)>4 else kw.get('wavelength',1030.),
            family=values[6] if len(values)>6 else kw.get('family',0)))
        b.save(progress,dict(status='running',optical_build_calls=trace_calls))
        return original(*values,**kw)
    b.optical=logged_optical
    started=time.monotonic()
    try:
        model,_,_=b.optical('triangular',plan['design'],32,1.,1030.,plan['period_nm'],0)
        spec,_,p,mesh=b.finite('triangular',model,plan['grid'],8,False)
        if sha(folder/'frozen_census.json')!=request['census_sha256']:raise RuntimeError('Closure census changed after job creation')
        with np.load(folder/plan.get('fields_file','frozen_fields.npz'),allow_pickle=False) as z:field=z[f'pole_{rank}_field'];C=z['coupling_m']
        seedrow=plan['poles'][rank];value=complex(seedrow['delta_per_m'],seedrow['alpha_field_per_m'])
        if np.linalg.norm(C-model.result.coupling_m)>2e-9*max(np.linalg.norm(C),1.):raise RuntimeError('Closure seed C mismatch')
        if relative_residual(p.center_operator(),field,value)>2e-7:raise RuntimeError('Closure seed original-pencil mismatch')
        wave,opt,g,ls,mode,trace=b.close_seed('triangular',plan['design'],plan['period_nm'],model,(value,field),spec,mesh,32,1.,0)
        name=f'TE0_pole{rank:03d}';mode=replace(mode,name=name);row,data=b.mode_row('triangular',opt,g,ls,mode,wave,201)
        axis,fx,fy,widths,qe,qg=data
        row.update(wavelength_nm=wave,closed_TE_family=0,frozen_rank=rank,frequency_trace=trace,frequency_residual_per_m=trace[-1]['frequency_residual_per_m'],group_index=opt.result.group_index,group_velocity_m_s=c/opt.result.group_index,seconds=time.monotonic()-started)
        arrays={name+'_field':mode.fields,name+'_C':opt.result.coupling_m,name+'_far_x':fx,name+'_far_y':fy,
            name+'_radiation_x':mode.radiation_x,name+'_radiation_y':mode.radiation_y,name+'_widths':widths,name+'_qenergy':qe,name+'_qgain':qg,
            name+'_lateral':b.binned_lateral_weights(mode.fields,mode.grid.x_um,300.,31),name+'_Lup':opt.upward_amplitude_loss_m,name+'_Ldown':opt.downward_amplitude_loss_m,
            'angle_deg':axis,'x_um':mode.grid.x_um,'y_um':mode.grid.y_um}
        np.savez_compressed(Path(request['output']).with_suffix('.npz'),**arrays)
        b.save(request['output'],dict(status='completed',row=row,driver_sha256=sha(__file__)))
    except Exception as error:b.save(request['output'],dict(status='unresolved',error=repr(error),seconds=time.monotonic()-started,optical_build_calls=trace_calls,driver_sha256=sha(__file__)))


def bounded_closure(folder,rank,signature):
    work=folder/'closure_queries';work.mkdir(parents=True,exist_ok=True);request=work/f'{rank:03d}.json';output=work/f'{rank:03d}_out.json'
    config=dict(case=str(folder),rank=rank,output=str(output),**signature)
    if request.exists() and read(request)!=config:raise RuntimeError('Closed-root cache source changed')
    b.save(request,config)
    if not output.exists():
        try:
            proc=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__)),'closure_worker','--request',str(request)],capture_output=True,text=True,timeout=240.)
            if not output.exists():b.save(output,dict(status='worker_failed',returncode=proc.returncode,stderr=proc.stderr[-2000:]))
        except subprocess.TimeoutExpired:b.save(output,dict(status='bounded_unresolved',timeout_s=240.,progress_path=str(request.with_name(request.stem+'_progress.json'))))
    result=read(output)
    if result['status']!='completed':raise RuntimeError(f'Bounded actual-frequency closure unresolved: {result}')
    with np.load(output.with_suffix('.npz'),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    return result['row'],arrays


def close(args):
    folder=OUT/args.tag;plan=read(folder/'frozen_census.json');period=plan['period_nm'];design=plan['design']
    if not (folder/plan.get('fields_file','frozen_fields.npz')).exists():
        raise RuntimeError('Intermediate frozen fields were compacted after validation; use a fresh survey tag to recompute')
    model,_,_=b.optical('triangular',design,32,1.,1030.,period,0);spec,_,p,mesh=b.finite('triangular',model,plan['grid'],8,False)
    ranks,clusters=cluster_ranks(plan['poles'],model.result.eigenvalues_m);ranks.update(range(min(args.closed_count,len(plan['poles']))))
    b.save(folder/'closure_plan.json',dict(ranks=sorted(ranks),delta_clusters=clusters,delta_gap_per_m=500.,alpha_limit_multiple=4.,force_Gamma_branches=True))
    path=folder/'closed_optical_modes.json';rows=read(path) if path.exists() else [];out=folder/'closed_fields.npz';arrays={}
    if out.exists():
        with np.load(out,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    failures=read(folder/'closure_failures.json') if (folder/'closure_failures.json').exists() else []
    # Independent fixed-geometry branch continuations run in isolated processes;
    # identity and ranking remain a deterministic serial merge in frozen order.
    # 各根物理求解彼此独立，跨根身份判定仍严格按冻结序号串行合并。
    pool=ThreadPoolExecutor(max_workers=args.parallel_roots)
    completed={r['frozen_rank'] for r in rows}|{r['frozen_rank'] for r in failures}
    signature=dict(census_sha256=sha(folder/'frozen_census.json'),fields_file=plan.get('fields_file','frozen_fields.npz'),fields_sha256=sha(folder/plan.get('fields_file','frozen_fields.npz')))
    futures={rank:pool.submit(bounded_closure,folder,rank,signature) for rank in sorted(ranks) if rank not in completed}
    with np.load(folder/plan.get('fields_file','frozen_fields.npz'),allow_pickle=False) as frozen:
        sequence=sorted(ranks)+[i for i in range(len(plan['poles'])) if i not in ranks]
        for rank in sequence:
            if rank not in ranks and len(rows)>=args.closed_count:break
            if any(row['frozen_rank']==rank for row in rows) or any(x['frozen_rank']==rank for x in failures):continue
            started=time.monotonic();r=plan['poles'][rank];value=complex(r['delta_per_m'],r['alpha_field_per_m']);field=frozen[f'pole_{rank}_field']
            try:
                if rank not in futures:futures[rank]=pool.submit(bounded_closure,folder,rank,signature)
                row,closed_arrays=futures[rank].result();name=row['name'];wave=row['wavelength_nm']
                closed_value=complex(row['delta_per_m'],row['alpha_field_per_m']);closed_field=closed_arrays[name+'_field'];closed_C=closed_arrays[name+'_C']
                peers=[old for old in rows if abs(old['wavelength_nm']-wave)<=1e-7]
                for peer in peers:
                    old_C=arrays[peer['name']+'_C']
                    if np.linalg.norm(old_C-closed_C)>2e-9*max(np.linalg.norm(old_C),1.):
                        raise RuntimeError('Closed identity requires the same TE0 operator at matched frequency')
                if not independent_eigenmode(closed_value,closed_field,[complex(x['delta_per_m'],x['alpha_field_per_m']) for x in peers],[arrays[x['name']+'_field'] for x in peers]):
                    failures.append(dict(frozen_rank=rank,status='duplicate_closed_pole',wavelength_nm=wave))
                    b.save(folder/'closure_failures.json',failures);continue
                rows.append(row);arrays.update(closed_arrays)
                b.save(path,rows);np.savez_compressed(out,**arrays)
                print('CLOSED',args.tag,rank,wave,row['required_material_gain_cm1'],row['beam']['center_to_peak'],flush=True)
            except Exception as error:failures.append(dict(frozen_rank=rank,status='unresolved',error=repr(error)));print('CLOSE_FAILED',args.tag,rank,repr(error),flush=True)
            b.save(folder/'closure_failures.json',failures)
    pool.shutdown(wait=True)
    b.save(folder/'closure_status.json',dict(status='bounded_with_unresolved' if any(r['status']=='unresolved' for r in failures) else 'completed',closed=len(rows),failures=failures,
        scope=f'Lowest{args.closed_count} frozen poles plus qualifying delta-cluster and every Gamma representative; actual frequency closed, not complete fine spectrum.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('freeze_outer','freeze_baseline','survey','worker','closure_worker','fine_stripes','close'))
    p.add_argument('--tag',default='candidate56_N26');p.add_argument('--grid',type=int,default=26);p.add_argument('--previous');p.add_argument('--request',type=Path)
    p.add_argument('--frozen',default='candidate56_frozen');p.add_argument('--closed-count',type=int,default=32)
    p.add_argument('--index',type=int,default=56)
    p.add_argument('--parallel-roots',type=int,default=3)
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.phase not in ('worker','closure_worker'):
        signature=sha(__file__);archive=OUT/'drivers'/f'{Path(__file__).stem}_{signature[:12]}.py';archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():archive.write_bytes(Path(__file__).read_bytes())
        b.save(OUT/f'bounded_runtime_{time.time_ns()}.json',dict(argv=sys.argv,driver_sha256=signature,driver_snapshot=str(archive),source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'src/pcselsim').glob('*.py')}))
    globals()[args.phase](args)


if __name__=='__main__':main()
