"""Post-audit Yb:YAG square-cell re-census and bounded multi-parameter search.

中文：旧正式数据只读；每个粗筛使用完整粗网格广义谱，而不是每Γ族平滑一态。
粗筛候选必须重新接受细网格谱、真实频率闭合、共享反转和光斑验证。
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
from pathlib import Path
from dataclasses import asdict,replace
import argparse,json,hashlib,time,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(Path(__file__).parent)]
import numpy as np
from scipy.stats import qmc
import matched300_yag_search as backend
from pcselsim.finite_discretization import relative_residual
from pcselsim.mode_identity import independent_mode_indices

OUT=ROOT/'results/current/strict_rerun_20261025/YbYAG_square'
OLD=ROOT/'results/current/matched300_20261021/yag/final/square_v2'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):backend.save(p,x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    stamp=time.time_ns()
    save(OUT/f'provenance_{stage}_{stamp}.json',dict(stage=stage,argv=sys.argv,
        script_sha256=sha(__file__),old_summary_sha256=sha(OLD/'summary.json'),
        old_fields_sha256=sha(OLD/'closed_fields.npz'),
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'src/pcselsim').glob('*.py')},
        scope='New actual simulations using audited code. Original data are read only.'))
    (OUT/f'executed_{stage}_{stamp}.py').write_bytes(Path(__file__).read_bytes())


def full_coarse(design,order=8,dz=3.,grid=13):
    started=time.monotonic()
    optical,period,guide=backend.optical('square',design,order,dz)
    fabrication=backend.evaluate_fabrication(backend.geometry('square',design).inclusions,
                                             period,None,backend.RULE)
    if not fabrication['margin_pass']:
        return dict(status='geometry_rejected',lattice='square',design=design,
                    period_nm=period,fabrication=fabrication)
    spec,_,problem,_=backend.finite('square',optical,grid,8,False)
    dim=problem.stiffness.shape[0]
    problem.near(complex(optical.result.eigenvalues_m[0]),8,np.ones(dim,complex),dense_limit=dim+1)
    values,fields=problem._dense_spectrum
    valid=np.flatnonzero(values.imag>=-1e-7)
    order_idx=valid[np.argsort(values[valid].imag)]
    # All valid coarse eigenpairs are retained for rank; duplicate solver
    # returns are absent in one QZ call. Near-collinear distinct poles survive.
    rows=[]
    for i in order_idx[:12]:
        field=fields[:,i].reshape(4,grid,grid)
        mode=backend.as_mode('square',field,complex(values[i]),optical.result,None,f'coarse_{i}')
        mode=replace(mode,eigen_residual=relative_residual(problem.center_operator(),field,values[i]))
        row,_=backend.mode_row('square',optical,guide,spec,mode,1030.,81)
        rows.append(row)
    rows.sort(key=lambda r:r['required_material_gain_cm1'])
    first=rows[0];b=first['beam'];gap=rows[1]['required_material_gain_cm1']/first['required_material_gain_cm1']-1
    beam_ok=(b['center_to_peak']>=.98 and b['ellipticity']<=1.05 and b['radial_anisotropy_l2']<=.10)
    # Scores are only a selection aid; final decisions retain each metric.
    penalty=(30*max(b['ellipticity']-1.05,0)+50*max(.98-b['center_to_peak'],0)
             +30*max(b['radial_anisotropy_l2']-.10,0)+8*max(.10-gap,0))
    score=first['required_material_gain_cm1']*(1+penalty)
    if first['required_material_gain_cm1']>=23.93265:score+=1000
    vertical=backend.solve_guided_te_modes(guide,1030.,dz,16)
    return dict(status='coarse_full_spectrum_screen',lattice='square',design=design,
        period_nm=period,settings=dict(order=order,dz_nm=dz,grid=grid),
        coarse_dimension=dim,complete_coarse_eigenpairs=len(values),modes=rows,
        lowest=first,relative_material_gap=gap,beam_pass=beam_ok,
        isolated_TE_count=len(vertical),fabrication=fabrication,score=score,
        optical_theorem_residual=optical.optical_theorem_residual,
        seconds=time.monotonic()-started,
        scope='Complete coarse discretized spectrum; frozen 1030nm C. Not actual-frequency/fine-spectrum or pump proof.')


def designs(count):
    d=read(OLD/'summary.json')['design']
    yield 'nominal',dict(d)
    variations={'fill':(.275,.285,.30),'main_share':(.50,.55,.60,.65),
        'distance':(.62,.66,.72,.75),'position_angle_deg':(40.,43.,45.,50.,53.),
        'main_aspect':(1.,1.15,1.25,1.4),'satellite_aspect':(1.08,1.18,1.35),
        'main_angle_deg':(-20.,-10.,10.,20.),'satellite_angle_deg':(-45.,0.,45.,90.)}
    for key,values in variations.items():
        for value in values:
            local={**d,key:value,'tag':f'{key}_{value:g}'}
            if key=='satellite_angle_deg':local['satellite_aspect']=1.18
            yield local['tag'],local
    for main,satellite in [('circle','ellipse'),('ellipse','rounded_triangle'),
        ('rounded_triangle','circle'),('superellipse','circle'),('ellipse','superellipse')]:
        local={**d,'main_shape':main,'satellite_shape':satellite,'tag':f'{main}_{satellite}'}
        yield local['tag'],local
    # Joint Latin-hypercube local perturbations supplement the independent axes.
    for i,p in enumerate(qmc.LatinHypercube(9,seed=251030).random(count)):
        local={**d,'fill':.27+.035*p[0],'main_share':.50+.17*p[1],
            'distance':.62+.13*p[2],'position_angle_deg':38+18*p[3],
            'main_aspect':1+.45*p[4],'satellite_aspect':1+.35*p[5],
            'main_angle_deg':-35+70*p[6],'satellite_angle_deg':-90+180*p[7],
            'main_exponent':2.5+1.5*p[8],'tag':f'joint_{i:03d}'}
        if i%7==0:local['satellite_shape']='rounded_triangle'
        elif i%7==1:local['main_shape']='rounded_triangle'
        elif i%7==2:local['main_shape']='superellipse'
        yield local['tag'],local


def focus_designs():
    d=read(OLD/'summary.json')['design']
    for share in (.535,.55,.565,.58):
        for angle in (41.,43.,45.,47.):
            name=f'focus_share{share:g}_position{angle:g}'
            yield name,{**d,'main_share':share,'position_angle_deg':angle,'tag':name}
    for aspect in (1.0,1.05,1.10,1.15):
        for orientation in (-12.,0.,12.):
            name=f'focus_aspect{aspect:g}_orientation{orientation:g}'
            yield name,{**d,'main_share':.55,'position_angle_deg':43.,
                        'main_aspect':aspect,'main_angle_deg':orientation,'tag':name}


def scan(args):
    prefix='focus' if args.stage=='focus' else 'local'
    path=OUT/(prefix+'_screen.jsonl');completed={}
    if path.exists():
        completed={r['id']:r for r in (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())}
    with path.open('a',encoding='utf-8') as stream:
        for name,design in (focus_designs() if args.stage=='focus' else designs(args.count)):
            if name in completed:
                if completed[name]['design']!=design:
                    raise ValueError(f'Resume design mismatch for {name}; do not reuse a Latin-hypercube ID with different parameters')
                continue
            try:row=full_coarse(design,args.order,args.dz,args.grid)
            except Exception as error:row=dict(status='failed',design=design,error=repr(error))
            row['id']=name;stream.write(json.dumps(backend.clean(row))+'\n');stream.flush();completed[name]=row
            print('screen',name,row['status'],row.get('score'),row.get('relative_material_gap'),
                  row.get('lowest',{}).get('required_material_gain_cm1'),flush=True)
    good=[r for r in completed.values() if r.get('beam_pass') and r.get('isolated_TE_count')==1
          and r.get('status')=='coarse_full_spectrum_screen']
    save(OUT/(prefix+'_best.json'),sorted(good,key=lambda r:r['score'])[:12])
    save(OUT/(prefix+'_isolation_pareto.json'),sorted(good,key=lambda r:(-r['relative_material_gap'],r['lowest']['required_material_gain_cm1']))[:12])


def close(args):
    if args.stage=='baseline':
        source=read(OLD/'summary.json');target=OUT/'baseline_full32'
        record=dict(lattice='square',design=source['design'],period_nm=source['period_nm'])
        period=source['period_nm']
    else:
        candidates=read(args.selection_file or OUT/'local_best.json');record=candidates[args.index]
        target=OUT/f'candidate_{args.index:02d}_full32';period=None
    input_path=OUT/(target.name+'_input.json');save(input_path,record)
    backend.command_close(argparse.Namespace(input=input_path,index=0,output=target,
        order=args.order,dz=args.dz,square_grid=args.grid,tri_grid=9,roots=12,
        extra_targets=1,closed_roots=args.closed_modes,_period_override=period))


def controls(args):
    """Track supplied competitors on higher D/N, retaining original raw fields.

    中文：精度控制不替代新网格全谱；每根在新算符重新求本征场并真实闭频。
    """
    folder=Path(args.case)
    summary=read(folder/'summary.json');design=summary['design'];period=summary['period_nm']
    source_rows=read(folder/'closed_optical_modes.json')
    with np.load(folder/'closed_fields.npz',allow_pickle=False) as raw:
        source_arrays={key:raw[key].copy() for key in raw.files}
    challenges=folder/'spectral_challenges'
    if (challenges/'closed_optical_modes.json').exists():
        source_rows+=read(challenges/'closed_optical_modes.json')
        with np.load(challenges/'closed_fields.npz',allow_pickle=False) as raw:
            source_arrays.update({key:raw[key].copy() for key in raw.files if key!='angle_deg'})
    identity_path=folder/'augmented_mode_identity_and_flux_audit.json'
    if identity_path.exists():
        rejected={r['name'] for r in read(identity_path)['rejected_challenge_modes']}
        source_rows=[r for r in source_rows if r['name'] not in rejected]
    source_rows=sorted(source_rows,key=lambda r:r['required_material_gain_cm1'])[:args.control_modes]
    destination=folder/'accuracy_controls';destination.mkdir(parents=True,exist_ok=True)
    references=[]
    for row in source_rows:
        reference,_,_=backend.optical('square',design,summary['settings']['order'],
            summary['settings']['dz_nm'],row['wavelength_nm'],period,row['closed_TE_family'])
        ss,_,_,sm=backend.finite('square',reference,summary['settings']['square_grid'],8,False)
        references.append((row,reference,ss,sm,source_arrays[row['name']+'_field'].copy()))
    tasks=((48,49),(32,65));results=[]
    for order,grid in tasks:
        path=destination/f'D{order}_N{grid}.json'
        if path.exists() and read(path).get('status')=='completed':
            results.append(read(path));continue
        rows=[];arrays={};failures=[]
        for row,reference,ss,sm,field in references:
            started=time.monotonic()
            try:
                optical,_,guide=backend.optical('square',design,order,1.,row['wavelength_nm'],period,row['closed_TE_family'])
                spec,_,problem,mesh=backend.finite('square',optical,grid,8,False)
                field=backend.interpolate_envelope('square',field,ss,sm,spec,mesh)
                flat=field.reshape(4,-1)
                change=optical.result.coupling_m-reference.result.coupling_m
                predictor=complex(row['delta_per_m'],row['alpha_field_per_m'])+np.einsum('ik,ij,jk->',flat.conj(),change,flat)/np.sum(abs(flat)**2)
                value,field,initialization=backend.follow_reference_pole(problem,field,predictor)
                wave,model,g,local_spec,mode,trace=backend.close_seed('square',design,period,
                    optical,(value,field),spec,mesh,order,1.,row['closed_TE_family'])
                mode=replace(mode,name=row['name'])
                record,data=backend.mode_row('square',model,g,local_spec,mode,wave,201)
                axis,fx,fy,widths,qe,qg=data
                record.update(wavelength_nm=wave,closed_TE_family=row['closed_TE_family'],
                    group_index=model.result.group_index,group_velocity_m_s=backend.c/model.result.group_index,
                    frequency_residual_per_m=trace[-1]['frequency_residual_per_m'],
                    frequency_trace=trace,initialization=initialization,seconds=time.monotonic()-started)
                rows.append(record);name=mode.name
                arrays.update({name+'_field':mode.fields,name+'_far_x':fx,name+'_far_y':fy,
                    name+'_radiation_x':mode.radiation_x,name+'_radiation_y':mode.radiation_y,
                    name+'_C':model.result.coupling_m,name+'_Lup':model.upward_amplitude_loss_m,
                    name+'_Ldown':model.downward_amplitude_loss_m,name+'_qgain':qg,
                    name+'_qenergy':qe,name+'_widths':widths,'angle_deg':axis})
                arrays[name+'_lateral']=backend.lateral_energy_weights(mode.fields,31)
                print('control',order,grid,name,record['required_material_gain_cm1'],flush=True)
            except Exception as error:failures.append(dict(name=row['name'],error=repr(error)))
            save(path,dict(status='running',order=order,grid=grid,closed_modes=rows,failures=failures))
        result=dict(status='completed' if not failures else 'partial',order=order,grid=grid,dz_nm=1.,
            fixed_period_nm=period,closed_modes=rows,failures=failures,
            scope='Actual-frequency closure of supplied low-threshold branches only; not a new complete fine spectrum.')
        np.savez_compressed(path.with_suffix('.npz'),**arrays);save(path,result);results.append(result)
    save(destination/'summary.json',dict(source_case=str(folder),results=results))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('baseline','scan','focus','final','controls'))
    p.add_argument('--count',type=int,default=40);p.add_argument('--order',type=int)
    p.add_argument('--dz',type=float);p.add_argument('--grid',type=int)
    p.add_argument('--index',type=int,default=0);p.add_argument('--closed-modes',type=int,default=32)
    p.add_argument('--selection-file',type=Path)
    p.add_argument('--case',type=Path);p.add_argument('--control-modes',type=int,default=4)
    a=p.parse_args()
    defaults=(8,3.,13) if a.stage in ('scan','focus') else (32,1.,49)
    if a.order is None:a.order=defaults[0]
    if a.dz is None:a.dz=defaults[1]
    if a.grid is None:a.grid=defaults[2]
    provenance(a.stage)
    if a.stage in ('scan','focus'):scan(a)
    elif a.stage=='controls':controls(a)
    else:close(a)


if __name__=='__main__':main()
