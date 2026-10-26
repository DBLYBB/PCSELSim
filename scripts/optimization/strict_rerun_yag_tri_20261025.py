"""Post-audit triangular Yb:YAG local search, with two actual branch closures.

中文：局扫只跟踪已知亮/暗两支，不能据此宣称完整模式隔离。入选点必须再跑
完整粗谱引导、细网格挑战和至少32根闭频，且检查非同对齐网格的实际复场。
All physical periods and dimensions remain fixed within each frequency solve.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(Path(__file__).parent)]
import numpy as np
from scipy.constants import c
from scipy.stats import qmc
import matched300_yag_search as b
from pcselsim.matched300_yag import binned_lateral_weights
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig
from pcselsim.ybyag_spatial_rates import SpatialYbYAGModel
from pcselsim.ybyag_threshold import shared_inversion_pump_threshold

OUT=ROOT/'results/current/strict_rerun_20261025/YbYAG_triangular'
SOURCE=ROOT/'results/current/matched300_20261021/yag/final/triangular_bright_round_v5'


def proposals(d):
    yield 'nominal',dict(d)
    groups=[('fill',[(-.004,),(-.002,),(.002,),(.004,)]),
            ('main_share',[(-.03,),(-.015,),(.015,),(.03,)]),
            ('distance_both',[(-.01,),(-.005,),(.005,),(.01,)]),
            ('angle_both',[(-1.0,),(-.5,),(.5,),(1.0,)]),
            ('orientation',[(-4.,),(-2.,),(2.,),(4.,)]),
            ('rounding',[(.08,),(.10,),(.15,),(.18,)])]
    for kind,entries in groups:
        for (step,) in entries:
            target=dict(d)
            if kind=='distance_both':target.update(distance1=d['distance1']+step,distance2=d['distance2']+step)
            elif kind=='angle_both':target.update(angle1_deg=d['angle1_deg']-step,angle2_deg=d['angle2_deg']+step)
            elif kind=='orientation':target['main_orientation_deg']=d['main_orientation_deg']+step
            elif kind=='rounding':target['rounding']=step
            else:target[kind]=d[kind]+step
            yield f'{kind}_{step:+g}',target
    for key,steps in [('distance1',[-.008,.008]),('angle1_deg',[-1.,1.]),
                      ('angle2_deg',[-1.,1.]),('satellite_balance',[.48,.52])]:
        for step in steps:
            target=dict(d);target[key]=step if key=='satellite_balance' else d[key]+step
            yield f'asymmetric_{key}_{step:g}',target
    for aspect in (1.03,1.08,1.15,1.25):
        yield f'area_preserving_aspect_{aspect:g}',dict(d,main_aspect=aspect,_yag_triangle_aspect_enabled=True)
    keys=('fill','main_share','distance1','distance2','angle1_deg','angle2_deg','main_orientation_deg','rounding')
    scales=np.array([.004,.025,.008,.008,1.,1.,4.,.04])
    for i,unit in enumerate(qmc.LatinHypercube(8,seed=251030).random(24)):
        changes=(2*unit-1)*scales
        target={**d,**{k:float(d[k]+delta) for k,delta in zip(keys,changes)}}
        target['rounding']=float(np.clip(target['rounding'],.08,.20))
        if i%4==0:target.update(_yag_triangle_aspect_enabled=True,main_aspect=float(1.+.20*unit[0]))
        yield f'joint_{i:02d}',target


def geometric_continuation(base,target,period,reference,spec,mesh,seed,order,dz):
    """Continuous geometry changes follow fields; final values solve original K/M."""
    start=dict(base)
    if target.get('_yag_triangle_aspect_enabled',False) and not base.get('_yag_triangle_aspect_enabled',False):
        start['main_aspect']=1.0  # old aspect label was inactive for equilateral polygons
    keys=[k for k,v in target.items() if isinstance(v,(int,float)) and not isinstance(v,bool)
          and k in start and float(v)!=float(start[k])]
    value,field=seed;old=reference;fraction=0.;step=1.;trace=[]
    while fraction<1.-1e-12:
        trial=min(1.,fraction+step)
        design={**target,**{k:float(start[k]+trial*(target[k]-start[k])) for k in keys}}
        model,_,_=b.optical('triangular',design,order,dz,reference.result.bragg_wavelength_nm,period)
        _,_,problem,_=b.finite('triangular',model,spec.radius_cells,8,False)
        flat=field.reshape(6,-1)
        predictor=value+np.einsum('ik,ij,jk->',flat.conj(),model.result.coupling_m-old.result.coupling_m,flat)/np.sum(abs(flat)**2)
        try:
            new_value,new_field,audit=b.follow_reference_pole(problem,field,predictor)
            if audit['interpolated_field_overlap']<.75:raise RuntimeError('Geometry branch overlap below0.75')
        except Exception:
            step/=2
            if step<1/32:raise RuntimeError('Bounded geometry continuation lost reference branch')
            continue
        value,field,old=new_value,new_field,model;fraction=trial
        trace.append(dict(fraction=fraction,**audit));step=min(2*step,1.-fraction)
    return old,(value,field),trace


def screen(args):
    summary=json.loads((SOURCE/'summary.json').read_text(encoding='utf-8'))
    rows=json.loads((SOURCE/'closed_optical_modes.json').read_text(encoding='utf-8'))
    d,period=summary['design'],summary['period_nm']
    data=np.load(SOURCE/'closed_fields.npz',allow_pickle=False)
    references=[]
    for row in rows[:2]:
        old,_,_=b.optical('triangular',d,32,1.,row['wavelength_nm'],period)
        old_spec,_,_,old_mesh=b.finite('triangular',old,52,8,False)
        model,_,_=b.optical('triangular',d,args.order,args.dz,row['wavelength_nm'],period)
        spec,_,problem,mesh=b.finite('triangular',model,args.grid,8,False)
        field=b.interpolate_envelope('triangular',data[row['name']+'_field'],old_spec,old_mesh,spec,mesh)
        flat=field.reshape(6,-1)
        predictor=complex(row['delta_per_m'],row['alpha_field_per_m'])+np.einsum('ik,ij,jk->',flat.conj(),model.result.coupling_m-old.result.coupling_m,flat)/np.sum(abs(flat)**2)
        value,field,init=b.follow_reference_pole(problem,field,predictor)
        references.append((row,model,spec,mesh,(value,field),init))
    output=OUT/'local_screen';output.mkdir(parents=True,exist_ok=True)
    saved=output/'records.json';records=json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else []
    best=min((r['score'] for r in records if r.get('joint_target_pass')),default=np.inf)
    plan=list(proposals(d))
    b.save(output/'plan.json',dict(count=len(plan),base_design=d,period_nm=period,order=args.order,
        dz_nm=args.dz,grid=args.grid,samples=81,scope='Two known independent branches only; screening cannot certify the actual global first mode.'))
    for index,(label,design) in enumerate(plan):
        if any(r['index']==index for r in records):continue
        started=time.monotonic();fab=b.evaluate_fabrication(b.geometry('triangular',design).inclusions,period,b.TRIANGULAR_DIRECT,b.RULE)
        record=dict(index=index,label=label,lattice='triangular',design=design,period_nm=period,
                    settings=dict(order=args.order,dz_nm=args.dz,grid=args.grid),fabrication=fab)
        models=[];closed=[];arrays={};identities=[];stored=[]
        if not fab['margin_pass']:record['status']='geometry_rejected'
        else:
            try:
                for row,reference,spec,mesh,seed,init in references:
                    model,continued,geom=geometric_continuation(d,design,period,reference,spec,mesh,seed,args.order,args.dz)
                    wl,optical,guide,local_spec,mode,trace=b.close_seed('triangular',design,period,model,continued,spec,mesh,args.order,args.dz)
                    mode=replace(mode,name=row['name'])
                    out,parts=b.mode_row('triangular',optical,guide,local_spec,mode,wl,81)
                    axis,fx,fy,widths,qe,qg=parts
                    out.update(wavelength_nm=wl,frequency_residual_per_m=trace[-1]['frequency_residual_per_m'],
                               geometry_trace=geom,frequency_trace=trace,closed_TE_family=0)
                    closed.append(out);stored.append(mode.fields)
                    lateral=binned_lateral_weights(mode.fields,mode.grid.x_um,300.,31)
                    models.append(SpatialYbYAGModel(b.material(wl),widths,qe,out['total_power_loss_per_m'],
                        out['up_power_loss_per_m'],out['down_power_loss_per_m'],c/optical.result.group_index,qg,lateral))
                    arrays.update({mode.name+'_field':mode.fields,mode.name+'_C':optical.result.coupling_m,
                        mode.name+'_far_x':fx,mode.name+'_far_y':fy,mode.name+'_qgain':qg,
                        mode.name+'_qenergy':qe,mode.name+'_widths':widths,mode.name+'_lateral':lateral})
                separate=True
                if abs(closed[0]['wavelength_nm']-closed[1]['wavelength_nm'])<=1e-7:
                    c0,c1=[arrays[row['name']+'_C'] for row in closed]
                    if np.linalg.norm(c0-c1)>1e-8*max(np.linalg.norm(c0),1.):
                        raise RuntimeError('Same-frequency identity requires the same optical operator')
                    separate=independent_eigenmode(complex(closed[1]['delta_per_m'],closed[1]['alpha_field_per_m']),stored[1],
                        [complex(closed[0]['delta_per_m'],closed[0]['alpha_field_per_m'])],[stored[0]])
                if not separate:record.update(status='tracked_branches_merged',closed_modes=closed)
                else:
                    system=SpatialMultimodeYbYAG(tuple(models),tuple(row['name'] for row in closed))
                    th=shared_inversion_pump_threshold(system,PumpGeometryConfig('edge_dual',0.,300.,31))
                    if len(th.first_modes)!=1:raise RuntimeError('No unique reachable two-branch winner')
                    first=next(r for r in closed if r['name']==th.first_modes[0]);other=next(r for r in closed if r['name']!=th.first_modes[0])
                    beam=first['beam'];gain=first['required_material_gain_cm1'];gap=other['required_material_gain_cm1']/gain-1
                    accepted=bool(beam['center_to_peak']>=.98 and beam['ellipticity']<=1.05 and beam['radial_anisotropy_l2']<=.10 and gap>=.10)
                    score=float(gain*(1+12*max(beam['ellipticity']-1.05,0)+20*max(.98-beam['center_to_peak'],0)+10*max(beam['radial_anisotropy_l2']-.10,0)+8*max(.10-gap,0)))
                    record.update(status='two_branch_screen_completed',lowest=first,second=other,closed_modes=closed,
                        relative_material_gap=gap,edge_two_branch_threshold=asdict(th),joint_target_pass=accepted,score=score,
                        wavelength_nm=first['wavelength_nm'])
                    if accepted and score<best:
                        best=score;arrays.update(angle_deg=axis,x_um=mesh.x_um,y_um=mesh.y_um)
                        np.savez_compressed(output/'best_fields.npz',**arrays)
                        b.save(output/'best_record.json',record)
            except Exception as error:record.update(status='failed',error=repr(error),closed_modes=closed)
        record['seconds']=time.monotonic()-started;records.append(record);b.save(saved,records)
        print('LOCAL',index,label,record['status'],record.get('lowest',{}).get('required_material_gain_cm1'),record.get('relative_material_gap'),record['seconds'],flush=True)
    b.save(output/'ranked.json',sorted([r for r in records if r.get('joint_target_pass')],key=lambda r:r['score']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--order',type=int,default=16)
    p.add_argument('--dz',type=float,default=2.);p.add_argument('--grid',type=int,default=26)
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'local_executed_script.py').write_bytes(Path(__file__).read_bytes())
    b.save(OUT/'local_runtime.json',dict(argv=sys.argv,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/pcselsim').glob('*.py')}))
    screen(args)


if __name__=='__main__':main()
