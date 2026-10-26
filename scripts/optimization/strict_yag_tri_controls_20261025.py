"""Fixed-geometry actual-field N/D controls for triangular Yb:YAG.

中文：输入局扫中两支实际复场，仅作为新算符初值。所有控制都重新求原K/M、
固定同一300um边界与物理周期并闭合实际频率。此控制不宣称完成新网格全谱。
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
import argparse,hashlib,json,sys,time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
import numpy as np
import matched300_yag_search as b
from pcselsim.finite_discretization import relative_residual
from pcselsim.mode_identity import independent_eigenmode

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--fields',type=Path,required=True)
    p.add_argument('--period',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=json.loads(a.input.read_text(encoding='utf8'));d=r['design']
    period=json.loads(a.period.read_text(encoding='utf8'))['frozen_period_nm']
    a.output.mkdir(parents=True,exist_ok=True);reference=[]
    with np.load(a.fields,allow_pickle=False) as raw:
        for row in r['closed_modes']:
            if row.get('closed_TE_family')!=0:raise RuntimeError('The two supplied branches must both be TE family0')
            model,_,_=b.optical('triangular',d,r['settings']['order'],r['settings']['dz_nm'],row['wavelength_nm'],r['period_nm'])
            spec,_,problem,mesh=b.finite('triangular',model,r['settings']['grid'],8,False)
            field=raw[row['name']+'_field'];stored=raw[row['name']+'_C']
            if np.linalg.norm(model.result.coupling_m-stored)>2e-9*max(np.linalg.norm(stored),1.):raise RuntimeError('Source coupling mismatch')
            residual=relative_residual(problem.center_operator(),field,complex(row['delta_per_m'],row['alpha_field_per_m']))
            if residual>2e-7:raise RuntimeError('Source original-pencil mismatch')
            reference.append((row,model,spec,mesh,field.copy(),residual))
    results=[]
    for order,grid,dz in ((32,52,1.),(32,64,1.),(48,52,1.),(48,64,1.),(32,64,.5)):
        path=a.output/f'D{order}_N{grid}_dz{dz:g}.json'
        if path.exists() and json.loads(path.read_text(encoding='utf8')).get('status')=='completed':
            results.append(json.loads(path.read_text(encoding='utf8')));continue
        rows=[];arrays={};failures=[]
        for row,old,source,source_mesh,initial,source_residual in reference:
            start=time.monotonic()
            try:
                model,_,guide=b.optical('triangular',d,order,dz,row['wavelength_nm'],period)
                spec,_,problem,mesh=b.finite('triangular',model,grid,8,False)
                field=b.interpolate_envelope('triangular',initial,source,source_mesh,spec,mesh)
                flat=field.reshape(6,-1);change=model.result.coupling_m-old.result.coupling_m
                predictor=complex(row['delta_per_m'],row['alpha_field_per_m'])+np.einsum('ik,ij,jk->',flat.conj(),change,flat)/np.sum(abs(flat)**2)
                value,field,initialization=b.follow_reference_pole(problem,field,predictor)
                wave,opt,g,local_spec,mode,trace=b.close_seed('triangular',d,period,model,(value,field),spec,mesh,order,dz)
                mode=replace(mode,name=row['name']);item,data=b.mode_row('triangular',opt,g,local_spec,mode,wave,201)
                axis,fx,fy,widths,qe,qg=data
                item.update(wavelength_nm=wave,closed_TE_family=0,frequency_residual_per_m=trace[-1]['frequency_residual_per_m'],
                    frequency_trace=trace,initialization=initialization,source_original_pencil_residual=source_residual,seconds=time.monotonic()-start)
                rows.append(item);name=mode.name
                arrays.update({name+'_field':mode.fields,name+'_C':opt.result.coupling_m,name+'_far_x':fx,name+'_far_y':fy,
                    name+'_radiation_x':mode.radiation_x,name+'_radiation_y':mode.radiation_y,
                    name+'_widths':widths,name+'_qenergy':qe,name+'_qgain':qg,'angle_deg':axis,
                    'x_um':mode.grid.x_um,'y_um':mode.grid.y_um})
                print('CONTROL',order,grid,dz,name,item['required_material_gain_cm1'],item['beam']['ellipticity'],flush=True)
            except Exception as e:failures.append(dict(name=row['name'],error=repr(e)))
            b.save(path,dict(status='running',D=order,N=grid,dz_nm=dz,period_nm=period,closed_modes=rows,failures=failures))
        repeated=[];independent=[]
        for item in rows:
            peers=[prior for prior in independent if prior['closed_TE_family']==item['closed_TE_family'] and
                   abs(prior['wavelength_nm']-item['wavelength_nm'])<=1e-7]
            for prior in peers:
                c0,c1=arrays[prior['name']+'_C'],arrays[item['name']+'_C']
                if np.linalg.norm(c0-c1)>2e-9*max(np.linalg.norm(c0),1.):
                    raise RuntimeError('Same-frequency mode identity requires matching TE-family C operators')
            value=complex(item['delta_per_m'],item['alpha_field_per_m'])
            if independent_eigenmode(value,arrays[item['name']+'_field'],
                [complex(prior['delta_per_m'],prior['alpha_field_per_m']) for prior in peers],
                [arrays[prior['name']+'_field'] for prior in peers]):independent.append(item)
            else:repeated.append(dict(name=item['name'],compared_with=[prior['name'] for prior in peers]))
        gap=None
        if len(independent)>=2:
            ranked=sorted(independent,key=lambda item:item['required_material_gain_cm1'])
            gap=ranked[1]['required_material_gain_cm1']/ranked[0]['required_material_gain_cm1']-1
        status='partial' if failures else ('branch_merged' if repeated else 'completed')
        result=dict(status=status,D=order,N=grid,dz_nm=dz,
            nominal_aperture_um=[300,300],period_nm=period,closed_modes=rows,failures=failures,
            independent_closed_modes=[item['name'] for item in independent],repeated_branch_records=repeated,
            independent_branch_count=len(independent),two_supplied_branch_relative_material_gap=gap,
            scope='Two supplied branches, each solved on the original new pencil at its own closed frequency. Not a complete new-grid spectrum.')
        np.savez_compressed(path.with_suffix('.npz'),**arrays);b.save(path,result);results.append(result)
    b.save(a.output/'summary.json',dict(results=results,input=str(a.input),fields=str(a.fields),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.input,a.fields,a.period,Path(__file__))}))

if __name__=='__main__':main()
