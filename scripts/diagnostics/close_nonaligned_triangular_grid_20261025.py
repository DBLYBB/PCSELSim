"""Close two N64 diagnostic branches without redesigning the YAG structure.

Uses the actual N64 frozen-C eigenfields as seeds, rebuilds D32/dz1 optical
models at each frequency, and checks the original new-grid K/M pencil.
中文：只跟踪两条已知分支，不宣称完整N64谱、全波或圆斑鲁棒性。
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/optimization')]
import numpy as np
from scipy.sparse.linalg import splu
import matched300_yag_search as backend
from pcselsim.triangular_finite import TriangularFiniteSpec,triangular_finite_eigenproblem


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=165.)
    p.add_argument('--directory',type=Path,default=ROOT/'results/current/physics_audit_20261025/triangular_nonaligned_grid_challenge')
    args=p.parse_args();started=time.monotonic();deadline=started+args.seconds
    frozen_path=args.directory/'summary.json';frozen=json.loads(frozen_path.read_text())
    field_path=args.directory/'diagnostic_fields.npz'
    manifest=ROOT/'results/current/matched300_20261021/comparison/selected_cases.json'
    case=next(c for c in json.loads(manifest.read_text())['cases'] if c['id']=='YbYAG_triangular')
    source_path=ROOT/case['summary_path'];source=json.loads(source_path.read_text())
    mode_path=source_path.parent/'closed_optical_modes.json'
    old_modes={r['name']:r for r in json.loads(mode_path.read_text())}
    names=['TE0_pole00','TE0_pole01'];results=[];stored={}
    inputs={str(p.relative_to(ROOT)):digest(p) for p in [frozen_path,field_path,manifest,source_path,mode_path]}
    original_optical=backend.optical
    def bounded_optical(*pos,**kw):
        if time.monotonic()>deadline:raise TimeoutError('Nonaligned-grid closure reached the declared wall-clock budget')
        return original_optical(*pos,**kw)
    backend.optical=bounded_optical
    try:
        with np.load(field_path,allow_pickle=False) as data:
            for name in names:
                trial=next(r for r in frozen['results'] if r['N']==64 and r['name']==name and r['status']=='completed')
                print('N64_CLOSURE_START',name,flush=True);t=time.monotonic()
                try:
                    old=old_modes[name];field=data['N64_'+name+'_field'];frozen_c=data['N64_'+name+'_C']
                    reference,_,_=bounded_optical('triangular',source['design'],32,1.,old['wavelength_nm'],source['period_nm'])
                    c_difference=float(np.linalg.norm(reference.result.coupling_m-frozen_c)/np.linalg.norm(frozen_c))
                    if c_difference>2e-10:raise RuntimeError('Saved D32 optical model no longer matches the reference')
                    spec=TriangularFiniteSpec(150.,64,0.,8,'square','staggered')
                    _,mesh=triangular_finite_eigenproblem(reference.result,spec)
                    seed=(complex(trial['frozen_delta_per_m'],trial['frozen_alpha_field_per_m']),field)
                    wl,model,guide,local_spec,mode,trace=backend.close_seed('triangular',source['design'],source['period_nm'],reference,seed,spec,mesh,32,1.)
                    problem,_=triangular_finite_eigenproblem(model.result,local_spec)
                    value=complex(mode.delta_per_m,mode.alpha_per_m)
                    links=splu(problem.mass.tocsc()).solve(mode.fields.ravel())
                    lhs=problem.stiffness@links;rhs=value*(problem.mass@links)
                    pencil=float(np.linalg.norm(lhs-rhs)/max(np.linalg.norm(lhs),np.linalg.norm(rhs),1e-30))
                    widths,qe,qg=backend.active_weights(model.result,guide,24)
                    gain=(2*value.imag+10)/(100*qg.sum())
                    beta=2*np.pi*model.result.effective_index/(wl*1e-9);beta0=4*np.pi/(np.sqrt(3)*source['period_nm']*1e-9)
                    frequency=float((beta*beta-beta0*beta0)/(2*beta0)-value.real)
                    if pencil>2e-7 or abs(frequency)>.05:raise RuntimeError('Independent final K/M or frequency residual failed')
                    record={'status':'completed','name':name,'N':64,'side_um':300.,'D':32,'dz_nm':1.,'period_nm':source['period_nm'],
                        'wavelength_nm':wl,'delta_per_m':value.real,'alpha_field_per_m':value.imag,
                        'material_gain_cm1':float(gain),'reference_N52_material_gain_cm1':old['required_material_gain_cm1'],
                        'gain_relative_to_N52':float(gain/old['required_material_gain_cm1']-1),
                        'original_K_M_relative_residual':pencil,'frequency_residual_per_m':frequency,
                        'source_frozen_C_relative_difference':c_difference,'gain_weight':float(qg.sum()),
                        'seconds':time.monotonic()-t,'frequency_trace':trace}
                    stored[name+'_field']=mode.fields;stored[name+'_C']=model.result.coupling_m
                    stored[name+'_qgain']=qg;stored[name+'_qenergy']=qe;stored[name+'_widths']=widths
                except Exception as error:
                    record={'status':'failed','name':name,'N':64,'error':repr(error),'seconds':time.monotonic()-t}
                results.append(record)
                summary={'status':'completed_two_tracked_frequency_closed_branches' if len(results)==2 and all(r['status']=='completed' for r in results) else 'partial_two_branch_closure',
                    'scope':'Actual fixed-period D32/dz1 N64 roots, only two branches seeded from the N64 frozen-C diagnostic. Not complete N64 spectrum, beam robustness, full-wave or experimental certification.',
                    'declared_wall_budget_seconds':args.seconds,'results':results,'source_sha256':inputs,
                    'script_sha256':digest(__file__),'elapsed_seconds':time.monotonic()-started}
                if len(results)==2 and all(r['status']=='completed' for r in results):
                    summary['tracked_two_mode_material_gap']=results[1]['material_gain_cm1']/results[0]['material_gain_cm1']-1
                (args.directory/'summary_closed.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
                np.savez_compressed(args.directory/'diagnostic_closed_fields.npz',**stored)
                print('N64_CLOSURE_DONE',json.dumps({k:v for k,v in record.items() if k!='frequency_trace'}),flush=True)
                if time.monotonic()>deadline:break
    finally:backend.optical=original_optical
    if any(digest(ROOT/p)!=d for p,d in inputs.items()):raise RuntimeError('Frozen inputs changed during closure')


if __name__=='__main__':main()
