"""Nonaligned-grid challenge of two saved YAG triangular finite modes.

Keep each own D32 C and wavelength fixed. Interpolation is only the initial
vector; accepted fields solve the new grid's unchanged K/M pencil. This is a
frozen-C mesh diagnostic, not frequency closure, full-spectrum search or beam
robustness certification / 只改变有限网格，不改变原C，不将插值当作最终解。
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
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse.linalg import splu
from pcselsim.triangular_finite import TriangularFiniteSpec,build_device_grid,triangular_finite_eigenproblem


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solve_follow(problem,initial,target):
    """Physical-center inverse iteration and strict original-pencil residual."""
    q=initial.ravel()/np.linalg.norm(initial);seed=q.copy();shift=complex(target)+.5+.7j
    mass=splu(problem.mass.tocsc());trace=[]
    for iteration in range(1,17):
        factor=splu((problem.stiffness-shift*problem.mass).tocsc())
        links=factor.solve(q);del factor
        field=problem.mass@links;field/=np.linalg.norm(field)
        new_links=mass.solve(field);applied=problem.stiffness@new_links
        value=complex(np.vdot(field,applied));rhs=value*(problem.mass@new_links)
        residual=float(np.linalg.norm(applied-rhs)/max(np.linalg.norm(applied),np.linalg.norm(rhs),1e-30))
        overlap=float(abs(np.vdot(seed,field))**2)
        trace.append({'iteration':iteration,'delta_per_m':value.real,'alpha_field_per_m':value.imag,'original_K_M_relative_residual':residual,'interpolated_seed_field_overlap_squared':overlap})
        if residual<1e-10 and overlap>.7:
            if value.imag<0:raise RuntimeError('A passive frozen-C mode has negative loss')
            return value,field.reshape(initial.shape),trace
        q=field;shift=value+1e-8+2e-8j
    raise RuntimeError('No strict residual/field-overlap accepted inverse-iteration root in 16 iterations')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grids',nargs='+',type=int,default=[53,64])
    p.add_argument('--output',type=Path,default=ROOT/'results/current/physics_audit_20261025/triangular_nonaligned_grid_challenge')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'results/current/matched300_20261021/comparison/selected_cases.json'
    case=next(x for x in json.loads(manifest.read_text())['cases'] if x['id']=='YbYAG_triangular')
    directory=(ROOT/case['summary_path']).parent
    rows=sorted(json.loads((directory/'closed_optical_modes.json').read_text()),key=lambda r:r['required_material_gain_cm1'])[:2]
    source=directory/'closed_fields.npz';sources={str(f.relative_to(ROOT)):digest(f) for f in [manifest,directory/'closed_optical_modes.json',source]}
    old=build_device_grid(case['grid'],150.,'square');input_points=np.column_stack((old.x_um,old.y_um))
    results=[];stored={};started=time.monotonic()
    with np.load(source,allow_pickle=False) as raw:
        seeds={r['name']:(raw[r['name']+'_field'].copy(),raw[r['name']+'_C'].copy(),raw[r['name']+'_qgain'].copy()) for r in rows}
    for grid in args.grids:
        spec=TriangularFiniteSpec(150.,grid,0.,8,'square','staggered')
        for row in rows:
            name=row['name'];field,coupling,qg=seeds[name]
            print('GRID_CHALLENGE_START',grid,name,flush=True);t=time.monotonic()
            try:
                problem,mesh=triangular_finite_eigenproblem(SimpleNamespace(coupling_m=coupling),spec)
                query=np.column_stack((mesh.x_um,mesh.y_um))
                initial=LinearNDInterpolator(input_points,field.T,fill_value=0.)(query).T
                value,new_field,trace=solve_follow(problem,initial,complex(row['delta_per_m'],row['alpha_field_per_m']))
                material=(2*value.imag+row['internal_power_loss_per_m'])/(100*qg.sum())
                local=problem.loss_budget(new_field)
                record={'status':'completed','name':name,'N':grid,'nodes':mesh.points,'unknowns':problem.stiffness.shape[0],
                    'reference_N':case['grid'],'frozen_reference_wavelength_nm':row['wavelength_nm'],'frozen_C_D':case['D'],
                    'frozen_delta_per_m':value.real,'frozen_alpha_field_per_m':value.imag,
                    'frozen_C_material_gain_estimate_cm1':float(material),'reference_material_gain_cm1':row['required_material_gain_cm1'],
                    'gain_relative_difference':float(material/row['required_material_gain_cm1']-1),
                    'original_K_M_relative_residual':trace[-1]['original_K_M_relative_residual'],
                    'interpolated_seed_field_overlap_squared':trace[-1]['interpolated_seed_field_overlap_squared'],
                    'frequency_equation_residual_at_frozen_reference_per_m':float(row['delta_per_m']+row['frequency_residual_per_m']-value.real),
                    'amplitude_loss_budget_closure_per_m':float(value.imag-local['local_amplitude_loss_per_m']-local['transport_amplitude_loss_per_m']),
                    'iterations':len(trace),'trace':trace,'seconds':time.monotonic()-t}
                stored[f'N{grid}_{name}_field']=new_field
                stored[f'N{grid}_{name}_C']=coupling
            except Exception as error:
                record={'status':'failed','name':name,'N':grid,'error':repr(error),'seconds':time.monotonic()-t}
            results.append(record)
            report={'status':'completed_frozen_C_diagnostic' if all(r['status']=='completed' for r in results) else 'contains_failed_challenges',
                'scope':'Fixed own saved D32 C and wavelength per root; actual new finite-grid poles. Not actual-frequency closure, not a complete mode census, not beam robustness or experimental validation.',
                'source_sha256':sources,'audit_source_sha256':{str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),ROOT/'src/pcselsim/triangular_finite.py',ROOT/'src/pcselsim/finite_discretization.py']},'results':results,'elapsed_seconds':time.monotonic()-started}
            (args.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            np.savez_compressed(args.output/'diagnostic_fields.npz',**stored)
            print('GRID_CHALLENGE_DONE',json.dumps(record),flush=True)
    changed=[s for s,d in sources.items() if digest(ROOT/s)!=d]
    if changed:raise RuntimeError('Frozen source inputs changed: '+repr(changed))


if __name__=='__main__':main()
