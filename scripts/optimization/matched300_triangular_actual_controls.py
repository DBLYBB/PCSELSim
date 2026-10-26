"""Follow saved finite poles at the actual closed reference for numerical controls.

These are fixed-reference branch controls, not an exhaustive competing-pole proof.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[_key]="1"
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.sparse.linalg import splu
from pcselsim.design_objectives import beam_metrics
from pcselsim.fabrication import TRIANGULAR_DIRECT,perturb_triangular_inclusions,evaluate_fabrication
from pcselsim.triangular_finite import TriangularFiniteSpec,TriangularFiniteMode,triangular_finite_eigenproblem,triangular_vector_far_field
from scripts.optimization.matched300_triangular_qz_guided import model_for,scalar_record
from scripts.optimization.matched300_triangular_search import ThreeHole,candidate,RULE
from pcselsim.finite_discretization import relative_residual
from scripts.optimization.triangular_semiconductor_search import json_write,write_csv,threadpool_limits


def separable_far_field(mode,wavelength_nm,view_deg=1.,samples=201):
    """Identical point sum, separated on the exact unique Cartesian coordinates.

    No coordinate rounding, spatial interpolation, FFT, or angular approximation
    is introduced.  The sum order alone differs from the direct shared routine.
    """
    if samples%2==0:samples+=1
    safe=float(np.rad2deg(np.arctan(2*wavelength_nm*1e-9/(3*mode.grid.spacing_m*(1+1/np.sqrt(3))))))
    evaluated=min(view_deg,.98*safe);axis=np.linspace(-evaluated,evaluated,samples)
    theta=np.deg2rad(axis);wave=2*np.pi/(wavelength_nm*1e-9)*np.tan(theta)
    x,ix=np.unique(mode.grid.x_um*1e-6,return_inverse=True)
    y,iy=np.unique(mode.grid.y_um*1e-6,return_inverse=True)
    ex=np.exp(-1j*wave[:,None]*x[None,:]);ey=np.exp(-1j*wave[:,None]*y[None,:])
    result=[]
    for radiation in (mode.radiation_x,mode.radiation_y):
        spatial=np.zeros((len(y),len(x)),complex);np.add.at(spatial,(iy,ix),radiation)
        result.append((ey@spatial)@ex.T)
    obliquity=np.cos(theta)[:,None]+np.cos(theta)[None,:]-1
    return SimpleNamespace(angle_deg=axis,field_x=result[0]*obliquity,field_y=result[1]*obliquity,
        alias_free_square_view_deg=safe,spatial_unique_coordinates=(len(x),len(y)))


def follow(problem,target,seed,count=24):
    """Physical-center inverse iteration, strictly checked; ARPACK fallback."""
    seed=np.asarray(seed).ravel();seed=seed/np.linalg.norm(seed)
    operator=problem.center_operator();field=seed.copy();shift=complex(target);trace=[]
    for iteration in range(12):
        try:
            links=splu((problem.stiffness-shift*problem.mass).tocsc()).solve(field)
            updated=problem.mass@links;updated/=np.linalg.norm(updated)
            applied=operator@updated;value=complex(np.vdot(updated,applied))
            residual=relative_residual(operator,updated,value)
            previous_overlap=float(abs(np.vdot(field,updated)))
            source_overlap=float(abs(np.vdot(seed,updated)))
            trace.append({"iteration":iteration,"delta_per_m":value.real,"alpha_per_m":value.imag,
                "original_pencil_relative_residual":residual,"previous_overlap":previous_overlap,"source_overlap":source_overlap})
            if previous_overlap<.75 or source_overlap<.75:break
            if residual<1e-11:
                return value,updated,source_overlap,{"solver":"physical_center_inverse_iteration","trace":trace}
            field=updated;shift=value+complex(1e-9,2e-9)
        except Exception as error:
            trace.append({"iteration":iteration,"exception":repr(error)});break
    values,vectors=problem.near(target,count,seed,tol=1e-10,maxiter=1800,dense_limit=0)
    overlaps=abs(vectors.conj().T@seed);index=int(np.argmax(overlaps))
    return values[index],vectors[:,index],float(overlaps[index]),{"solver":"ARPACK_fallback","inverse_trace":trace}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--controls",nargs="+",default=["39:48:1","39:64:1","39:48:0.5"])
    parser.add_argument("--roots",type=int,default=3)
    parser.add_argument("--process",action="store_true",help="13 fixed-reference geometric perturbation branch controls")
    parser.add_argument("--contour-samples",type=int,help="Explicit equal-area rounded-triangle contour resolution, without changing historical defaults")
    parser.add_argument("--inverse",action="store_true",help="Strict physical-center inverse iteration with ARPACK fallback")
    parser.add_argument("--separable-far-field",action="store_true",help="Validated exact Cartesian-separable point sum")
    parser.add_argument("--spec-input",type=Path,help="Optional new three-hole spec for a controlled continuation")
    parser.add_argument("--spec-name",help="Select a named row in a candidate list")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=json.loads((args.case/"summary.json").read_text())
    if args.spec_input:
        specification=json.loads(args.spec_input.read_text())
        if isinstance(specification,list):specification=next(r for r in specification if r["name"]==args.spec_name)
        source["three_hole_spec"]=specification.get("three_hole_spec",specification)
    if args.contour_samples is not None:
        source["three_hole_spec"]["triangle_samples_per_corner"]=args.contour_samples
    closed_directory=args.case/"frequency_closed" if (args.case/"frequency_closed").is_dir() else args.case
    poles=json.loads((closed_directory/"frequency_closed_poles.json").read_text())
    poles=sorted([row for row in poles if row["status"]=="completed"],key=lambda row:row["g_th_cm"])[:args.roots]
    arrays=np.load(closed_directory/"frequency_closed_fields.npz")
    triangulation=Delaunay(np.column_stack((arrays["x_um"],arrays["y_um"])))
    controls=[(text,None) for text in args.controls]
    if args.process:
        spec=ThreeHole(**source["three_hole_spec"]);original=candidate(spec).inclusions
        perturbations=[("nominal",original)]
        for bias in (-3.,3.):
            perturbations.append((f"all_edges_{bias:+g}nm",perturb_triangular_inclusions(original,spec.lattice_nm,np.random.default_rng(20261021),etch_bias_nm=bias)))
            for index in range(3):
                holes=list(original)
                holes[index]=perturb_triangular_inclusions((holes[index],),spec.lattice_nm,np.random.default_rng(20261021),etch_bias_nm=bias)[0]
                perturbations.append((f"hole{index}_edge_{bias:+g}nm",tuple(holes)))
        for index in (1,2):
            for sign in (-1,1):
                holes=list(original);shift=np.linalg.solve(TRIANGULAR_DIRECT,np.asarray((3.*sign,0.))/spec.lattice_nm)
                hole=holes[index]
                if hasattr(hole,"center_fractional"):
                    holes[index]=replace(hole,center_fractional=tuple(np.asarray(hole.center_fractional)+shift))
                else:holes[index]=replace(hole,vertices_fractional=tuple(map(tuple,np.asarray(hole.vertices_fractional)+shift)))
                perturbations.append((f"satellite{index}_x_{3*sign:+g}nm",tuple(holes)))
        controls=[(args.controls[0],item) for item in perturbations]
    trials=[]
    with threadpool_limits(limits=1):
        for control,perturbation in controls:
            started=time.perf_counter();n,d,dz=map(float,control.split(":"));n,d=int(n),int(d)
            spec,parameters,model=model_for(source,d,dz,None if perturbation is None else perturbation[1])
            problem,grid=triangular_finite_eigenproblem(model,TriangularFiniteSpec(150.,n,0.,24,"square","staggered"))
            label=f"N{n}_D{d}_dz{dz:g}" if perturbation is None else perturbation[0]
            output=args.output/label;output.mkdir(exist_ok=True)
            records=[];fields=[]
            for rank,pole in enumerate(poles):
                old=arrays[f"root_{pole['seed_rank']}_fields"]
                interpolator=LinearNDInterpolator(triangulation,old.T,fill_value=0.)
                seed=interpolator(np.column_stack((grid.x_um,grid.y_um))).T.ravel()
                seed/=np.linalg.norm(seed)
                # Avoid factorizing exactly at the already known eigenvalue.
                target=complex(np.vdot(seed,problem.center_operator()@seed)/np.vdot(seed,seed))+complex(50.,5.)
                if args.inverse:
                    value,field,overlap,solver_info=follow(problem,target,seed)
                else:
                    values,vectors=problem.near(target,24,seed,tol=1e-10,maxiter=1800,dense_limit=0)
                    overlaps=abs(vectors.conj().T@seed);index=int(np.argmax(overlaps))
                    value,field,overlap=values[index],vectors[:,index],float(overlaps[index]);solver_info={"solver":"ARPACK"}
                if overlap<.6:raise RuntimeError(f"Lost branch {rank} in {control}: overlap {overlap}")
                row=scalar_record(value,field,model,parameters,problem,grid,model.bragg_wavelength_nm,rank)
                rx,ry=model.radiation_amplitudes(field.reshape(6,grid.points))
                mode=TriangularFiniteMode(f"control_{rank}",value.real,value.imag,field.reshape(6,grid.points),1.,grid,rx,ry,discretization="staggered")
                far=(separable_far_field if args.separable_far_field else triangular_vector_far_field)(mode,row["frozen_CWT_pole_wavelength_nm"],1.,201)
                row["beam"]=beam_metrics(far.angle_deg,far.field_x,far.field_y,requested_view_deg=1.,source_nyquist_deg=far.alias_free_square_view_deg)
                row["source_closed_rank"]=rank;row["field_overlap"]=overlap;row["solver_info"]=solver_info;records.append(row);fields.append(field.reshape(6,grid.points))
            result={"three_hole_spec":asdict(spec),"N":n,"D":d,"dz_nm":dz,"reference_wavelength_nm":model.bragg_wavelength_nm,
                "nominal_size_um":[300,300],"sampled_node_span_um":[float(np.ptp(grid.x_um)),float(np.ptp(grid.y_um))],
                "scope":"Tracked source branches at fixed actual reference and period; not frequency-closed or a complete fine-grid spectrum",
                "tracked_modes":records,"elapsed_seconds":time.perf_counter()-started}
            if perturbation is not None:
                result.update(perturbation=label,fabrication_actual=evaluate_fabrication(perturbation[1],spec.lattice_nm,TRIANGULAR_DIRECT,RULE))
            json_write(output/"summary.json",result)
            np.savez_compressed(output/"tracked_fields.npz",fields=fields,x_um=grid.x_um,y_um=grid.y_um,coupling_m=model.coupling_m)
            trials.append(result);json_write(args.output/"trials.json",trials)
            write_csv(args.output/"trials.csv",[{"perturbation":r.get("perturbation"),"N":r["N"],"D":r["D"],"dz_nm":r["dz_nm"],"reference_wavelength_nm":r["reference_wavelength_nm"],
                "g_main_cm":r["tracked_modes"][0]["g_th_cm"],"ellipse":r["tracked_modes"][0]["beam"]["ellipticity"],
                "center_to_peak":r["tracked_modes"][0]["beam"]["center_to_peak"],"elapsed_seconds":r["elapsed_seconds"]} for r in trials])
            print(label,records[0]["g_th_cm"],records[0]["beam"]["ellipticity"],flush=True)


if __name__=="__main__":main()
