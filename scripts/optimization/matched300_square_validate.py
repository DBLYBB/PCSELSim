"""Accuracy, full coarse QZ, and independent lateral-pole challenges."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"
import argparse,json,sys,time
from dataclasses import asdict,replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from pcselsim.custom_analysis import FourWaveOpticalSpec,LinearMode,finite_area_eigenproblem
from pcselsim.finite_discretization import relative_residual
from pcselsim.polarization import polarization_metrics
from scripts.optimization.matched300_square_search import (Design,build_model,evaluate,
    acceptable,save_json,clean,threshold,direct_vector_far_field,BASE,OUT)
from scripts.optimization.square_semiconductor_validate import independent_pole


def spectral_challenge(d,order=8,dz=3.,grid=9,full=True,count=24,wide=12,output=None,tune_period=False,vertices=64,extra_targets=None):
    started=time.monotonic();d,cell,model=build_model(d,order,dz,tune_period=tune_period,vertices=vertices)
    spec=FourWaveOpticalSpec(950.,model.effective_index,model.group_index,300.,5.,
        tuple(model.eigenvalues_m.real/100),tuple(model.eigenvalues_m.imag/100),grid,
        "staggered",count,None,1500)
    problem=finite_area_eigenproblem(spec,model.coupling_m);size=4*grid*grid
    rng=np.random.default_rng(30021021);v0=rng.normal(size=size)+1j*rng.normal(size=size)
    failures=[];targets=[];values=[];fields=[]
    if full:
        problem.near(0j,4,v0,dense_limit=size);v,f=problem._dense_spectrum
        for value,field in zip(v,f.T):
            if value.imag>=-1e-7:values.append(value);fields.append(field.reshape(4,grid,grid))
    else:
        spacing=np.pi/300e-6
        target_values=[complex(v.real+offset*spacing,max(v.imag,.1)) for v in model.eigenvalues_m for offset in (-2,-1,0,1,2)]
        target_values.extend(complex(x,.1) for x in np.linspace(model.eigenvalues_m.real.min()-3*spacing,model.eigenvalues_m.real.max()+3*spacing,wide))
        # A regular real-axis grid can miss narrow low-loss lateral clusters.
        # Complete coarse-QZ poles supply independent complex and near-axis
        # targets rather than trusting four selected Gamma-family states.
        for value in (extra_targets or ()):
            target_values.extend((complex(value)+10+10j,complex(value.real,.1)))
        for index,target in enumerate(target_values):
            before=len(values)
            try:
                vs,fs=problem.near(target,count,v0,maxiter=1500)
                for value,field in zip(vs,fs.T):
                    field=field.reshape(4,grid,grid)
                    if value.imag>=-1e-7 and independent_pole(value,field,values,fields):
                        values.append(value);fields.append(field)
            except Exception as e:failures.append({"target":[target.real,target.imag],"error":str(e)})
            targets.append({"target":[target.real,target.imag],"new":len(values)-before})
            if output and ((index+1)%8==0 or index==len(target_values)-1):
                print(f"spectral clusters {index+1}/{len(target_values)}; {len(values)} independent passive poles",flush=True)
    gamma=model.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME];rows=[];arrays={"coupling_m":model.coupling_m}
    operator=problem.center_operator()
    for rank,index in enumerate(np.argsort(np.imag(values))[:24]):
        value,field=values[index],fields[index];rx,ry=model.radiation_fields(field)
        mode=LinearMode(f"pole_{rank}",float(value.real),float(value.imag),field,0.,rx,ry,
            discretization="staggered",eigen_residual=relative_residual(operator,field,value))
        axis,fx,fy,beam=direct_vector_far_field(mode,950.,300.,161)
        flat=field.reshape(4,-1);local=float(np.vdot(flat,model.coupling_m@flat).imag/np.vdot(flat,flat).real)
        rows.append({"mode":mode.name,"alpha_field_cm1":value.imag/100,"delta_per_m":value.real,
            "eigen_residual":mode.eigen_residual,**threshold(value.imag,gamma,model.effective_index,300.),
            "beam":beam,"radiative_extraction_fraction":2*local/(2*value.imag+500),
            "polarization":polarization_metrics(axis,fx,fy,core_radius_deg=.2),
            "detuning_fraction_Bragg":abs(value.real)/(2*np.pi/(d.lattice_nm*1e-9))})
        if output and rank<12:arrays.update({f"pole_{rank}_field":field,f"pole_{rank}_far_x":fx,f"pole_{rank}_far_y":fy})
    result={"status":"completed","id":d.identifier,"design":asdict(d),
        "D":order,"dz_nm":dz,"grid":grid,"vertices":vertices,"full_QZ":full,"distinct_poles":len(values),
        "targets":targets,"failures":failures,"ordered_poles":rows,"lowest":rows[0],
        "extra_coarse_QZ_targets":[[v.real,v.imag] for v in (extra_targets or ())],
        "relative_material_gap":rows[1]["g_th_cm1"]/rows[0]["g_th_cm1"]-1,
        "elapsed_seconds":time.monotonic()-started,"basic_beam_pass":acceptable(rows[0]),
        "scope":"complete finite coarse pencil" if full else "finite multi-shift lateral survey; not exhaustive"}
    if output:
        output.mkdir(parents=True,exist_ok=True);save_json(output/"spectral_challenge.json",result)
        np.savez_compressed(output/"lateral_fields.npz",axis_deg=axis,**arrays)
    return result


def top_diverse(records,n=24):
    good=[r for r in records if r.get("status")=="completed" and r["fabrication"]["margin_pass"] and r["basic_beam_pass"]]
    # The first branch is a low-threshold list; the other preserves high-gap
    # beam-shaped alternatives, so a weighted score cannot hide the Pareto set.
    lists=[sorted(good,key=lambda r:r["lowest"]["g_th_cm1"]),
           sorted(good,key=lambda r:r["lowest"]["g_th_cm1"]*(1+2*max(0,.10-r["relative_material_gap"]))),
           sorted([r for r in good if r["relative_material_gap"]>=.10],key=lambda r:r["lowest"]["g_th_cm1"])]
    chosen=[];seen=set()
    for i in range(n):
        seq=lists[i%len(lists)]
        if not seq:continue
        r=seq.pop(0)
        while r["id"] in seen and seq:r=seq.pop(0)
        if r["id"] not in seen:chosen.append(r);seen.add(r["id"])
    return chosen


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--top",type=int,default=24)
    p.add_argument("--D",type=int,default=8);p.add_argument("--dz",type=float,default=3.)
    p.add_argument("--grid",type=int,default=25);p.add_argument("--qz",action="store_true")
    p.add_argument("--shift",action="store_true");p.add_argument("--keep-fields",action="store_true")
    p.add_argument("--retune-bragg",action="store_true")
    p.add_argument("--vertices",type=int,default=64)
    p.add_argument("--coarse-census",type=Path)
    args=p.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.source.read_text(encoding="utf-8"))
    chosen=top_diverse(source,args.top) if isinstance(source,list) and "lowest" in source[0] else [{"design":x} for x in source]
    coarse=[] if args.coarse_census is None else json.loads(args.coarse_census.read_text(encoding="utf-8"))
    results=[]
    for i,r in enumerate(chosen):
        d=Design(**r["design"]);folder=args.output/f"case_{i:03d}" if args.keep_fields else None
        signature=lambda design:json.dumps({k:v for k,v in design.items() if k not in ("tag","lattice_nm")},sort_keys=True)
        seed=next((x for x in coarse if signature(x["design"])==signature(asdict(d))),None)
        extras=[] if seed is None else [complex(x["delta_per_m"],100*x["alpha_field_cm1"]) for x in seed["ordered_poles"]]
        try:
            result=spectral_challenge(d,args.D,args.dz,args.grid,args.qz,output=folder,tune_period=args.retune_bragg,vertices=args.vertices,extra_targets=extras) if args.qz or args.shift else evaluate(d,args.D,args.dz,args.grid,args.vertices,24,folder,tune_period=args.retune_bragg)
            results.append(result)
            print(f"{i+1}/{len(chosen)} {d.identifier} g={result['lowest']['g_th_cm1']:.5f} ell={result['lowest']['beam']['ellipticity']:.5f} gap={result['relative_material_gap']:.4f}",flush=True)
        except Exception as e:
            results.append({"status":"failed","design":asdict(d),"error":str(e)});print(i+1,str(e),flush=True)
        save_json(args.output/"compact_results.json",results)
    save_json(args.output/"run_status.json",{"status":"completed","total":len(results),"success":sum(r["status"]=="completed" for r in results)})

if __name__=="__main__":main()
