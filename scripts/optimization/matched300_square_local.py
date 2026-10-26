"""Multivariate local refinement scored against every coarse QZ pole."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import argparse,json,sys
from pathlib import Path
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from pcselsim.fabrication import evaluate_fabrication
from scripts.optimization.matched300_square_search import Design,build_cell,save_json,OUT,RULES
from scripts.optimization.matched300_square_validate import spectral_challenge

def proposals(source,count=256,seed=30021022):
    data=[r for r in source if r.get("status")=="completed" and r["lowest"]["beam"]["ellipticity"]<1.2
          and r["lowest"]["beam"]["center_to_peak"]>.9]
    groups=[sorted(data,key=lambda r:r["lowest"]["g_th_cm1"])[:5],
        sorted(data,key=lambda r:r["lowest"]["g_th_cm1"]/(1+4*r["relative_material_gap"]))[:5],
        sorted(data,key=lambda r:-r["relative_material_gap"])[:4]]
    seeds=[];seen=set()
    for group in groups:
        for r in group:
            if r["id"] not in seen:seeds.append(Design(**r["design"]));seen.add(r["id"])
    rng=np.random.default_rng(seed);accepted=[];rejected=0
    names=("fill","main_share","distance","position_angle_deg","main_aspect","satellite_aspect",
           "main_angle_deg","satellite_angle_deg","main_exponent","satellite_exponent")
    scale=np.array([.025,.09,.035,5.,.18,.18,18.,18.,.4,.4])
    lower=np.array([.04,.50,.34,25.,1.,1.,-90.,-90.,2.3,2.3]);upper=np.array([.28,.88,.69,65.,2.,2.,90.,90.,4.8,4.8])
    while len(accepted)<count:
        parent=seeds[len(accepted)%len(seeds)];spread=(.45,1.,1.5)[len(accepted)%3]
        value=np.array([getattr(parent,k) for k in names])+rng.normal(size=10)*scale*spread
        value=np.clip(value,lower,upper)
        d=replace(parent,**dict(zip(names,map(float,value))),tag=f"local{seed}_{len(accepted)}")
        fab=evaluate_fabrication(build_cell(d,40).inclusions,d.lattice_nm,rules=RULES)
        if fab["margin_pass"]:accepted.append(d)
        else:rejected+=1
    return accepted,rejected

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",type=Path)
    p.add_argument("--output",type=Path,default=OUT/"local_QZ");p.add_argument("--count",type=int,default=256)
    p.add_argument("--D",type=int,default=6);p.add_argument("--dz",type=float,default=5.)
    args=p.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    designs,rejected=proposals(json.loads(args.source.read_text(encoding="utf-8")),args.count)
    save_json(args.output/"designs.json",[asdict(d) for d in designs]);save_json(args.output/"plan.json",{"count":args.count,"geometry_rejections":rejected,"D":args.D,"dz_nm":args.dz,"grid":9,"full_QZ":True,"scope":"coarse screen; angular +/-1 deg not yet alias-free at N9"})
    results=[]
    for i,d in enumerate(designs):
        try:r=spectral_challenge(d,args.D,args.dz,9,True)
        except Exception as e:r={"status":"failed","design":asdict(d),"error":str(e)}
        results.append(r)
        if (i+1)%8==0 or i==len(designs)-1:
            save_json(args.output/"compact_results.json",results)
            print(f"{i+1}/{len(designs)} basic={sum(x.get('basic_beam_pass',False) for x in results)}",flush=True)
    save_json(args.output/"run_status.json",{"status":"completed","count":len(results)})

if __name__=="__main__":main()
