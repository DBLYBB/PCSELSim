"""Separate reciprocal, vertical and lateral numerical controls for a finalist."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[key]="1"
import argparse
import json
from pathlib import Path
import sys
import time
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scripts.optimization.matched300_triangular_search import ThreeHole,evaluate,candidate,base_spec
from scripts.optimization.triangular_semiconductor_search import json_write,write_csv,threadpool_limits,evaluate_finite


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--name");p.add_argument("--kind",choices=("D","dz","N","dense","process"),required=True)
    p.add_argument("--values",type=float,nargs="+",help="Override the values for one convergence coordinate")
    p.add_argument("--process-order",type=int,default=32)
    p.add_argument("--process-cells",type=int,default=23)
    p.add_argument("--audit-order",type=int,default=32)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.input.read_text(encoding="utf-8"))
    if isinstance(data,list):row=next(r for r in data if r["name"]==a.name) if a.name else data[0]
    else:row=data
    spec=ThreeHole(**row.get("three_hole_spec",row));trials=[]
    if a.kind=="process":
        from pcselsim.fabrication import perturb_triangular_inclusions,TRIANGULAR_DIRECT,evaluate_fabrication
        from scripts.optimization.matched300_triangular_search import RULE
        original=candidate(spec); perturbations=[("nominal",original.inclusions)]
        for bias in (-3.,3.):
            perturbations.append((f"all_edges_{bias:+g}nm",perturb_triangular_inclusions(original.inclusions,spec.lattice_nm,np.random.default_rng(20261021),etch_bias_nm=bias)))
            for index in range(3):
                holes=list(original.inclusions)
                holes[index]=perturb_triangular_inclusions((holes[index],),spec.lattice_nm,np.random.default_rng(20261021),etch_bias_nm=bias)[0]
                perturbations.append((f"hole{index}_edge_{bias:+g}nm",tuple(holes)))
        for index in (1,2):
            for sign in (-1,1):
                holes=list(original.inclusions);shift=np.linalg.solve(TRIANGULAR_DIRECT,np.asarray((3.*sign,0.))/spec.lattice_nm)
                h=holes[index]
                if hasattr(h,"center_fractional"):
                    holes[index]=replace(h,center_fractional=tuple(np.asarray(h.center_fractional)+shift))
                else:holes[index]=replace(h,vertices_fractional=tuple(map(tuple,np.asarray(h.vertices_fractional)+shift)))
                perturbations.append((f"satellite{index}_x_{3*sign:+g}nm",tuple(holes)))
        with threadpool_limits(limits=1):
            for label,holes in perturbations:
                t=time.perf_counter()
                try:
                    r=evaluate_finite(base_spec(spec),a.process_cells,a.process_order,1.,24,None,candidate_override=replace(original,inclusions=holes),extra_targets=9)
                    r["nominal_three_hole_spec"]=asdict(spec);r["fabrication_actual"]=evaluate_fabrication(holes,spec.lattice_nm,TRIANGULAR_DIRECT,RULE)
                except Exception as error:r={"status":"failed","exception":repr(error)}
                r["perturbation"]=label;r["elapsed_seconds"]=time.perf_counter()-t;trials.append(r)
                json_write(a.output/"trials.json",trials);write_csv(a.output/"trials.csv",trials)
                print(label,r.get("g_th_cm"),r.get("ellipticity"),r.get("center_to_peak"),r.get("g_th_gap_fraction"),flush=True)
        return
    controls={"D":[(23,d,2.,0,False) for d in (6,8,10,14,18,24)],
        "dz":[(23,a.audit_order,dz,0,False) for dz in (4.,2.,1.,.5)],
        "N":[(n,a.audit_order,1.,17,False) for n in (17,23,31,39)],
        "dense":[(5,a.audit_order,1.,0,True)]}[a.kind]
    if a.values:
        controls=[(23,int(v),2.,0,False) for v in a.values] if a.kind=="D" else (
            [(23,a.audit_order,float(v),0,False) for v in a.values] if a.kind=="dz" else [(int(v),a.audit_order,1.,17,False) for v in a.values])
    with threadpool_limits(limits=1):
        for n,d,dz,extra,dense in controls:
            t=time.perf_counter()
            save=a.output/f"N{n}_D{d}_dz{dz:g}" if a.kind=="N" and n in (31,39,52) else None
            try:r=evaluate(spec,n,d,dz,24,save,extra,dense)
            except Exception as error:r={"status":"failed","N":n,"D":d,"dz_nm":dz,"exception":repr(error)}
            r["elapsed_seconds"]=time.perf_counter()-t;trials.append(r)
            json_write(a.output/"trials.json",trials);write_csv(a.output/"trials.csv",trials)
            print(a.kind,n,d,dz,r.get("g_th_cm"),r.get("ellipticity"),r.get("g_th_gap_fraction"),r["elapsed_seconds"],flush=True)


if __name__=="__main__":main()
