"""Retune a selected motif to an actual 950-nm finite pole, then close rivals."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[key]="1"
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.optimization.matched300_triangular_search import ThreeHole,evaluate
from scripts.optimization.matched300_triangular_close import close
from scripts.optimization.triangular_semiconductor_search import json_write,threadpool_limits


def recover_summary(output,target=950.):
    output=output.resolve()
    closed=json.loads((output/"frequency_closed/summary.json").read_text(encoding="utf-8"))
    closed["period_retuning"]=json.loads((output/"period_retuning.json").read_text(encoding="utf-8"))
    closed["actual_target_wavelength_nm"]=target;closed["actual_target_tolerance_nm"]=.05
    closed["actual_target_pass"]=bool(closed["lowest"] and abs(closed["lowest"]["wavelength_nm"]-target)<=.05)
    closed["finite_seed_path"]=str((output/"final_seed/result.json").relative_to(ROOT))
    json_write(output/"summary.json",closed)
    return closed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cells",type=int,default=31);p.add_argument("--order",type=int,default=32);p.add_argument("--dz",type=float,default=1.)
    p.add_argument("--roots",type=int,default=6);p.add_argument("--target-nm",type=float,default=950.);p.add_argument("--use-existing-period",action="store_true")
    p.add_argument("--recover",action="store_true",help="Write final metadata from already completed numerical results")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.recover:
        recover_summary(a.output,a.target_nm);return
    source=json.loads(a.input.read_text(encoding="utf-8"));spec=ThreeHole(**source["three_hole_spec"])
    if not a.use_existing_period:
        spec=replace(spec,lattice_nm=spec.lattice_nm*a.target_nm/source["frozen_CWT_pole_wavelength_nm"])
    trace=[]
    with threadpool_limits(limits=1):
        for iteration in range(4):
            seed_dir=a.output/f"retune_{iteration}_seed"
            seed=evaluate(spec,23,a.order,a.dz,18,seed_dir,0)
            if seed["status"]!="finite_solved":raise RuntimeError("Retuned motif violates the shared process-margin rules")
            closed=close(seed_dir,a.output/f"retune_{iteration}_closed",1,a.order,a.dz)
            if closed["roots_closed"]!=1:raise RuntimeError("Retuning closure failed")
            wl=closed["lowest"]["wavelength_nm"]
            trace.append({"iteration":iteration,"period_nm":spec.lattice_nm,"actual_closed_wavelength_nm":wl,
                "g_th_cm":closed["lowest"]["g_th_cm"],"beam":closed["lowest"]["beam"]})
            json_write(a.output/"period_retuning.json",trace)
            print("outer retune",iteration,spec.lattice_nm,wl,flush=True)
            if abs(wl-a.target_nm)<.01:break
            spec=replace(spec,lattice_nm=spec.lattice_nm*a.target_nm/wl)
        else:raise RuntimeError("Actual pole wavelength did not converge to 950 +/- .01 nm")
        json_write(a.output/"design.json",asdict(spec))
        final_seed=evaluate(spec,a.cells,a.order,a.dz,24,a.output/"final_seed",17)
        if final_seed["status"]!="finite_solved":raise RuntimeError("Final motif violates shared fabrication margin")
        closed=close(a.output/"final_seed",a.output/"frequency_closed",a.roots,a.order,a.dz)
        closed["period_retuning"]=trace
        closed["actual_target_wavelength_nm"]=a.target_nm
        closed["actual_target_tolerance_nm"]=.05
        closed["actual_target_pass"]=bool(closed["lowest"] and abs(closed["lowest"]["wavelength_nm"]-a.target_nm)<=.05)
        closed["finite_seed_path"]=str((a.output/"final_seed/result.json").resolve().relative_to(ROOT))
        json_write(a.output/"summary.json",closed)


if __name__=="__main__":main()
