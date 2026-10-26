"""Close continued finite modes and optionally retune their actual period."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[_key]="1"
import argparse
from dataclasses import replace,asdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from pcselsim.mode_identity import independent_eigenmode
from scripts.optimization.matched300_triangular_search import ThreeHole,geometry
from scripts.optimization.matched300_triangular_close import close
from scripts.optimization.triangular_semiconductor_search import json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--roots",type=int,default=3);parser.add_argument("--target",type=float,default=950.)
    parser.add_argument("--retune",action="store_true")
    parser.add_argument("--newton",action="store_true",help="Use exact-residual checked group-index Newton with Brent fallback")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    original=json.loads((args.input/"summary.json").read_text());arrays=np.load(args.input/"tracked_fields.npz")
    fields=arrays["fields"];records=original["tracked_modes"]
    order=np.argsort([r["g_th_cm"] for r in records]);fields=fields[order];records=[records[i] for i in order]
    # A mesh continuation can bring two source seeds to one fine eigenvector.
    # Preserve independent near-degenerate vectors, but remove repeats of the
    # same span before interpreting the nearest competitor.
    kept=[]
    for index,(record,field) in enumerate(zip(records,fields)):
        value=complex(record["delta_per_m"],record["alpha_per_m"])
        if not independent_eigenmode(value,field,
                [complex(records[j]["delta_per_m"],records[j]["alpha_per_m"]) for j in kept],
                [fields[j] for j in kept]):continue
        kept.append(index)
    fields=fields[kept];records=[records[i] for i in kept]
    seed=args.output/"finite_seed";seed.mkdir(exist_ok=True)
    row={"three_hole_spec":original["three_hole_spec"],"N":original["N"],"D":original["D"],"dz_nm":original["dz_nm"],"spectrum":records,
        "scope":"Saved actual-reference continuation seeds; tracked-pole pool, not complete fine spectrum","source_control_path":str(args.input.resolve()),
        "seed_reference_three_hole_spec":original["three_hole_spec"],"seed_reference_wavelength_nm":original["reference_wavelength_nm"],
        "seed_coupling_role":"The NPZ coupling and eigenpairs are a predictor reference at seed_reference_three_hole_spec. Period retuning updates the target three_hole_spec; every closed pole independently rebuilds C(lambda,a)."}
    json_write(seed/"result.json",row)
    np.savez_compressed(seed/"complex_fields.npz",fields=fields,
        eigenvalues=[complex(r["delta_per_m"],r["alpha_per_m"]) for r in records],coupling_m=arrays["coupling_m"],x_um=arrays["x_um"],y_um=arrays["y_um"])
    trace=[]
    if args.retune:
        for iteration in range(3):
            result=close(seed,args.output/f"retune_{iteration}",1,original["D"],original["dz_nm"],True,True,args.newton)
            if not result["lowest"]:raise RuntimeError("Actual period retuning closure failed")
            wavelength=result["lowest"]["wavelength_nm"];spec=ThreeHole(**row["three_hole_spec"])
            trace.append({"iteration":iteration,"period_nm":spec.lattice_nm,"actual_closed_wavelength_nm":wavelength})
            if abs(wavelength-args.target)<=.002:break
            row["three_hole_spec"]=asdict(replace(spec,lattice_nm=spec.lattice_nm*args.target/wavelength))
            json_write(seed/"result.json",row)
        else:raise RuntimeError("Fine-grid actual period did not reach target tolerance")
    spec=ThreeHole(**row["three_hole_spec"])
    if not geometry(spec)["margin_pass"]:raise RuntimeError("Retuned actual geometry violates common manufacturing margin")
    final=close(seed,args.output/"frequency_closed",min(args.roots,len(records)),original["D"],original["dz_nm"],True,True,args.newton)
    final.update(actual_target_wavelength_nm=args.target,actual_target_tolerance_nm=.05,
        actual_target_pass=bool(final["lowest"] and abs(final["lowest"]["wavelength_nm"]-args.target)<=.05),
        finite_seed_path=str((seed/"result.json").resolve().relative_to(ROOT)),period_retuning=trace,
        source_control_path=str(args.input.resolve().relative_to(ROOT)))
    json_write(args.output/"summary.json",final)
    print("Formal continued case",final["N"],final["lowest"]["g_th_cm"],final["lowest"]["wavelength_nm"],final["lowest"]["beam"]["ellipticity"],final["g_th_gap_fraction"],flush=True)


if __name__=="__main__":main()
