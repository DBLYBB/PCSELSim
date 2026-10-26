"""Verify the 400-um single-ellipse triangular alternative found by size scaling.

This retains the 950-nm unit cell and changes the square aperture only.
It is an alternative optimized tradeoff, with larger estimated total current.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key]="1"
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from pcselsim.fabrication import perturb_triangular_inclusions
from pcselsim.run_output import prepare_output,write_provenance
from scripts._triangular_workflow import TriangularCandidate
from scripts.optimization.triangular_semiconductor_search import (
    Motif,evaluate_finite,json_write,make_candidate,parameters,threadpool_limits)
import numpy as np


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    output=prepare_output(args.output,ROOT)
    write_provenance(output,ROOT)
    (output/"executed_script.py").write_bytes(Path(__file__).read_bytes())
    base_path=ROOT/"results/current/optimization_20261021/triangular/highmesh_retuned_v1/candidates.json"
    base=next(r for r in json.loads(base_path.read_text()) if r["name"]=="local_0067_950nm")
    spec=replace(Motif(**base["spec"]),name="local_0067_950nm_400um",radius_um=200.)
    json_write(output/"run_plan.json",{"spec":asdict(spec),"base_path":str(base_path),
        "base_sha256":hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "checks":["N35 local 24 poles per distinct Gamma target","N23 additional 17 real-frequency targets","N7 dense generalized complete spectrum","N19 global +/-3 nm etch"],
        "scope":"linear six-wave passive poles; not full-wave, nonlinear lasing, or complete N35 spectrum"})
    json_write(output/"run_status.json",{"status":"running"})
    rows=[]
    with threadpool_limits(limits=1):
        for label,n,extra,dense in (("N35",35,0,False),("wide_N23",23,17,False),("dense_N7",7,0,True)):
            row=evaluate_finite(spec,n,10,2.,24,output/label,extra_targets=extra,full_dense=dense)
            row["check"]=label
            rows.append(row)
            json_write(output/"checks.json",rows)
            print(label,"g",row["g_th_cm"],"gap",row["g_th_gap_fraction"],"ellipse",row["ellipticity"],flush=True)
        candidate=make_candidate(spec)
        p=parameters(spec)
        for bias in (-3.,3.):
            holes=perturb_triangular_inclusions(candidate.inclusions,p.LATTICE_CONSTANT_NM,
                np.random.default_rng(20261021),etch_bias_nm=bias)
            override=TriangularCandidate(spec.name+f"_etch{bias:+g}",holes)
            row=evaluate_finite(spec,19,10,2.,24,output/f"etch{bias:+g}",candidate_override=override)
            row.update(check=f"etch{bias:+g}",etch_bias_nm=bias,actual_inclusions=[asdict(h) for h in holes])
            rows.append(row);json_write(output/"checks.json",rows)
            print(row["check"],"g",row["g_th_cm"],"gap",row["g_th_gap_fraction"],"ellipse",row["ellipticity"],flush=True)
    final=rows[0].copy()
    final.update(checks=rows[1:],scope="N35 local poles plus wide N23 and complete coarse N7. Uniform etched unit cells, not random disorder. Larger area raises conditional current.")
    json_write(output/"summary.json",final)
    json_write(output/"candidates.json",[final])
    json_write(output/"run_status.json",{"status":"completed","checks":len(rows)})


if __name__=="__main__":main()
