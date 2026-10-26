"""Process sensitivity using spectra and independent real roots per sample.

The simultaneous worst-case process corner is not represented by these
one-variable +/-3-nm deviations. No nonlinear/thermal stability is inferred.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from pcselsim.run_output import prepare_output,write_provenance
from scripts.optimization.square_semiconductor_search import Design,save_json
from scripts.optimization.square_semiconductor_validate import survey,perturbed_designs
from scripts.optimization.square_extended_validate import close_survey


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("--design-index",type=int,default=0)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--length",type=float,default=165.)
    parser.add_argument("--wavelength",type=float,default=916.47)
    parser.add_argument("--D",type=int,default=10)
    parser.add_argument("--dz",type=float,default=2.)
    parser.add_argument("--grid",type=int,default=33)
    args=parser.parse_args(argv)
    source=json.loads(args.source.read_text(encoding="utf8"))
    record=source["finalists"][args.design_index]["design"] if "finalists" in source else source["design"]
    d=replace(Design(**record),length_um=args.length)
    output=prepare_output(args.output,ROOT)
    write_provenance(output,ROOT)
    designs=perturbed_designs(d)
    save_json(output/"run_plan.json",vars(args)|{"designs":[asdict(item) for item in designs],
        "scope":"one-variable process sensitivity, not simultaneous corner or statistical yield"})
    save_json(output/"run_status.json",{"status":"running"})
    summaries=[]
    for index,changed in enumerate(designs):
        folder=output/f"case_{index:02d}_{changed.tag}"
        folder.mkdir()
        try:
            spectrum=survey(changed,folder,args.D,args.dz,args.grid,24,args.wavelength,17)
            result=close_survey(changed,folder,spectrum,order=args.D,dz=args.dz,roots=4)
            result["status"]="completed"
        except Exception as error:
            result={"status":"failed","design":asdict(changed),"error":str(error)}
        summaries.append(result)
        save_json(output/"robustness_summary.json",summaries)
        print(f"PROCESS {index+1}/{len(designs)} {changed.tag}: {result['status']}",flush=True)
    save_json(output/"run_status.json",{"status":"completed","cases":len(summaries),
        "failed":sum(r["status"]!="completed" for r in summaries)})


if __name__=="__main__":
    main()
