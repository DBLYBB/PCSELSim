"""Directed quarter-period diagonal anchors, including process counterexamples."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import sys
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from pcselsim.fabrication import evaluate_fabrication
from scripts.optimization.matched300_square_search import Design,build_cell,save_json,OUT,RULES
from scripts.optimization.matched300_square_validate import spectral_challenge

def main():
    output=OUT/"quarter_period_anchors";output.mkdir(parents=True,exist_ok=True);rows=[]
    designs=[]
    for fill in (.025,.035,.045,.055,.07,.09):
        for share in (.50,.52,.55,.60):
            for variant in range(3):
                designs.append(Design(fill=fill,main_share=share,distance=np.sqrt(2)*.25,
                    main_shape="circle" if variant==0 else "ellipse",
                    satellite_shape="circle" if variant==0 else "ellipse",
                    main_aspect=1. if variant==0 else 1.3,
                    satellite_aspect=(1.,1.3,1.15)[variant],main_angle_deg=-35.,
                    satellite_angle_deg=-35. if variant<2 else 25.,tag=f"quarter_anchor_{len(designs)}"))
    save_json(output/"designs.json",[asdict(d) for d in designs])
    for i,d in enumerate(designs):
        try:
            fab=evaluate_fabrication(build_cell(d).inclusions,280.,rules=RULES)
            if fab["minimum_periodic_gap_nm"]<=0:raise ValueError("overlapping motif")
            r=spectral_challenge(d,8,3.,9,True);r["fabrication"]=fab
        except Exception as e:r={"status":"failed","design":asdict(d),"error":str(e)}
        rows.append(r)
        if (i+1)%6==0:
            save_json(output/"compact_results.json",rows);print(f"anchor {i+1}/{len(designs)}",flush=True)
    save_json(output/"run_status.json",{"status":"completed","count":len(rows),"scope":"quarter-period anchors; some explicitly fail the common research process assumption"})

if __name__=="__main__":main()
