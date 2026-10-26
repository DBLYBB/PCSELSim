"""Merge residual-verified finite pole pools at one geometry/reference/grid."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[_key]="1"
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from pcselsim.mode_identity import independent_eigenmode
from scripts.optimization.triangular_semiconductor_search import json_write,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--inputs",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    retained=[];values=[];fields=[];reference=None;provenance=[]
    for folder in args.inputs:
        row=json.loads((folder/"result.json").read_text());arrays=np.load(folder/"complex_fields.npz")
        if reference is None:reference=(row,{key:arrays[key] for key in ("coupling_m","x_um","y_um")})
        else:
            first,saved=reference
            if {k:v for k,v in row["three_hole_spec"].items() if k!="name"}!={k:v for k,v in first["three_hole_spec"].items() if k!="name"}:raise ValueError("Different physical geometry/period")
            if row["N"]!=first["N"] or row["D"]!=first["D"] or row["dz_nm"]!=first["dz_nm"]:raise ValueError("Different grid/truncation")
            if np.linalg.norm(arrays["coupling_m"]-saved["coupling_m"])>1e-10*np.linalg.norm(saved["coupling_m"]):raise ValueError("Different actual-reference C")
            if not np.array_equal(arrays["x_um"],saved["x_um"]) or not np.array_equal(arrays["y_um"],saved["y_um"]):raise ValueError("Different node coordinates")
        for rank,(value,field) in enumerate(zip(arrays["eigenvalues"],arrays["fields"])):
            vector=field.ravel()/np.linalg.norm(field)
            if not independent_eigenmode(value,vector,values,fields):continue
            record={**row["spectrum"][rank],"source_pool":str(folder),"source_rank":rank}
            retained.append(record);values.append(value);fields.append(vector.reshape(field.shape))
        provenance.append({"path":str(folder),"saved_field_poles":len(arrays["fields"]),"complete_scalar_poles":len(row["spectrum"])})
    order=np.argsort(np.imag(values),kind="stable");values=np.asarray(values)[order];fields=np.asarray(fields)[order]
    records=[retained[i] for i in order]
    for index,record in enumerate(records):record["spectral_index"]=index
    first,saved=reference
    result={"three_hole_spec":first["three_hole_spec"],"N":first["N"],"D":first["D"],"dz_nm":first["dz_nm"],
        "spectrum":records,"retained_poles":len(records),"g_th_cm":records[0]["g_th_cm"],"lowest":records[0],"second":records[1],
        "g_th_gap_fraction":records[1]["g_th_cm"]/records[0]["g_th_cm"]-1,"source_pools":provenance,
        "scope":"Union of complete-coarse-guided mesh continuation and explicit stripe-target challenges; not a complete finest-grid spectrum"}
    json_write(args.output/"result.json",result);write_csv(args.output/"all_poles.csv",records)
    np.savez_compressed(args.output/"complex_fields.npz",eigenvalues=values[:24],fields=fields[:24],**saved)
    print("Merged",len(records),"independent fields; lowest",records[0]["g_th_cm"],"next",records[1]["g_th_cm"],flush=True)


if __name__=="__main__":main()
