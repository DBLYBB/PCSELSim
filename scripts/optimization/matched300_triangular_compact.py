"""Retain final Jones fields and scalar search evidence; remove bulky trials."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scripts.optimization.triangular_semiconductor_search import json_write


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(base):
    base=base.resolve()
    if base!=ROOT/"results/current/matched300_20261021/triangular":raise ValueError("Unexpected cleanup root")
    def checked(path):
        result=path.resolve()
        if not result.is_relative_to(base):raise ValueError("Path escapes explicit triangular result root")
        return result
    before=sum(p.stat().st_size for p in base.rglob("*") if p.is_file());deleted=[];trimmed=[]
    finals=("formal_isolated_baseline","formal_isolated_baseline_N39","formal_low_threshold","formal_isolated_optimized")
    closed=[base/n/"frequency_closed/frequency_closed_fields.npz" for n in finals]
    fingerprints={str(p.relative_to(base)):digest(p) for p in closed}
    seeds={base/"formal_isolated_baseline/final_seed/complex_fields.npz":24,
        base/"formal_isolated_N_audit/N39_D32_dz1/complex_fields.npz":24,
        base/"formal_low_threshold/final_seed/complex_fields.npz":24,
        base/"formal_isolated_optimized/final_seed/complex_fields.npz":12}
    keep=set(closed)|set(seeds)|{base/n/"stokes_maps.npz" for n in finals}
    for p in list(base.rglob("*.npz")):
        p=checked(p)
        if p not in keep:
            deleted.append({"path":str(p.relative_to(base)),"bytes":p.stat().st_size,"sha256":digest(p)})
            p.unlink()
    for p,count in seeds.items():
        checked(p);old_bytes=p.stat().st_size
        with np.load(p) as z:
            original=len(z["eigenvalues"]);arrays={key:z[key].copy() for key in z.files}
        for key in ("eigenvalues","fields","all_radiation_x","all_radiation_y"):
            if key in arrays:arrays[key]=arrays[key][:count]
        temp=p.with_name("retained_low_poles.tmp.npz");checked(temp)
        np.savez_compressed(temp,**arrays);os.replace(temp,p)
        trimmed.append({"path":str(p.relative_to(base)),"original_poles":original,"retained_lowest_field_indices":list(range(min(count,original))),
            "original_bytes":old_bytes,"retained_bytes":p.stat().st_size,"note":"All scalar poles/residuals remain in result.json; only high-loss complex field arrays are discarded."})
    for p in list(base.rglob("*.png")):
        p=checked(p);rel=p.relative_to(base)
        allowed=(len(rel.parts)==2 and rel.parts[0] in finals and rel.name in ("final_overview.png","polarization_stokes.png")) or (
            len(rel.parts)==3 and rel.parts[0] in finals and rel.parts[1]=="frequency_closed" and rel.name in ("lattice.png","closed_modes.png"))
        if not allowed:
            deleted.append({"path":str(rel),"bytes":p.stat().st_size,"sha256":digest(p)});p.unlink()
    def strip(value):
        if isinstance(value,dict):return {k:strip(v) for k,v in value.items() if k!="spectrum"}
        if isinstance(value,list):return [strip(v) for v in value]
        return value
    for name in ("candidates.json","leaderboard.json","run_status.json"):
        for p in base.rglob(name):
            checked(p);value=json.loads(p.read_text(encoding="utf-8"));json_write(p,strip(value))
    after_fingerprints={str(p.relative_to(base)):digest(p) for p in closed}
    if fingerprints!=after_fingerprints:raise RuntimeError("Final closed complex fields changed")
    after=sum(p.stat().st_size for p in base.rglob("*") if p.is_file())
    result={"explicit_root":str(base),"bytes_before":before,"bytes_after":after,"bytes_released":before-after,
        "deleted_files":deleted,"trimmed_seed_arrays":trimmed,"closed_field_sha256_before":fingerprints,
        "closed_field_sha256_after":after_fingerprints,"all_final_closed_fields_unchanged":True,
        "retention":"Each accepted/rejected final case retains all six independently closed poles; raw fine seeds retain 24 lowest fields (12 for rejected design) and complete scalar spectra. Exploratory catalogues retain exact input parameters, beam metrics and scalar CSV/JSON."}
    json_write(base/"retention_manifest.json",result)
    print(json.dumps({k:result[k] for k in ("bytes_before","bytes_after","bytes_released","all_final_closed_fields_unchanged")}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True);a=p.parse_args();compact(a.input)


if __name__=="__main__":main()
