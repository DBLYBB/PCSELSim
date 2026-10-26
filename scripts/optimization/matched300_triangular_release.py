"""Create the compact, read-back-verified triangular PCSEL delivery index."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[_key]="1"
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from pcselsim.design_objectives import beam_metrics
from pcselsim.polarization import polarization_metrics
from scripts.optimization.matched300_triangular_search import ThreeHole
from scripts.optimization.matched300_triangular_package import actual_geometry
from scripts.optimization.triangular_semiconductor_search import json_write,write_csv


def relative(path):return str(Path(path).resolve().relative_to(ROOT)).replace("\\","/")


def case_record(folder):
    folder=folder.resolve();summary=json.loads((folder/"summary.json").read_text())
    geometry=actual_geometry(ThreeHole(**summary["three_hole_spec"]));json_write(folder/"actual_geometry_nm.json",geometry)
    arrays=np.load(folder/"frequency_closed/frequency_closed_fields.npz")
    rank=summary["lowest"]["seed_rank"];fx,fy=arrays[f"root_{rank}_far_x"],arrays[f"root_{rank}_far_y"];axis=arrays["angle_deg"]
    metrics=beam_metrics(axis,fx,fy,requested_view_deg=1.,source_nyquist_deg=summary["lowest"]["beam"].get("source_nyquist_deg"))
    pol=polarization_metrics(axis,fx,fy,core_radius_deg=.2);json_write(folder/"polarization.json",pol)
    numeric=[]
    for key in arrays.files:
        item=arrays[key]
        if np.issubdtype(item.dtype,np.number):
            if not np.isfinite(item).all():raise ValueError(f"Nonfinite retained array {folder}/{key}")
            numeric.append({"key":key,"shape":list(item.shape)})
    for key in ("ellipticity","center_to_peak","radial_anisotropy_l2"):
        if abs(metrics[key]-summary["lowest"]["beam"][key])>1e-10:raise ValueError(f"Beam readback failed: {key}")
    if abs(summary["lowest"]["frequency_residual_per_m"])>.05:raise ValueError("Actual-frequency residual exceeds .05/m")
    if summary["lowest"]["eigen_residual"]>2e-6:raise ValueError("Original finite pencil residual failed")
    if not geometry["central_main_has_largest_area"] or not geometry["fabrication"]["margin_pass"]:raise ValueError("Final geometry fails common rules")
    mode_keys={"fields":f"root_{rank}_fields","coupling_m":f"root_{rank}_coupling_m","eigenvalue_per_m":f"root_{rank}_eigenvalue_per_m",
        "radiation_x":f"root_{rank}_radiation_x","radiation_y":f"root_{rank}_radiation_y",
        "far_x":f"root_{rank}_far_x","far_y":f"root_{rank}_far_y","angle_deg":"angle_deg","x_um":"x_um","y_um":"y_um"}
    compact={key:summary[key] for key in ("three_hole_spec","N","D","dz_nm","roots_closed","roots_requested","g_th_gap_fraction")}
    for key in ("lowest","second"):
        compact[key]={k:v for k,v in summary[key].items() if k!="trace"} if summary.get(key) else None
    compact.update(actual_target_wavelength_nm=950.,actual_target_tolerance_nm=.05,
        actual_target_pass=abs(compact["lowest"]["wavelength_nm"]-950)<=.05,
        current_threshold_gap_fraction=compact["second"]["I_th_A"]/compact["lowest"]["I_th_A"]-1 if compact["second"] else None,
        summary_path=relative(folder/"summary.json"),closed_fields_path=relative(folder/"frequency_closed/frequency_closed_fields.npz"),
        geometry_path=relative(folder/"actual_geometry_nm.json"),polarization_path=relative(folder/"polarization.json"),
        overview_path=relative(folder/"final_overview.png"),primary_array_keys=mode_keys,
        actual_geometry={k:v for k,v in geometry.items() if k!="holes"},
        grid_geometry={"nominal_size_um":[300,300],"sampled_node_span_um":[float(np.ptp(arrays["x_um"])),float(np.ptp(arrays["y_um"]))]})
    readback={"path":relative(folder),"all_numeric_arrays_finite":True,"arrays":numeric,"beam_reconstructed_from_Jones":metrics,
        "original_pencil_relative_residual":summary["lowest"]["eigen_residual"],"actual_frequency_residual_per_m":summary["lowest"]["frequency_residual_per_m"]}
    return compact,readback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True);parser.add_argument("--primary",type=Path,required=True)
    parser.add_argument("--pareto",type=Path);parser.add_argument("--coarse",type=Path,required=True)
    parser.add_argument("--guided",type=Path,required=True);parser.add_argument("--audit",type=Path,nargs="*",default=[])
    args=parser.parse_args();base=args.base.resolve();primary,readback=case_record(args.primary);cases={args.primary.name:primary};checks=[readback]
    if not (args.coarse/"complete_coarse_poles.json").exists() or not (args.guided/"result.json").exists():
        raise ValueError("Complete coarse and guided fine spectral evidence is required before creating the canonical recommendation")
    coarse_summary=json.loads((args.coarse/"summary.json").read_text())
    if coarse_summary["retained_passive_poles"]!=coarse_summary["matrix_dimension"]:
        raise ValueError("The retained coarse spectrum is not the full matrix dimension; review filtering explicitly")
    guided=json.loads((args.guided/"result.json").read_text())
    actual_spec={k:v for k,v in primary["three_hole_spec"].items() if k!="name"}
    guided_spec={k:v for k,v in guided["three_hole_spec"].items() if k!="name"}
    if actual_spec!=guided_spec:raise ValueError("Guided spectrum is from a different physical geometry or period")
    coarse_plan=json.loads((args.coarse/"run_plan.json").read_text())
    if {k:v for k,v in coarse_plan["three_hole_spec"].items() if k!="name"}!=actual_spec:
        raise ValueError("Complete coarse spectrum is from a different physical geometry or period")
    if abs(coarse_plan["reference_wavelength_nm"]-guided["reference_wavelength_nm"])>1e-9:
        raise ValueError("Coarse and guided fine actual references differ")
    if primary["roots_closed"]!=primary["roots_requested"]:raise ValueError("Requested final competing poles have not all frequency-closed")
    if not primary["actual_target_pass"]:raise ValueError("Final lowest pole is outside 950 +/- .05 nm")
    if primary["lowest"]["beam"]["ellipticity"]>1.10 or primary["lowest"]["beam"]["center_to_peak"]<.90:raise ValueError("Final lowest pole is not bright and round")
    if args.pareto:
        other,check=case_record(args.pareto);other["qualification_status"]="More circular tracked-mode Pareto; its complete-spectrum scope is separate from primary"
        cases[args.pareto.name]=other;checks.append(check)
    protocol={"nominal_aperture_um":[300,300],"aperture":"square","semiconductor_stack":"Unchanged common Inoue Table-I complete 190nm-PC stack",
        "internal_intensity_loss_cm-1":5,"actual_lowest_pole_target_nm":950.,"target_tolerance_nm":.05,
        "gain_threshold":"(2*passive field loss + internal intensity loss)/(Gamma*n_active/n_eff)",
        "gap_definition":"Material threshold-gain separation; current gap uses a nonlinear gain law and is separately reported",
        "eta_rad_definition":"2*local radiative amplitude loss/(2*passive finite amplitude loss+internal intensity loss). Includes internal5/cm in denominator; not wall-plug or pump efficiency",
        "beam":"Complex Jones intensity |Ex|^2+|Ey|^2 in the common +/-1 degree window; intensity circularity is distinct from circular polarization",
        "fabrication":"Assumed gap40/span50/edge3/placement3/corner5nm; model rule, not foundry certification",
        "contour":"equal_area_rounded_regular_triangle_polygon_v1, samples_per_corner recorded in three_hole_spec; aspect applies only to ellipse"}
    scans={};input_catalogues={}
    for folder in base.iterdir():
        file=folder/"candidates.json"
        if not file.exists():continue
        if folder.name.endswith("_parents"):
            input_catalogues[folder.name]=relative(file);continue
        rows=json.loads(file.read_text());statuses={}
        for row in rows:statuses[row.get("status","unknown")]=statuses.get(row.get("status","unknown"),0)+1
        scans[folder.name]={"attempts":len(rows),"status_counts":statuses,"parameters":relative(file)}
    broad_names=("sobol1024","circle_grid567","local256","local_targeted192","highD_adaptive128")
    fine_names=[name for name in scans if name.startswith("stripe_safe_actual_")]
    workload={"broad_parameter_attempts":sum(scans[name]["attempts"] for name in broad_names),
        "broad_finite_solved":sum(scans[name]["status_counts"].get("finite_solved",0) for name in broad_names),
        "N17_pareto_rescreen_attempts":scans.get("pareto_rescreen_N17",{}).get("attempts",0),
        "fine_N65_local_attempts":sum(scans[name]["attempts"] for name in fine_names),
        "fine_N65_completed_branch_follows":sum(scans[name]["status_counts"].get("completed",0) for name in fine_names),
        "interpretation":"Counts are actual attempted evaluations/reruns, not unique physical geometries. Copied *_parents input catalogues are excluded. Fine branch follows are not complete spectra; validation/timing catalogues are listed separately in scans."}
    source_paths=list((ROOT/"scripts/optimization").glob("matched300_triangular_*.py"))
    source_paths.extend(ROOT/"src/pcselsim"/name for name in ("triangular_six_wave.py","triangular_finite.py","finite_discretization.py","geometry.py","vertical.py","polarization.py"))
    sources={relative(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    history_path=base/"source_provenance_history.json"
    history=json.loads(history_path.read_text()) if history_path.exists() else {"snapshots":[]}
    old_path=base/"final_summary.json"
    if old_path.exists():
        previous=json.loads(old_path.read_text()).get("source_sha256",{})
        if previous and previous!=sources and not any(row["source_sha256"]==previous for row in history["snapshots"]):
            changed=[name for name in previous if previous[name]!=sources.get(name)]
            history["snapshots"].append({"role":"previous_packaging_snapshot","archived_utc":datetime.now(timezone.utc).isoformat(),
                "source_sha256":previous,"changed_paths_at_next_release":changed,
                "interpretation":"Archived exactly from the previous canonical packaging; not asserted to be a complete run-time source manifest. Subsequent control_close edits only clarify predictor seed-reference metadata; release edits only package catalogues, retention and provenance. Final numerical arrays are unchanged; core sources are independently rebuilt and checked."})
    history["current_release_source_sha256"]=sources
    json_write(history_path,history)
    result={"three_hole_spec":primary["three_hole_spec"],"N":primary["N"],"D":primary["D"],"dz_nm":primary["dz_nm"],
        "primary_case":args.primary.name,"primary":primary,"cases":cases,"protocol":protocol,
        "qualification_status":"Lowest bright-round candidate found in the stated finite-spectrum and mesh challenges; not global or dynamic-single-mode proof",
        "coarse_complete_spectrum_path":relative(args.coarse),"coarse_guided_fine_spectrum_path":relative(args.guided),
        "audit_paths":{p.name:relative(p) for p in args.audit},"scans":scans,"search_workload":workload,
        "copied_input_catalogue_paths":input_catalogues,"source_sha256":sources,"source_sha256_at_release":sources,
        "source_provenance_history_path":relative(history_path),
        "source_provenance_note":"source_sha256 is the current-release compatibility alias. Previous packaging hashes remain in the linked history; final fields were not recomputed for packaging-only metadata edits.",
        "decisive_previous_counterexample_path":relative(base/"formal_stripe_challenge_N39/frequency_closed/summary.json"),
        "scope":"Complete N13 coarse spectrum guides complex-shift N39 finite poles; finer-grid continuation and explicit stripe challenges are not an exhaustive finest-grid spectrum. No nonlinear, experimental, full-Maxwell or all-band single-mode guarantee."}
    config=ROOT/"configs/optimized_matched300_20261021/triangular_finalists.json"
    result["portable_config_path"]=relative(config);result["readback_validation_path"]=relative(base/"readback_validation_final.json")
    json_write(base/"readback_validation_final.json",{"cases":checks,"source_sha256":sources})
    json_write(base/"final_summary.json",result)
    json_write(config,{"primary_case":args.primary.name,"protocol":protocol,"designs":[primary["three_hole_spec"]],
        "actual_geometry_paths":{args.primary.name:primary["geometry_path"]},"primary_fields":primary["closed_fields_path"],
        "primary_array_keys":primary["primary_array_keys"],"source_sha256":sources,"source_sha256_at_release":sources,
        "source_provenance_history_path":relative(history_path),
        "reproduction":"ThreeHole/candidate define fractional shapes; actual period and explicit contour resolution are authoritative."})
    write_csv(base/"final_comparison.csv",[{"case":name,"N":r["N"],"D":r["D"],"dz_nm":r["dz_nm"],"period_nm":r["three_hole_spec"]["lattice_nm"],
        "g1_cm":r["lowest"]["g_th_cm"],"g2_cm":r["second"]["g_th_cm"],"gain_gap_fraction":r["g_th_gap_fraction"],
        "current_threshold_gap_fraction":r["current_threshold_gap_fraction"],"wavelength_nm":r["lowest"]["wavelength_nm"],
        "ellipse":r["lowest"]["beam"]["ellipticity"],"center_to_peak":r["lowest"]["beam"]["center_to_peak"],
        "radial_anisotropy":r["lowest"]["beam"]["radial_anisotropy_l2"],"eta_rad":r["lowest"]["eta_rad"]} for name,r in cases.items()])
    print("Packaged",args.primary.name,primary["lowest"]["g_th_cm"],primary["lowest"]["beam"]["ellipticity"],flush=True)


if __name__=="__main__":main()
