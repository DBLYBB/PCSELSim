"""Package audited actual-period three-hole PCSEL designs without PPT assets."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[_key]="1"
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from pcselsim.fabrication import TRIANGULAR_DIRECT,inclusion_boundary
from scripts.optimization.matched300_triangular_search import ThreeHole,candidate,geometry
from scripts.optimization.triangular_semiconductor_search import json_write,write_csv


def relative(path):return str(Path(path).resolve().relative_to(ROOT)).replace("\\","/")


def actual_geometry(spec):
    holes=candidate(spec).inclusions;rows=[]
    shares=(spec.main_share,(1-spec.main_share)*spec.satellite_balance,(1-spec.main_share)*(1-spec.satellite_balance))
    shapes=(spec.main_shape,spec.satellite1_shape,spec.satellite2_shape)
    centers=[(0.,0.)]+[tuple(d*spec.lattice_nm*np.asarray((np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a)))))
        for d,a in ((spec.distance1,spec.angle1_deg),(spec.distance2,spec.angle2_deg))]
    orientations=(spec.main_orientation_deg,spec.satellite1_orientation_deg,spec.satellite2_orientation_deg)
    for i,(h,share,shape) in enumerate(zip(holes,shares,shapes,strict=True)):
        row={"role":("central_main","satellite_1","satellite_2")[i],"shape":shape,"area_share":share,
            "area_nm2":h.fill_fraction*np.sqrt(3)/2*spec.lattice_nm**2,
            "center_cartesian_nm":list(centers[i]),"orientation_deg":orientations[i],
            "boundary_cartesian_nm":(inclusion_boundary(h,TRIANGULAR_DIRECT,samples=384)*spec.lattice_nm).tolist()}
        if hasattr(h,"center_fractional"):
            row.update(center_fractional=list(h.center_fractional),
                center_cartesian_nm=(TRIANGULAR_DIRECT@np.asarray(h.center_fractional)*spec.lattice_nm).tolist(),
                radii_nm=(np.asarray(h.radii_over_a)*spec.lattice_nm).tolist(),
                full_axes_nm=(2*np.asarray(h.radii_over_a)*spec.lattice_nm).tolist(),orientation_deg=h.angle_deg,
                exact_minimum_caliper_nm=2*min(h.radii_over_a)*spec.lattice_nm)
        if shape=="triangle":
            vertices=np.asarray(h.vertices_fractional)@TRIANGULAR_DIRECT.T*spec.lattice_nm
            row.update(rounding_corner_fraction=spec.rounding,triangle_samples_per_corner=spec.triangle_samples_per_corner,
                actual_polygon_vertices=len(h.vertices_fractional),aspect_parameter_applied=False,
                vertices_cartesian_nm=vertices.tolist(),
                bounding_box_nm=[[float(vertices[:,0].min()),float(vertices[:,0].max())],[float(vertices[:,1].min()),float(vertices[:,1].max())]],
                maximum_vertex_radius_nm=float(np.linalg.norm(vertices-np.asarray(centers[i]),axis=1).max()))
        if shape=="circle":row["orientation_has_physical_effect"]=False
        rows.append(row)
    return {"lattice_constant_nm":spec.lattice_nm,"nominal_aperture_um":[300,300],"aperture_shape":"square",
        "central_main_has_largest_area":bool(shares[0]>=max(shares[1:])),"holes":rows,"fabrication":geometry(spec),
        "contour_version":"equal_area_rounded_regular_triangle_polygon_v1",
        "triangle_samples_per_corner":spec.triangle_samples_per_corner,
        "aspect_semantics":"Only ellipse uses aspect; circle and rounded regular triangle ignore aspect."}


def package(base):
    base=base.resolve()
    cases={}
    for label in ("formal_isolated_baseline","formal_isolated_optimized","formal_low_threshold","formal_isolated_baseline_N39"):
        folder=base/label
        if not (folder/"summary.json").exists():continue
        s=json.loads((folder/"summary.json").read_text(encoding="utf-8"));spec=ThreeHole(**s["three_hole_spec"])
        geom=actual_geometry(spec);json_write(folder/"actual_geometry_nm.json",geom)
        grid=np.load(folder/"frequency_closed/frequency_closed_fields.npz")
        metrics={"nominal_size_um":[300,300],"sampled_node_span_um":[float(np.ptp(grid["x_um"])),float(np.ptp(grid["y_um"]))]}
        cases[label]={**s,"current_threshold_gap_fraction":s["second"]["I_th_A"]/s["lowest"]["I_th_A"]-1,
            "actual_geometry":geom,"grid_geometry":metrics,
            "summary_path":relative(folder/"summary.json"),"closed_fields_path":relative(folder/"frequency_closed/frequency_closed_fields.npz"),
            "seed_path":s.get("finite_seed_path",relative(folder/"final_seed/result.json")),"overview_path":relative(folder/"final_overview.png"),
            "polarization_path":relative(folder/"polarization.json"),"geometry_path":relative(folder/"actual_geometry_nm.json")}
    if not cases:raise RuntimeError("No completed formal candidates")
    eligible=[(name,r) for name,r in cases.items() if r["actual_target_pass"] and r["roots_closed"]==r["roots_requested"]
        and r["actual_geometry"]["fabrication"]["margin_pass"] and r["actual_geometry"]["central_main_has_largest_area"]
        and r["lowest"]["beam"]["ellipticity"]<=1.10 and r["lowest"]["beam"]["center_to_peak"]>=.90
        and r["lowest"]["beam"]["radial_anisotropy_l2"]<=.15]
    isolated=[(name,r) for name,r in eligible if r["g_th_gap_fraction"]>=.10]
    ranking_pool=[(name,r) for name,r in (isolated or eligible) if not(name=="formal_isolated_baseline" and "formal_isolated_baseline_N39" in cases)]
    numerical_best=min(ranking_pool,key=lambda item:item[1]["lowest"]["g_th_cm"])[0]
    primary="formal_isolated_baseline_N39" if any(name=="formal_isolated_baseline_N39" for name,r in isolated) else numerical_best
    stripe_file=base/"formal_stripe_challenge_N39/frequency_closed/summary.json"
    stripe=json.loads(stripe_file.read_text(encoding="utf-8")) if stripe_file.exists() else None
    invalidated=bool(stripe and stripe["lowest"] and stripe["lowest"]["g_th_cm"]<cases["formal_isolated_baseline_N39"]["lowest"]["g_th_cm"])
    scans={}
    for label in ("sobol1024","circle_grid567","local256","local_targeted192","highD_adaptive128","pareto_rescreen_N17"):
        file=base/label/"candidates.json"
        if not file.exists():continue
        rows=json.loads(file.read_text(encoding="utf-8"));counts={}
        for row in rows:counts[row["status"]]=counts.get(row["status"],0)+1
        scans[label]={"attempts":len(rows),"status_counts":counts,"scalar_csv_path":relative(base/label/"candidates.csv"),
            "input_specs_path":relative(base/label/"input_specs.json")}
    protocols={"nominal_aperture":"300 x 300 um square, common to all four material/lattice comparisons",
        "semiconductor_stack":"Unchanged Inoue Table-I layers, PC190 nm, GaAs110nm, spacer25nm, 3x10nm InGaAs QWs and 3x20nm barriers, common claddings",
        "internal_intensity_loss_cm-1":5,"actual_lowest_pole_target_nm":950.,"target_tolerance_nm":.05,
        "threshold_gain_definition":"(2*passive field loss + 5/cm)/(Gamma*n_active/n_eff)",
        "gap_definition":"g_th_gap_fraction is material threshold-gain separation; current_threshold_gap_fraction uses the common nonlinear carrier-gain law and is a distinct quantity.",
        "eta_rad_definition":"2*modeled local radiative amplitude loss / (2*passive finite-cavity amplitude loss + internal intensity loss); internal5/cm is included in the denominator. A threshold-loss fraction, not pump absorption or wall-plug efficiency.",
        "beam_metric_domain":"Same +/-1 degree vector far-field window; circularity denotes intensity, not circular polarization",
        "manufacturing_scope":"Assumed common gap40nm/span50nm, edge3nm/placement3nm, corner5nm; margin checks are not foundry qualification"}
    result={"primary_case":primary,"primary":cases[primary],"numerical_best_isolated_case":numerical_best,
        "low_threshold_pareto_case":min(eligible,key=lambda item:item[1]["lowest"]["g_th_cm"])[0],
        "spectral_isolation_status":"provisional_pending_full_coarse_QZ_guided_fine_challenge" if not (base/"formal_QZ_guided_N39/frequency_closed/summary.json").exists() else "coarse_QZ_guided_fine_challenge_available_review_required",
        "recommendation_note":"Primary prioritizes verified finer-mesh circularity and process span margin; a lower-g isolated numerical design is listed separately if its span budget is nearly exhausted.",
        "cases":cases,"protocol":protocols,"scans":scans,
        "audit_paths":{p.name:relative(p) for p in base.iterdir() if p.is_dir() and ("audit" in p.name or "process" in p.name)},
        "scope":"Actual finite lateral poles independently real-frequency-closed at fixed redesigned period; local spectral search with wide real shifts, not exhaustive fine-grid Maxwell, dynamic single-mode, experimental polarization or global optimum proof."}
    if invalidated:
        result.update(primary_case=None,primary=None,qualification_status="previous_bright_candidate_disqualified_by_lower_nonround_stripe_poles; new designs under optimization",
            numerical_best_isolated_case=None,
            recommendation_note="No main recommendation is currently certified. The previous selected bright mode was contradicted by lower nonround stripe poles; a lower-threshold alternative also awaits expanded spectral challenges.",
            previous_primary_case=primary,previous_primary=cases[primary],
            decisive_counterexample={"summary_path":relative(stripe_file),"lowest":stripe["lowest"],"second":stripe["second"]})
    json_write(base/"final_summary_full.json",result)
    def compact_case(record):
        keep=("three_hole_spec","N","D","dz_nm","roots_closed","roots_requested","g_th_gap_fraction","current_threshold_gap_fraction",
            "actual_target_wavelength_nm","actual_target_tolerance_nm","actual_target_pass","grid_geometry",
            "summary_path","closed_fields_path","seed_path","overview_path","polarization_path","geometry_path")
        value={k:record[k] for k in keep if k in record}
        for k in ("lowest","second"):
            value[k]={key:item for key,item in record[k].items() if key not in ("trace",)} if record.get(k) else None
        value["actual_geometry"]={k:v for k,v in record["actual_geometry"].items() if k!="holes"}
        value["eligible"] = any(name==n for n,_ in eligible for name,r in cases.items() if r is record)
        return value
    compact_cases={name:compact_case(r) for name,r in cases.items()}
    if invalidated:
        for name in ("formal_isolated_baseline","formal_isolated_baseline_N39"):
            compact_cases[name].update(eligible=False,qualification_status="disqualified_by_lower_nonround_stripe_poles",selected_bright_mode_is_not_actual_lowest=True)
    compact_result={**result,"cases":compact_cases,"primary":None if invalidated else compact_cases[primary],
        "full_summary_path":relative(base/"final_summary_full.json"),
        "portable_config_path":"configs/optimized_matched300_20261021/triangular_finalists.json",
        "retention_manifest_path":relative(base/"retention_manifest.json")}
    if invalidated:
        compact_result["previous_primary"]=compact_cases[primary]
        compact_result["decisive_counterexample"]={"summary_path":relative(stripe_file),
            "lowest":{k:v for k,v in stripe["lowest"].items() if k!="trace"},
            "second":{k:v for k,v in stripe["second"].items() if k!="trace"}}
    json_write(base/"final_summary.json",compact_result)
    config=ROOT/"configs/optimized_matched300_20261021/triangular_finalists.json";config.parent.mkdir(exist_ok=True,parents=True)
    portable=[(name,r) for name,r in eligible if not(name=="formal_isolated_baseline" and "formal_isolated_baseline_N39" in cases)]
    json_write(config,{"protocol":protocols,"primary_case":None if invalidated else primary,
        "qualification_status":"No currently certified recommendation; previous main candidate is contradicted by lower stripe modes" if invalidated else "Selected finite-spectrum candidate, subject to stated audit scope",
        "designs":[{**r["three_hole_spec"],"name":"triangular_"+name.removeprefix("formal_")} for name,r in portable],
        "case_paths":{"triangular_"+name.removeprefix("formal_"):r["summary_path"] for name,r in portable},
        "rejected_counterexamples":{name:r["summary_path"] for name,r in cases.items() if not any(n==name for n,_ in eligible)},
        "actual_geometry_paths":{name:r["geometry_path"] for name,r in cases.items()},
        "reproduction":"Use ThreeHole and candidate from scripts/optimization/matched300_triangular_search.py; fixed actual period is authoritative, not the Bragg950 screening period."})
    write_csv(base/"final_comparison.csv",[{"case":name,
        "status":"eligible" if any(n==name for n,_ in eligible) else "rejected_actual_wavelength_or_nonround_lowest_mode",
        "N":r["N"],"D":r["D"],"dz_nm":r["dz_nm"],
        "period_nm":r["three_hole_spec"]["lattice_nm"],"g1_cm":r["lowest"]["g_th_cm"],"g2_cm":r["second"]["g_th_cm"],
        "gain_gap_fraction":r["g_th_gap_fraction"],"current_threshold_gap_fraction":r["current_threshold_gap_fraction"],
        "wavelength_nm":r["lowest"]["wavelength_nm"],"I_th_A":r["lowest"]["I_th_A"],
        "ellipse":r["lowest"]["beam"]["ellipticity"],"center_to_peak":r["lowest"]["beam"]["center_to_peak"],
        "radial_anisotropy":r["lowest"]["beam"]["radial_anisotropy_l2"],"eta_rad":r["lowest"]["eta_rad"],
        "actual_target_pass":r["actual_target_pass"],"nominal_gap_nm":r["actual_geometry"]["fabrication"]["minimum_periodic_gap_nm"],
        "nominal_span_nm":r["actual_geometry"]["fabrication"]["minimum_caliper_nm"]} for name,r in cases.items()])
    print("primary_disqualified" if invalidated else primary,relative(config),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True);a=p.parse_args();package(a.input)


if __name__=="__main__":main()
