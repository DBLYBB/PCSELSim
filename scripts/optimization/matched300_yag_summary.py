"""Assemble portable same-stack YAG finals and their compact root manifest."""
from matched300_yag_search import *
from collections import Counter


def relative(path):
    p=Path(path).resolve()
    try:return p.relative_to(ROOT).as_posix()
    except ValueError:return str(p)


def scan_statistics():
    logs=[];by_lattice={}
    for p in sorted(OUT.rglob("*.jsonl")):
        rows=[json.loads(line) for line in p.read_text(encoding="utf8").splitlines() if line.strip()]
        lattice=rows[0]["lattice"] if rows else "unknown"
        completed=[r for r in rows if r.get("lowest")]
        rejected=Counter(r.get("status","unknown") for r in rows)
        design_keys=sorted({k for r in rows for k in r.get("design",{})})
        ranges={}
        for k in design_keys:
            values=[r["design"][k] for r in rows if k in r.get("design",{})]
            if all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in values):ranges[k]=[min(values),max(values)]
            elif k.endswith("shape"):ranges[k]=sorted(set(values))
        unreachable=[r for r in completed if r["lowest"]["required_material_gain_cm1"]>=23.93265]
        near=[r for r in completed if r["lowest"]["beam"]["ellipticity"]<=1.05 and r["lowest"]["beam"]["center_to_peak"]>=.98]
        item=dict(path=relative(p),lattice=lattice,evaluations=len(rows),status_counts=dict(rejected),
            reachable_frozen_candidates=len(completed)-len(unreachable),nominal_gain_ceiling_exceeded=len(unreachable),
            frozen_bright_round_candidates=len(near),parameter_ranges=ranges,
            run_completed=(p.parent.name!="highfill_local" or lattice!="triangular"))
        if not item["run_completed"]:item["interruption_reason"]="45 early trials retained; old smooth selector replaced by bounded multi-pole worker with 192 completed trials"
        logs.append(item);by_lattice[lattice]=by_lattice.get(lattice,0)+len(rows)
    return dict(scalar_logs=logs,trial_evaluations_by_lattice=by_lattice,
        total_trial_evaluations=sum(by_lattice.values()),
        counting="Every logged evaluation, including repeated anchors, rejected fabrication and explicit failures. Additional accuracy/census/contour/tolerance trials are separate, not inflated into independent broad-search geometries.")


def case_record(path, config_filename=None):
    summary=json.loads((path/"summary.json").read_text(encoding="utf8"))
    qa=json.loads((path/"readback_qa.json").read_text(encoding="utf8"))
    if qa["status"]!="passed":raise ValueError(f"Readback QA did not pass: {path}")
    row=summary["actual_winners"]["edge_dual"][0]
    lattice=summary["lattice"];name=row["name"]
    edge,top=summary["pump_results"]
    rows=json.loads((path/"closed_optical_modes.json").read_text())
    material_sorted=sorted(r["required_material_gain_cm1"] for r in rows)
    geometry_path=path/"actual_geometry_nm.json"
    case=dict(id=f"yag_{lattice}",label=f"Yb:YAG {lattice}",material="nominal 10 at% Yb:YAG",lattice=lattice,
        nominal_aperture_um=[300,300],target_wavelength_nm=1030.,wavelength_nm=row["wavelength_nm"],
        period_nm=summary["period_nm"],g_th_cm_inverse=row["required_material_gain_cm1"],
        internal_edge_pump_W=edge["threshold"]["first_threshold_internal_W"],
        internal_top_pump_W=top["threshold"]["first_threshold_internal_W"],
        unlased_edge_branch_gap_fraction=edge["threshold"]["next_to_first_ratio"]-1 if edge["threshold"]["next_to_first_ratio"] is not None else None,
        unlased_top_branch_gap_fraction=top["threshold"]["next_to_first_ratio"]-1 if top["threshold"]["next_to_first_ratio"] is not None else None,
        relative_gain_gap=material_sorted[1]/material_sorted[0]-1 if len(material_sorted)>1 else None,
        radiative_fraction_of_threshold_power_loss=(row["up_power_loss_per_m"]+row["down_power_loss_per_m"])/row["total_power_loss_per_m"],
        useful_upward_fraction_of_threshold_power_loss=row["up_fraction"],
        up_power_loss_per_m=row["up_power_loss_per_m"],down_power_loss_per_m=row["down_power_loss_per_m"],
        boundary_transport_power_loss_per_m=row["boundary_transport_power_loss_per_m"],internal_intensity_loss_cm1=.1,
        fields_path=relative(path/"closed_fields.npz"),axis_key="angle_deg",far_x_key=name+"_far_x",far_y_key=name+"_far_y",
        field_key=name+"_field",coupling_key=name+"_C",upward_operator_key=name+"_Lup",downward_operator_key=name+"_Ldown",
        source_nyquist_deg=row["beam"]["source_nyquist_deg"],reported_beam=row["beam"],polarization=row["polarization"],
        geometry_path=relative(geometry_path),summary_path=relative(path/"summary.json"),qa_path=relative(path/"readback_qa.json"),
        mode_name=name,actual_winners=summary["actual_winners"],closed_mode_count=summary["included_closed_modes"],
        failed_closed_roots=summary["failed_roots"],common_analysis_band_nm=[1000,1060],
        caveats="Conditional nominal-material result and common INTERNAL first-entry pump inputs; edge_dual is the optimization objective. Top winner is separately recorded. Unlased crossing gap is not saturated turn-on or dynamic single-mode stability. Bounded near-1030 TE model, not an all-spectrum Maxwell proof.")
    cfg=dict(schema="matched300_yag_v1",material_protocol=json.loads((ROOT/"configs/optimized_matched300_20261021/yag_protocol.json").read_text()),
        lattice=lattice,nominal_aperture_um=[300,300],design=summary["design"],physical_period_nm=summary["period_nm"],
        actual_geometry=json.loads(geometry_path.read_text()),numerical_settings=summary["settings"],
        selected_actual_edge_mode=name,actual_edge_wavelength_nm=row["wavelength_nm"],
        declared_scope=summary["scope"],source_case=relative(path))
    config_path=ROOT/"configs/optimized_matched300_20261021"/(config_filename or f"yag_{lattice}_final.json")
    save(config_path,cfg);case["config_path"]=relative(config_path)
    return case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--square",type=Path,required=True);p.add_argument("--triangular",type=Path,required=True)
    a=p.parse_args()
    protocol=json.loads((ROOT/"configs/optimized_matched300_20261021/yag_protocol.json").read_text());protocol["status"]="completed_with_bounded_model_scope"
    save(ROOT/"configs/optimized_matched300_20261021/yag_protocol.json",protocol)
    cases=[case_record(a.square),case_record(a.triangular)]
    statistics=scan_statistics();save(OUT/"scan_statistics.json",statistics)
    summary=dict(status="completed",cases=cases,primary={r["lattice"]:r for r in cases},
        pareto_note="Retained exploratory low-threshold/nonround/near-cutoff negatives are scalar evidence, not certified alternatives. No required 10% gap is imposed.",
        scan_statistics=statistics,protocol_path="configs/optimized_matched300_20261021/yag_protocol.json",
        nominal_gain_ceiling_cm1=23.93265,common_analysis_band_nm=[1000,1060],
        all_cavities_nominal_aperture_um=[300,300],internal_intensity_loss_cm1=.1,
        output_caveat="Normalized beam shape does not imply high useful power; side loss budgets and shared-inversion CW outputs are retained in linked summaries.")
    save(OUT/"final_summary.json",summary)
    print([(r["id"],r["internal_edge_pump_W"],r["reported_beam"]["ellipticity"]) for r in cases],flush=True)


if __name__=="__main__":main()
