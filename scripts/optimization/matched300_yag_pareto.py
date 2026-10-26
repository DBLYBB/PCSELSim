"""Compact actual-winner records for completed, read-back-verified YAG cases."""
from matched300_yag_summary import ROOT, OUT, save, relative
import argparse
import json
from pathlib import Path


def compact_case(path, role, config_path):
    summary=json.loads((path/"summary.json").read_text())
    qa=json.loads((path/"readback_qa.json").read_text())
    if qa["status"]!="passed":raise ValueError(f"Unverified case: {path}")
    rows=json.loads((path/"closed_optical_modes.json").read_text())
    cold=sorted(rows,key=lambda r:r["required_material_gain_cm1"])
    pumps={}
    for pump in summary["pump_results"]:
        architecture=pump["architecture"];threshold=pump["threshold"]
        winners=[]
        for row in summary["actual_winners"][architecture]:
            winners.append(dict(mode=row["name"],wavelength_nm=row["wavelength_nm"],
                gain_cm1=row["required_material_gain_cm1"],
                beam={key:row["beam"][key] for key in ("ellipticity","center_to_peak","radial_anisotropy_l2","centroid_offset_deg","peak_offset_deg")},
                upward_fraction_of_threshold_loss=row["up_fraction"],
                far_x_key=row["name"]+"_far_x",far_y_key=row["name"]+"_far_y"))
        cw=[]
        for point in pump["cw"]:
            cw.append({key:point[key] for key in ("status","ratio","power_internal_W","active_modes") if key in point})
            if point["status"]=="completed":
                cw[-1].update(upward_output_W=sum(point["upward_W"].values()),
                    energy_closure_residual_W=point["energy_budget_W"]["energy_closure_residual_W"],
                    ncp_residual=point["ncp_residual"])
        pumps[architecture]=dict(first_internal_W=threshold["first_threshold_internal_W"],
            zero_photon_next_to_first_ratio=threshold["next_to_first_ratio"],
            zero_photon_branch_gap_fraction=(threshold["next_to_first_ratio"]-1
                if threshold["next_to_first_ratio"] is not None else None),
            actual_first_modes=winners,cw=cw)
    edge=pumps["edge_dual"]["actual_first_modes"][0]
    return dict(lattice=summary["lattice"],selection_status=role,
        nominal_aperture_um=[300,300],period_nm=summary["period_nm"],
        recommended_pump_architecture="edge_dual",actual_edge_first=edge,
        cold_gain_gap_fraction=cold[1]["required_material_gain_cm1"]/cold[0]["required_material_gain_cm1"]-1,
        cold_first_mode=cold[0]["name"],cold_second_mode=cold[1]["name"],
        pumps=pumps,numerical_settings=summary["settings"],
        independently_closed_modes=len(rows),failed_roots=summary["failed_roots"],
        summary_path=relative(path/"summary.json"),fields_path=relative(path/"closed_fields.npz"),
        config_path=config_path,geometry_path=relative(path/"actual_geometry_nm.json"),
        readback_qa_path=relative(path/"readback_qa.json"),axis_key="angle_deg")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",action="append",nargs=3,required=True,
        metavar=("PATH","SELECTION_STATUS","CONFIG_PATH"))
    parser.add_argument("--output",type=Path,default=OUT/"pareto.json")
    args=parser.parse_args()
    cases=[compact_case(Path(path),role,config) for path,role,config in args.case]
    save(args.output,dict(status="completed_cases_only",cases=cases,
        comparison_scope="Same nominal 300um square aperture and YAG stack/material/pump entrances. Different periods and motif geometry are optimized. Zero-photon branch gaps do not predict saturated turn-on or dynamic stability. Nominal constant cross sections; internal first-entry pump power; bounded1000-1060nm TE model.",
        internal_intensity_loss_cm1=.1,nominal_gain_ceiling_cm1=23.93265))
    print([(r["lattice"],r["selection_status"],r["actual_edge_first"]["gain_cm1"]) for r in cases])


if __name__=="__main__":main()
