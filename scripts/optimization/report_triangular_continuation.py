"""Combine saved triangular size/beam checks and three finite-grid finalists."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pcselsim.design_objectives import beam_acceptance
from pcselsim.run_output import prepare_output,write_provenance
from scripts.optimization.triangular_semiconductor_search import json_write,write_csv


def load(path):return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    output=prepare_output(args.output,ROOT)
    write_provenance(output,ROOT)
    (output/"executed_script.py").write_bytes(Path(__file__).read_bytes())
    base=ROOT/"results/current/optimization_20261021/triangular"
    paths=[base/p for p in ("final_report_v2/summary.json","beam_audit_v2/beam_audit.json",
        "size_scaling_v1/size_candidates.json","scaled_400um_finalist_v1/summary.json",
        "scaled_400um_beam_v2/beam_audit.json","scaled_400um_refinement_v1/numerical_sensitivity.json")]
    old,beams,sizes,scaled,scaled_beam,refine=map(load,paths)
    beam_map={r["name"]:r for r in beams+scaled_beam}
    finalists=old["finalists"]
    keys=("name","spec","N","D","dz_nm","g_th_cm","J_th_A_cm2","I_th_A","alpha_field_cm",
        "alpha_radiation_field_cm","eta_rad","field_loss_gap_cm","g_th_gap_fraction","Gamma","effective_index",
        "group_index","bragg_nm","frozen_CWT_pole_wavelength_nm","beam","ellipticity","center_to_peak",
        "radial_anisotropy_l2","fine_process_pass","research_process_pass","geometry","gap_nm","width_nm",
        "maximum_residual","maximum_energy_closure_per_m")
    new={k:scaled[k] for k in keys}
    checks={r["check"]:r for r in scaled["checks"]}
    new.update(combined_lowest_two_method_and_grid_sensitivity_cm=refine["combined_lowest_two_cm"],
        lowest_pole_ordering_resolved=refine["ordering_resolved_on_tested_discretizations"],
        beam_acceptance=beam_acceptance(scaled["beam"],require_sampling=True),radiation_fraction_pass=scaled["eta_rad"]>=.01,
        balanced_5percent_radiation_fraction_pass=scaled["eta_rad"]>=.05,
        ten_percent_material_gain_separation_pass=scaled["g_th_gap_fraction"]>=.1,
        complete_N7_discrete_spectrum_poles=checks["dense_N7"]["retained_poles"],
        complete_N7_lowest_g_th_cm=checks["dense_N7"]["g_th_cm"],
        N23_wide_shift_search_poles=checks["wide_N23"]["retained_poles"],
        fine_spectrum_scope="N35 local six-Gamma-neighborhood poles; N23 17 additional real shifts and complete coarse N7; not exhaustive N35 spectrum",
        robustness_N19=[{key:r[key] for key in ("check","g_th_cm","g_th_gap_fraction","ellipticity","eta_rad","fine_process_pass","research_process_pass")}
                         for r in scaled["checks"] if r["check"].startswith("etch")],
        limitations=old["finalists"][0]["limitations"]+["400-um aperture requires larger total injection current; no current spreading or thermal solve"])
    finalists.append(new)
    sources={r["name"]:base/"highmesh_retuned_v1"/r["name"]/"N35/complex_fields.npz" for r in finalists[:-1]}
    sources[new["name"]]=base/"scaled_400um_finalist_v1/N35/complex_fields.npz"
    roles={"local_0010_950nm":"small-area low-current candidate; tighter fabrication margin",
        "local_0067_950nm":"larger mode-separation and simple fabrication candidate",
        "local_0067_950nm_400um":"low-material-gain, simple-fabrication candidate; larger total current"}
    for row in finalists:
        row["engineering_role"]=roles[row["name"]]
        row["dense_angle_beam_audit"]=beam_map[row["name"]]
        row["internal_loss_only_gain_floor_cm"]=5/(row["Gamma"]*3.584/row["effective_index"])
        row["double_internal_loss_gain_cm"]=(2*row["alpha_field_cm"]+10)/(row["Gamma"]*3.584/row["effective_index"])
        row["double_internal_loss_gap_fraction"]=row["g_th_gap_fraction"]*row["g_th_cm"]/row["double_internal_loss_gain_cm"]
        source=sources[row["name"]]
        data=np.load(source)
        compact={k:data[k][:8] if k in ("fields","eigenvalues","all_radiation_x","all_radiation_y") else data[k] for k in data.files}
        np.savez_compressed(output/(row["name"]+"_complex_fields.npz"),**compact)
        row["complex_field_subset"]={"path":row["name"]+"_complex_fields.npz","pole_indices":list(range(8)),
            "source":source.relative_to(ROOT).as_posix(),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest()}
        dense_path=(base/"scaled_400um_beam_v2" if row["name"]==new["name"] else base/"beam_audit_v2")/(row["name"]+"_dense_beam.npz")
        beam=np.load(dense_path)
        angle=beam["angle_deg"]
        far=abs(beam["far_field_x"])**2+abs(beam["far_field_y"])**2
        far/=far.max()
        near=np.sum(abs(data["fields"][0])**2,axis=0)
        eig=data["eigenvalues"][:8]
        gains=(2*eig.imag/100+5)/(row["Gamma"]*3.584/row["effective_index"])
        fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
        axes[0].tripcolor(data["x_um"],data["y_um"],near/near.max(),shading="gouraud",cmap="inferno")
        axes[0].set(aspect="equal",xlabel="x (um)",ylabel="y (um)",title=row["name"]+": actual N35")
        axes[1].imshow(far,origin="lower",extent=(angle[0],angle[-1],angle[0],angle[-1]),cmap="inferno")
        axes[1].set(xlim=(-.5,.5),ylim=(-.5,.5),xlabel="theta x (deg)",ylabel="theta y (deg)",title=f"Ellipse {row['dense_angle_beam_audit']['beam']['ellipticity']:.3f}; bright center")
        axes[2].plot(np.arange(1,9),gains,"o-")
        axes[2].set(xlabel="Found pole, loss sorted",ylabel="Material gain threshold (cm$^{-1}$)",title=f"g={row['g_th_cm']:.2f}; gap={row['g_th_gap_fraction']:.1%}")
        axes[2].grid(alpha=.25)
        fig.savefig(output/(row["name"]+"_overview.png"),dpi=190);plt.close(fig)
    summary={"finalists":finalists,"new_work":{"finite_size_points":len(sizes),"distinct_aperture_sizes":8,
        "dense_angle_samples":301,"additional_finalist":"local_0067_950nm_400um",
        "size_figure":str((base/"size_scaling_v1/size_tradeoffs.png").relative_to(ROOT)),
        "findings":["Larger apertures lower the threshold and improve intensity roundness while decreasing mode separation and increasing conditional total current.",
        "The 400-um single-ellipse alternative approximately matches the small-area three-hole material gain, with generous fabrication gaps and larger current.",
        "C6 directional-dispersion value requires complete symmetry, non-degenerate analytic reciprocal bands; optimized broken-symmetry cells do not have C6 protection."]},
        "scope":"linear six-wave CWT engineering screening; no global optimum, exhaustive N35 spectrum, nonlinear single-mode or fabricated-device proof",
        "source_input_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    json_write(output/"summary.json",summary)
    flat=[]
    for r in finalists:
        extra=r["dense_angle_beam_audit"]["extra"]
        flat.append({k:r[k] for k in ("name","g_th_cm","J_th_A_cm2","I_th_A","g_th_gap_fraction","eta_rad","ellipticity","center_to_peak","radial_anisotropy_l2","gap_nm","width_nm","research_process_pass","combined_lowest_two_method_and_grid_sensitivity_cm","field_loss_gap_cm","lowest_pole_ordering_resolved")}|{"aperture":r["spec"]["aperture"],"half_size_um":r["spec"]["radius_um"],"FWHM_min_deg":extra["FWHM_diameter_min_deg"],"FWHM_max_deg":extra["FWHM_diameter_max_deg"],"outer_peak_dB":extra["outside_first_radial_mean_minimum_peak_dB"]})
    write_csv(output/"finalists.csv",flat)
    json_write(output/"run_status.json",{"status":"completed","finalists":len(finalists),"new_work_only":True})
    print(output,flush=True)


if __name__=="__main__":main()
