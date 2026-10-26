"""Assemble the completed deterministic triangular search without re-solving it."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pcselsim.design_objectives import beam_acceptance
from pcselsim.fabrication import pareto_mask
from pcselsim.run_output import prepare_output, write_provenance
from scripts.optimization.triangular_semiconductor_search import json_write, write_csv


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT/"results/current/optimization_20261021/triangular")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    base = args.base.resolve()
    out = prepare_output(args.output or base/"final_report_v1", ROOT)
    write_provenance(out, ROOT)
    (out/"executed_script.py").write_bytes(Path(__file__).read_bytes())
    phases = {}
    source_hashes = {}
    for directory in sorted(base.iterdir()):
        candidate_file = directory/"candidates.json"
        if candidate_file.exists():
            phases[directory.name] = load(candidate_file)
            source_hashes[candidate_file.relative_to(ROOT).as_posix()] = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    final = phases["highmesh_retuned_v1"]
    validated = {r["name"]: r for r in phases["validated_retuned_v1"]}
    robustness = {r["name"]: r for r in phases["robustness_retuned_v1"]}
    coarse = {r["name"]: r for r in phases["fullcoarse_retuned_v1"]}
    tuned_wide = {r["name"]: r for r in phases["retuned_finalists_v1"]}
    finalists = []
    for row in final:
        name = row["name"]
        lower = validated[name]
        method_sensitivity = sum(s["D8_D10_cm"]+s["dz3_dz2_cm"]
                                 for s in lower["lowest_two_numerical_sensitivities"])
        uncertainty = row["combined_lowest_two_grid_sensitivity_cm"]+method_sensitivity
        accepted = beam_acceptance(row["beam"], require_sampling=True)
        robust = robustness[name]["perturbation_trials"]
        pole_info = [{"perturbation": trial["perturbation"], "g_th_cm": trial["g_th_cm"],
            "J_th_A_cm2": trial["J_th_A_cm2"], "I_th_A": trial["I_th_A"],
            "ellipse": trial["ellipticity"], "radial_anisotropy_l2": trial["radial_anisotropy_l2"],
            "eta_rad": trial["eta_rad"], "g_th_gap_fraction": trial["g_th_gap_fraction"],
            "fine_nominal_pass_after_perturbation": trial["geometry"][0]["nominal_pass"],
            "research_nominal_pass_after_perturbation": trial["geometry"][1]["nominal_pass"],
            "fine_pass_with_another_full_error_budget": trial["fine_process_pass"],
            "research_pass_with_another_full_error_budget": trial["research_process_pass"],
            "double_internal_loss_g_th_cm": trial["double_internal_loss_audit"]["required_material_gain_cm-1"],
            "double_internal_loss_g_th_gap_fraction": trial["g_th_gap_fraction"]*trial["g_th_cm"]/
                trial["double_internal_loss_audit"]["required_material_gain_cm-1"],
            "double_internal_loss_eta_rad": trial["double_internal_loss_eta_rad"]} for trial in robust]
        output_row = {key: row[key] for key in ("name", "spec", "N", "D", "dz_nm", "g_th_cm",
            "J_th_A_cm2", "I_th_A", "alpha_field_cm", "alpha_radiation_field_cm", "eta_rad",
            "field_loss_gap_cm", "g_th_gap_fraction", "Gamma", "effective_index", "group_index",
            "bragg_nm", "frozen_CWT_pole_wavelength_nm", "beam", "ellipticity", "center_to_peak",
            "radial_anisotropy_l2", "fine_process_pass", "research_process_pass", "geometry",
            "gap_nm", "width_nm", "maximum_residual", "maximum_energy_closure_per_m")}
        output_row.update(combined_lowest_two_method_and_grid_sensitivity_cm=uncertainty,
            lowest_pole_ordering_resolved=bool(row["field_loss_gap_cm"] > uncertainty),
            beam_acceptance=accepted, radiation_fraction_pass=row["eta_rad"] >= .01,
            balanced_5percent_radiation_fraction_pass=row["eta_rad"] >= .05,
            ten_percent_material_gain_separation_pass=row["g_th_gap_fraction"] >= .10,
            complete_N7_discrete_spectrum_poles=coarse[name]["retained_poles"],
            complete_N7_lowest_g_th_cm=coarse[name]["g_th_cm"],
            N23_wide_shift_search_poles=tuned_wide[name]["retained_poles"],
            fine_spectrum_scope="6 Gamma neighborhoods (24 eigenpairs per target); independent N23 17-real-shift wide search and full N7 spectrum; not exhaustive N35 spectrum",
            robustness_N19=pole_info,
            limitations=["Linear, uniform material-gain threshold only; no nonlinear cross saturation, carrier spreading or thermal stability proof",
                "Six-wave coupling and scalar vertical mode frozen at self-consistent Bragg frequency; reported pole wavelength uses group-index conversion",
                "High-index/air effective-medium and radiation Green approximation require an independent GME/RCWA/full-wave check",
                "No proven global optimum; actual fine-grid eigenpairs retained; no fitted intercept reported as physical loss"])
        source = base/"highmesh_retuned_v1"/name/"N35"/"complex_fields.npz"
        arrays = np.load(source)
        compact = {}
        for key in arrays.files:
            value = arrays[key]
            compact[key] = value[:8] if key in ("fields", "eigenvalues", "all_radiation_x", "all_radiation_y") else value
        np.savez_compressed(out/f"{name}_complex_fields.npz", **compact)
        output_row["complex_field_subset"] = {"path": f"{name}_complex_fields.npz", "pole_indices": list(range(min(8,len(arrays['eigenvalues'])))),
            "source": source.relative_to(ROOT).as_posix(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        fields = arrays["fields"][0]
        x, y = arrays["x_um"], arrays["y_um"]
        intensity = np.sum(abs(fields)**2, axis=0)
        intensity /= intensity.max()
        angle = arrays["angle_deg"]
        far = abs(arrays["far_field_x"])**2+abs(arrays["far_field_y"])**2
        far /= far.max()
        fig, axes = plt.subplots(1,4,figsize=(15.5,3.7),constrained_layout=True)
        axes[0].tripcolor(x,y,intensity,shading="gouraud",cmap="viridis")
        axes[0].set(aspect="equal",xlabel="x (um)",ylabel="y (um)",title="Actual N=35 envelope")
        axes[1].imshow(far,extent=(angle[0],angle[-1],angle[0],angle[-1]),origin="lower",cmap="inferno")
        axes[1].set(xlabel="theta x (deg)",ylabel="theta y (deg)",title=f"ellipse={row['ellipticity']:.3f}; C/P={row['center_to_peak']:.3f}")
        middle = len(angle)//2
        axes[2].plot(angle,far[middle,:],label="theta x")
        axes[2].plot(angle,far[:,middle],"--",label="theta y")
        axes[2].set(xlim=(-.5,.5),xlabel="Angle (deg)",ylabel="Normalized intensity",title="Actual angular cuts")
        axes[2].legend()
        first = row["spectrum"][:12]
        axes[3].plot(np.arange(len(first)),[r["g_th_cm"] for r in first],"o-")
        axes[3].set(xlabel="Loss-sorted found pole",ylabel="g_th (cm^-1)",title=f"gain gap={row['g_th_gap_fraction']:.1%}; eta={row['eta_rad']:.1%}")
        fig.suptitle(f"{name}: linear engineering candidate; not nonlinear single-mode certification")
        fig.savefig(out/f"{name}_overview.png",dpi=180)
        plt.close(fig)
        finalists.append(output_row)
    balanced = phases.get("validated_balanced_v1", [])
    json_write(out/"summary.json", {"finalists": finalists,
        "additional_balanced_outcoupling_N23_comparisons": balanced,
        "phase_counts": {name: {"attempted":len(rows), "status_counts": {status:sum(r["status"]==status for r in rows)
            for status in sorted({r["status"] for r in rows})}} for name,rows in phases.items()},
        "gain_formula": "g_th=(2*alpha_field_cm+alpha_internal_intensity_cm)/(Gamma*n_active/n_eff)",
        "current_scope": "Uniform-injection Inoue rational-gain law, 30-nm active thickness, 1.5-ns lifetime; J/I are conditional estimates, area changes dominate the current difference",
        "source_input_sha256":source_hashes})
    write_csv(out/"finalists.csv",finalists)
    if balanced:
        write_csv(out/"balanced_comparisons_N23.csv", balanced)
    for row in finalists:
        write_csv(out/f"{row['name']}_robustness.csv",row["robustness_N19"])
    exploratory = []
    for phase in ("finite_broad_v1","finite_symmetry_v1","local_targets_v1", "finite_balanced_v1"):
        for row in phases.get(phase, []):
            record = row.copy()
            record["phase"] = phase
            exploratory.append(record)
    write_csv(out/"all_finite_candidates.csv", exploratory)
    admissible = [r for r in exploratory if r["status"] == "finite_solved" and r["eta_rad"] >= .01]
    objectives = np.asarray([[r["g_th_cm"],r["ellipticity"]-1,r["radial_anisotropy_l2"],1-r["center_to_peak"],-r["eta_rad"]] for r in admissible])
    pareto = pareto_mask(objectives)
    json_write(out/"coarse_pareto.json",[r for r,keep in zip(admissible,pareto,strict=True) if keep])
    fig, axes = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for phase,marker in (("finite_broad_v1","o"),("finite_symmetry_v1","^"),("local_targets_v1","s"),("finite_balanced_v1","d")):
        data = [r for r in admissible if r["phase"]==phase]
        axes[0].scatter([r["g_th_cm"]for r in data],[r["ellipticity"]for r in data],s=16,marker=marker,label=phase)
        axes[1].scatter([r["g_th_cm"]for r in data],[r["g_th_gap_fraction"]*100 for r in data],s=16,marker=marker)
    axes[0].axhline(1.10,color="gray",ls="--")
    axes[0].set(xlabel="Material gain threshold (cm^-1)",ylabel="Window moment ellipticity",ylim=(.99,2),title="Coarse search; eta_rad >= 1%")
    axes[0].legend(fontsize=8)
    axes[1].axhline(10,color="gray",ls="--")
    axes[1].set(xlabel="Material gain threshold (cm^-1)",ylabel="Closest found-pole gain gap (%)",ylim=(-1,60),title="Roundness alone does not establish mode separation")
    fig.savefig(out/"search_tradeoffs.png",dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for ax,row in zip(axes,final,strict=True):
        low=validated[row["name"]]
        trials=low["validation_trials"][-3:]+[low]+row["highmesh_trials"]+[row]
        for rank in (0,1):
            ax.plot([r["N"]for r in trials],[r["spectrum"][rank]["alpha_per_m"]/100 for r in trials],"o-",label=f"pole rank {rank}")
        ax.set(xlabel="Radius cells N",ylabel="Actual field loss (cm^-1)",title=row["name"])
        ax.legend()
        ax.grid(alpha=.25)
    fig.suptitle("Actual eigenpair mesh study; fits are sensitivity estimates only")
    fig.savefig(out/"mesh_ordering.png",dpi=180)
    plt.close(fig)
    print(out)


if __name__ == "__main__": main()
