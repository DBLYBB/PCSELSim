"""Square PCSEL: circle/ellipse screening under explicit fabrication rules.

This compares 300-um cold cavities on the same Inoue vertical stack. Only one
band-connected finite state per A/B/C/D is retained, so the reported gap is
not a proof of single-transverse-mode lasing. Hard geometry checks precede
optical ranking; score weights are never used to waive a failed design rule.
中文：圆/椭圆避免锐角，显式检查实际nm孔宽和间隙；阈值损耗统一为振幅系数。
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np

from pcselsim.custom_analysis import (
    plot_grid_convergence, plot_threshold_summary,
    plot_vector_far_field_diagnostics, solve_finite_modes,
    solve_finite_modes_converged, vector_far_field_metrics,
)
from pcselsim.fabrication import FabricationRules, evaluate_fabrication, pareto_mask
from pcselsim.geometry import Ellipse

if __package__:
    from .design_square_lattice_multiobjective import (
        CANDIDATES, SquareCandidate, build_model, ellipse_for_fill, plot_cell,
    )
    from .compare_square_triangular_equal_size import threshold_audit, cold_cavity_q
    from .run_custom_semiconductor_pcsel import ACTIVE_LAYER_NAME, LATTICE_CONSTANT_NM
else:
    from design_square_lattice_multiobjective import (
        CANDIDATES, SquareCandidate, build_model, ellipse_for_fill, plot_cell,
    )
    from compare_square_triangular_equal_size import threshold_audit, cold_cavity_q
    from run_custom_semiconductor_pcsel import ACTIVE_LAYER_NAME, LATTICE_CONSTANT_NM

OUTPUT = PROJECT_ROOT / "results" / "square_fabrication_20261008"
DOMAIN_UM = 300.0
COARSE = (6, 4.0, 13)
FINAL = (10, 3.0, (13, 17, 21))
RULES = {
    "research": FabricationRules("research", 40.0, 50.0, 3.0, 3.0),
    "easier": FabricationRules("easier", 60.0, 65.0, 3.0, 3.0),
}


def candidate_panel() -> tuple[SquareCandidate, ...]:
    values = [item for item in CANDIDATES if all(isinstance(hole, Ellipse) for hole in item.inclusions)]
    # Separations below are coordinate offsets dx=dy=d*a, as in the square
    # double lattice, rather than the radial distance d*a.
    for fill in (0.10, 0.12, 0.14):
        for separation in (0.35, 0.40, 0.45, 0.50):
            for fraction, aspect in ((0.50, 1.0), (0.42, 1.0), (0.50, 1.15), (0.42, 1.15)):
                name = f"smooth_F{fill:.2f}_d{separation:.2f}_w{fraction:.2f}_a{aspect:.2f}".replace(".", "p")
                centers = ((-0.5 * separation, -0.5 * separation), (0.5 * separation, 0.5 * separation))
                inclusions = tuple(
                    ellipse_for_fill(center, fill * weight, aspect, -35.0)
                    for center, weight in zip(centers, (fraction, 1.0 - fraction), strict=True)
                )
                values.append(SquareCandidate(name, "smooth_double", "two circle/ellipse holes, no sharp corners", inclusions))
    return tuple(values)


def geometry_row(candidate) -> dict:
    row = {}
    for name, rules in RULES.items():
        metrics = evaluate_fabrication(candidate.inclusions, LATTICE_CONSTANT_NM, rules=rules)
        for key, value in metrics.items():
            if key != "rules":
                row[f"{name}_{key}"] = value
    return row


def evaluate(candidate, settings, final=False):
    cell, model, base = build_model(candidate, settings[0], settings[1])
    spec = replace(base, domain_um=DOMAIN_UM, internal_loss_cm=0.0, grid_points=settings[2][-1] if final else settings[2])
    convergence = None
    if final:
        modes, convergence = solve_finite_modes_converged(spec, settings[2], model.coupling_m, model.eigenvectors, model.radiation_fields)
    else:
        modes = solve_finite_modes(spec, model.coupling_m, model.eigenvectors, model.radiation_fields)
    mode = min(modes.values(), key=lambda value: value.alpha_per_m)
    far = vector_far_field_metrics(spec, mode)
    gap = min(value.alpha_per_m for value in modes.values() if value.name != mode.name) - mode.alpha_per_m
    row = {
        "candidate": candidate.name, "mode": mode.name,
        "bragg_wavelength_nm": model.bragg_wavelength_nm,
        "amplitude_optical_loss_cm-1": mode.alpha_per_m / 100.0,
        "band_connected_gap_cm-1": gap / 100.0,
        "center_to_peak": far.center_to_peak,
        "full_rms_deg": far.full_rms_divergence_deg,
        "ellipticity": far.ellipticity,
        "encircled_power_0p5deg": far.encircled_power_0p5deg,
        "encircled_power_1deg": far.encircled_power_1deg,
        "peak_offset_deg": far.peak_offset_deg,
        "centroid_offset_deg": far.centroid_offset_deg,
        "cold_cavity_Q": cold_cavity_q(model.bragg_wavelength_nm, model.group_index, mode.alpha_per_m),
        **threshold_audit(mode.alpha_per_m, (DOMAIN_UM * 1e-6)**2,
                          model.vertical_mode.confinement[ACTIVE_LAYER_NAME], model.effective_index),
        **geometry_row(candidate),
    }
    row["beam_qualified"] = bool(row["center_to_peak"] >= 0.75 and row["encircled_power_0p5deg"] >= 0.90
                                 and row["ellipticity"] <= 1.4 and row["peak_offset_deg"] <= 0.2)
    # This transparent utility ranks within the hard fabrication/beam set.
    row["utility"] = (100 * row["encircled_power_0p5deg"] + 30 * row["center_to_peak"]
                      - 20 * row["full_rms_deg"] - row["amplitude_optical_loss_cm-1"]
                      - 15 * abs(np.log(row["ellipticity"])))
    if final:
        # Loss may be the provisional 1/N intercept, or the finest-grid
        # fallback if that intercept is rejected. The field remains N=21.
        row["extrapolation_status"] = mode.extrapolation_status
        row["extrapolation_sensitivity_cm-1"] = mode.extrapolation_uncertainty_per_m / 100.0
        row["finest_grid_amplitude_loss_cm-1"] = mode.grid_alpha_per_m / 100.0
        row["raw_amplitude_loss_intercept_cm-1"] = float(convergence[f"{mode.name}_alpha_diagnostics"][0]) / 100.0
    return row, (cell, model, spec, modes, convergence)


def write_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value in {"True", "False"}:
                row[key] = value == "True"
            elif key not in {"candidate", "mode"}:
                row[key] = float(value)
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = candidate_panel()
    geometries = [{"candidate": item.name, **geometry_row(item)} for item in candidates]
    write_csv(geometries, OUTPUT / "01_geometry_all.csv")
    survivors = [item for item, row in zip(candidates, geometries, strict=True) if row["research_nominal_pass"]]
    print(f"geometry screen: {len(candidates)} defined, {len(survivors)} pass 40/50-nm nominal rule", flush=True)
    # Retain completed coarse data for interrupted plotting/final-stage runs.
    # Delete/rename the dated result folder to start a new parameter panel.
    rows = read_csv(OUTPUT / "02_coarse.csv") if (OUTPUT / "02_coarse.csv").exists() else []
    done = {row["candidate"] for row in rows}
    for index, candidate in enumerate(survivors, 1):
        if candidate.name in done:
            continue
        print(f"[{index}/{len(survivors)}] {candidate.name}", flush=True)
        row, _ = evaluate(candidate, COARSE)
        rows.append(row)
        write_csv(rows, OUTPUT / "02_coarse.csv")
        print(f"  {row['mode']} alpha={row['amplitude_optical_loss_cm-1']:.4f}, center={row['center_to_peak']:.3f}, E.5={row['encircled_power_0p5deg']:.4f}", flush=True)
    lookup = {item.name: item for item in survivors}
    final_names = []
    for rule in RULES:
        acceptable = [row for row in rows if row[f"{rule}_margin_pass"] and row["beam_qualified"]]
        if acceptable:
            final_names.append(max(acceptable, key=lambda row: row["utility"])["candidate"])
        nominal = [row for row in rows if row[f"{rule}_nominal_pass"] and row["beam_qualified"]]
        if nominal:
            final_names.append(max(nominal, key=lambda row: row["utility"])["candidate"])
    final_names.append(min(rows, key=lambda row: row["amplitude_optical_loss_cm-1"])["candidate"])
    final_rows = []
    for name in dict.fromkeys(final_names):
        print(f"FINAL: {name}", flush=True)
        row, data = evaluate(lookup[name], FINAL, final=True)
        final_rows.append(row)
        folder = OUTPUT / "finalists" / name
        folder.mkdir(parents=True, exist_ok=True)
        cell, model, spec, modes, convergence = data
        np.savez_compressed(folder / "02_convergence_arrays.npz", **convergence)
        (folder / "02_numerical_quality.json").write_text(json.dumps({
            key: {"reported_amplitude_loss_cm-1": value.alpha_per_m / 100.0,
                  "finest_grid_amplitude_loss_cm-1": value.grid_alpha_per_m / 100.0,
                  "raw_intercept_cm-1": float(convergence[f"{key}_alpha_diagnostics"][0]) / 100.0,
                  "extrapolation_sensitivity_cm-1": value.extrapolation_uncertainty_per_m / 100.0,
                  "extrapolation_status": value.extrapolation_status}
            for key, value in modes.items()
        }, indent=2), encoding="utf-8")
        plot_cell(lookup[name], folder / "01_lattice.png")
        plot_grid_convergence(spec, convergence, folder / "02_convergence.png")
        plot_threshold_summary(spec, modes, folder / "03_thresholds.png")
        plot_vector_far_field_diagnostics(spec, modes[row["mode"]], folder / "04_far_field.png")
        write_csv(final_rows, OUTPUT / "03_final.csv")
    objectives = np.asarray([[row["amplitude_optical_loss_cm-1"], row["full_rms_deg"],
                              1-row["encircled_power_0p5deg"], -row["easier_minimum_periodic_gap_nm"]] for row in rows])
    write_csv([row for row, keep in zip(rows, pareto_mask(objectives)) if keep], OUTPUT / "04_pareto_coarse.csv")
    report = {
        "date": "2026-10-08", "device_side_um": DOMAIN_UM,
        "defined_count": len(candidates), "geometry_survivors": len(survivors),
        "rules": {name: rules.__dict__ for name, rules in RULES.items()},
        "finalists": final_rows,
        "recommendations": {name: max((row for row in final_rows if row[f"{name}_margin_pass"] and row["beam_qualified"]),
                                       key=lambda row: row["utility"], default=None) for name in RULES},
        "scope": ["local finite candidate screen, not global optimum", "four band-connected fundamental states only, not all transverse modes",
                  "RMS and encircled energy are computed over the represented angular aperture", "uniform injection static current audit, no electrothermal closure"],
    }
    (OUTPUT / "05_recommendation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"complete: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
