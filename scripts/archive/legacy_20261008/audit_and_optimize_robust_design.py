"""Manufacturing-constrained six-wave design audit / 三角六波制造与容差审计.

Run directly in PyCharm. Parameters and switches are below. Every trial
rebuilds the geometry-derived optical model. No loss or beam shape is fitted.
Outputs are dated; the previous design outputs remain available for comparison.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np
import scipy

from pcselsim.fabrication import (FabricationRules, TRIANGULAR_DIRECT,
    evaluate_fabrication, pareto_mask, perturb_triangular_inclusions)
from pcselsim.triangular_six_wave import TriangularCWTSettings, TriangularEllipse, TriangularPolygon, plot_triangular_lattice
from pcselsim.triangular_finite import (TriangularFiniteSpec, solve_triangular_finite_modes,
    solve_triangular_finite_modes_converged, triangular_vector_far_field,
    plot_triangular_far_field_diagnostics, plot_triangular_grid_convergence, plot_triangular_mode_atlas)

if __package__:
    from .design_triangular_three_hole_shape_position import ShapeCandidate
    from .run_best_triangular_three_triangle_pcsel import (DESIGN, LATTICE_CONSTANT_NM,
        build_model_for_candidate, plot_device_overview, analyze_lasing_mode)
    from .design_triangular_triangle_main_refinement import candidate_from_parameters
else:
    from design_triangular_three_hole_shape_position import ShapeCandidate
    from run_best_triangular_three_triangle_pcsel import (DESIGN, LATTICE_CONSTANT_NM,
        build_model_for_candidate, plot_device_overview, analyze_lasing_mode)
    from design_triangular_triangle_main_refinement import candidate_from_parameters


# PARAMETERS / 参数区：工艺值是明确的假设档位，需替换为你的加工平台规则。
OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "robust_design_20261008"
PROCESS_TIERS = (
    FabricationRules("fine_ebeam_assumption", 40, 50, 3, 3, 5),
    FabricationRules("research_ebeam_assumption", 60, 65, 3, 3, 10),
    FabricationRules("relaxed_feature_assumption", 80, 80, 3, 3, 15),
)
COARSE_SETTINGS = TriangularCWTSettings(truncation_order=5, vertical_step_nm=5)
FINAL_SETTINGS = TriangularCWTSettings(truncation_order=10, vertical_step_nm=3)
FINITE_HALF_SIZE_UM = 150.0
COARSE_GRID = 9
FINAL_GRIDS = (11, 15, 19)
FAR_FIELD_VIEW_DEG = 1.0  # fixed no-alias observation window across screening grids
MONTE_CARLO_SAMPLES = 12  # preliminary periodic-error study, not fabrication yield
PLACEMENT_SIGMA_NM = 2.0
SIZE_SIGMA_NM = 1.5
ANGLE_SIGMA_DEG = 2.0
ETCH_BIASES_NM = (-3.0, 0.0, 3.0)
RANDOM_SEED = 20261008
STEPS = {"01_inventory": True, "02_geometry": True, "03_screen": True,
         "04_finalists": True, "05_tolerance": True}


@dataclass(frozen=True)
class Design:
    name: str
    main_shape: str
    satellite_shape: str
    fill: float
    main_area_share: float
    distance: float
    spread_deg: float
    rounding: float = 0.3
    main_angle_deg: float = 0.0
    ellipse_aspect: float = 1.3


def build_candidate(p: Design):
    area = np.sqrt(3) / 2
    angles = (-30 - p.spread_deg, 30 + p.spread_deg)
    centers = ((0, 0),) + tuple(tuple(np.linalg.solve(TRIANGULAR_DIRECT,
        p.distance * np.asarray((np.cos(np.deg2rad(t)), np.sin(np.deg2rad(t)))))) for t in angles)
    shapes = (p.main_shape, p.satellite_shape, p.satellite_shape)
    shares = (p.main_area_share, (1-p.main_area_share)/2, (1-p.main_area_share)/2)
    inclusions = []
    for i, (shape, center, share) in enumerate(zip(shapes, centers, shares, strict=True)):
        angle = p.main_angle_deg if i == 0 else angles[i-1] + (180 if shape == "triangle" else 0)
        if shape == "triangle":
            unit = TriangularPolygon.rounded_regular((0, 0), 1, corner_fraction=p.rounding,
                                                    samples_per_corner=9)
            radius = np.sqrt(p.fill * share / unit.fill_fraction)
            hole = TriangularPolygon.rounded_regular(center, radius, angle_deg=angle,
                                                     corner_fraction=p.rounding, samples_per_corner=9)
        else:
            radius = np.sqrt(p.fill * share * area / np.pi)
            aspect = p.ellipse_aspect if shape == "ellipse" else 1
            hole = TriangularEllipse(center, (radius*np.sqrt(aspect), radius/np.sqrt(aspect)), angle)
        inclusions.append(hole)
    return ShapeCandidate(p.name, f"{p.main_shape}_{p.satellite_shape}", "hard-constraint study", tuple(inclusions))


def parameter_panel():
    rows = []
    # Equal and unequal hole areas; smooth shapes need no nanoscale corner control.
    for main, satellite in (("triangle", "triangle"), ("triangle", "circle"),
                            ("triangle", "ellipse"), ("circle", "circle"),
                            ("ellipse", "circle"), ("ellipse", "ellipse")):
        for fill in (0.10, 0.15, 0.20):
            for share in (0.4, 0.55):
                for distance, spread in ((0.40, 0), (0.44, 10), (0.48, 15), (0.46, 0), (0.50, 0)):
                    name = f"{main}_{satellite}_f{fill:.2f}_m{share:.2f}_d{distance:.2f}_s{spread}"
                    rows.append(Design(name, main, satellite, fill, share, distance, spread))
        for share in (0.4, 0.55):
            rows.append(Design(f"{main}_{satellite}_f0.18_m{share:.2f}_d0.50_s0",
                               main, satellite, .18, share, .5, 0))
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory(output):
    """Syntax/source snapshot; physical interpretation is in the three audit reports."""
    rows = []
    for directory in ("src", "scripts", "tests", "configs"):
        for path in sorted((PROJECT_ROOT / directory).rglob("*")):
            if path.suffix not in (".py", ".yaml"):
                continue
            raw = path.read_bytes()
            valid = True
            if path.suffix == ".py":
                try:
                    ast.parse(raw.decode("utf-8-sig"), filename=str(path))
                except SyntaxError:
                    valid = False
            rows.append({"file": path.relative_to(PROJECT_ROOT).as_posix(),
                         "sha256": hashlib.sha256(raw).hexdigest(), "syntax_valid": valid,
                         "bytes": len(raw)})
    write_json(output / "01_source_inventory.json", {"date": "2026-10-08", "files": rows,
        "scope": "AST syntax and SHA256 only; see audit documents for physical review"})
    return rows


def optical_summary(model, modes):
    mode = min(modes.values(), key=lambda item: item.alpha_per_m)
    gap = min(item.alpha_per_m for item in modes.values() if item.name != mode.name) - mode.alpha_per_m
    far = triangular_vector_far_field(mode, model.bragg_wavelength_nm, FAR_FIELD_VIEW_DEG, 81)
    return {"mode": mode.name, "alpha_field_cm-1": mode.alpha_per_m/100,
            "bragg_wavelength_nm": model.bragg_wavelength_nm,
            "required_modal_gain_cm-1": 5 + 2*mode.alpha_per_m/100,
            "band_family_gap_cm-1": gap/100, "center_to_peak": far.center_to_peak,
            "window_RMS_deg": far.full_rms_divergence_deg,
            "window_E0p5": far.encircled_power_0p5deg,
            "ellipticity": far.ellipticity, "peak_offset_deg": far.peak_offset_deg,
            "window_half_angle_deg": float(far.angle_deg[-1]),
            "band_overlap": mode.band_overlap}


def loss_decomposition(model, mode):
    """Energy identity for the discrete operator, not a physical leakage fit."""
    norm = float(np.sum(np.abs(mode.fields)**2))
    radiation = (model.cr_m-model.cr_m.conj().T)/(2j)
    radiative = float(np.einsum("ap,ab,bp->", mode.fields.conj(), radiation, mode.fields).real/norm)
    numerator = 0.0
    for channel, (dj, dk) in enumerate(((1, 0), (1, 0), (0, 1), (0, 1), (1, 1), (1, 1))):
        for i, (j, k) in enumerate(zip(mode.grid.axial_j, mode.grid.axial_k, strict=True)):
            neighbour = mode.grid.index.get((j+dj, k+dk))
            if neighbour is not None:
                numerator += abs(mode.fields[channel, i]-mode.fields[channel, neighbour])**2
    diffusion = float(numerator/(2*mode.grid.spacing_m*norm))
    grid_alpha = mode.grid_alpha_per_m if mode.grid_alpha_per_m is not None else mode.alpha_per_m
    return {"discrete_radiation_loss_cm-1": radiative/100,
            "upwind_volume_diffusion_cm-1": diffusion/100,
            "discrete_boundary_remainder_cm-1": (grid_alpha-radiative-diffusion)/100,
            "numerical_diffusion_fraction_of_grid_loss": diffusion/grid_alpha}


def make_model(candidate, settings):
    return build_model_for_candidate(candidate, settings)[-1]


def make_spec(grid):
    return TriangularFiniteSpec(radius_um=FINITE_HALF_SIZE_UM, radius_cells=grid,
        aperture_shape="square", eigensolutions_per_target=8)


def plot_tradeoffs(rows, path):
    """Display numerical sensitivity, not a certified physical ranking.

    alpha is a field loss; the gain sensitivity is twice that quantity.
    中文：误差横线是拟合网格敏感度，不是统计置信区间或工艺误差。
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, row in enumerate(rows):
        ax.errorbar(row["required_modal_gain_cm-1"], row["window_RMS_deg"],
                    xerr=2 * row["empirical_grid_sensitivity_cm-1"], fmt="o", capsize=4,
                    label=f"{i+1}: gap {row['minimum_periodic_gap_nm']:.0f} nm; {row['candidate']}")
    ax.set(xlabel="Extrapolated modal power gain (cm$^{-1}$; unverified)",
           ylabel="1-deg window radial RMS diameter (deg)",
           title="Grid-sensitive candidates: no resolved threshold ranking")
    ax.grid(alpha=.2)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(.5, -.18))
    fig.text(.5, .005, "Bars: empirical grid-fit sensitivity, NOT confidence intervals or all model errors.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .02, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=tuple(STEPS))
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY,
                        help="new result directory; keep earlier runs separate")
    parser.add_argument("--quick", action="store_true", help="small diagnostic; not final recommendations")
    args = parser.parse_args()
    switches = {k: k in args.only for k in STEPS} if args.only else STEPS.copy()
    output = args.output.resolve() / "quick" if args.quick else args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "00_run_settings.json", {"python": sys.version, "numpy": np.__version__,
        "scipy": scipy.__version__, "coarse_optics": asdict(COARSE_SETTINGS),
        "final_optics": asdict(FINAL_SETTINGS), "finite_grids": FINAL_GRIDS,
        "device_half_size_um": FINITE_HALF_SIZE_UM, "lattice_nm": LATTICE_CONSTANT_NM,
        "process_tiers": [asdict(p) for p in PROCESS_TIERS],
        "far_field_half_window_deg": FAR_FIELD_VIEW_DEG, "random_seed": RANDOM_SEED})
    if switches["01_inventory"]:
        print("[01] syntax and source hash inventory", flush=True)
        inventory(output)
    if not any(switches[key] for key in STEPS if key != "01_inventory"):
        return
    parameters = parameter_panel()
    if args.quick:
        parameters = parameters[::18]
    candidates = {p.name: build_candidate(p) for p in parameters}
    candidates["previous_0deg_q0p10"] = candidate_from_parameters(DESIGN)
    geometry_rows, eligible = [], []
    print(f"[02] periodic geometry rules: {len(candidates)} candidates", flush=True)
    for name, candidate in candidates.items():
        process = [evaluate_fabrication(candidate.inclusions, LATTICE_CONSTANT_NM,
                   TRIANGULAR_DIRECT, rules) for rules in PROCESS_TIERS]
        record = {"candidate": name, **{k: v for k, v in process[0].items() if k != "rules"},
                  "fine_pass": process[0]["margin_pass"], "research_pass": process[1]["margin_pass"],
                  "relaxed_pass": process[2]["margin_pass"]}
        geometry_rows.append(record)
        if record["fine_pass"] or name.startswith("previous"):
            eligible.append(name)
    geometry_lookup = {row["candidate"]: row for row in geometry_rows}
    if switches["02_geometry"]:
        write_csv(output / "02_geometry_rules.csv", geometry_rows)
        write_json(output / "02_parameter_panel.json", [asdict(p) for p in parameters])
        write_json(output / "02_geometry_inputs.json", {name: [asdict(hole) for hole in candidate.inclusions]
                   for name, candidate in candidates.items()})
        # Also archive exact inputs for completed finalists, without recomputing
        # optical results. A run's source hashes distinguish later metadata edits.
        for name, candidate in candidates.items():
            folder = output / "finalists" / name
            if folder.is_dir() and not (folder / "geometry_inputs.json").exists():
                write_json(folder / "geometry_inputs.json", [asdict(hole) for hole in candidate.inclusions])
    if not switches["03_screen"]:
        return
    print(f"[03] six-wave optical screen: {len(eligible)} geometries", flush=True)
    rows = []
    for i, name in enumerate(eligible):
        model = make_model(candidates[name], COARSE_SETTINGS)
        modes = solve_triangular_finite_modes(model, make_spec(COARSE_GRID))
        row = {"candidate": name, **optical_summary(model, modes), **{
            k: geometry_lookup[name][k] for k in ("minimum_periodic_gap_nm", "minimum_caliper_nm",
                                                "minimum_sampled_corner_radius_nm", "fine_pass", "research_pass", "relaxed_pass")}}
        rows.append(row)
        write_csv(output / "03_screen.csv", rows)
        print(f"  {i+1}/{len(eligible)} {name}: {row['mode']} alpha={row['alpha_field_cm-1']:.3f}, center={row['center_to_peak']:.3f}", flush=True)
    # Do not reward low threshold if its lowest family is dark/off-axis.
    beam_candidates = [r for r in rows if r["center_to_peak"] >= .8 and r["ellipticity"] < 1.5
                       and r["peak_offset_deg"] <= .1 and r["fine_pass"]]
    if not beam_candidates:
        beam_candidates = [r for r in rows if r["fine_pass"]]
    objectives = np.asarray([[r["alpha_field_cm-1"], r["window_RMS_deg"],
                            -r["minimum_periodic_gap_nm"], -r["band_family_gap_cm-1"]] for r in beam_candidates])
    frontier = [r for r, keep in zip(beam_candidates, pareto_mask(objectives), strict=True) if keep]
    write_json(output / "03_pareto.json", frontier)
    # Keep candidates from feasible process tiers rather than one subjective score.
    finalists = ["previous_0deg_q0p10"]
    for criterion in ("fine_pass", "research_pass", "relaxed_pass"):
        tier = [r for r in beam_candidates if r[criterion]]
        if tier:
            finalists.append(min(tier, key=lambda r: r["alpha_field_cm-1"])["candidate"])
    if beam_candidates:
        finalists.append(min(beam_candidates, key=lambda r: r["window_RMS_deg"])["candidate"])
    finalists = list(dict.fromkeys(finalists))
    if args.quick:
        finalists = finalists[:2]
    if not switches["04_finalists"]:
        return
    final_rows, final_models = [], {}
    grids = (7, 9, 11) if args.quick else FINAL_GRIDS
    print(f"[04] multi-grid finalists: {finalists}", flush=True)
    for name in finalists:
        model = make_model(candidates[name], FINAL_SETTINGS)
        modes, convergence = solve_triangular_finite_modes_converged(model, make_spec(grids[-1]), grids)
        row = {"candidate": name, **optical_summary(model, modes), **{
            k: geometry_lookup[name][k] for k in ("minimum_periodic_gap_nm", "minimum_caliper_nm",
                                                "minimum_sampled_corner_radius_nm", "fine_pass", "research_pass", "relaxed_pass")}}
        mode = modes[row["mode"]]
        raw = convergence[mode.name]
        row["finest_grid_alpha_cm-1"] = float(raw[-1].imag/100)
        row["extrapolation_shift_cm-1"] = float(abs(mode.alpha_per_m-raw[-1].imag)/100)
        row["extrapolation_status"] = mode.extrapolation_status
        row["empirical_grid_sensitivity_cm-1"] = mode.extrapolation_uncertainty_per_m / 100
        row.update(loss_decomposition(model, mode))
        final_rows.append(row)
        final_models[name] = model
        folder = output / "finalists" / name
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "geometry_inputs.json", [asdict(hole) for hole in candidates[name].inclusions])
        cell = build_model_for_candidate(candidates[name], FINAL_SETTINGS)[0]
        plot_triangular_lattice(cell, folder / "01_lattice.png")
        plot_device_overview(candidates[name], cell, folder / "02_device_overview.png")
        plot_triangular_grid_convergence(convergence, FINITE_HALF_SIZE_UM, folder / "03_convergence.png")
        plot_triangular_far_field_diagnostics(mode, model, folder / "04_far_field.png", view_deg=1.5)
        plot_triangular_mode_atlas(modes, model, folder / "05_mode_atlas.png", view_deg=1.5)
        np.savez_compressed(folder / "06_raw_modes_and_coupling.npz", x_um=mode.grid.x_um,
            y_um=mode.grid.y_um, Cb_m=model.cb_m, Cr_m=model.cr_m, Ch_m=model.ch_m,
            wavelength_nm=model.bragg_wavelength_nm,
            **{f"fields_{key}": value.fields for key, value in modes.items()},
            **{f"radiation_x_{key}": value.radiation_x for key, value in modes.items()},
            **{f"radiation_y_{key}": value.radiation_y for key, value in modes.items()})
        write_json(folder / "summary.json", row)
        print(f"  {name}: alpha={row['alpha_field_cm-1']:.4f}, grid shift={row['extrapolation_shift_cm-1']:.4f}", flush=True)
    write_csv(output / "04_finalists.csv", final_rows)
    plot_tradeoffs(final_rows, output / "04_tradeoffs.png")
    if not switches["05_tolerance"]:
        return
    print("[05] systematic etch + periodic motif placement/size/angle errors", flush=True)
    tolerance_rows = []
    count = 3 if args.quick else MONTE_CARLO_SAMPLES
    for name in finalists:
        rng = np.random.default_rng(RANDOM_SEED)
        trials = [("etch", bias, 0, 0, 0) for bias in ETCH_BIASES_NM]
        trials += [("periodic_mc", 0, PLACEMENT_SIGMA_NM, SIZE_SIGMA_NM, ANGLE_SIGMA_DEG) for _ in range(count)]
        for j, (kind, bias, position, size, angle) in enumerate(trials):
            holes = perturb_triangular_inclusions(candidates[name].inclusions, LATTICE_CONSTANT_NM, rng,
                etch_bias_nm=bias, placement_sigma_nm=position, scale_sigma_nm=size, angle_sigma_deg=angle)
            candidate = replace(candidates[name], inclusions=holes)
            fab = evaluate_fabrication(holes, LATTICE_CONSTANT_NM, TRIANGULAR_DIRECT, PROCESS_TIERS[0])
            if fab["minimum_periodic_gap_nm"] <= 0:
                tolerance_rows.append({"candidate": name, "trial": j, "type": kind, "etch_nm": bias,
                                       "valid": False, "mode": "overlap", "alpha_field_cm-1": None,
                                       "center_to_peak": None, "window_RMS_deg": None})
                continue
            model = make_model(candidate, COARSE_SETTINGS)
            modes = solve_triangular_finite_modes(model, make_spec(11))
            summary = optical_summary(model, modes)
            tolerance_rows.append({"candidate": name, "trial": j, "type": kind, "etch_nm": bias,
                                   "valid": True, **{k: summary[k] for k in ("mode", "alpha_field_cm-1", "center_to_peak", "window_RMS_deg")}})
            write_csv(output / "05_tolerance_trials.csv", tolerance_rows)
        print(f"  completed {name}: {len(trials)} trials", flush=True)
    report = {"finalists": final_rows, "periodic_error_trials": tolerance_rows,
        "process_rules": [asdict(p) for p in PROCESS_TIERS], "seed": RANDOM_SEED,
        "candidates": len(candidates), "optically_screened": len(rows),
        "scope": ["conditional energy within fixed 1-deg observation window",
                  "six band-connected fundamental families, not exhaustive lateral spectrum",
                  "periodically repeated unit-cell perturbations, not spatial disorder or production yield",
                  "grid extrapolation is estimated, not certified physical convergence",
                  "local discrete candidate set; no global optimality proof"]}
    write_json(output / "00_design_audit.json", report)
    inventory(output)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
