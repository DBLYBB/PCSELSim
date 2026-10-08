"""Independent high-accuracy check of a research-process triangular/circle motif.

Run in PyCharm or from the project root.  This restores a candidate excluded
by a coarse ellipticity cutoff and evaluates its tradeoff explicitly.  No
global-optimum or fabrication-yield claim is made.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

if __package__:
    from . import audit_and_optimize_robust_design as study
else:
    import audit_and_optimize_robust_design as study

from pcselsim.fabrication import (
    TRIANGULAR_DIRECT, evaluate_fabrication, inclusion_boundary,
)
from pcselsim.triangular_finite import (
    plot_triangular_far_field_diagnostics,
    plot_triangular_grid_convergence,
    plot_triangular_mode_atlas,
    solve_triangular_finite_modes_converged,
    triangular_vector_far_field,
)
from pcselsim.triangular_six_wave import plot_triangular_lattice


# Parameters / 参数：与193候选主研究的指定方案逐项一致。
CANDIDATE_NAME = "triangle_circle_f0.15_m0.40_d0.50_s0"
GRIDS = (11, 15, 19)
VIEW_DEG = 1.0
OUTPUT_DIRECTORY = study.OUTPUT_DIRECTORY / "research_tier_diagnostic"


def plot_generic_overview(candidate, cell, path):
    """Complete square device plus lattice and primitive-cell zooms."""
    half_um = study.FINITE_HALF_SIZE_UM
    primitive_area_um2 = np.sqrt(3.0) / 2 * (study.LATTICE_CONSTANT_NM * 1e-3) ** 2
    boundaries = [inclusion_boundary(hole, TRIANGULAR_DIRECT, 120) for hole in candidate.inclusions]
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.8))
    axes[0].add_patch(Rectangle((-half_um, -half_um), 2 * half_um, 2 * half_um,
                               facecolor="#d9ecf7", edgecolor="#174a7e", lw=2))
    axes[0].text(0, 0, f"{2*half_um:g} um square\n~{(2*half_um)**2/primitive_area_um2:,.0f} primitive cells",
                 ha="center", va="center")
    axes[0].set(aspect="equal", xlim=(-170, 170), ylim=(-170, 170),
                title="complete finite PC / uniform gain area", xlabel="x (um)", ylabel="y (um)")
    for u in range(-4, 5):
        for v in range(-4, 5):
            origin = TRIANGULAR_DIRECT @ np.asarray((u, v))
            for boundary in boundaries:
                axes[1].add_patch(Polygon(boundary + origin, closed=True, facecolor="white",
                                          edgecolor="black", lw=0.6))
    axes[1].set(aspect="equal", xlim=(-3.5, 3.5), ylim=(-3.0, 3.0),
                title="periodic motif zoom", xlabel="x/a", ylabel="y/a")
    for i, boundary in enumerate(boundaries):
        axes[2].add_patch(Polygon(boundary, closed=True, facecolor="#f7f7f7", edgecolor="black", lw=1.4))
        center = boundary.mean(axis=0)
        axes[2].text(*center, "main" if i == 0 else f"sat {i}", ha="center", va="center", fontsize=8)
    a1, a2 = TRIANGULAR_DIRECT.T
    primitive = np.asarray((-.5*(a1+a2), .5*(a1-a2), .5*(a1+a2), .5*(-a1+a2)))
    axes[2].add_patch(Polygon(primitive, closed=True, fill=False, edgecolor="#d62728", lw=1.6))
    axes[2].set(aspect="equal", xlim=(-.7, 1.0), ylim=(-.72, .72),
                title=f"rounded triangle + two circles; fill={cell.fill_fraction:.3f}",
                xlabel="x/a", ylabel="y/a")
    fig.suptitle("Research-process candidate: beam ellipticity versus fabrication margin")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main():
    output = OUTPUT_DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    parameters = next(p for p in study.parameter_panel() if p.name == CANDIDATE_NAME)
    candidate = study.build_candidate(parameters)
    cell, layers, model = study.build_model_for_candidate(candidate, study.FINAL_SETTINGS)
    fabrication = [evaluate_fabrication(candidate.inclusions, study.LATTICE_CONSTANT_NM,
                                        TRIANGULAR_DIRECT, rules) for rules in study.PROCESS_TIERS]
    print(f"Research-tier check: {CANDIDATE_NAME}; D=10, dz=3 nm, grids={GRIDS}", flush=True)
    modes, convergence = solve_triangular_finite_modes_converged(model, study.make_spec(GRIDS[-1]), GRIDS)
    lowest = min(modes.values(), key=lambda mode: mode.alpha_per_m)
    mode_rows = []
    for name, mode in modes.items():
        far = triangular_vector_far_field(mode, model.bragg_wavelength_nm, VIEW_DEG, 81)
        mode_rows.append({
            "mode": name, "alpha_field_cm-1": mode.alpha_per_m / 100,
            "finest_grid_alpha_cm-1": mode.grid_alpha_per_m / 100,
            "empirical_grid_sensitivity_cm-1": mode.extrapolation_uncertainty_per_m / 100,
            "extrapolation_status": mode.extrapolation_status,
            "delta_per_m": mode.delta_per_m, "band_overlap": mode.band_overlap,
            "center_to_peak": far.center_to_peak,
            "ellipticity": far.ellipticity, "window_E0p5": far.encircled_power_0p5deg,
            "window_RMS_diameter_deg": far.full_rms_divergence_deg,
            "evaluated_half_window_deg": far.evaluated_view_deg,
        })
    grid_rows = []
    for i, grid in enumerate(GRIDS):
        order = sorted(modes, key=lambda name: convergence[name][i].imag)
        grid_rows.append({
            "grid": grid, "lowest_selected_family": order[0],
            "alpha_grid_cm-1": float(convergence[order[0]][i].imag / 100),
            "second_family": order[1],
            "family_gap_cm-1": float((convergence[order[1]][i].imag - convergence[order[0]][i].imag) / 100),
        })
    finest_lowest_name = grid_rows[-1]["lowest_selected_family"]
    summary = {
        "candidate": CANDIDATE_NAME, "parameters": asdict(parameters),
        "optical_settings": asdict(study.FINAL_SETTINGS), "grids": GRIDS,
        "device_side_um": 2 * study.FINITE_HALF_SIZE_UM,
        "lattice_constant_nm": study.LATTICE_CONSTANT_NM,
        "bragg_wavelength_nm": model.bragg_wavelength_nm,
        "coarse_rejection_reason": "N9 ellipticity=1.602 exceeded strict <1.5 cutoff",
        "fabrication": fabrication, "gridwise_lowest_families": grid_rows,
        "modes": mode_rows, "lowest_extrapolated_family": lowest.name,
        "lowest_finest_grid_family": finest_lowest_name,
        "ordering_changed_by_extrapolation": any(
            row["lowest_selected_family"] != lowest.name for row in grid_rows
        ),
        "lowest_summary": study.optical_summary(model, modes),
        "loss_decomposition_at_finest_grid": study.loss_decomposition(model, lowest),
        "scope": [
            "Research process tier is an assumed rule set, not foundry approval",
            "Energy and RMS use a fixed +/-1 degree square observation window",
            "An ellipse-like centered beam may be a useful fabrication tradeoff",
            "Six band-connected fundamental families; higher lateral modes are not exhausted",
            "First-order grid extrapolation remains provisional",
            "B1 wins the intercept but B2 wins every actual grid; lasing-family choice is unresolved",
        ],
    }
    study.write_json(output / "00_summary.json", summary)
    study.write_json(output / "geometry_inputs.json", [asdict(h) for h in candidate.inclusions])
    study.write_csv(output / "mode_summary.csv", mode_rows)
    study.write_csv(output / "grid_lowest_families.csv", grid_rows)
    plot_triangular_lattice(cell, output / "01_lattice.png")
    plot_generic_overview(candidate, cell, output / "02_device_overview.png")
    plot_triangular_grid_convergence(convergence, study.FINITE_HALF_SIZE_UM, output / "03_convergence.png")
    plot_triangular_far_field_diagnostics(lowest, model, output / "04_far_field.png", view_deg=VIEW_DEG)
    plot_triangular_far_field_diagnostics(modes[finest_lowest_name], model,
                                        output / "04b_finest_grid_lowest_far_field.png", view_deg=VIEW_DEG)
    plot_triangular_mode_atlas(modes, model, output / "05_mode_atlas.png", view_deg=VIEW_DEG)
    np.savez_compressed(output / "06_raw_modes_and_coupling.npz",
        x_um=lowest.grid.x_um, y_um=lowest.grid.y_um,
        Cb_m=model.cb_m, Cr_m=model.cr_m, Ch_m=model.ch_m, C_m=model.coupling_m,
        wavelength_nm=model.bragg_wavelength_nm, grids=np.asarray(GRIDS),
        **{f"fields_{name}": mode.fields for name, mode in modes.items()},
        **{f"radiation_x_{name}": mode.radiation_x for name, mode in modes.items()},
        **{f"radiation_y_{name}": mode.radiation_y for name, mode in modes.items()},
        **{f"convergence_{name}": convergence[name] for name in modes})
    print(summary["lowest_summary"], flush=True)
    print(f"Saved separate diagnostic: {output}", flush=True)


if __name__ == "__main__":
    main()
