"""Matched square/triangular controls, separate from each optimized design.

Equal reciprocal-shell magnitude, filling, vertical stack, material gain/loss,
and 200-um square aperture. Only Bravais lattice and the corresponding hole
area differ. The selected Gamma families are not an exhaustive modal spectrum.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
from datetime import datetime
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pcselsim.geometry import Ellipse, SquareLatticeCell
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from pcselsim.triangular_six_wave import (TriangularEllipse, TriangularLatticeCell,
    TriangularCWTSettings, build_triangular_coupling)
from pcselsim.triangular_finite import (TriangularFiniteSpec,
    solve_triangular_finite_modes_converged, triangular_vector_far_field)
from pcselsim.custom_analysis import FourWaveOpticalSpec, solve_finite_modes_converged
from pcselsim.design_objectives import beam_metrics, beam_acceptance
from pcselsim.run_output import default_output, prepare_output, write_provenance
from scripts.optimization.compare_lattice_symmetry import stack_for
from scripts.optimization.square_semiconductor_search import direct_vector_far_field


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = prepare_output(args.output or default_output(ROOT, "optimization_20261021/controlled"), ROOT)
    write_provenance(output, ROOT)
    (output/"run_status.json").write_text(json.dumps({"status": "running"}))
    rows, arrays = [], {}
    for fill in (.15, .29):
        for aspect in (1., 1.08):
            for lattice in ("square", "triangular"):
                identifier = f"{lattice}_f{fill:.2f}_asp{aspect:.2f}"
                primitive_area = 1. if lattice == "square" else np.sqrt(3)/2
                radius = np.sqrt(fill*primitive_area/np.pi)
                radii = (float(radius*np.sqrt(aspect)), float(radius/np.sqrt(aspect)))
                a = 277. if lattice == "square" else 277.*2/np.sqrt(3)
                cell = (SquareLatticeCell(3.554**2, (Ellipse((0, 0), radii),)) if lattice == "square"
                        else TriangularLatticeCell(3.554**2, (TriangularEllipse(radii_over_a=radii),)))
                stack = stack_for(cell)
                stack = replace(stack, layers=tuple(replace(layer, thickness_nm=120.)
                    if layer.name == "photonic crystal" else layer for layer in stack.layers))
                if lattice == "square":
                    model = build_geometry_coupling(cell, stack, "photonic crystal", a, 950., ThreeDCWTSettings(8, 3.))
                    spec = FourWaveOpticalSpec(model.bragg_wavelength_nm, model.effective_index,
                        model.group_index, 200., 5., tuple(model.eigenvalues_m.real/100),
                        tuple(model.eigenvalues_m.imag/100), 25)
                    modes, convergence = solve_finite_modes_converged(spec, (13, 19, 25),
                        model.coupling_m, model.eigenvectors, model.radiation_fields)
                else:
                    model = build_triangular_coupling(cell, stack, "photonic crystal", a, 950., TriangularCWTSettings(8, 3.))
                    # The triangular legacy offset is a field-amplitude loss.
                    # Keep it zero: the material POWER loss is added once in g.
                    spec = TriangularFiniteSpec(radius_um=100., radius_cells=13,
                        internal_loss_cm=0., eigensolutions_per_target=12, aperture_shape="square")
                    modes, convergence = solve_triangular_finite_modes_converged(model, spec, (7, 10, 13))
                gamma = model.vertical_mode.confinement["active InGaAs QWs"]
                mode_rows = []
                for name, mode in modes.items():
                    fields = mode.fields.reshape(len(model.coupling_m), -1)
                    alpha_local = float(np.vdot(fields, model.coupling_m@fields).imag/np.vdot(fields, fields).real)
                    if lattice == "square":
                        angle, fx, fy, metrics = direct_vector_far_field(mode, model.bragg_wavelength_nm, 200., 161)
                    else:
                        ff = triangular_vector_far_field(mode, model.bragg_wavelength_nm, 1., 161)
                        angle, fx, fy = ff.angle_deg, ff.field_x, ff.field_y
                        metrics = beam_metrics(angle, fx, fy, requested_view_deg=1., source_nyquist_deg=ff.alias_free_square_view_deg)
                    g = (2*mode.alpha_per_m/100+5)/(gamma*3.584/model.effective_index)
                    row = {"name": name, "g_th_cm1": g, "alpha_field_cm1": mode.alpha_per_m/100,
                        "grid_sensitivity_cm1": mode.extrapolation_uncertainty_per_m/100,
                        "eta_rad": 2*alpha_local/(2*mode.alpha_per_m+500),
                        "beam": metrics, "beam_pass": beam_acceptance(metrics, require_sampling=True),
                        "eigen_residual": mode.eigen_residual}
                    mode_rows.append(row)
                    arrays[identifier+"_"+name+"_field"] = mode.fields
                    arrays[identifier+"_"+name+"_far_x"] = fx
                    arrays[identifier+"_"+name+"_far_y"] = fy
                ranked = sorted(mode_rows, key=lambda row: row["g_th_cm1"])
                rows.append({"id": identifier, "lattice": lattice, "fill": fill, "aspect": aspect,
                    "a_nm": a, "pc_nm": 120., "side_um": 200., "wavelength_nm": model.bragg_wavelength_nm,
                    "gamma_active": gamma, "n_eff": model.effective_index,
                    "grids": [13,19,25] if lattice == "square" else [7,10,13],
                    "lowest_selected_family": ranked[0], "all_selected_families": mode_rows,
                    "family_relative_gain_gap": ranked[1]["g_th_cm1"]/ranked[0]["g_th_cm1"]-1,
                    "scope": "matched controls, one selected actual state per Gamma family; not complete spectrum"})
                print(identifier, ranked[0]["name"], "g", round(ranked[0]["g_th_cm1"], 4),
                    "ellipse", round(ranked[0]["beam"]["ellipticity"], 4),
                    "center", round(ranked[0]["beam"]["center_to_peak"], 4), flush=True)
                (output/"matched_controls.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output/"matched_controls_arrays.npz", **arrays)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for lattice, marker in (("square", "s"), ("triangular", "^")):
        data = [r for r in rows if r["lattice"] == lattice]
        axes[0].plot(range(4), [r["lowest_selected_family"]["g_th_cm1"] for r in data], marker+"-", label=lattice)
        axes[1].plot(range(4), [r["lowest_selected_family"]["beam"]["center_to_peak"] for r in data], marker+"-", label=lattice)
    labels = ["f.15 circle", "f.15 ellipse", "f.29 circle", "f.29 ellipse"]
    for ax in axes:
        ax.set_xticks(range(4), labels, rotation=15)
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].set(ylabel="Material gain threshold (cm$^{-1}$)", title="Equal 200-um aperture / stack / reciprocal shell")
    axes[1].set(ylabel="Surface-normal intensity / peak", title="Low threshold does not ensure a bright center")
    fig.savefig(output/"matched_controls.png", dpi=190)
    plt.close(fig)
    (output/"run_status.json").write_text(json.dumps({"status":"completed", "cases":len(rows),
        "completed_at":datetime.now().isoformat(), "note":"Matched CWT controls, no universal lattice ranking"}, indent=2))


if __name__ == "__main__":
    main()
