"""Recheck the previous 300-um recommendation without overwriting old outputs.

Compare extrapolation windows and sampled far fields independently.  This is
an audit calculation, not a new design optimizer. / 检验历史推荐的数值可信区间。
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project
PROJECT_ROOT = bootstrap_project()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIRECTORY))

import numpy as np

from run_best_triangular_three_triangle_pcsel import build_model
from pcselsim.numerical_quality import extrapolate_grid_loss
from pcselsim.triangular_finite import (
    TriangularFiniteSpec,
    solve_triangular_finite_modes,
    triangular_vector_far_field,
)


def main():
    output = PROJECT_ROOT / "results" / "audit_20261008" / "optical_convergence"
    output.mkdir(parents=True, exist_ok=True)
    candidate, cell, layers, model = build_model()
    geometry_fingerprint = hashlib.sha256(
        json.dumps(asdict(cell), sort_keys=True).encode("utf-8")
    ).hexdigest()
    grids = (7, 9, 11, 13, 15, 17, 21, 25)
    table = []
    cache = output / "grid_and_window_metrics.csv"
    identity = output / "extrapolation_audit.json"
    if cache.exists() and identity.exists():
        previous = json.loads(identity.read_text(encoding="utf-8"))
        if previous.get("geometry_fingerprint") == geometry_fingerprint and previous["design"] == candidate.name and abs(
            previous["bragg_wavelength_nm"] - model.bragg_wavelength_nm
        ) < 1e-4:
            with cache.open(encoding="utf-8-sig", newline="") as stream:
                table = [
                    {key: (value if key == "mode" else float(value) if value else None) for key, value in row.items()}
                    for row in csv.DictReader(stream)
                ]
    for grid in grids:
        if len([row for row in table if row["grid"] == grid]) == 6:
            print(f"Using matching audit cache N={grid}", flush=True)
            continue
        print(f"Optical audit: 300-um square, characteristic N={grid}", flush=True)
        modes = solve_triangular_finite_modes(
            model, TriangularFiniteSpec(radius_um=150.0, radius_cells=grid, aperture_shape="square")
        )
        for name, mode in modes.items():
            far = triangular_vector_far_field(mode, model.bragg_wavelength_nm, view_deg=1.5)
            radiation_matrix = (model.cr_m - model.cr_m.conj().T) / (2j)
            norm = float(np.sum(np.abs(mode.fields) ** 2))
            radiation_loss = float(np.einsum(
                "ap,ab,bp->", mode.fields.conj(), radiation_matrix, mode.fields
            ).real / norm)
            numerical_diffusion = 0.0
            for channel, (dj, dk) in enumerate(((1, 0), (1, 0), (0, 1), (0, 1), (1, 1), (1, 1))):
                for left, (j, k) in enumerate(zip(mode.grid.axial_j, mode.grid.axial_k)):
                    right = mode.grid.index.get((j + dj, k + dk))
                    if right is not None:
                        numerical_diffusion += abs(mode.fields[channel, left] - mode.fields[channel, right]) ** 2
            numerical_diffusion /= 2.0 * mode.grid.spacing_m * norm
            table.append({
                "grid": grid, "mode": name,
                "alpha_grid_cm-1": mode.alpha_per_m / 100.0,
                "delta_grid_m-1": mode.delta_per_m,
                "band_overlap": mode.band_overlap,
                "window_rms_diameter_deg": far.full_rms_divergence_deg,
                "E0p5_window_fraction": far.encircled_power_0p5deg,
                "center_to_peak": far.center_to_peak,
                "evaluated_view_deg": far.evaluated_view_deg,
                "alias_free_square_view_deg": far.alias_free_square_view_deg,
                "direct_radiation_loss_cm-1": radiation_loss / 100.0,
                "upwind_volume_diffusion_cm-1": numerical_diffusion / 100.0,
                "remaining_boundary_loss_cm-1": (mode.alpha_per_m - radiation_loss - numerical_diffusion) / 100.0,
            })
    with (output / "grid_and_window_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        columns = list(dict.fromkeys(key for row in table for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    report = {
        "design": candidate.name, "bragg_wavelength_nm": model.bragg_wavelength_nm,
        "geometry_fingerprint": geometry_fingerprint,
        "normalization": "E0p5 and RMS are conditional on the sampled angular window",
        "fits": {},
    }
    for first, last in ((0, 3), (1, 4), (0, 4), (2, 5), (3, 6), (5, 8), (0, 8)):
        fit_grids = grids[first:last]
        entries = {}
        for name in ("A", "B1", "B2", "C", "D1", "D2"):
            losses = np.asarray([
                next(row["alpha_grid_cm-1"] for row in table if row["grid"] == grid and row["mode"] == name) * 100.0
                for grid in fit_grids
            ])
            accepted, raw, sensitivity, status = extrapolate_grid_loss(1.0 / np.asarray(fit_grids), losses)
            entries[name] = {
                "alpha_cm-1": accepted / 100.0,
                "raw_intercept_cm-1": raw / 100.0,
                "empirical_sensitivity_cm-1": sensitivity / 100.0,
                "status": status,
            }
        report["fits"][str(fit_grids)] = entries
    (output / "extrapolation_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
