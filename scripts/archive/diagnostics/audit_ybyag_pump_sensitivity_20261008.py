"""Independent pump-only sensitivity of a saved 8-um crystal PCSEL mode.

No optical coupling, finite field, old solver, baseline or study is modified.
This does NOT correct frozen-Bragg frequency sensitivity, material spectra,
multi-TE competition, thermal effects or a real pump relay/DBR wave solution.
中文：固定保存的同一光学模，仅比较泵效率、有限程数、外部耦合和z网格。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "src" / "pcselsim").is_dir()
                    and (parent / "configs" / "inoue2019.yaml").is_file())
sys.path.insert(0, str(PROJECT_ROOT))
from scripts._project_bootstrap import bootstrap_project
PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
from scipy.integrate import trapezoid

from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.ybyag_spatial_rates import SpatialYbYAGModel
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig, solve_ybyag_pump_geometry
from pcselsim.run_output import default_output, prepare_output, write_provenance

DEFAULT_SOURCE = PROJECT_ROOT / "results" / "archive" / "legacy_before_20261021" / "ybyag_thickness_pump" / "20261008_180021_822771" / "case_02_gain_8um"


def write_json(path, value):
    def default(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def hashes(source):
    paths = [Path(__file__).resolve(), PROJECT_ROOT / "scripts" / "run_ybyag_pcsel.py",
             PROJECT_ROOT / "scripts" / "archive" / "diagnostics" / "run_ybyag_thickness_pump_study.py",
             PROJECT_ROOT / "src" / "pcselsim" / "solid_state.py",
             PROJECT_ROOT / "src" / "pcselsim" / "ybyag_spatial_rates.py",
             PROJECT_ROOT / "src" / "pcselsim" / "ybyag_pump_geometry.py"]
    paths += [source / name for name in ("optical_report.json", "optical_arrays.npz",
                                       "pump_edge_dual.json", "pump_edge_dual_arrays.npz")]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def load_saved_model(source):
    report = json.loads((source / "optical_report.json").read_text(encoding="utf-8"))
    pump_report = json.loads((source / "pump_edge_dual.json").read_text(encoding="utf-8"))
    with np.load(source / "optical_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    with np.load(source / "pump_edge_dual_arrays.npz", allow_pickle=False) as archive:
        lateral = archive["lateral_energy_fractions"].copy()
    if not np.isclose(report["gain_um"], 8.0):
        raise ValueError("This dated sensitivity contract is the saved 8-um candidate")
    neff = report["bragg_wavelength_nm"] / report["period_redesigned_nm"]
    z = arrays["vertical_z_um"] * 1e-6
    z = z - z[0]
    density = np.abs(arrays["vertical_field"])**2
    norm_energy = float(trapezoid(arrays["vertical_index"]**2 * density, z))
    ng = norm_energy / neff  # the same non-dispersive TE identity as 3D CWT
    medium = YbYAGMediumConfig(**pump_report["medium"])
    model = SpatialYbYAGModel(
        medium, arrays["gain_cell_width_m"], arrays["q_energy"],
        100.0 * report["raw_total_power_loss_cm1"],
        100.0 * report["upward_power_loss_cm1"],
        100.0 * report["downward_power_loss_cm1"], c / ng,
        arrays["q_gain"], lateral,
    )
    np.testing.assert_allclose(model.optical_gain_weights,
                               model.optical_energy_fractions * ng / medium.refractive_index, rtol=1e-11)
    # Reconstruct physical guided-layer position without importing/editing the
    # baseline or re-solving the optical eigenproblem. Numerical padding is the
    # saved grid length minus finite guide-layer thickness, divided by two.
    guide_layers = [layer for layer in report["stack"] if not layer["name"].startswith("DBR_")]
    padding = (z[-1] - sum(layer["thickness_nm"] * 1e-9 for layer in guide_layers)) / 2.0
    cursor = padding
    for layer in guide_layers:
        if layer["name"] == "Yb:YAG gain":
            gain_start, gain_stop = cursor, cursor + layer["thickness_nm"] * 1e-9
            break
        cursor += layer["thickness_nm"] * 1e-9
    else:
        raise ValueError("Exactly identified Yb:YAG gain layer required")
    np.testing.assert_allclose(gain_stop - gain_start, model.gain_thickness_m, rtol=1e-11)
    integration = dict(z=z, density=density, norm_energy=norm_energy,
                       neff=neff, group_index=ng, gain_start=gain_start, gain_stop=gain_stop,
                       reconstructed_padding_um=padding * 1e6)
    return model, integration, report, pump_report


def regrid_gain(model, integration, cells):
    """Integrate saved piecewise-linear |Theta|², not renormalized cell samples."""
    edges = np.linspace(integration["gain_start"], integration["gain_stop"], cells + 1)
    z, density = integration["z"], integration["density"]
    integrals = []
    for left, right in zip(edges[:-1], edges[1:]):
        points = np.r_[left, z[(z > left) & (z < right)], right]
        integrals.append(trapezoid(np.interp(points, z, density), points))
    integrals = np.asarray(integrals)
    index = model.medium.refractive_index
    energy = index**2 * integrals / integration["norm_energy"]
    gain = index / integration["neff"] * integrals
    np.testing.assert_allclose(energy.sum(), model.optical_energy_fractions.sum(), rtol=1e-11)
    np.testing.assert_allclose(gain.sum(), model.optical_gain_weights.sum(), rtol=1e-11)
    return replace(model, cell_widths_m=np.diff(edges), optical_energy_fractions=energy,
                   optical_gain_weights=gain)


def record_result(sweep, value, model, pump, result, coupling=None, external_W=None):
    budget = result.energy_budget_W
    threshold = (None if result.threshold_pump_scale is None
                 else result.threshold_pump_scale * pump.internal_incident_power_W)
    return dict(
        sweep=sweep, varied_value=value, architecture=pump.architecture,
        gain_um=model.gain_thickness_m * 1e6, gain_z_cells=len(model.cell_widths_m),
        x_cells=pump.x_cells, top_passes=pump.top_passes, return_efficiency=pump.return_efficiency,
        assumed_edge_external_coupling=coupling, incident_external_pump_W=external_W,
        threshold_internal_pump_W=threshold,
        threshold_external_pump_W=None if coupling is None or threshold is None else threshold / coupling,
        lasing_gain_condition=result.lasing_gain_condition, photon_density_m3=result.photon_density_m3,
        unlased_modal_gain_cm1=result.unlased_modal_gain_per_m / 100.0,
        saturated_modal_gain_cm1=result.modal_gain_per_m / 100.0,
        gain_energy_overlap=float(model.optical_energy_fractions.sum()),
        gain_flux_overlap=float(model.optical_gain_weights.sum()),
        population_residual_per_s=result.population_residual_per_s,
        photon_fraction_residual_per_s=result.photon_fraction_residual_per_s,
        **budget,
    )


def plot_power_sensitivity(output, name, rows, xlabel):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = [row["varied_value"] for row in rows]
    axes[0].plot(x, [row["upward_output_W"] for row in rows], "o-")
    axes[0].set(ylabel="Conditional upward CW output (W)")
    axes[1].plot(x, [row["threshold_internal_pump_W"] for row in rows], "o-", label="Internal threshold")
    if any(row["threshold_external_pump_W"] is not None for row in rows):
        axes[1].plot(x, [row["threshold_external_pump_W"] for row in rows], "s-", label="Assumed external threshold")
    axes[1].set(ylabel="Threshold pump power (W)"); axes[1].legend(fontsize=8)
    for key in ("absorbed_pump_W", "pump_relay_loss_W", "pump_escape_W"):
        axes[2].plot(x, [row[key] for row in rows], "o-", label=key)
    axes[2].set(ylabel="First-entry pump budget (W)"); axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set(xlabel=xlabel); ax.grid(alpha=.2)
    fig.suptitle("Saved 8-um frozen optical mode: pump assumptions only, not experimental validation")
    fig.tight_layout(); fig.savefig(output / name, dpi=170); plt.close(fig)


def plot_z_sensitivity(output, rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for architecture in ("edge_dual", "top_single"):
        selected = [row for row in rows if row["architecture"] == architecture]
        label = "side dual" if architecture == "edge_dual" else "top 20 passes, eta=0.98"
        axes[0].plot([row["gain_z_cells"] for row in selected], [row["upward_output_W"] for row in selected], "o-", label=label)
        axes[1].plot([row["gain_z_cells"] for row in selected], [row["threshold_internal_pump_W"] for row in selected], "o-", label=label)
    axes[0].set(ylabel="Conditional upward CW output (W)")
    axes[1].set(ylabel="Internal threshold (W)")
    for ax in axes:
        ax.set(xlabel="Active z control volumes"); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle("Exact saved-field weights: numerical z sensitivity, not optical C/frequency convergence")
    fig.tight_layout(); fig.savefig(output / "gain_z_grid_sensitivity.png", dpi=170); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, help="new directory only; default results/current/legacy/ybyag_pump_sensitivity/timestamp")
    args = parser.parse_args(argv)
    source = args.source.resolve()
    source_hashes = hashes(source)
    model, integration, report, original_pump = load_saved_model(source)
    output = prepare_output(args.output or default_output(PROJECT_ROOT, "legacy/ybyag_pump_sensitivity"), PROJECT_ROOT)
    write_provenance(output, PROJECT_ROOT)
    write_json(output / "run_status.json", {"status": "running", "source": source,
                                           "source_sha256_start": source_hashes})
    side_um = float(np.sqrt(model.medium.pumped_area_um2))
    nx = len(model.lateral_energy_fractions)
    cache = {}

    def solve(local_model, architecture="top_single", power_W=32.0, passes=20, efficiency=.98):
        is_top = architecture == "top_single"
        pump = PumpGeometryConfig(architecture, power_W, side_um, nx,
                                  top_passes=passes if is_top else 1,
                                  return_efficiency=efficiency if is_top else 1.0)
        key = (len(local_model.cell_widths_m), architecture, power_W, pump.top_passes, pump.return_efficiency)
        if key not in cache:
            print(f"[pump] Nz={key[0]}, {architecture}, internal={power_W:g} W, passes={pump.top_passes}, return={pump.return_efficiency:g}", flush=True)
            cache[key] = solve_ybyag_pump_geometry(local_model, pump)
        return pump, cache[key]

    baseline_pump, baseline = solve(model, "edge_dual")
    recovered = baseline.energy_budget_W["upward_output_W"]
    np.testing.assert_allclose(recovered, original_pump["upward_output_W"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(recovered, 0.7760243421, rtol=1e-8, atol=1e-10)
    print(f"[recovery] side32W upward={recovered:.12g} W, saved={original_pump['upward_output_W']:.12g} W", flush=True)
    rows = []
    for efficiency in (.80, .85, .90, .95, .98, 1.0):
        pump, result = solve(model, efficiency=efficiency)
        rows.append(record_result("top_return_efficiency", efficiency, model, pump, result))
    for passes in (1, 2, 4, 8, 20, 40):
        pump, result = solve(model, passes=passes)
        rows.append(record_result("top_pass_count", passes, model, pump, result))
    for coupling in (.25, .5, .8, 1.0):
        pump, result = solve(model, "edge_dual", power_W=32.0 * coupling)
        rows.append(record_result("edge_external_coupling", coupling, model, pump, result,
                                  coupling=coupling, external_W=32.0))
    z_rows = []
    for cells in (40, 80, 160):
        local = regrid_gain(model, integration, cells)
        for architecture in ("edge_dual", "top_single"):
            pump, result = solve(local, architecture)
            row = record_result("gain_z_grid", cells, local, pump, result)
            z_rows.append(row); rows.append(row)
    grid_changes = {}
    for architecture in ("edge_dual", "top_single"):
        selected = [row for row in z_rows if row["architecture"] == architecture]
        low, high = selected[0], selected[-1]
        grid_changes[architecture] = {
            "upward_output_160_over_40_relative_change": high["upward_output_W"] / low["upward_output_W"] - 1,
            "threshold_160_over_40_relative_change": high["threshold_internal_pump_W"] / low["threshold_internal_pump_W"] - 1,
        }
    write_csv(output / "all_pump_sensitivities.csv", rows)
    for sweep, filename, xlabel in (
        ("top_return_efficiency", "top_return_efficiency.png", "Assumed per-return survival efficiency"),
        ("top_pass_count", "top_pass_count.png", "Ideal incoherent pass count (return=0.98)"),
        ("edge_external_coupling", "edge_external_coupling.png", "Assumed external-to-internal side coupling"),
    ):
        selected = [row for row in rows if row["sweep"] == sweep]
        write_csv(output / f"{sweep}.csv", selected)
        plot_power_sensitivity(output, filename, selected, xlabel)
    write_csv(output / "gain_z_grid.csv", z_rows)
    plot_z_sensitivity(output, z_rows)
    payload = dict(
        source=source, selected_frozen_mode=report["selected_raw_family"],
        recovered_side32W_upward_output_W=recovered,
        saved_side32W_upward_output_W=original_pump["upward_output_W"],
        reconstructed_effective_index=integration["neff"], reconstructed_group_index=integration["group_index"],
        reconstructed_padding_um=integration["reconstructed_padding_um"],
        optical_C_changed=False, source_sha256_start=source_hashes, rows=rows,
        gain_z_grid_sensitivity=grid_changes,
        limitations=["All return and side-coupling efficiencies are assumptions, not measured couplers",
                     "Top relay is noncoherent ideal re-imaging; no physical DBR pump-return boundary or standing wave",
                     "Threshold is first-entry INTERNAL W; only edge coupling scan gives external threshold by dividing eta",
                     "Same saved optical C, losses, finite field and full-mode q normalization; no optical frequency correction",
                     "z refinement integrates the saved piecewise-linear |Theta|^2 exactly; it is not a new optical vertical solve",
                     "No true spectral, thermal or shared multimode competition validation"],
    )
    write_json(output / "summary.json", payload)
    end_hashes = hashes(source)
    changed = [path for path in source_hashes if source_hashes[path] != end_hashes[path]]
    write_json(output / "run_status.json", dict(status="complete" if not changed else "complete_source_changed",
               completed_at=datetime.now().isoformat(), source_sha256_end=end_hashes,
               source_changed_during_run=changed, result_rows=len(rows)))
    print(f"Output: {output}\nRecovery={recovered:.12g} W; rows={len(rows)}; grid sensitivity={grid_changes}", flush=True)
    return output


if __name__ == "__main__":
    main()
