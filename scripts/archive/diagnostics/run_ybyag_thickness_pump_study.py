"""Yb:YAG PCSEL thickness and equal-INTERNAL-power pump study / 独立厚度泵浦扫描。

No baseline globals are patched. Every thickness redesigns a at ~1030 nm;
this is NOT a sweep of one already fabricated fixed-period sample. The PC
retains in-plane feedback; the bottom DBR only enters the layered radiation.
Top pumping compares single pass and an explicitly idealized incoherent
re-imaging relay, NOT a coherent DBR pump standing-wave model. Use --quick.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

# This diagnostic is nested; locate the real project before importing helpers.
PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents
                    if (parent/"src"/"pcselsim").is_dir() and (parent/"configs"/"inoue2019.yaml").is_file())
sys.path.insert(0, str(PROJECT_ROOT))
from scripts._project_bootstrap import bootstrap_project
PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

from scripts import run_ybyag_pcsel as baseline
from scripts.run_ybyag_pcsel import make_structure, derive, active_weights, lateral_energy_weights
from pcselsim.custom_analysis import (FourWaveOpticalSpec, plot_grid_convergence,
                                     plot_mode_atlas, vector_far_field_metrics)
from pcselsim.hybrid_vertical import solve_guided_te_modes
from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.stratified_optics import StratifiedStack, solve_stack
from pcselsim.ybyag_spatial_rates import SpatialYbYAGModel
from pcselsim.run_output import default_output, prepare_output, write_provenance


@dataclass(frozen=True)
class StudySettings:
    thickness_um: tuple[float, ...] = (4., 8., 12., 20., 30., 50.)
    cap_nm: float = 245.
    pc_nm: float = 300.
    lower_spacer_nm: float = 2000.
    target_nm: float = 1030.
    pump_nm: float = 940.
    internal_pump_W: float = 32.
    cwt_order: int = 6
    vertical_step_nm: float = 5.
    finite_grids: tuple[int, ...] = (17, 25, 33)
    selected_te_family: int = 0
    max_te_families: int = 256
    pump_x_cells: int = 41
    gain_z_cells: int = 40
    maximum_gain_z_cell_um: float = .2
    top_passes: int = 20
    pump_return_efficiency: float = .98
    cap_scan_nm: tuple[float, ...] = (100., 150., 200., 245., 300., 400.)
    cap_scan_gain_um: float = 8.
    beam_requested_view_deg: float = 1.
    architectures: tuple[str, ...] = ("edge_single", "edge_dual", "top_single", "top_multipass20")
    quick_smoke: bool = False


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    paths = sorted((PROJECT_ROOT/"src"/"pcselsim").glob("*.py"))
    paths += [Path(__file__).resolve(), Path(baseline.__file__).resolve(),
              PROJECT_ROOT/"scripts"/"_project_bootstrap.py", PROJECT_ROOT/"requirements.txt",
              PROJECT_ROOT/"pyproject.toml"]
    return {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def verify_baseline_contract(settings):
    """Fail if an inherited hidden default changed; do not silently patch it."""
    expected = {"TARGET_WAVELENGTH_NM": settings.target_nm, "PC_DEPTH_NM": settings.pc_nm,
                "VERTICAL_STEP_NM": settings.vertical_step_nm}
    for name, value in expected.items():
        if not np.isclose(getattr(baseline, name), value):
            raise ValueError(f"Baseline {name}={getattr(baseline,name)} differs from study contract {value}; "
                             "provide an explicit parameter API instead of changing module globals")


def pump_optics(full, settings):
    """The entrance T is not full-stack T escaping past gain and DBR."""
    gain_index = next(i for i, layer in enumerate(full.layers) if layer.name == "Yb:YAG gain")
    entrance = StratifiedStack(full.layers[gain_index+1:],
                              bottom_index=baseline.YBYAG_INDEX, top_index=full.top_index)
    entry = solve_stack(entrance, settings.pump_nm, incident_side="top")
    whole = solve_stack(full, settings.pump_nm, incident_side="top")
    return dict(
        wavelength_nm=settings.pump_nm,
        entrance_R=entry.R, entrance_T=entry.T, entrance_A=entry.A,
        full_stack_R=whole.R, full_stack_T_escape_bottom=whole.T, full_stack_A=whole.A,
        entrance_layers=[asdict(layer) for layer in entrance.layers],
        interpretation="Entrance stack stops at semi-infinite YAG; full-stack T is NOT entering-gain coupling.",
        limitations=["real fixed indices borrowed from baseline; no measured 940-nm dispersion or absorption",
                     "laterally averaged passive PC, no pump diffraction",
                     "no coherent/multiple-pass pump including finite DBR return"],
    )


def material(gain_um, wavelength_nm, qenergy):
    return YbYAGMediumConfig(
        pump_wavelength_nm=940., laser_wavelength_nm=wavelength_nm,
        refractive_index=baseline.YBYAG_INDEX, dopant_density_cm3=baseline.YB_DENSITY_CM3,
        confinement_factor=min(float(qenergy.sum()), 1.),
        upper_state_lifetime_ms=baseline.LIFETIME_MS,
        pump_absorption_cross_section_cm2=baseline.PUMP_ABSORPTION_CM2,
        pump_emission_cross_section_cm2=baseline.PUMP_EMISSION_CM2,
        laser_absorption_cross_section_cm2=baseline.LASER_ABSORPTION_CM2,
        laser_emission_cross_section_cm2=baseline.LASER_EMISSION_CM2,
        spontaneous_emission_factor=0., pumped_area_um2=baseline.DEVICE_SIZE_UM**2,
        gain_thickness_um=gain_um,
    )


def optical_case(settings, gain_um, directory, reference_period_nm):
    """Same geometry/layer API and four-wave core, retaining ALL mode losses."""
    directory.mkdir()
    print(f"[optical] gain={gain_um:g} um; CWT D={settings.cwt_order}; grids={settings.finite_grids}", flush=True)
    optical, cell, guided, full = derive(gain_nm=1000*gain_um, cap_nm=settings.cap_nm,
                                        spacer_nm=settings.lower_spacer_nm,
                                        te_family=settings.selected_te_family, truncation=settings.cwt_order)
    result = optical.result
    te_modes = solve_guided_te_modes(guided, settings.target_nm, settings.vertical_step_nm,
                                    settings.max_te_families)
    zcells=max(settings.gain_z_cells, int(np.ceil(gain_um/settings.maximum_gain_z_cell_um)))
    zwidth, qenergy, qgain = active_weights(result, guided, zcells)
    spec = FourWaveOpticalSpec(result.bragg_wavelength_nm, result.effective_index, result.group_index,
                               baseline.DEVICE_SIZE_UM, baseline.INTERNAL_POWER_LOSS_CM,
                               tuple(result.eigenvalues_m.real/100), tuple(result.eigenvalues_m.imag/100),
                               grid_points=settings.finite_grids[-1])
    modes, convergence = baseline.solve_finite_modes_converged(
        spec, settings.finite_grids, result.coupling_m, result.eigenvectors, result.radiation_fields)
    spec = replace(spec, grid_points=next(iter(modes.values())).fields.shape[-1])
    chosen = min(modes.values(), key=lambda mode: mode.grid_alpha_per_m)
    best_extrap = min(modes.values(), key=lambda mode: mode.alpha_per_m)
    up, down = optical.side_power_losses(chosen.fields)
    total = 2*chosen.grid_alpha_per_m + 100*baseline.INTERNAL_POWER_LOSS_CM
    beam = asdict(vector_far_field_metrics(spec, chosen, view_deg=settings.beam_requested_view_deg, padding=16))
    arrays = dict(C_m=result.coupling_m, C1D_m=result.c1d_m, Crad_m=result.crad_m, C2D_m=result.c2d_m,
                  vertical_z_um=result.vertical_mode.z_um, vertical_field=result.vertical_mode.field,
                  vertical_index=result.vertical_mode.index, gain_cell_width_m=zwidth,
                  q_energy=qenergy, q_gain=qgain, upward_loss_operator_m=optical.upward_amplitude_loss_m,
                  downward_loss_operator_m=optical.downward_amplitude_loss_m)
    arrays.update({f"convergence_{key}": value for key, value in convergence.items()})
    losses = []
    k0_bragg=2*np.pi/(result.bragg_wavelength_nm*1e-9)
    def shifted_wavelength_nm(delta_per_m):
        return 2*np.pi/(k0_bragg+delta_per_m/result.group_index)*1e9
    for name, mode in modes.items():
        arrays.update({f"fields_{name}": mode.fields, f"radiation_x_{name}": mode.radiation_x,
                       f"radiation_y_{name}": mode.radiation_y})
        raw_delta=float(convergence[name][-1].real)
        loss = dict(mode=name, raw_amplitude_loss_cm1=mode.grid_alpha_per_m/100,
                    extrap_amplitude_loss_cm1=mode.alpha_per_m/100,
                    empirical_sensitivity_cm1=mode.extrapolation_uncertainty_per_m/100,
                    extrapolation_status=mode.extrapolation_status, band_overlap=mode.band_overlap,
                    detuning_raw_per_m=raw_delta, detuning_extrap_per_m=mode.delta_per_m,
                    wavelength_raw_shifted_nm=shifted_wavelength_nm(raw_delta),
                    wavelength_extrap_shifted_nm=shifted_wavelength_nm(mode.delta_per_m),
                    detuning_raw_wavelength_shift_nm=shifted_wavelength_nm(raw_delta)-result.bragg_wavelength_nm,
                    raw_total_power_loss_cm1=2*mode.grid_alpha_per_m/100+baseline.INTERNAL_POWER_LOSS_CM)
        losses.append(loss)
    np.savez_compressed(directory/"optical_arrays.npz", **arrays)
    write_csv(directory/"all_four_wave_losses.csv", losses)
    plot_grid_convergence(spec, convergence, directory/"grid_convergence.png")
    plot_mode_atlas(baseline.lattice_spec(result), spec, modes, directory/"all_mode_atlas.png", result.unit_cell_fields)
    baseline.plot_beam_zoom(spec, chosen, directory/"selected_raw_beam_zoom.png")
    optical_row = dict(gain_um=gain_um, cap_nm=settings.cap_nm, period_redesigned_nm=result.lattice_constant_nm,
        bragg_wavelength_nm=result.bragg_wavelength_nm, selected_te_family=settings.selected_te_family,
        guided_TE_family_count=len(te_modes), TE_count_may_be_truncated=len(te_modes)==settings.max_te_families,
        gain_energy_overlap=float(qenergy.sum()), gain_flux_overlap=float(qgain.sum()),
        gain_z_cells=zcells, minimum_gain_z_cell_um=float(zwidth.min()*1e6),
        maximum_gain_z_cell_um=float(zwidth.max()*1e6),
        pc_overlap=result.pc_confinement, selected_raw_family=chosen.name, selected_extrap_family=best_extrap.name,
        raw_amplitude_loss_cm1=chosen.grid_alpha_per_m/100,
        selected_raw_family_extrap_loss_cm1=chosen.alpha_per_m/100,
        selected_raw_family_fit_status=chosen.extrapolation_status,
        selected_raw_family_sensitivity_cm1=chosen.extrapolation_uncertainty_per_m/100,
        selected_raw_detuning_per_m=float(convergence[chosen.name][-1].real),
        selected_extrap_detuning_per_m=chosen.delta_per_m,
        selected_raw_wavelength_shifted_nm=shifted_wavelength_nm(float(convergence[chosen.name][-1].real)),
        selected_extrap_wavelength_shifted_nm=shifted_wavelength_nm(chosen.delta_per_m),
        selected_raw_wavelength_shift_nm=shifted_wavelength_nm(float(convergence[chosen.name][-1].real))-result.bragg_wavelength_nm,
        raw_total_power_loss_cm1=total/100, upward_power_loss_cm1=up/100, downward_power_loss_cm1=down/100,
        fixed_reference_period_nm=reference_period_nm,
        fixed_a_linearized_Bragg_reminder_nm=reference_period_nm*result.effective_index,
        fixed_a_reminder_is_NOT_frequency_self_consistent=True, **beam)
    optics_pump = pump_optics(full, settings)
    family_rows = [dict(family=i, neff=mode.effective_index, **mode.confinement)
                   for i, mode in enumerate(te_modes)]
    save_json(directory/"optical_report.json", dict(
        **optical_row, settings=asdict(settings), all_losses=losses, vertical_families=family_rows,
        pump_optics=optics_pump, stack=[asdict(layer) for layer in full.layers],
        selection_rule="lowest finest-grid raw loss among four retained Gamma-connected families of stated TE0",
        threshold_loss_rule="2*raw_amplitude_loss + separate_internal_power_loss; internal loss counted once",
        limitations=["TE families counted, but other families are not given full CWT/finite-loss/pump competition",
                     "raw upwind loss includes numerical diffusion, not all physical loss or heat",
                     "q_gain and q_energy retain full-mode normalization, neither is renormalized inside gain",
                     "modal detuning wavelength is first-order; C and Theta remain built at Bragg, no frozen-Theta Green frequency sensitivity performed here",
                     "spectroscopic cross sections and DBR reflection phase may change at the shifted modal wavelength",
                     "gain-thickness change modifies the DBR propagation phase as well as optical overlap",
                     "fixed-a reminder uses neff at 1030nm, not a fixed-period frequency solution"],
    ))
    return optical_row, (zwidth, qenergy, qgain, total, up, down, chosen, result), optics_pump


def pump_case(settings, architecture, gain_um, optical_data, optics_pump, directory):
    # Imported after setup: the new adapter changes pump geometry only, not old rate equations.
    from pcselsim.ybyag_pump_geometry import PumpGeometryConfig, solve_ybyag_pump_geometry
    zwidth, qenergy, qgain, total, up, down, chosen, result = optical_data
    lateral = lateral_energy_weights(chosen.fields, settings.pump_x_cells)
    medium = material(gain_um, result.bragg_wavelength_nm, qenergy)
    model = SpatialYbYAGModel(medium, zwidth, qenergy, total, up, down,
                              c/result.group_index, qgain, lateral)
    is_relay=architecture.startswith("top_multipass")
    is_top=architecture=="top_single" or is_relay
    pump = PumpGeometryConfig(architecture="top_single" if is_relay else architecture,
                              internal_incident_power_W=settings.internal_pump_W,
                              device_side_um=baseline.DEVICE_SIZE_UM, x_cells=settings.pump_x_cells,
                              top_passes=settings.top_passes if is_relay else 1,
                              return_efficiency=settings.pump_return_efficiency if is_relay else 1.)
    print(f"[pump] {architecture}: total entered gain={settings.internal_pump_W:g} W", flush=True)
    steady = solve_ybyag_pump_geometry(model, pump, find_threshold=True)
    threshold = None if steady.threshold_pump_scale is None else steady.threshold_pump_scale*settings.internal_pump_W
    budget = steady.energy_budget_W
    transmission = optics_pump["entrance_T"] if is_top else None
    external = settings.internal_pump_W/transmission if transmission and transmission > 0 else None
    external_threshold = threshold/transmission if transmission and transmission > 0 and threshold is not None else None
    row = dict(gain_um=gain_um, architecture=architecture, incident_internal_pump_W=settings.internal_pump_W,
        threshold_internal_pump_W=threshold, lasing_gain_condition=steady.lasing_gain_condition,
        upward_output_W=budget["upward_output_W"], downward_output_W=budget["downward_output_W"],
        absorbed_pump_W=budget["absorbed_pump_W"], absorbed_fraction_of_internal=budget["absorbed_pump_W"]/settings.internal_pump_W,
        pump_relay_loss_W=budget.get("pump_relay_loss_W",0.), pump_escape_W=budget.get("pump_escape_W"),
        unlased_modal_gain_cm1=steady.unlased_modal_gain_per_m/100, saturated_modal_gain_cm1=steady.modal_gain_per_m/100,
        external_top_power_estimate_W=external, external_top_threshold_estimate_W=external_threshold,
        assumed_eta0p5_external_power_W=settings.internal_pump_W/.5,
        assumed_eta0p8_external_power_W=settings.internal_pump_W/.8,
        population_residual_per_s=steady.population_residual_per_s,
        photon_fraction_residual_per_s=steady.photon_fraction_residual_per_s,
        energy_closure_residual_W=budget["energy_closure_residual_W"],
        gain_energy_overlap=float(qenergy.sum()), gain_flux_overlap=float(qgain.sum()),
        selected_raw_family=chosen.name, first_pass_internal_power_is_total_input=True,
        top_passes=settings.top_passes if is_relay else 1,
        pump_return_efficiency=settings.pump_return_efficiency if is_relay else 1.,
        pump_is_effective_single_pass=not is_relay)
    if not np.isclose(budget["incident_internal_pump_W"], settings.internal_pump_W, rtol=1e-10):
        raise RuntimeError("Pump geometry adapter did not preserve the requested total internal power")
    save_json(directory/f"pump_{architecture}.json", dict(
        **row, medium=asdict(medium), pump=asdict(pump), energy_budget_W=budget,
        photon_density_m3=steady.photon_density_m3, iterations=steady.iterations,
        notes=["threshold None means no bracketed unsaturated gain threshold, not an experimentally measured threshold",
               "CW of a single retained TE/four-wave family, not competing multimode stable operation",
               "top external estimate only uses passive entrance-stack T940, NOT full-stack T940",
               "top relay assumes ideal re-imaging, incoherent passes and stated return efficiency; not an actual pump-cavity design",
               "other cavity loss contains numerical diffusion; it is not all material heat"],
    ))
    np.savez_compressed(directory/f"pump_{architecture}_arrays.npz",
        excited_fraction=steady.excited_fraction, pump_intensity_W_cm2=steady.pump_intensity_W_cm2,
        pump_forward_W_cm2=steady.pump_forward_W_cm2, pump_backward_W_cm2=steady.pump_backward_W_cm2,
        lateral_energy_fractions=lateral, q_energy=qenergy, q_gain=qgain,
        z_edges_in_gain_um=np.r_[0,np.cumsum(zwidth)*1e6],
        x_edges_um=np.linspace(0,baseline.DEVICE_SIZE_UM,settings.pump_x_cells+1))
    return row, steady


def inversion_plot(directory, gain_um, pump_results):
    fig, axes = plt.subplots(len(pump_results), 2, figsize=(11, 3.5*len(pump_results)), squeeze=False)
    for (architecture, result), pair in zip(pump_results, axes):
        for ax, values, title in ((pair[0], result.excited_fraction, "Yb excited fraction"),
                                  (pair[1], result.pump_intensity_W_cm2, "Local total pump (W/cm2)")):
            im = ax.imshow(values.T, origin="lower", aspect="auto", extent=(0,baseline.DEVICE_SIZE_UM,0,gain_um))
            ax.set(xlabel="x (um)", ylabel="z within gain (um, bottom -> top)", title=f"{architecture}: {title}")
            fig.colorbar(im, ax=ax)
    fig.suptitle("Equal first-entry pump power; finite-pass relay is an ideal incoherent engineering assumption")
    fig.tight_layout()
    fig.savefig(directory/"pump_and_inversion_comparison.png", dpi=160)
    plt.close(fig)


def cap_scan(settings, output):
    rows = []
    for cap in settings.cap_scan_nm:
        print(f"[cap vertical-only] 8-um gain, cap={cap:g} nm", flush=True)
        _, guided, _, period = make_structure(1000*settings.cap_scan_gain_um, cap,
                                               settings.lower_spacer_nm, settings.selected_te_family)
        modes = solve_guided_te_modes(guided, settings.target_nm, settings.vertical_step_nm, settings.max_te_families)
        for family, mode in enumerate(modes):
            # Use the same quadrature as the CW gain model, without solving Crad or finite thresholds.
            proxy = type("VerticalResult", (), {"vertical_mode": mode, "effective_index": mode.effective_index})()
            zcells=max(settings.gain_z_cells,int(np.ceil(settings.cap_scan_gain_um/settings.maximum_gain_z_cell_um)))
            _, qenergy, qgain = active_weights(proxy, guided, zcells)
            rows.append(dict(cap_nm=cap, gain_um=settings.cap_scan_gain_um, family=family,
                TE_family_count=len(modes), family_count_may_be_truncated=len(modes)==settings.max_te_families,
                neff=mode.effective_index, gain_energy_overlap=float(qenergy.sum()), gain_flux_overlap=float(qgain.sum()),
                pc_overlap=mode.confinement["passive YAG photonic crystal"],
                cap_overlap=mode.confinement["unpatterned SiN upper guide"], redesigned_period_TE0_nm=period,
                calculation="vertical_only_no_loss_or_threshold_prediction"))
    write_csv(output/"cap_scan_vertical_only.csv", rows)
    selected = [row for row in rows if row["family"] == settings.selected_te_family]
    fig, axes = plt.subplots(1, 3, figsize=(13,4))
    for key in ("gain_energy_overlap", "gain_flux_overlap", "pc_overlap", "cap_overlap"):
        axes[0].plot([r["cap_nm"] for r in selected], [r[key] for r in selected], "o-", label=key)
    axes[0].legend(fontsize=7)
    axes[1].plot([r["cap_nm"] for r in selected], [r["neff"] for r in selected], "o-")
    axes[2].plot([r["cap_nm"] for r in selected], [r["TE_family_count"] for r in selected], "o-")
    for ax, label in zip(axes,("Full-mode overlap", "TE0 effective index", "Guided TE families")):
        ax.set(xlabel="Unpatterned SiN cap thickness (nm)", ylabel=label)
        ax.grid(alpha=.2)
    fig.suptitle("8-um Yb:YAG, vertical-only cap diagnostic: no threshold inferred")
    fig.tight_layout()
    fig.savefig(output/"cap_scan_vertical_only.png", dpi=170)
    plt.close(fig)
    return rows


def summary_plots(output, optical_rows, pump_rows):
    fig, axes = plt.subplots(2, 3, figsize=(15,9))
    for architecture in sorted({row["architecture"] for row in pump_rows}):
        rows = [row for row in pump_rows if row["architecture"] == architecture]
        for ax,key in zip(axes[0],("threshold_internal_pump_W","upward_output_W","absorbed_fraction_of_internal")):
            ax.plot([r["gain_um"] for r in rows], [np.nan if r[key] is None else r[key] for r in rows], "o-", label=architecture)
    for ax,label in zip(axes[0],("Threshold internal pump (W)","Top CW output (W)","Absorbed / internal pump")):
        ax.set(ylabel=label);ax.legend(fontsize=8)
    x=[row["gain_um"] for row in optical_rows]
    for key in ("gain_energy_overlap","gain_flux_overlap","pc_overlap"):
        axes[1,0].plot(x,[r[key] for r in optical_rows],"o-",label=key)
    axes[1,0].legend(fontsize=8);axes[1,0].set(ylabel="Full-mode overlap")
    axes[1,1].plot(x,[r["guided_TE_family_count"] for r in optical_rows],"o-")
    axes[1,1].set(ylabel="Guided TE-family count (not lasing modes)")
    for key in ("raw_amplitude_loss_cm1","selected_raw_family_extrap_loss_cm1"):
        axes[1,2].plot(x,[r[key] for r in optical_rows],"o-",label=key)
    axes[1,2].legend(fontsize=7);axes[1,2].set(ylabel="Amplitude loss (cm^-1)")
    for ax in axes.ravel():
        ax.set(xlabel="Yb:YAG gain thickness (um)");ax.grid(alpha=.2)
    fig.suptitle("Thickness redesign at ~1030 nm; equal internal pump; TE0-family CW feasibility only")
    fig.tight_layout()
    fig.savefig(output/"thickness_pump_summary.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thickness-um", type=float, nargs="+", default=(4,8,12,20,30,50))
    parser.add_argument("--pump-power-W", type=float, default=32., help="total effective power entering YAG, NOT per-edge power")
    parser.add_argument("--quick", action="store_true", help="one 8-um case by default, D3,N9/11/13,small pump grid; smoke only")
    parser.add_argument("--skip-top", action="store_true")
    parser.add_argument("--top-passes", type=int, default=20, help="ideal incoherent re-imaged top-pump passes; first input counted once")
    parser.add_argument("--pump-return-efficiency", type=float, default=.98)
    parser.add_argument("--output", type=Path, help="must not already exist; otherwise use timestamped results")
    args=parser.parse_args(argv)
    if any(not np.isfinite(t) or t<=0 for t in args.thickness_um) or not np.isfinite(args.pump_power_W) or args.pump_power_W<=0:
        parser.error("positive finite thicknesses and total internal pump power are required")
    if args.top_passes<1 or not np.isfinite(args.pump_return_efficiency) or not 0<=args.pump_return_efficiency<=1:
        parser.error("top-passes must be positive and return efficiency in [0,1]")
    thicknesses=tuple(float(t) for t in args.thickness_um)
    if args.quick and thicknesses==(4.,8.,12.,20.,30.,50.):
        thicknesses=(8.,)
    settings=StudySettings(thickness_um=thicknesses,internal_pump_W=args.pump_power_W,
                           top_passes=args.top_passes,pump_return_efficiency=args.pump_return_efficiency,
                           architectures=("edge_single","edge_dual") if args.skip_top else
                           ("edge_single","edge_dual","top_single",f"top_multipass{args.top_passes}"))
    if args.quick:
        settings=replace(settings,cwt_order=3,finite_grids=(9,11,13),pump_x_cells=9,gain_z_cells=12,maximum_gain_z_cell_um=.8,
                         cap_scan_nm=(150.,245.),quick_smoke=True)
    verify_baseline_contract(settings)
    output=prepare_output(args.output or default_output(PROJECT_ROOT, "legacy/ybyag_thickness_pump"), PROJECT_ROOT)
    write_provenance(output, PROJECT_ROOT)
    start_hash=source_hashes()
    baseline_parameters={key:value for key,value in vars(baseline).items() if key.isupper() and isinstance(value,(int,float,str,tuple))}
    save_json(output/"run_plan.json",dict(settings=asdict(settings),baseline_parameters=baseline_parameters,
        source_sha256_start=start_hash,python=sys.version,platform=platform.platform(),numpy=np.__version__,
        command=sys.argv,started_at=datetime.now().isoformat(),units="nm/um,m^-1 amplitude or power explicitly labelled,W,cm2 cross sections",
        scope="per-thickness period redesign; TE0 CWT only; common first-entry internal pump; top single/ideal relay; no thermal/stress/pulses"))
    reference_period=make_structure(8000.,settings.cap_nm,settings.lower_spacer_nm,settings.selected_te_family)[3]
    optical_rows=[];pump_rows=[];failures=[]
    for index,thickness in enumerate(settings.thickness_um):
        directory=output/f"case_{index+1:02d}_gain_{thickness:g}um"
        try:
            row,data,pump_optical=optical_case(settings,thickness,directory,reference_period)
            optical_rows.append(row)
            pump_results=[]
            for architecture in settings.architectures:
                try:
                    pump_row,steady=pump_case(settings,architecture,thickness,data,pump_optical,directory)
                    pump_rows.append(pump_row);pump_results.append((architecture,steady))
                except Exception as error:
                    failure=dict(gain_um=thickness,stage=architecture,error_type=type(error).__name__,error=str(error))
                    failures.append(failure);save_json(directory/f"FAILED_{architecture}.json",failure)
                    print(f"FAILED: {failure}",flush=True)
            if pump_results:
                inversion_plot(directory,thickness,pump_results)
        except Exception as error:
            failure=dict(gain_um=thickness,stage="optical",error_type=type(error).__name__,error=str(error))
            failures.append(failure);directory.mkdir(exist_ok=True);save_json(directory/"FAILED_optical.json",failure)
            print(f"FAILED: {failure}",flush=True)
        # Save incrementally: failures and completed cases are never concealed.
        write_csv(output/"thickness_optical_summary.csv",optical_rows)
        write_csv(output/"thickness_pump_summary.csv",pump_rows)
    cap_rows=cap_scan(settings,output)
    summary_plots(output,optical_rows,pump_rows)
    end_hash=source_hashes()
    changed=[path for path in sorted(set(start_hash)|set(end_hash)) if start_hash.get(path)!=end_hash.get(path)]
    save_json(output/"run_status.json",dict(status="completed" if not failures else "completed_with_failed_cases",
        completed_at=datetime.now().isoformat(),settings=asdict(settings),failures=failures,
        successful_optical_cases=len(optical_rows),successful_pump_cases=len(pump_rows),cap_family_rows=len(cap_rows),
        source_sha256_end=end_hash,source_changed_during_run=changed,
        warning="A completed CW gate is not fabrication/thermal/single-mode/pulse validation; raw finite loss has numerical diffusion"))
    print(f"Output: {output}\noptical={len(optical_rows)}, pump={len(pump_rows)}, failures={len(failures)}",flush=True)
    if failures:
        raise RuntimeError("Some study cases failed; inspect FAILED_*.json and run_status.json before using summaries")
    return output


if __name__=="__main__":
    main()
