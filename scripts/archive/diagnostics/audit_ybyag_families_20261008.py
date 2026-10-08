"""Independent fixed-period multi-TE and cap-absorption feasibility audit.

Imports the crystal main program without changing its parameters or models.
The SiN absorption scan is perturbative: it does not rebuild complex-index
Green kernels, thermal fields, or wavelength-dependent Yb cross sections.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

if __package__:
    from scripts._project_bootstrap import bootstrap_project
else:
    # Standalone archived script: first locate the helper, then let it resolve
    # the root using project marker files. Works with arbitrary PyCharm cwd.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _project_bootstrap import bootstrap_project

ROOT = bootstrap_project()

import numpy as np
from scipy.constants import c

from scripts import run_ybyag_pcsel as main_program
from pcselsim.custom_analysis import FourWaveOpticalSpec, plot_grid_convergence, solve_finite_modes_converged
from pcselsim.hybrid_vertical import solve_guided_te_modes
from pcselsim.reflector_cwt import build_reflector_coupling
from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.stratified_optics import StratifiedStack, solve_stack
from pcselsim.three_d_cwt import ThreeDCWTSettings
from pcselsim.ybyag_spatial_rates import SpatialPumpConfig, SpatialYbYAGModel, solve_spatial_ybyag_steady

GRIDS = (9, 13, 17)
TE_FAMILIES = (0, 1, 2, 3)
CAP_BULK_POWER_ABSORPTION_CM = (0.0, 0.1, 1.0, 6.9)
OUTPUT = ROOT / "results" / "ybyag_crystal_pcsel" / "independent_family_audit_20261008"


def layer_flux_overlap(result, guided, name):
    """q_abs=(n/neff) integral_layer |Theta|² dz, not energy overlap."""
    z = result.vertical_mode.z_um * 1e-6
    z = z-z[0]
    density = abs(result.vertical_mode.field)**2
    cursor = guided.padding_um*1e-6
    for layer in guided.layers:
        stop = cursor+layer.thickness_nm*1e-9
        if layer.name == name:
            samples = np.r_[cursor, z[(z>cursor)&(z<stop)], stop]
            integral = np.trapz(np.interp(samples, z, density), samples)
            return float(layer.refractive_index/result.effective_index*integral)
        cursor = stop
    raise ValueError(f"Missing layer {name}")


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, reference_guide, _, fixed_period = main_program.make_structure(te_family=0)
    all_guided = solve_guided_te_modes(reference_guide, main_program.TARGET_WAVELENGTH_NM,
                                       main_program.VERTICAL_STEP_NM, max_modes=256)
    settings = ThreeDCWTSettings(truncation_order=3, vertical_step_nm=main_program.VERTICAL_STEP_NM)
    summary = {
        "status": "independent_fixed_period_exploratory_audit_NOT_calibrated_prediction",
        "fixed_period_TE0_nm": fixed_period,
        "finite_grids": GRIDS,
        "guided_family_count_at_target_wavelength": len(all_guided),
        "all_guided_neff_at_target_wavelength": [mode.effective_index for mode in all_guided],
        "material_index_hypotheses": {
            "YbYAG_n": main_program.YBYAG_INDEX, "passive_YAG_n": main_program.PASSIVE_YAG_INDEX,
            "SiN_n": main_program.SIN_INDEX, "SiO2_n": main_program.SILICA_INDEX,
            "DBR_high_n": main_program.DBR_HIGH_INDEX,
            "imaginary_index_in_core_green": "all zero; cap absorption added perturbatively after optics",
            "dispersion": "constant indices in this audit, not measured n(lambda),k(lambda)",
        },
        "medium_defaults": asdict(YbYAGMediumConfig(dopant_density_cm3=main_program.YB_DENSITY_CM3)),
        "cross_sections_status": "unchanged defaults at all family wavelengths; spectral/temperature calibration absent",
        "absorption_status": "SiN perturbative bulk-to-modal absorption; 6.9 cm^-1 example measured at 1064 nm, not this 1030-nm process",
        "thermal_status": "no temperature, stress, photothermal or saturation-induced refractive-index feedback",
        "lateral_rate_status": "uniform x optical weighting; no measured finite-envelope lateral burn or shared multimode competition",
        "families": [],
    }
    _, _, reference_radiation, _ = main_program.make_structure(te_family=0)
    reference_dbr = StratifiedStack(reference_radiation.layers[:2*main_program.DBR_PAIRS],
                                    main_program.PASSIVE_YAG_INDEX, main_program.SILICA_INDEX)
    k0_target = 2*np.pi/(main_program.TARGET_WAVELENGTH_NM*1e-9)
    all_family_spacer_rows = []
    for family, mode in enumerate(all_guided):
        beta = k0_target*mode.effective_index
        load = solve_stack(reference_dbr, main_program.TARGET_WAVELENGTH_NM,
                           incident_side="top", q_parallel_per_m=beta)
        kappa = k0_target*np.sqrt(mode.effective_index**2-main_program.SILICA_INDEX**2)
        all_family_spacer_rows.append(dict(family=family, neff_at_target=mode.effective_index,
            spacer_roundtrip_reflected_amplitude=float(abs(load.r)*np.exp(-2*kappa*main_program.BOTTOM_SPACER_NM*1e-9))))
    summary["all_families_spacer_isolation_at_target"] = all_family_spacer_rows
    with (OUTPUT/"all_TE_spacer_isolation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_family_spacer_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_family_spacer_rows)
    absorption_rows = []
    mode_rows = []
    for family in TE_FAMILIES:
        folder = OUTPUT/f"TE{family}"
        folder.mkdir(exist_ok=True)
        cell, guide, radiation, _ = main_program.make_structure(te_family=family)
        # DO NOT use derive(family): it would redesign the period for each mode.
        wrapper = build_reflector_coupling(cell, guide, radiation, "passive YAG photonic crystal",
                                           fixed_period, main_program.TARGET_WAVELENGTH_NM, settings)
        result = wrapper.result
        widths, qenergy, qgain = main_program.active_weights(result, guide)
        qcap = layer_flux_overlap(result, guide, "unpatterned SiN upper guide")
        spec = FourWaveOpticalSpec(result.bragg_wavelength_nm, result.effective_index, result.group_index,
                                   main_program.DEVICE_SIZE_UM, main_program.INTERNAL_POWER_LOSS_CM,
                                   tuple(result.eigenvalues_m.real/100), tuple(result.eigenvalues_m.imag/100))
        print(f"[TE{family}] fixed a={fixed_period:.6f}nm lambda={result.bragg_wavelength_nm:.6f}nm "
              f"q_gain={qgain.sum():.6f} q_cap={qcap:.6f}", flush=True)
        modes, convergence = solve_finite_modes_converged(spec, GRIDS, result.coupling_m,
                                                         result.eigenvectors, result.radiation_fields)
        plot_grid_convergence(spec, convergence, folder/"finite_convergence.png")
        selected = min(modes.values(), key=lambda item: item.grid_alpha_per_m)
        up, down = wrapper.side_power_losses(selected.fields)
        raw_loss_cm = 2*selected.grid_alpha_per_m/100 + main_program.INTERNAL_POWER_LOSS_CM
        medium = YbYAGMediumConfig(laser_wavelength_nm=result.bragg_wavelength_nm,
                                  refractive_index=main_program.YBYAG_INDEX,
                                  dopant_density_cm3=main_program.YB_DENSITY_CM3,
                                  confinement_factor=min(float(qenergy.sum()), 1), spontaneous_emission_factor=0,
                                  pumped_area_um2=main_program.DEVICE_SIZE_UM**2,
                                  gain_thickness_um=main_program.GAIN_THICKNESS_NM/1000)
        pump = SpatialPumpConfig("edge", main_program.PUMP_INTENSITY_PER_EDGE_W_CM2,
                                 main_program.PUMP_INTENSITY_PER_EDGE_W_CM2,
                                 main_program.DEVICE_SIZE_UM, main_program.PUMP_CELLS)
        pump_limit = medium.pump_absorption_cross_section_cm2/(
            medium.pump_absorption_cross_section_cm2+medium.pump_emission_cross_section_cm2)
        ceiling_gain_cm = float(qgain.sum()*medium.dopant_density_cm3*(
            (medium.laser_emission_cross_section_cm2+medium.laser_absorption_cross_section_cm2)*pump_limit
            -medium.laser_absorption_cross_section_cm2))
        k0 = 2*np.pi/(result.bragg_wavelength_nm*1e-9)
        beta = k0*result.effective_index
        kappa = np.sqrt(beta**2-(main_program.SILICA_INDEX*k0)**2)
        dbr = StratifiedStack(radiation.layers[:2*main_program.DBR_PAIRS],
                              main_program.PASSIVE_YAG_INDEX, main_program.SILICA_INDEX)
        mirror_load = solve_stack(dbr, result.bragg_wavelength_nm,
                                  incident_side="top", q_parallel_per_m=beta)
        roundtrip_load = float(abs(mirror_load.r)*np.exp(-2*kappa*main_program.BOTTOM_SPACER_NM*1e-9))
        family_summary = {
            "family": family, "wavelength_nm": result.bragg_wavelength_nm,
            "period_nm": result.lattice_constant_nm, "effective_index": result.effective_index,
            "energy_overlap_gain": float(qenergy.sum()), "flux_overlap_gain": float(qgain.sum()),
            "flux_overlap_SiN_cap": qcap, "pc_overlap": result.pc_confinement,
            "selected_lowest_raw_mode": selected.name, "raw_total_power_loss_cm": raw_loss_cm,
            "upward_power_loss_cm": up/100, "downward_power_loss_cm": down/100,
            "optical_theorem_residual": wrapper.optical_theorem_residual,
            "spacer_reflected_roundtrip_amplitude": roundtrip_load,
            "default_sigma_pump_limited_modal_gain_ceiling_cm": ceiling_gain_cm,
            "cross_sections_uncalibrated": True, "absorption_scan": [],
        }
        for mode in modes.values():
            mu, md = wrapper.side_power_losses(mode.fields)
            mode_rows.append(dict(family=family, mode=mode.name, wavelength_bragg_nm=result.bragg_wavelength_nm,
                                  period_nm=fixed_period, raw_power_loss_cm=2*mode.grid_alpha_per_m/100,
                                  extrap_power_loss_cm=2*mode.alpha_per_m/100,
                                  fit_status=mode.extrapolation_status,
                                  extrap_power_sensitivity_cm=2*mode.extrapolation_uncertainty_per_m/100,
                                  upward_power_loss_cm=mu/100, downward_power_loss_cm=md/100))
        for absorption in CAP_BULK_POWER_ABSORPTION_CM:
            extra_loss_cm = absorption*qcap
            total_loss_cm = raw_loss_cm+extra_loss_cm
            model = SpatialYbYAGModel(medium, widths, qenergy, 100*total_loss_cm,
                                      up, down, c/result.group_index, qgain)
            rates = solve_spatial_ybyag_steady(model, pump)
            bulk_unlased_gain_cm = rates.unlased_modal_gain_per_m/100/float(qgain.sum())
            weighted_excitation = (bulk_unlased_gain_cm/medium.dopant_density_cm3
                                   +medium.laser_absorption_cross_section_cm2)/(
                medium.laser_emission_cross_section_cm2+medium.laser_absorption_cross_section_cm2)
            # Conditional spectral-sensitivity criterion, holding pump and sigma_a fixed.
            minimum_emission_sigma = (total_loss_cm/(medium.dopant_density_cm3*float(qgain.sum()))
                                      +medium.laser_absorption_cross_section_cm2*(1-weighted_excitation))/weighted_excitation
            row = dict(family=family, SiN_bulk_power_absorption_cm=absorption,
                       perturbative_cap_modal_absorption_cm=extra_loss_cm,
                       raw_total_power_loss_cm=total_loss_cm,
                       default_sigma_unlased_modal_gain_cm=rates.unlased_modal_gain_per_m/100,
                       finite_pump_gain_margin_cm=rates.unlased_modal_gain_per_m/100-total_loss_cm,
                       default_sigma_gain_ceiling_margin_cm=ceiling_gain_cm-total_loss_cm,
                       default_unlased_weighted_excitation=float(weighted_excitation),
                       minimum_emission_sigma_with_fixed_absorption_cm2=float(minimum_emission_sigma),
                       lasing_gate_with_default_sigma=rates.lasing_gain_condition,
                       threshold_scale_with_default_sigma=rates.threshold_pump_scale,
                       upward_W=rates.energy_budget_W["upward_output_W"],
                       downward_W=rates.energy_budget_W["downward_output_W"],
                       energy_closure_W=rates.energy_budget_W["energy_closure_residual_W"])
            absorption_rows.append(row)
            family_summary["absorption_scan"].append(row)
            print(f"  SiNabs={absorption:g} bulk cm^-1 margin={row['finite_pump_gain_margin_cm']:.6f} "
                  f"gate={row['lasing_gate_with_default_sigma']}", flush=True)
        arrays = dict(C=result.coupling_m, C1D=result.c1d_m, Crad=result.crad_m, C2D=result.c2d_m,
                      Lup=wrapper.upward_amplitude_loss_m, Ldown=wrapper.downward_amplitude_loss_m,
                      z_um=result.vertical_mode.z_um, field=result.vertical_mode.field,
                      q_energy=qenergy, q_gain=qgain)
        for name, mode in modes.items():
            arrays[f"fields_{name}"] = mode.fields
        for name, values in convergence.items():
            arrays[f"convergence_{name}"] = values
        np.savez_compressed(folder/"optical_and_finite.npz", **arrays)
        _json(folder/"summary.json", family_summary)
        summary["families"].append(family_summary)
        _json(OUTPUT/"summary.json", summary)
    for name, rows in (("family_mode_losses.csv", mode_rows), ("SiN_absorption_sensitivity.csv", absorption_rows)):
        with (OUTPUT/name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print("Completed independent fixed-period audit:", OUTPUT, flush=True)
    return summary


if __name__ == "__main__":
    run()
