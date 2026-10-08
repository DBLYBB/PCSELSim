"""Compare saturated normal/edge pumping of explicit Yb:YAG slab volumes.

This is an absorption and local-gain screening tool, not an optical PCSEL
optimizer. Changing doping/thickness requires a new measured index profile,
mode solver and thermal model before selecting a fabricated design.
中文：用一维饱和泵浦传播比较泵浦路线，不把厚度/掺杂扫描伪装成已验证的激光设计。
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
from scipy.integrate import trapezoid

from pcselsim.solid_state import (
    YbYAGMediumConfig, gain_per_m, solve_bidirectional_unlased_pump,
    solve_unlased_pump_propagation,
)


OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "ybyag_pump_architectures_20261008"
DOPING_MULTIPLIERS = (1.0, 5.0, 10.0)  # relative to default ~1 at.% Yb
THICKNESSES_UM = (4.0, 15.0, 100.0)
SQUARE_SIDES_UM = (500.0, 5000.0)
PUMP_INTENSITY_W_CM2 = 2.5e4


def evaluate_architectures() -> list[dict]:
    rows = []
    default = YbYAGMediumConfig(pump_intensity_W_cm2=PUMP_INTENSITY_W_CM2)
    for doping in DOPING_MULTIPLIERS:
        for thickness in THICKNESSES_UM:
            for side in SQUARE_SIDES_UM:
                for direction in ("normal_single_pass", "edge_single_pass"):
                    length = thickness if direction.startswith("normal") else side
                    entrance_area_um2 = side**2 if direction.startswith("normal") else side * thickness
                    medium = replace(default, dopant_density_cm3=default.dopant_density_cm3 * doping,
                                     gain_thickness_um=thickness, pumped_area_um2=side**2)
                    profile = solve_unlased_pump_propagation(medium, length)
                    incident = PUMP_INTENSITY_W_CM2 * entrance_area_um2 * 1e-8
                    local_gain_cm = gain_per_m(profile.unlased_excited_fraction, medium) / 100.0
                    rows.append({
                        "doping_multiplier": doping,
                        "gain_thickness_um": thickness,
                        "square_side_um": side,
                        "pump_direction": direction,
                        "pump_path_um": length,
                        "pump_entrance_area_um2": entrance_area_um2,
                        "incident_power_W": incident,
                        "saturated_absorbed_fraction": profile.net_absorbed_fraction,
                        "absorbed_power_W": incident * profile.net_absorbed_fraction,
                        "quantum_defect_heat_lower_bound_W": incident * profile.net_absorbed_fraction * (
                            1.0 - medium.pump_wavelength_nm / medium.laser_wavelength_nm
                        ),
                        "entrance_unlased_excited_fraction": float(profile.unlased_excited_fraction[0]),
                        "exit_unlased_excited_fraction": float(profile.unlased_excited_fraction[-1]),
                        "minimum_unlased_material_gain_cm-1": float(local_gain_cm.min()),
                        "mean_unlased_material_gain_cm-1": float(trapezoid(local_gain_cm, profile.distance_um) / length),
                        "assumptions": "fixed cross sections/lifetime; ideal entrance coupling; no signal/thermal solve",
                    })
    # Three paired comparisons at identical total pump power: avoid selecting
    # a heavily doped device merely because its entrance absorbs strongly.
    for doping in DOPING_MULTIPLIERS:
        source = next(row for row in rows if row["doping_multiplier"] == doping
                      and row["gain_thickness_um"] == 4.0 and row["square_side_um"] == 5000.0
                      and row["pump_direction"] == "edge_single_pass")
        row = dict(source)
        medium = replace(default, dopant_density_cm3=default.dopant_density_cm3 * doping)
        profile = solve_bidirectional_unlased_pump(medium, source["pump_path_um"])
        local_gain = gain_per_m(profile.unlased_excited_fraction, medium) / 100.0
        row.update({
            "pump_direction": "edge_bidirectional_equal_total_power",
            "saturated_absorbed_fraction": profile.net_absorbed_fraction,
            "absorbed_power_W": source["incident_power_W"] * profile.net_absorbed_fraction,
            "quantum_defect_heat_lower_bound_W": source["incident_power_W"] * profile.net_absorbed_fraction * (
                1.0 - medium.pump_wavelength_nm / medium.laser_wavelength_nm
            ),
            "entrance_unlased_excited_fraction": float(profile.unlased_excited_fraction[0]),
            "exit_unlased_excited_fraction": float(profile.unlased_excited_fraction[-1]),
            "minimum_unlased_material_gain_cm-1": float(local_gain.min()),
            "mean_unlased_material_gain_cm-1": float(trapezoid(local_gain, profile.distance_um) / source["pump_path_um"]),
        })
        rows.append(row)
    return rows


def main() -> None:
    output = OUTPUT_DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    rows = evaluate_architectures()
    with (output / "01_architecture_screen.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "00_scope_and_results.json").write_text(json.dumps({
        "scope": "pump absorption / local gain screen; no optical or thermal optimum claim",
        "pump_intensity_W_cm2": PUMP_INTENSITY_W_CM2,
        "dopant_note": "Concentration scaling alone is not validated spectroscopy or index scaling",
        "recommended_next_step": "experiment-backed guided edge pump, measured loss and index, then full-wave PC validation",
        "rows": rows,
    }, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for doping in DOPING_MULTIPLIERS:
        for side in SQUARE_SIDES_UM:
            normal = [r for r in rows if r["doping_multiplier"] == doping and r["square_side_um"] == side
                      and r["pump_direction"] == "normal_single_pass"]
            edge = [r for r in rows if r["doping_multiplier"] == doping and r["square_side_um"] == side
                    and r["pump_direction"] == "edge_single_pass"]
            label = f"{doping:g}x doping, {side/1000:g} mm side"
            axes[0].semilogx(THICKNESSES_UM, [r["saturated_absorbed_fraction"] * 100 for r in normal], "o-", label=label)
            axes[1].semilogx(THICKNESSES_UM, [r["saturated_absorbed_fraction"] * 100 for r in edge], "o-", label=label)
    for ax, title in zip(axes, ("Normal incidence", "Guided edge pumping"), strict=True):
        ax.set(xlabel="gain film thickness (um)", ylabel="saturated absorbed fraction (%)", title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle("Single-pass pump screen: fixed local intensity; entrance powers differ")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "02_normal_vs_edge_absorption.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    default = YbYAGMediumConfig(pump_intensity_W_cm2=PUMP_INTENSITY_W_CM2)
    for doping in DOPING_MULTIPLIERS:
        medium = replace(default, dopant_density_cm3=default.dopant_density_cm3 * doping)
        profile = solve_unlased_pump_propagation(medium, 5000.0)
        axes[0].plot(profile.distance_um / 1000, profile.intensity_W_cm2 / 1e3, label=f"{doping:g}x doping")
        axes[1].plot(profile.distance_um / 1000, gain_per_m(profile.unlased_excited_fraction, medium) / 100,
                     label=f"{doping:g}x doping")
    axes[0].set(xlabel="edge-pump distance (mm)", ylabel="pump intensity (kW/cm2)")
    axes[1].set(xlabel="edge-pump distance (mm)", ylabel="unlased material gain (cm-1)")
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Longer absorption path improves utilization but makes inversion nonuniform")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "03_edge_pump_nonuniformity.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for doping in DOPING_MULTIPLIERS:
        medium = replace(default, dopant_density_cm3=default.dopant_density_cm3 * doping)
        for solve, style, label in (
            (solve_unlased_pump_propagation, "--", "single end"),
            (solve_bidirectional_unlased_pump, "-", "two ends, same total power"),
        ):
            profile = solve(medium, 5000.0)
            axes[0].plot(profile.distance_um / 1000, profile.intensity_W_cm2 / 1e3, style,
                         label=f"{doping:g}x, {label}")
            axes[1].plot(profile.distance_um / 1000, gain_per_m(profile.unlased_excited_fraction, medium) / 100,
                         style, label=f"{doping:g}x, {label}")
    axes[0].set(xlabel="pump distance (mm)", ylabel="local total intensity (kW/cm2)")
    axes[1].set(xlabel="pump distance (mm)", ylabel="unlased material gain (cm-1)")
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6)
    fig.suptitle("Opposite-end pump: compare uniformity at the same total incident power")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "04_same_power_bidirectional_pump.png", dpi=190)
    plt.close(fig)
    print(f"Evaluated {len(rows)} pump configurations; outputs: {output}")
    for row in rows:
        if row["doping_multiplier"] == 1 and row["gain_thickness_um"] == 4 and row["square_side_um"] == 500:
            print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
