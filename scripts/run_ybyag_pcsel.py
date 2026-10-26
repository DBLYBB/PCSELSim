"""Crystal-gain PCSEL: independent layered radiation + spatial Yb gain.

PyCharm: edit PARAMETER PANEL / STEPS, then right click this file -> Run.
中文：上下透明层只导光；底部另置有限DBR。此候选是需实验/全波验证的设计假设。
No semiconductor model is overwritten. No mode-locking is implied by CW gain.

Configuration scope / 配置范围：本文件是可修改厚度/孔径的开发入口；默认参数
不等于 matched300 对照组。300 um 四腔比较请使用 matched300_yag_search.py
及已保存的正式配置，避免混用不同孔径、层厚和扫描结果。
"""

from __future__ import annotations
import argparse
import csv
import json
import hashlib
import os
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project
PROJECT_ROOT = bootstrap_project()
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
from pcselsim.geometry import Ellipse, SquareLatticeCell
from pcselsim.vertical import Layer
from pcselsim.hybrid_vertical import HybridLayerStack, solve_guided_te_modes
from pcselsim.stratified_optics import (
    StratifiedLayer,
    StratifiedStack,
    quarter_wave_dbr,
    solve_stack,
)
from pcselsim.reflector_cwt import build_reflector_coupling
from pcselsim.three_d_cwt import ThreeDCWTSettings
from pcselsim.custom_analysis import (
    FourWaveOpticalSpec,
    LatticeSpec,
    LatticeInclusionSpec,
    LayerSpec,
    plot_lattice,
    plot_layer_stack,
    plot_k_space,
    plot_vertical_mode,
    solve_finite_modes_converged,
    plot_grid_convergence,
    plot_mode_atlas,
    plot_vector_far_field_diagnostics,
    vector_far_field_metrics,
    vector_far_field_complex,
    write_mode_table,
)
from pcselsim.band_structure import square_four_wave_band_diagram, plot_band_diagram
from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.ybyag_spatial_rates import (
    SpatialYbYAGModel,
    SpatialPumpConfig,
    solve_spatial_ybyag_steady,
)

# ===================== PARAMETER PANEL / 可修改参数 =====================
TARGET_WAVELENGTH_NM = 1030.0
GAIN_THICKNESS_NM = 8000.0  # thickness scan tests whether extra YAG helps
YBYAG_INDEX = 1.8166  # 2016 10%-Yb:YAG ceramic measured @1029nm
PASSIVE_YAG_INDEX = 1.8154
SILICA_INDEX = 1.45  # nominal film index; measure n,k before fabrication
SIN_INDEX = 2.00  # deposition-dependent, not a universal constant
PC_DEPTH_NM = 300.0
CAP_THICKNESS_NM = 245.0  # unpatterned high-index guide, NOT an upper mirror
TOP_SILICA_NM = 300.0
BOTTOM_SPACER_NM = 2000.0  # separates guided mode from the distinct bottom DBR
FILL_FRACTION = 0.10
SMALL_HOLE_AREA_SHARE = 0.42
HOLE_DISPLACEMENT_A = 0.45  # two circular SiO2-filled holes, easier than sharp tips
DEVICE_SIZE_UM = 1000.0  # 1-mm aperture: not a tiny single-mode microdisk
INTERNAL_POWER_LOSS_CM = (
    0.10  # ASSUMPTION, includes material/scatter but no thermal model
)
DBR_PAIRS = 16
DBR_HIGH_INDEX = 2.10  # Ta2O5, measured thin-film optical constants required
DBR_CENTER_NM = 1030.0
DBR_FIRST_LAYER = "low"  # order is bottom -> top, last layer touches spacer
TE_FAMILY = 0  # reference-period / legacy helper only; active search uses ACTIVE_TE_FAMILIES
ACTIVE_TE_FAMILIES = (0, 1, 2, 3)
VERTICAL_STEP_NM = 5.0
CWT_TRUNCATION = 6
FINITE_GRIDS = (17, 25, 33)
PUMP_INTENSITY_PER_EDGE_W_CM2 = (
    2.0e5  # legacy/reference helper only; new main holds INTERNAL_TOTAL_PUMP_POWER_W fixed
)
INTERNAL_TOTAL_PUMP_POWER_W = 32.0  # all incident beams combined, already coupled inside the gain
YB_DENSITY_CM3 = 1.38e21  # nominal 10-at.% site density; verify actual crystal doping
LIFETIME_MS = 0.95
PUMP_ABSORPTION_CM2 = (
    0.70e-20  # room-temperature nominal, must replace by sample spectrum
)
PUMP_EMISSION_CM2 = 0.10e-20
LASER_ABSORPTION_CM2 = 0.126e-20
LASER_EMISSION_CM2 = 2.00e-20
PUMP_CELLS = 41
GAIN_Z_CELLS = 40
STEPS = {
    "01_structure": True,  # physical full stack, PC and vertical families
    "02_bands": True,  # local four-wave M-Gamma-X, not full BZ photonic bands
    "03_finite_modes": True,  # closed-frequency modes; mesh sensitivity at that frequency
    "04_fields": True,  # whole-device envelope, cell field, top vector farfield
    "05_spatial_gain": True,  # selected pump geometry + shared inversion + static CW closure
    "06_design_scan": True,  # family/search scope; vary geometry through separate fresh runs
}


def make_structure(
    gain_nm=GAIN_THICKNESS_NM,
    cap_nm=CAP_THICKNESS_NM,
    spacer_nm=BOTTOM_SPACER_NM,
    te_family=TE_FAMILY,
):
    radii = np.sqrt(
        FILL_FRACTION
        * np.array([SMALL_HOLE_AREA_SHARE, 1 - SMALL_HOLE_AREA_SHARE])
        / np.pi
    )
    centres = (-HOLE_DISPLACEMENT_A / 2, HOLE_DISPLACEMENT_A / 2)
    cell = SquareLatticeCell(
        PASSIVE_YAG_INDEX**2,
        tuple(
            Ellipse((d, d), (float(r), float(r)), 0, SILICA_INDEX**2)
            for d, r in zip(centres, radii)
        ),
    )
    average_index = float(np.sqrt(cell.fourier_epsilon(0, 0).real))
    layers = (
        Layer("lower SiO2 guide spacer", spacer_nm, SILICA_INDEX),
        Layer("Yb:YAG gain", gain_nm, YBYAG_INDEX),
        Layer("passive YAG photonic crystal", PC_DEPTH_NM, average_index),
        Layer("unpatterned SiN upper guide", cap_nm, SIN_INDEX),
        Layer("SiO2 upper cladding", TOP_SILICA_NM, SILICA_INDEX),
    )
    guided = HybridLayerStack(layers, 1.0, SILICA_INDEX, 2.0, te_family)
    mode = guided.solve_te0(TARGET_WAVELENGTH_NM, VERTICAL_STEP_NM)
    period_nm = TARGET_WAVELENGTH_NM / mode.effective_index
    dbr = quarter_wave_dbr(
        DBR_CENTER_NM,
        DBR_HIGH_INDEX,
        SILICA_INDEX,
        DBR_PAIRS,
        first_layer=DBR_FIRST_LAYER,
    )
    full = StratifiedStack(
        dbr
        + tuple(
            StratifiedLayer(l.name, l.thickness_nm, l.refractive_index) for l in layers
        ),
        bottom_index=PASSIVE_YAG_INDEX,
        top_index=1.0,
    )
    return cell, guided, full, period_nm


def derive(
    gain_nm=GAIN_THICKNESS_NM,
    cap_nm=CAP_THICKNESS_NM,
    spacer_nm=BOTTOM_SPACER_NM,
    te_family=TE_FAMILY,
    truncation=CWT_TRUNCATION,
):
    cell, guided, full, period = make_structure(gain_nm, cap_nm, spacer_nm, te_family)
    optical = build_reflector_coupling(
        cell,
        guided,
        full,
        "passive YAG photonic crystal",
        period,
        TARGET_WAVELENGTH_NM,
        ThreeDCWTSettings(truncation, VERTICAL_STEP_NM),
    )
    return optical, cell, guided, full


def active_weights(result, guided, cells=GAIN_Z_CELLS):
    """Exact integrals of piecewise-linear |Theta|², preserving full-mode norm.

    q_energy=n²|Theta|² dz / int(n²|Theta|² dz), q_gain=(n/neff)|Theta|² dz.
    These are not normalized to one inside the gain layer!

    中文：q_energy 与 q_gain 保留全纵向模归一化，增益层内权重和通常小于 1。
    不可在增益层内重新归一化，否则会丢失约束因子并低估材料增益阈值。
    """
    z = result.vertical_mode.z_um * 1e-6
    z -= z[0]
    cursor = guided.padding_um * 1e-6
    for layer in guided.layers:
        if layer.name == "Yb:YAG gain":
            break
        cursor += layer.thickness_nm * 1e-9
    edges = np.linspace(cursor, cursor + layer.thickness_nm * 1e-9, cells + 1)
    density = abs(result.vertical_mode.field) ** 2
    trap = getattr(np, "trapezoid", np.trapz)
    norm_energy = trap(result.vertical_mode.index**2 * density, z)
    weights = []
    for left, right in zip(edges[:-1], edges[1:]):
        samples = np.r_[left, z[(z > left) & (z < right)], right]
        weights.append(trap(np.interp(samples, z, density), samples))
    weights = np.asarray(weights)
    return (
        np.diff(edges),
        YBYAG_INDEX**2 * weights / norm_energy,
        YBYAG_INDEX / result.effective_index * weights,
    )


def lattice_spec(result):
    a = result.lattice_constant_nm
    rs = (
        np.sqrt(
            FILL_FRACTION
            * np.array([SMALL_HOLE_AREA_SHARE, 1 - SMALL_HOLE_AREA_SHARE])
            / np.pi
        )
        * a
    )
    inc = tuple(
        LatticeInclusionSpec("circle", float(r), float(r), d * a, d * a)
        for d, r in zip((-HOLE_DISPLACEMENT_A / 2, HOLE_DISPLACEMENT_A / 2), rs)
    )
    return LatticeSpec(
        "square", a, "circle", 0, 0, 0, PASSIVE_YAG_INDEX, SILICA_INDEX, inc
    )


def lateral_energy_weights(fields, cells):
    """Conservative x-bin projection of the SAME finite mode, integrating y.

    No interpolation-generated gain: integrate overlap of uniform finite cells.
    This projection alone does not solve y-resolved inversion or mode competition.

    中文：只做守恒的 y 积分和 x 分箱，所得权重供共享反转求解器使用；
    投影自身不求反转，也不引入额外的模式竞争物理。
    """
    density = np.sum(abs(np.asarray(fields)) ** 2, axis=(0, 1))
    if density.ndim != 1 or density.sum() <= 0:
        raise ValueError("Expected nonzero fields[4,y,x]")
    old_edges = np.linspace(0, 1, len(density) + 1)
    new_edges = np.linspace(0, 1, cells + 1)
    overlap = np.maximum(
        0,
        np.minimum(new_edges[1:, None], old_edges[None, 1:])
        - np.maximum(new_edges[:-1, None], old_edges[None, :-1]),
    )
    return overlap @ (density / density.sum()) * len(density)


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: float(x)),
        encoding="utf-8",
    )


def plot_beam_zoom(spec, mode, path):
    """Display-only zoom; reported beam metrics retain their wider angle window."""
    angle, fx, fy = vector_far_field_complex(
        mode.radiation_x,
        mode.radiation_y,
        spec.wavelength_nm,
        spec.domain_um,
        view_deg=0.2,
        padding=16,
    )
    power = abs(fx) ** 2 + abs(fy) ** 2
    power /= power.max()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(
        power,
        origin="lower",
        cmap="inferno",
        extent=(angle[0], angle[-1], angle[0], angle[-1]),
    )
    axes[0].set(
        xlabel="theta x (deg)",
        ylabel="theta y (deg)",
        title=f"Mode {mode.name}: top-output beam zoom",
    )
    centre = int(np.argmin(abs(angle)))
    axes[1].plot(angle, power[centre], label="x cut")
    axes[1].plot(angle, power[:, centre], label="y cut")
    axes[1].set(
        xlabel="Angle (deg)",
        ylabel="Normalized intensity",
        title="Top-output angular cuts (display zoom)",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def design_scan(output):
    rows = []
    # Vertical-only scan is explicit: no threshold is inferred from overlap alone.
    for thickness in (4000, 8000, 15000, 30000, 50000):
        for cap in (150, 230, 245, 270, 400):
            _, stack, _, a = make_structure(thickness, cap)
            modes = solve_guided_te_modes(
                stack, TARGET_WAVELENGTH_NM, VERTICAL_STEP_NM, 12
            )
            for family, m in enumerate(modes):
                rows.append(
                    dict(
                        gain_nm=thickness,
                        cap_nm=cap,
                        family=family,
                        neff=m.effective_index,
                        gain_overlap=m.confinement["Yb:YAG gain"],
                        pc_overlap=m.confinement["passive YAG photonic crystal"],
                        guide_overlap=m.confinement["unpatterned SiN upper guide"],
                        period_TE0_nm=a,
                    )
                )
    with (output / "06_thickness_cap_families.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    # Scan reflector phase using same field/period/PC. Spacer variation is small
    # relative to evanescent decay length separation; full Green is re-evaluated.
    phase = []
    for spacer in np.linspace(2000, 2355, 9):
        optical, _, _, _ = derive(spacer_nm=float(spacer), truncation=3)
        rad = np.linalg.eigvalsh(
            (optical.result.crad_m - optical.result.crad_m.conj().T) / (2j)
        )
        phase.append((spacer, float(rad.sum()), optical.optical_theorem_residual))
    np.savetxt(
        output / "06_spacer_phase.csv",
        phase,
        delimiter=",",
        header="spacer_nm,trace_amplitude_radiation_loss_per_m,optical_theorem_residual",
        comments="",
    )
    fig, ax = plt.subplots()
    ax.plot(np.asarray(phase)[:, 0], np.asarray(phase)[:, 1] / 100, "o-")
    ax.set(
        xlabel="Lower spacer thickness (nm)",
        ylabel="Trace radiation amplitude loss (cm^-1)",
        title="DBR phase changes Crad, not only output direction",
    )
    fig.tight_layout()
    fig.savefig(output / "06_spacer_phase.png", dpi=180)
    plt.close(fig)


def frozen_reference_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--quick", action="store_true", help="coarse smoke test, not validated design"
    )
    parser.add_argument("--only", nargs="+", choices=STEPS)
    args = parser.parse_args(argv)
    selected = {
        k: v and (args.only is None or k in args.only) for k, v in STEPS.items()
    }
    output = (
        args.output
        or PROJECT_ROOT
        / "results"
        / "current"
        / "legacy"
        / "ybyag_crystal_pcsel"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    print("Output:", output, flush=True)
    print("[core] geometry -> guided TE -> reflected Green -> C1D+Crad+C2D", flush=True)
    optical, cell, guided, full = derive(truncation=3 if args.quick else CWT_TRUNCATION)
    result = optical.result
    lattice = lattice_spec(result)
    spec = FourWaveOpticalSpec(
        result.bragg_wavelength_nm,
        result.effective_index,
        result.group_index,
        DEVICE_SIZE_UM,
        INTERNAL_POWER_LOSS_CM,
        tuple(result.eigenvalues_m.real / 100),
        tuple(result.eigenvalues_m.imag / 100),
    )
    zwidth, qenergy, qgain = active_weights(result, guided)
    k0 = 2 * np.pi / (result.bragg_wavelength_nm * 1e-9)
    kappa = k0 * np.sqrt(result.effective_index**2 - SILICA_INDEX**2)
    np.savez_compressed(
        output / "optical_core.npz",
        C=result.coupling_m,
        C1D=result.c1d_m,
        Crad=result.crad_m,
        C2D=result.c2d_m,
        Lup=optical.upward_amplitude_loss_m,
        Ldown=optical.downward_amplitude_loss_m,
        z_um=result.vertical_mode.z_um,
        field=result.vertical_mode.field,
        q_energy=qenergy,
        q_gain=qgain,
    )
    summary = dict(
        status="exploratory_TE_CWT_candidate_NOT_experimental_validation",
        wavelength_nm=result.bragg_wavelength_nm,
        period_nm=result.lattice_constant_nm,
        effective_index=result.effective_index,
        group_index=result.group_index,
        gain_energy_overlap=qenergy.sum(),
        gain_flux_overlap=qgain.sum(),
        pc_overlap=result.pc_confinement,
        optical_theorem_residual=optical.optical_theorem_residual,
        guide_DBR_roundtrip_attenuation=float(
            np.exp(-2 * kappa * BOTTOM_SPACER_NM * 1e-9)
        ),
        numerical_quality="quick_coarse" if args.quick else "multi_grid_not_full_wave",
        full_stack=[asdict(x) for x in full.layers],
        steps=selected,
    )
    summary["assumptions"] = {
        "gain_model": "single selected TE-family CW branch with x-z inversion; no shared multimode competition",
        "spectroscopy": "nominal room-temperature cross sections, not calibrated concentration/temperature spectra",
        "film_optics": "real fixed indices; transparent radiation stack plus assumed separate internal loss",
        "thermal": "energy bookkeeping only; no temperature, stress or thermal lens solution",
        "finite_loss": "finest-grid raw upwind loss includes numerical diffusion; not all other loss is material heat",
    }
    summary["parameters"] = {
        key: value
        for key, value in globals().items()
        if key.isupper() and isinstance(value, (str, int, float, tuple))
    }
    summary["source_sha256"] = {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in [
            Path(__file__).resolve(),
            *sorted((PROJECT_ROOT / "src" / "pcselsim").glob("*.py")),
        ]
    }
    if selected["01_structure"]:
        print("[01] full architecture, lattice, guided-family audit", flush=True)
        layers = tuple(
            LayerSpec(x.name, x.thickness_nm, float(np.real(x.refractive_index)))
            for x in full.layers
        )
        plot_layer_stack(layers, output / "01_full_stack.png")
        plot_lattice(lattice, output / "01_lattice.png")
        plot_k_space(lattice, output / "01_k_space.png")
        plot_vertical_mode(
            result.vertical_mode,
            "passive YAG photonic crystal",
            output / "01_vertical.png",
        )
        modes = solve_guided_te_modes(
            guided, result.bragg_wavelength_nm, VERTICAL_STEP_NM, 256
        )
        summary["bound_TE_family_count"] = len(modes)
        summary["vertical_family_count_cap"] = 256
        summary["vertical_families"] = [
            dict(family=i, neff=m.effective_index, confinement=m.confinement)
            for i, m in enumerate(modes)
        ]
        wavelengths = np.linspace(850, 1150, 301)
        # Reflectivity seen from silica just above the DBR, not from air above whole device.
        dbrstack = StratifiedStack(
            full.layers[: 2 * DBR_PAIRS], PASSIVE_YAG_INDEX, SILICA_INDEX
        )
        rt = np.array(
            [
                (
                    solve_stack(dbrstack, float(w), incident_side="top").R,
                    solve_stack(dbrstack, float(w), incident_side="top").T,
                )
                for w in wavelengths
            ]
        )
        np.savetxt(
            output / "01_DBR_spectrum.csv",
            np.c_[wavelengths, rt],
            delimiter=",",
            header="wavelength_nm,R,T",
            comments="",
        )
        fig, ax = plt.subplots()
        ax.plot(wavelengths, rt[:, 0], label="R")
        ax.plot(wavelengths, rt[:, 1], label="T")
        for w in (940, 1030):
            ax.axvline(w, ls=":", color="gray")
        ax.set(
            xlabel="Wavelength (nm)",
            ylabel="Power fraction",
            title="Finite quarter-wave DBR: do not assume 940-nm transparency",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "01_DBR_spectrum.png", dpi=180)
        plt.close(fig)
    if selected["02_bands"]:
        print("[02] near-Gamma M-Gamma-X bands", flush=True)
        bands = square_four_wave_band_diagram(
            result.coupling_m,
            result.lattice_constant_nm,
            result.bragg_wavelength_nm,
            result.effective_index,
            group_index=result.group_index,
        )
        plot_band_diagram(
            bands, output / "02_bands.png", "Crystal PCSEL: local M-Gamma-X dispersion"
        )
    needs_finite = any(
        selected[k] for k in ("03_finite_modes", "04_fields", "05_spatial_gain")
    )
    if needs_finite:
        grids = (9, 13, 17) if args.quick else FINITE_GRIDS
        print("[03 core] finite aperture eigenmodes, grids", grids, flush=True)
        modes, convergence = solve_finite_modes_converged(
            spec, grids, result.coupling_m, result.eigenvectors, result.radiation_fields
        )
        spec = replace(spec, grid_points=grids[-1])
        arrays = {f"convergence_{key}": value for key, value in convergence.items()}
        for name, mode in modes.items():
            arrays[f"{name}_fields"] = mode.fields
            arrays[f"{name}_radiation_x"] = mode.radiation_x
            arrays[f"{name}_radiation_y"] = mode.radiation_y
            arrays[f"{name}_loss_raw_extrap_sensitivity_per_m"] = np.array(
                [
                    mode.grid_alpha_per_m,
                    mode.alpha_per_m,
                    mode.extrapolation_uncertainty_per_m,
                ]
            )
        np.savez_compressed(output / "finite_modes.npz", **arrays)
        if selected["03_finite_modes"]:
            write_mode_table(output, spec, modes)
            plot_grid_convergence(spec, convergence, output / "03_convergence.png")
        # Conservative numerical gate: use finest-grid raw POWER loss, not an
        # optimistic zero-grid intercept. Same finest-grid field yields side flux.
        ranked = sorted(modes.values(), key=lambda m: m.grid_alpha_per_m)
        chosen = ranked[0]
        summary["selected_mode"] = chosen.name
        summary["selection_rule"] = (
            "lowest finest-grid raw loss among retained four-wave families; no global multimode guarantee"
        )
        summary["mode_losses"] = {
            m.name: dict(
                raw_amplitude_cm=m.grid_alpha_per_m / 100,
                extrap_amplitude_cm=m.alpha_per_m / 100,
                fit_status=m.extrapolation_status,
                sensitivity_cm=m.extrapolation_uncertainty_per_m / 100,
            )
            for m in ranked
        }
        if selected["04_fields"]:
            print("[04] envelope, one-cell Bloch fields, top far field", flush=True)
            plot_mode_atlas(
                lattice,
                spec,
                modes,
                output / "04_mode_atlas.png",
                result.unit_cell_fields,
            )
            for mode in ranked:
                plot_vector_far_field_diagnostics(
                    spec, mode, output / f"04_far_field_{mode.name}.png"
                )
            summary["beam_metrics"] = asdict(vector_far_field_metrics(spec, chosen))
            plot_beam_zoom(spec, chosen, output / "04_beam_zoom.png")
        if selected["05_spatial_gain"]:
            print(
                "[05] depleted two-edge pump, spatial inversion, CW gain/loss and power closure",
                flush=True,
            )
            up, down = optical.side_power_losses(chosen.fields)
            total = 2 * chosen.grid_alpha_per_m + 100 * INTERNAL_POWER_LOSS_CM
            medium = YbYAGMediumConfig(
                laser_wavelength_nm=result.bragg_wavelength_nm,
                refractive_index=YBYAG_INDEX,
                dopant_density_cm3=YB_DENSITY_CM3,
                confinement_factor=min(float(qenergy.sum()), 1),
                upper_state_lifetime_ms=LIFETIME_MS,
                pump_absorption_cross_section_cm2=PUMP_ABSORPTION_CM2,
                pump_emission_cross_section_cm2=PUMP_EMISSION_CM2,
                laser_absorption_cross_section_cm2=LASER_ABSORPTION_CM2,
                laser_emission_cross_section_cm2=LASER_EMISSION_CM2,
                spontaneous_emission_factor=0,
                pumped_area_um2=DEVICE_SIZE_UM**2,
                gain_thickness_um=GAIN_THICKNESS_NM / 1000,
            )
            lateral = lateral_energy_weights(chosen.fields, PUMP_CELLS)
            model = SpatialYbYAGModel(
                medium,
                zwidth,
                qenergy,
                total,
                up,
                down,
                c / result.group_index,
                qgain,
                lateral,
            )
            pump = SpatialPumpConfig(
                "edge",
                PUMP_INTENSITY_PER_EDGE_W_CM2,
                PUMP_INTENSITY_PER_EDGE_W_CM2,
                DEVICE_SIZE_UM,
                PUMP_CELLS,
            )
            rates = solve_spatial_ybyag_steady(model, pump)
            summary["CW_spatial_gain"] = dict(
                lasing_condition=rates.lasing_gain_condition,
                interpretation="single selected TE-family branch; other TE families compete, not certified stable single-mode output",
                raw_total_power_loss_cm=total / 100,
                upward_power_loss_cm=up / 100,
                downward_power_loss_cm=down / 100,
                top_fraction_of_vertical_output=up / (up + down),
                threshold_pump_scale=rates.threshold_pump_scale,
                modal_gain_cm=rates.modal_gain_per_m / 100,
                unlased_gain_cm=rates.unlased_modal_gain_per_m / 100,
                photon_density_m3=rates.photon_density_m3,
                energy_budget_W=rates.energy_budget_W,
                population_residual_per_s=rates.population_residual_per_s,
                photon_residual_per_s=rates.photon_fraction_residual_per_s,
            )
            summary["medium"] = asdict(medium)
            summary["pump"] = asdict(pump)
            np.savez_compressed(
                output / "05_spatial_gain.npz",
                excited_fraction=rates.excited_fraction,
                pump_intensity_W_cm2=rates.pump_intensity_W_cm2,
                lateral_energy_fractions=lateral,
            )
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, data, title in zip(
                axes,
                (rates.excited_fraction.T, rates.pump_intensity_W_cm2.T),
                ("Yb excited fraction", "Internal pump intensity (W/cm2)"),
            ):
                image = ax.imshow(
                    data,
                    origin="lower",
                    aspect="auto",
                    extent=[0, DEVICE_SIZE_UM, 0, GAIN_THICKNESS_NM / 1000],
                )
                fig.colorbar(image, ax=ax)
                ax.set(xlabel="x (um)", ylabel="Active depth (um)", title=title)
            fig.tight_layout()
            fig.savefig(output / "05_inversion_pump.png", dpi=180)
            plt.close(fig)
            scan = []
            for scale in (0, 0.25, 0.5, 0.75, 1, 1.5, 2):
                point = solve_spatial_ybyag_steady(
                    model,
                    replace(
                        pump,
                        forward_intensity_W_cm2=pump.forward_intensity_W_cm2 * scale,
                        backward_intensity_W_cm2=pump.backward_intensity_W_cm2 * scale,
                    ),
                    find_threshold=False,
                )
                scan.append(
                    dict(
                        scale=scale,
                        lasing_condition=point.lasing_gain_condition,
                        **point.energy_budget_W,
                    )
                )
            save_json(output / "05_pump_scan.json", scan)
    if selected["06_design_scan"]:
        print("[06] thickness/cap/family audit and DBR phase scan", flush=True)
        design_scan(output)
    save_json(output / "summary.json", summary)
    save_json(
        output / "run_status.json", {"status": "complete", "selected_steps": selected}
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k
                in (
                    "selected_mode",
                    "gain_energy_overlap",
                    "pc_overlap",
                    "CW_spatial_gain",
                )
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(
        "Finished. This CW feasibility gate does NOT establish fabrication, thermal stability or mode locking.",
        flush=True,
    )
    return output


def main(argv=None):
    """Fixed-period frequency closure followed by shared-population CW competition.

    The old frozen reference is retained as a named comparison function only;
    normal execution always closes C(omega) before calculating gain or output.

    中文：通过 --only 可只输出静态冷腔结构/能带/模场；请求增益步骤后，
    才求共享反转静态 CW。任何步骤都不据此证明时域稳定或锁模。
    """
    from pcselsim.frequency_cwt import solve_frequency_consistent_modes
    from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG, solve_multimode_spatial_ybyag
    from pcselsim.ybyag_spectroscopy import YbYAGSpectroscopy
    from pcselsim.run_output import default_output, prepare_output, write_provenance
    from pcselsim.custom_analysis import solve_finite_modes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true", help="D2/N9/TE0 pipeline smoke; no grid-convergence claim")
    parser.add_argument("--dry-run", action="store_true", help="print parameters and dependencies without creating a result directory")
    parser.add_argument("--only", nargs="+", choices=STEPS)
    parser.add_argument("--te-families", nargs="+", type=int)
    parser.add_argument("--modes", nargs="+", choices=("A", "B", "C", "D"), default=("A", "B", "C", "D"))
    parser.add_argument("--gain-um", type=float, default=GAIN_THICKNESS_NM/1000)
    parser.add_argument("--period-nm", type=float, help="fixed physical period; omitted designs TE0 Bragg once, then holds it fixed")
    parser.add_argument("--grid-points", type=int)
    parser.add_argument("--spectroscopy", type=Path, help="bounded measured cross-section table JSON; no extrapolation")
    parser.add_argument("--temperature-K", type=float, default=293.15)
    parser.add_argument("--pump-power-W", type=float, default=INTERNAL_TOTAL_PUMP_POWER_W, help="total first-entry power already coupled inside gain; all beams combined")
    parser.add_argument("--pump-architecture", choices=("edge_single", "edge_dual", "top_single"), default="edge_dual")
    parser.add_argument("--top-passes", type=int, default=1, help="finite noncoherent ideal relay for top pumping")
    parser.add_argument("--return-efficiency", type=float, default=.98, help="assumed survival between top passes; not measured DBR feedback")
    args = parser.parse_args(argv)
    if args.gain_um <= 0 or args.pump_power_W < 0 or args.temperature_K <= 0:
        parser.error("Positive thickness/temperature and nonnegative internal pump power required")
    families = tuple(args.te_families or ((0,) if args.quick else ACTIVE_TE_FAMILIES))
    if min(families) < 0 or len(set(families)) != len(families):
        parser.error("TE families must be unique nonnegative integers")
    grid = args.grid_points or (9 if args.quick else FINITE_GRIDS[-1])
    if grid < 9:
        parser.error("grid-points must be at least 9")
    selected = {k: v and (args.only is None or k in args.only) for k, v in STEPS.items()}
    if args.dry_run:
        plan = dict(selected_steps=selected, gain_um=args.gain_um, fixed_period_nm=args.period_nm,
                    te_families=families, finite_grid_points=grid,
                    requires_finite_frequency_solve=any(selected[k] for k in ("03_finite_modes","04_fields","05_spatial_gain")),
                    requires_shared_CW=selected["05_spatial_gain"], internal_pump_power_W=args.pump_power_W,
                    spectroscopy=str(args.spectroscopy) if args.spectroscopy else "uncalibrated_nominal")
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return plan
    output = prepare_output(args.output or default_output(PROJECT_ROOT, "ybyag_crystal_pcsel"), PROJECT_ROOT)
    write_provenance(output, PROJECT_ROOT)
    save_json(output/"run_status.json", dict(status="running"))
    try:
        cell, reference_guide, full, design_period = make_structure(gain_nm=args.gain_um*1000)
        period = args.period_nm or design_period
        run_plan = dict(
            arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
            selected_steps=selected, fixed_physical_period_nm=period,
            period_selection="specified" if args.period_nm is not None else "TE0 reference design once, fixed during frequency search",
            gain_thickness_um=args.gain_um, device_side_um=DEVICE_SIZE_UM,
            active_TE_families=families, retained_branches=tuple(args.modes),
            frequency_root_grid_points=grid,
            mesh_diagnostic_grids=(grid,) if args.quick else tuple(sorted(set(n for n in (9,13,*FINITE_GRIDS,grid) if n<=grid))),
            cwt_order=2 if args.quick else CWT_TRUNCATION, vertical_step_nm=VERTICAL_STEP_NM,
            gain_z_cells=max(GAIN_Z_CELLS,int(np.ceil(args.gain_um/.2))),pump_x_cells=9 if args.quick else PUMP_CELLS,
            inbound_pump=dict(architecture=args.pump_architecture,total_internal_first_entry_W=args.pump_power_W,
                              top_passes=args.top_passes,assumed_return_efficiency=args.return_efficiency,
                              external_coupling="not modelled"),
            parameter_panel={key:value for key,value in globals().items()
                             if key.isupper() and isinstance(value,(int,float,str,tuple))},
            source_provenance_file="source_provenance.json",numpy_version=np.__version__)
        run_plan["blas_thread_environment"] = {key:os.environ.get(key) for key in
                                                ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}
        if args.spectroscopy:
            raw = args.spectroscopy.read_bytes()
            (output/"spectroscopy_input.json").write_bytes(raw)
            run_plan["spectroscopy_input_sha256"] = hashlib.sha256(raw).hexdigest()
        save_json(output/"run_plan.json",run_plan)
        table = YbYAGSpectroscopy.from_json(args.spectroscopy) if args.spectroscopy else None
        spectrum_metadata = (table.metadata() if table else dict(
            calibration_status="uncalibrated_nominal_constants_no_spectral_validation",
            provenance="explicit parameter-panel assumptions; no measured wavelength/temperature spectrum supplied",
            temperature_K=args.temperature_K, extrapolation="no measured-spectrum claim"))
        settings = ThreeDCWTSettings(2 if args.quick else CWT_TRUNCATION, VERTICAL_STEP_NM)
        def reference_assets():
            from types import SimpleNamespace
            if selected["01_structure"]:
                plot_layer_stack(tuple(LayerSpec(x.name,x.thickness_nm,float(np.real(x.refractive_index))) for x in full.layers),output/"01_full_stack.png")
                plot_lattice(lattice_spec(SimpleNamespace(lattice_constant_nm=period)),output/"01_lattice.png")
                plot_vertical_mode(reference_guide.solve_te0(TARGET_WAVELENGTH_NM,VERTICAL_STEP_NM),
                                   "passive YAG photonic crystal",output/"01_reference_vertical.png")
        reference_assets()
        needs_finite = any(selected[k] for k in ("03_finite_modes","04_fields","05_spatial_gain"))
        if not needs_finite:
            count = len(solve_guided_te_modes(reference_guide,TARGET_WAVELENGTH_NM,VERTICAL_STEP_NM,256))
            if selected["02_bands"]:
                reference = build_reflector_coupling(cell,reference_guide,full,"passive YAG photonic crystal",
                                                    period,TARGET_WAVELENGTH_NM,settings,
                                                    fixed_wavelength_nm=TARGET_WAVELENGTH_NM).result
                band = square_four_wave_band_diagram(reference.coupling_m,period,TARGET_WAVELENGTH_NM,
                                                      reference.effective_index,group_index=reference.group_index)
                plot_band_diagram(band,output/"02_reference_local_bands.png",
                                  "Reference-frequency local bands; no finite laser frequency or CW solve")
            summary = dict(status="reference_structure_or_local_band_only_no_finite_mode_or_CW_claim",
                           fixed_period_nm=period,reference_wavelength_nm=TARGET_WAVELENGTH_NM,
                           isolated_TE_family_count=count,selected_steps=selected,spectroscopy=spectrum_metadata)
            save_json(output/"summary.json",summary)
            if selected["06_design_scan"]:
                save_json(output/"06_family_scope.json",dict(isolated_TE_family_count=count,
                          scope="vertical family count only; no frequency-closed finite spectrum, threshold or power"))
            save_json(output/"run_status.json",dict(status="complete",scope=summary["status"]))
            print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
            return output
        rows, spatial_models, names, branches = [], [], [], {}
        zcells = max(GAIN_Z_CELLS, int(np.ceil(args.gain_um/.2)))
        nx = 9 if args.quick else PUMP_CELLS
        for family in families:
            family_started = perf_counter()
            _, guided, radiation, _ = make_structure(gain_nm=args.gain_um*1000, te_family=family)
            print(f"[frequency] TE{family}, fixed a={period:.9f} nm, D={settings.truncation_order}, N={grid}", flush=True)
            from pcselsim.three_d_cwt import solve_bragg_vertical_mode
            # Find a local initial spectral reference for this TE family at the
            # SAME period. This is only an initializer: every finite branch is
            # subsequently solved at independent trial frequencies with no
            # Bragg locking and a nonzero finite coupling detuning.
            family_reference_nm, _ = solve_bragg_vertical_mode(guided,period,TARGET_WAVELENGTH_NM,settings)
            solved = solve_frequency_consistent_modes(
                cell, guided, radiation, "passive YAG photonic crystal", period,
                family_reference_nm, settings, DEVICE_SIZE_UM, INTERNAL_POWER_LOSS_CM,
                grid, wavelength_interval_nm=(TARGET_WAVELENGTH_NM-20, TARGET_WAVELENGTH_NM+20),
                mode_names=tuple(args.modes))
            family_elapsed = perf_counter()-family_started
            for label, branch in solved.items():
                name = f"TE{family}_{label}"
                branches[name] = branch
                optical, mode = branch.optical, branch.finite_mode
                result = optical.result
                zwidth, qenergy, qgain = active_weights(result, guided, zcells)
                up, down = optical.side_power_losses(mode.fields)
                total = 2*mode.alpha_per_m + 100*INTERNAL_POWER_LOSS_CM
                medium = YbYAGMediumConfig(
                    laser_wavelength_nm=branch.wavelength_nm, refractive_index=YBYAG_INDEX,
                    dopant_density_cm3=YB_DENSITY_CM3, upper_state_lifetime_ms=LIFETIME_MS,
                    pump_absorption_cross_section_cm2=PUMP_ABSORPTION_CM2,
                    pump_emission_cross_section_cm2=PUMP_EMISSION_CM2,
                    laser_absorption_cross_section_cm2=LASER_ABSORPTION_CM2,
                    laser_emission_cross_section_cm2=LASER_EMISSION_CM2,
                    spontaneous_emission_factor=0, pumped_area_um2=DEVICE_SIZE_UM**2,
                    gain_thickness_um=args.gain_um)
                if table:
                    medium = table.at(medium, branch.wavelength_nm, args.temperature_K)
                spatial_models.append(SpatialYbYAGModel(
                    medium, zwidth, qenergy, total, up, down, c/result.group_index,
                    qgain, lateral_energy_weights(mode.fields, nx)))
                names.append(name)
                # Grid sensitivity is evaluated at this physical frequency;
                # the reported root retains the actual finest eigenpair.
                grid_audit = []
                grids = (grid,) if args.quick else tuple(sorted(set(n for n in (9, 13, *FINITE_GRIDS, grid) if n <= grid)))
                for n in grids:
                    if n == grid:
                        matched = mode
                    else:
                        trial = solve_finite_modes(replace(branch.spec, grid_points=n), result.coupling_m,
                                                   result.eigenvectors, result.radiation_fields)
                        # Different grids cannot directly compare vector arrays;
                        # use the retained band label for this mesh diagnostic.
                        matched = trial[label]
                    grid_audit.append(dict(grid_points=n, amplitude_loss_per_m=matched.alpha_per_m,
                                           detuning_per_m=matched.delta_per_m,
                                           frequency_residual_at_fixed_finest_root_per_m=branch.background_detuning_per_m-matched.delta_per_m,
                                           eigen_residual=getattr(matched, "eigen_residual", None)))
                row = dict(name=name, te_family=family, branch=label, wavelength_nm=branch.wavelength_nm,
                           family_Bragg_initializer_nm=family_reference_nm,
                           period_nm=period, amplitude_loss_per_m=mode.alpha_per_m,
                           total_power_loss_per_m=total, upward_power_loss_per_m=up, downward_power_loss_per_m=down,
                           frequency_residual_per_m=branch.frequency_residual_per_m,
                           reference_field_overlap=branch.reference_overlap,
                           frequency_elapsed_seconds=branch.elapsed_seconds,
                           family_frequency_elapsed_seconds=family_elapsed,
                           reference_dense_limit=1400, sparse_tracking_maxiter=400,
                           eigen_residual=getattr(mode, "eigen_residual", None),
                           discretization=getattr(mode, "discretization", "unknown"),
                           gain_energy_overlap=float(qenergy.sum()), gain_weight=float(qgain.sum()),
                           optical_theorem_residual=optical.optical_theorem_residual,
                           grid_sensitivity=grid_audit, frequency_evaluations=branch.evaluations)
                rows.append(row)
                save_json(output/"optical_modes.json",rows)
                print(f"  {name}: {branch.wavelength_nm:.9f} nm; root residual {branch.frequency_residual_per_m:.3g}/m; "
                      f"field overlap {branch.reference_overlap:.5f}; {branch.elapsed_seconds:.2f}s",flush=True)
                np.savez_compressed(output/f"{name}_optical.npz", C=result.coupling_m, C1D=result.c1d_m,
                                    Crad=result.crad_m, C2D=result.c2d_m,
                                    Lup=optical.upward_amplitude_loss_m, Ldown=optical.downward_amplitude_loss_m,
                                    field=mode.fields, z_um=result.vertical_mode.z_um,
                                    theta=result.vertical_mode.field, q_energy=qenergy, q_gain=qgain)
        save_json(output/"optical_modes.json", rows)
        if not selected["05_spatial_gain"]:
            # A cold-cavity/field request must not silently run a gain simulation.
            summary = dict(status="frequency_closed_cold_cavity_only_no_CW_or_lasing_claim",
                           fixed_period_nm=period,included_TE_families=families,
                           supplied_modes=names,selected_steps=selected,spectroscopy=spectrum_metadata)
            save_json(output/"summary.json",summary)
            if selected["04_fields"]:
                chosen = min(branches.values(),key=lambda b:b.finite_mode.alpha_per_m)
                plot_vector_far_field_diagnostics(chosen.spec,chosen.finite_mode,output/"04_cold_cavity_far_field.png")
                plot_beam_zoom(chosen.spec,chosen.finite_mode,output/"04_beam_zoom.png")
            if selected["02_bands"]:
                chosen = next(iter(branches.values())); r=chosen.optical.result
                band = square_four_wave_band_diagram(r.coupling_m,period,chosen.wavelength_nm,
                                                      r.effective_index,group_index=r.group_index)
                plot_band_diagram(band,output/"02_local_bands.png","Local dispersion at one closed cold-cavity frequency")
            if selected["06_design_scan"]:
                save_json(output/"06_family_scope.json",dict(included_TE_families=families,
                          scope="retained frequency-closed cold-cavity branches; no shared gain competition requested"))
            save_json(output/"run_status.json",dict(status="complete",scope=summary["status"]))
            return output
        system = SpatialMultimodeYbYAG(tuple(spatial_models), tuple(names))
        from pcselsim.ybyag_pump_geometry import PumpGeometryConfig, pump_geometry_metadata
        pump = PumpGeometryConfig(args.pump_architecture, args.pump_power_W, DEVICE_SIZE_UM, nx,
                                  top_passes=args.top_passes if args.pump_architecture=="top_single" else 1,
                                  return_efficiency=args.return_efficiency if args.pump_architecture=="top_single" else 1.)
        print(f"[CW] shared inversion, {len(names)} supplied branches, {args.pump_power_W:g} W internal input", flush=True)
        rates = solve_multimode_spatial_ybyag(system, pump)
        count = len(solve_guided_te_modes(reference_guide, TARGET_WAVELENGTH_NM, VERTICAL_STEP_NM, 256))
        summary = dict(status="frequency_closed_scalar_CWT_shared_inversion_CW_conditional_NOT_full_wave_or_thermal_validation",
                       fixed_period_nm=period, gain_thickness_um=args.gain_um,
                       included_TE_families=families, isolated_TE_family_count_at_reference=count,
                       supplied_mode_count=len(names), active_modes=rates.active_modes,
                       interpretation="static complementarity among supplied branches; no dynamical single-mode certification",
                       frequency_model="fixed geometry; re-solved Theta/C1D/Crad/C2D/finite fields at each real frequency; squared-beta CWT background",
                       spectroscopy=spectrum_metadata, temperature_K=args.temperature_K,
                       internal_incident_pump_W=args.pump_power_W,
                       pump_geometry=pump_geometry_metadata(spatial_models[0],pump),
                       external_coupling="not calculated; pump power is after coupler",
                       thermal="effective energy budget only; no temperature/stress/thermal-lens feedback",
                       film_optics="constant real nominal indices; absorbing-film CWT rejected; actual n(lambda),k(lambda) still required",
                       competition_scope="x-z shared inversion; y averaged; no coherent population gratings, field pulling, ASE or dynamic stability",
                       population_residual_per_s=rates.population_residual_per_s,
                       complementarity_residual=rates.complementarity_residual,
                       maximum_inactive_gain_excess_per_m=rates.maximum_inactive_gain_excess_per_m,
                       energy_budget_W=rates.energy_budget_W, numerical_quality="smoke_only" if args.quick else "finest_staggered_frequency_root; mesh_sensitivity_at_same_frequency_not_separate_mesh_roots",
                       modes=[dict(name=name, wavelength_nm=branches[name].wavelength_nm,
                                   photon_density_m3=float(rates.photon_density_m3[i]),
                                   saturated_modal_gain_per_m=float(rates.modal_gain_per_m[i]),
                                   unlased_modal_gain_per_m=float(rates.unlased_modal_gain_per_m[i]),
                                   upward_output_W=float(rates.upward_output_W[i]),
                                   downward_output_W=float(rates.downward_output_W[i])) for i, name in enumerate(names)])
        save_json(output/"summary.json", summary)
        np.savez_compressed(output/"shared_inversion.npz", excited_fraction=rates.excited_fraction,
                            photon_density_m3=rates.photon_density_m3,
                            modal_gain_per_m=rates.modal_gain_per_m)
        if selected["01_structure"]:
            plot_layer_stack(tuple(LayerSpec(x.name,x.thickness_nm,float(np.real(x.refractive_index))) for x in full.layers), output/"01_full_stack.png")
            plot_lattice(lattice_spec(next(iter(branches.values())).optical.result),output/"01_lattice.png")
            plot_vertical_mode(next(iter(branches.values())).optical.result.vertical_mode,
                               "passive YAG photonic crystal",output/"01_vertical.png")
        if selected["04_fields"]:
            chosen = names[int(np.argmax(rates.upward_output_W))]
            branch = branches[chosen]
            plot_vector_far_field_diagnostics(branch.spec, branch.finite_mode, output/"04_conditional_far_field.png")
            plot_beam_zoom(branch.spec, branch.finite_mode, output/"04_beam_zoom.png")
        if selected["02_bands"]:
            branch = next(iter(branches.values()))
            r = branch.optical.result
            bands = square_four_wave_band_diagram(r.coupling_m, period, branch.wavelength_nm,
                                                   r.effective_index, group_index=r.group_index)
            plot_band_diagram(bands,output/"02_local_bands.png",
                              "Local dispersion at one closed frequency; not a global C(omega,k) band solve")
        if selected["05_spatial_gain"]:
            fig, ax = plt.subplots(figsize=(7,4))
            image = ax.imshow(rates.excited_fraction.T, origin="lower", aspect="auto",
                              extent=[0,DEVICE_SIZE_UM,0,args.gain_um])
            fig.colorbar(image,ax=ax,label="Shared excited fraction")
            ax.set(xlabel="x (um)",ylabel="z in gain (um)",title="Shared inversion of supplied frequency-closed modes")
            fig.tight_layout(); fig.savefig(output/"05_shared_inversion.png",dpi=180); plt.close(fig)
        if selected["06_design_scan"]:
            save_json(output/"06_family_scope.json",dict(
                isolated_TE_family_count=count, included_TE_families=families,
                included_finite_branches=names,
                interpretation="Other TE/TM and higher lateral branches remain untested; thickness studies require separate fixed-geometry frequency solves",
                parameter_scan="Use --gain-um and --period-nm in separate new output directories; no frozen-frequency design ranking is emitted"))
        save_json(output/"run_status.json", dict(status="complete", selected_steps=selected,
                                                 scope="conditional frequency-closed scalar CW; uncalibrated inputs remain explicit"))
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
        return output
    except Exception as error:
        save_json(output/"run_status.json",dict(status="failed",error_type=type(error).__name__,error=str(error)))
        raise


if __name__ == "__main__":
    main()
