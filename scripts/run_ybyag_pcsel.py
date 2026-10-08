"""Crystal-gain PCSEL: independent layered radiation + spatial Yb gain.

PyCharm: edit PARAMETER PANEL / STEPS, then right click this file -> Run.
中文：上下透明层只导光；底部另置有限DBR。此候选是需实验/全波验证的设计假设。
No semiconductor model is overwritten. No mode-locking is implied by CW gain.
"""

from __future__ import annotations
import argparse
import csv
import json
import hashlib
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

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
TE_FAMILY = 0
VERTICAL_STEP_NM = 5.0
CWT_TRUNCATION = 6
FINITE_GRIDS = (17, 25, 33)
PUMP_INTENSITY_PER_EDGE_W_CM2 = (
    2.0e5  # INSIDE YAG after pump coupler, each of two edges
)
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
    "03_finite_modes": True,  # same C, multiple grids, raw/extrap losses separately
    "04_fields": True,  # whole-device envelope, cell field, top vector farfield
    "05_spatial_gain": True,  # saturated two-edge pump + inversion + CW steady closure
    "06_design_scan": True,  # thickness/cap/spacer diagnostics, not global optimization
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
    Y-resolved inversion and simultaneous multimode competition remain absent.
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


def main(argv=None):
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


if __name__ == "__main__":
    main()
