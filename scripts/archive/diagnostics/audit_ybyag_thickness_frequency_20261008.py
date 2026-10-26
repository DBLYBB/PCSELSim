"""Saved-field frequency sensitivity, NOT a nonlinear C(omega) eigenproblem.

Read a completed thickness study with --source. Re-evaluate only the q=0
layered radiation Green at Bragg and at its raw finite-mode linearized
wavelength, keeping Theta, a, geometry, dielectric indices and finite fields
fixed. No existing data, model or CW result is modified.

中文：检查厚度扫描中的反射相位敏感性；不可将新损耗拼回旧CW结果当作已修正预测。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "src" / "pcselsim").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from pcselsim.geometry import Ellipse, SquareLatticeCell
from pcselsim.stratified_optics import StratifiedLayer, StratifiedStack, local_green_kernel
from pcselsim.run_output import default_output, prepare_output, write_provenance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trapezoid_weights(axis: np.ndarray) -> np.ndarray:
    if len(axis) < 2 or np.any(np.diff(axis) <= 0):
        raise ValueError("PC source quadrature requires at least two increasing coordinates")
    weights = np.empty_like(axis)
    weights[0] = .5 * (axis[1] - axis[0])
    weights[-1] = .5 * (axis[-1] - axis[-2])
    weights[1:-1] = .5 * (axis[2:] - axis[:-2])
    return weights


def audit_case(folder: Path, parameters: dict, guided_padding_um: float) -> dict:
    """Use the stored layer stack and cell parameters, never current main defaults."""
    report_path, arrays_path = folder / "optical_report.json", folder / "optical_arrays.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as core:
        passive_n = parameters["PASSIVE_YAG_INDEX"]
        gain_n = parameters["YBYAG_INDEX"]
        fill = parameters["FILL_FRACTION"]
        share = parameters["SMALL_HOLE_AREA_SHARE"]
        displacement = parameters["HOLE_DISPLACEMENT_A"]
        radii = np.sqrt(fill * np.array([share, 1 - share]) / np.pi)
        cell = SquareLatticeCell(passive_n**2, tuple(
            Ellipse((d, d), (float(r), float(r)), 0, parameters["SILICA_INDEX"]**2)
            for d, r in zip((-displacement / 2, displacement / 2), radii)
        ))
        xi = cell.fourier_epsilon
        sources = np.array([[xi(-1, 0), xi(1, 0), 0, 0],
                            [0, 0, xi(0, -1), xi(0, 1)]], complex)
        gram = sources.conj().T @ sources
        # In the source study the full-stack exterior is passive YAG/air.
        full = StratifiedStack(tuple(StratifiedLayer(**layer) for layer in report["stack"]),
                               bottom_index=passive_n, top_index=1.0)
        host = next(i for i, layer in enumerate(full.layers)
                    if layer.name == "passive YAG photonic crystal")
        if not np.isclose(complex(full.layers[host].refractive_index)**2,
                          cell.fourier_epsilon(0, 0), rtol=1e-10):
            raise ValueError("Stored PC host disagrees with the saved geometry parameters")
        guided = [layer for layer in full.layers if not layer.name.startswith("DBR_")]
        guide_pc = next(i for i, layer in enumerate(guided)
                        if layer.name == "passive YAG photonic crystal")
        total_nm = 2000 * guided_padding_um + sum(layer.thickness_nm for layer in guided)
        z_nm = core["vertical_z_um"] * 1000 + .5 * total_nm
        start_nm = 1000 * guided_padding_um + sum(layer.thickness_nm for layer in guided[:guide_pc])
        mask = (z_nm >= start_nm) & (z_nm <= start_nm + full.layers[host].thickness_nm)
        local_z = (z_nm[mask] - start_nm) * 1e-9 + full.boundaries_m[host]
        weighted = core["vertical_field"][mask] * _trapezoid_weights(local_z)
        period_nm = report["period_redesigned_nm"]
        beta = 2 * np.pi / (period_nm * 1e-9)
        wavelength_b = report["bragg_wavelength_nm"]
        delta = report["selected_raw_detuning_per_m"]
        # Existing study q_gain=(ng/nYAG)*q_energy; recover the saved ng.
        group_index = float(gain_n * core["q_gain"].sum() / core["q_energy"].sum())
        wavelength = 2 * np.pi / (2 * np.pi / (wavelength_b * 1e-9) + delta / group_index) * 1e9
        if not np.isclose(wavelength, report["selected_raw_wavelength_shifted_nm"], atol=1e-9):
            raise ValueError("Raw-mode wavelength reconstruction disagrees with source report")

        def evaluate(wavelength_nm):
            kernel = local_green_kernel(full, host, wavelength_nm)
            if not kernel.lossless:
                raise ValueError("This escaped-flux optical theorem audit requires lossless layers")
            integral = weighted.conj() @ kernel.matrix(local_z) @ weighted
            escape = kernel.outgoing_amplitudes(local_z)
            top, bottom = escape.top @ weighted, escape.bottom @ weighted
            factor = (2 * np.pi / (wavelength_nm * 1e-9))**4 / (2 * beta)
            crad = -factor * integral * gram
            up = factor * escape.top_admittance_m.real * abs(top)**2 * gram
            down = factor * escape.bottom_admittance_m.real * abs(bottom)**2 * gram
            antihermitian = (crad - crad.conj().T) / (2j)
            closure = float(np.linalg.norm(antihermitian - up - down) /
                            max(np.linalg.norm(up + down), 1e-30))
            if closure > 1e-8:
                raise ValueError(f"Layered-radiation optical theorem failed: {closure}")
            return crad, up, down, integral, closure

        old, new = evaluate(wavelength_b), evaluate(wavelength)
        recovery = float(np.linalg.norm(old[0] - core["Crad_m"]) /
                         max(np.linalg.norm(core["Crad_m"]), 1e-30))
        side_recovery = max(float(np.linalg.norm(old[1] - core["upward_loss_operator_m"]) /
                                  max(np.linalg.norm(core["upward_loss_operator_m"]), 1e-30)),
                            float(np.linalg.norm(old[2] - core["downward_loss_operator_m"]) /
                                  max(np.linalg.norm(core["downward_loss_operator_m"]), 1e-30)))
        if max(recovery, side_recovery) > 1e-8:
            raise ValueError(f"Stored Bragg operator recovery failed: Crad={recovery}, side={side_recovery}")
        name = report["selected_raw_family"]
        fields = core[f"fields_{name}"].reshape(4, -1)
        norm = np.sum(abs(fields)**2)

        def power(operator):
            return float(2 * np.einsum("ik,ij,jk->", fields.conj(), operator, fields).real / norm / 100)

        up_b, up_w, down_b, down_w = power(old[1]), power(new[1]), power(old[2]), power(new[2])
        edges = np.r_[0, np.cumsum(core["gain_cell_width_m"]) * 1e6]
        fraction_top_um = np.clip((edges[1:] - np.maximum(edges[:-1], report["gain_um"] - 1)) /
                                  np.diff(edges), 0, 1)
        return dict(
            source_case=folder.name, gain_um=report["gain_um"], selected_raw_family=name,
            source_CWT_order=report["settings"]["cwt_order"],
            source_finite_grid_points=int(fields.shape[1]**.5),
            fixed_geometry_period_nm=period_nm, recovered_frozen_group_index=group_index,
            detuning_raw_per_m=delta, Bragg_wavelength_nm=wavelength_b,
            linearized_raw_mode_wavelength_nm=wavelength,
            wavelength_shift_nm=wavelength - wavelength_b,
            Bragg_Crad_recovery_relative_error=recovery,
            Bragg_side_operator_recovery_relative_error=side_recovery,
            Crad_relative_matrix_norm_change=float(np.linalg.norm(new[0] - old[0]) / np.linalg.norm(old[0])),
            Crad_matrix_norm_ratio=float(np.linalg.norm(new[0]) / np.linalg.norm(old[0])),
            Crad_amplitude_loss_trace_ratio=float(np.trace(new[1] + new[2]).real /
                                                np.trace(old[1] + old[2]).real),
            top_power_loss_Bragg_cm1=up_b, top_power_loss_shifted_cm1=up_w,
            top_power_loss_ratio=up_w / up_b, top_power_loss_percent_change=100 * (up_w / up_b - 1),
            bottom_power_loss_Bragg_cm1=down_b, bottom_power_loss_shifted_cm1=down_w,
            bottom_power_loss_ratio=down_w / down_b,
            Green_integral_Bragg_real=float(old[3].real), Green_integral_Bragg_imag=float(old[3].imag),
            Green_integral_shifted_real=float(new[3].real), Green_integral_shifted_imag=float(new[3].imag),
            Bragg_optical_theorem_relative_residual=old[4], shifted_optical_theorem_relative_residual=new[4],
            gain_energy_overlap=float(core["q_energy"].sum()),
            gain_top_1um_full_mode_energy_fraction=float(fraction_top_um @ core["q_energy"]),
            source_report_sha256=_sha256(report_path), source_arrays_sha256=_sha256(arrays_path),
            interpretation="frozen_Theta_geometry_finite_fields_q0_ONLY_NOT_frequency_selfconsistent_eigenmode",
        )


def plot_summary(rows: list[dict], path: Path) -> None:
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for key, label, offset in (("top_power_loss_Bragg_cm1", "Frozen Bragg", -.18),
                               ("top_power_loss_shifted_cm1", "Linearized mode wavelength", .18)):
        axes[0].bar(x + offset, [r[key] for r in rows], width=.36, label=label)
    axes[0].set(yscale="log", ylabel="Same-field top power loss (cm^-1)")
    axes[0].legend(fontsize=7)
    axes[1].plot(x, [r["Crad_relative_matrix_norm_change"] * 100 for r in rows], "o-")
    axes[1].set(ylabel="q=0 Crad matrix change (%)")
    axes[2].plot(x, [r["wavelength_shift_nm"] for r in rows], "o-")
    axes[2].axhline(0, color="gray", linewidth=.7)
    axes[2].set(ylabel="Linearized raw-mode shift from Bragg (nm)")
    for ax in axes:
        ax.set_xticks(x, [f"{r['gain_um']:g}" for r in rows])
        ax.set_xlabel("Yb:YAG thickness (um); each period redesigned")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Radiation sensitivity only: Theta / a / finite fields frozen; NO CW update or self-consistent C(omega)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="existing thickness-study directory")
    parser.add_argument("--output", type=Path,
                        help="new audit directory; defaults to results/current/legacy/ybyag_frequency_sensitivity/timestamp")
    parser.add_argument("--guided-padding-um", type=float, default=2.,
                        help="source isolated-guided padding on each side (baseline is 2 um)")
    args = parser.parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve() if args.output is not None else default_output(PROJECT_ROOT, "legacy/ybyag_frequency_sensitivity")
    if output.exists():
        parser.error(f"output already exists; choose a new --output directory: {output}")
    if not np.isfinite(args.guided_padding_um) or args.guided_padding_um <= 0:
        parser.error("guided-padding-um must be positive and finite")
    plan = json.loads((source / "run_plan.json").read_text(encoding="utf-8"))
    folders = sorted(source.glob("case_*"))
    if len(folders) != len(plan["settings"]["thickness_um"]):
        raise ValueError("Not all planned thickness cases are available")
    rows = [audit_case(folder, plan["baseline_parameters"], args.guided_padding_um) for folder in folders]
    prepare_output(output, PROJECT_ROOT)
    write_provenance(output, PROJECT_ROOT)
    with (output / "frozen_frequency_sensitivity.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    payload = dict(source=str(source), source_run_plan_sha256=_sha256(source / "run_plan.json"),
        status="frozen_Theta_geometry_finite_fields_q0_ONLY_NOT_frequency_selfconsistent_eigenproblem",
        source_guided_padding_um=args.guided_padding_um, constant_material_indices=True,
        scope="Stored raw finest-grid mode; q=0 Crad and its top/bottom flux expectation only",
        omissions=["C1D, C2D, Theta, ng and finite eigenfields not re-solved",
                   "no dielectric dispersion, gain spectrum, pump, thermal or CW update",
                   "group-converted wavelength is linearized, not a solved stable lasing frequency",
                   "gain top-1um statistic uses conservative overlap of saved rate cells"],
        prohibition="Do NOT replace one loss in old CW budget and call it a corrected prediction.", rows=rows)
    (output / "frozen_frequency_sensitivity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_summary(rows, output / "frozen_frequency_sensitivity.png")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(f"Output: {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
