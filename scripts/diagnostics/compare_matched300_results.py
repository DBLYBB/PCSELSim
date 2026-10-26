"""Read final saved Jones fields back and compare matched 300-um designs.

The manifest is deliberately explicit about NPZ keys and scalar provenance.
This script never infers a lasing threshold from the appearance of a beam.
Plots and tables are rebuilt from the selected final files, not scan proxies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from pcselsim.design_objectives import beam_metrics
from pcselsim.polarization import polarization_metrics, stokes_maps
from matched300_result_provenance import (DEFAULT_PROTOCOL, capture_case_inputs,
    validate_case_audit_binding, validate_completed_manifest)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_case(case, protocol_path=None):
    inputs = capture_case_inputs(case, protocol_path=protocol_path)
    pump_selection = inputs["selection"]
    path = root_path(case["fields_path"])
    with np.load(path, allow_pickle=False) as saved:
        axis = saved[case["axis_key"]].copy()
        fx, fy = (saved[case[key]].copy() for key in ("far_x_key", "far_y_key"))
    beam = beam_metrics(axis, fx, fy, requested_view_deg=1.,
                        source_nyquist_deg=case["source_nyquist_deg"])
    pol = polarization_metrics(axis, fx, fy, core_radius_deg=.2)
    if case["nominal_aperture_um"] != [300, 300]:
        raise ValueError(f"Unmatched nominal footprint in {case['id']}")
    if abs(case["wavelength_nm"] - case["target_wavelength_nm"]) > .05:
        raise ValueError(f"Actual wavelength misses the common target: {case['id']}")
    checks = {}
    for key in ("ellipticity", "center_to_peak", "radial_anisotropy_l2"):
        expected = case["reported_beam"][key]
        delta = abs(beam[key] - expected)
        checks[key] = {"reported": expected, "readback": beam[key],
                       "absolute_difference": delta}
        if delta > 1e-8 * max(1., abs(expected)):
            raise ValueError(f"Saved-field readback mismatch: {case['id']} / {key}")
    geometry = inputs["geometry"]
    if "nominal_aperture_um" in geometry and geometry["nominal_aperture_um"] != [300, 300]:
        raise ValueError(f"Unmatched stored geometry footprint in {case['id']}")
    if "device_square_side_um" in geometry and geometry["device_square_side_um"] != 300:
        raise ValueError(f"Unmatched stored geometry side in {case['id']}")
    audit = dict(id=case["id"], fields_path=case["fields_path"],
                 fields_sha256=inputs["fields_sha256"],
                 geometry_sha256=inputs["geometry_sha256"],
                 summary_path=case["summary_path"],
                 summary_sha256=inputs["summary_sha256"],
                 protocol_sha256=inputs["protocol_sha256"], case_inputs=inputs["case_inputs"],
                 case_fingerprint=inputs["case_fingerprint"],
                 pump_selection=pump_selection,
                 beam_readback=beam, polarization_readback=pol,
                 reported_beam_checks=checks)
    validate_case_audit_binding(case, audit, protocol_path=protocol_path)
    return axis, fx, fy, geometry, audit


def plot_geometry(ax, geometry, lattice):
    period = geometry["lattice_constant_nm"]
    direct = (np.eye(2) if lattice == "square" else
              np.array([[np.sqrt(3)/2, np.sqrt(3)/2], [-.5, .5]]))
    colors = ("#0072B2", "#D55E00", "#009E73")
    # Periodic copies make a satellite crossing a primitive boundary visible.
    for i in range(-2, 3):
        for j in range(-2, 3):
            shift = direct @ [i, j]
            for h, color in zip(geometry["holes"], colors):
                points = np.asarray(h.get("boundary_cartesian_nm", h.get("boundary_nm"))) / period + shift
                ax.add_patch(Polygon(points, facecolor=color, edgecolor="white", lw=.3))
    vertices = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) @ direct.T
    ax.add_patch(Polygon(vertices, fill=False, edgecolor="#202020", lw=1.5, ls="--"))
    ax.set(xlim=(-.9, .9), ylim=(-.9, .9), xlabel="x / a", ylabel="y / a")
    ax.set_aspect("equal")
    ax.set_title(f"Optimized holes | a = {period:.2f} nm", fontsize=10)


def overview(cases, loaded, output):
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 3.5 * len(cases)),
                             squeeze=False, constrained_layout=True)
    for row, (case, data) in enumerate(zip(cases, loaded)):
        axis, fx, fy, geometry, audit = data
        plot_geometry(axes[row, 0], geometry, case["lattice"])
        axes[row, 0].text(-.32, .5, case["label"], rotation=90,
                          transform=axes[row, 0].transAxes, ha="center", va="center",
                          fontsize=12, fontweight="bold")
        intensity = abs(fx)**2 + abs(fy)**2
        intensity /= intensity.max()
        extent = [axis[0], axis[-1], axis[0], axis[-1]]
        view = .5
        image = axes[row, 1].imshow(intensity, origin="lower", extent=extent,
                                     cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 1].contour(axis, axis, intensity, levels=[.5], colors="cyan", linewidths=.8)
        beam = audit["beam_readback"]
        axes[row, 1].set_title(
            f"Vector intensity | axial ratio {beam['ellipticity']:.4f}\n"
            f"Center / peak {beam['center_to_peak']:.4f}", fontsize=10)
        maps = stokes_maps(fx, fy, intensity_floor=.01)
        s1 = np.ma.array(maps["s1_over_s0"], mask=~maps["valid_mask"])
        pimage = axes[row, 2].imshow(s1, origin="lower", extent=extent,
                                     cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
        stride = max(1, len(axis)//18)
        xx, yy = np.meshgrid(axis, axis)
        valid = maps["axis_valid_mask"][::stride, ::stride]
        angle = maps["axis_angle_rad"][::stride, ::stride]
        axes[row, 2].quiver(xx[::stride, ::stride][valid], yy[::stride, ::stride][valid],
                            np.cos(angle)[valid], np.sin(angle)[valid],
                            color="#202020", pivot="mid", headwidth=0, headlength=0,
                            headaxislength=0, scale=25, width=.003)
        pol = audit["polarization_readback"]
        core = pol["core"]
        axes[row, 2].set_title(
            f"Polarization S1 / S0 + linear axes\n"
            f"Core DOLP {core['degree_of_linear_polarization']:.3f}, "
            f"S3/S0 {core['signed_circular_fraction_s3_over_s0']:+.3f}", fontsize=10)
        for ax in axes[row, 1:]:
            ax.set(xlim=(-view, view), ylim=(-view, view),
                   xlabel=r"$\theta_x$ (deg)", ylabel=r"$\theta_y$ (deg)")
            ax.set_aspect("equal")
        fig.colorbar(image, ax=axes[row, 1], fraction=.046, pad=.02, label="Normalized intensity")
        fig.colorbar(pimage, ax=axes[row, 2], fraction=.046, pad=.02, label="S1 / S0")
    fig.suptitle("Matched 300 x 300 um cavities | saved final eigenfields", fontsize=15)
    fig.savefig(output / "four_cavity_overview.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/current/matched300_20261021/comparison")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    cases = manifest["cases"]
    validate_completed_manifest(manifest)
    loaded = [read_case(case, args.protocol) for case in cases]
    protocol_sha = file_hash(args.protocol)
    if (file_hash(args.manifest) != manifest_sha
            or any(item[-1]["protocol_sha256"] != protocol_sha for item in loaded)):
        raise ValueError("Manifest or protocol changed during field readback; rerun on frozen inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    audit = dict(manifest_sha256=manifest_sha, protocol_sha256=protocol_sha,
                 cases=[item[-1] for item in loaded],
                 scope="Readback validates stored Jones-field metrics; spectral completeness and threshold evidence remain in each linked final summary.")
    (args.output / "saved_field_readback.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    rows = []
    for case, data in zip(cases, loaded):
        beam, pol = data[-1]["beam_readback"], data[-1]["polarization_readback"]
        pump = data[-1]["pump_selection"] or {}
        rows.append(dict(id=case["id"], wavelength_nm=case["wavelength_nm"],
                         selected_mode=pump.get("selected_mode", "lowest_material_gain_mode"),
                         selected_pump_architecture=pump.get("selected_pump_architecture"),
                         period_nm=data[3]["lattice_constant_nm"],
                         g_th_cm_inverse=case["g_th_cm_inverse"],
                         relative_gain_gap=case.get("relative_gain_gap"),
                         threshold_current_A=case.get("threshold_current_A"),
                         relative_current_gap=case.get("relative_current_gap"),
                         radiative_fraction_of_threshold_power_loss=case.get("radiative_fraction_of_threshold_power_loss"),
                         internal_edge_pump_W=case.get("internal_edge_pump_W"),
                         internal_top_pump_W=case.get("internal_top_pump_W"),
                         edge_first_modes=";".join(pump.get("first_modes_by_architecture", {}).get("edge_dual", [])),
                         top_first_modes=";".join(pump.get("first_modes_by_architecture", {}).get("top_single", [])),
                         top_first_wavelengths_nm=";".join(str(v) for v in pump.get("first_wavelengths_nm_by_architecture", {}).get("top_single", [])),
                         unlased_edge_branch_gap_fraction=case.get("unlased_edge_branch_gap_fraction"),
                         unlased_top_branch_gap_fraction=case.get("unlased_top_branch_gap_fraction"),
                         axial_ratio=beam["ellipticity"],
                         center_to_peak=beam["center_to_peak"],
                         radial_anisotropy_l2=beam["radial_anisotropy_l2"],
                         encircled_0p2deg=beam["encircled_0p2deg"],
                         core_DOLP=pol["core"]["degree_of_linear_polarization"],
                         core_S3_over_S0=pol["core"]["signed_circular_fraction_s3_over_s0"],
                         on_axis_linear_angle_deg=pol["on_axis"]["linear_axis_angle_deg"]))
    with (args.output / "four_cavity_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    overview(cases, loaded, args.output)
    print(json.dumps({"cases": len(cases), "output": str(args.output), "readback": "passed"}))


if __name__ == "__main__":
    main()
