"""Plot the fixed-reference high-grid semiconductor triangle rotation scan.

The tracked-branch local scan is exploratory. The final selected design
must separately pass the full-coarse-spectrum-guided census and frequency
closure; this plot must never supply its reported modal isolation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def signature(row):
    specification = dict(row["three_hole_spec"])
    specification.pop("name", None)
    specification.pop("main_orientation_deg")
    return specification, row["N"], row["D"], row["dz_nm"], row["reference_wavelength_nm"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results/current/matched300_20261021")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results/"comparison"
    paths = sorted((args.results/"triangular").glob("stripe_safe_actual_angle*/candidates.json"))
    paths += [args.results/"triangular/stripe_safe_actual_local_N65_18/candidates.json"]
    rows, provenance, reference, failed_angles = {}, [], None, set()
    for path in paths:
        records = json.loads(path.read_text(encoding="utf8"))
        provenance.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for row in records:
            if row.get("status") == "failed" and row["name"].startswith("main_orientation_deg_"):
                failed_angles.add(row["three_hole_spec"]["main_orientation_deg"])
            if row.get("status") != "completed" or not (row["name"] == "nominal" or row["name"].startswith("main_orientation_deg_")):
                continue
            if reference is None:
                reference = signature(row)
            if signature(row) != reference:
                raise ValueError("Angle plot cannot combine different designs, grids or frozen optical references")
            angle = row["three_hole_spec"]["main_orientation_deg"]
            rows[angle] = dict(main_orientation_deg=angle, g_tracked_bright_cm1=row["g_th_cm"],
                nearest_tracked_competitor_cm1=min(p["g_th_cm"] for p in row["tracked_modes"][1:]),
                tracked_branches=len(row["tracked_modes"]),
                axial_ratio=row["beam"]["ellipticity"], center_to_peak=row["beam"]["center_to_peak"])
    failed_angles.difference_update(rows)
    rows = [rows[key] for key in sorted(rows)]
    if len(rows) < 2:
        raise ValueError("At least two completed orientations required")
    output.mkdir(parents=True, exist_ok=True)
    with (output/"semiconductor_triangle_angle_scan.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    x = [row["main_orientation_deg"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
    axes[0].plot(x, [row["g_tracked_bright_cm1"] for row in rows], "o-", label="Tracked bright branch")
    axes[0].plot(x, [row["nearest_tracked_competitor_cm1"] for row in rows], "s--", label="Nearest tracked competitor")
    axes[0].set(xlabel="Central rounded-triangle orientation (deg)", ylabel="Material gain estimate (cm$^{-1}$)")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, [row["axial_ratio"] for row in rows], "o-", color="#D55E00", label="Axial ratio")
    axes[1].axhline(1.05, color="gray", linestyle="--", label="Preferred axial-ratio target")
    axes[1].set(xlabel="Central rounded-triangle orientation (deg)", ylabel="Far-field axial ratio")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
        for failed_angle in sorted(failed_angles):
            axis.axvline(failed_angle, color="#CC3311", linestyle=":", alpha=.5)
    if failed_angles:
        axes[0].plot([], [], ":", color="#CC3311", label="Unresolved continuation")
        axes[0].legend(fontsize=8)
    fig.suptitle("300 x 300 um semiconductor / triangular lattice | fixed-reference local scan", fontsize=12)
    fig.text(.5, -.025, "N65, D48, dz = 1 nm | Tracked branch continuation only; no independent frequency closure or full spectrum claimed", ha="center", fontsize=8)
    fig.savefig(output/"semiconductor_triangle_angle_scan.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (output/"semiconductor_triangle_angle_scan_provenance.json").write_text(json.dumps(dict(
        scope="Exploratory same-geometry rotation scan; not final closed threshold or isolation evidence",
        common_signature=reference, sources=provenance, points=len(rows),
        unresolved_orientations_deg=sorted(failed_angles)), indent=2), encoding="utf8")
    print(json.dumps(dict(points=len(rows), output=str(output))))


if __name__ == "__main__":
    main()
