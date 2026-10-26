"""Rebuild a four-case gain budget from the audited passive-loss ledger."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from matched300_result_provenance import (DEFAULT_PROTOCOL, validate_completed_manifest,
    validate_case_audit_binding)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_loss_rows(cases, rebuilt, protocol_path=DEFAULT_PROTOCOL):
    """Bind every saved loss ledger to current inputs before plotting or writing."""
    if rebuilt.get("status") != "passed":
        raise ValueError("A passed optical reconstruction is required")
    ids = [row["id"] for row in rebuilt["cases"]]
    if len(ids) != len(set(ids)) or set(ids) != {case["id"] for case in cases}:
        raise ValueError("Optical rebuild must contain exactly the selected distinct cases")
    audits = {row["id"]: row for row in rebuilt["cases"]}
    rows = []
    for case in cases:
        item = audits[case["id"]]
        inputs = validate_case_audit_binding(case, item, protocol_path=protocol_path)
        q = item["rebuilt_gain_weight"]
        internal = inputs["internal_intensity_loss_cm_inverse"]
        local = 2*item["loss_budget_per_m"]["local_amplitude_loss_per_m"]/100
        boundary = 2*item["loss_budget_per_m"]["transport_amplitude_loss_per_m"]/100
        if not math.isfinite(q) or q <= 0 or min(local, boundary) < -1e-10:
            raise ValueError("Invalid passive threshold-loss ledger")
        if abs(local+boundary-2*case["eigenvalue_per_m"][1]/100) > 1e-8:
            raise ValueError("Passive loss ledger differs from selected complex pole")
        total = (internal+local+boundary)/q
        if abs(total-case["g_th_cm_inverse"]) > 1e-8:
            raise ValueError(f"Material-gain budget does not close: {case['id']}")
        if "rebuilt_up_down_power_loss_per_m" in item:
            if abs(sum(item["rebuilt_up_down_power_loss_per_m"])/100-local) > 1e-8:
                raise ValueError("YAG local radiation does not equal upward plus downward flux")
        rows.append(dict(id=case["id"], selected_mode=inputs["case_inputs"]["selected_mode"],
                         wavelength_nm=case["wavelength_nm"], period_nm=inputs["case_inputs"]["actual_period_nm"],
                         case_fingerprint=inputs["case_fingerprint"], protocol_sha256=inputs["protocol_sha256"],
                         gain_weight=q, internal_intensity_loss_cm_inverse=internal,
                         local_radiation_intensity_loss_cm_inverse=local,
                         boundary_transport_intensity_loss_cm_inverse=boundary,
                         internal_material_gain_cm_inverse=internal/q,
                         radiation_material_gain_cm_inverse=local/q,
                         boundary_material_gain_cm_inverse=boundary/q,
                         total_material_gain_cm_inverse=total,
                         passive_intensity_loss_cm_inverse=local+boundary))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results/current/matched300_20261021/comparison")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    manifest_path, rebuilt_path = args.results/"selected_cases.json", args.results/"optical_rebuild.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest, rebuilt = json.loads(manifest_bytes), read(rebuilt_path)
    validate_completed_manifest(manifest)
    rows = build_loss_rows(manifest["cases"], rebuilt, args.protocol)
    protocol_sha = digest(args.protocol)
    if (digest(manifest_path) != manifest_sha
            or any(row["protocol_sha256"] != protocol_sha for row in rows)):
        raise ValueError("Manifest or protocol changed during budget validation")
    with (args.results/"threshold_loss_ledger.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(status="passed", manifest_sha256=manifest_sha,
                  optical_rebuild_sha256=digest(rebuilt_path), protocol_sha256=protocol_sha,
                  rows=rows, scope="Same 300 x 300 um aperture; sum of material-gain contributions. Different optimized motifs and gain weights do not isolate a Bravais-lattice effect or wall-plug efficiency.")
    (args.results/"threshold_loss_ledger.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    components = [("internal_material_gain_cm_inverse", "Internal loss", "#999999"),
                  ("radiation_material_gain_cm_inverse", "Radiation", "#0072B2"),
                  ("boundary_material_gain_cm_inverse", "Boundary transport", "#D55E00")]
    for ax, prefix, title in zip(axes, ("semiconductor", "YbYAG"), ("Semiconductor / 950 nm", "Yb:YAG / 1030 nm")):
        selected = [next(row for row in rows if row["id"] == prefix+suffix) for suffix in ("_square", "_triangular")]
        lower = [0., 0.]
        for key, label, color in components:
            heights = [row[key] for row in selected]
            ax.bar([0, 1], heights, bottom=lower, width=.6, label=label, color=color)
            lower = [a+b for a, b in zip(lower, heights)]
        for x, row in enumerate(selected):
            ax.text(x, row["total_material_gain_cm_inverse"]*1.025,
                    f"{row['total_material_gain_cm_inverse']:.2f}\nq = {row['gain_weight']:.3f}", ha="center", fontsize=9)
        ax.set(xticks=[0, 1], xticklabels=["Square", "Triangular"], title=title,
               ylabel="Equivalent material gain (cm$^{-1}$)", ylim=(0, max(lower)*1.19))
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Threshold loss budget | matched 300 x 300 um aperture", fontsize=13)
    fig.savefig(args.results/"threshold_loss_budget.png", dpi=200)
    plt.close(fig)
    print(json.dumps(dict(status="passed", cases=len(rows))))


if __name__ == "__main__":
    main()
