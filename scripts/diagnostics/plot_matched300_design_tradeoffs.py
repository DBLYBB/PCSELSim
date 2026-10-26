"""Plot retained, actual-frequency designs and their competing objectives.

The points are qualified candidates, not the many frozen-reference screen
points. Their labels retain the source modal-isolation scope. Different
optimized motifs do not constitute a controlled Bravais-lattice experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from matched300_result_provenance import validate_completed_manifest


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results/current/matched300_20261021")
    args = parser.parse_args()
    validate_completed_manifest(read(args.results/"comparison/selected_cases.json"))
    sources = [args.results/"square/final_summary.json",
               args.results/"triangular/final_summary.json", args.results/"yag/pareto.json"]
    square, triangle, yag = map(read, sources)
    rows = []
    square_roles = {"best_low_threshold": "Lowest gain", "best_balanced": "Axial ratio / low gain",
                    "best_isolation": "Isolation primary"}
    for case in square["candidates"]:
        rows.append(dict(material="Semiconductor", lattice="square", role=square_roles[case["role"]],
                         primary=case["role"] == "best_isolation", gain=case["g1_cm1"],
                         beam=case["beam"], cold_gap=case["relative_material_gap"],
                         summary_path=case["summary_path"]))
    for name, case in triangle["cases"].items():
        mode = case["lowest"]
        primary = name == triangle["primary_case"]
        rows.append(dict(material="Semiconductor", lattice="triangular",
                         role="Low gain primary" if primary else "Axial-ratio Pareto",
                         primary=primary, gain=mode["g_th_cm"], beam=mode["beam"],
                         cold_gap=case["g_th_gap_fraction"], summary_path=case["summary_path"]))
    for case in yag["cases"]:
        mode = case["actual_edge_first"]
        rows.append(dict(material="Yb:YAG", lattice=case["lattice"],
                         role=case["selection_status"].replace("_", " "),
                         primary=case["selection_status"] == "primary", gain=mode["gain_cm1"],
                         beam=mode["beam"], cold_gap=(case["cold_gain_gap_fraction"]
                             if mode["mode"] == case["cold_first_mode"] else None),
                         edge_internal_W=case["pumps"]["edge_dual"]["first_internal_W"],
                         zero_photon_edge_gap=case["pumps"]["edge_dual"]["zero_photon_branch_gap_fraction"],
                         summary_path=case["summary_path"]))
    if sum(r["primary"] for r in rows) != 4:
        raise ValueError("Exactly four qualified primary designs must be identified")
    for row in rows:
        if row["beam"]["center_to_peak"] < .9:
            raise ValueError("A retained bright-center design has a dark first mode")
        if not (ROOT/row["summary_path"]).is_file():
            raise FileNotFoundError(row["summary_path"])
    out = args.results/"comparison"
    document = dict(rows=rows, source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sources},
                    scope="Retained actual-frequency designs, same 300 x 300 um square aperture. Cold gain gaps have each branch's stated spectral scope. YAG points use actual edge_dual first modes; a zero-photon gap is not saturated CW turn-on. Different optimized motifs are not a controlled lattice-only comparison.")
    (out/"design_tradeoffs.json").write_text(json.dumps(document, indent=2), encoding="utf8")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), constrained_layout=True)
    colors = {"square": "#0072B2", "triangular": "#D55E00"}
    for ax, material in zip(axes, ("Semiconductor", "Yb:YAG")):
        group = [r for r in rows if r["material"] == material]
        for index, row in enumerate(group):
            x, y = row["beam"]["radial_anisotropy_l2"], row["gain"]
            ax.scatter(x, y, color=colors[row["lattice"]], s=155 if row["primary"] else 65,
                       marker="*" if row["primary"] else "o", edgecolor="white", linewidth=.6, zorder=3)
            gap_label = (f"Cold gap {100*row['cold_gap']:.2f}%" if row["cold_gap"] is not None
                         else "Edge winner differs from cold first")
            text = f"{row['lattice'].capitalize()}: {row['role']}\n{gap_label}"
            # Alternating offsets keep nearby retained designs distinguishable.
            dx, dy = (8, 13) if index % 2 == 0 else (8, -26)
            if material == "Semiconductor" and row["role"] == "Lowest gain":
                dx, dy = 8, -30
            ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                        color=colors[row["lattice"]], arrowprops=dict(arrowstyle="-", lw=.4, color=".65"))
        xmax = max(r["beam"]["radial_anisotropy_l2"] for r in group)
        ymin, ymax = min(r["gain"] for r in group), max(r["gain"] for r in group)
        span = max(ymax-ymin, 1.)
        ax.set(xlim=(0, xmax*1.8), ylim=(ymin-.25*span, ymax+.30*span),
               xlabel="Radial intensity anisotropy L2 (lower is rounder)",
               ylabel="First-mode material gain (cm$^{-1}$)", title=material)
        ax.grid(alpha=.18); ax.set_axisbelow(True)
    fig.suptitle("Retained design tradeoffs | matched 300 x 300 um aperture", fontsize=14)
    fig.supxlabel("Stars: four comparison primaries. Labels: cold material-gain gap within each documented modal search.\n"
                  "Lower gain and rounder intensity can trade against modal isolation; circular intensity is independent of polarization.", fontsize=9)
    fig.savefig(out/"design_tradeoffs.png", dpi=200)
    plt.close(fig)
    print(json.dumps(dict(status="completed", retained_designs=len(rows))))


if __name__ == "__main__":
    main()
