"""Plot the two primary YAG designs' stored shared-inversion CW results."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results/current/matched300_20261021")
    args = parser.parse_args()
    source = args.results/"yag/pareto.json"
    manifest_path = args.results/"comparison/selected_cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    validate_completed_manifest(manifest)
    cases = [r for r in json.loads(source.read_text(encoding="utf8"))["cases"]
             if r["selection_status"] == "primary"]
    if len(cases) != 2 or {r["lattice"] for r in cases} != {"square", "triangular"}:
        raise ValueError("Require one completed primary for each YAG lattice")
    for case in cases:
        selected = next(r for r in manifest["cases"] if r["id"] == "YbYAG_"+case["lattice"])
        if (selected["summary_path"] != case["summary_path"]
                or selected["selected_mode"] != case["actual_edge_first"]["mode"]):
            raise ValueError("The CW plot and four-cavity comparison select different YAG results")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    rows = []
    colors = {"square": "#0072B2", "triangular": "#D55E00"}
    for ax, architecture, title in zip(axes, ("edge_dual", "top_single"),
                                     ("Dual-edge pump", "Single-pass top pump")):
        for case in sorted(cases, key=lambda r: r["lattice"]):
            pump = case["pumps"][architecture]
            samples = pump["cw"]
            if any(p["status"] != "completed" for p in samples):
                raise ValueError("A failed CW sample must be resolved or explicitly presented as incomplete")
            threshold = pump["first_internal_W"]
            if threshold is None:
                raise ValueError("Cannot normalize CW powers by an unreachable threshold")
            for point in samples:
                if abs(point["power_internal_W"]/threshold-point["ratio"]) > 1e-9:
                    raise ValueError("Stored CW ratio does not match this architecture's own threshold")
                rows.append(dict(lattice=case["lattice"], architecture=architecture,
                                 first_internal_W=threshold, **point))
            label = f"{case['lattice'].capitalize()} | Pth = {threshold*1000:.3f} mW" if architecture == "edge_dual" else f"{case['lattice'].capitalize()} | Pth = {threshold:.3f} W"
            xs = [p["ratio"] for p in samples]
            ys = [1000*p["upward_output_W"] for p in samples]
            ax.plot(xs, ys, "o-", color=colors[case["lattice"]], label=label, lw=1.7, ms=5)
            for i, point in enumerate(samples):
                ax.annotate(str(len(point["active_modes"])), (xs[i], ys[i]),
                            xytext=(0, 7 if case["lattice"] == "triangular" else -13),
                            textcoords="offset points", ha="center", fontsize=8,
                            color=colors[case["lattice"]])
        ax.set(title=title, xlabel="Internal pump / own first threshold", ylabel="Total upward output (mW)",
               xlim=(.95, 4.1))
        ax.set_ylim(bottom=-.30)  # Keep low-power mode-count labels inside the panel.
        ax.grid(alpha=.18); ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Yb:YAG | shared-inversion static CW competition", fontsize=14)
    fig.supxlabel("Numbers give active-mode counts at the sampled powers. Both cavities: 300 x 300 um.\n"
                  "Nominal material data, fixed passive-mode basis and x-z inversion with y averaging; no dynamic stability inference.", fontsize=9)
    output = args.results/"comparison"
    fig.savefig(output/"YAG_shared_CW.png", dpi=200)
    plt.close(fig)
    report = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), samples=rows,
                  scope="Own-architecture first-entry internal pump; total upward output from the same stored shared-inversion CW solve. Sampled active-mode counts do not locate an exact second-mode turn-on or prove dynamic stability.")
    (output/"YAG_shared_CW.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(dict(status="completed", samples=len(rows))))


if __name__ == "__main__":
    main()
