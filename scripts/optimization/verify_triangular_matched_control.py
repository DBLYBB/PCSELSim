"""Check the matched 200-um triangular control with a wider pole search.

Internal intensity loss belongs in g_th after the passive optical solve.
This explicitly contrasts zero/internal field shifts to prevent conventions
from masquerading as a square-versus-triangular effect.
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
import numpy as np
from scripts.optimization.triangular_semiconductor_search import (
    Motif, build_model, evaluate_finite, json_write, threadpool_limits)
from pcselsim.triangular_finite import TriangularFiniteSpec, solve_triangular_finite_modes
from pcselsim.run_output import prepare_output, default_output, write_provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = prepare_output(args.output or default_output(ROOT, "optimization_20261021/triangular/matched_control"), ROOT)
    write_provenance(output, ROOT)
    spec = Motif("tri_f029_asp108_200um", "ellipse", .29, aspect=1.08,
        pc_thickness_nm=120., aperture="square", radius_um=100.)
    rows = []
    with threadpool_limits(limits=1):
        p, candidate, cell, model = build_model(spec, 8, 3.)
        for n in (7, 10, 13):
            searched = evaluate_finite(spec, n, 8, 3., 24,
                output/f"wide_N{n}", extra_targets=17)
            selected = solve_triangular_finite_modes(model,
                TriangularFiniteSpec(100., n, 0., 12, "square", "staggered"))
            alpha_native = min(mode.alpha_per_m for mode in selected.values())
            rows.append({"N": n, "wide_search": searched,
                "native_lowest_alpha_field_cm": alpha_native/100,
                "native_lowest_family": min(selected, key=lambda name: selected[name].alpha_per_m),
                "native_minus_wide_alpha_cm": alpha_native/100-searched["alpha_field_cm"]})
            json_write(output/"matched_control_audit.json", rows)
            print(n, searched["alpha_field_cm"], alpha_native/100, flush=True)
    json_write(output/"interpretation.json", {
        "gain_definition": "(2*passive_field_loss_cm+5)/(Gamma*3.584/n_eff)",
        "internal_loss_convention": "TriangularFiniteSpec.internal_loss_cm shifts amplitude eigenvalues by the stated number in cm^-1. It is set to zero here, matching the square solver's passive coupling convention.",
        "spectrum_scope": "17 additional real-frequency targets plus all six Gamma edges; not exhaustive fine-grid spectrum",
        "native_selector_missed_lower_found_pole": any(r["native_minus_wide_alpha_cm"] > 1e-6 for r in rows)})


if __name__ == "__main__": main()
