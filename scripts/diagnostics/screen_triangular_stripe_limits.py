"""Screen existing bright triangular candidates against frozen-C stripe limits.

This diagnostic supplies new geometries to the genuine two-dimensional search.
Its proxy gap is never a replacement for finite transverse mode competition.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from pcselsim.vertical import Layer, LayerStack
from pcselsim.triangular_six_wave import TriangularLatticeCell, TriangularCWTSettings, build_triangular_coupling
from scripts.optimization.matched300_triangular_search import ThreeHole, candidate, base_spec
from scripts.optimization.triangular_semiconductor_search import parameters
from scripts.diagnostics.audit_counterpropagating_stripes import stripe_pair


def fixed_model(design, order=16, dz=2., wavelength_nm=950.):
    p = parameters(base_spec(design))
    cell = TriangularLatticeCell(p.BACKGROUND_INDEX**2, candidate(design).inclusions)
    average_index = np.sqrt(cell.fourier_epsilon(0, 0).real)
    stack = LayerStack(tuple(Layer(l.name, l.thickness_nm,
        average_index if l.name == p.PC_LAYER_NAME else l.refractive_index) for l in p.LAYERS),
        p.TOP_CLADDING_INDEX, p.BOTTOM_CLADDING_INDEX, p.VERTICAL_PADDING_UM)
    model = build_triangular_coupling(cell, stack, p.PC_LAYER_NAME, design.lattice_nm,
        wavelength_nm, TriangularCWTSettings(order, dz), fixed_wavelength_nm=wavelength_nm)
    return p, model


def evaluate(row, order=16, dz=2.):
    design = ThreeHole(**row["three_hole_spec"])
    p, model = fixed_model(design, order, dz)
    gain_weight = model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME] * p.ACTIVE_INDEX / model.effective_index
    poles = []
    for indices, length in (((0, 1), 600/np.sqrt(3)), ((2, 3), 600/np.sqrt(3)), ((4, 5), 300.)):
        pair = model.coupling_m[np.ix_(indices, indices)]
        diagnostic = stripe_pair(pair, length, gain_weight, 5., grids=(31,))
        pole = diagnostic["independent_continuous_transfer_pole"]
        seed = diagnostic["discrete_runs"][0]["poles"][0]
        # A false root reached by escaping to infinite loss is not usable.
        pole["converged"] = bool(pole["converged"] and pole["alpha_field_per_m"] >= -1e-7 and
            pole["alpha_field_per_m"] < seed["alpha_field_per_m"] + 20e6/length and
            abs(pole["delta_per_m"]-seed["delta_per_m"]) < 20*np.pi*1e6/length)
        poles.append(dict(indices=indices, chord_um=length,
                          coarse_seed_alpha_per_m=seed["alpha_field_per_m"], **pole))
    valid = all(pole["converged"] for pole in poles)
    minimum = min(p["g_th_cm_inverse"] for p in poles) if valid else None
    return dict(name=design.name, three_hole_spec=asdict(design),
        source_directory=row["_source_directory"], source_g_th_cm=row["g_th_cm"],
        source_N=row["N"], source_D=row["D"],
        source_ellipticity=row["ellipticity"], source_center_to_peak=row["center_to_peak"],
        source_radial_anisotropy_l2=row["radial_anisotropy_l2"], source_eta_rad=row["eta_rad"],
        source_material_gain_gap_fraction=row["g_th_gap_fraction"],
        order=order, dz_nm=dz, fixed_wavelength_nm=950., gain_weight=gain_weight,
        minimum_stripe_g_th_cm=minimum,
        stripe_vs_source_gain_gap_proxy=minimum/row["g_th_cm"]-1 if valid else None,
        stripe_poles=poles, coupling_sha256=hashlib.sha256(model.coupling_m.tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/current/matched300_20261021/stripe_screen")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    base = ROOT / "results/current/matched300_20261021/triangular"
    selected, seen = [], set()
    folders = ("highD_adaptive128", "pareto_rescreen_N17", "sobol1024", "circle_grid567", "local256", "local_targeted192")
    for folder in folders:
        for row in json.loads((base / folder / "candidates.json").read_text(encoding="utf8")):
            if row.get("status") != "finite_solved" or row["ellipticity"] > 1.15 or row["center_to_peak"] < .9:
                continue
            design = dict(row["three_hole_spec"])
            if design["main_share"] < (1-design["main_share"])*max(design["satellite_balance"], 1-design["satellite_balance"]):
                continue
            design.pop("name")
            signature = json.dumps(design, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            row["_source_directory"] = (base / folder).relative_to(ROOT).as_posix()
            selected.append(row)
    if args.limit:
        selected = selected[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.monotonic()
    with (args.output / "stripe_screen.jsonl").open("w", encoding="utf8") as stream:
        for index, row in enumerate(selected):
            try:
                result = evaluate(row)
            except Exception as error:
                result = dict(name=row["name"], error=repr(error))
            records.append(result)
            stream.write(json.dumps(result)+"\n")
            stream.flush()
            if (index+1) % 8 == 0 or index+1 == len(selected):
                valid = [r for r in records if r.get("minimum_stripe_g_th_cm") is not None]
                ranked = sorted(valid, key=lambda r: -r["stripe_vs_source_gain_gap_proxy"])
                (args.output / "ranked_candidates.json").write_text(json.dumps(ranked, indent=2), encoding="utf8")
                isolated = sorted([r for r in valid if r["stripe_vs_source_gain_gap_proxy"] >= .1
                    and r["source_material_gain_gap_fraction"] >= .1
                    and r["source_ellipticity"] <= 1.1 and r["source_center_to_peak"] >= .98],
                    key=lambda r: r["source_g_th_cm"])
                (args.output / "isolated_proxy_candidates.json").write_text(json.dumps(isolated, indent=2), encoding="utf8")
                (args.output / "run_status.json").write_text(json.dumps(dict(completed=index+1,
                    attempted=len(selected), valid_stripe_roots=len(valid), elapsed_seconds=time.monotonic()-started,
                    scope="Different-fidelity frozen-C stripe-vs-source gain proxy only; every proposed design needs genuine fine 2D QZ-guided competition and actual frequency closure."), indent=2), encoding="utf8")
                print(f"{index+1}/{len(selected)}; valid {len(valid)}; proxy pass10% {sum(r['stripe_vs_source_gain_gap_proxy']>=.1 for r in valid)}", flush=True)


if __name__ == "__main__":
    main()
