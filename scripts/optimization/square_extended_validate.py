"""Sample lateral spectra and close the real frequency of arbitrary low poles.

The source four-family labels do not choose the winner. Independent poles from
20 shift-invert clusters are ranked, then tracked by their complex fields while
C(lambda) and the vertical mode are rebuilt. This remains a sampled scalar CWT
spectrum, not a full Maxwell or nonlinear single-mode certification.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from pcselsim.custom_analysis import FourWaveOpticalSpec, LinearMode, finite_area_eigenproblem
from pcselsim.design_objectives import beam_acceptance
from pcselsim.finite_discretization import relative_residual
from pcselsim.frequency_cwt import background_detuning
from pcselsim.run_output import prepare_output, write_provenance
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from scripts.optimization.square_semiconductor_search import (
    BASE, Design, build_cell, build_stack, direct_vector_far_field,
    fabrication, save_json, threshold)
from scripts.optimization.square_semiconductor_validate import survey


def close_survey(d, folder, spectrum, *, order=10, dz=2., roots=12):
    """Close saved leading poles, with fixed geometry and normalized overlaps."""
    grid = spectrum["grid"]
    cell = build_cell(d, 64)
    stack = build_stack(d, cell)
    settings = ThreeDCWTSettings(order, dz)
    optical_cache = {}

    def model_at(wavelength):
        key = None if wavelength is None else float(wavelength)
        if key not in optical_cache:
            optical_cache[key] = build_geometry_coupling(cell, stack, BASE.PC_LAYER_NAME,
                d.lattice_nm, BASE.WAVELENGTH_GUESS_NM, settings,
                fixed_wavelength_nm=key)
        return optical_cache[key]

    reference = model_at(spectrum.get("fixed_wavelength_nm"))
    wavelength0 = reference.bragg_wavelength_nm
    saved = np.load(folder / "lateral_poles.npz", allow_pickle=False)
    records, arrays = [], {}
    for rank, row in enumerate(spectrum["ordered_poles"][:min(roots, 24)]):
        started = time.monotonic()
        field0 = saved[f"pole_{row['pole_id']}_field"]
        value0 = complex(row["detuning_per_m"], row["alpha_field_cm1"]*100)
        seed = field0.ravel()
        trace, cache = [], {}

        def evaluate(wavelength):
            key = float(wavelength)
            if key in cache:
                return cache[key]
            model = model_at(wavelength)
            spec = FourWaveOpticalSpec(wavelength, model.effective_index, model.group_index,
                d.length_um, BASE.INTERNAL_LOSS_CM, (0.,)*4, (0.,)*4,
                grid, "staggered", 16, None, 1600)
            problem = finite_area_eigenproblem(spec, model.coupling_m)
            change = model.coupling_m-reference.coupling_m
            flat = field0.reshape(4, -1)
            predictor = value0 + np.einsum("ik,ij,jk->", flat.conj(), change, flat)/np.sum(abs(flat)**2)
            for count in (8, 24, 48):
                values, fields = problem.near(predictor, count, seed, tol=1e-10, maxiter=1600)
                overlaps = abs(seed.conj() @ fields)**2 / np.sum(abs(seed)**2)
                chosen = int(np.argmax(overlaps))
                overlap = float(overlaps[chosen])
                if overlap >= .60:
                    break
            if overlap < .60:
                raise RuntimeError(f"Arbitrary pole {rank} lost its reference field: overlap={overlap}")
            value = values[chosen]
            field = fields[:, chosen].reshape(field0.shape)
            if value.imag < -1e-7:
                raise RuntimeError("Frequency tracking returned active passive pole")
            residual = background_detuning(wavelength, model.effective_index, d.lattice_nm)-value.real
            trace.append({"wavelength_nm":wavelength, "effective_index":model.effective_index,
                "finite_detuning_per_m":value.real, "alpha_field_cm1":value.imag/100,
                "frequency_residual_per_m":residual, "reference_field_overlap":overlap,
                "eigenpairs_returned":len(values)})
            cache[key] = model, spec, problem, value, field, overlap, float(residual)
            return cache[key]

        try:
            half = .5
            while True:
                left, right = wavelength0-half, wavelength0+half
                fl, fr = evaluate(left)[-1], evaluate(right)[-1]
                if fl*fr <= 0:
                    break
                half *= 2
                if half > 40:
                    raise RuntimeError("No real frequency root within +/-32 nm")
            root = float(brentq(lambda w:evaluate(w)[-1], left, right, xtol=2e-8))
            model, spec, problem, value, field, overlap, residual = evaluate(root)
            if abs(residual) > .05:
                raise RuntimeError(f"Frequency residual {residual} /m exceeds .05 /m")
            rx, ry = model.radiation_fields(field)
            mode = LinearMode(f"seed_{rank}", float(value.real), float(value.imag), field,
                0., rx, ry, discretization="staggered")
            axis, fx, fy, beam = direct_vector_far_field(mode, root, d.length_um, 201)
            budget = problem.loss_budget(field)
            record = {"status":"completed", "seed_rank":rank, "seed_pole_id":row["pole_id"],
                "seed_g_th_cm1":row["g_th_cm1"], "wavelength_nm":root,
                "detuning_per_m":float(value.real), "alpha_field_cm1":float(value.imag)/100,
                "active_Gamma":model.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME],
                "effective_index":model.effective_index,
                **threshold(float(value.imag), model.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME],
                            model.effective_index, d.length_um),
                "frequency_residual_per_m":residual, "reference_field_overlap":overlap,
                "eigen_residual":relative_residual(problem.center_operator(),field,value),
                "beam":beam, "beam_acceptance":beam_acceptance(beam, require_sampling=True),
                "budget":budget, "radiative_extraction_fraction":2*budget["local_amplitude_loss_per_m"]/(2*value.imag+100*BASE.INTERNAL_LOSS_CM),
                "frequency_trace":trace, "elapsed_seconds":time.monotonic()-started}
            arrays.update({f"seed_{rank}_field":field, f"seed_{rank}_far_x":fx,
                f"seed_{rank}_far_y":fy, f"seed_{rank}_radiation_x":rx,
                f"seed_{rank}_radiation_y":ry, f"seed_{rank}_coupling_m":model.coupling_m})
            arrays["axis_deg"] = axis
            print(f"N{grid} root {rank+1}/{roots}: {root:.6f} nm, g={record['g_th_cm1']:.6f}, ellipse={beam['ellipticity']:.5f}, overlap={overlap:.6f}",flush=True)
        except Exception as error:
            record = {"status":"failed", "seed_rank":rank, "error":str(error), "frequency_trace":trace}
            print(f"N{grid} root {rank+1}: FAILED {error}", flush=True)
        records.append(record)
        save_json(folder/"frequency_closed_poles.json", {"design":asdict(d), "grid":grid,
            "D":order, "dz_nm":dz, "reference_wavelength_nm":wavelength0,
            "records":records, "scope":"independent real roots of lowest sampled poles; not exhaustive spectrum"})
    saved.close()
    np.savez_compressed(folder/"frequency_closed_poles.npz", **arrays)
    ordered = sorted((r for r in records if r["status"]=="completed"), key=lambda r:r["g_th_cm1"])
    summary = {"design":asdict(d), "grid":grid, "D":order, "dz_nm":dz,
        "reference_wavelength_nm":wavelength0, "sampled_distinct_poles":spectrum["distinct_poles"],
        "closed_successfully":len(ordered), "failed_roots":sum(r["status"]!="completed" for r in records),
        "lowest":ordered[0] if ordered else None, "second":ordered[1] if len(ordered)>1 else None,
        "relative_material_gain_gap":ordered[1]["g_th_cm1"]/ordered[0]["g_th_cm1"]-1 if len(ordered)>1 else None,
        "fabrication":fabrication(d, cell),
        "scope":"lowest frequency-closed state among sampled lateral poles; scalar four-wave CWT, not Maxwell or nonlinear lasing proof"}
    save_json(folder/"closed_summary.json", summary)
    if ordered:
        fig, axes = plt.subplots(min(4,len(ordered)), 3, figsize=(11,3.1*min(4,len(ordered))), squeeze=False, constrained_layout=True)
        for row, record in enumerate(ordered[:4]):
            rank = record["seed_rank"]
            field = arrays[f"seed_{rank}_field"]
            power = abs(arrays[f"seed_{rank}_far_x"])**2+abs(arrays[f"seed_{rank}_far_y"])**2
            power /= power.max()
            axes[row,0].imshow(np.sum(abs(field)**2,axis=0),origin="lower",extent=(0,d.length_um,0,d.length_um))
            axes[row,0].set(title=f"Rank {row+1}; g={record['g_th_cm1']:.2f} /cm",xlabel="x (um)",ylabel="y (um)")
            axes[row,1].imshow(power,origin="lower",extent=(-1,1,-1,1),cmap="hot",vmin=0,vmax=1)
            axes[row,1].set(title=f"lambda={record['wavelength_nm']:.3f} nm; ellipse={record['beam']['ellipticity']:.3f}",xlabel="theta x (deg)",ylabel="theta y (deg)")
            centre = len(axis)//2
            axes[row,2].plot(axis,power[centre],label="x cut")
            axes[row,2].plot(axis,power[:,centre],label="y cut")
            axes[row,2].set(xlabel="theta (deg)",ylabel="normalized intensity",yscale="log",ylim=(1e-5,1.1))
            axes[row,2].legend()
        fig.suptitle(f"Actual low sampled poles; N={grid}, D={order}, dz={dz} nm; frequency closed")
        fig.savefig(folder/"closed_low_poles.png",dpi=170)
        plt.close(fig)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("--design-index",type=int,default=0,
                        help="Index into a frozen finalists configuration (default: 0)")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--grids",nargs="+",type=int,default=[33,49,65])
    parser.add_argument("--D",type=int,default=10)
    parser.add_argument("--dz",type=float,default=2.)
    parser.add_argument("--count",type=int,default=32)
    parser.add_argument("--roots",type=int,default=12)
    parser.add_argument("--length",type=float)
    parser.add_argument("--fixed-wavelength",type=float)
    parser.add_argument("--wide-targets",type=int,default=0)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    source=json.loads(args.source.read_text(encoding="utf8"))
    if isinstance(source,dict) and "finalists" in source:
        record=source["finalists"][args.design_index]["design"]
    else:
        record=source[args.design_index] if isinstance(source,list) else source.get("design",source)
    d=Design(**record)
    if args.length is not None:
        from dataclasses import replace
        d=replace(d,length_um=args.length,tag=d.tag+f"_L{args.length:g}")
    if args.dry_run:
        print(json.dumps({"design":asdict(d),"D":args.D,"dz_nm":args.dz,
            "grids":args.grids,"roots":args.roots,"wide_targets":args.wide_targets,
            "fabrication":fabrication(d,build_cell(d))},indent=2))
        return
    output=prepare_output(args.output,ROOT)
    write_provenance(output,ROOT)
    save_json(output/"run_plan.json",vars(args)|{"design":asdict(d)})
    save_json(output/"run_status.json",{"status":"running"})
    summaries=[]
    for grid in args.grids:
        folder=output/f"N{grid}"
        folder.mkdir()
        spectrum=survey(d,folder,args.D,args.dz,grid,args.count,args.fixed_wavelength,args.wide_targets)
        summaries.append(close_survey(d,folder,spectrum,order=args.D,dz=args.dz,roots=args.roots))
        save_json(output/"mesh_summaries.json",summaries)
    save_json(output/"run_status.json",{"status":"completed","grids":args.grids,
        "closed_roots":sum(r["closed_successfully"] for r in summaries)})


if __name__=="__main__":
    main()
