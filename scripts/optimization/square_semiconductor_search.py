"""Reproducible square-lattice geometry/vertical/finite-device search.

This is a separate research entry. Existing curated presets are never edited.
All candidates use the current real-interface vertical quadrature, corrected
three-barrier stack and staggered finite-area eigenproblem. A low-resolution
screen is explicitly distinct from the D8/D10 and mesh checks. No dynamic or
global single-mode claim follows from the four retained Gamma families.
"""
from __future__ import annotations

import os
for _thread_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_name] = "1"

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from pcselsim.custom_analysis import (FourWaveOpticalSpec, MODE_NAMES,
    finite_area_eigenproblem, solve_finite_modes, solve_finite_modes_converged)
from pcselsim.design_objectives import beam_acceptance, beam_metrics
from pcselsim.fabrication import (FabricationRules, evaluate_fabrication,
    inclusion_boundary, pareto_mask)
from pcselsim.geometry import Ellipse, PolygonInclusion, SquareLatticeCell
from pcselsim.run_output import default_output, prepare_output, write_provenance
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from pcselsim.vertical import Layer, LayerStack
from scripts import run_square_double_hole as BASE

RESEARCH = FabricationRules("research_span40_gap50_nm", 50, 40, 3, 3, 6)
EASIER = FabricationRules("easier_span65_gap60_nm", 60, 65, 3, 3, 10)


def clean(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                                     allow_nan=False), encoding="utf-8")


@dataclass(frozen=True)
class Design:
    family: str
    fill: float
    share: float = .42
    dx: float = .45
    dy: float = .45
    aspect: float = 1.
    second_aspect: float = 1.
    angle: float = 0.
    second_angle: float = 0.
    exponent: float = 2.
    pc_nm: float = 190.
    gaas_nm: float = 110.
    qw_spacer_nm: float = 25.
    length_um: float = 200.
    lattice_nm: float = 277.
    tag: str = "search"

    @property
    def identifier(self):
        text = json.dumps(asdict(self), sort_keys=True)
        return self.family + "_" + hashlib.sha256(text.encode()).hexdigest()[:12]


def shape(center, fill, aspect, angle, exponent=2., vertices=40):
    if abs(exponent - 2.) < 1e-10:
        radius = np.sqrt(fill / np.pi)
        return Ellipse(tuple(center), (float(radius*np.sqrt(aspect)),
                                      float(radius/np.sqrt(aspect))), float(angle))
    phase = 2*np.pi*np.arange(vertices)/vertices
    points = np.column_stack((np.sign(np.cos(phase))*abs(np.cos(phase))**(2/exponent),
                              np.sign(np.sin(phase))*abs(np.sin(phase))**(2/exponent)))
    points *= np.array([np.sqrt(aspect), 1/np.sqrt(aspect)])
    next_points=np.roll(points,-1,axis=0)
    area = .5*abs(np.sum(points[:,0]*next_points[:,1]-points[:,1]*next_points[:,0]))
    points *= np.sqrt(fill/area)
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points = points@rotation.T+center
    return PolygonInclusion(tuple(map(tuple, points)))


def build_cell(d: Design, vertices=40):
    if d.family == "inoue_reference":
        holes = (Ellipse((-.125, -.125), (.12, .075), -35.),
                 Ellipse((.125, .125), (.095, .060), -35.))
    elif d.family.startswith("single"):
        holes = (shape((0, 0), d.fill, d.aspect, d.angle, d.exponent, vertices),)
    elif d.family.startswith("double"):
        holes = (shape((-d.dx/2, -d.dy/2), d.fill*d.share, d.aspect, d.angle, d.exponent, vertices),
                 shape((d.dx/2, d.dy/2), d.fill*(1-d.share), d.second_aspect,
                       d.second_angle, d.exponent, vertices))
    elif d.family == "triple":
        centers = ((-d.dx/2, -d.dy/2), (d.dx/2, -d.dy/2), (0, d.dy/2))
        weights = (d.share, (1-d.share)/2, (1-d.share)/2)
        holes = tuple(shape(c, d.fill*w, d.aspect if i == 0 else d.second_aspect,
                            d.angle if i == 0 else d.second_angle) for i, (c, w) in enumerate(zip(centers, weights)))
    elif d.family == "quad":
        centers = ((-d.dx/2, -d.dy/2), (d.dx/2, -d.dy/2),
                   (d.dx/2, d.dy/2), (-d.dx/2, d.dy/2))
        weights = (d.share/2, (1-d.share)/2, d.share/2, (1-d.share)/2)
        holes = tuple(shape(c, d.fill*w, d.aspect if i%2 == 0 else d.second_aspect,
                            d.angle if i%2 == 0 else d.second_angle) for i, (c,w) in enumerate(zip(centers, weights)))
    else:
        raise ValueError("Unknown geometry family " + d.family)
    return SquareLatticeCell(BASE.BACKGROUND_INDEX**2, tuple(holes))


def build_stack(d, cell):
    average = float(np.sqrt(cell.fourier_epsilon(0, 0).real))
    layers = []
    for layer in BASE.LAYERS:
        thickness = {BASE.PC_LAYER_NAME: d.pc_nm, "GaAs": d.gaas_nm,
                     "AlGaAs spacer": d.qw_spacer_nm}.get(layer.name, layer.thickness_nm)
        index = average if layer.name == BASE.PC_LAYER_NAME else layer.refractive_index
        layers.append(Layer(layer.name, thickness, index))
    return LayerStack(tuple(layers), BASE.TOP_CLADDING_INDEX,
                      BASE.BOTTOM_CLADDING_INDEX, BASE.VERTICAL_PADDING_UM)


def fabrication(d, cell):
    return {"research": evaluate_fabrication(cell.inclusions, d.lattice_nm, rules=RESEARCH),
            "easier": evaluate_fabrication(cell.inclusions, d.lattice_nm, rules=EASIER)}


def broad_designs(count=192, seed=21026):
    # Exact geometry seed and all accepted/rejected proposals are persisted.
    references = [Design("double", .14, length_um=300, tag="current_main_reference"),
                  Design("inoue_reference", .046, length_um=300, tag="paper_process_fail_reference")]
    families = ("single", "double", "double", "double", "double_rounded", "triple", "quad", "single_rounded")
    proposals = qmc.LatinHypercube(12, seed=seed).random(max(count*24, 2048))
    accepted, rejected = list(references), []
    for index, p in enumerate(proposals):
        family = families[index % len(families)]
        fill = (.055+.15*p[0]) if family in {"triple", "quad"} else .065+.17*p[0]
        aspect = 1+(.65 if family != "quad" else .30)*p[3]
        d = Design(family, fill, .28+.35*p[1], .32+.19*p[2], .32+.19*p[4],
                   aspect, 1+.60*p[5], -90+180*p[6], -90+180*p[7],
                   2.6+1.6*p[8] if "rounded" in family else 2.,
                   120+140*p[9], 65+115*p[10], 10+40*p[11], 200.,
                   tag=f"broad_seed{seed}_{index}")
        audit = fabrication(d, build_cell(d))
        if not audit["research"]["margin_pass"]:
            rejected.append({"design": asdict(d), "fabrication": audit})
            continue
        accepted.append(d)
        if len(accepted) >= count+len(references):
            break
    if len(accepted) < count+len(references):
        raise RuntimeError("Not enough feasible motifs; increase proposal pool")
    return accepted, rejected


def direct_vector_far_field(mode, wavelength_nm, length_um, samples=129):
    n = mode.fields.shape[-1]
    axis = np.linspace(-1., 1., samples)
    positions = ((np.arange(n)+.5)/n-.5)*length_um*1e-6
    frequency = np.tan(np.deg2rad(axis))/(wavelength_nm*1e-9)
    transform = np.exp(-2j*np.pi*frequency[:, None]*positions[None, :])
    xx, yy = np.meshgrid(np.deg2rad(axis), np.deg2rad(axis))
    obliquity = np.cos(xx)+np.cos(yy)-1
    fx = (transform@mode.radiation_x@transform.T)*obliquity
    fy = (transform@mode.radiation_y@transform.T)*obliquity
    nyquist = float(np.rad2deg(np.arctan(wavelength_nm*1e-9/(2*length_um*1e-6/n))))
    metrics = beam_metrics(axis, fx, fy, requested_view_deg=1., source_nyquist_deg=nyquist)
    return axis, fx, fy, metrics


def threshold(alpha_per_m, gamma, neff, length_um):
    overlap = gamma*BASE.ACTIVE_INDEX/neff
    material_cm = (2*alpha_per_m/100+BASE.INTERNAL_LOSS_CM)/overlap
    if material_cm >= BASE.MAXIMUM_GAIN_CM:
        return {"g_th_cm1": material_cm, "threshold_carrier_cm3": None,
                "J_th_A_cm2": None, "I_th_A": None, "gain_law_reachable": False}
    maximum, negative_zero = BASE.MAXIMUM_GAIN_CM, -BASE.ZERO_CARRIER_GAIN_CM
    carrier = BASE.TRANSPARENCY_DENSITY_CM3*(maximum+material_cm*maximum/negative_zero)/(maximum-material_cm)
    current_density = 1.602176634e-19*(BASE.ACTIVE_THICKNESS_NM*1e-7)*carrier/(BASE.CARRIER_LIFETIME_NS*1e-9)
    return {"g_th_cm1": material_cm, "threshold_carrier_cm3": carrier,
            "J_th_A_cm2": current_density, "I_th_A": current_density*(length_um*1e-4)**2,
            "gain_law_reachable": True}


def minimum_grid(length_um, wavelength_nm=900., minimum=9):
    count = max(minimum, int(np.ceil(2*length_um*1e-6*np.tan(np.deg2rad(1.))/(wavelength_nm*1e-9)))+1)
    return count if count % 2 else count+1


def solve_design(d, output, order=4, dz=8., grids=None, samples=129, vertices=40,
                 eigensolutions=12, make_figure=False, fixed_wavelength_nm=None):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    cell = build_cell(d, vertices)
    fab = fabrication(d, cell)
    model = build_geometry_coupling(cell, build_stack(d, cell), BASE.PC_LAYER_NAME,
        d.lattice_nm, BASE.WAVELENGTH_GUESS_NM, ThreeDCWTSettings(order, dz),
        fixed_wavelength_nm=fixed_wavelength_nm)
    minimum = minimum_grid(d.length_um, model.bragg_wavelength_nm)
    grid_list = tuple(grids or (minimum,))
    if grid_list[0] < minimum:
        raise ValueError(f"N={grid_list[0]} cannot support +/-1 degree for this aperture")
    spec = FourWaveOpticalSpec(model.bragg_wavelength_nm, model.effective_index,
        model.group_index, d.length_um, BASE.INTERNAL_LOSS_CM,
        tuple(model.eigenvalues_m.real/100), tuple(model.eigenvalues_m.imag/100),
        grid_list[-1], "staggered", eigensolutions, None, 1000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if len(grid_list) >= 3:
            modes, convergence = solve_finite_modes_converged(spec, grid_list,
                model.coupling_m, model.eigenvectors, model.radiation_fields)
        else:
            modes = solve_finite_modes(spec, model.coupling_m, model.eigenvectors, model.radiation_fields)
            convergence = {}
    problem = finite_area_eigenproblem(spec, model.coupling_m)
    gamma = model.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME]
    rows, arrays, figures = [], {}, {}
    for name, mode in modes.items():
        budget = problem.loss_budget(mode.fields)
        outcoupling = 2*budget["local_amplitude_loss_per_m"]/(2*mode.alpha_per_m+100*BASE.INTERNAL_LOSS_CM)
        axis, fx, fy, metrics = direct_vector_far_field(mode, spec.wavelength_nm, d.length_um, samples)
        row = {"mode": name, "alpha_field_cm1": mode.alpha_per_m/100,
            "detuning_per_m": mode.delta_per_m, "band_overlap": mode.band_overlap,
            "eigen_residual": mode.eigen_residual, "budget": budget,
            "radiative_extraction_fraction": outcoupling,
            "grid_sensitivity_cm1": None if mode.extrapolation_uncertainty_per_m is None else mode.extrapolation_uncertainty_per_m/100,
            "grid_status": mode.extrapolation_status,
            **threshold(mode.alpha_per_m, gamma, model.effective_index, d.length_um),
            "beam": metrics, "beam_acceptance": beam_acceptance(metrics, require_sampling=True)}
        rows.append(row)
        arrays.update({f"{name}_field": mode.fields, f"{name}_radiation_x": mode.radiation_x,
            f"{name}_radiation_y": mode.radiation_y, f"{name}_far_x": fx, f"{name}_far_y": fy})
        figures[name] = (mode, axis, fx, fy)
    ranked = sorted(rows, key=lambda row: row["alpha_field_cm1"])
    first, second = ranked[:2]
    combined = None if first["grid_sensitivity_cm1"] is None else first["grid_sensitivity_cm1"]+second["grid_sensitivity_cm1"]
    result = {"status": "completed", "id": d.identifier, "design": asdict(d),
        "settings": {"D": order, "dz_nm": dz, "grids": grid_list, "angular_samples": samples,
                     "polygon_vertices": vertices, "eigenpairs_per_target": eigensolutions,
                     "fixed_wavelength_nm":fixed_wavelength_nm},
        "fabrication": fab, "wavelength_nm": model.bragg_wavelength_nm,
        "effective_index": model.effective_index, "group_index": model.group_index,
        "active_Gamma": gamma, "pc_Gamma": model.pc_confinement,
        "passivity_correction_per_m": model.passivity_correction_m,
        "modes": rows, "lowest": first, "second": second["mode"],
        "family_gap_cm1": second["alpha_field_cm1"]-first["alpha_field_cm1"],
        "material_gain_relative_gap": second["g_th_cm1"]/first["g_th_cm1"]-1,
        "combined_grid_sensitivity_cm1": combined,
        "family_order_grid_separated": bool(combined is not None and
            second["alpha_field_cm1"]-first["alpha_field_cm1"] > combined),
        "useful_radiation": bool(first["radiative_extraction_fraction"] >= .01),
        "balanced_radiation": bool(first["radiative_extraction_fraction"] >= .05),
        "warnings": [str(w.message) for w in caught],
        "elapsed_seconds": time.monotonic()-started,
        "scope": "one selected state per four Gamma-connected families; no full spectrum, dynamic stability or full-wave certification"}
    arrays.update(coupling_m=model.coupling_m, c1d_m=model.c1d_m, c2d_m=model.c2d_m,
        crad_m=model.crad_m, axis_deg=axis, vertical_z_um=model.vertical_mode.z_um,
        vertical_field=model.vertical_mode.field, vertical_index=model.vertical_mode.index,
        **{f"convergence_{k}": v for k,v in convergence.items()})
    np.savez_compressed(output/"raw_fields.npz", **arrays)
    save_json(output/"result.json", result)
    if make_figure:
        plot_design(d, cell, result, figures, output/"all_four_modes.png")
    return result


def plot_design(d, cell, result, figures, path):
    fig, axes = plt.subplots(4, 3, figsize=(12, 14), constrained_layout=True)
    for row, (name, (mode, axis, fx, fy)) in enumerate(figures.items()):
        info = next(m for m in result["modes"] if m["mode"] == name)
        for hole in cell.inclusions:
            boundary = inclusion_boundary(hole, np.eye(2))
            axes[row,0].fill(boundary[:,0], boundary[:,1], color="white", ec="black")
        axes[row,0].set(xlim=(-.6,.6), ylim=(-.6,.6), aspect="equal", title=name+" unit cell")
        axes[row,1].imshow(mode.intensity, origin="lower", extent=(0,d.length_um,0,d.length_um))
        axes[row,1].set(title=f"{name}: field loss {info['alpha_field_cm1']:.4g} cm⁻¹", xlabel="x (um)",ylabel="y (um)")
        power = abs(fx)**2+abs(fy)**2; power /= power.max()
        axes[row,2].imshow(power, origin="lower", extent=(axis[0],axis[-1],axis[0],axis[-1]), cmap="hot")
        axes[row,2].set(title=f"gth={info['g_th_cm1']:.1f} cm⁻¹; ellipse={info['beam']['ellipticity']:.3f}", xlabel="theta x (deg)", ylabel="theta y (deg)")
    fig.suptitle(d.identifier+f"; actual lowest {result['lowest']['mode']}; +/-1 degree, vector FFP")
    fig.savefig(path, dpi=170); plt.close(fig)


def summarize(results, output):
    completed = [r for r in results if r.get("status") == "completed"]
    rows = []
    for r in completed:
        m=r["lowest"]; b=m["beam"]; d=r["design"]
        rows.append({"id": r["id"], "family":d["family"], "lowest_mode":m["mode"],
            "g_th_cm1":m["g_th_cm1"], "J_th_A_cm2":m["J_th_A_cm2"], "I_th_A":m["I_th_A"],
            "relative_gain_gap":r["material_gain_relative_gap"], "family_gap_cm1":r["family_gap_cm1"],
            "grid_sensitivity_cm1":r["combined_grid_sensitivity_cm1"],
            "ellipticity":b["ellipticity"], "center_to_peak":b["center_to_peak"],
            "radial_anisotropy_l2":b["radial_anisotropy_l2"],
            "beam_accepted":m["beam_acceptance"]["accepted"],
            "radiative_fraction":m["radiative_extraction_fraction"],
            "research_process_pass":r["fabrication"]["research"]["margin_pass"],
            "easier_process_pass":r["fabrication"]["easier"]["margin_pass"],
            "pc_nm":d["pc_nm"], "gaas_nm":d["gaas_nm"], "length_um":d["length_um"],
            "Gamma":r["active_Gamma"], "wavelength_nm":r["wavelength_nm"]})
    if rows:
        objectives=np.array([[r["g_th_cm1"],r["ellipticity"]-1,
            r["radial_anisotropy_l2"],-r["relative_gain_gap"]] for r in rows])
        feasible=np.array([r["research_process_pass"] and r["radiative_fraction"]>=.01 for r in rows])
        pareto=np.zeros(len(rows),bool)
        if feasible.any(): pareto[feasible]=pareto_mask(objectives[feasible])
        for row, flag in zip(rows,pareto): row["pareto"]=bool(flag)
        with (output/"summary.csv").open("w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        save_json(output/"pareto.json",[r for r,flag in zip(completed,pareto) if flag])
        fig,ax=plt.subplots(figsize=(8,5), constrained_layout=True)
        scatter=ax.scatter([r["g_th_cm1"] for r in rows],[r["ellipticity"] for r in rows],
            c=[r["relative_gain_gap"] for r in rows],cmap="viridis",s=25)
        ax.axhline(1.1,color="grey",ls="--");ax.set(xlabel="Material threshold gain (cm⁻¹)",ylabel="Moment ellipticity",ylim=(1,min(3,max(r["ellipticity"] for r in rows))))
        fig.colorbar(scatter,ax=ax,label="Relative material-threshold gap, retained families")
        fig.savefig(output/"threshold_roundness.png",dpi=170);plt.close(fig)
    save_json(output/"all_results.json",results)
    return rows


def select_seeds(source, count=6):
    results=json.loads((source/"all_results.json").read_text(encoding="utf-8"))
    valid=[r for r in results if r["status"]=="completed" and r["useful_radiation"]
           and r["fabrication"]["research"]["margin_pass"]]
    def score(r):
        m=r["lowest"];b=m["beam"]
        return (m["g_th_cm1"]*(1+4*max(b["ellipticity"]-1.1,0)
            +2*max(b["radial_anisotropy_l2"]-.15,0)+3*max(.9-b["center_to_peak"],0))
            / (1+min(r["material_gain_relative_gap"],.5)))
    ordered=sorted(valid,key=score)
    # Preserve a process-easier alternative and rounded-contour alternatives,
    # rather than spending every refinement on one tiny process margin.
    easy=[r for r in ordered if r["fabrication"]["easier"]["margin_pass"]
          and r["lowest"]["beam_acceptance"]["accepted"]
          and r["material_gain_relative_gap"]>=.1]
    rounded=[r for r in ordered if "rounded" in r["design"]["family"]
             and r["lowest"]["beam_acceptance"]["accepted"]]
    priority=ordered[:max(1,count-3)]+easy[:2]+rounded[:1]+ordered
    selected=[];seen=set()
    for r in priority:
        if r["id"] not in seen:
            selected.append(r);seen.add(r["id"])
        if len(selected)>=count:break
    return selected


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("broad","local","verify"),default="broad")
    parser.add_argument("--source",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--count",type=int,default=192)
    parser.add_argument("--seed",type=int,default=21026)
    parser.add_argument("--seeds",type=int,default=6)
    parser.add_argument("--D",type=int,default=None)
    parser.add_argument("--dz",type=float,default=None)
    parser.add_argument("--grids",nargs="+",type=int)
    parser.add_argument("--id",nargs="*")
    parser.add_argument("--designs-json",type=Path,help="explicit frozen Design records; bypass seed selection")
    args=parser.parse_args(argv)
    output=prepare_output(args.output or default_output(ROOT,"optimization_20261021/square/"+args.stage),ROOT)
    write_provenance(output,ROOT)
    save_json(output/"run_plan.json",vars(args)|{"source":str(args.source),"output":str(output),
        "threads":{k:os.environ[k] for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
        "objectives":"material gain, retained-family gap, true minimum-family beam and finite radiation; no global/dynamic single-mode proof"})
    save_json(output/"run_status.json",{"status":"running","started_at":datetime.now().isoformat()})
    if args.designs_json is not None:
        designs=[Design(**record) for record in json.loads(args.designs_json.read_text(encoding="utf-8"))]
    elif args.stage=="broad":
        designs,rejected=broad_designs(args.count,args.seed)
        save_json(output/"fabrication_rejections.json",rejected)
    else:
        if args.source is None: raise ValueError("--source is required")
        seeds=select_seeds(args.source,args.seeds)
        if args.id:seeds=[r for r in json.loads((args.source/"all_results.json").read_text(encoding="utf-8")) if r.get("id") in args.id]
        designs=[]
        for r in seeds:
            d=Design(**r["design"])
            if args.stage=="verify":designs.append(d);continue
            designs.append(replace(d,tag=d.tag+"_local_center"))
            for key,changes in (("fill",(-.015,.015)),("share",(-.05,.05)),("dx",(-.035,.035)),
                ("dy",(-.035,.035)),("aspect",(-.08,.08)),("second_aspect",(-.08,.08)),
                ("angle",(-12.,12.)),("second_angle",(-12.,12.)),("pc_nm",(-20.,20.)),
                ("gaas_nm",(-20.,20.)),("qw_spacer_nm",(-8.,8.)),("length_um",(150.,250.,300.,400.))):
                for change in changes:
                    value=change if key=="length_um" else getattr(d,key)+change
                    if key in {"aspect","second_aspect"}:value=max(1.,value)
                    designs.append(replace(d,**{key:value,"tag":d.tag+"_local_"+key+str(change)}))
        designs=list({d.identifier:d for d in designs}.values())
    save_json(output/"designs.json",[asdict(d) for d in designs])
    results=[]
    for index,d in enumerate(designs):
        case=output/f"case_{index:04d}_{d.identifier}"
        order=args.D or (10 if args.stage=="verify" else 4)
        dz=args.dz or (3. if args.stage=="verify" else 8.)
        grids=tuple(args.grids) if args.grids else None
        if args.stage=="verify" and grids is None:
            n=minimum_grid(d.length_um,minimum=17);grids=(n,n+8,n+16)
        try:
            result=solve_design(d,case,order,dz,grids,161 if args.stage=="verify" else 129,
                64 if args.stage=="verify" else 40,20 if args.stage=="verify" else 12,
                make_figure=args.stage=="verify")
        except Exception as error:
            result={"status":"failed","id":d.identifier,"design":asdict(d),
                    "error_type":type(error).__name__,"error":str(error),"traceback":traceback.format_exc()}
            case.mkdir(parents=True,exist_ok=True);save_json(case/"failure.json",result)
        results.append(result)
        print(f"{index+1}/{len(designs)} {d.identifier}: "+
            (f"g={result['lowest']['g_th_cm1']:.1f}, ellipse={result['lowest']['beam']['ellipticity']:.3f}, gap={result['material_gain_relative_gap']:.3%}" if result['status']=="completed" else result['error']),flush=True)
        if (index+1)%10==0:summarize(results,output)
    rows=summarize(results,output)
    save_json(output/"run_status.json",{"status":"completed","finished_at":datetime.now().isoformat(),
        "case_count":len(results),"completed":len(rows),"failed":len(results)-len(rows),
        "beam_accepted":sum(r["beam_accepted"] for r in rows),
        "scope":"Execution success is not physical validation or global optimization"})


if __name__=="__main__":main()
