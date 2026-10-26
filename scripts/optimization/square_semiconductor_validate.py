"""Additional lateral-pole and process tests of a chosen square design.

The lateral survey samples several independently shifted spectral clusters.
It is more extensive than four selected family states, but remains finite and
does not certify the entire open Maxwell spectrum or nonlinear stability.
"""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_name]="1"
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import json
import numpy as np

from pcselsim.custom_analysis import (FourWaveOpticalSpec, LinearMode,
    finite_area_eigenproblem)
from pcselsim.finite_discretization import relative_residual
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.run_output import default_output, prepare_output, write_provenance
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from scripts.optimization.square_semiconductor_search import (
    BASE, Design, build_cell, build_stack, clean, direct_vector_far_field,
    save_json, solve_design, threshold)


def independent_pole(value, field, values, fields, relative_tolerance=5e-9,
                     identical_overlap=.9999):
    """Compatible wrapper using span rank inside one numerical complex pole.

    The legacy overlap argument can tighten, but never loosen, the 1e-6 span
    residual. Pairwise overlap cannot identify rotations of a degenerate basis.
    """
    if not 0 < identical_overlap <= 1:
        raise ValueError("identical_overlap must lie in (0,1]")
    span_tolerance = min(1e-6, np.sqrt(1-identical_overlap)+64*np.finfo(float).eps)
    return independent_eigenmode(value,field,values,fields,
        eigenvalue_rtol=relative_tolerance,span_tolerance=span_tolerance)


def survey(d, output, order=10, dz=3., grid=33, count=32, fixed_wavelength_nm=None,
           wide_targets=0):
    cell=build_cell(d,64)
    model=build_geometry_coupling(cell,build_stack(d,cell),BASE.PC_LAYER_NAME,
        d.lattice_nm,BASE.WAVELENGTH_GUESS_NM,ThreeDCWTSettings(order,dz),
        fixed_wavelength_nm=fixed_wavelength_nm)
    spec=FourWaveOpticalSpec(model.bragg_wavelength_nm,model.effective_index,
        model.group_index,d.length_um,BASE.INTERNAL_LOSS_CM,
        tuple(model.eigenvalues_m.real/100),tuple(model.eigenvalues_m.imag/100),grid,
        "staggered",count,None,1200)
    problem=finite_area_eigenproblem(spec,model.coupling_m)
    operator=problem.center_operator()
    rng=np.random.default_rng(222617)
    v0=rng.normal(size=4*grid*grid)+1j*rng.normal(size=4*grid*grid)
    # +/- pi/L and +/-2pi/L include the first few distinct spatial oscillations.
    spacing=np.pi/(d.length_um*1e-6)
    targets=[complex(value.real+offset*spacing,max(value.imag,.1))
             for value in model.eigenvalues_m for offset in (-2,-1,0,1,2)]
    if wide_targets:
        # Independently cover the intervals between the four Gamma edges. The
        # extra targets sit close to the real axis, favouring low-loss poles.
        lower=float(model.eigenvalues_m.real.min()-3*spacing)
        upper=float(model.eigenvalues_m.real.max()+3*spacing)
        targets.extend(complex(value,.1) for value in np.linspace(lower,upper,wide_targets))
    values=[];fields=[];failures=[];target_records=[]
    started=time.monotonic()
    for index,target in enumerate(targets):
        before=len(values)
        try:
            poles,states=problem.near(target,count,v0,maxiter=1200)
            for value,field in zip(poles,states.T):
                if value.imag < -1e-7:continue
                field=field.reshape(4,grid,grid)
                if independent_pole(value,field,values,fields):
                    values.append(value);fields.append(field)
            target_records.append({"target":[target.real,target.imag],"new_poles":len(values)-before,
                                   "returned":len(poles)})
        except Exception as error:
            failures.append({"target":[target.real,target.imag],"error":str(error),"type":type(error).__name__})
        print(f"lateral cluster {index+1}/{len(targets)}: {len(values)} distinct poles",flush=True)
    ranking=np.argsort(np.imag(values))
    gamma=model.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME]
    modes=[];arrays={"coupling_m":model.coupling_m}
    for index in ranking:
        value=values[index];field=fields[index]
        residual=relative_residual(operator,field,value)
        budget=problem.loss_budget(field)
        rx,ry=model.radiation_fields(field)
        mode=LinearMode(f"pole_{index}",float(value.real),float(value.imag),field,0.,rx,ry,
                        discretization="staggered",eigen_residual=residual)
        row={"pole_id":int(index),"detuning_per_m":value.real,"alpha_field_cm1":value.imag/100,
            "eigen_residual":residual,"budget":budget,
            "radiative_extraction_fraction":2*budget["local_amplitude_loss_per_m"]/(2*value.imag+100*BASE.INTERNAL_LOSS_CM),
            **threshold(value.imag,gamma,model.effective_index,d.length_um)}
        if len(modes)<24:
            axis,fx,fy,metrics=direct_vector_far_field(mode,spec.wavelength_nm,d.length_um,161)
            row["beam"]=metrics
            arrays[f"pole_{index}_field"]=field
            arrays[f"pole_{index}_far_x"]=fx;arrays[f"pole_{index}_far_y"]=fy
            arrays["axis_deg"]=axis
        modes.append(row)
    result={"status":"completed","design":asdict(d),"D":order,"dz_nm":dz,"grid":grid,
        "fixed_wavelength_nm":fixed_wavelength_nm,"evaluated_wavelength_nm":model.bragg_wavelength_nm,
        "eigenpairs_per_target":count,"wide_real_axis_targets":wide_targets,
        "targets":target_records,"target_failures":failures,
        "distinct_poles":len(values),"ordered_poles":modes,"elapsed_seconds":time.monotonic()-started,
        "scope":f"{len(targets)} shift clusters about four band edges and optional real-axis coverage; finite sampled lateral spectrum, no exhaustive or dynamic single-mode proof"}
    np.savez_compressed(output/"lateral_poles.npz",**arrays)
    save_json(output/"lateral_survey.json",result)
    return result


def perturbed_designs(d):
    result=[]
    for bias in (-3.,3.):
        cell=build_cell(d)
        if d.family=="double" and all(hasattr(h,"radii") for h in cell.inclusions):
            radii=[np.array(h.radii)+bias/d.lattice_nm for h in cell.inclusions]
            fills=[np.pi*np.prod(r) for r in radii]
            result.append(replace(d,fill=sum(fills),share=fills[0]/sum(fills),
                aspect=radii[0][0]/radii[0][1],second_aspect=radii[1][0]/radii[1][1],
                tag=f"etch_bias_{bias:+g}nm"))
        result.append(replace(d,dx=d.dx+2*bias/d.lattice_nm,tag=f"opposed_x_shift_{bias:+g}nm_per_hole"))
        result.append(replace(d,dy=d.dy+2*bias/d.lattice_nm,tag=f"opposed_y_shift_{bias:+g}nm_per_hole"))
        result.append(replace(d,pc_nm=d.pc_nm+bias,tag=f"PC_depth_{bias:+g}nm"))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case",type=Path,help="result.json of a selected actual-grid case")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--D",type=int,default=10)
    parser.add_argument("--dz",type=float,default=3.)
    parser.add_argument("--grid",type=int,default=33)
    parser.add_argument("--count",type=int,default=32)
    parser.add_argument("--perturb",action="store_true")
    parser.add_argument("--fixed-wavelength",type=float)
    args=parser.parse_args(argv)
    source=json.loads(args.case.read_text(encoding="utf-8"));design=Design(**source["design"])
    output=prepare_output(args.output or default_output(ROOT,"optimization_20261021/square/lateral"),ROOT)
    write_provenance(output,ROOT)
    save_json(output/"run_plan.json",{"source_case":str(args.case),"design":asdict(design),
        "D":args.D,"dz_nm":args.dz,"grid":args.grid,"count_per_cluster":args.count,
        "perturb":args.perturb,"fixed_wavelength_nm":args.fixed_wavelength})
    save_json(output/"run_status.json",{"status":"running"})
    result=survey(design,output,args.D,args.dz,args.grid,args.count,args.fixed_wavelength)
    # Uniform internal loss changes every branch threshold algebraically. It
    # does not change these passive cold-cavity spatial eigenvectors.
    loss_cases=[]
    for mode in source.get("modes",source.get("frequency_closed_modes",[])):
        gamma=source.get("active_Gamma",mode.get("Gamma"))
        neff=source.get("effective_index",mode.get("frequency_trace",[{}])[-1].get("effective_index"))
        overlap=gamma*BASE.ACTIVE_INDEX/neff
        g=(2*mode["alpha_field_cm1"]+2*BASE.INTERNAL_LOSS_CM)/overlap
        base_g=mode["g_th_cm1"]
        loss_cases.append({"mode":mode["mode"],"base_g_th_cm1":base_g,
                           "double_internal_loss_g_th_cm1":g,"relative_increase":g/base_g-1})
    save_json(output/"double_internal_loss.json",{"internal_loss_cm1":[BASE.INTERNAL_LOSS_CM,2*BASE.INTERNAL_LOSS_CM],
        "modes":loss_cases,"scope":"uniform passive loss sensitivity; same cold fields, not thermal recalculation"})
    perturbations=[]
    if args.perturb:
        for index,changed in enumerate(perturbed_designs(design)):
            try:
                item=solve_design(changed,output/f"perturb_{index:02d}",args.D,args.dz,
                    (args.grid,),161,64,24,make_figure=True,fixed_wavelength_nm=args.fixed_wavelength)
            except Exception as error:
                item={"status":"failed","design":asdict(changed),"error":str(error)}
            perturbations.append(item)
            print(f"perturb {index+1}: {changed.tag} {item['status']}",flush=True)
        save_json(output/"perturbation_summary.json",perturbations)
    save_json(output/"run_status.json",{"status":"completed","distinct_poles":result["distinct_poles"],
        "failed_targets":len(result["target_failures"]),"perturb_cases":len(perturbations)})


if __name__=="__main__":main()
