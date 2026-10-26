"""Optional 950-nm square design and independent fixed-period frequency roots.

The period is a design variable only during the first root. It is then frozen;
each retained branch re-solves C(lambda), vertical TE0 and finite fields at its
own real frequency. This still uses the stated reflection-free scalar CWT and
nominal nondispersive material indices/gain parameters.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scipy.optimize import brentq
from pcselsim.custom_analysis import FourWaveOpticalSpec,solve_finite_modes,finite_area_eigenproblem
from pcselsim.design_objectives import beam_acceptance
from pcselsim.frequency_cwt import background_detuning,solve_frequency_consistent_modes
from pcselsim.run_output import default_output,prepare_output,write_provenance
from pcselsim.three_d_cwt import ThreeDCWTSettings,build_geometry_coupling
from scripts.optimization.square_semiconductor_search import (
    BASE,Design,build_cell,build_stack,save_json,threshold,direct_vector_far_field,
    fabrication,solve_design)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--target",type=float,default=950.)
    parser.add_argument("--length",type=float)
    parser.add_argument("--D",type=int,default=10)
    parser.add_argument("--dz",type=float,default=2.)
    parser.add_argument("--grid",type=int,default=25)
    parser.add_argument("--branch",default="D")
    args=parser.parse_args(argv)
    source=json.loads(args.case.read_text(encoding="utf-8"));base=Design(**source["design"])
    if args.length is not None:base=replace(base,length_um=args.length)
    output=prepare_output(args.output or default_output(ROOT,"optimization_20261021/square/target950"),ROOT)
    write_provenance(output,ROOT)
    settings=ThreeDCWTSettings(args.D,args.dz)
    trace=[]
    def period_equation(period):
        d=replace(base,lattice_nm=float(period));cell=build_cell(d,64);stack=build_stack(d,cell)
        model=build_geometry_coupling(cell,stack,BASE.PC_LAYER_NAME,period,args.target,
                                      settings,fixed_wavelength_nm=args.target)
        spec=FourWaveOpticalSpec(args.target,model.effective_index,model.group_index,d.length_um,
            BASE.INTERNAL_LOSS_CM,tuple(model.eigenvalues_m.real/100),
            tuple(model.eigenvalues_m.imag/100),args.grid,"staggered",24,None,1000)
        modes=solve_finite_modes(spec,model.coupling_m,model.eigenvectors,model.radiation_fields)
        mode=modes[args.branch]
        gap=background_detuning(args.target,model.effective_index,period)-mode.delta_per_m
        trace.append({"period_nm":period,"branch":args.branch,"residual_per_m":gap,
                      "lowest_family_at_target":min(modes,key=lambda n:modes[n].alpha_per_m)})
        print(f"period design a={period:.8f} nm, residual={gap:.5g}/m",flush=True)
        return gap
    period=float(brentq(period_equation,base.lattice_nm*.94,base.lattice_nm*1.12,xtol=2e-8))
    period_residual=period_equation(period)
    d=replace(base,lattice_nm=period,tag=base.tag+f"_target{args.target:g}nm")
    cell=build_cell(d,64);stack=build_stack(d,cell)
    def factory(wavelength):
        model=build_geometry_coupling(cell,stack,BASE.PC_LAYER_NAME,period,args.target,
            settings,fixed_wavelength_nm=wavelength)
        return SimpleNamespace(result=model,optical_theorem_residual=None)
    branches=solve_frequency_consistent_modes(cell,stack,None,BASE.PC_LAYER_NAME,period,args.target,
        settings,d.length_um,BASE.INTERNAL_LOSS_CM,args.grid,
        wavelength_interval_nm=(args.target-45,args.target+45),optical_factory=factory,
        discretization="staggered",reference_dense_limit=0)
    records=[];arrays={}
    for name,branch in branches.items():
        model=branch.optical.result;mode=branch.finite_mode
        problem=finite_area_eigenproblem(branch.spec,model.coupling_m)
        budget=problem.loss_budget(mode.fields)
        gamma=model.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME]
        axis,fx,fy,beam=direct_vector_far_field(mode,branch.wavelength_nm,d.length_um,201)
        records.append({"mode":name,"wavelength_nm":branch.wavelength_nm,
            "alpha_field_cm1":mode.alpha_per_m/100,"Gamma":gamma,
            **threshold(mode.alpha_per_m,gamma,model.effective_index,d.length_um),
            "eigen_residual":mode.eigen_residual,"frequency_residual_per_m":branch.frequency_residual_per_m,
            "reference_overlap":branch.reference_overlap,"beam":beam,
            "beam_acceptance":beam_acceptance(beam,require_sampling=True),"budget":budget,
            "radiative_extraction_fraction":2*budget["local_amplitude_loss_per_m"]/(2*mode.alpha_per_m+100*BASE.INTERNAL_LOSS_CM),
            "frequency_trace":branch.evaluations})
        arrays.update({name+"_fields":mode.fields,name+"_far_x":fx,name+"_far_y":fy,
            name+"_radiation_x":mode.radiation_x,name+"_radiation_y":mode.radiation_y,
            name+"_coupling_m":model.coupling_m})
        print(f"closed {name}: lambda={branch.wavelength_nm:.8f}nm, g={records[-1]['g_th_cm1']:.5f}",flush=True)
    ordered=sorted(records,key=lambda x:x["g_th_cm1"])
    save_json(output/"summary.json",{"status":"completed","design":asdict(d),
        "source_case":str(args.case),"D":args.D,"dz_nm":args.dz,"grid":args.grid,
        "period_design_root":trace,"period_design_residual_per_m":period_residual,
        "fabrication":fabrication(d,cell),"frequency_closed_modes":records,
        "lowest_material_threshold":ordered[0],
        "relative_material_gain_gap":ordered[1]["g_th_cm1"]/ordered[0]["g_th_cm1"]-1,
        "scope":"reflection-free scalar CWT; four independent retained real roots; nominal non-dispersive material indices and gain; lateral spectrum incomplete"})
    np.savez_compressed(output/"frequency_closed_fields.npz",axis_deg=axis,**arrays)
    save_json(output/"tuned_design_manifest.json",[asdict(d)])
    save_json(output/"run_status.json",{"status":"completed","period_nm":period,
        "lowest_mode":ordered[0]["mode"],"target_branch":args.branch})


if __name__=="__main__":main()
