"""Re-rank saved geometries with the complete coarse generalized QZ spectrum.

No band-edge target or hand-picked smooth envelope determines the winner.
All passive finite residual-verified poles are retained, without an arbitrary
detuning gate. The coarse full spectrum is still a screen, not convergence.
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from pcselsim.custom_analysis import FourWaveOpticalSpec,LinearMode,finite_area_eigenproblem
from pcselsim.design_objectives import beam_acceptance
from pcselsim.finite_discretization import relative_residual
from pcselsim.run_output import default_output,prepare_output,write_provenance
from scripts.optimization.square_semiconductor_search import (
    BASE,Design,save_json,threshold,direct_vector_far_field,summarize,minimum_grid)


def radiation_map(saved):
    fields=np.concatenate([saved[name+"_field"].reshape(4,-1) for name in ("A","B","C","D")],axis=1)
    maps=[];residuals=[]
    for component in ("x","y"):
        samples=np.concatenate([saved[name+"_radiation_"+component].ravel() for name in ("A","B","C","D")])
        row,_,rank,_=np.linalg.lstsq(fields.T,samples,rcond=1e-12)
        error=float(np.linalg.norm(row@fields-samples)/max(np.linalg.norm(samples),1e-300))
        if rank<4 or error>1e-9:
            raise RuntimeError(f"Saved four-wave radiation operator is underdetermined: rank{rank}, residual{error}")
        maps.append(row);residuals.append(error)
    return np.asarray(maps),residuals


def evaluate_case(source,case,output,grid=9):
    d=Design(**source["design"])
    with np.load(case/"raw_fields.npz",allow_pickle=False) as saved:
        coupling=saved["coupling_m"]
        radiation,residuals=radiation_map(saved)
    spec=FourWaveOpticalSpec(source["wavelength_nm"],source["effective_index"],source["group_index"],
        d.length_um,BASE.INTERNAL_LOSS_CM,(0.,)*4,(0.,)*4,grid,"staggered",12,4*grid*grid,1000)
    problem=finite_area_eigenproblem(spec,coupling)
    dimension=4*grid*grid
    problem.near(0j,4,np.ones(dimension,complex),dense_limit=dimension)
    values,fields=problem._dense_spectrum
    indices=np.argsort(values.imag)
    indices=[index for index in indices if values[index].imag>=-1e-7]
    rows=[];arrays={"eigenvalues_per_m":values,"radiation_map":radiation,"coupling_m":coupling}
    for rank,index in enumerate(indices[:16]):
        value=values[index];field=fields[:,index].reshape(4,grid,grid)
        rad=np.einsum("ac,cxy->axy",radiation,field)
        mode=LinearMode(f"global_{rank}",float(value.real),float(value.imag),field,0.,rad[0],rad[1],discretization="staggered")
        budget=problem.loss_budget(field)
        axis,fx,fy,beam=direct_vector_far_field(mode,spec.wavelength_nm,d.length_um,129)
        row={"mode":f"global_{rank}","alpha_field_cm1":value.imag/100,"detuning_per_m":value.real,
            "detuning_fraction_of_Bragg_beta":abs(value.real)/(2*np.pi/(d.lattice_nm*1e-9)),
            "eigen_residual":relative_residual(problem.center_operator(),field,value),
            **threshold(value.imag,source["active_Gamma"],source["effective_index"],d.length_um),
            "budget":budget,"radiative_extraction_fraction":2*budget["local_amplitude_loss_per_m"]/(2*value.imag+100*BASE.INTERNAL_LOSS_CM),
            "beam":beam,"beam_acceptance":beam_acceptance(beam,require_sampling=True),
            "grid_sensitivity_cm1":None,"grid_status":"single_grid_complete_QZ"}
        rows.append(row)
        if rank<4:
            arrays.update({f"global_{rank}_fields":field,f"global_{rank}_far_x":fx,f"global_{rank}_far_y":fy})
    if len(rows)<2:raise RuntimeError("Fewer than two passive QZ poles")
    result={"status":"completed","id":source["id"],"design":source["design"],
        "source_case":str(case),"source_settings":source["settings"],"spectrum_grid":grid,
        "verified_QZ_pole_count":len(values),"passive_pole_count":len(indices),
        "detuning_window":"all finite residual-verified QZ poles; no hand-selected detuning exclusion",
        "detuning_range_per_m":[float(values.real.min()),float(values.real.max())],
        "radiation_map_fit_residuals":residuals,
        "fabrication":source["fabrication"],"wavelength_nm":source["wavelength_nm"],
        "active_Gamma":source["active_Gamma"],"effective_index":source["effective_index"],
        "modes":rows,"lowest":rows[0],"second":rows[1]["mode"],
        "family_gap_cm1":rows[1]["alpha_field_cm1"]-rows[0]["alpha_field_cm1"],
        "material_gain_relative_gap":rows[1]["g_th_cm1"]/rows[0]["g_th_cm1"]-1,
        "combined_grid_sensitivity_cm1":None,
        "useful_radiation":bool(rows[0]["radiative_extraction_fraction"]>=.01),
        "selected_family_winner_g_th_cm1":source["lowest"]["g_th_cm1"],
        "selected_family_winner_is_lowest":bool(abs(source["lowest"]["g_th_cm1"]-rows[0]["g_th_cm1"])/rows[0]["g_th_cm1"]<.02),
        "scope":"complete coarse finite QZ pencil; mesh/truncation/real frequency remain to be checked"}
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/"result.json",result)
    np.savez_compressed(output/"global_raw.npz",axis_deg=axis,**arrays)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources",nargs="+",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--grid",type=int,default=9)
    parser.add_argument("--id",nargs="*")
    args=parser.parse_args(argv)
    output=prepare_output(args.output or default_output(ROOT,"optimization_20261021/square/full_QZ"),ROOT)
    write_provenance(output,ROOT)
    save_json(output/"run_plan.json",{"sources":[str(s) for s in args.sources],"grid":args.grid,
        "spectral_window":"none; all residual-verified finite generalized eigenpairs",
        "large_apertures":"sampling may fail +/-1 degree at N9; retained as unresolved, never accepted"})
    save_json(output/"run_status.json",{"status":"running"})
    cases=[(case.parent,json.loads(case.read_text(encoding="utf-8")))
           for folder in args.sources for case in sorted(folder.glob("case_*/result.json"))]
    if args.id:cases=[(case,record) for case,record in cases if record["id"] in args.id]
    results=[]
    for index,(case,source) in enumerate(cases):
        try:
            result=evaluate_case(source,case,output/f"case_{index:04d}_{source['id']}",args.grid)
        except Exception as error:
            result={"status":"failed","id":source["id"],"source_case":str(case),
                    "design":source["design"],"error":str(error)}
        results.append(result)
        if (index+1)%20==0:
            summarize(results,output)
            accepted=sum(r["status"]=="completed" and r["useful_radiation"] and r["lowest"]["beam_acceptance"]["accepted"] for r in results)
            print(f"complete coarse QZ {index+1}/{len(cases)}, bright/round/useful winners {accepted}",flush=True)
    rows=summarize(results,output)
    save_json(output/"run_status.json",{"status":"completed","cases":len(results),"success":len(rows),
        "failures":len(results)-len(rows),"bright_round_useful":sum(r["status"]=="completed" and r["useful_radiation"] and r["lowest"]["beam_acceptance"]["accepted"] for r in results)})


if __name__=="__main__":main()
