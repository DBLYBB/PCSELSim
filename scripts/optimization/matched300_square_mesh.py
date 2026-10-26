"""Continue known physical finite fields on a new mesh or nearby fixed period.

This is explicitly a mesh-sensitivity census, not a complete fine-mesh spectrum.
All actual returned passive poles are ranked, including non-round competitors.
"""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import argparse,json,sys,time
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pcselsim.custom_analysis import FourWaveOpticalSpec,LinearMode,finite_area_eigenproblem
from pcselsim.finite_discretization import relative_residual
from pcselsim.polarization import polarization_metrics
from scripts.optimization.square_semiconductor_validate import independent_pole
from scripts.optimization.matched300_square_search import (Design,build_model,save_json,BASE,
    threshold,direct_vector_far_field,acceptable)


def interpolate_field(field,grid):
    old=field.shape[-1]
    if old==grid:return field.copy()/np.linalg.norm(field)
    axis=(np.arange(old)+.5)/old;target=(np.arange(grid)+.5)/grid
    yy,xx=np.meshgrid(target,target,indexing="ij");points=np.column_stack((yy.ravel(),xx.ravel()))
    result=np.stack([RegularGridInterpolator((axis,axis),part,bounds_error=False,fill_value=None)(points).reshape(grid,grid) for part in field])
    return result/np.linalg.norm(result)


def seeded_survey(source,raw_path,output,grid=65,order=64,dz=1.,design=None,count=12,vertices=96):
    started=time.monotonic();d=Design(**(design or source["design"]))
    _,cell,m=build_model(d,order,dz,950.,False,vertices)
    with np.load(raw_path,allow_pickle=False) as z:
        old_C=z["coupling_m"].copy()
        seeds=[(i,z[f"pole_{i}_field"].copy()) for i in range(min(count,len(source["ordered_poles"]))) if f"pole_{i}_field" in z]
    spec=FourWaveOpticalSpec(950.,m.effective_index,m.group_index,300.,5.,(0.,)*4,(0.,)*4,grid,"staggered",16,0,400)
    problem=finite_area_eigenproblem(spec,m.coupling_m);values=[];fields=[];matches=[];failures=[]
    for index,old_field in seeds:
        seed=interpolate_field(old_field,grid);flat=old_field.reshape(4,-1)
        info=source["ordered_poles"][index];v0=complex(info["delta_per_m"],100*info["alpha_field_cm1"])
        v0+=np.vdot(flat,(m.coupling_m-old_C)@flat)/np.vdot(flat,flat)
        errors=[]
        for requested,offset in ((16,10+10j),(8,100+40j),(8,-100+40j)):
            try:
                vs,fs=problem.near(v0+offset,requested,seed.ravel(),maxiter=400,dense_limit=0)
                overlaps=abs(seed.ravel().conj()@fs)**2;best=int(np.argmax(overlaps))
                matches.append({"reference_rank":index,"reference_mode":info["mode"],"requested":requested,
                    "target_per_m":[float((v0+offset).real),float((v0+offset).imag)],
                    "best_overlap":float(overlaps[best]),"matched_value_per_m":[float(vs[best].real),float(vs[best].imag)],
                    "matched_eigen_residual":relative_residual(problem.center_operator(),fs[:,best],vs[best]),"retry_errors":errors})
                for value,field in zip(vs,fs.T):
                    field=field.reshape(4,grid,grid)
                    if value.imag>=-1e-7 and independent_pole(value,field,values,fields):values.append(value);fields.append(field)
                break
            except Exception as error:errors.append(str(error))
        else:failures.append({"reference_rank":index,"errors":errors})
        print(f"mesh/geometry continuation {index+1}/{len(seeds)}; independent poles {len(values)}",flush=True)
    rows=[];arrays={"coupling_m":m.coupling_m};operator=problem.center_operator()
    gamma=m.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME]
    for rank,index in enumerate(np.argsort(np.imag(values))[:24]):
        value,field=values[index],fields[index];rx,ry=m.radiation_fields(field)
        mode=LinearMode(f"pole_{rank}",float(value.real),float(value.imag),field,0.,rx,ry,discretization="staggered")
        axis,fx,fy,beam=direct_vector_far_field(mode,950.,300.,161)
        budget=problem.loss_budget(field)
        rows.append({"mode":mode.name,"delta_per_m":value.real,"alpha_field_cm1":value.imag/100,
            **threshold(value.imag,gamma,m.effective_index,300.),"beam":beam,"eigen_residual":relative_residual(operator,field,value),
            "radiative_extraction_fraction":2*budget["local_amplitude_loss_per_m"]/(2*value.imag+500),
            "polarization":polarization_metrics(axis,fx,fy,core_radius_deg=.2)})
        if rank<12:arrays.update({f"pole_{rank}_field":field,f"pole_{rank}_far_x":fx,f"pole_{rank}_far_y":fy})
    if len(rows)<2:raise RuntimeError("Mesh continuation returned fewer than two physical poles")
    result={"status":"completed","id":d.identifier,"design":asdict(d),"D":order,"dz_nm":dz,"grid":grid,"vertices":vertices,
        "ordered_poles":rows,"lowest":rows[0],"distinct_poles":len(values),"full_QZ":False,
        "relative_material_gap":rows[1]["g_th_cm1"]/rows[0]["g_th_cm1"]-1,"reference_matches":matches,"failures":failures,
        "source_grid":source["grid"],"source_pole_count":len(seeds),"source_census_scope":source["scope"],
        "extra_coarse_QZ_targets":source.get("extra_coarse_QZ_targets",[]),"basic_beam_pass":acceptable(rows[0]),
        "elapsed_seconds":time.monotonic()-started,
        "scope":"mesh/nearby-period continuation of the lowest known 12 physical fields; all returned passive poles ranked; not complete fine spectrum"}
    output.mkdir(parents=True,exist_ok=True);save_json(output/"spectral_challenge.json",result)
    np.savez_compressed(output/"lateral_fields.npz",axis_deg=axis,**arrays)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("case",type=Path)
    p.add_argument("--output",required=True,type=Path);p.add_argument("--grid",type=int,default=65)
    p.add_argument("--D",type=int,default=64);p.add_argument("--dz",type=float,default=1.)
    p.add_argument("--vertices",type=int,default=96)
    p.add_argument("--design",type=Path)
    args=p.parse_args(argv);source=json.loads((args.case/"spectral_challenge.json").read_text(encoding="utf-8"))
    design=None if args.design is None else json.loads(args.design.read_text(encoding="utf-8"))
    if isinstance(design,list):design=design[0]
    if design and "design" in design:design=design["design"]
    seeded_survey(source,args.case/"lateral_fields.npz",args.output,args.grid,args.D,args.dz,design,vertices=args.vertices)

if __name__=="__main__":main()
