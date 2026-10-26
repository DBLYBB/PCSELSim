"""Continue a coarse-guided finite pole pool to a finer nominal 300-um grid."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[_key]="1"
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from pcselsim.design_objectives import beam_metrics
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.triangular_finite import TriangularFiniteSpec,TriangularFiniteMode,triangular_finite_eigenproblem
from scripts.optimization.matched300_triangular_actual_controls import follow,separable_far_field
from scripts.optimization.matched300_triangular_qz_guided import model_for,scalar_record
from scripts.optimization.triangular_semiconductor_search import json_write,write_csv,threadpool_limits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True);parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--cells",type=int,default=91)
    parser.add_argument("--order",type=int,default=48);parser.add_argument("--dz",type=float,default=1.)
    parser.add_argument("--poles",type=int,default=12)
    args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    source=json.loads((args.case/"summary.json").read_text());prior=json.loads((args.input/"result.json").read_text())
    arrays=np.load(args.input/"complex_fields.npz");traces=[];found_values=[];found_fields=[];started=time.perf_counter()
    triangulation=Delaunay(np.column_stack((arrays["x_um"],arrays["y_um"])))
    with threadpool_limits(limits=1):
        spec,parameters,model=model_for(source,args.order,args.dz)
        if np.linalg.norm(arrays["coupling_m"]-model.coupling_m)>1e-10*np.linalg.norm(model.coupling_m):
            raise ValueError("Fine-mesh continuation requires the identical actual-reference coupling matrix")
        problem,grid=triangular_finite_eigenproblem(model,TriangularFiniteSpec(150.,args.cells,0.,24,"square","staggered"))
        for rank in range(min(args.poles,len(arrays["fields"]))):
            step=time.perf_counter()
            interpolation=LinearNDInterpolator(triangulation,arrays["fields"][rank].T,fill_value=0.)
            seed=interpolation(np.column_stack((grid.x_um,grid.y_um))).T.ravel();seed/=np.linalg.norm(seed)
            target=complex(np.vdot(seed,problem.center_operator()@seed))+complex(50.,5.)
            try:
                value,field,overlap,solver=follow(problem,target,seed,24)
                if overlap<.75:raise RuntimeError(f"Insufficient physical-center branch overlap {overlap}")
                duplicate=not independent_eigenmode(value,field,found_values,found_fields)
                if not duplicate:found_values.append(value);found_fields.append(field)
                row={"source_rank":rank,"status":"completed","delta_per_m":value.real,"alpha_per_m":value.imag,
                    "field_overlap":overlap,"duplicate_of_existing_span":bool(duplicate),"solver":solver,"elapsed_seconds":time.perf_counter()-step}
            except Exception as error:
                row={"source_rank":rank,"status":"failed","exception":repr(error),"elapsed_seconds":time.perf_counter()-step}
            traces.append(row);json_write(args.output/"progress.json",traces)
            print("Mesh challenge",args.cells,rank,row["status"],row.get("alpha_per_m"),flush=True)
        order=np.argsort(np.imag(found_values));values=np.asarray(found_values)[order];fields=np.asarray(found_fields)[order].reshape(-1,6,grid.points)
        records=[]
        for rank,(value,field) in enumerate(zip(values,fields)):
            row=scalar_record(value,field,model,parameters,problem,grid,model.bragg_wavelength_nm,rank)
            if rank<6:
                rx,ry=model.radiation_amplitudes(field)
                mode=TriangularFiniteMode(f"continued_{rank}",value.real,value.imag,field,1.,grid,rx,ry,discretization="staggered")
                far=separable_far_field(mode,row["frozen_CWT_pole_wavelength_nm"],1.,201)
                row["beam"]=beam_metrics(far.angle_deg,far.field_x,far.field_y,requested_view_deg=1.,source_nyquist_deg=far.alias_free_square_view_deg)
            records.append(row)
        result={"three_hole_spec":asdict(spec),"N":args.cells,"D":args.order,"dz_nm":args.dz,"spectrum":records,
            "reference_wavelength_nm":model.bragg_wavelength_nm,"source_mesh_N":prior["N"],"source_targets":args.poles,
            "retained_poles":len(records),"g_th_cm":records[0]["g_th_cm"],"lowest":records[0],"second":records[1],
            "g_th_gap_fraction":records[1]["g_th_cm"]/records[0]["g_th_cm"]-1,"elapsed_seconds":time.perf_counter()-started,
            "scope":"Physical-center field continuation from the low-pole pool of the complete-coarse-spectrum-guided N39 search; not a complete fine-grid spectrum"}
        json_write(args.output/"result.json",result);write_csv(args.output/"all_poles.csv",records)
        np.savez_compressed(args.output/"complex_fields.npz",eigenvalues=values,fields=fields,x_um=grid.x_um,y_um=grid.y_um,coupling_m=model.coupling_m)


if __name__=="__main__":main()
