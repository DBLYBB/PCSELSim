"""Small actual-reference, fine-mesh shape/position correction of a saved motif."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[_key]="1"
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from pcselsim.design_objectives import beam_metrics
from pcselsim.triangular_finite import TriangularFiniteSpec,TriangularFiniteMode,triangular_finite_eigenproblem,triangular_vector_far_field
from scripts.optimization.matched300_triangular_actual_controls import follow,separable_far_field
from scripts.optimization.matched300_triangular_qz_guided import model_for,scalar_record
from scripts.optimization.matched300_triangular_search import ThreeHole,geometry
from scripts.optimization.triangular_semiconductor_search import json_write,write_csv,threadpool_limits


def covariance_q(axis,fx,fy):
    xx,yy=np.meshgrid(axis,axis);mass=abs(fx)**2+abs(fy)**2
    weights=np.ones(len(axis));weights[[0,-1]]=.5;mass*=weights[:,None]*weights[None,:]
    mass/=mass.sum();dx=xx-(xx*mass).sum();dy=yy-(yy*mass).sum()
    qxx=(mass*dx*dx).sum();qyy=(mass*dy*dy).sum();qxy=(mass*dx*dy).sum()
    return [float((qxx-qyy)/(qxx+qyy)),float(2*qxy/(qxx+qyy))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cells",type=int,default=52);parser.add_argument("--order",type=int,default=48)
    parser.add_argument("--dz",type=float,default=1.);parser.add_argument("--samples",type=int,default=121)
    parser.add_argument("--orientation-offsets",type=float,nargs="+",help="Replace the 18-point correction with an explicit main-angle control sweep")
    parser.add_argument("--roots",type=int,default=3,help="Number of known branches for exploratory continuation; final spectrum requires independent audit")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=json.loads((args.case/"summary.json").read_text());spec=ThreeHole(**source["three_hole_spec"])
    arrays=np.load(args.case/"frequency_closed/frequency_closed_fields.npz")
    poles=json.loads((args.case/"frequency_closed/frequency_closed_poles.json").read_text())
    poles=sorted([r for r in poles if r["status"]=="completed"],key=lambda r:r["g_th_cm"])[:args.roots]
    triangulation=Delaunay(np.column_stack((arrays["x_um"],arrays["y_um"])))
    controls=[("nominal",{})]
    for key,values in (("main_orientation_deg",(-2.,-1.,1.,2.)),("distance1",(-.003,-.001,.001,.003)),
        ("distance2",(-.003,-.001,.001,.003)),("satellite_balance",(-.005,.005))):
        for offset in values:controls.append((f"{key}_{offset:+g}",{key:getattr(spec,key)+offset}))
    if args.orientation_offsets:
        controls=[("nominal",{})]+[(f"main_orientation_deg_{offset:+g}",{"main_orientation_deg":spec.main_orientation_deg+offset}) for offset in args.orientation_offsets]
    rows=[];saved=[];seeds=None;reference_values=None;reference_c=None
    with threadpool_limits(limits=1):
        index=0
        while index<len(controls):
            label,changes=controls[index];index+=1;started=time.perf_counter()
            trial=replace(spec,**changes,name=f"actual_local_{index-1:03d}")
            fabrication=geometry(trial)
            if not fabrication["margin_pass"]:
                rows.append({"name":label,"status":"geometry_rejected","three_hole_spec":asdict(trial),"fabrication":fabrication});continue
            try:
                current_source={**source,"three_hole_spec":asdict(trial)}
                _,parameters,model=model_for(current_source,args.order,args.dz)
                problem,grid=triangular_finite_eigenproblem(model,TriangularFiniteSpec(150.,args.cells,0.,16,"square","staggered"))
                if seeds is None:
                    seeds=[]
                    for pole in poles:
                        interpolation=LinearNDInterpolator(triangulation,arrays[f"root_{pole['seed_rank']}_fields"].T,fill_value=0.)
                        field=interpolation(np.column_stack((grid.x_um,grid.y_um))).T.ravel();seeds.append(field/np.linalg.norm(field))
                records=[];fields=[]
                for rank,seed in enumerate(seeds):
                    target=complex(np.vdot(seed,problem.center_operator()@seed))+complex(50.,5.)
                    value,field,overlap,solver=follow(problem,target,seed,16)
                    if overlap<.75:raise RuntimeError(f"Branch {rank} overlap {overlap}")
                    record=scalar_record(value,field,model,parameters,problem,grid,model.bragg_wavelength_nm,rank)
                    record.update(field_overlap=overlap,solver=solver);records.append(record);fields.append(field.reshape(6,grid.points))
                if index==1:
                    seeds=[f.ravel().copy() for f in fields]
                main=records[0];rx,ry=model.radiation_amplitudes(fields[0])
                mode=TriangularFiniteMode("tracked_lowest",main["delta_per_m"],main["alpha_per_m"],fields[0],1.,grid,rx,ry,discretization="staggered")
                far=separable_far_field(mode,main["frozen_CWT_pole_wavelength_nm"],1.,args.samples)
                beam=beam_metrics(far.angle_deg,far.field_x,far.field_y,requested_view_deg=1.,source_nyquist_deg=far.alias_free_square_view_deg)
                q=covariance_q(far.angle_deg,far.field_x,far.field_y)
                gap=min(r["g_th_cm"] for r in records[1:])/main["g_th_cm"]-1
                row={"name":label,"status":"completed","three_hole_spec":asdict(trial),"N":args.cells,"D":args.order,"dz_nm":args.dz,
                    "reference_wavelength_nm":model.bragg_wavelength_nm,"tracked_modes":records,"spectrum":records,"beam":beam,"covariance_q":q,
                    "g_th_cm":main["g_th_cm"],"ellipticity":beam["ellipticity"],"center_to_peak":beam["center_to_peak"],
                    "radial_anisotropy_l2":beam["radial_anisotropy_l2"],"g_th_gap_fraction":gap,
                    "frozen_CWT_pole_wavelength_nm":main["frozen_CWT_pole_wavelength_nm"],"fabrication":fabrication,
                    "known_branches":len(records),
                    "scope":f"{len(records)} known finite branches followed at fixed actual-reference C; exploratory ranking, not complete spectrum or independent frequency closure"}
                row["score"]=beam["ellipticity"]+5*max(.98-beam["center_to_peak"],0)+5*max(.06-gap,0)+5*max(beam["radial_anisotropy_l2"]-.10,0)
                saved.append((row,fields,model.coupling_m,grid.x_um,grid.y_um))
            except Exception as error:row={"name":label,"status":"failed","three_hole_spec":asdict(trial),"exception":repr(error)}
            row["elapsed_seconds"]=time.perf_counter()-started;rows.append(row)
            json_write(args.output/"candidates.json",rows);write_csv(args.output/"candidates.csv",rows)
            print(label,row.get("g_th_cm"),row.get("ellipticity"),row.get("g_th_gap_fraction"),flush=True)
            if not args.orientation_offsets and index==15 and all(r.get("status")=="completed" for r in rows):
                # Two covariance components, three local knobs. The least-norm
                # correction is clipped to the scanned trust region.
                lookup={r["name"]:r for r in rows};columns=[]
                for key,step in (("main_orientation_deg",1.),("distance1",.001),("distance2",.001)):
                    positive=np.asarray(lookup[f"{key}_{step:+g}"]["covariance_q"])
                    negative=np.asarray(lookup[f"{key}_{-step:+g}"]["covariance_q"])
                    columns.append((positive-negative)/2)
                jacobian=np.asarray(columns).T;correction=np.linalg.lstsq(jacobian,-np.asarray(rows[0]["covariance_q"]),rcond=None)[0]
                correction=np.clip(correction,(-2.,-3.,-3.),(2.,3.,3.))
                for scale in (.5,1.,1.5):
                    controls.append((f"covariance_correction_{scale:g}",{
                        "main_orientation_deg":spec.main_orientation_deg+scale*correction[0],
                        "distance1":spec.distance1+scale*correction[1]*.001,
                        "distance2":spec.distance2+scale*correction[2]*.001}))
        ranked=sorted(saved,key=lambda item:item[0]["score"])
        for rank,(row,fields,coupling,x,y) in enumerate(ranked[:4]):
            folder=args.output/f"selected_{rank:02d}";folder.mkdir(exist_ok=True)
            json_write(folder/"result.json",row)
            np.savez_compressed(folder/"complex_fields.npz",fields=fields,eigenvalues=np.asarray([complex(r["delta_per_m"],r["alpha_per_m"]) for r in row["spectrum"]]),
                coupling_m=coupling,x_um=x,y_um=y)
        json_write(args.output/"summary.json",{"attempts":len(rows),"completed":len(saved),"best":[row for row,*_ in ranked[:4]],"scope":"Actual-reference fine-grid local branch search; winners require independent actual-950 retuning and expanded spectral audit"})


if __name__=="__main__":main()
