"""Challenge a 300-um candidate with pair-limit complex shifts/high-perp seeds."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[key]="1"
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from pcselsim.design_objectives import beam_metrics
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.triangular_finite import TriangularFiniteSpec,TriangularFiniteMode,triangular_finite_eigenproblem,triangular_vector_far_field
from scripts.optimization.matched300_triangular_qz_guided import model_for,scalar_record
from scripts.optimization.triangular_semiconductor_search import json_write,write_csv,threadpool_limits


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--case",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cells",type=int,default=39);p.add_argument("--target-real",type=float,default=-77204.543)
    p.add_argument("--target-imags",type=float,nargs="+",default=(0.,50.,100.));p.add_argument("--q-fractions",type=float,nargs="+",default=(.35,.65))
    p.add_argument("--eigen-count",type=int,default=24)
    p.add_argument("--order",type=int,default=32);p.add_argument("--dz",type=float,default=1.)
    p.add_argument("--target-file",type=Path,help="Independent three-pair continuous-transfer audit at the identical actual reference")
    p.add_argument("--imag-scales",type=float,nargs="+",default=(0.,1.))
    p.add_argument("--pairs",type=int,nargs="+",default=(0,1,2),help="Order of analytic opposite-wave pairs")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    source=json.loads((a.case/"summary.json").read_text());start=time.perf_counter();traces=[];values_out=[];fields_out=[]
    with threadpool_limits(limits=1):
        spec,params,model=model_for(source,a.order,a.dz)
        problem,grid=triangular_finite_eigenproblem(model,TriangularFiniteSpec(150.,a.cells,0.,a.eigen_count,"square","staggered"))
        ref=model.bragg_wavelength_nm
        plan=[(pair,q,a.target_real,imag) for pair in (0,1) for q in a.q_fractions for imag in a.target_imags]
        if a.target_file:
            audit=json.loads(a.target_file.read_text())
            if abs(audit["actual_wavelength_nm"]-ref)>1e-9:raise ValueError("Stripe audit and model actual references differ")
            audit_c=np.asarray(audit["coupling_real_per_m"])+1j*np.asarray(audit["coupling_imag_per_m"])
            if np.linalg.norm(audit_c-model.coupling_m)>1e-10*np.linalg.norm(model.coupling_m):raise ValueError("Stripe audit coupling differs")
            plan=[]
            pair_lookup={pair["indices"][0]//2:pair for pair in audit["pairs"]}
            for pair_index in a.pairs:
                pair=pair_lookup[pair_index]
                pole=pair["independent_continuous_transfer_pole"]
                if not pole["usable"]:continue
                plan.extend((pair["indices"][0]//2,q,pole["delta_per_m"],scale*pole["alpha_field_per_m"])
                    for q in a.q_fractions for scale in a.imag_scales)
        json_write(a.output/"run_plan.json",{"N":a.cells,"D":a.order,"dz_nm":a.dz,"actual_period_nm":spec.lattice_nm,"reference_wavelength_nm":ref,
            "target_real_per_m":a.target_real,"target_imag_per_m":a.target_imags,"high_perp_q_fraction_of_pi_over_h":a.q_fractions,
            "target_file":str(a.target_file) if a.target_file else None,"explicit_targets":[list(row) for row in plan],
            "origin":"Independent two-wave stripe-limit diagnostic supplies a challenge target, not a claimed 2D threshold"})
        def retain(value,field):
            if not independent_eigenmode(value,field,values_out,fields_out):return
            values_out.append(value);fields_out.append(field)
        for index,(pair,q,real,imag) in enumerate(plan):
            t=time.perf_counter();theta=np.deg2rad((-60,60,0)[pair])
            xx,yy=grid.x_um*1e-6,grid.y_um*1e-6
            longitudinal=np.cos(theta)*xx+np.sin(theta)*yy;perpendicular=-np.sin(theta)*xx+np.cos(theta)*yy
            chord=300e-6/max(abs(np.sin(theta)),abs(np.cos(theta)))
            envelope=np.maximum(np.cos(np.pi*longitudinal/chord),0.)*np.exp(-(perpendicular/40e-6)**2)
            stripe=envelope*np.exp(1j*q*np.pi/grid.spacing_m*perpendicular)
            initial=np.zeros((6,grid.points),complex);initial[2*pair]=stripe;initial[2*pair+1]=1j*stripe
            initial=initial.ravel()/np.linalg.norm(initial)
            rng=np.random.default_rng(20261021+index);initial+=.01*(rng.normal(size=initial.size)+1j*rng.normal(size=initial.size))/np.sqrt(initial.size)
            target=complex(real,imag)
            try:
                values,fields=problem.near(target,a.eigen_count,initial,tol=1e-10,maxiter=1600,dense_limit=0)
                for value,field in zip(values,fields.T):
                    if value.imag>=-1e-7:retain(value,field)
                trace={"index":index,"pair":pair,"q_fraction":q,"target_real_per_m":real,"target_imag_per_m":imag,"returned":len(values),"status":"completed"}
            except Exception as error:trace={"index":index,"status":"failed","exception":repr(error)}
            trace["elapsed_seconds"]=time.perf_counter()-t;traces.append(trace);json_write(a.output/"progress.json",traces)
            print("stripe probe",index+1,len(plan),"pair",pair,"q",q,"imag",imag,"lowest alpha",min((v.imag for v in values_out),default=None),"s",trace["elapsed_seconds"],flush=True)
            if values_out:
                order=np.argsort(np.imag(values_out));best=np.asarray(values_out)[order][:12]
                json_write(a.output/"lowest_progress.json",[{"delta_per_m":float(v.real),"alpha_per_m":float(v.imag)} for v in best])
        order=np.argsort(np.imag(values_out),kind="stable");values=np.asarray(values_out)[order];fields=np.asarray(fields_out)[order].reshape(-1,6,grid.points)
        records=[scalar_record(v,f,model,params,problem,grid,ref,i) for i,(v,f) in enumerate(zip(values,fields))]
        for i in range(min(8,len(values))):
            rx,ry=model.radiation_amplitudes(fields[i]);mode=TriangularFiniteMode(f"stripe_probe_{i}",float(values[i].real),float(values[i].imag),fields[i],1.,grid,rx,ry,discretization="staggered")
            far=triangular_vector_far_field(mode,records[i]["frozen_CWT_pole_wavelength_nm"],1.,161)
            records[i]["beam"]=beam_metrics(far.angle_deg,far.field_x,far.field_y,requested_view_deg=1.,source_nyquist_deg=far.alias_free_square_view_deg)
        row={"three_hole_spec":asdict(spec),"N":a.cells,"D":a.order,"dz_nm":a.dz,"spectrum":records,"retained_poles":len(records),
            "g_th_cm":records[0]["g_th_cm"],"g_th_gap_fraction":records[1]["g_th_cm"]/records[0]["g_th_cm"]-1,
            "lowest":records[0],"elapsed_seconds":time.perf_counter()-start}
        json_write(a.output/"result.json",row);write_csv(a.output/"all_poles.csv",records)
        np.savez_compressed(a.output/"complex_fields.npz",eigenvalues=values[:24],fields=fields[:24],x_um=grid.x_um,y_um=grid.y_um,coupling_m=model.coupling_m)


if __name__=="__main__":main()
