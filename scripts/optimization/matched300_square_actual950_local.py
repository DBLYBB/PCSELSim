"""Local screen with the actual lowest coarse QZ pole designed at 950 nm.

This corrects a practical distinction: a 950-nm frozen Bragg reference does
not ensure the real-frequency pole or its final on-axis intensity is 950 nm.
"""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse,json,sys,time
from dataclasses import asdict,replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from pcselsim.custom_analysis import FourWaveOpticalSpec,finite_area_eigenproblem,LinearMode
from pcselsim.fabrication import evaluate_fabrication
from pcselsim.frequency_cwt import background_detuning
from scripts.optimization.matched300_square_search import (Design,build_model,build_cell,RULES,
    OUT,save_json,threshold,direct_vector_far_field,BASE,acceptable)


def evaluate_actual(d,order=10,dz=3.,grid=9):
    started=time.monotonic();trace=[]
    for iteration in range(12):
        d,cell,m=build_model(d,order,dz,950.,False,64)
        spec=FourWaveOpticalSpec(950.,m.effective_index,m.group_index,300.,5.,(0.,)*4,(0.,)*4,grid,"staggered",12,4*grid*grid,1200)
        p=finite_area_eigenproblem(spec,m.coupling_m);p.near(0j,4,np.ones(4*grid*grid,complex),dense_limit=4*grid*grid)
        values,fields=p._dense_spectrum;indices=[i for i in np.argsort(values.imag) if values[i].imag>=-1e-7]
        winner=values[indices[0]];residual=background_detuning(950.,m.effective_index,d.lattice_nm)-winner.real
        trace.append({"iteration":iteration,"period_nm":d.lattice_nm,"residual_per_m":residual,"lowest_delta_per_m":winner.real,"lowest_alpha_per_m":winner.imag})
        if abs(residual)<.05:break
        beta=2*np.pi*m.effective_index/950e-9
        beta0=-winner.real+np.sqrt(winner.real**2+beta**2)
        d=replace(d,lattice_nm=float(2*np.pi/beta0*1e9))
    else:raise RuntimeError("Actual lowest coarse-QZ pole did not close at 950nm")
    gamma=m.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME];rows=[]
    for rank,index in enumerate(indices[:6]):
        value=values[index];field=fields[:,index].reshape(4,grid,grid);rx,ry=m.radiation_fields(field)
        mode=LinearMode(f"pole_{rank}",float(value.real),float(value.imag),field,0.,rx,ry,discretization="staggered")
        axis,fx,fy,beam=direct_vector_far_field(mode,950.,300.,129)
        rows.append({"mode":mode.name,"alpha_field_cm1":value.imag/100,"delta_per_m":value.real,
            **threshold(value.imag,gamma,m.effective_index,300.),"beam":beam})
    fab=evaluate_fabrication(cell.inclusions,d.lattice_nm,rules=RULES)
    result={"status":"completed","id":d.identifier,"design":asdict(d),"D":order,"dz_nm":dz,"grid":grid,
        "period_trace":trace,"period_design_residual_per_m":residual,"wavelength_nm":950.,
        "lowest":rows[0],"ordered_poles":rows,"relative_material_gap":rows[1]["g_th_cm1"]/rows[0]["g_th_cm1"]-1,
        "fabrication":fab,"basic_beam_pass":acceptable(rows[0]),"strict_beam_pass":acceptable(rows[0],True),
        "elapsed_seconds":time.monotonic()-started,"scope":"complete coarse QZ and actual-lowest-pole period design; final high-D/high-N frequency recheck still required"}
    return result


def proposals(count=192):
    parents=[Design(**json.loads((OUT/name/"period_design.json").read_text(encoding="utf-8"))["design"])
             for name in ("final_balanced","final_isolation","final_low_threshold")]
    parents.extend(Design(**x) for x in json.loads((OUT/"core_finalist_designs.json").read_text(encoding="utf-8"))[1:])
    names=("fill","main_share","distance","position_angle_deg","main_aspect","satellite_aspect","main_angle_deg","satellite_angle_deg")
    scales=np.array([.012,.035,.02,3.,.09,.11,6.,6.]);low=np.array([.08,.5,.43,33.,1.,1.,-90.,-90.]);high=np.array([.26,.8,.67,57.,2.,2.,90.,90.])
    rng=np.random.default_rng(30021950);accepted=list(parents);attempts=0
    while len(accepted)<count:
        parent=parents[(len(accepted)-len(parents))%len(parents)];spread=(.4,.75,1.3)[len(accepted)%3]
        value=np.clip([getattr(parent,k) for k in names]+rng.normal(size=8)*scales*spread,low,high)
        d=replace(parent,**dict(zip(names,map(float,value))),tag=f"actual950_local_{len(accepted)}")
        attempts+=1
        if evaluate_fabrication(build_cell(d,48).inclusions,d.lattice_nm,rules=RULES)["margin_pass"]:accepted.append(d)
        if attempts>count*30:raise RuntimeError("insufficient feasible actual-frequency proposals")
    return accepted


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--count",type=int,default=192)
    p.add_argument("--D",type=int,default=10);p.add_argument("--dz",type=float,default=3.)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args(argv);output=OUT/"actual950_local_QZ";output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        designs=[Design(**x) for x in json.loads((output/"designs.json").read_text(encoding="utf-8"))]
        records=json.loads((output/"compact_results.json").read_text(encoding="utf-8"))
    else:
        designs=proposals(args.count);save_json(output/"designs.json",[asdict(d) for d in designs]);records=[]
    for index,d in enumerate(designs[len(records):],start=len(records)):
        try:r=evaluate_actual(d,args.D,args.dz)
        except Exception as e:r={"status":"failed","design":asdict(d),"error":str(e)}
        records.append(r)
        if (index+1)%8==0 or index==len(designs)-1:
            save_json(output/"compact_results.json",records)
            print(f"actual950 {index+1}/{len(designs)} strict={sum(r.get('strict_beam_pass',False) for r in records)} basic={sum(r.get('basic_beam_pass',False) for r in records)}",flush=True)
    save_json(output/"run_status.json",{"status":"completed","count":len(records)})

if __name__=="__main__":main()
