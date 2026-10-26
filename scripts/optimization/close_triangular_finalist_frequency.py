"""Fixed-period real-frequency closure of independently tracked triangular poles.

Rebuilds TE0, all six-wave couplings and radiation at each trial wavelength.
The finite grid and exact hole geometry remain fixed. This is sampled CWT,
not an exhaustive fine-grid nonlinear or full-Maxwell laser calculation.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from pcselsim.design_objectives import beam_metrics,beam_acceptance
from pcselsim.finite_discretization import relative_residual
from pcselsim.run_output import prepare_output,write_provenance
from pcselsim.triangular_six_wave import TriangularCWTSettings,build_triangular_coupling,TriangularLatticeCell
from pcselsim.triangular_finite import TriangularFiniteSpec,TriangularFiniteMode,triangular_finite_eigenproblem,triangular_vector_far_field
from pcselsim.vertical import Layer,LayerStack
from scripts._triangular_workflow import threshold_audit,aperture_area_m2
from scripts.optimization.triangular_semiconductor_search import Motif,make_candidate,parameters,evaluate_finite,json_write,write_csv,threadpool_limits


def background_delta(wavelength,index,period):
    beta=2*np.pi*index/(wavelength*1e-9)
    beta0=4*np.pi/(np.sqrt(3)*period*1e-9)
    return float((beta*beta-beta0*beta0)/(2*beta0))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-json",type=Path,default=ROOT/"configs/optimized_20261021/triangular_finalists.json")
    parser.add_argument("--name",default="local_0067_950nm_400um")
    parser.add_argument("--cells",type=int,default=35)
    parser.add_argument("--roots",type=int,default=4)
    parser.add_argument("--order",type=int,default=10)
    parser.add_argument("--dz",type=float,default=2.)
    parser.add_argument("--seed-dir",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    output=prepare_output(args.output,ROOT)
    write_provenance(output,ROOT)
    (output/"executed_script.py").write_bytes(Path(__file__).read_bytes())
    configuration=json.loads(args.spec_json.read_text(encoding="utf-8"))
    spec=Motif(**next(s for s in configuration["designs"] if s["name"]==args.name))
    json_write(output/"run_plan.json",{"spec":asdict(spec),"cells":args.cells,"roots":args.roots,"D":args.order,"dz_nm":args.dz,
        "config_sha256":hashlib.sha256(args.spec_json.read_bytes()).hexdigest(),
        "scope":"real-frequency C(lambda) closure at fixed actual period and geometry, independently tracked sampled finite lateral modes"})
    json_write(output/"run_status.json",{"status":"running"})
    records=[];arrays={};model_cache={}
    with threadpool_limits(limits=1):
        seed_dir=args.seed_dir.resolve() if args.seed_dir else output/"frozen_seed"
        if args.seed_dir is None:evaluate_finite(spec,args.cells,args.order,args.dz,24,seed_dir)
        source=json.loads((seed_dir/"result.json").read_text())
        if source["spec"]!=asdict(spec) or source["N"]!=args.cells:
            raise ValueError("Seed geometry or finite grid does not match requested design")
        seed=np.load(seed_dir/"complex_fields.npz")
        p=parameters(spec);candidate=make_candidate(spec)
        cell=TriangularLatticeCell(p.BACKGROUND_INDEX**2,candidate.inclusions)
        average=float(np.sqrt(cell.fourier_epsilon(0,0).real))
        stack=LayerStack(tuple(Layer(layer.name,layer.thickness_nm,average if layer.name==p.PC_LAYER_NAME else layer.refractive_index) for layer in p.LAYERS),p.TOP_CLADDING_INDEX,p.BOTTOM_CLADDING_INDEX,p.VERTICAL_PADDING_UM)
        settings=TriangularCWTSettings(args.order,args.dz)
        finite_spec=TriangularFiniteSpec(spec.radius_um,args.cells,0.,24,spec.aperture,"staggered")
        def model_at(wavelength):
            key=float(wavelength)
            if key not in model_cache:
                model_cache[key]=build_triangular_coupling(cell,stack,p.PC_LAYER_NAME,spec.lattice_nm,wavelength,settings,fixed_wavelength_nm=wavelength)
            return model_cache[key]
        for rank in range(min(args.roots,len(seed["eigenvalues"]))):
            start=time.perf_counter();value0=seed["eigenvalues"][rank];field0=seed["fields"][rank]
            vector0=field0.ravel()/np.linalg.norm(field0);reference_c=seed["coupling_m"]
            trace=[];cache={}
            def evaluate(wavelength):
                key=float(wavelength)
                if key in cache:return cache[key]
                model=model_at(key)
                problem,grid=triangular_finite_eigenproblem(model,finite_spec)
                correction=np.einsum("ik,ij,jk->",field0.conj(),model.coupling_m-reference_c,field0)/np.sum(abs(field0)**2)
                predictor=value0+correction
                for count in (8,24,48):
                    values,fields=problem.near(predictor,count,vector0,tol=1e-10,maxiter=1600,dense_limit=0)
                    overlaps=abs(vector0.conj()@fields)**2
                    index=int(np.argmax(overlaps));overlap=float(overlaps[index])
                    if overlap>=.8:break
                if overlap<.8:raise RuntimeError(f"Lost finite pole {rank}: overlap={overlap}")
                value=values[index];field=fields[:,index].reshape(field0.shape)
                if value.imag < -1e-7:raise RuntimeError("Active pole from passive model")
                residual=background_delta(key,model.effective_index,spec.lattice_nm)-value.real
                trace.append({"wavelength_nm":key,"background_delta_per_m":background_delta(key,model.effective_index,spec.lattice_nm),
                    "finite_delta_per_m":float(value.real),"alpha_field_cm":float(value.imag/100),
                    "frequency_residual_per_m":float(residual),"reference_overlap":overlap,"returned_eigenpairs":len(values),
                    "Gamma":model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME],"n_eff":model.effective_index})
                cache[key]=(model,problem,grid,value,field,overlap,float(residual))
                return cache[key]
            try:
                guess=source["spectrum"][rank]["frozen_CWT_pole_wavelength_nm"]
                half=.5
                while True:
                    left,right=guess-half,guess+half
                    fl,fr=evaluate(left)[-1],evaluate(right)[-1]
                    if fl*fr<=0:break
                    half*=2
                    if half>32:raise RuntimeError("No closure root within 32 nm of frozen estimate")
                root=float(brentq(lambda wavelength:evaluate(wavelength)[-1],left,right,xtol=2e-8))
                model,problem,grid,value,field,overlap,residual=evaluate(root)
                if abs(residual)>.05:raise RuntimeError(f"Frequency residual {residual} exceeds .05 /m")
                gamma=model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME]
                audit=threshold_audit(p,float(value.imag),aperture_area_m2(spec.aperture,spec.radius_um),gamma,model.effective_index)
                rx,ry=model.radiation_amplitudes(field)
                mode=TriangularFiniteMode(f"closed_{rank}",float(value.real),float(value.imag),field,overlap,grid,rx,ry,discretization="staggered")
                far=triangular_vector_far_field(mode,root,1.,201)
                beam=beam_metrics(far.angle_deg,far.field_x,far.field_y,requested_view_deg=1.,source_nyquist_deg=far.alias_free_square_view_deg)
                budget=problem.loss_budget(field)
                record={"status":"completed","seed_rank":rank,"wavelength_nm":root,"frozen_wavelength_nm":guess,
                    "g_th_cm":audit["required_material_gain_cm-1"],"J_th_A_cm2":audit["threshold_current_density_A_cm-2"],"I_th_A":audit["threshold_current_A"],
                    "alpha_field_cm":float(value.imag/100),"delta_per_m":float(value.real),"Gamma":gamma,"n_eff":model.effective_index,
                    "frequency_residual_per_m":residual,"reference_overlap":overlap,
                    "eigen_residual":relative_residual(problem.center_operator(),field,value),"beam":beam,"beam_acceptance":beam_acceptance(beam,require_sampling=True),
                    "eta_rad":2*budget["local_amplitude_loss_per_m"]/(2*value.imag+100*p.INTERNAL_LOSS_CM),"budget":budget,
                    "frozen_g_th_cm":source["spectrum"][rank]["g_th_cm"],"trace":trace,"elapsed_seconds":time.perf_counter()-start}
                arrays.update({f"root_{rank}_fields":field,f"root_{rank}_far_x":far.field_x,f"root_{rank}_far_y":far.field_y,
                    f"root_{rank}_coupling_m":model.coupling_m,f"root_{rank}_radiation_x":rx,f"root_{rank}_radiation_y":ry})
                arrays.update(angle_deg=far.angle_deg,x_um=grid.x_um,y_um=grid.y_um)
                print("closed",rank,root,"g",record["g_th_cm"],"ellipse",beam["ellipticity"],"overlap",overlap,flush=True)
            except Exception as error:
                record={"status":"failed","seed_rank":rank,"exception":repr(error),"trace":trace}
                print("failed",rank,repr(error),flush=True)
            records.append(record);json_write(output/"frequency_closed_poles.json",records)
        ordered=sorted((r for r in records if r["status"]=="completed"),key=lambda r:r["g_th_cm"])
        summary={"spec":asdict(spec),"N":args.cells,"D":args.order,"dz_nm":args.dz,"roots_requested":args.roots,
            "roots_closed":len(ordered),"lowest":ordered[0] if ordered else None,"second":ordered[1] if len(ordered)>1 else None,
            "g_th_gap_fraction":ordered[1]["g_th_cm"]/ordered[0]["g_th_cm"]-1 if len(ordered)>1 else None,
            "seed_source":str(seed_dir),"seed_sha256":hashlib.sha256((seed_dir/"complex_fields.npz").read_bytes()).hexdigest(),
            "scope":"Each sampled mode has its own fixed-period TE0/C(lambda)/finite-field real-frequency closure; not exhaustive fine-grid spectrum, independent Maxwell, dispersive-material or nonlinear single-mode proof."}
        json_write(output/"summary.json",summary);write_csv(output/"closed_modes.csv",[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in ordered])
        np.savez_compressed(output/"frequency_closed_fields.npz",**arrays)
        if ordered:
            fig,axes=plt.subplots(len(ordered),3,figsize=(11,3.3*len(ordered)),squeeze=False,constrained_layout=True)
            for index,r in enumerate(ordered):
                rank=r["seed_rank"];field=arrays[f"root_{rank}_fields"]
                axes[index,0].tripcolor(arrays["x_um"],arrays["y_um"],np.sum(abs(field)**2,axis=0),shading="gouraud",cmap="inferno")
                axes[index,0].set(aspect="equal",title=f"seed {rank}: {r['g_th_cm']:.3f} /cm",xlabel="x (um)",ylabel="y (um)")
                power=abs(arrays[f"root_{rank}_far_x"])**2+abs(arrays[f"root_{rank}_far_y"])**2;power/=power.max()
                axes[index,1].imshow(power,origin="lower",extent=(-1,1,-1,1),cmap="inferno")
                axes[index,1].set(xlim=(-.5,.5),ylim=(-.5,.5),title=f"lambda={r['wavelength_nm']:.6f} nm; ellipse={r['beam']['ellipticity']:.3f}",xlabel="theta x (deg)",ylabel="theta y (deg)")
                axes[index,2].plot([t["wavelength_nm"] for t in r["trace"]],[t["frequency_residual_per_m"] for t in r["trace"]],"o")
                axes[index,2].axhline(0,color="grey",ls=":");axes[index,2].set(xlabel="Wavelength (nm)",ylabel="Frequency residual (/m)",title=f"Overlap {r['reference_overlap']:.6f}")
            fig.savefig(output/"closed_low_poles.png",dpi=180);plt.close(fig)
        json_write(output/"run_status.json",{"status":"completed" if len(ordered)==args.roots else "completed_with_failed_roots","roots_closed":len(ordered),"roots_requested":args.roots})


if __name__=="__main__":main()
