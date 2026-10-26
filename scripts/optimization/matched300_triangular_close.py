"""Fixed-period C(lambda) closure of sampled three-hole finite lateral poles.

中文：固定三孔结构及晶格周期，以实际场重叠跟踪各有限腔极点，逐条求实频闭合。
闭合后的阈值、近场和 Jones 远场来自同一次实际求解；不以 Γ 点六模代替横向谱。
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"
import argparse
from dataclasses import asdict
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
from pcselsim.design_objectives import beam_metrics
from pcselsim.finite_discretization import relative_residual
from pcselsim.triangular_finite import TriangularFiniteSpec,TriangularFiniteMode,triangular_finite_eigenproblem,triangular_vector_far_field
from pcselsim.triangular_six_wave import TriangularLatticeCell,TriangularCWTSettings,build_triangular_coupling,plot_triangular_lattice
from pcselsim.vertical import Layer,LayerStack
from scripts._triangular_workflow import threshold_audit,aperture_area_m2
from scripts.optimization.triangular_semiconductor_search import parameters,json_write,write_csv,threadpool_limits
from scripts.optimization.close_triangular_finalist_frequency import background_delta
from scripts.optimization.matched300_triangular_search import ThreeHole,candidate,base_spec


def close(seed_dir,output,roots=6,order=10,dz=2.,inverse=False,separable=False,newton=False):
    """Close supplied branches; preserve failed attempts / 闭合输入分支并保留失败记录。

    Both the eigenpair and background_delta(lambda)-Re(pole) residuals must pass.
    Newton predicts a wavelength only; exact C(lambda) residuals and a Brent
    fallback decide acceptance. This is not an exhaustive fine-grid mode search.
    本征残差和实频失谐残差必须分别通过。Newton 只预测波长，最终仍用真实
    C(lambda) 残差和 Brent 回退判定；本函数不负责穷尽精细网格全部模式。
    """
    output.mkdir(parents=True,exist_ok=True)
    source=json.loads((seed_dir/"result.json").read_text(encoding="utf-8"))
    spec=ThreeHole(**source["three_hole_spec"]); n=source["N"]
    p=parameters(base_spec(spec)); holes=candidate(spec)
    cell=TriangularLatticeCell(p.BACKGROUND_INDEX**2,holes.inclusions)
    average=float(np.sqrt(cell.fourier_epsilon(0,0).real))
    stack=LayerStack(tuple(Layer(l.name,l.thickness_nm,average if l.name==p.PC_LAYER_NAME else l.refractive_index)
        for l in p.LAYERS),p.TOP_CLADDING_INDEX,p.BOTTOM_CLADDING_INDEX,p.VERTICAL_PADDING_UM)
    settings=TriangularCWTSettings(order,dz)
    finite_spec=TriangularFiniteSpec(150.,n,0.,24,"square","staggered")
    seed=np.load(seed_dir/"complex_fields.npz"); cache_models={}; records=[]; arrays={}
    def model_at(wl):
        if float(wl) not in cache_models:
            cache_models[float(wl)]=build_triangular_coupling(cell,stack,p.PC_LAYER_NAME,spec.lattice_nm,wl,settings,fixed_wavelength_nm=wl)
        return cache_models[float(wl)]
    with threadpool_limits(limits=1):
        for rank in range(min(roots,len(seed["eigenvalues"]))):
            start=time.perf_counter(); value0=seed["eigenvalues"][rank]; field0=seed["fields"][rank]
            vector0=field0.ravel()/np.linalg.norm(field0);reference_c=seed["coupling_m"];trace=[];cache={}
            def evaluate(wl):
                if float(wl) in cache:return cache[float(wl)]
                model=model_at(wl); problem,grid=triangular_finite_eigenproblem(model,finite_spec)
                correction=np.einsum("ik,ij,jk->",field0.conj(),model.coupling_m-reference_c,field0)/np.sum(abs(field0)**2)
                # The source may already be an exact pole at this reference.
                # Offset only the shift-invert solver target, never the matrix,
                # to avoid a singular factorization at the known eigenvalue.
                predictor=value0+correction+complex(50.,5.)
                for count in (8,24,48):
                    if inverse:
                        from scripts.optimization.matched300_triangular_actual_controls import follow
                        value,vector,amplitude_overlap,solver=follow(problem,predictor,vector0,count)
                        values=np.asarray([value]);fields=vector[:,None];index=0;overlap=amplitude_overlap**2
                    else:
                        values,fields=problem.near(predictor,count,vector0,tol=1e-10,maxiter=1600,dense_limit=0)
                        overlaps=abs(vector0.conj()@fields)**2;index=int(np.argmax(overlaps));overlap=float(overlaps[index]);solver={"solver":"ARPACK"}
                    if overlap>=.8:break
                if overlap<.8:raise RuntimeError(f"Lost finite pole {rank}: overlap={overlap}")
                value=values[index];field=fields[:,index].reshape(field0.shape)
                if value.imag< -1e-7:raise RuntimeError("Active pole in passive model")
                residual=background_delta(wl,model.effective_index,spec.lattice_nm)-value.real
                trace.append({"wavelength_nm":wl,"residual_per_m":float(residual),"overlap":overlap,"alpha_field_cm":float(value.imag/100),"solver_info":solver})
                cache[float(wl)]=(model,problem,grid,value,field,overlap,float(residual))
                return cache[float(wl)]
            try:
                guess=source["spectrum"][rank]["frozen_CWT_pole_wavelength_nm"]
                wl=None
                if newton:
                    trial=float(guess)
                    for iteration in range(6):
                        current=evaluate(trial);residual=current[-1]
                        if abs(residual)<=.005:
                            wl=trial;break
                        slope=2*np.pi*current[0].group_index*1e9/trial**2
                        increment=residual/slope
                        if abs(increment)>.5:break
                        trial+=increment
                if wl is None:
                    half=.5
                    while evaluate(guess-half)[-1]*evaluate(guess+half)[-1]>0:
                        half*=2
                        if half>32:raise RuntimeError("No real frequency closure within 32 nm")
                    wl=float(brentq(lambda x:evaluate(x)[-1],guess-half,guess+half,xtol=2e-8))
                model,problem,grid,value,field,overlap,residual=evaluate(wl)
                if abs(residual)>.05:raise RuntimeError("Frequency closure residual exceeds .05/m")
                gamma=model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME]
                audit=threshold_audit(p,float(value.imag),aperture_area_m2("square",150.),gamma,model.effective_index)
                rx,ry=model.radiation_amplitudes(field)
                mode=TriangularFiniteMode(f"closed_{rank}",float(value.real),float(value.imag),field,overlap,grid,rx,ry,discretization="staggered")
                if separable:
                    from scripts.optimization.matched300_triangular_actual_controls import separable_far_field
                    far=separable_far_field(mode,wl,1.,201)
                else:far=triangular_vector_far_field(mode,wl,1.,201)
                beam=beam_metrics(far.angle_deg,far.field_x,far.field_y,requested_view_deg=1.,source_nyquist_deg=far.alias_free_square_view_deg)
                budget=problem.loss_budget(field)
                # alpha is field loss /m; 2*alpha and internal 500 /m are power losses.
                # alpha 是振幅损耗 /m；2*alpha 与内部 500 /m 才能按功率损耗相加。
                row={"status":"completed","seed_rank":rank,"wavelength_nm":wl,"g_th_cm":audit["required_material_gain_cm-1"],
                    "I_th_A":audit["threshold_current_A"],"J_th_A_cm2":audit["threshold_current_density_A_cm-2"],
                    "alpha_field_cm":float(value.imag/100),"delta_per_m":float(value.real),"delta_origin":"direct_eigensolver_output",
                    "Gamma":gamma,"n_eff":model.effective_index,"beam":beam,"loss_budget":budget,
                    "eta_rad":2*budget["local_amplitude_loss_per_m"]/(2*value.imag+500),
                    "reference_overlap":overlap,"frequency_residual_per_m":residual,
                    "solver_shift_offset_per_m":[50.,5.],
                    "far_field_evaluator":"cartesian_separable_exact_point_sum" if separable else "direct_shared_point_sum",
                    "frequency_root_solver":"group_index_Newton_with_exact_C_lambda_residual_and_Brent_fallback" if newton else "bracketed_Brent",
                    "eigen_residual":relative_residual(problem.center_operator(),field,value),"trace":trace,
                    "elapsed_seconds":time.perf_counter()-start}
                try:
                    from pcselsim.polarization import polarization_metrics
                    row["polarization"]=polarization_metrics(far.angle_deg,far.field_x,far.field_y,core_radius_deg=.2)
                except ImportError:pass
                arrays.update({f"root_{rank}_fields":field,f"root_{rank}_far_x":far.field_x,f"root_{rank}_far_y":far.field_y,
                    f"root_{rank}_coupling_m":model.coupling_m,f"root_{rank}_radiation_x":rx,f"root_{rank}_radiation_y":ry,
                    f"root_{rank}_eigenvalue_per_m":np.asarray(value)})
                arrays.update(angle_deg=far.angle_deg,x_um=grid.x_um,y_um=grid.y_um)
                print("closed",rank,wl,row["g_th_cm"],beam["ellipticity"],flush=True)
            except Exception as error:
                row={"status":"failed","seed_rank":rank,"exception":repr(error),"trace":trace}
                print("failed",rank,repr(error),flush=True)
            records.append(row);json_write(output/"frequency_closed_poles.json",records)
    ordered=sorted((r for r in records if r["status"]=="completed"),key=lambda r:r["g_th_cm"])
    summary={"three_hole_spec":asdict(spec),"N":n,"D":order,"dz_nm":dz,"roots_closed":len(ordered),"roots_requested":roots,
        "lowest":ordered[0] if ordered else None,"second":ordered[1] if len(ordered)>1 else None,
        "g_th_gap_fraction":ordered[1]["g_th_cm"]/ordered[0]["g_th_cm"]-1 if len(ordered)>1 else None,
        "scope":"Individually frequency-closed sampled finite lateral poles; not exhaustive fine-grid or nonlinear single-mode proof."}
    json_write(output/"summary.json",summary);write_csv(output/"closed_modes.csv",ordered)
    np.savez_compressed(output/"frequency_closed_fields.npz",**arrays)
    plot_triangular_lattice(cell,output/"lattice.png",cells=3)
    if ordered:
        fig,axes=plt.subplots(len(ordered),3,figsize=(11,3.1*len(ordered)),squeeze=False,constrained_layout=True)
        for i,r in enumerate(ordered):
            rank=r["seed_rank"];field=arrays[f"root_{rank}_fields"]
            axes[i,0].tripcolor(arrays["x_um"],arrays["y_um"],np.sum(abs(field)**2,axis=0),shading="gouraud",cmap="inferno")
            axes[i,0].set(aspect="equal",title=f"Pole {rank}: g = {r['g_th_cm']:.3f} /cm",xlabel="x (um)",ylabel="y (um)")
            power=abs(arrays[f"root_{rank}_far_x"])**2+abs(arrays[f"root_{rank}_far_y"])**2;power/=power.max()
            axes[i,1].imshow(power,origin="lower",extent=(-1,1,-1,1),cmap="inferno")
            axes[i,1].set(xlim=(-.5,.5),ylim=(-.5,.5),title=f"Ellipticity {r['beam']['ellipticity']:.3f}; center {r['beam']['center_to_peak']:.3f}",xlabel="theta x (deg)",ylabel="theta y (deg)")
            axes[i,2].plot([t["wavelength_nm"] for t in r["trace"]],[t["residual_per_m"] for t in r["trace"]],"o")
            axes[i,2].axhline(0,color="gray",ls=":");axes[i,2].set(title=f"lambda = {r['wavelength_nm']:.5f} nm",xlabel="Wavelength (nm)",ylabel="Closure residual (/m)")
        fig.savefig(output/"closed_modes.png",dpi=180);plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--roots",type=int,default=6);p.add_argument("--order",type=int,default=10);p.add_argument("--dz",type=float,default=2.)
    p.add_argument("--inverse",action="store_true");p.add_argument("--separable",action="store_true")
    p.add_argument("--newton",action="store_true")
    p.add_argument("--promote-case",action="store_true",help="Also write the parent case summary with actual-target and source-pool provenance")
    a=p.parse_args();result=close(a.input,a.output,a.roots,a.order,a.dz,a.inverse,a.separable,a.newton)
    if a.promote_case:
        result.update(actual_target_wavelength_nm=950.,actual_target_tolerance_nm=.05,
            actual_target_pass=bool(result["lowest"] and abs(result["lowest"]["wavelength_nm"]-950)<=.05),
            finite_seed_path=str((a.input/"result.json").resolve().relative_to(ROOT)))
        json_write(a.output.parent/"summary.json",result)


if __name__=="__main__":main()
