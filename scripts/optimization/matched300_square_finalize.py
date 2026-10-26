"""Fixed-period real-frequency closure of saved actual low-loss lateral poles.

Unlike a four-band-edge closure, every input pole is continued by its physical
field overlap. A weak match is rejected rather than renamed as convergence.

中文：先确定实际有限腔首模所需周期，再固定几何逐条闭合 C(lambda)。
极点残差和实频失谐残差是两种独立检查；仅有极小本征残差并不说明频率已闭合。
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"
import argparse,json,sys,time,hashlib
from pathlib import Path
from dataclasses import asdict,replace
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pcselsim.custom_analysis import FourWaveOpticalSpec,LinearMode,finite_area_eigenproblem
from pcselsim.finite_discretization import relative_residual
from pcselsim.frequency_cwt import background_detuning
from pcselsim.polarization import polarization_metrics,stokes_maps
from pcselsim.mode_identity import independent_mode_indices
from pcselsim.fabrication import evaluate_fabrication,inclusion_boundary
from scripts.optimization.matched300_square_search import (Design,build_model,build_cell,save_json,
    threshold,direct_vector_far_field,BASE,RULES)


def tracked_near(problem,target,seed):
    """Avoid a numerically singular factorization at an already known pole."""
    failures=[]
    for offset in (10+10j,100+40j,-100+40j):
        try:
            values,fields=problem.near(target+offset,8,seed.ravel(),maxiter=1600,dense_limit=0)
            overlaps=abs(seed.ravel().conj()@fields)**2;index=int(np.argmax(overlaps))
            if overlaps[index] >= .65:return values[index],fields[:,index].reshape(seed.shape),float(overlaps[index])
            failures.append(f"overlap {overlaps[index]:g}")
        except (RuntimeError,np.linalg.LinAlgError) as error:failures.append(str(error))
    raise RuntimeError("No residual-verified physical branch match: "+"; ".join(failures))


def design_period(source,raw_path,target_nm=950.,order=16,dz=2.):
    """Design the *actual lowest reference pole* at target, then freeze a."""
    d=Design(**source["design"]);grid=source["grid"]
    with np.load(raw_path,allow_pickle=False) as saved:
        field=saved["pole_0_field"].copy();reference_C=saved["coupling_m"].copy()
    value0=source["ordered_poles"][0]["delta_per_m"]+100j*source["ordered_poles"][0]["alpha_field_cm1"]
    trace=[]
    def equation(period):
        candidate=replace(d,lattice_nm=float(period))
        model=build_model(candidate,order,dz,target_nm,False,96)[2]
        spec=FourWaveOpticalSpec(target_nm,model.effective_index,model.group_index,300.,5.,(0.,)*4,(0.,)*4,grid,"staggered",24,0,1600)
        problem=finite_area_eigenproblem(spec,model.coupling_m)
        flat=field.reshape(4,-1);shift=np.vdot(flat,(model.coupling_m-reference_C)@flat)/np.vdot(flat,flat)
        value,tracked,overlap=tracked_near(problem,value0+shift,field)
        residual=background_detuning(target_nm,model.effective_index,float(period))-value.real
        trace.append({"period_nm":float(period),"target_wavelength_nm":target_nm,
                      "residual_per_m":residual,"reference_overlap":overlap})
        return residual
    period=brentq(equation,d.lattice_nm*.95,d.lattice_nm*1.05,xtol=2e-8)
    residual=equation(period)
    if abs(residual)>.05:raise RuntimeError("Period design frequency residual exceeds .05/m")
    return replace(d,lattice_nm=float(period),tag=d.tag+"_actual950nm"),trace,residual


def close_saved(source,raw_path,output,order=16,dz=2.,count=10):
    """Continue saved poles at fixed period / 固定周期，按真实场重叠延续已保存极点。

    Each wavelength rebuilds the vertical mode and coupling matrix. The real
    equation is background_detuning(lambda)-Re(pole(lambda))=0, in m^-1.
    每个试探波长均重建纵模及耦合矩阵；闭合方程为背景失谐减极点实部，
    单位 m^-1。求解器只延续输入分支，并不保证找到完整精细网格谱。
    """
    started=time.monotonic();d=Design(**source["design"]);grid=source["grid"]
    reference=np.load(raw_path,allow_pickle=False);reference_C=reference["coupling_m"]
    cache={};records=[];arrays={};closure_failures=[];axis=np.linspace(-1.,1.,241)
    source_sha=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    progress=output/"closure_progress.json";progress_fields=output/"closure_progress_fields.npz"
    if progress.exists() and progress_fields.exists():
        saved=json.loads(progress.read_text(encoding="utf-8"))
        if saved.get("source_sha256")==source_sha and saved.get("D")==order and saved.get("dz_nm")==dz:
            records=saved["records"];closure_failures=saved.get("closure_failures",[])
            with np.load(progress_fields,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    def optical(w):
        key=float(w)
        if key not in cache:cache[key]=build_model(d,order,dz,w,False,96)[2]
        return cache[key]
    for index,seed in enumerate(source["ordered_poles"]):
        if len(records)>=count:break
        key=f"pole_{index}_field"
        if key not in reference:break
        if any(r["mode"]==f"pole_{index}" for r in records) or any(r["reference_rank"]==index for r in closure_failures):continue
        seedfield=reference[key];seedflat=seedfield.reshape(4,-1)
        seedvalue=seed["delta_per_m"]+100j*seed["alpha_field_cm1"];trace=[];evaluations={}
        def equation(w):
            if float(w) not in evaluations:
                m=optical(w)
                spec=FourWaveOpticalSpec(w,m.effective_index,m.group_index,300.,5.,(0.,)*4,(0.,)*4,grid,"staggered",24,0,1600)
                problem=finite_area_eigenproblem(spec,m.coupling_m)
                shift=np.vdot(seedflat,(m.coupling_m-reference_C)@seedflat)/np.vdot(seedflat,seedflat)
                if abs(w-950.)<1e-12 and np.allclose(m.coupling_m,reference_C,rtol=1e-13,atol=1e-8):
                    value,field,overlap=seedvalue,seedfield.copy(),1.
                else:
                    value,field,overlap=tracked_near(problem,seedvalue+shift,seedfield)
                residual=background_detuning(w,m.effective_index,d.lattice_nm)-value.real
                evaluations[float(w)]=(residual,m,spec,problem,value,field,overlap)
                trace.append({"wavelength_nm":w,"frequency_residual_per_m":residual,"overlap":overlap})
            return evaluations[float(w)][0]
        # Start with the exact background derivative, then use actual rebuilt
        # C(lambda) / finite fields in a bounded secant. Every accepted root is
        # directly residual-checked; a bracket remains the bounded fallback.
        def find_root():
            residual0=equation(950.)
            beta_scale=2*np.pi*optical(950.).group_index/(950e-9)**2*1e-9
            estimate=950.+residual0/beta_scale
            root=None
            if abs(residual0)<.005:root=950.
            else:
                x0,f0,x1=950.,residual0,float(estimate)
                for _ in range(6):
                    if not 930.<x1<970.:break
                    f1=equation(x1)
                    if abs(f1)<.005:root=x1;break
                    if abs(f1-f0)<1e-12:break
                    x0,f0,x1=x1,f1,x1-f1*(x1-x0)/(f1-f0)
            if root is None:
                left,right=min(950.,estimate)-1.,max(950.,estimate)+1.
                for _ in range(6):
                    if equation(left)*equation(right)<=0:break
                    left-=2.;right+=2.
                root=brentq(equation,left,right,xtol=2e-8,rtol=1e-12)
            equation(root)
            return root
        try:root=find_root()
        except (RuntimeError,np.linalg.LinAlgError,ValueError) as error:
            closure_failures.append({"reference_rank":index,"mode":f"pole_{index}","error":str(error),"frequency_trace":trace})
            if index<2:raise
            print(f"unresolved higher branch {index}: {error}",flush=True)
            continue
        residual,m,spec,problem,value,field,overlap=evaluations[float(root)]
        if abs(residual)>.05:raise RuntimeError("Closed frequency residual exceeds .05/m")
        rx,ry=m.radiation_fields(field);mode=LinearMode(f"pole_{index}",float(value.real),float(value.imag),field,0.,rx,ry,discretization="staggered")
        axis,fx,fy,beam=direct_vector_far_field(mode,root,300.,241)
        gamma=m.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME];budget=problem.loss_budget(field)
        row={"mode":mode.name,"reference_rank":index,"wavelength_nm":root,"delta_per_m":value.real,
            "alpha_field_cm1":value.imag/100,"effective_index":m.effective_index,"Gamma":gamma,
            **threshold(value.imag,gamma,m.effective_index,300.),"beam":beam,
            "eigen_residual":relative_residual(problem.center_operator(),field,value),
            "frequency_residual_per_m":residual,"reference_overlap":overlap,"frequency_trace":trace,
            "radiative_extraction_fraction":2*budget["local_amplitude_loss_per_m"]/(2*value.imag+500),
            "polarization":polarization_metrics(axis,fx,fy,core_radius_deg=.2)}
        records.append(row)
        arrays.update({mode.name+"_field":field,mode.name+"_far_x":fx,mode.name+"_far_y":fy,
                       mode.name+"_radiation_x":rx,mode.name+"_radiation_y":ry,
                       mode.name+"_coupling_m":m.coupling_m})
        output.mkdir(parents=True,exist_ok=True)
        save_json(progress,{"source_sha256":source_sha,"D":order,"dz_nm":dz,"records":records,"closure_failures":closure_failures})
        np.savez_compressed(progress_fields,**arrays)
        print(f"closed {index+1}/{count} lambda={root:.9f} g={row['g_th_cm1']:.5f} ellipse={beam['ellipticity']:.5f}",flush=True)
    # Eigenvector span comparisons are meaningful for the same operator.
    # Group only numerically equal closed wavelengths, then deduplicate within
    # numerical complex-eigenvalue clusters without removing true degeneracy.
    groups=[]
    # Same-frequency operators only; distinct resolved complex poles stay distinct.
    # 只对同频算符做子空间去重；非正交但已分辨的不同复数极点仍需保留。
    for index,row in enumerate(records):
        group=next((g for g in groups if abs(records[g[0]]["wavelength_nm"]-row["wavelength_nm"])<2e-7),None)
        if group is None:groups.append([index])
        else:group.append(index)
    accepted=[];independence=[]
    for group in groups:
        values=np.array([records[i]["delta_per_m"]+100j*records[i]["alpha_field_cm1"] for i in group])
        fields=np.stack([arrays[records[i]["mode"]+"_field"] for i in group])
        indices=independent_mode_indices(values,fields)
        selected=[group[i] for i in indices];accepted.extend(selected)
        independence.append({"closed_wavelength_nm":records[group[0]]["wavelength_nm"],"input_modes":[records[i]["mode"] for i in group],
            "independent_modes":[records[i]["mode"] for i in selected]})
    ranked=sorted([records[i] for i in accepted],key=lambda r:r["g_th_cm1"])
    for row in ranked:
        name=row["mode"];maps=stokes_maps(arrays[name+"_far_x"],arrays[name+"_far_y"])
        arrays.update({name+"_stokes_"+k:v for k,v in maps.items()})
    # Retain the winner plus three nearest competitors as actual Jones/near
    # fields. Polarization maps are only needed for the winner; every closed
    # pole retains its complete numerical metrics and continuation trace.
    kept={r["mode"] for r in ranked[:4]};winner=ranked[0]["mode"]
    arrays={k:v for k,v in arrays.items() if (k.endswith("_field") or any(k.startswith(n+"_") for n in kept))
            and ("_stokes_" not in k or k.startswith(winner+"_"))}
    result={"status":"completed","design":asdict(d),"id":d.identifier,"D":order,"dz_nm":dz,"grid":grid,
        "frequency_closed_modes":ranked,"lowest":ranked[0],"relative_material_gap":ranked[1]["g_th_cm1"]/ranked[0]["g_th_cm1"]-1,
        "requested_frequency_closed_count":count,"closure_failures":closure_failures,
        "fabrication":evaluate_fabrication(build_cell(d,96).inclusions,d.lattice_nm,rules=RULES),
        "saved_field_modes":[r["mode"] for r in ranked],"saved_jones_modes":sorted(kept),"saved_stokes_map_mode":winner,
        "elapsed_seconds":time.monotonic()-started,"source_lateral_survey":source,
        "scope":"fixed-period real-frequency continuation of the lowest saved lateral poles; sampled cold scalar CWT, not a nonlinear or full-wave single-mode proof"}
    output.mkdir(parents=True,exist_ok=True);save_json(output/"summary.json",result)
    save_json(output/"mode_independence_audit.json",{"frequency_group_tolerance_nm":2e-7,"input_count":len(records),
        "independent_count":len(ranked),"groups":independence,"scope":"only equal-frequency operator groups use SVD span tests"})
    np.savez_compressed(output/"final_fields.npz",axis_deg=axis,**arrays)
    plot_final(result,arrays,axis,output)
    return result


def plot_final(result,arrays,axis,output):
    d=Design(**result["design"]);row=result["lowest"];name=row["mode"]
    field=arrays[name+"_field"];fx=arrays[name+"_far_x"];fy=arrays[name+"_far_y"]
    total=abs(fx)**2+abs(fy)**2;scale=total.max();extent=(axis[0],axis[-1],axis[0],axis[-1])
    maps=stokes_maps(fx,fy);fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for h in build_cell(d,96).inclusions:
        points=inclusion_boundary(h,np.eye(2))
        for i in (-1,0,1):
            for j in (-1,0,1):ax[0,0].fill((points[:,0]+i)*d.lattice_nm,(points[:,1]+j)*d.lattice_nm,fc="#d6eef4",ec="black",lw=.7)
    a=d.lattice_nm;ax[0,0].plot(np.array([-.5,.5,.5,-.5,-.5])*a,np.array([-.5,-.5,.5,.5,-.5])*a,"r--",lw=1.)
    ax[0,0].set(xlim=(-.65*a,.65*a),ylim=(-.65*a,.65*a),aspect="equal",xlabel="x (nm)",ylabel="y (nm)",title=f"Central main + diagonal satellite; a={a:.3f} nm")
    ax[0,1].imshow(np.sum(abs(field)**2,axis=0),origin="lower",extent=(-150,150,-150,150),cmap="viridis")
    ax[0,1].set(xlabel="x (um)",ylabel="y (um)",title="Actual lowest-pole envelope intensity")
    plotdata=((0,2,total/scale,"Total far-field intensity", "hot",0,1),
              (1,0,abs(fx)**2/scale,"x-polarized intensity (common scale)","hot",0,1),
              (1,1,abs(fy)**2/scale,"y-polarized intensity (common scale)","hot",0,1),
              (1,2,np.ma.masked_where(~maps["valid_mask"],maps["s3_over_s0"]),"Signed circular Stokes S3 / S0","coolwarm",-1,1))
    for i,j,data,title,cmap,lo,hi in plotdata:
        im=ax[i,j].imshow(data,origin="lower",extent=extent,cmap=cmap,vmin=lo,vmax=hi)
        ax[i,j].set(xlim=(-.35,.35),ylim=(-.35,.35),xlabel="theta x (deg)",ylabel="theta y (deg)",title=title)
        fig.colorbar(im,ax=ax[i,j],shrink=.75)
    pol=row["polarization"]["on_axis"]
    fig.suptitle(f"300 x 300 um square PCSEL | gth={row['g_th_cm1']:.3f} cm^-1 | lambda={row['wavelength_nm']:.4f} nm\n"
        f"ellipse={row['beam']['ellipticity']:.4f}; center/peak={row['beam']['center_to_peak']:.4f}; checked gap={100*result['relative_material_gap']:.2f}%")
    fig.savefig(output/"final_overview.png",dpi=190);plt.close(fig)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("case",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--D",type=int,default=16)
    p.add_argument("--dz",type=float,default=2.);p.add_argument("--count",type=int,default=10)
    p.add_argument("--target",type=float)
    args=p.parse_args(argv);source=json.loads((args.case/"spectral_challenge.json").read_text(encoding="utf-8"))
    raw=args.case/"lateral_fields.npz"
    if args.target:
        from scripts.optimization.matched300_square_validate import spectral_challenge
        d,trace,residual=design_period(source,raw,args.target,args.D,args.dz)
        args.output.mkdir(parents=True,exist_ok=True)
        save_json(args.output/"period_design.json",{"design":asdict(d),"target_nm":args.target,"trace":trace,"residual_per_m":residual})
        extras=[complex(x[0],x[1]) for x in source.get("extra_coarse_QZ_targets",[])]
        extras.extend(complex(x["delta_per_m"],100*x["alpha_field_cm1"]) for x in source["ordered_poles"])
        source=spectral_challenge(d,args.D,args.dz,source["grid"],False,24,24,args.output/"period_lateral",vertices=96,extra_targets=extras)
        raw=args.output/"period_lateral/lateral_fields.npz"
    close_saved(source,raw,args.output,args.D,args.dz,args.count)

if __name__=="__main__":main()
