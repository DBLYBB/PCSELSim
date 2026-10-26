"""Full coarse QZ poles guide complex shifts and interpolated fields on fine grids.

The reference matrix is rebuilt at the actual lowest closed-mode wavelength,
not assumed to be the unperturbed Bragg frequency.  No selected-family gap is
used as a substitute for this finite-pole audit.

中文：粗网格用完整广义谱产生复数移位和初始场，精细网格重新求解原算符。
插值场只作初值，不是精细结果；完整“粗”谱不构成完整“精细”谱的证明。
"""
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
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.linalg import eig as dense_standard_eig
from scipy.sparse.linalg import splu
from pcselsim.design_objectives import beam_metrics
from pcselsim.finite_discretization import relative_residual
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.triangular_finite import TriangularFiniteSpec,TriangularFiniteMode,triangular_finite_eigenproblem,triangular_vector_far_field
from pcselsim.triangular_six_wave import TriangularLatticeCell,TriangularCWTSettings,build_triangular_coupling
from pcselsim.vertical import Layer,LayerStack
from scripts._triangular_workflow import threshold_audit,aperture_area_m2
from scripts.optimization.triangular_semiconductor_search import parameters,json_write,write_csv,threadpool_limits
from scripts.optimization.matched300_triangular_search import ThreeHole,candidate,base_spec
from scripts.optimization.close_triangular_finalist_frequency import background_delta


def model_for(source,order=32,dz=1.,inclusions=None):
    """Rebuild at the saved actual closed wavelength / 在已保存的真实闭合波长重建模型。"""
    spec=ThreeHole(**source["three_hole_spec"]);p=parameters(base_spec(spec))
    cell=TriangularLatticeCell(p.BACKGROUND_INDEX**2,candidate(spec).inclusions if inclusions is None else inclusions)
    average=float(np.sqrt(cell.fourier_epsilon(0,0).real))
    stack=LayerStack(tuple(Layer(l.name,l.thickness_nm,average if l.name==p.PC_LAYER_NAME else l.refractive_index)
        for l in p.LAYERS),p.TOP_CLADDING_INDEX,p.BOTTOM_CLADDING_INDEX,p.VERTICAL_PADDING_UM)
    wavelength=source["lowest"]["wavelength_nm"]
    model=build_triangular_coupling(cell,stack,p.PC_LAYER_NAME,spec.lattice_nm,wavelength,
        TriangularCWTSettings(order,dz),fixed_wavelength_nm=wavelength)
    return spec,p,model


def scalar_record(value,field,model,p,problem,grid,reference_wavelength,index):
    """Record a frozen-operator pole and wavelength predictor / 记录冻结算符极点及波长估计。

    frozen_CWT_pole_wavelength_nm uses a group-index linearization. It is a
    continuation seed, not the result of rebuilding and closing C(lambda).
    frozen_CWT_pole_wavelength_nm 来自群折射率线性估计，只用于后续延续初值，
    不能把它标为已经重建 C(lambda) 的实频闭合结果。
    """
    gamma=model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME]
    audit=threshold_audit(p,float(value.imag),aperture_area_m2("square",150.),gamma,model.effective_index)
    budget=problem.loss_budget(field)
    delta_ref=background_delta(reference_wavelength,model.effective_index,model.lattice_constant_nm)
    guessed=1/(1/reference_wavelength+(value.real-delta_ref)*1e-9/(2*np.pi*model.group_index))
    return {"spectral_index":index,"delta_per_m":float(value.real),"alpha_per_m":float(value.imag),
        "g_th_cm":audit["required_material_gain_cm-1"],"I_th_A":audit["threshold_current_A"],
        "frozen_CWT_pole_wavelength_nm":float(guessed),"reference_wavelength_nm":reference_wavelength,
        "relative_eigen_residual":relative_residual(problem.center_operator(),field,value),
        "eta_rad":2*budget["local_amplitude_loss_per_m"]/(2*value.imag+500),**budget}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=("coarse","fine"),required=True)
    p.add_argument("--case",type=Path,required=True);p.add_argument("--input",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cells",type=int,default=13);p.add_argument("--order",type=int,default=32);p.add_argument("--dz",type=float,default=1.)
    p.add_argument("--poles",type=int,default=40);p.add_argument("--eigen-count",type=int,default=24)
    p.add_argument("--imag-scales",type=float,nargs="+",default=[1.,.5]);p.add_argument("--start",type=int,default=0)
    p.add_argument("--coarse-solver",choices=("qz","qr"),default="qz")
    p.add_argument("--retain-fields",type=int,default=64)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    source=json.loads((a.case/"summary.json").read_text(encoding="utf-8"));start=time.perf_counter()
    with threadpool_limits(limits=1):
        spec,params,model=model_for(source,a.order,a.dz)
        finite=TriangularFiniteSpec(150.,a.cells,0.,a.eigen_count,"square","staggered")
        problem,grid=triangular_finite_eigenproblem(model,finite)
        ref=model.bragg_wavelength_nm
        json_write(a.output/"run_plan.json",{"stage":a.stage,"three_hole_spec":asdict(spec),"N":a.cells,"D":a.order,"dz_nm":a.dz,
            "reference_wavelength_nm":ref,"matrix_dimension":problem.stiffness.shape[0],"coarse_targets":a.poles,
            "imag_scales":a.imag_scales,"coarse_solver":a.coarse_solver,"scope":"Actual-period finite operator at the same closed reference wavelength; full coarse spectrum followed by fine complex-shift challenges. Not a complete fine spectrum."})
        if a.stage=="coarse":
            initial=np.ones(problem.stiffness.shape[0],complex)
            print("Starting full coarse",a.coarse_solver,a.cells,"dimension",len(initial),flush=True)
            if a.coarse_solver=="qz":
                problem.near(complex(model.eigenvalues_m[0]),a.eigen_count,initial,dense_limit=len(initial))
                values,fields=problem._dense_spectrum
                filter_counts={"matrix_dimension":len(initial),"raw_eigenpairs":len(initial),
                    "finite_original_residual_verified":len(values),
                    "nonfinite_or_original_residual_rejected":len(initial)-len(values)}
            else:
                # M is invertible. Work on the identical coefficient-field
                # operator H=K M^-1=C+i D M^-1, not an eigensolver's Hermitian-M
                # interface. Every root is rechecked against the original K,M.
                inverse_mass=splu(problem.mass).solve(np.eye(len(initial),dtype=complex))
                operator_dense=problem.local.toarray()+1j*(problem.transport@inverse_mass)
                values,fields=dense_standard_eig(operator_dense,overwrite_a=True,check_finite=False)
                del inverse_mass,operator_dense
                fields/=np.linalg.norm(fields,axis=0)[None,:]
                operator=problem.center_operator()
                valid=np.array([np.isfinite(v) and relative_residual(operator,f,v)<=2e-6 for v,f in zip(values,fields.T)])
                filter_counts={"matrix_dimension":len(initial),"raw_eigenpairs":len(values),
                    "nonfinite_eigenvalue_rejected":int(np.count_nonzero(~np.isfinite(values))),
                    "finite_original_residual_verified":int(np.count_nonzero(valid)),
                    "finite_original_residual_rejected":int(np.count_nonzero(np.isfinite(values)&~valid))}
                values,fields=values[valid],fields[:,valid]
            mask=values.imag>=-1e-7
            filter_counts["negative_alpha_below_minus_1e_minus7_rejected"]=int(np.count_nonzero(~mask))
            filter_counts["retained_passive_poles"]=int(np.count_nonzero(mask))
            values=values[mask];fields=fields[:,mask]
            order=np.argsort(values.imag,kind="stable");values=values[order];fields=fields[:,order]
            records=[]
            for i,value in enumerate(values):
                # Save the complete scalar spectrum, while retaining only the
                # first 64 interpolatable complex fields.
                if i<a.retain_fields:r=scalar_record(value,fields[:,i],model,params,problem,grid,ref,i)
                else:r={"spectral_index":i,"delta_per_m":float(value.real),"alpha_per_m":float(value.imag)}
                records.append(r)
            np.savez_compressed(a.output/"qz_low_fields.npz",eigenvalues=values,fields=fields[:,:a.retain_fields].T.reshape(-1,6,grid.points),
                x_um=grid.x_um,y_um=grid.y_um,coupling_m=model.coupling_m)
            json_write(a.output/"complete_coarse_poles.json",records);write_csv(a.output/"complete_coarse_poles.csv",records)
            json_write(a.output/"summary.json",{"N":a.cells,"retained_passive_poles":len(values),"matrix_dimension":len(initial),
                "lowest_64":records[:64],"coarse_solver":a.coarse_solver,"elapsed_seconds":time.perf_counter()-start})
            json_write(a.output/"filter_counts.json",filter_counts)
            print("Full spectrum completed",a.coarse_solver,len(values),"lowest",records[0],"seconds",time.perf_counter()-start,flush=True)
            return
        if not a.input:raise ValueError("fine stage requires --input coarse QZ directory")
        coarse=np.load(a.input/"qz_low_fields.npz")
        if np.linalg.norm(coarse["coupling_m"]-model.coupling_m)>1e-10*max(1.,np.linalg.norm(model.coupling_m)):
            raise ValueError("Coarse and fine reference coupling matrices differ")
        triangle=Delaunay(np.column_stack((coarse["x_um"],coarse["y_um"])))
        targets=[]
        for index,value in enumerate(coarse["eigenvalues"][:a.poles]):
            for scale in a.imag_scales:
                target=complex(value.real,value.imag*scale)
                # A 24-pole Arnoldi window resolves neighbouring coarse poles;
                # retain exact target provenance while avoiding numerical duplicates.
                if any(abs(target-old[1])<1e-7*max(abs(target),1.) for old in targets):continue
                targets.append((index,target,scale))
        json_write(a.output/"targets.json",[{"coarse_rank":i,"real_per_m":float(t.real),"imag_per_m":float(t.imag),"imag_scale":sc} for i,t,sc in targets])
        found_values=[];found_fields=[];traces=[]
        def retain(value,field):
            """Add a resolved complex pole or new span / 保留不同复极点或同极点的独立场。"""
            if not independent_eigenmode(value,field,found_values,found_fields):return
            found_values.append(value);found_fields.append(field)
        for index,(rank,target,scale) in enumerate(targets[a.start:],a.start):
            t=time.perf_counter()
            interpolation=LinearNDInterpolator(triangle,coarse["fields"][rank].T,fill_value=0.)
            # Interpolation seeds only: the residual-verified fine eigenvector is saved.
            # 插值仅生成初始场；保存的是随后通过残差检查的精细网格本征矢量。
            initial=interpolation(np.column_stack((grid.x_um,grid.y_um))).T.ravel()
            initial=initial/np.linalg.norm(initial)
            rng=np.random.default_rng(20261021+index)
            initial+=.01*(rng.normal(size=initial.size)+1j*rng.normal(size=initial.size))/np.sqrt(initial.size)
            try:
                values,fields=problem.near(target,a.eigen_count,initial,tol=1e-10,maxiter=1800,dense_limit=0)
                for value,field in zip(values,fields.T):
                    if value.imag>=-1e-7:retain(value,field)
                trace={"index":index,"coarse_rank":rank,"target_real_per_m":target.real,"target_imag_per_m":target.imag,
                    "imag_scale":scale,"returned":len(values),"status":"completed","elapsed_seconds":time.perf_counter()-t}
            except Exception as error:trace={"index":index,"status":"failed","exception":repr(error),"elapsed_seconds":time.perf_counter()-t}
            traces.append(trace);json_write(a.output/"progress.json",traces)
            print("QZ guided",index+1,len(targets),"coarse",rank,"scale",scale,"found",len(found_values),"sec",trace["elapsed_seconds"],flush=True)
            if found_values:
                order=np.argsort(np.imag(found_values),kind="stable")
                best=np.asarray(found_values)[order][:12]
                json_write(a.output/"lowest_progress.json",[{"delta_per_m":float(v.real),"alpha_per_m":float(v.imag)} for v in best])
        order=np.argsort(np.imag(found_values),kind="stable");values=np.asarray(found_values)[order];fields=np.asarray(found_fields)[order].reshape(-1,6,grid.points)
        records=[scalar_record(v,f,model,params,problem,grid,ref,i) for i,(v,f) in enumerate(zip(values,fields))]
        for i in range(min(12,len(values))):
            rx,ry=model.radiation_amplitudes(fields[i]);mode=TriangularFiniteMode(f"qz_guided_{i}",float(values[i].real),float(values[i].imag),fields[i],1.,grid,rx,ry,discretization="staggered")
            far=triangular_vector_far_field(mode,records[i]["frozen_CWT_pole_wavelength_nm"],1.,161)
            records[i]["beam"]=beam_metrics(far.angle_deg,far.field_x,far.field_y,requested_view_deg=1.,source_nyquist_deg=far.alias_free_square_view_deg)
        row={"three_hole_spec":asdict(spec),"N":a.cells,"D":a.order,"dz_nm":a.dz,"reference_wavelength_nm":ref,
            "spectrum":records,"g_th_cm":records[0]["g_th_cm"],"g_th_gap_fraction":records[1]["g_th_cm"]/records[0]["g_th_cm"]-1,
            "targets":len(targets),"retained_poles":len(records),"lowest":records[0],"second":records[1],"elapsed_seconds":time.perf_counter()-start}
        json_write(a.output/"result.json",row);write_csv(a.output/"all_poles.csv",records)
        np.savez_compressed(a.output/"complex_fields.npz",eigenvalues=values[:24],fields=fields[:24],x_um=grid.x_um,y_um=grid.y_um,coupling_m=model.coupling_m)


if __name__=="__main__":main()
