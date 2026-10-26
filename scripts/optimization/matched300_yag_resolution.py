"""Fixed-period D/N/contour continuation audit of actual YAG pump competitors.

Every tracked pole is independently frequency-closed using original pencils.
This is convergence of supplied branches; it does not replace a fresh census.
"""
from matched300_yag_search import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("case",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--modes",type=int,default=3)
    p.add_argument("--orders",type=int,nargs="+",default=[24,32,48])
    p.add_argument("--grids",type=int,nargs="+");p.add_argument("--contours",type=int,nargs="+",default=[0])
    p.add_argument("--live",action="store_true",help="Audit supplied closed branches before the shared-pump final completes; no first-mode claim")
    a=p.parse_args();rows=json.loads((a.case/"closed_optical_modes.json").read_text())
    if a.live:
        plan=json.loads((a.case/"plan.json").read_text());outer=json.loads((a.case/"outer_period_design.json").read_text())
        summary=dict(lattice=plan["record"]["lattice"],design=plan["record"]["design"],period_nm=outer["frozen_period_nm"],
            settings=dict(order=plan["order"],dz_nm=plan["dz_nm"],square_grid=plan["grid"],tri_radius_cells=plan["grid"]),
            pump_results=[dict(threshold=dict(branch_thresholds_internal_W={r["name"]:r["required_material_gain_cm1"] for r in rows}))],actual_winners={})
    else:summary=json.loads((a.case/"summary.json").read_text())
    lattice,design,period=summary["lattice"],summary["design"],summary["period_nm"]
    settings=summary["settings"];old_grid=settings["square_grid"] if lattice=="square" else settings["tri_radius_cells"]
    thresholds=summary["pump_results"][0]["threshold"]["branch_thresholds_internal_W"]
    chosen=sorted((r for r in rows if thresholds.get(r["name"]) is not None),key=lambda r:thresholds[r["name"]])[:a.modes]
    for actual in summary["actual_winners"].values():
        for row in actual:
            if row["name"] not in {r["name"] for r in chosen}:chosen.append(row)
    saved=np.load(a.case/"closed_fields.npz") if (a.case/"closed_fields.npz").exists() and not a.live else None
    references=[]
    for row in chosen:
        model,_,_=optical(lattice,design,settings["order"],settings["dz_nm"],row["wavelength_nm"],period,row["closed_TE_family"])
        spec,_,problem,mesh=finite(lattice,model,old_grid,8,False)
        if saved is not None:field=saved[row["name"]+"_field"].copy()
        else:
            target=complex(row["delta_per_m"],row["alpha_field_per_m"])
            vv,ff=problem.near(target+complex(.5,.7),8,np.ones(problem.stiffness.shape[0],complex),tol=1e-10,maxiter=500,dense_limit=0)
            chosen_source=int(np.argmin(abs(vv-target)))
            shape=(4,old_grid,old_grid) if lattice=="square" else (6,mesh.points)
            field=ff[:,chosen_source].reshape(shape)
        references.append((row,model,spec,mesh,field))
    tasks=[(D,old_grid,0) for D in a.orders]
    tasks += [(settings["order"],N,0) for N in (a.grids or []) if N!=old_grid]
    tasks += [(settings["order"],old_grid,n) for n in a.contours if n]
    results=[];a.output.mkdir(parents=True,exist_ok=True)
    for order,grid,contour in tasks:
        t=time.monotonic();d={**design}
        if contour:d["_yag_triangle_samples_per_corner"]=contour
        closed=[];models=[];names=[];failures=[]
        for row,reference,source_spec,source_mesh,source_field in references:
            try:
                model,_,guide=optical(lattice,d,order,settings["dz_nm"],row["wavelength_nm"],period,row["closed_TE_family"])
                spec,_,problem,mesh=finite(lattice,model,grid,8,False)
                field=interpolate_envelope(lattice,source_field,source_spec,source_mesh,spec,mesh)
                flat=field.reshape(len(model.result.coupling_m),-1)
                change=model.result.coupling_m-reference.result.coupling_m
                predictor=complex(row["delta_per_m"],row["alpha_field_per_m"])+np.einsum("ik,ij,jk->",flat.conj(),change,flat)/np.sum(abs(flat)**2)
                value,field,initialization=follow_reference_pole(problem,field,predictor)
                w,opt,g,local_spec,mode,trace=close_seed(lattice,d,period,model,(value,field),spec,mesh,order,settings["dz_nm"],row["closed_TE_family"])
                mode=replace(mode,name=row["name"])
                result,data=mode_row(lattice,opt,g,local_spec,mode,w,201)
                _,_,_,widths,qe,qg=data
                lateral=lateral_energy_weights(mode.fields,31) if lattice=="square" else binned_lateral_weights(mode.fields,mode.grid.x_um,300.,31)
                up,down=opt.side_power_losses(mode.fields)
                models.append(SpatialYbYAGModel(material(w),widths,qe,2*mode.alpha_per_m+10.,up,down,c/opt.result.group_index,qg,lateral));names.append(mode.name)
                result.update(wavelength_nm=w,closed_TE_family=row["closed_TE_family"],frequency_trace=trace,initialization=initialization)
                closed.append(result)
            except Exception as error:failures.append(dict(source_mode=row["name"],error=repr(error)))
        pumps=[]
        if models:
            system=SpatialMultimodeYbYAG(tuple(models),tuple(names))
            for arch in ("edge_dual","top_single"):
                threshold=shared_inversion_pump_threshold(system,PumpGeometryConfig(arch,0.,300.,31))
                pumps.append(dict(architecture=arch,threshold=asdict(threshold)))
        result=dict(order=order,grid=grid,contour_samples_per_corner=contour or design.get("_yag_triangle_samples_per_corner",9),
            fixed_period_nm=period,closed_modes=closed,failures=failures,pumps=pumps,seconds=time.monotonic()-t,
            scope="Convergence of actual first pump competitors; not a replacement complete spectrum")
        results.append(result);save(a.output/"results.json",results)
        print("resolution",lattice,order,grid,contour,[(r["name"],r["required_material_gain_cm1"],r["beam"]["ellipticity"]) for r in closed],"fails",len(failures),flush=True)
    save(a.output/"summary.json",dict(source=str(a.case),source_modes=[r["name"] for r in chosen],results=results))


if __name__=="__main__":main()
