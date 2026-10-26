"""Physical nm process perturbations of final closed YAG modes.

Continue the twelve lowest nominal closed competitors on the SAME final grid,
period and layer thicknesses. This is a sampled periodic-motif sensitivity
screen, not random-device disorder or a guaranteed manufacturing yield.
"""
from matched300_yag_search import *
from pcselsim.fabrication import perturb_triangular_inclusions


def encode(holes,lattice):
    return [dict(type=("square_ellipse" if isinstance(h,Ellipse) else "square_polygon") if lattice=="square" else
        ("triangular_ellipse" if isinstance(h,TriangularEllipse) else "triangular_polygon"),parameters=asdict(h)) for h in holes]


def transformed_holes(cell,lattice,period,biases,shifts,angles):
    out=[];direct=np.eye(2) if lattice=="square" else TRIANGULAR_DIRECT
    for hole,bias,shift,angle in zip(cell.inclusions,biases,shifts,angles):
        if isinstance(hole,(Ellipse,TriangularEllipse)):
            radii=np.asarray(hole.radii if isinstance(hole,Ellipse) else hole.radii_over_a)+bias/period
            if np.min(radii)<=0:raise ValueError("Etch eliminates hole")
            if isinstance(hole,Ellipse):
                out.append(replace(hole,radii=tuple(radii),center=tuple(np.asarray(hole.center)+np.asarray(shift)/period),angle_deg=hole.angle_deg+angle))
            else:
                center=TRIANGULAR_DIRECT@np.asarray(hole.center_fractional)+np.asarray(shift)/period
                out.append(replace(hole,radii_over_a=tuple(radii),center_fractional=tuple(np.linalg.solve(TRIANGULAR_DIRECT,center)),angle_deg=hole.angle_deg+angle))
        else:
            points=inclusion_boundary(hole,direct)
            # Reuse the tested half-plane normal-offset routine via a pure
            # coordinate conversion; it does not change the Bravais lattice.
            tri=TriangularPolygon(tuple(map(tuple,np.linalg.solve(TRIANGULAR_DIRECT,points.T).T)),epsilon=hole.epsilon)
            changed=perturb_triangular_inclusions((tri,),period,np.random.default_rng(1030),etch_bias_nm=bias)[0]
            points=inclusion_boundary(changed,TRIANGULAR_DIRECT)
            x,y=points.T;cross=x*np.roll(y,-1)-np.roll(x,-1)*y
            center=np.array([np.sum((x+np.roll(x,-1))*cross),np.sum((y+np.roll(y,-1))*cross)])/(3*np.sum(cross))
            theta=np.deg2rad(angle);R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            points=(points-center)@R.T+center+np.asarray(shift)/period
            out.append(PolygonInclusion(tuple(map(tuple,points)),epsilon=hole.epsilon) if lattice=="square" else
                       TriangularPolygon(tuple(map(tuple,np.linalg.solve(TRIANGULAR_DIRECT,points.T).T)),epsilon=hole.epsilon))
    return tuple(out)


def variants(cell,lattice,period):
    n=len(cell.inclusions);zero=np.zeros(n);shifts=np.zeros((n,2))
    yield "nominal",cell.inclusions,dict(radial_bias_nm=zero,cartesian_shift_nm=shifts,rotation_deg=zero)
    for bias in (-3.,3.):
        for which in ("all","main","satellites"):
            b=np.full(n,bias) if which=="all" else np.r_[bias,np.zeros(n-1)] if which=="main" else np.r_[0.,np.full(n-1,bias)]
            yield f"{which}_edge_{bias:+g}nm",transformed_holes(cell,lattice,period,b,shifts,zero),dict(radial_bias_nm=b,cartesian_shift_nm=shifts,rotation_deg=zero)
    rng=np.random.default_rng(3001030)
    for i in range(6):
        b=rng.uniform(-3.,3.,n);s=rng.uniform(-3.,3.,(n,2));a=rng.uniform(-1.,1.,n)
        yield f"independent_bounded_{i}",transformed_holes(cell,lattice,period,b,s,a),dict(radial_bias_nm=b,cartesian_shift_nm=s,rotation_deg=a)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("case",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--modes",type=int,default=12)
    a=p.parse_args();summary=json.loads((a.case/"summary.json").read_text());rows=json.loads((a.case/"closed_optical_modes.json").read_text())
    rows.sort(key=lambda r:r["required_material_gain_cm1"]);rows=rows[:a.modes]
    lattice,design,period=summary["lattice"],summary["design"],summary["period_nm"]
    settings=summary["settings"];order,dz=settings["order"],settings["dz_nm"]
    grid=settings["square_grid"] if lattice=="square" else settings["tri_radius_cells"]
    saved=np.load(a.case/"closed_fields.npz");references=[]
    for row in rows:
        opt,_,g=optical(lattice,design,order,dz,row["wavelength_nm"],period,row["closed_TE_family"])
        spec,_,problem,mesh=finite(lattice,opt,grid,8,False)
        references.append((row,opt,spec,mesh,saved[row["name"]+"_field"].copy()))
    a.output.mkdir(parents=True,exist_ok=True);results=[];cell=geometry(lattice,design)
    for case_index,(name,holes,perturbation) in enumerate(variants(cell,lattice,period)):
        t=time.monotonic();d={**design,"_yag_explicit_holes":encode(holes,lattice)}
        fab=evaluate_fabrication(holes,period,None if lattice=="square" else TRIANGULAR_DIRECT,RULE)
        closed=[];models=[];names=[];failures=[];stored=[]
        for rank,(source,ref,spec,mesh,field) in enumerate(references):
            try:
                seed=(complex(source["delta_per_m"],source["alpha_field_per_m"]),field)
                w,opt,g,local_spec,mode,trace=close_seed(lattice,d,period,ref,seed,spec,mesh,order,dz,source["closed_TE_family"])
                mode=replace(mode,name=f"continued_{rank}")
                row,data=mode_row(lattice,opt,g,local_spec,mode,w,129)
                _,_,_,widths,qe,qg=data
                lateral=lateral_energy_weights(mode.fields,31) if lattice=="square" else binned_lateral_weights(mode.fields,mode.grid.x_um,300.,31)
                up,down=opt.side_power_losses(mode.fields)
                models.append(SpatialYbYAGModel(material(w),widths,qe,2*mode.alpha_per_m+10.,up,down,c/opt.result.group_index,qg,lateral));names.append(mode.name)
                row.update(wavelength_nm=w,source_mode=source["name"],frequency_trace=trace,closed_TE_family=source["closed_TE_family"])
                closed.append(row);stored.append((mode,None,None,None))
            except Exception as error:failures.append(dict(source_mode=source["name"],error=repr(error)))
        retained,identity=distinct_closed_modes(closed,stored)
        closed=[closed[i] for i in retained];models=[models[i] for i in retained];names=[names[i] for i in retained]
        pumps=[]
        if models:
            system=SpatialMultimodeYbYAG(tuple(models),tuple(names))
            for arch in ("edge_dual","top_single"):
                threshold=shared_inversion_pump_threshold(system,PumpGeometryConfig(arch,0.,300.,31))
                winners=[next(r for r in closed if r["name"]==n) for n in threshold.first_modes]
                pumps.append(dict(architecture=arch,threshold=asdict(threshold),actual_winners=winners))
        result=dict(name=name,lattice=lattice,frozen_period_nm=period,perturbation=perturbation,fabrication=fab,
            closed_modes=closed,failed_modes=failures,pumps=pumps,seconds=time.monotonic()-t,
            mode_identity_audit=identity,
            scope="Periodic unit-cell systematic errors; first twelve nominal global-census competitors continued at fixed period/layers/grid, not a new exhaustive spectrum or random disorder/yield calculation")
        results.append(result);save(a.output/"process_cases.json",results)
        print(case_index,name,"closed",len(closed),"fail",len(failures),"threshold",pumps[0]["threshold"]["first_threshold_internal_W"] if pumps else None,flush=True)
    save(a.output/"summary.json",dict(source=str(a.case),cases=len(results),nominal_grid=grid,order=order,dz_nm=dz,
        sampled_modes_per_case=len(rows),all_modes_continued=all(not r["failed_modes"] for r in results),
        assumptions="Bounded +/-3nm normal edge offsets and independent Cartesian placements; +/-1degree rotations; each periodic cell changes identically",results=results))


if __name__=="__main__":main()
