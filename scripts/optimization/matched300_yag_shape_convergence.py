"""Frozen-period contour-resolution and orientation checks of a known YAG pole.

This follows a known physical branch rather than claiming a new first mode.
Every proposed improvement must subsequently pass the complete bounded census.
"""
from matched300_yag_search import *


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path);p.add_argument("--index",type=int,default=0)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--order",type=int,default=32);p.add_argument("--dz",type=float,default=1.)
    p.add_argument("--tri-grid",type=int,default=39)
    p.add_argument("--period",type=float)
    p.add_argument("--symmetric-angle-scan",action="store_true")
    a=p.parse_args();record=json.loads(a.input.read_text());record=record[a.index] if isinstance(record,list) else record
    d=record["design"];settings=record["settings"];lattice=record["lattice"]
    assert lattice=="triangular"
    source,_,_=optical(lattice,d,settings["order"],settings["dz_nm"],period=record["period_nm"])
    ss,_,sp,sm=finite(lattice,source,settings["grid"],12,False)
    target=complex(record["lowest"]["delta_per_m"],record["lowest"]["alpha_field_per_m"])
    vv,ff=sp.near(target+complex(.5,.7),12,np.ones(sp.stiffness.shape[0],complex),tol=1e-10,maxiter=500,dense_limit=0)
    chosen=int(np.argmin(abs(vv-target)));source_field=ff[:,chosen].reshape(6,sm.points)
    period=a.period or record["period_nm"]
    reference,_,_=optical(lattice,d,a.order,a.dz,period=period)
    spec,_,problem,mesh=finite(lattice,reference,a.tri_grid,12,False)
    field=interpolate_envelope(lattice,source_field,ss,sm,spec,mesh)
    flat=field.reshape(6,-1);change=reference.result.coupling_m-source.result.coupling_m
    predictor=vv[chosen]+np.einsum("ik,ij,jk->",flat.conj(),change,flat)/np.sum(abs(flat)**2)
    value,field,initialization=follow_reference_pole(problem,field,predictor)
    cases=[(f"contour_{n}",{**d,"_yag_triangle_samples_per_corner":n}) for n in (9,18,27)]
    cases += [(f"orientation_{da:+g}",{**d,"_yag_triangle_samples_per_corner":27,
        "main_orientation_deg":d["main_orientation_deg"]+da}) for da in (-5.,-2.5,2.5,5.,7.5)]
    if a.symmetric_angle_scan:
        cases=[(f"symmetric_satellite_angle_{theta:g}",{**d,"_yag_triangle_samples_per_corner":27,
            "main_orientation_deg":180.,"angle1_deg":-theta,"angle2_deg":theta}) for theta in (31.,34.,37.,40.,43.)]
    a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for name,design in cases:
        start=time.monotonic()
        try:
            model,_,guide=optical(lattice,design,a.order,a.dz,period=period)
            _,_,local_problem,_=finite(lattice,model,a.tri_grid,12,False)
            change=model.result.coupling_m-reference.result.coupling_m
            flat=field.reshape(6,-1);prediction=value+np.einsum("ik,ij,jk->",flat.conj(),change,flat)/np.sum(abs(flat)**2)
            v,f,audit=follow_reference_pole(local_problem,field,prediction)
            wavelength,opt,g,closed_spec,mode,trace=close_seed(lattice,design,period,model,(v,f),spec,mesh,a.order,a.dz)
            row,_=mode_row(lattice,opt,g,closed_spec,mode,wavelength,201)
            fab=evaluate_fabrication(geometry(lattice,design).inclusions,period,TRIANGULAR_DIRECT,RULE)
            result=dict(name=name,status="completed",lattice=lattice,design=design,period_nm=period,
                settings=dict(order=a.order,dz_nm=a.dz,grid=a.tri_grid),wavelength_nm=wavelength,
                lowest=row,branch_follow=audit,frequency_trace=trace,fabrication=fab,
                seconds=time.monotonic()-start,scope="Known-branch contour/orientation continuation; not a first-mode or isolation claim")
            print(name,wavelength,row["required_material_gain_cm1"],row["beam"]["ellipticity"],row["beam"]["center_to_peak"],flush=True)
        except Exception as error:
            result=dict(name=name,status="failed",error=repr(error));print(name,"failed",repr(error),flush=True)
        rows.append(result);save(a.output/"all_results.json",rows)
    save(a.output/"ranked.json",sorted((r for r in rows if r["status"]=="completed"),key=lambda r:r["lowest"]["beam"]["ellipticity"]))
    save(a.output/"initialization.json",initialization)


if __name__=="__main__":main()
