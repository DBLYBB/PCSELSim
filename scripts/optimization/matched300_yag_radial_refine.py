"""Fine-grid local refinement using the COMPLETE intensity radial residual.

Geometry trials are frozen at the source wavelength and follow one supplied
branch; only shortlisted final points are frequency-closed. A new full census
must still decide the actual first mode and shared-pump threshold afterwards.
"""
from matched300_yag_search import *
from scipy.optimize import least_squares


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("input",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--evaluations",type=int,default=12)
    p.add_argument("--rounding-grid",type=float,nargs="+")
    p.add_argument("--minimum-rounding",type=float,default=.30)
    a=p.parse_args();record=json.loads(a.input.read_text());record=record[0] if isinstance(record,list) else record
    d=record["design"];s=record["settings"];period=record["period_nm"];w=record.get("wavelength_nm",1030.);lattice="triangular"
    reference,_,guide=optical(lattice,d,s["order"],s["dz_nm"],w,period)
    spec,_,problem,mesh=finite(lattice,reference,s["grid"],8,False)
    target=complex(record["lowest"]["delta_per_m"],record["lowest"]["alpha_field_per_m"])
    indices=np.arange(problem.stiffness.shape[0]);initial=1+.19*np.cos(indices)+.17j*np.sin(3*indices)
    vv,ff=problem.near(target+complex(.5,.7),4,initial,tol=1e-10,maxiter=500,dense_limit=0)
    chosen=int(np.argmin(abs(vv-target)));source=ff[:,chosen].reshape(6,mesh.points)
    source_value=complex(vv[chosen]);flat=source.reshape(6,-1)
    results=[];cache={};a.output.mkdir(parents=True,exist_ok=True)
    def evaluate(parameters):
        key=tuple(map(float,parameters))
        if key in cache:return cache[key]
        theta,distance,rounding=key
        design={**d,"main_orientation_deg":180.,"angle1_deg":-theta,"angle2_deg":theta,
            "distance1":distance,"distance2":distance,"rounding":rounding,"_yag_triangle_samples_per_corner":27}
        start=time.monotonic();fab=evaluate_fabrication(geometry(lattice,design).inclusions,period,TRIANGULAR_DIRECT,RULE)
        if not fab["margin_pass"]:
            result=dict(lattice=lattice,design=design,period_nm=period,wavelength_nm=w,settings=s,
                lowest=None,fabrication=fab,residual=np.ones(5)*10,score=float(np.sqrt(500)),
                status="geometry_rejected",scope="Common process-margin rejection; no optical score")
            cache[key]=result;results.append(result);save(a.output/"all_results.json",results)
            print(len(results),key,"geometry_rejected",fab,flush=True);return result
        model,_,g=optical(lattice,design,s["order"],s["dz_nm"],w,period)
        _,_,local_problem,_=finite(lattice,model,s["grid"],8,False)
        predictor=source_value+np.einsum("ik,ij,jk->",flat.conj(),model.result.coupling_m-reference.result.coupling_m,flat)/np.sum(abs(flat)**2)
        value,field,follow=follow_reference_pole(local_problem,source,predictor)
        mode=as_mode(lattice,field,value,model.result,mesh,"known_branch");mode.eigen_residual=follow["eigen_residual"]
        row,_=mode_row(lattice,model,g,spec,mode,w,201)
        b=row["beam"];gain=row["required_material_gain_cm1"]
        residual=np.array([b["radial_anisotropy_l2"],4*max(b["ellipticity"]-1.03,0),
            20*max(.98-b["center_to_peak"],0),.1*max(gain-10.,0),.2*max(.5-row["up_fraction"],0)])
        result=dict(lattice=lattice,design=design,period_nm=period,wavelength_nm=w,settings=s,
            lowest=row,fabrication=fab,branch_follow=follow,residual=residual,score=float(np.linalg.norm(residual)),
            seconds=time.monotonic()-start,status="frozen_known_branch_high_grid",
            scope="Fine-grid local radial/brightness/ellipse design at fixed source wavelength; not a frequency-closed or first-mode claim")
        cache[key]=result;results.append(result);save(a.output/"all_results.json",results)
        print(len(results),key,gain,b["ellipticity"],b["center_to_peak"],b["radial_anisotropy_l2"],result["seconds"],flush=True)
        return result
    def residual(x):return evaluate(x)["residual"]
    if a.rounding_grid:
        for rounding in a.rounding_grid:evaluate([d["angle2_deg"],d["distance1"],rounding])
        save(a.output/"solver.json",dict(method="explicit rounding boundary grid",values=a.rounding_grid,
            reason="Do not interpret an arbitrary .30 local trust-region bound as a physical optimum"))
    else:
        fit=least_squares(residual,[d["angle2_deg"],d["distance1"],d["rounding"]],
            bounds=([35.,.49,a.minimum_rounding],[39.,.515,.49]),x_scale=[1.,.005,.05],diff_step=[1e-4,2e-4,2e-4],
            max_nfev=a.evaluations,ftol=1e-5,xtol=1e-6,gtol=1e-5)
        best=evaluate(fit.x)
        save(a.output/"solver.json",dict(success=fit.success,parameters=fit.x,residual=fit.fun,message=fit.message,nfev=fit.nfev))
    ranked=sorted((r for r in results if r.get("lowest")),key=lambda r:r["score"]);save(a.output/"ranked_frozen.json",ranked)
    unique=[];scale=np.array([1.,.005,.05])
    for result in ranked:
        parameters=np.array([result["design"][k] for k in ("angle2_deg","distance1","rounding")])
        if all(np.linalg.norm((parameters-old)/scale)>.01 for _,old in unique):unique.append((result,parameters))
    closed=[]
    for frozen,_ in unique[:3]:
        design=frozen["design"];model,_,g=optical(lattice,design,s["order"],s["dz_nm"],w,period)
        _,_,lp,_=finite(lattice,model,s["grid"],8,False)
        predictor=complex(frozen["lowest"]["delta_per_m"],frozen["lowest"]["alpha_field_per_m"])
        v,f,_=follow_reference_pole(lp,source,predictor)
        wavelength,opt,g,cs,mode,trace=close_seed(lattice,design,period,model,(v,f),spec,mesh,s["order"],s["dz_nm"])
        row,_=mode_row(lattice,opt,g,cs,mode,wavelength,201)
        result={**frozen,"wavelength_nm":wavelength,"lowest":row,"frequency_trace":trace,"status":"actual_closed_known_branch"}
        closed.append(result);save(a.output/"ranked.json",closed)


if __name__=="__main__":main()
