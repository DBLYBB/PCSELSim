"""High-grid mirror-symmetric motif adjustment to equal far-field moments.

The central triangular hole is mirror symmetric at180deg and circular satellites
occupy plus/minus theta. Equalizing xx and yy moments is then a one-dimensional
design equation. This is only a supplied-branch design; final census is separate.
"""
from matched300_yag_search import *


def moments(axis,fx,fy):
    widths=np.diff(axis);w=np.r_[widths[0]/2,(widths[:-1]+widths[1:])/2,widths[-1]/2]
    intensity=abs(fx)**2+abs(fy)**2;mass=intensity*w[:,None]*w[None,:];mass/=mass.sum()
    xx,yy=np.meshgrid(axis,axis);cx,cy=float(np.sum(xx*mass)),float(np.sum(yy*mass));xx-=cx;yy-=cy
    vx,vy,cov=np.sum(xx*xx*mass),np.sum(yy*yy*mass),np.sum(xx*yy*mass)
    return dict(centroid_x_deg=cx,centroid_y_deg=cy,xx_deg2=float(vx),yy_deg2=float(vy),xy_deg2=float(cov),
        normalized_xx_minus_yy=float((vx-vy)/(vx+vy)),normalized_xy=float(2*cov/(vx+vy)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("input",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--grid",type=int,default=52)
    p.add_argument("--order",type=int,default=32);p.add_argument("--dz",type=float,default=1.)
    p.add_argument("--angles",type=float,nargs=2,default=[36.5,38.5])
    p.add_argument("--center",action="store_true",help="Jointly adjust symmetric angle and distance to equal moments and zero centroid")
    a=p.parse_args()
    record=json.loads(a.input.read_text());record=record[0] if isinstance(record,list) else record
    d=record["design"];period=record["period_nm"];s=record["settings"];lattice="triangular"
    source,_,_=optical(lattice,d,s["order"],s["dz_nm"],record.get("wavelength_nm",1030.),period)
    ss,_,sp,sm=finite(lattice,source,s["grid"],12,False)
    target=complex(record["lowest"]["delta_per_m"],record["lowest"]["alpha_field_per_m"])
    indices=np.arange(sp.stiffness.shape[0]);initial=1+.19*np.cos(indices)+.17j*np.sin(3*indices)
    vv,ff=sp.near(target+complex(.5,.7),4,initial,tol=1e-10,maxiter=500,dense_limit=0)
    chosen=int(np.argmin(abs(vv-target)));source_field=ff[:,chosen].reshape(6,sm.points)
    reference,_,_=optical(lattice,d,a.order,a.dz,period=period)
    spec,_,problem,mesh=finite(lattice,reference,a.grid,12,False)
    field=interpolate_envelope(lattice,source_field,ss,sm,spec,mesh)
    flat=field.reshape(6,-1);change=reference.result.coupling_m-source.result.coupling_m
    predictor=vv[chosen]+np.einsum("ik,ij,jk->",flat.conj(),change,flat)/np.sum(abs(flat)**2)
    value,field,_=follow_reference_pole(problem,field,predictor)
    a.output.mkdir(parents=True,exist_ok=True);results=[];cache={}
    def evaluate(theta,distance=None):
        distance=d["distance1"] if distance is None else float(distance);key=(float(theta),distance)
        if key in cache:return cache[key]
        design={**d,"_yag_triangle_samples_per_corner":27,"main_orientation_deg":180.,"angle1_deg":-float(theta),"angle2_deg":float(theta)}
        design.update(distance1=distance,distance2=distance)
        model,_,_=optical(lattice,design,a.order,a.dz,period=period)
        _,_,local_problem,_=finite(lattice,model,a.grid,12,False)
        change=model.result.coupling_m-reference.result.coupling_m;flat=field.reshape(6,-1)
        prediction=value+np.einsum("ik,ij,jk->",flat.conj(),change,flat)/np.sum(abs(flat)**2)
        val,f,follow=follow_reference_pole(local_problem,field,prediction)
        wavelength,opt,g,closed_spec,mode,trace=close_seed(lattice,design,period,model,(val,f),spec,mesh,a.order,a.dz)
        row,data=mode_row(lattice,opt,g,closed_spec,mode,wavelength,201);axis,fx,fy,*_=data
        mm=moments(axis,fx,fy)
        result=dict(lattice=lattice,design=design,period_nm=period,wavelength_nm=wavelength,
            settings=dict(order=a.order,dz_nm=a.dz,grid=a.grid),lowest=row,moments=mm,
            branch_follow=follow,frequency_trace=trace,status="actual_closed_known_branch",
            scope="Mirror-symmetric supplied-branch shape design only; must pass fresh whole-QZ/stripe census and shared pump")
        cache[key]=result;results.append(result);save(a.output/"all_results.json",results)
        print(theta,distance,row["required_material_gain_cm1"],row["beam"]["ellipticity"],mm["centroid_x_deg"],mm["normalized_xx_minus_yy"],mm["normalized_xy"],flush=True)
        return result
    if a.center:
        from scipy.optimize import least_squares
        def residual(x):
            result=evaluate(x[0],x[1]);mm=result["moments"]
            return np.array([mm["normalized_xx_minus_yy"],10*mm["centroid_x_deg"]])
        fit=least_squares(residual,[d["angle2_deg"],d["distance1"]],bounds=([35.,.45],[39.,.56]),
            x_scale=[1.,.02],diff_step=[1e-4,2e-4],max_nfev=20,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        best=evaluate(*fit.x)
        save(a.output/"joint_moment_solver.json",dict(success=fit.success,parameters=fit.x,residual=fit.fun,message=fit.message,nfev=fit.nfev))
    else:
        lo,hi=a.angles
        objective=lambda theta:evaluate(theta)["moments"]["normalized_xx_minus_yy"]
        if objective(lo)*objective(hi)<0:
            theta=float(brentq(objective,lo,hi,xtol=2e-4));best=evaluate(theta)
        else:best=min(results,key=lambda r:r["lowest"]["beam"]["ellipticity"])
    save(a.output/"best.json",best);save(a.output/"ranked.json",[best])


if __name__=="__main__":main()
