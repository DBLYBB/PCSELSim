"""Resolution/contour audit of a known branch at a proposed final period.

The actual shared-pump first mode is determined by the separate full census.
This early audit prevents optimizing a beam that changes with the envelope grid.
"""
from matched300_yag_search import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("input",type=Path)
    p.add_argument("--index",type=int,default=0);p.add_argument("--period",type=float,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    r=json.loads(a.input.read_text());r=r[a.index] if isinstance(r,list) else r
    d,s=r["design"],r["settings"];lattice=r["lattice"];w=r.get("wavelength_nm",1030.)
    reference,_,_=optical(lattice,d,s["order"],s["dz_nm"],w,r["period_nm"])
    ss,_,sp,sm=finite(lattice,reference,s["grid"],8,False)
    target=complex(r["lowest"]["delta_per_m"],r["lowest"]["alpha_field_per_m"])
    indices=np.arange(sp.stiffness.shape[0]);v0=1+.19*np.cos(indices)+.17j*np.sin(3*indices)
    vv,ff=sp.near(target+complex(.5,.7),4,v0,tol=1e-10,maxiter=500,dense_limit=0)
    chosen=int(np.argmin(abs(vv-target)));source=ff[:,chosen].reshape(6,sm.points)
    tasks=[(32,52,27),(48,52,27),(32,65,27),(32,52,54)]
    a.output.mkdir(parents=True,exist_ok=True);results=[]
    for order,grid,contour in tasks:
        start=time.monotonic();design={**d,"_yag_triangle_samples_per_corner":contour}
        try:
            model,_,_=optical(lattice,design,order,1.,w,a.period)
            spec,_,problem,mesh=finite(lattice,model,grid,8,False)
            field=interpolate_envelope(lattice,source,ss,sm,spec,mesh);flat=field.reshape(6,-1)
            predictor=vv[chosen]+np.einsum("ik,ij,jk->",flat.conj(),model.result.coupling_m-reference.result.coupling_m,flat)/np.sum(abs(flat)**2)
            value,field,follow=follow_reference_pole(problem,field,predictor)
            wave,opt,g,cs,mode,trace=close_seed(lattice,design,a.period,model,(value,field),spec,mesh,order,1.)
            row,_=mode_row(lattice,opt,g,cs,mode,wave,201)
            result=dict(status="completed",order=order,grid=grid,contour_samples_per_corner=contour,
                period_nm=a.period,wavelength_nm=wave,mode=row,initialization=follow,frequency_trace=trace,
                seconds=time.monotonic()-start,
                scope="Known branch convergence at fixed proposed final period; global first mode remains subject to the complete separate census")
            print(order,grid,contour,wave,row["required_material_gain_cm1"],row["beam"]["ellipticity"],row["beam"]["radial_anisotropy_l2"],flush=True)
        except Exception as error:
            result=dict(status="failed",order=order,grid=grid,contour=contour,error=repr(error));print(result,flush=True)
        results.append(result);save(a.output/"results.json",results)


if __name__=="__main__":main()
