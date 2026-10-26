"""Track complete-census shortlist modes while refining YAG triangle shape.

This cheap local stage is explicitly a tracked subset. Its best geometry must
pass a new complete coarse QZ / fine-wide / frequency-closed challenge later.
"""
from matched300_yag_search import *


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("input",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--grid",type=int,default=21)
    p.add_argument("--order",type=int,default=16);p.add_argument("--dz",type=float,default=2.)
    a=p.parse_args();parent=json.loads(a.input.read_text(encoding="utf8"))
    if isinstance(parent,list):parent=parent[0]
    d0=parent["design"];period=parent["period_nm"]
    ref,_,guide=optical("triangular",d0,a.order,a.dz,period=period)
    spec,_,problem,mesh=finite("triangular",ref,a.grid,8,False)
    init=np.ones(problem.stiffness.shape[0],complex);seeds=[]
    for row in parent["modes"][:8]:
        target=complex(row["delta_per_m"],row["alpha_field_per_m"])
        vv,ff=problem.near(target+complex(.5,.7),4,init,tol=1e-10,maxiter=400,dense_limit=0)
        index=int(np.argmin(abs(vv-target)));value=vv[index];field=ff[:,index].reshape(6,mesh.points)
        if abs(value-target)>1e-5*max(1.,abs(target)):raise RuntimeError("Cannot reconstruct a source census pole")
        if independent_eigenmode(value,field,[v for v,_ in seeds],[f for _,f in seeds]):seeds.append((value,field))
    save(a.output/"source_seed_audit.json",dict(source=str(a.input),independent_reconstructed=len(seeds),
        eigenvalues=[dict(delta_per_m=v.real,alpha_field_per_m=v.imag) for v,_ in seeds]))
    proposals=[]
    for aspect in np.linspace(.75,1.30,23):
        proposals.append({**d0,"_yag_triangle_aspect_enabled":True,"main_aspect":float(aspect)})
    for aspect in (.85,1.,1.15):
        for shift in (-10.,-5.,5.,10.):
            proposals.append({**d0,"_yag_triangle_aspect_enabled":True,"main_aspect":aspect,
                "main_orientation_deg":d0["main_orientation_deg"]+shift})
    results=[]
    for index,d in enumerate(proposals):
        t=time.monotonic()
        try:
            model,_,g=optical("triangular",d,a.order,a.dz,period=period)
            fab=evaluate_fabrication(geometry("triangular",d).inclusions,period,TRIANGULAR_DIRECT,RULE)
            if not fab["margin_pass"]:raise ValueError("Geometry margin failed")
            local_spec,_,local_problem,local_mesh=finite("triangular",model,a.grid,8,False)
            values=[];fields=[]
            for value0,field0 in seeds:
                predictor=value0+np.einsum("ik,ij,jk->",field0.conj(),model.result.coupling_m-ref.result.coupling_m,field0)/np.sum(abs(field0)**2)
                vv,ff=local_problem.near(predictor+complex(.5,.7),4,field0.ravel(),tol=1e-9,maxiter=300,dense_limit=0)
                for value,field in zip(vv,ff.T):
                    if value.imag>=-1e-7 and independent_eigenmode(value,field,values,fields):
                        values.append(value);fields.append(field)
            order=np.argsort(np.imag(values));lowest=order[0]
            mode=as_mode("triangular",fields[lowest].reshape(6,mesh.points),values[lowest],model.result,mesh,"tracked_lowest")
            mode.eigen_residual=relative_residual(local_problem.center_operator(),mode.fields,values[lowest])
            row,_=mode_row("triangular",model,g,local_spec,mode,1030.,129)
            gap=(2*values[order[1]].imag+10)/(2*values[lowest].imag+10)-1
            b=row["beam"];score=(row["required_material_gain_cm1"]+150*max(b["ellipticity"]-1.02,0)+
                200*max(.99-b["center_to_peak"],0)+120*max(b["radial_anisotropy_l2"]-.10,0)+100*max(.10-gap,0))
            result=dict(lattice="triangular",design=d,period_nm=period,index=index,status="tracked_local_subset",
                lowest=row,relative_material_gap=gap,score=score,fabrication=fab,independent_returned=len(values),
                settings=dict(order=a.order,dz_nm=a.dz,grid=a.grid),seconds=time.monotonic()-t,
                scope="8 whole-census source poles and their 4 nearest local neighbours; final whole census still required")
            print(index,"aspect",d["main_aspect"],"g",row["required_material_gain_cm1"],"ellipse",b["ellipticity"],"center",b["center_to_peak"],flush=True)
        except Exception as error:result=dict(lattice="triangular",design=d,status="failed",error=repr(error));print("failed",index,repr(error),flush=True)
        results.append(result);save(a.output/"all_results.json",results)
    save(a.output/"ranked.json",sorted((r for r in results if r["status"]!="failed"),key=lambda r:r["score"]))


if __name__=="__main__":main()
