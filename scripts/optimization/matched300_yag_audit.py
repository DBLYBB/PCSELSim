"""Wide-spectrum scalar audit of shortlisted same-stack YAG geometries."""
from matched300_yag_search import *


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--top",type=int,default=8);p.add_argument("--order",type=int,default=16)
    p.add_argument("--dz",type=float,default=2.)
    p.add_argument("--square-grid",type=int,default=31);p.add_argument("--tri-grid",type=int,default=21)
    p.add_argument("--roots",type=int,default=12);p.add_argument("--wide",action="store_true")
    a=p.parse_args();source=json.loads(a.input.read_text(encoding="utf8"))
    if not isinstance(source,list):source=[source]
    a.output.mkdir(parents=True,exist_ok=True);results=[]
    for index,parent in enumerate(source[:a.top]):
        t=time.monotonic();lattice,d=parent["lattice"],parent["design"]
        try:
            model,period,g=optical(lattice,d,a.order,a.dz)
            grid=a.square_grid if lattice=="square" else a.tri_grid
            spec,modes,problem,mesh=finite(lattice,model,grid,a.roots,not a.wide)
            if a.wide:
                pool=seed_pool(lattice,model,spec,problem,mesh,a.roots,2,True,a.output/f"candidate_{index:02d}_QZ_audit.json")
                physical=[as_mode(lattice,f,v,model.result,mesh,f"pole_{rank}") for rank,(v,f) in enumerate(pool[:24])]
            else:physical=list(modes.values())
            rows=[mode_row(lattice,model,g,spec,m,1030.,129)[0] for m in physical]
            rows.sort(key=lambda r:r["required_material_gain_cm1"])
            b=rows[0]["beam"];gap=rows[1]["required_material_gain_cm1"]/rows[0]["required_material_gain_cm1"]-1
            score=rows[0]["required_material_gain_cm1"]*(1+10*max(b["ellipticity"]-1.05,0)+
                20*max(.98-b["center_to_peak"],0)+10*max(b["radial_anisotropy_l2"]-.10,0)+4*max(.1-gap,0))
            if rows[0]["required_material_gain_cm1"]>=23.93265:score+=1000+100*(rows[0]["required_material_gain_cm1"]-23.93265)
            r=dict(lattice=lattice,design=d,period_nm=period,settings=dict(order=a.order,dz_nm=a.dz,grid=grid,wide=a.wide),
                parent_settings=parent.get("settings"),parent_score=parent.get("score"),
                lowest=rows[0],modes=rows,relative_material_gap=gap,score=score,
                status="wide_frozen_audit" if a.wide else "accuracy_frozen_audit",seconds=time.monotonic()-t)
            print(index,lattice,"g",rows[0]["required_material_gain_cm1"],"ellipse",b["ellipticity"],"center",b["center_to_peak"],"gap",gap,"s",r["seconds"],flush=True)
        except Exception as error:
            r=dict(lattice=lattice,design=d,status="failed",error=repr(error));print(index,"failed",repr(error),flush=True)
        results.append(r);save(a.output/"all_results.json",results)
    ranked=sorted((r for r in results if r["status"]!="failed"),key=lambda r:r["score"])
    save(a.output/"ranked.json",ranked)


if __name__=="__main__":main()
