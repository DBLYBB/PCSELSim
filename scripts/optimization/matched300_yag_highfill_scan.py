"""Additional YAG-only high-fill search, independent of semiconductor ranges."""
from matched300_yag_search import *
from scipy.stats import qmc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=OUT/"highfill")
    p.add_argument("--count",type=int,default=160)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/"triangular_highfill.jsonl"
    if path.exists():raise FileExistsError(path)
    designs=[]
    for i,u in enumerate(qmc.LatinHypercube(18,seed=1030301).random(a.count)):
        kinds=("circle","ellipse","triangle")
        symmetric=i%2==0
        designs.append(ThreeHole(f"highfill_{i:04d}",fill=.225+.105*u[0],main_share=.5+.28*u[1],
            satellite_balance=.5 if symmetric else .3+.4*u[2],main_shape=kinds[i%3],
            satellite1_shape=kinds[(i//3)%3],satellite2_shape=kinds[(i//3)%3] if symmetric else kinds[(i//9)%3],
            main_aspect=1+.7*u[3],satellite1_aspect=1+.7*u[4],satellite2_aspect=1+.7*u[4 if symmetric else 5],
            distance1=.36+.22*u[6],distance2=.36+.22*u[6 if symmetric else 7],
            angle1_deg=-30+20*(u[8]-.5),angle2_deg=30-20*(u[8]-.5) if symmetric else 30+20*(u[9]-.5),
            main_orientation_deg=180*u[10],satellite1_orientation_deg=180*u[11],
            satellite2_orientation_deg=-180*u[11] if symmetric else 180*u[12],rounding=.18+.25*u[13]))
    for fill in (.24,.26,.28,.30):
        for share in (.50,.60,.70):
            for distance in (.40,.46,.52):
                designs.append(ThreeHole(f"highfill_anchor_{len(designs):04d}",fill=fill,main_share=share,
                    distance1=distance,distance2=distance))
    rows=[]
    with path.open("w",encoding="utf8") as stream:
        for i,d in enumerate(designs):
            try:r=evaluate("triangular",asdict(d),4,5.,9,12)
            except Exception as error:r=dict(status="failed",lattice="triangular",design=asdict(d),error=repr(error))
            r["index"]=i;stream.write(json.dumps(clean(r))+"\n");stream.flush()
            if r["status"]=="frozen_screen":rows.append(r)
            if i%10==0:print("highfill",i,r["status"],r.get("score"),flush=True)
    save(a.output/"triangular_highfill_best.json",sorted(rows,key=lambda r:r["score"])[:12])


if __name__=="__main__":main()
