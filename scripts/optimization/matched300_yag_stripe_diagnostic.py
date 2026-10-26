"""YAG counterpropagating-pair limits supply independent spectral challenges."""
from matched300_yag_search import *
from scripts.diagnostics.audit_counterpropagating_stripes import stripe_pair


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("input",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--top",type=int,default=6)
    p.add_argument("--order",type=int,default=16);p.add_argument("--dz",type=float,default=2.)
    a=p.parse_args();rows=json.loads(a.input.read_text(encoding="utf8"))
    if not isinstance(rows,list):rows=[rows]
    out=[]
    for i,row in enumerate(rows[:a.top]):
        t=time.monotonic();lattice,d=row["lattice"],row["design"]
        optical_model,period,guide=optical(lattice,d,a.order,a.dz,period=row["period_nm"])
        _,qe,qg=active_weights(optical_model.result,guide,24)
        C=optical_model.result.coupling_m;pairs=[]
        for pair in range(2 if lattice=="square" else 3):
            length=300. if lattice=="square" or pair==2 else 300./np.sin(np.pi/3)
            diag=stripe_pair(C[2*pair:2*pair+2,2*pair:2*pair+2],length,float(qg.sum()),.1,grids=(51,101))
            pairs.append(dict(pair=pair,length_um=length,**diag))
        threshold=min(p["independent_continuous_transfer_pole"]["g_th_cm_inverse"] for p in pairs
                      if p["independent_continuous_transfer_pole"]["converged"])
        result=dict(index=i,lattice=lattice,design=d,period_nm=period,order=a.order,dz_nm=a.dz,
            reported_bright_candidate_material_g_cm1=row["lowest"]["required_material_gain_cm1"],
            gain_weight=qg.sum(),stripe_limit_material_g_cm1=threshold,pairs=pairs,
            candidate_below_all_pair_limits=row["lowest"]["required_material_gain_cm1"]<threshold,
            seconds=time.monotonic()-t,
            scope="Independent continuous two-wave transfer roots plus finite 1D convergence; asymptotic transverse-detuning challenge, not an actual 2D Jones field or closed pump threshold")
        out.append(result);save(a.output,out)
        print(i,lattice,"bright",result["reported_bright_candidate_material_g_cm1"],"pairlimit",threshold,"s",result["seconds"],flush=True)


if __name__=="__main__":main()
