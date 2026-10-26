"""Compare complete equivalent-operator QR and original generalized QZ spectra."""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[_key]="1"
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.optimization.triangular_semiconductor_search import json_write


def validate(qr_path,qz_path):
    qa=np.load(qr_path/"qz_low_fields.npz");qz=np.load(qz_path/"qz_low_fields.npz")
    va,vb=qa["eigenvalues"],qz["eigenvalues"]
    if len(va)!=len(vb):raise ValueError("Residual-verified full spectrum counts differ")
    cost=abs(va[:,None]-vb[None,:])/np.maximum(np.maximum(abs(va[:,None]),abs(vb[None,:])),1.)
    ia,ib=linear_sum_assignment(cost)
    mapping=np.empty(len(va),int);mapping[ia]=ib
    fa=qa["fields"].reshape(len(va),-1).T;fb=qz["fields"].reshape(len(vb),-1).T[:,mapping]
    if fa.shape!=fb.shape:raise ValueError("Validation requires retaining every eigenfield")
    assigned_relative=cost[ia,ib];assigned_absolute=abs(va-vb[mapping])
    groups=[];remaining=set(range(len(va)))
    while remaining:
        start=min(remaining);cluster={start};front=[start];remaining.remove(start)
        while front:
            left=front.pop()
            close=[right for right in remaining if abs(va[left]-va[right])<=2e-7*max(abs(va[left]),abs(va[right]),1.)]
            cluster.update(close);front.extend(close);remaining.difference_update(close)
        groups.append(sorted(cluster))
    errors=[];phase_errors=[]
    for group in groups:
        xa,xb=fa[:,group],fb[:,group]
        ua,sa,_=np.linalg.svd(xa,full_matrices=False);ub,sb,_=np.linalg.svd(xb,full_matrices=False)
        ua=ua[:,sa>1e-8*sa[0]];ub=ub[:,sb>1e-8*sb[0]]
        subspace=float(max(np.linalg.norm(ua-ub@(ub.conj().T@ua),ord=2),np.linalg.norm(ub-ua@(ua.conj().T@ub),ord=2)))
        errors.append({"indices":group,"dimension":len(group),"qr_rank":ua.shape[1],"qz_rank":ub.shape[1],"subspace_error":subspace})
        if len(group)==1:
            left=xa[:,0];right=xb[:,0];right=right*np.exp(-1j*np.angle(np.vdot(left,right)))
            phase_errors.append(float(np.linalg.norm(left-right)))
    reports=[]
    for folder in (qr_path,qz_path):
        records=json.loads((folder/"complete_coarse_poles.json").read_text())
        reports.append({"path":str(folder),"maximum_original_pencil_relative_residual":max(r["relative_eigen_residual"] for r in records),
            "maximum_loss_budget_closure_per_m":max(abs(r["alpha_per_m"]-r["local_amplitude_loss_per_m"]-r["transport_amplitude_loss_per_m"]) for r in records),
            "seconds":json.loads((folder/"summary.json").read_text())["elapsed_seconds"]})
    return {"full_pole_count":len(va),"one_to_one_assignment_maximum_relative_error":float(assigned_relative.max()),
        "one_to_one_assignment_maximum_absolute_error_per_m":float(assigned_absolute.max()),
        "symmetric_Hausdorff_relative_error":float(max(cost.min(0).max(),cost.min(1).max())),
        "lowest_64_max_relative_pole_error":float(assigned_relative[:64].max()),
        "maximum_nondegenerate_phase_aligned_field_error":max(phase_errors,default=0.),
        "maximum_cluster_subspace_error":max(r["subspace_error"] for r in errors),
        "clusters":errors,"residual_and_energy":reports,
        "exact_equation":"K=C M+iD, H=K M^-1=C+iD M^-1, phi=M u; QR fields are phi and are checked against original K,M",
        "scope":"Full residual-verified coarse spectra, not only Gamma modes; worst non-normal and near-degenerate discrepancies are retained."}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--qr",type=Path,required=True);p.add_argument("--qz",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=validate(a.qr,a.qz);json_write(a.output,r)
    print({k:v for k,v in r.items() if k not in ("clusters",)},flush=True)


if __name__=="__main__":main()
