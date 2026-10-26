"""Identical YAG closure with a validated complete coarse QR spectrum.

中文：只替换驱动内的完整粗网格求谱算法，不修改物理模型、细网格移位反演、
模式身份、阈值或闭频方程。H=K M^-1=C+iD M^-1，输出为物理中心场phi。
Every coarse eigenpair is checked against the original K/M operator. Run
``validate`` before using ``close``; the small-pencil QR/QZ comparison is saved.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(Path(__file__).parent)]
import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import splu
from pcselsim.finite_discretization import FiniteEigenproblem,relative_residual
import matched300_yag_search as b

OUT=ROOT/'results/current/strict_rerun_20261025/YbYAG_triangular'
_original_near=FiniteEigenproblem.near
_original_save=b.save
QR_CALLS=[]

def qr_spectrum(problem):
    """Ordinary non-Hermitian QR on the exact center-field operator / 不做对称化。"""
    started=time.monotonic();n=problem.stiffness.shape[0]
    inverse=splu(problem.mass).solve(np.eye(n,dtype=complex))
    operator=problem.local.toarray()+1j*(problem.transport@inverse)
    values,fields=eig(operator,overwrite_a=True,check_finite=True)
    del inverse,operator
    norms=np.linalg.norm(fields,axis=0)
    finite=np.isfinite(values)&np.isfinite(norms)&(norms>1e-30)
    fields[:,finite]/=norms[finite][None,:]
    original=problem.center_operator()
    residual=np.full(n,np.inf)
    for i in np.flatnonzero(finite):residual[i]=relative_residual(original,fields[:,i],values[i])
    valid=finite&(residual<=2e-6)
    audit=dict(solver='complete ordinary non-Hermitian QR of H=K M^-1',dimension=n,
        raw_eigenpairs=n,finite_original_KM_verified=int(valid.sum()),
        nonfinite_rejected=int((~finite).sum()),original_KM_residual_rejected=int((finite&~valid).sum()),
        maximum_original_KM_relative_residual=float(residual[valid].max()),seconds=time.monotonic()-started,
        eigenvalues_per_m=[[float(x.real),float(x.imag)] for x in values[valid]],
        equivalence='K=C M+iD, phi=M u; M is invertible. No Hermitian mass assumption or change to the pencil.')
    return values[valid],fields[:,valid],audit

def checked_near(self,target,count,v0,tol=1e-9,maxiter=8000,*,dense_limit=None):
    if dense_limit is not None and dense_limit>=self.stiffness.shape[0] and not hasattr(self,'_dense_spectrum'):
        values,fields,audit=qr_spectrum(self)
        self._dense_spectrum=(values,fields);QR_CALLS.append(audit)
        print('complete_coarse_QR',audit['dimension'],len(values),audit['seconds'],flush=True)
    return _original_near(self,target,count,v0,tol,maxiter,dense_limit=dense_limit)

def save_with_solver(path,value):
    """Correct legacy QZ field names in this QR driver's records / 审计记录注明实际算法。"""
    if isinstance(value,dict) and 'residual_verified_full_QZ_count' in value:
        value=dict(value);value['residual_verified_full_QR_count']=value.pop('residual_verified_full_QZ_count')
        value['coarse_solver']='ordinary QR on exact physical-center operator'
        value['complete_coarse_spectrum']=value.pop('coarse_qz',True)
        value['complete_coarse_QR_audit']=QR_CALLS[-1]
        value['targets']=[{**r,'kind':r.get('kind','').replace('complete_coarse_QZ','complete_coarse_QR')} for r in value.get('targets',[])]
    return _original_save(path,value)

def validate():
    source=json.loads((OUT/'baseline_input.json').read_text(encoding='utf8'))
    model,_,_=b.optical('triangular',source['design'],16,2.,1030.,source['period_nm'])
    _,_,problem,_=b.finite('triangular',model,5,8,False)
    qr,qf,audit=qr_spectrum(problem)
    _original_near(problem,complex(model.result.eigenvalues_m[0]),8,np.ones(problem.stiffness.shape[0]),dense_limit=10000)
    qz,zf=problem._dense_spectrum
    ri,ci=linear_sum_assignment(abs(qr[:,None]-qz[None,:]))
    error=abs(qr[ri]-qz[ci]);relative=error/np.maximum(abs(qz[ci]),1.)
    passed=bool(len(qr)==len(qz)==problem.stiffness.shape[0] and relative.max()<1e-8)
    result=dict(passed=passed,count_QR=len(qr),count_QZ=len(qz),
        max_eigenvalue_absolute_error_per_m=float(error.max()),maximum_relative_error=float(relative.max()),
        coarse_QR=audit,scope='Full one-to-one small YAG triangular pencil comparison; larger pencils still verify every original K/M residual.')
    _original_save(OUT/'QR_QZ_validation.json',result)
    if not passed:raise RuntimeError('Independent QR/QZ comparison failed')
    print('VALIDATED',len(qr),float(relative.max()),flush=True)

def main():
    if len(sys.argv)>1 and sys.argv[1]=='validate':validate();return
    check=json.loads((OUT/'QR_QZ_validation.json').read_text(encoding='utf8'))
    if not check['passed']:raise RuntimeError('QR/QZ validation required')
    FiniteEigenproblem.near=checked_near;b.save=save_with_solver
    args=sys.argv[1:];output=Path(args[args.index('--output')+1])
    if output.exists():raise FileExistsError(output)
    # Only provenance files are written after the existing close command creates its directory.
    b.main(args)
    _original_save(output/'QR_runtime_provenance.json',dict(argv=sys.argv,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validation=str(OUT/'QR_QZ_validation.json'),coarse_QR_calls=QR_CALLS))

if __name__=='__main__':main()
