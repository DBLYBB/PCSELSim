"""Close additional frozen spectral clusters without spending all roots in one stripe band.

中文：前 32 根可能集中在同一条纹簇；补查不同 δ 簇及每个 Γ 支。
The explicit cluster/alpha cutoffs define a bounded challenge, not a proof of a complete spectrum.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from dataclasses import replace
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(Path(__file__).parent)]
import numpy as np
import matched300_yag_search as backend
from pcselsim.finite_discretization import relative_residual


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def reconstruct_frozen(problem,target,v0,grid):
    """Offset a nearly singular shift; still accept only the original pencil.

    中文：精确移位可能令反演病态；只改数值移位，不修改原算符或验收阈值。
    """
    history=[]
    for offset,count in ((complex(.5,.7),6),(complex(.03,.07),4),
                         (complex(.003,.007),2),(complex(1.,2.),2)):
        try:
            vv,ff=problem.near(target+offset,count,v0,tol=1e-11,maxiter=1200,dense_limit=0)
            j=int(np.argmin(abs(vv-target)));value=complex(vv[j]);field=ff[:,j].reshape(4,grid,grid)
            residual=relative_residual(problem.center_operator(),field,value)
            history.append(dict(offset=[offset.real,offset.imag],count=count,
                distance=abs(value-target),original_residual=residual))
            if residual<=2e-7 and abs(value-target)<=.05:return value,field,residual,history
        except Exception as error:history.append(dict(offset=[offset.real,offset.imag],count=count,error=repr(error)))
    raise RuntimeError(f'Original frozen pole could not be reconstructed: {history}')


def challenge(folder):
    plan=read(folder/'plan.json');design=plan['record']['design']
    order,dz,grid=plan['order'],plan['dz_nm'],plan['grid']
    period=read(folder/'outer_period_design.json')['frozen_period_nm']
    optical,_,guide=backend.optical('square',design,order,dz,period=period)
    spec,_,problem,mesh=backend.finite('square',optical,grid,8,False)
    poles=read(folder/'TE0_frozen_census.json')['poles']
    clusters=[]
    for rank,row in sorted(enumerate(poles),key=lambda pair:pair[1]['delta_per_m']):
        if not clusters or row['delta_per_m']-clusters[-1][-1][1]['delta_per_m']>500.:
            clusters.append([])
        clusters[-1].append((rank,row))
    selected={}; cluster_audit=[]
    cutoff=4*poles[0]['alpha_field_per_m']
    for group in clusters:
        rank,row=min(group,key=lambda pair:pair[1]['alpha_field_per_m'])
        covered=any(i<32 for i,_ in group)
        qualifying=row['alpha_field_per_m']<=cutoff
        cluster_audit.append(dict(min_delta=group[0][1]['delta_per_m'],max_delta=group[-1][1]['delta_per_m'],
            count=len(group),lowest_frozen_alpha_rank=rank,already_represented_in_first32=covered,
            lowest_frozen_alpha=row['alpha_field_per_m'],qualifying_for_extra_challenge=qualifying))
        if qualifying and not covered:selected.setdefault(rank,[]).append('minimum_alpha_of_uncovered_delta_cluster')
    values=np.array([complex(row['delta_per_m'],row['alpha_field_per_m']) for row in poles])
    for i,value in enumerate(optical.result.eigenvalues_m):
        rank=int(np.argmin(abs(values-value)))
        if rank>=32:selected.setdefault(rank,[]).append(f'nearest_frozen_pole_to_Gamma_branch_{i}')
    out=folder/'spectral_challenges';out.mkdir(parents=True,exist_ok=True)
    path=out/'closed_optical_modes.json';rawpath=out/'closed_fields.npz'
    records=read(path) if path.exists() else []
    arrays={}
    if rawpath.exists():
        with np.load(rawpath,allow_pickle=False) as raw:arrays={key:raw[key].copy() for key in raw.files}
    completed={row['frozen_rank'] for row in records}
    failures=[]
    if (out/'failures.json').exists():
        history_path=out/'historical_failed_attempts.json'
        history=read(history_path) if history_path.exists() else []
        backend.save(history_path,history+read(out/'failures.json'))
    backend.save(out/'plan.json',dict(fixed_period_nm=period,order=order,dz_nm=dz,grid=grid,
        delta_cluster_adjacent_gap_m1=500.,frozen_alpha_cutoff_m1=cutoff,
        clusters=cluster_audit,selected_ranks=selected,
        scope='Additional representatives of uncovered low-alpha delta clusters plus each Gamma branch. All modes are re-solved and real-frequency closed at fixed physical period; bounded challenge, not exhaustive full fine spectrum.'))
    n=problem.stiffness.shape[0];axis=np.arange(n);v0=1+.19*np.cos(axis)+.17j*np.sin(3*axis)
    for rank,reasons in sorted(selected.items()):
        if rank in completed:continue
        started=time.monotonic()
        try:
            target=values[rank]
            seed_value,field,residual,search_history=reconstruct_frozen(problem,target,v0,grid)
            wave,model,g,local_spec,mode,trace=backend.close_seed('square',design,period,optical,
                (seed_value,field),spec,mesh,order,dz,0)
            name=f'TE0_crossrank{rank:03d}';mode=replace(mode,name=name)
            row,data=backend.mode_row('square',model,g,local_spec,mode,wave,201)
            far_axis,fx,fy,widths,qe,qg=data
            row.update(wavelength_nm=wave,closed_TE_family=0,frozen_rank=rank,reasons=reasons,
                frozen_eigen_residual=residual,frozen_target_distance_m1=abs(seed_value-target),
                frozen_reconstruction_attempts=search_history,
                frequency_trace=trace,frequency_residual_per_m=trace[-1]['frequency_residual_per_m'],
                group_velocity_m_s=backend.c/model.result.group_index,group_index=model.result.group_index,
                seconds=time.monotonic()-started)
            records.append(row)
            arrays.update({name+'_field':mode.fields,name+'_far_x':fx,name+'_far_y':fy,
                name+'_radiation_x':mode.radiation_x,name+'_radiation_y':mode.radiation_y,
                name+'_C':model.result.coupling_m,name+'_Lup':model.upward_amplitude_loss_m,
                name+'_Ldown':model.downward_amplitude_loss_m,name+'_qgain':qg,name+'_qenergy':qe,
                name+'_widths':widths,name+'_lateral':backend.lateral_energy_weights(mode.fields,31),
                'angle_deg':far_axis})
            np.savez_compressed(rawpath,**arrays);backend.save(path,records)
            print('cross_branch',folder.name,rank,wave,row['required_material_gain_cm1'],flush=True)
        except Exception as error:
            failures.append(dict(frozen_rank=rank,reasons=reasons,error=repr(error)))
            print('cross_branch_failed',folder.name,rank,repr(error),flush=True)
        backend.save(out/'failures.json',failures)
    backend.save(out/'summary.json',dict(status='completed' if not failures else 'partial',
        source_case=str(folder),closed=len(records),failed=failures,
        scope='Supplementary actual-frequency roots; merge only after original-pencil and independent-mode checks.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case',type=Path)
    challenge(parser.parse_args().case)
