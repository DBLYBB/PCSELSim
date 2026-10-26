"""Consolidate the independent triangular-PCSEL rerun without altering old data.

中文：汇总实际闭频、完整粗谱/有界细谱与非对齐网格检查，保留失败而不把失败
算作模式消失。只写本轮目录；小幅备选不冒充已通过完整独立谱复核的新主方案。
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
OUT=ROOT/'results/current/strict_rerun_20261025/semiconductor_triangular'
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def compact(case):
    s=load(OUT/case/'summary.json');low=s['lowest'];second=s['second']
    return dict(case=case,path=str(OUT/case/'summary.json'),N=s['N'],D=s['D'],dz_nm=s['dz_nm'],
        roots_closed=s['roots_closed'],period_nm=s['three_hole_spec']['lattice_nm'],wavelength_nm=low['wavelength_nm'],
        g1_cm1=low['g_th_cm'],g2_cm1=second['g_th_cm'],relative_material_gap=s['g_th_gap_fraction'],
        beam=low['beam'],polarization=low['polarization'],
        maximum_KM_residual=max(r['eigen_residual'] for r in load(OUT/case/'frequency_closed/frequency_closed_poles.json') if r['status']=='completed'))

def main():
    selected=load(OUT/'selected_case.json');primary=compact('baseline_N91')
    controls=[compact(n) for n in ('baseline_nonaligned_N90','baseline_order64_N91','baseline_contour27_N91')]
    for row in controls:row['relative_g1_difference']=row['g1_cm1']/primary['g1_cm1']-1
    modest=[compact(n) for n in ('modest_d2p003_N91','modest_d2p003_N90')]
    aggressive=[compact(n) for n in ('promoted_precheck_N91','promoted_precheck_N90')]
    for row in modest+aggressive:row['relative_g1_improvement']=1-row['g1_cm1']/primary['g1_cm1']
    coarse=load(OUT/'baseline_complete_QR_N13/summary.json');filters=load(OUT/'baseline_complete_QR_N13/filter_counts.json')
    fine=load(OUT/'baseline_QR_guided_N39/result.json');mesh=load(OUT/'baseline_QR_guided_N91/result.json')
    stripe=load(OUT/'baseline_stripe_N39/result.json');repeat=load(OUT/'baseline_stability_N39/summary.json')
    supplied=np.load(OUT/'baseline_N91/seeds/complex_fields.npz')['eigenvalues']
    challenged=np.load(OUT/'baseline_QR_guided_N91/complex_fields.npz')['eigenvalues']
    errors=[float(np.min(abs(challenged-v))) for v in supplied[:6]]
    if max(errors)>1e-3:raise RuntimeError('Fresh wide spectrum no longer matches the six actually closed branches')
    failures=[];scans=[]
    for label in ('local_scan','highD_local_scan','highD_extension_scan'):
        rows=load(OUT/label/'candidates.json');status={}
        for row in rows:
            status[row['status']]=status.get(row['status'],0)+1
            if row['status']!='completed':failures.append(dict(scan=label,label=row['label'],status=row['status'],error=row.get('error')))
        scans.append(dict(scan=label,attempts=len(rows),status_counts=status,path=str(OUT/label/'candidates.json')))
    old_path=ROOT/'results/current/matched300_20261021/triangular/formal_low_gain_verified_N91/summary.json'
    old=load(old_path);old_union=ROOT/'results/current/matched300_20261021/triangular/low_gain_union_N91/complex_fields.npz'
    old_hashes=dict(old_summary_sha256=digest(old_path),old_union_sha256=digest(old_union))
    for path in OUT.glob('runtime_*.json'):
        run=load(path)
        if any(run.get(k)!=v for k,v in old_hashes.items()):raise RuntimeError('Original frozen baseline changed during this task')
    spectral=dict(complete_coarse_N=13,complete_coarse_pole_count=coarse['retained_passive_poles'],
        full_matrix_dimension=coarse['matrix_dimension'],coarse_solver='ordinary QR of exact H=K M^-1, independently checked against original generalized QZ',
        coarse_filter_counts=filters,QR_QZ_validation_path=str(OUT/'QR_QZ_check/validation.json'),
        fine_N39_complex_targets=80,fine_N39_distinct_poles=fine['retained_poles'],
        fine_N91_continued_low_poles=mesh['retained_poles'],N91_original_KM_max_residual=max(r['relative_eigen_residual'] for r in mesh['spectrum']),
        supplied_closed_six_to_fresh_N91_max_pole_error_per_m=max(errors),
        fine_N39_independent_stripe_targets=12,fine_N39_stripe_distinct_poles=stripe['retained_poles'],
        fine_N39_stripe_minimum_g_cm1=stripe['lowest']['g_th_cm'],
        fine_N91_direct_stripe=load(OUT/'baseline_stripe_N91/bounded_unresolved.json'),
        repeats=repeat['attempts'],repeats_completed=sum(r['status']=='completed' for r in repeat['rows']),
        maximum_repeat_pole_error_per_m=max(r['pole_error_absolute_per_m'] for r in repeat['rows'] if r['status']=='completed'),
        maximum_repeat_KM_residual=max(r['original_pencil_relative_residual'] for r in repeat['rows'] if r['status']=='completed'),
        maximum_low24_pairwise_field_overlap=repeat['maximum_pairwise_field_overlap'],
        identity='Distinct complex poles are retained even with near-collinear fields; only same-pole repeated spans are merged.',
        scope='Complete coarse spectrum and bounded fine-grid challenges; no exhaustive fine-grid or full-Maxwell spectral proof.')
    goals=dict(maximum_axis_ratio=1.05,minimum_center_to_peak=.98,maximum_radial_anisotropy_l2=.10,minimum_material_gain_gap=.10)
    all_controls=[primary]+controls
    observed=dict(g1_range_cm1=[min(r['g1_cm1'] for r in all_controls),max(r['g1_cm1'] for r in all_controls)],
        g2_range_cm1=[min(r['g2_cm1'] for r in all_controls),max(r['g2_cm1'] for r in all_controls)],
        first_mode_axis_range=[min(r['beam']['ellipticity'] for r in all_controls),max(r['beam']['ellipticity'] for r in all_controls)],
        smallest_tested_material_gap=min(r['relative_material_gap'] for r in all_controls),
        interpretation='Observed grid/harmonic/vertical-step/contour sensitivity, not a statistical confidence interval or complete model uncertainty. No area-based loss correction.')
    decision=dict(primary_geometry='same as previous canonical main case',geometry_changed=False,
        reason='The full rerun reproduces the old selected case. A modest d2+0.003a change improves nominal threshold by about1.65%, but shifts a satellite by only0.96nm, below the assumed3nm placement budget, and lacks its own complete independent fine census. It is retained as a nominal Pareto point rather than replacing the main report.',
        modest_nominal_improvement_fraction=modest[0]['relative_g1_improvement'],
        modest_satellite_displacement_nm=.003*selected['three_hole_spec']['lattice_nm'],
        assumed_placement_budget_nm=3.,aggressive_rejection='d2+0.0075a has a branch-order swap on N91 and only9.27% tested gap; coarse12.2% gap is not used.',
        previous_g1_cm1=old['lowest']['g_th_cm'],new_g1_cm1=primary['g1_cm1'],
        g1_change_cm1=primary['g1_cm1']-old['lowest']['g_th_cm'],
        PPT_metric_change_required=False,scope='Nominal static CWT design choice; no fabrication or above-threshold mode-stability validation.')
    qualification=dict(status='selected_static_CWT_case_with_disclosed_spectral_and_mesh_limits',goals=goals,
        selected=primary,numerical_controls=controls,observed_sensitivity=observed,selection_decision=decision,
        modest_Pareto=modest,aggressive_Pareto=aggressive,spectral_audit=spectral,scans=scans,
        local_failures=failures,old_inputs_unchanged=True,old_input_sha256=old_hashes,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'scripts/optimization/strict_rerun_sem_tri_20261025.py',
            ROOT/'scripts/optimization/matched300_triangular_qz_guided.py',ROOT/'scripts/optimization/matched300_triangular_close.py',
            ROOT/'src/pcselsim/finite_discretization.py',ROOT/'src/pcselsim/mode_identity.py',ROOT/'src/pcselsim/triangular_six_wave.py',ROOT/'src/pcselsim/band_structure.py']})
    save('qualification.json',qualification)
    selected.update(status='completed',case_id='semiconductor_triangular_strict_20261025',material='semiconductor',lattice='triangular',
        fields_path=str(ROOT/selected['closed_fields_path']),grid=selected['N'],actual_period_nm=selected['three_hole_spec']['lattice_nm'],
        wavelength_nm=selected['lowest']['wavelength_nm'],g_th_cm1=selected['lowest']['g_th_cm'],
        relative_material_gap=selected['g_th_gap_fraction'],frequency_closed_modes=selected['modes'],
        qualification_path=str(OUT/'qualification.json'),qualification=qualification,
        parameter_scan=dict(attempts=sum(r['attempts'] for r in scans),scans=scans),selection_decision=decision)
    save('selected_case.json',selected)
    rows=['# 半导体三角晶格：严格复算与局部优化（2026-10-25）','',
        '主汇报保留旧几何。严格复算后首模、阈值及光斑与旧正式结果一致；新扫描的小幅改善作为名义Pareto，不强行替换。',
        f"300×300μm方形物理边界；a={selected['three_hole_spec']['lattice_nm']:.9f}nm；D48、dz1nm、N91。实际首模λ={primary['wavelength_nm']:.9f}nm，材料阈值{primary['g1_cm1']:.8f}cm⁻¹，已检模式材料增益间隔{100*primary['relative_material_gap']:.5f}%。",
        f"首模轴比{primary['beam']['ellipticity']:.6f}、中心/峰值{primary['beam']['center_to_peak']:.6f}、径向L2非圆度{primary['beam']['radial_anisotropy_l2']:.6f}；轴上线偏振度{primary['polarization']['on_axis']['degree_of_linear_polarization']:.8f}。光斑圆形不等于圆偏振。",'',
        '## 实际闭频模式','', '| 模式 | λ/nm | 材料阈值/cm⁻¹ | 轴比 | 中心/峰值 |','|---|---:|---:|---:|---:|']
    for i,r in enumerate(selected['modes']):rows.append(f"| M{i+1} | {r['wavelength_nm']:.8f} | {r['g_th_cm']:.6f} | {r['beam']['ellipticity']:.5f} | {r['beam']['center_to_peak']:.5f} |")
    rows+=['','6根都分别重建C(λ)、纵向模与原有限K/M后闭合实际频率；保持同一几何与周期。不是每个Γ族挑一根，也不是把冻结C的频率估计标为闭频。原始复NF、Jones远场、辐射包络、C及偏振均可按selected_case.json索引。','',
        '## 谱与数值复核','',
        f"N13完整4926维粗谱；QR与独立小网格原广义QZ逐根匹配。N39的80个复移位得到{fine['retained_poles']}个独立极点；最低12根在N91重新求解，包含旧6根，最大极点差{max(errors):.3g}/m。N39另12个高横向条纹挑战得到{stripe['retained_poles']}个独立极点，最低材料阈值{stripe['lowest']['g_th_cm']:.5f}cm⁻¹，没有推翻主模。",
        'N91直接条纹首target约17分钟未返回而按墙钟预算中止；未返回不表示没有模式，保留bounded_unresolved.json并明确不声称完整精细谱。6根×3扰动移位/初场重复均通过原残差核验；不同复极点不会因场相似被删去。','',
        '| 控制（固定物理周期） | g₁/cm⁻¹ | λ₁/nm | g₂/cm⁻¹ | 间隔/% |','|---|---:|---:|---:|---:|']
    for r in all_controls:rows.append(f"| {r['case']} | {r['g1_cm1']:.6f} | {r['wavelength_nm']:.8f} | {r['g2_cm1']:.6f} | {100*r['relative_material_gap']:.4f} |")
    rows+=['','N90不与N91/78/65共享13倍数对齐；两根实际闭频，首模及最近竞争枝都检查。误差区间是已测离散/截断/轮廓敏感性，不是全部模型误差；不按面积修正损耗。','',
        '## 新局部扫描与决策','',
        f"跨三级精度共{sum(r['attempts'] for r in scans)}次参数试算，覆盖主/副孔位置、面积分配、形状与有效朝向；圆孔朝向、普通正三角形aspect是无效元数据，不作为独立形变自由度。失败记录保留，不作模式消失证据。",
        f"d2+0.003a：N91三根已知枝闭频g₁={modest[0]['g1_cm1']:.6f}、g₂={modest[0]['g2_cm1']:.6f}，名义首模比旧低{100*modest[0]['relative_g1_improvement']:.4f}%；N90对应g₁={modest[1]['g1_cm1']:.6f}。位置仅移动{decision['modest_satellite_displacement_nm']:.4f}nm，小于既定3nm工艺位置预算，且无自身完整独立谱，所以列为名义Pareto。",
        f"d2+0.0075a：N39看似更低，但N91竞争枝次序交换；已闭频g₁={aggressive[0]['g1_cm1']:.6f}、g₂={aggressive[0]['g2_cm1']:.6f}，间隔仅{100*aggressive[0]['relative_material_gap']:.4f}%，低于10%目标。不得沿用粗筛约12.2%的间隔。",'',
        '## 模型范围与复现','',
        '阈值为(2α_field+内部强度损耗5cm⁻¹)/(Γn_active/n_eff)。这是材料增益阈值，区别于振幅损耗、阈值电流和泵浦能量；归一化保持声明的近Bragg模型近似。仍为标量纵向TE0/六波CWT、有限高阶谐波与阶梯方形边界，不是完整Maxwell、制造误差、热效应或阈值以上非线性单模证明。',
        '主入口scripts/optimization/strict_rerun_sem_tri_20261025.py；最终整理scripts/diagnostics/finalize_strict_sem_tri_20261025.py。old frozen输入只读且SHA不变。qualifications、完整失败、来源SHA及每次执行快照均在本目录。PPT资产及其工作脚本仅位于会议工作区。']
    (OUT/'report_zh.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    (OUT/'README.md').write_text('''# Strict triangular-PCSEL rerun / 严格三角晶格复算

Read `selected_case.json` first; it points to the six individually frequency-closed complex modes, physical geometry, local bands, Jones polarization, and `qualification.json`. The selected geometry is unchanged from the previous canonical main case.

`report_zh.md` explains the decision and numerical limits in Chinese. `qualification.json` preserves all scan counts/failures, full-coarse and bounded-fine spectral evidence, QR/QZ equivalence, repeated-solve residuals, and actual N90/N91/D64/contour controls. The `modest_d2p003_*` folders contain only three tracked actual branches, and do not certify a complete new first-mode spectrum.

The main driver is `scripts/optimization/strict_rerun_sem_tri_20261025.py` (`known`, `local`, `retune`, `stripe-audit`, `stability`, `package`). Coarse/fine census uses the audited `matched300_triangular_qz_guided.py`; interpolation is only an eigensolver initial condition. Every displayed field solves the current original K/M. `finalize_strict_sem_tri_20261025.py` consolidates the completed files without altering old results.

The N91 direct stripe attempt exceeded its explicit wall-clock budget and remains unresolved. It is not evidence of mode absence. The completed N39 stripe search and 12-pole N91 continuation are bounded tests, not an exhaustive fine-grid or full-Maxwell spectral proof. Beam circularity concerns intensity; on-axis Jones polarization is reported separately.

All cavities have a nominal 300 x 300 um physical square boundary. Do not rescale loss by sampled node area. Internal5/cm is an intensity loss; finite Im(delta) is a field-amplitude loss per metre. The material gain weight is Gamma*n_active/n_eff within the declared near-Bragg approximation.

PPT files, plotting-workspace scripts and report assets are outside PCSELSim, in the meeting workspace. Existing canonical results were read-only and their SHA256 values remain unchanged.
''',encoding='utf-8')
    print('FINALIZED',primary['g1_cm1'],len(selected['modes']),sum(r['attempts'] for r in scans),'attempts')

if __name__=='__main__':main()
