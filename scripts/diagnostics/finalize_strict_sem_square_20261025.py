"""Package the post-audit square-PCSEL evidence without touching old results.

中文：汇总本轮正式设计、独立复核、参数来源和被拒绝候选；不制作PPT。
All paths in selected_case.json are explicit and all beam maps derive from
saved complex fields. Packaging does not change any eigenfield or gain.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import sys,json,hashlib
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from pcselsim.polarization import stokes_maps
from scripts.optimization.strict_rerun_sem_square_20261025 import OUT,save,old_case
from scripts.optimization.matched300_square_search import Design,build_cell,BASE
from scripts.optimization.square_semiconductor_search import build_stack


def read(name,file='summary.json'):
    path=OUT/name/file
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary_row(name,role):
    s=read(name);row=s['lowest']
    if not s.get('strict_readback'):raise RuntimeError(f'{name} still needs strict field readback')
    return dict(role=role,summary_path=str(OUT/name/'summary.json'),fields_path=str(OUT/name/'final_fields.npz'),
        design=s['design'],D=s['D'],dz_nm=s['dz_nm'],grid=s['grid'],wavelength_nm=row['wavelength_nm'],
        g_th_cm1=row['g_th_cm1'],relative_material_gap=s['relative_material_gap'],beam=row['beam'],
        polarization=row['polarization'],mode_count=len(s['frequency_closed_modes']),
        maximum_original_pencil_residual=max(x['original_pencil_residual'] for x in s['strict_readback']),
        maximum_frequency_residual_per_m=max(abs(x['true_frequency_residual_per_m']) for x in s['strict_readback']),
        closure_failures=s.get('closure_failures',[]))


def main():
    primary_name='candidate041_closed_N65';primary=read(primary_name)
    names=[('baseline_closed_N65','old_primary_revalidated'),
        (primary_name,'selected_lower_gain_with_10pct_gap'),
        ('candidate013_closed_N65','lower_radial_anisotropy_alternative'),
        ('old_low_closed_N65','lowest_gain_pareto_small_gap'),
        ('old_balanced_closed_N65','historical_roundness_pareto_small_gap'),
        ('candidate041_fixed_D48_N65','fixed_geometry_D48_control'),
        ('candidate041_fixed_D80_N65','fixed_geometry_D80_control'),
        ('candidate041_finalperiod_closed_N81','fixed_geometry_N81_control')]
    comparisons=[summary_row(name,role) for name,role in names]
    reference=comparisons[0];winner=comparisons[1]

    # Complete Stokes maps for all saved primary Jones fields. Pure derived
    # quantities; preserve the exact solved near fields and Jones amplitudes.
    # 为全部正式闭频态补全Stokes，只派生已有Jones，不改变模式或归一化。
    path=OUT/primary_name/'final_fields.npz'
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    modes=primary['frequency_closed_modes']
    for row in modes:
        n=row['mode'];arrays.update({n+'_stokes_'+key:value for key,value in stokes_maps(arrays[n+'_far_x'],arrays[n+'_far_y']).items()})
    np.savez_compressed(path,**arrays)
    primary['saved_stokes_map_modes']=[x['mode'] for x in modes]
    save(OUT/primary_name/'summary.json',primary)

    d=Design(**primary['design']);cell=build_cell(d,96)
    config=dict(material='semiconductor InGaAs/AlGaAs',lattice='square',design=asdict(d),
        actual_period_nm=d.lattice_nm,physical_aperture_um=[300.,300.],
        aperture_boundary='zero incoming envelopes on the 300um square; not a carrier/current-spreading calculation',
        stack=asdict(build_stack(d,cell)),background_index=3.554,hole_index=1.,
        physics_source='Inoue et al., Physical Review B 99, 035308 (2019), Tables I-II; scalar TE0 / four-wave projected CWT',
        source_entry=str(ROOT/'scripts/run_square_double_hole.py'),source_entry_sha256=sha(ROOT/'scripts/run_square_double_hole.py'),
        geometry_source=str(ROOT/'scripts/optimization/matched300_square_search.py'),
        geometry_active_fields=['fill','main_share','distance','position_angle_deg','main_shape','satellite_shape','satellite_aspect','satellite_angle_deg','lattice_nm'],
        inactive_metadata_in_selected_geometry=['main_aspect (circle)','main_angle_deg (circle)','main_exponent (circle)','satellite_exponent (fixed Minkowski rounded triangle)'],
        rounded_triangle_definition='area-centroid normalized triangle plus disk of radius0.35 in shape units, aspect stretch, area normalization and rotation; 96 contour samples',
        numerical=dict(D=64,dz_nm=1.,grid=65,contour_samples=96,angle_window_deg=[-1.,1.],angle_samples=241,BLAS_threads=1),
        internal_intensity_loss_cm1=BASE.INTERNAL_LOSS_CM,active_index=BASE.ACTIVE_INDEX,
        semiconductor_gain_parameters=dict(maximum_gain_cm1=BASE.MAXIMUM_GAIN_CM,zero_carrier_gain_cm1=BASE.ZERO_CARRIER_GAIN_CM,
            transparency_density_cm3=BASE.TRANSPARENCY_DENSITY_CM3,carrier_lifetime_ns=BASE.CARRIER_LIFETIME_NS,active_thickness_nm=BASE.ACTIVE_THICKNESS_NM),
        threshold_formula='g_material=(2*alpha_field_cm1+5)/(Gamma*n_active/n_eff); current is uniform-pump empirical gain-law estimate',
        strict_beam_objectives=dict(maximum_axis_ratio=1.05,minimum_center_to_peak=.98,maximum_radial_anisotropy_l2=.10,minimum_material_gap=.10),
        approximations=['scalar TE0 vertical projection and near-Bragg four-wave CWT','finite high-order harmonic truncation',
            'bounded fine-grid spectrum, complete N17 coarse QZ only','static cold thresholds; no nonlinear single-mode guarantee','ideal deterministic contours; no fabrication or thermal validation'])
    save(OUT/'selected_config.json',config)

    surveys={}
    for dirname in ('baseline_N65','candidate041_N65','candidate013_N65','candidate041_finalperiod_N81','old_low_N65','old_balanced_N65','lowpareto060_N41'):
        s=read(dirname,'spectral_challenge.json')
        surveys[dirname]=dict(distinct_fine_poles=s['distinct_poles'],coarse_grid=s['coarse_grid'],complete_coarse_QZ_count=s['complete_coarse_QZ_count'],
            target_count=len(s['search_targets']),failed_targets=[r for r in s['search_targets'] if 'error' in r],
            spectrum_json=str(OUT/dirname/'spectral_challenge.json'),scope=s['scope'])
    screen=read('screen','records.json');rejected=read('lowpareto060_N41','spectral_challenge.json')
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'scripts/optimization/strict_rerun_sem_square_20261025.py',Path(__file__),
        ROOT/'scripts/optimization/matched300_square_finalize.py',ROOT/'scripts/optimization/matched300_square_search.py',ROOT/'scripts/run_square_double_hole.py',
        ROOT/'src/pcselsim/mode_identity.py',ROOT/'src/pcselsim/finite_discretization.py',ROOT/'src/pcselsim/frequency_cwt.py',ROOT/'src/pcselsim/band_structure.py']}
    numerical_controls=[]
    for row in comparisons[5:]:
        numerical_controls.append({**row,'gain_relative_difference_from_selected':row['g_th_cm1']/winner['g_th_cm1']-1,
            'wavelength_difference_nm':row['wavelength_nm']-winner['wavelength_nm']})
    payload=dict(status='completed',case_id='semiconductor_square_strict_20261025',material='semiconductor',lattice='square',
        summary_path=str(OUT/primary_name/'summary.json'),fields_path=str(path),geometry_path=str(OUT/primary_name/'actual_geometry_nm.json'),
        config_path=str(OUT/'selected_config.json'),science_figures={n:str(OUT/primary_name/n) for n in ('closed_modes.png','final_overview.png','local_bands.png')},
        design=primary['design'],actual_period_nm=d.lattice_nm,D=64,dz_nm=1.,grid=65,physical_aperture_um=[300.,300.],
        target_wavelength_nm=950.,target_tolerance_nm=.05,wavelength_nm=primary['lowest']['wavelength_nm'],actual_target_pass=primary['actual_target_pass'],
        lowest=primary['lowest'],g_th_cm1=primary['lowest']['g_th_cm1'],relative_material_gap=primary['relative_material_gap'],
        beam=primary['lowest']['beam'],polarization=primary['lowest']['polarization'],frequency_closed_modes=modes,
        fields=dict(file=str(path),sha256=sha(path),axis_key='axis_deg',near_field_pattern='{mode}_field',near_field_shape=[4,65,65],
            Jones_x_pattern='{mode}_far_x',Jones_y_pattern='{mode}_far_y',Jones_shape=[241,241],radiation_x_pattern='{mode}_radiation_x',
            radiation_y_pattern='{mode}_radiation_y',coupling_matrix_pattern='{mode}_coupling_m',Stokes_pattern='{mode}_stokes_{quantity}',
            Stokes_quantities=list(stokes_maps(arrays[modes[0]['mode']+'_far_x'],arrays[modes[0]['mode']+'_far_y'])),
            mode_keys=[r['mode'] for r in modes],normalization='near fields are numerical eigenvectors; Jones components share one scale, no independent component normalization'),
        strict_readback=primary['strict_readback'],mode_independence_path=str(OUT/primary_name/'mode_independence_audit.json'),
        comparisons=comparisons,numerical_controls=numerical_controls,spectral_challenges=surveys,
        parameter_scan=dict(count=len(screen),completed=sum(r['status']=='completed' for r in screen),grid=9,D=12,dz_nm=2.,records_path=str(OUT/'screen/records.json'),
            interpretation='screening heuristics only; N9 +/-1degree beam not envelope-Nyquist qualified'),
        rejected_low_gain_counterexample=dict(design=rejected['design'],lowest=rejected['lowest'],relative_material_gap=rejected['relative_material_gap'],
            reason='A lower dark high-transverse stripe beats the attractive coarse bright branch; rejected as a bright-first-mode design'),
        historical_pareto_verdict='Both old57.628 and58.248 designs remain tested bright-first candidates; only1.122% and1.764% checked gain isolation, so retain as low-threshold Pareto and do not claim robust single-mode selection.',
        improvement=dict(relative_material_gain_reduction=1-winner['g_th_cm1']/reference['g_th_cm1'],
            relative_threshold_current_reduction=1-primary['lowest']['I_th_A']/read('baseline_closed_N65')['lowest']['I_th_A'],
            caveat='axis ratio and gain improve; center/peak and radial L2 are slightly worse than old primary. Current includes transparency-density floor and improves much less than gth.'),
        source_sha256=sources,driver_path=str(ROOT/'scripts/optimization/strict_rerun_sem_square_20261025.py'),
        driver_sha256=sources['scripts/optimization/strict_rerun_sem_square_20261025.py'],
        execution_snapshot_paths=[str(p) for p in sorted(OUT.glob('runtime_*.json'))],
        scope='Post-audit bounded finite-mode validation under scalar TE0 near-Bragg CWT. No full-Maxwell, thermal, fabrication, or nonlinear single-mode certification.')
    save(OUT/'selected_case.json',payload);save(OUT/'summary.json',payload)

    rows=['# 严格审查后半导体正方PCSEL复算（2026-10-25）','',
        f'在固定300×300μm物理边界、文献层栈和同一损耗定义下，新正式方案材料阈值从{reference["g_th_cm1"]:.6f}降至{winner["g_th_cm1"]:.6f}cm⁻¹，降低{100*payload["improvement"]["relative_material_gain_reduction"]:.2f}%。',
        f'实际首模950.000000nm；a={d.lattice_nm:.9f}nm；已闭频最近竞争模的材料阈值间隔{100*winner["relative_material_gap"]:.4f}%。这是已挑战谱内的间隔，不是全谱或非线性单模证明。','',
        '## 正式方案与权衡','',
        f'中心主孔为圆孔，副孔为面积归一化的圆角三角形。总填充比{d.fill:.9f}，主孔面积份额{d.main_share:.9f}；孔心距{d.distance:.9f}a，位置角{d.position_angle_deg:.6f}°；副孔伸长比{d.satellite_aspect:.6f}、转角{d.satellite_angle_deg:.6f}°。实际轮廓保存在actual_geometry_nm.json。圆主孔的角度/伸长比和圆角三角形的exponent只是继承元数据，不是独立生效参数。',
        f'首模远场轴比{winner["beam"]["ellipticity"]:.6f}，中心/峰值{winner["beam"]["center_to_peak"]:.6f}，径向L2非圆度{winner["beam"]["radial_anisotropy_l2"]:.6f}。阈值与轴比优于旧主方案，中心/峰值及径向L2略逊，因此不宣称全指标支配。013方案保留为径向轮廓更圆的备选。',
        '阈值电流为均匀泵浦和经验增益逆式的估计，透明载流子密度占主导，材料增益降低11%左右并不表示电流同步降低11%。','',
        '| 方案 | 材料阈值/cm⁻¹ | 最近间隔/% | 轴比 | 中心/峰值 | 径向L2 |','|---|---:|---:|---:|---:|---:|']
    for row in comparisons[:5]:
        b=row['beam'];rows.append(f'| {row["role"]} | {row["g_th_cm1"]:.6f} | {100*row["relative_material_gap"]:.4f} | {b["ellipticity"]:.6f} | {b["center_to_peak"]:.6f} | {b["radial_anisotropy_l2"]:.6f} |')
    rows+=['','## 数值与身份复核','',
        '96个新参数组合覆盖填充比、面积分配、孔心距/角度、副孔形貌/伸长/方向，以及主孔椭圆形变；完整N9粗QZ只筛选，粗网格±1°光斑不用于正式验收。',
        '正式N65采用完整N17粗谱引导、Γ邻域、较宽实轴和独立高横向条纹挑战；最终调节周期后，再用N81独立宽谱和固定物理几何闭频检验。所有模式保持原K/M残差，不通过改场或去除暗模提升分数。','',
        '| 固定几何控制 | g₁/cm⁻¹ | λ₁/nm | 对正式阈值差/% |','|---|---:|---:|---:|']
    for r in numerical_controls:rows.append(f'| D{r["D"]}, dz={r["dz_nm"]}nm, N{r["grid"]} | {r["g_th_cm1"]:.6f} | {r["wavelength_nm"]:.9f} | {100*r["gain_relative_difference_from_selected"]:+.5f} |')
    rows+=['','复本征值已分辨时即使场高度重叠也保留；只有同算符、同数值复极点簇才用SVD子空间去重。10根正式闭频态保存各自近场、Jones、辐射场、C矩阵及Stokes，逐根重建原算符并读回验证。','',
        '## 旧Pareto与失败记录','',
        '旧57.628和58.248方案分别经205和202根独立细网格极点挑战，仍为本次找到的亮首模；最近条纹竞争模只高约1.122%和1.764%，可以保留低阈值权衡，不能作为≥10%隔离的正式方案。旧低阈值方案一根较高分支跟踪失败并已记录，未把失败算作模式消失。',
        f'新候选060在粗网格看似低阈值亮圆，但细网格首模为暗条纹：g={rejected["lowest"]["g_th_cm1"]:.6f}cm⁻¹、轴比{rejected["lowest"]["beam"]["ellipticity"]:.3f}、中心/峰值{rejected["lowest"]["beam"]["center_to_peak"]:.3f}，因此拒绝。',
        '每条细谱的补充shift失败和真实闭频失败保存在spectral_challenges、closure_failures中。N81有一个Γ补充shift未收敛，其他目标重复挑战并未找出更低模式；仍明确保留有界谱搜索的限制。','',
        '## 偏振和适用范围','',
        f'首模轴上归一化(S1,S2,S3)={primary["lowest"]["polarization"]["on_axis"]["normalized_stokes_s1_s2_s3"]}，接近线偏振；光斑接近圆形不意味着圆偏振。约定S3=+2Im(ExEy*)，不附加左右旋命名。',
        '模型仍为标量纵向TE0投影及近Bragg四波CWT，且包含有限谐波截断。此处只汇报静态冷腔，不包含热效应、制造误差、完整Maxwell验证及阈值以上的非线性单模稳定性。全部源参数、驱动SHA、实际几何、原方程残差与原始复场索引见selected_case.json。']
    (OUT/'report_zh.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print(json.dumps(dict(selected=primary_name,gth=winner['g_th_cm1'],gap=winner['relative_material_gap'],case_path=str(OUT/'selected_case.json')),ensure_ascii=False))


if __name__=='__main__':main()
