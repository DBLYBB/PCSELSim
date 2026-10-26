"""Summarize actual fixed-motif size points, preserving censored thresholds.

中文：不插值缺失/不可达阈值，不按1/L拟合损耗。所有图只是已求解点，
1030±1nm条件子集与总已检谱分开。300um主比较尺寸不变。
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,csv
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=ROOT/'results/current/strict_rerun_20261025/yag_feasibility'
SOURCE=ROOT/'results/current/strict_rerun_20261025/YbYAG_square/candidate_01_full32/merged_summary.json'
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def compact(path):
    r=load(path);modes=sorted(r.get('modes',[]),key=lambda m:m['material_gain_cm1'])
    target=[m for m in modes if 1029.<=m['wavelength_nm']<=1031.]
    def pump(source):
        out={}
        for arch,data in source.items():
            names=data.get('first_modes',[])
            out[arch]=dict(first_threshold_internal_W=data.get('first_threshold_internal_W'),first_modes=names,
                wavelengths_nm=[next(m['wavelength_nm'] for m in modes if m['name']==name) for name in names],
                interpretation=data.get('interpretation',data.get('status')))
        return out
    attempts=r.get('census',{}).get('attempts',[])
    return dict(path=str(path),sha256=sha(path),status=r['status'],side_um=r['side_um'],order=r['order'],dz_nm=r['dz_nm'],grid=r['grid'],
        period_nm=r['period_nm'],mode_count=len(modes),lowest_returned_material_mode=modes[0] if modes else None,
        target_window_lowest_material_mode=target[0] if target else None,target_window_nm=[1029.,1031.],
        target_window_mode_count=len(target),pumps=pump(r.get('pumps',{})),target_window_pumps=pump(r.get('target_band_pumps',{})),
        closure_failures=r.get('closure_failures',[]),search_target_failures=[a for a in attempts if 'error' in a],
        census=r.get('census'),physically_invalid_rows=r.get('physically_invalid_rows',[]),
        anchor=r.get('anchor',False),original_field_readback=r.get('strict_original_field_readback',[]))

def main():
    anchor=OUT/'sizes/square_L300_formal_merged40_anchor.json'
    coarse=[compact(OUT/f'sizes/square_L{length}_D12_dz2_N25_K8.json') for length in (50,100,200,500,750)]
    formal=compact(anchor)
    controls=[compact(OUT/f'sizes/square_L{length}_D32_dz1_N49_K10.json') for length in (100,200,500)]
    allpoints=coarse+[formal]+controls;source=load(SOURCE)
    if any(r['period_nm']!=source['period_nm'] for r in allpoints):raise RuntimeError('Physical period changed in size scan')
    if any(r['physically_invalid_rows'] for r in allpoints):raise RuntimeError('Invalid fields must not contribute to a curve')
    for point in allpoints:
        for pump in point['pumps'].values():
            if pump['first_threshold_internal_W'] is None and pump.get('interpretation')!='unreachable_under_nominal_pump_inversion_ceiling':
                raise RuntimeError('An unresolved pump root must be labelled separately from an analytically unreachable point')
    comparisons=[]
    for high in controls:
        low=next(r for r in coarse if r['side_um']==high['side_um'])
        comparisons.append(dict(side_um=high['side_um'],coarse_path=low['path'],verification_path=high['path'],
            coarse_g_cm1=low['lowest_returned_material_mode']['material_gain_cm1'],
            verified_g_cm1=high['lowest_returned_material_mode']['material_gain_cm1'],
            relative_g_difference=high['lowest_returned_material_mode']['material_gain_cm1']/low['lowest_returned_material_mode']['material_gain_cm1']-1,
            coarse_lowest_wavelength_nm=low['lowest_returned_material_mode']['wavelength_nm'],
            verified_lowest_wavelength_nm=high['lowest_returned_material_mode']['wavelength_nm']))
    cap=1.38e21*((2.e-20+.126e-20)*(.70/(.70+.10))-.126e-20)
    readbacks=[r for point in allpoints for r in point['original_field_readback']]
    validation=dict(new_actual_root_readbacks=len(readbacks),reused_formal_anchor_roots=formal['mode_count'],
        maximum_original_KM_residual=max(r['original_KM_residual'] for r in readbacks),
        maximum_frequency_residual_per_m=max(abs(r['frequency_residual_per_m']) for r in readbacks),
        maximum_energy_error_per_m=max(abs(r['energy_error_per_m']) for r in readbacks),
        failed_bounded_search_targets=sum(len(point['search_target_failures']) for point in allpoints),
        closure_failures=sum(len(point['closure_failures']) for point in allpoints))
    summary=dict(status='completed',source_summary=str(SOURCE),source_summary_sha256=sha(SOURCE),validation=validation,
        design=source['design'],fixed_period_nm=source['period_nm'],formal_comparison_aperture_um=[300,300],
        size_variation_scope='Additional fixed-motif study; the four main square/triangular semiconductor/YAG cavities remain300x300um.',
        material_protocol='Nominal constant laser/pump cross sections only; no measured wavelength-resolved gain spectrum or reciprocity sensitivity used for these size curves.',
        nominal_pump_limited_material_gain_ceiling_cm1=cap,points=sorted(coarse+[formal],key=lambda r:r['side_um']),
        higher_accuracy_controls=controls,accuracy_comparisons=comparisons,
        target_window_interpretation='The1029–1031nm subset gives a conditional threshold. It is not a physical filter and never removes roots from the all-found-mode pump calculation.',
        negative_missing_unreachable_policy='No fabricated numbers, negative-loss acceptance,1/L loss scaling, period retuning or interpolation of unreachable/missing thresholds.',
        scope='Complete N11 coarse generalized spectra and bounded fine-grid TE0 searches followed by actual fixed-period closure. Missing branches remain recorded; no exhaustive fine-grid, full-Maxwell, fabrication, thermal or dynamic-stability proof.')
    (OUT/'sizes_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    table=[]
    for r in sorted(allpoints,key=lambda r:(r['side_um'],r['order'])):
        mode=r['lowest_returned_material_mode'];target=r['target_window_lowest_material_mode']
        table.append(dict(side_um=r['side_um'],D=r['order'],N=r['grid'],dz_nm=r['dz_nm'],closed_modes=r['mode_count'],
            minimum_returned_g_cm1=mode['material_gain_cm1'],minimum_returned_wavelength_nm=mode['wavelength_nm'],
            target_window_g_cm1=target['material_gain_cm1'] if target else None,
            edge_dual_internal_W=r['pumps'].get('edge_dual',{}).get('first_threshold_internal_W'),
            top1_internal_W=r['pumps'].get('top1',{}).get('first_threshold_internal_W'),
            top20_internal_W=r['pumps'].get('top20',{}).get('first_threshold_internal_W'),
            target_edge_dual_internal_W=r['target_window_pumps'].get('edge_dual',{}).get('first_threshold_internal_W'),
            unresolved_closures=len(r['closure_failures']),anchor=r['anchor']))
    with (OUT/'square_size_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=table[0].keys());writer.writeheader();writer.writerows(table)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.5),layout='constrained')
    sets=[(coarse,'o','D12 / N25 exploration'),(controls,'D','D32 / N49 verification'),([formal],'*','300 um formal anchor (40 modes)')]
    for points,marker,label in sets:
        axes[0].scatter([r['side_um'] for r in points],[r['lowest_returned_material_mode']['material_gain_cm1'] for r in points],
            marker=marker,s=80 if marker=='*' else 40,label=label)
    axes[0].axhline(cap,color='.45',ls='--',lw=1,label='Nominal pump-limited gain ceiling')
    axes[0].set(yscale='log',xlabel='Square side L (um)',ylabel='Lowest returned material gain (cm$^{-1}$)',title='Actual finite-cavity thresholds')
    axes[0].legend(fontsize=7,loc='upper right');axes[0].grid(alpha=.2)
    colors=dict(edge_dual='#0072B2',top1='#D55E00',top20='#009E73')
    for points,marker,_ in sets:
        for architecture,index,scale in (('edge_dual',1,1000.),('top1',2,1.),('top20',2,1.)):
            valid=[r for r in points if r['pumps'].get(architecture,{}).get('first_threshold_internal_W') is not None]
            axes[index].scatter([r['side_um'] for r in valid],
                [scale*r['pumps'][architecture]['first_threshold_internal_W'] for r in valid],
                marker=marker,s=80 if marker=='*' else 40,color=colors[architecture])
    for ax in axes[1:]:
        ax.text(.03,.60,'50-200 um: tested roots\nhave no finite nominal threshold',
            transform=ax.transAxes,va='top',fontsize=8,color='.35')
        ax.set_xlabel('Square side L (um)');ax.grid(alpha=.2)
    axes[1].set(title='Double-side first-entry pump',ylabel='Internal threshold (mW)')
    axes[2].set(title='Top first-entry pump',ylabel='Internal threshold (W)',yscale='log')
    axes[2].legend(handles=[Line2D([],[],marker='o',linestyle='',color=colors['top1'],label='Single pass'),
        Line2D([],[],marker='o',linestyle='',color=colors['top20'],label='20-pass ideal relay (98% return)')],fontsize=8,loc='center right')
    for ax in axes:
        ax.set_xlim(20,800);ax.set_xticks([50,100,200,300,500,750])
    fig.suptitle(f"Fixed square motif: a={source['period_nm']:.6f}nm | independently solved sizes",fontsize=12)
    fig.text(.5,-.015,'Only solved finite thresholds are plotted. The 1030±1nm conditional subset is reported separately in JSON/CSV; it is not a physical filter.',ha='center',fontsize=8)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    fig.savefig(figures/'square_fixed_period_size_thresholds.png',dpi=210,bbox_inches='tight');plt.close(fig)
    rows=['# 新YAG正方晶格：固定几何的尺寸—阈值复算','',
        f"同一候选01孔形、层栈和实际周期a={source['period_nm']:.9f}nm。只改变方腔边长，不给各尺寸重调到1030nm；四个主腔的比较仍保持300×300μm。300μm点直接采用经独立审查的40根合并正式结果，其他尺寸均重新求有限K/M和实际频率。",'',
        '材料为名义平截面模型。泵浦阈值是进入增益几何的总首入射内部功率，不是吸收功率，也不是实验外部源功率。top20只是98%返回效率的理想20程重成像假设。','',
        '| L/μm | D/N | 闭频根 | 最低已检g/cm⁻¹ | 对应λ/nm | 侧泵内部/W | 顶泵单程内部/W |','|---:|---|---:|---:|---:|---:|---:|']
    fnum=lambda x:'名义不可达' if x is None else f'{x:.8g}'
    for r in table:rows.append(f"| {r['side_um']:g} | {r['D']}/{r['N']} | {r['closed_modes']} | {r['minimum_returned_g_cm1']:.6f} | {r['minimum_returned_wavelength_nm']:.7f} | {fnum(r['edge_dual_internal_W'])} | {fnum(r['top1_internal_W'])} |")
    rows+=['','## 解释','',
        f'名义泵吸收/受激发射给出的反转上限为0.875，材料增益上限{cap:.5f}cm⁻¹。50/100/200μm已找到的所有分支都超过该上限，因此这些已检模式没有有限名义泵阈值。这不是用有限搜索证明全部Maxwell模式均不可激射。',
        '增大腔后已检冷腔材料阈值明显下降，但总输入泵功率还随入口面积、反转透明度底限和泵耗尽改变。尤其750μm的材料阈值更低，而顶泵总功率高于500μm，不能把低材料阈值直接换算成更低总泵能。',
        '50/100/200/500/750μm的D12、dz2nm、N25仅为探索；100/200/500μm另用D32、dz1nm、N49重新完整粗→有界细谱和实频闭合。最低模式可能随尺寸或精度交换，所有原始模式和未解失败记录保留，未按名字强行跟随亮支。','',
        '## 数值控制','', '| L/μm | 探索g/cm⁻¹ | 高精度g/cm⁻¹ | 差/% |','|---:|---:|---:|---:|']
    for r in comparisons:rows.append(f"| {r['side_um']:g} | {r['coarse_g_cm1']:.6f} | {r['verified_g_cm1']:.6f} | {100*r['relative_g_difference']:+.4f} |")
    rows+=['','N11完整粗广义谱只产生移位和初场；细网格每个保留根都解原K/M并闭合实际频率。各根另外重建自身C、辐射、增益权重和能流收支，保存NPZ供读回。没有用负损耗、拟合1/L、面积修正或插值补出结果。','',
        '## 1030nm窗口的含义','',
        'JSON中pumps使用全部已解独立根；target_band_pumps只报告1029–1031nm子集的条件阈值。窗口不是实际滤光器，也没有删除总谱中窗外竞争模。100μm高精度最小g根在1023.5nm，另保留1030.9nm根供条件比较；二者在名义材料下均不可达。',
        '材料截面保持常数只是一种条件假设，不能替代真实波长谱、热效应、外部泵耦合或完整矢量Maxwell求解。尺寸曲线只比较阈值，没有证明大尺寸首模仍具备300μm主方案的亮中心和近圆远场，也没有证明阈值以上动态单模稳定性。','',
        '文件：sizes_summary.json汇总已解点/误差与所有路径；square_size_summary.csv给出单位一致表；sizes/*.json和*.npz保存原模式、目标窗条件结果、失败和自身C/辐射。科学图位于figures/，不用于本次静态PPT增加动态页。']
    (OUT/'size_report_zh.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    (OUT/'SIZE_README.md').write_text('''# New square-YAG fixed-motif size study

Entry: `scripts/optimization/strict_yag_square_size_20261025.py`; summary: `sizes_summary.json`. The wrapper rebinds the audited `yag_feasibility_study.size_point` selection/output only. It never retunes the period at each size or rescales loss by1/L or sampled area.

The300um anchor explicitly reads new merged40-mode files. It does not call the old helper that loads32 modes by directory name. All other points are fresh complete-coarse/bounded-fine searches with actual fixed-period closures. Per-root original K/M, background-frequency residuals, energy identities, current C and radiation are saved.

Nominal flat spectroscopy only. `pumps` uses every retained root. `target_band_pumps` is a1029–1031nm conditional subset, not a physical spectral filter. None means unreachable/missing, never zero. The plot contains solved points only. D12/N25 exploration and D32/N49 controls are displayed separately, with a formal300um anchor. The main four-cavity comparison remains300x300um.

This folder also holds other feasibility work managed by the parent task. Size files use the `square_` prefix and `size_report_zh.md`; this script does not edit pulse-energy results or PPT files.
''',encoding='utf-8')
    print('SUMMARY',[(r['side_um'],r['verified_g_cm1']) for r in comparisons])

if __name__=='__main__':main()
