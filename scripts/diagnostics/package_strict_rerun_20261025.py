"""Package four completed post-audit cases without mutating scientific inputs.

中文：统一字段、单位、旧/新对照和导航，不重新拟合/修正物理结果，也不代替
独立原方程读回。历史冻结结果保持只读，新清单逐个绑定本轮实际复场与来源。
"""
from __future__ import annotations
import csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/current/strict_rerun_20261025'
OUT=BASE/'comparison'
IDS=('semiconductor_square','semiconductor_triangular','YbYAG_square','YbYAG_triangular')
LABELS=('半导体正方','半导体三角','Yb:YAG正方','Yb:YAG三角')

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8')
def resolve(p):return Path(p) if Path(p).is_absolute() else ROOT/p
def gain(row):
    return float(next(row[k] for k in ('required_material_gain_cm1','g_th_cm1','g_th_cm','material_gain_cm1') if k in row))

def normalize(case_id,label,old):
    cp=BASE/case_id/'selected_case.json';s=read(cp)
    if s.get('status','completed')!='completed':raise RuntimeError((case_id,'not completed'))
    if case_id.startswith('YbYAG'):
        rp=resolve(s['closed_modes_path']);rows=read(rp)
    else:rp=None;rows=s.get('frequency_closed_modes',s.get('modes',[]))
    rows=sorted(rows,key=gain)
    if len(rows)<6:raise RuntimeError((case_id,'fewer than six closed roots'))
    first,second=rows[:2];g=gain(first);gap=gain(second)/g-1;beam=first['beam']
    period=s.get('actual_period_nm',s.get('period_nm',s.get('three_hole_spec',{}).get('lattice_nm')))
    design=s.get('design',s.get('three_hole_spec'))
    if period is None:period=design['lattice_nm']
    target=1030. if case_id.startswith('YbYAG') else 950.
    if abs(first['wavelength_nm']-target)>.05:raise RuntimeError((case_id,'wrong actual first frequency'))
    joint=beam['center_to_peak']>=.98 and beam['ellipticity']<=1.05 and beam['radial_anisotropy_l2']<=.10 and gap>=.10
    if not joint:raise RuntimeError((case_id,'selected case misses declared balanced engineering screen'))
    stated_gap=s.get('relative_material_gap',s.get('g_th_gap_fraction',gap))
    if abs(stated_gap-gap)>1e-8:raise RuntimeError((case_id,'stale competitor gap'))
    fields=s.get('fields_path',s.get('closed_fields_path'))
    sp=s['summary_path'];geometry=s['geometry_path'];config=s.get('config_path',s.get('configuration_path'))
    # Assert source dimensions before assigning the common comparison label.
    # 中文：统一300um标签必须来自源数据，不能靠汇总程序直接写常数。
    geometry_data=read(resolve(geometry));aperture_checks=[]
    for origin,obj in (('selected',s),('geometry',geometry_data),('design',design)):
        for key in ('physical_aperture_um','nominal_aperture_um','nominal_size_um','aperture_um'):
            if key in obj:
                values=obj[key] if isinstance(obj[key],list) else [obj[key],obj[key]]
                if len(values)!=2 or any(abs(float(v)-300.)>1e-9 for v in values):raise RuntimeError((case_id,origin,key,'unmatched aperture'))
                aperture_checks.append(dict(source=origin,key=key,value=obj[key]))
        for key in ('side_um','device_square_side_um','length_um'):
            if key in obj:
                if abs(float(obj[key])-300.)>1e-9:raise RuntimeError((case_id,origin,key,'unmatched side'))
                aperture_checks.append(dict(source=origin,key=key,value=obj[key]))
    if not aperture_checks:raise RuntimeError((case_id,'no physical aperture bound in source records'))
    actual_geometry_period=geometry_data.get('lattice_constant_nm',geometry_data.get('period_nm'))
    if actual_geometry_period is None or abs(float(actual_geometry_period)-float(period))>1e-7:raise RuntimeError((case_id,'geometry period mismatch'))
    sources=list(dict.fromkeys(str(resolve(p)) for p in (cp,sp,fields,geometry,config,rp) if p is not None))
    result=dict(id=case_id,label=label,lattice='triangular' if case_id.endswith('triangular') else 'square',
        material='Yb:YAG' if case_id.startswith('YbYAG') else 'semiconductor',
        period_nm=float(period),D=int(s['D']),dz_nm=float(s['dz_nm']),N=int(s.get('N',s.get('grid'))),
        nominal_aperture_um=[300.,300.],target_wavelength_nm=target,wavelength_nm=first['wavelength_nm'],
        g_cm1=g,gap=gap,axis=beam['ellipticity'],center=beam['center_to_peak'],radial=beam['radial_anisotropy_l2'],
        first_mode=first.get('name',first.get('mode',first.get('seed_rank'))),
        closest_competitor=second.get('name',second.get('mode',second.get('seed_rank'))),
        closest_competitor_wavelength_nm=second['wavelength_nm'],
        closed_count=len(rows),gains_cm1=[gain(r) for r in rows],wavelengths_nm=[r['wavelength_nm'] for r in rows],
        current_A=first.get('I_th_A'),beam=beam,polarization=first.get('polarization',s.get('polarization')),
        engineering_screen_pass=bool(joint),engineering_screen_interpretation='Selection targets, not physical feasibility boundaries or confidence intervals.',
        source_aperture_checks=aperture_checks,
        design=design,sources=sources,source_sha256={p:sha(p) for p in sources},
        selected_case_path=str(cp),summary_path=str(resolve(sp)),fields_path=str(resolve(fields)),
        geometry_path=str(resolve(geometry)),closed_modes_path=str(rp) if rp else None,
        previous=dict(g_cm1=old['g_th_cm_inverse'],gap=old['relative_gain_gap'],beam=old['reported_beam'],
                      wavelength_nm=old['wavelength_nm'],period_nm=read(ROOT/old['geometry_path'])['lattice_constant_nm']),
        threshold_gain_change_percent=100*(g/old['g_th_cm_inverse']-1),
        gap_change_percentage_points=100*(gap-old['relative_gain_gap']))
    for key in ('nominal','reciprocity_300K','controls','spectral_scope','scientific_scope','recommendation'):
        if key in s:result[key]=s[key]
    prefix='root_'+str(first['seed_rank']) if case_id=='semiconductor_triangular' else str(result['first_mode'])
    result['field_bindings']=dict(field_key=prefix+('_fields' if case_id=='semiconductor_triangular' else '_field'),
        far_x_key=prefix+'_far_x',far_y_key=prefix+'_far_y',radiation_x_key=prefix+'_radiation_x',radiation_y_key=prefix+'_radiation_y',
        coupling_key=prefix+('_C' if case_id.startswith('YbYAG') else '_coupling_m'),
        angle_key='axis_deg' if case_id=='semiconductor_square' else 'angle_deg')
    return result

def report(cases):
    sq,tr,ys,yt=cases
    rows='\n'.join(f"|{x['label']}|{x['period_nm']:.6f}|{x['wavelength_nm']:.6f}|{x['g_cm1']:.6f}|{100*x['gap']:.3f}%|{x['axis']:.5f}|{x['center']:.5f}|{x['radial']:.5f}|{x['closed_count']}|" for x in cases)
    changes='\n'.join(f"|{x['label']}|{x['previous']['g_cm1']:.6f}|{x['g_cm1']:.6f}|{x['threshold_gain_change_percent']:+.3f}%|{100*x['previous']['gap']:.3f}%|{100*x['gap']:.3f}%|" for x in cases)
    links='\n'.join(f"- [{x['label']}参数、谱与实际复场](../results/current/strict_rerun_20261025/{x['id']}/selected_case.json)" for x in cases)
    return f'''# 2026-10-25 严格审查后的四腔重算与优化

本轮采用审查后的同一套物理核心，重新扫描半导体PCSEL与Yb:YAG的正方/三角晶格。四个主方案均为 **300×300 μm正方形物理边界**；目标为半导体950±0.05nm、YAG1030±0.05nm的实际首模。旧文献复现和旧优化结果只读，新结果在独立目录中保存。

## 1. 最终静态指标

g为有源材料增益阈值。间隔为已查独立实频根中的g₂/g₁−1，不是电流间隔、泵浦间隔、饱和多模开启点或动态稳定性。

|方案|a / nm|λ₁ / nm|g₁ / cm⁻¹|材料增益间隔|光斑轴比|中心/峰|径向L2|独立闭频根|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

各方案的[实际晶格、近远场及轴上偏振总图](../results/current/strict_rerun_20261025/comparison/four_cavity_overview.png)、[前六个已查模式阈值](../results/current/strict_rerun_20261025/comparison/closed_mode_thresholds.png)和[偏振CSV](../results/current/strict_rerun_20261025/comparison/four_cavity_polarization.csv)分别保存；偏振图中的近直线对应近线偏振，不能把圆形光斑称为圆偏振。

本轮平衡筛选线为中心/峰≥0.98、轴比≤1.05、径向L2≤0.10、已查增益间隔≥10%。这些是优化目标，不是物理可行性边界。光斑圆度与圆偏振分别计算；轴比接近1也不能替代完整径向残差。各图片按自身峰值归一化，不能直接比较绝对输出功率。

## 2. 和旧主方案相比

|方案|旧g₁ / cm⁻¹|新g₁ / cm⁻¹|材料阈值变化|旧间隔|新间隔|
|---|---:|---:|---:|---:|---:|
{changes}

半导体正方主方案将材料阈值由72.9538降至{sq['g_cm1']:.4f}cm⁻¹，降低约11.21%，同时保留约{100*sq['gap']:.2f}%的已查间隔。轴比改善，但中心/峰和径向L2略逊旧方案。经验均匀注入电流从0.455515A降至{sq['current_A']:.6f}A，仅约0.58%，不能把材料增益降幅称为电流降幅。57.6288/58.2482cm⁻¹两种低阈值取舍仍保留，但间隔只有约1.12%/1.76%。另一粗扫亮圆候选因更低暗条纹态出现而淘汰。

半导体三角主几何保持原方案。新的完整粗谱和有界细谱重算复现原首模。较大副孔位移方案虽降低名义阈值，却在细网格失去10%模式间隔；温和位移只降低名义材料阈值约1.65%，该约0.96nm位移小于3nm位置工差预算，且只验证少数已知枝，作为局部名义取舍保留，不冒充完整新最优。

YAG正方主方案的主要改进是模式隔离，材料阈值变化很小，其约0.5%的变化与已测D截断敏感性同量级。轴比及径向L2略逊原来更圆但模式间隔很小的方案。完整跨频段补查把最危险次模识别为约1024.43nm的独立根，未因偏离1030nm而删去。用原32根直接算出的更大间隔已撤回，以合并根的{100*ys['gap']:.4f}%为准。YAG三角主方案和较低阈值/圆度取舍的具体选择见分支报告；不能把略过工程圆度线的方案称为物理不可行。

## 3. 物理与数值验收

1. **物理周期固定。** 只在外层设计目标首模的周期；每条竞争根随后使用同一几何和周期，不逐模改周期凑到目标波长。
2. **实际频率闭合。** 每根在自己的实波长重建纵模、C和原始有限腔算符，再满足 F(λ)=[β(λ)²−β₀²]/(2β₀)−Reμ(λ)=0。这里“闭频”指方程自洽，不表示封闭腔；出射边界仍保留。
3. **原方程验收。** Ku=μMu，物理中心场φ=Mu。移位反演、几何连续延续或插值只提供初值，最终场必须通过未经修改K/M的残差与能流收支。
4. **模式身份。** 同一频率、同一纵向族、同一算符下同时比较复极点和子空间维数。不能只用高场重叠删掉近共线但不同的非正规极点；跨波长场不直接按重叠去重。
5. **谱搜索。** 完整粗离散谱、多个复移位、横向条纹及不同频段代表分别挑战。独立闭频根数量不是谱完备性的证明；所有失败、超时和未覆盖范围保留。不能按首模是否漂亮筛掉更低暗模。
6. **数值敏感性。** 固定几何检查D、dz及实际网格；三角方形mask另用非同对齐网格，不用面积缩放阈值，也不把外推图当本征场。
7. **局域能带。** 使用同一参考失谐和平方失谐的β₀/βref导数转换。四/六条局域Bloch带不是有限腔横模列表，也不是全布里渊区全波能带。

孔形参数以导出的实际nm边界为准。圆孔的方向、圆孔长宽比以及未启用三角拉伸的aspect标签可能不生效，不能把失活标签的变化宣称为不同几何或新的优化维度。扫描尝试数不应自动等同于有效独立几何数；已计算独立数的分支另行记录。

阈值统一用g_th=(2α_field+α_internal,power)/(100q_gain)。α均先用m⁻¹，半导体内部强度损耗500m⁻¹，YAG为10m⁻¹；分母100负责换成cm⁻¹。偏振从复Jones场计算，轴上近线偏振不表示整个角窗的统一偏振；空间积分DOLP也不是时间退偏振。

本轮相关79项回归通过；一次临时目录权限setup错误改用任务目录后单独重试通过，未改公式或验收阈值。独立代理从实际复场交叉重建的审查记录与原始测试XML均保留。此前342项全仓审计回归属于同日的审计阶段，不能把其数量与本轮79项简单相加为不同测试数。

## 4. YAG泵浦、能量和尺寸研究

新主几何的输入输出、脉冲能量及尺寸数据单列于 `results/current/strict_rerun_20261025/yag_feasibility/`，不会与旧几何的同日研究混用。CW使用全部已保留模式共享x-z反转、y平均；脉冲为单个冷场的均匀反转降阶。脉冲输入是外部首入射能量，入口耦合70%为假设；CW阈值是内部首入射泵功率，两者不能混标。20程、回收效率0.98为理想泵光回收方案。

尺寸变化保持各自最终孔形、物理周期和层栈不变，每个尺寸重新求有限腔本征场与实际波长，不用1/L缩放，也不把每个尺寸重新调到1030nm。300μm点取正式合并根作为锚点，低精度探索点与高D/N验证分开标明。1029–1031nm只用于目标窗内的条件分析，不是物理滤波器，不能把窗外竞争模式从实际总谱删掉。

名义材料沿用约10at.%、Nt=1.38×10²¹cm⁻³、τ=0.95ms、940nm泵浦和原标量截面。把1030nm峰值量级截面赋给1000–1060nm各模只是flat-spectrum条件模型。300K Stark互易配对另作一致性敏感性，不能冒充实测光谱。因此当前结果支持继续研究这一微腔方案，不能证明给定薄膜和泵光结构已能实现实验激光。尤其需要实测波长/温度相关吸收发射谱、薄膜损耗、寿命、实际泵耦合和热边界。

原参数及速率方程来源仍见[文献与速率审计](yag_rates_literature_audit_20261025_zh.md)和[此前YAG研究](yag_feasibility_20261025_zh.md)。晶膜Yb:YAG微盘激光的一手实验依据是[Li等, Optics Letters48,5359(2023)](https://doi.org/10.1364/OL.502795)，其WGM反馈与本PCSEL不同；谱配对参考[Körner等, JOSA B29,2493(2012)](https://doi.org/10.1364/JOSAB.29.002493)。

## 5. 三角晶格的研究价值

三组反向传播波和两个可独立控制的副孔，提供额外的方向、相位及辐射通道调节自由度。这使“阈值—间隔—方向性—偏振”的受控Pareto比较有研究价值，也可以研究方形有限边界的各向异性如何由非对称motif补偿。当前不对称motif已经破坏完整C6，不能直接套用理想C6简并或对称保护论证。

本轮半导体正方首模{sq['g_cm1']:.2f}cm⁻¹、三角{tr['g_cm1']:.2f}cm⁻¹，YAG两晶格也有不同阈值和圆度取舍；孔形、填充和参数不同，不能从这四个优化点推出三角或正方晶格具有普适先天优势。更严格的研究应固定材料、边界、孔形族、工差与收敛预算，比较两晶格可达的整体Pareto前沿。

## 6. 文献复现仍然存在的差别

保留的理想双椭圆D10/dz3复现，Γ点ΔBC/ΔAB约1.2164，原文读图约3.1；尚未定量吻合。每个Γ家族选一态，也不等于原文A族内多个分裂横模。三维SEM重建孔形可能影响耦合，但没有同一三维几何的独立复算，不能把差异全部归因于它，更不能删掉C₂D项或人为压缩AB间距来凑图。

Inoue原文与勘误：[Physical Review B99,035308](https://doi.org/10.1103/PhysRevB.99.035308)、[169904](https://doi.org/10.1103/PhysRevB.99.169904)。严格审查修复不等于已经完全复现论文。标量TE、近Bragg投影、实频冻结局域耦合及实际材料/三维结构缺失等边界继续适用，见[总物理审计](physics_audit_overall_20261025_zh.md)。

## 7. 原始结果入口

{links}
- [统一科学清单](../results/current/strict_rerun_20261025/comparison/selected_cases.json)
- [新旧指标CSV](../results/current/strict_rerun_20261025/comparison/four_cavity_metrics.csv)
- [本轮79项相关回归](../results/current/strict_rerun_20261025/physics_regression_summary.json)
- [38个审查后物理核心文件SHA一致性](../results/current/strict_rerun_20261025/core_source_freeze_check.json)
- [新几何YAG泵浦、能量及尺寸详细报告](strict_yag_feasibility_20261025_zh.md)

扫描仅保留紧凑参数、标量、失败原因和必要反例/取舍；正式复场、几何、配置及数值控制保留。删除的数组有路径/字节/SHA清单，但清单不是备份。各分支报告给出实际命令与可复算范围。历史冻结清单不回填本次数字。
'''

def main():
    old={r['id']:r for r in read(ROOT/'results/current/matched300_20261021/comparison/selected_cases.json')['cases']}
    cases=[normalize(i,label,old[i]) for i,label in zip(IDS,LABELS)]
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=dict(status='completed',date='2026-10-25',cases=cases,
        spectrum_scope='Complete coarse discrete spectra and bounded fine challenges, not a complete continuum or fine-grid spectrum.',
        physics_scope='Audited scalar near-Bragg TE CWT and stated material models; not independent full-wave or experimental certification.')
    save(OUT/'selected_cases.json',manifest)
    columns=('id','label','period_nm','wavelength_nm','g_cm1','gap','axis','center','radial','closed_count','D','dz_nm','N','threshold_gain_change_percent','gap_change_percentage_points')
    with (OUT/'four_cavity_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows({k:r[k] for k in columns} for r in cases)
    text=report(cases)
    (ROOT/'docs/strict_rerun_four_cavities_20261025_zh.md').write_text(text,encoding='utf8')
    (BASE/'README.md').write_text('# 审查后同尺寸四腔结果\n\n[主报告](../../../docs/strict_rerun_four_cavities_20261025_zh.md)；[科学清单](comparison/selected_cases.json)；[指标](comparison/four_cavity_metrics.csv)。\n\n'+'\n'.join(f"- [{c['label']}]({c['id']}/selected_case.json)：g={c['g_cm1']:.6f}cm⁻¹，已查间隔{100*c['gap']:.3f}%。" for c in cases)+'\n\n历史数据只读。新谱为有界挑战范围，原始失败和未收敛目标保留。\n',encoding='utf8')
    print('Packaged',len(cases),'completed scientific cases')

if __name__=='__main__':main()
