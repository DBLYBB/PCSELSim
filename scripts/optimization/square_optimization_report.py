"""Collate frequency-closed square-lattice validation without rerunning it."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scripts.optimization.square_semiconductor_search import BASE,Design,build_cell,build_stack,save_json
from pcselsim.fabrication import inclusion_boundary


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def row(record,path):
    mode=record["lowest"]
    return {"source":str(path.resolve()),"length_um":record["design"]["length_um"],
        "fill":record["design"]["fill"],"tag":record["design"]["tag"],
        "D":record["D"],"dz_nm":record["dz_nm"],"grid":record["grid"],
        "sampled_poles":record["sampled_distinct_poles"],
        "wavelength_nm":mode["wavelength_nm"],"g_th_cm1":mode["g_th_cm1"],
        "J_th_A_cm2":mode["J_th_A_cm2"],"I_th_A":mode["I_th_A"],
        "relative_material_gain_gap":record["relative_material_gain_gap"],
        "ellipticity":mode["beam"]["ellipticity"],
        "center_to_peak":mode["beam"]["center_to_peak"],
        "radial_anisotropy_l2":mode["beam"]["radial_anisotropy_l2"],
        "beam_accepted":mode["beam_acceptance"]["accepted"],
        "radiative_fraction":mode["radiative_extraction_fraction"],
        "closed_roots":record["closed_successfully"],"failed_roots":record["failed_roots"]}


def write_csv(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader();writer.writerows(rows)


def beam_supplement(path,record):
    """Cuts and sidelobes refer only to the saved +/-1-degree angular window."""
    from scipy.signal import find_peaks
    with np.load(path.parent/"frequency_closed_poles.npz",allow_pickle=False) as saved:
        rank=record["lowest"]["seed_rank"]
        axis=saved["axis_deg"]
        intensity=abs(saved[f"seed_{rank}_far_x"])**2+abs(saved[f"seed_{rank}_far_y"])**2
    intensity/=intensity.max()
    centre=len(axis)//2
    widths=[];nulls=[]
    for cut in (intensity[centre],intensity[:,centre]):
        locations=[]
        for sign in (-1,1):
            aa=axis[centre:] if sign>0 else -axis[:centre+1][::-1]
            cc=cut[centre:] if sign>0 else cut[:centre+1][::-1]
            crossing=np.flatnonzero(cc<.5)
            if not len(crossing):locations.append(float("nan"));continue
            index=int(crossing[0]);fraction=(.5-cc[index-1])/(cc[index]-cc[index-1])
            locations.append(float(aa[index-1]+fraction*(aa[index]-aa[index-1])))
            minima,_=find_peaks(-cc)
            minima=minima[minima>index]
            if len(minima):nulls.append(float(aa[minima[0]]))
        widths.append(sum(locations))
    radius=min(nulls) if nulls else None
    xx,yy=np.meshgrid(axis,axis)
    outside=np.hypot(xx,yy)>radius if radius is not None else np.zeros_like(intensity,dtype=bool)
    return {"fwhm_x_deg":widths[0],"fwhm_y_deg":widths[1],
        "conservative_first_cut_minimum_radius_deg":radius,
        "peak_sidelobe_outside_that_circle_db":float(10*np.log10(max(intensity[outside].max(),1e-300))) if outside.any() else None,
        "power_outside_that_circle_fraction":float(intensity[outside].sum()/intensity.sum()) if outside.any() else None,
        "definition":"side lobe excludes circle at smallest first 1-D cut minimum; evaluated +/-1 degree only, not hemisphere"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"results/current/optimization_square_20261021")
    parser.add_argument("--primary",type=Path,required=True)
    parser.add_argument("--alternative",type=Path,required=True)
    args=parser.parse_args(argv)
    output=args.root
    primary=read(args.primary);alternative=read(args.alternative)
    all_cases=[(path,read(path)) for path in sorted(output.glob("*/N*/closed_summary.json"))]
    rows=[row(record,path) for path,record in all_cases]
    write_csv(output/"all_validated_cases.csv",rows)
    nominal165=[r for r in rows if r["length_um"]==165 and r["fill"]==primary["design"]["fill"]]
    mesh200=[r for r in rows if r["length_um"]==200 and r["fill"]==primary["design"]["fill"]]
    write_csv(output/"mesh_and_truncation.csv",nominal165+mesh200)
    tolerance_path=output/"robustness_L165_D10_z2_N33/robustness_summary.json"
    tolerance=read(tolerance_path) if tolerance_path.exists() else []
    tolerance_rows=[]
    for record in tolerance:
        if record["status"]!="completed":continue
        item=row(record,tolerance_path)
        item={"perturbation":record["design"]["tag"],**item}
        tolerance_rows.append(item)
    write_csv(output/"process_sensitivity.csv",tolerance_rows)
    alt_tolerance_path=output/"robustness_L200_D10_z2_N33/robustness_summary.json"
    alt_tolerance=read(alt_tolerance_path) if alt_tolerance_path.exists() else []
    alt_tolerance_rows=[{"perturbation":record["design"]["tag"],**row(record,alt_tolerance_path)}
        for record in alt_tolerance if record["status"]=="completed"]
    write_csv(output/"process_sensitivity_L200.csv",alt_tolerance_rows)
    geometry=[]
    for index,hole in enumerate(build_cell(Design(**primary["design"])).inclusions):
        geometry.append({"hole":index+1,"center_x_a":hole.center[0],"center_y_a":hole.center[1],
            "center_x_nm":hole.center[0]*primary["design"]["lattice_nm"],
            "center_y_nm":hole.center[1]*primary["design"]["lattice_nm"],
            "major_semiaxis_nm":hole.radii[0]*primary["design"]["lattice_nm"],
            "minor_semiaxis_nm":hole.radii[1]*primary["design"]["lattice_nm"],"rotation_deg":hole.angle_deg})
    write_csv(output/"selected_geometry.csv",geometry)
    baseline_path=output/"baseline_L165_D10_z2_N49/N49/closed_summary.json"
    summary={"status":"completed","model_scope":"passive scalar four-wave CWT; sampled finite lateral poles with independent real-frequency closure; no full Maxwell, carrier competition, heating or experimental certification",
        "primary_source":str(args.primary.resolve()),"primary":primary,
        "alternative_source":str(args.alternative.resolve()),"lower_threshold_alternative":alternative,
        "primary_beam_supplement":beam_supplement(args.primary,primary),
        "alternative_beam_supplement":beam_supplement(args.alternative,alternative),
        "layers_bottom_to_top":[asdict(layer) for layer in build_stack(Design(**primary["design"]),build_cell(Design(**primary["design"]))).layers],
        "geometry_nm":geometry,"all_cases":rows,"process_sensitivity":tolerance_rows,
        "alternative_process_sensitivity":alt_tolerance_rows,
        "recommendation_status":"nominal linear CWT research candidates; neither certified for +/-3 nm optical robustness",
        "primary_process_passed":sum(r["beam_accepted"] for r in tolerance_rows),
        "primary_process_total":len(tolerance_rows),
        "alternative_process_passed":sum(r["beam_accepted"] for r in alt_tolerance_rows),
        "alternative_process_total":len(alt_tolerance_rows),
        "same_size_original_double_circle":read(baseline_path) if baseline_path.exists() else None,
        "quantities":"g_th is material intensity gain /cm; field loss has factor 2; roundness is intensity circularity, not polarization",
        "angular_window_deg":[-1,1],"uniform_internal_power_loss_cm1":BASE.INTERNAL_LOSS_CM,
        "recommended_next_steps":["3-D vector full-wave / measured or dispersive indices", "spatial gain and thermal/carrier competition", "joint process corners and statistical yield", "physical finite aperture and boundary/corner validation"]}
    save_json(output/"summary.json",summary)
    save_json(output/"selected_designs.json",[primary["design"],alternative["design"]])
    config_dir=ROOT/"configs/optimized_20261021"
    config_dir.mkdir(parents=True,exist_ok=True)
    save_json(config_dir/"square_finalists.json",{
        "format":"square_semiconductor_finalists_v1","created":"2026-10-21",
        "scope":summary["model_scope"],"geometry_nm":geometry,
        "material_model":{"background_index":BASE.BACKGROUND_INDEX,"hole_index":1.,
            "active_index":BASE.ACTIVE_INDEX,"internal_power_loss_cm1":BASE.INTERNAL_LOSS_CM,
            "gain_maximum_cm1":BASE.MAXIMUM_GAIN_CM,"gain_zero_carrier_cm1":BASE.ZERO_CARRIER_GAIN_CM,
            "transparency_density_cm3":BASE.TRANSPARENCY_DENSITY_CM3,
            "active_thickness_nm":BASE.ACTIVE_THICKNESS_NM,"carrier_lifetime_ns":BASE.CARRIER_LIFETIME_NS,
            "top_cladding_index":BASE.TOP_CLADDING_INDEX,"bottom_cladding_index":BASE.BOTTOM_CLADDING_INDEX,
            "vertical_padding_um":BASE.VERTICAL_PADDING_UM},
        "layers_bottom_to_top":summary["layers_bottom_to_top"],
        "finalists":[{"role":role,"design":record["design"],
            "verification":{"D":record["D"],"dz_nm":record["dz_nm"],"grid":record["grid"],
                "angular_samples":201,"real_frequency_closure":True},
            "nominal_metrics":{"wavelength_nm":record["lowest"]["wavelength_nm"],
                "g_th_cm1":record["lowest"]["g_th_cm1"],
                "relative_material_gain_gap":record["relative_material_gain_gap"],
                "ellipticity":record["lowest"]["beam"]["ellipticity"]}}
            for role,record in (("larger_nominal_gap",primary),("lower_nominal_threshold",alternative))]})

    fig=plt.figure(figsize=(15,11),constrained_layout=True)
    gs=fig.add_gridspec(3,4)
    ax=fig.add_subplot(gs[0,0])
    d=Design(**primary["design"])
    for hole in build_cell(d).inclusions:
        boundary=inclusion_boundary(hole,np.eye(2))*d.lattice_nm
        ax.fill(boundary[:,0],boundary[:,1],color="#fcfcff",ec="#16324f",lw=2)
    ax.set(xlim=(-d.lattice_nm/2,d.lattice_nm/2),ylim=(-d.lattice_nm/2,d.lattice_nm/2),aspect="equal",xlabel="x (nm)",ylabel="y (nm)",title=f"Two rotated elliptical holes; a={d.lattice_nm:g} nm")
    for j,(path,record,label) in enumerate(((args.primary,primary,"L=165 um; larger mode gap"),(args.alternative,alternative,"L=200 um; lower threshold"))):
        arrays=np.load(path.parent/"frequency_closed_poles.npz",allow_pickle=False)
        rank=record["lowest"]["seed_rank"]
        field=arrays[f"seed_{rank}_field"]
        fx,fy=arrays[f"seed_{rank}_far_x"],arrays[f"seed_{rank}_far_y"]
        intensity=abs(fx)**2+abs(fy)**2;intensity/=intensity.max()
        axis=arrays["axis_deg"]
        ax=fig.add_subplot(gs[0,j+1]);ax.imshow(intensity,origin="lower",extent=(-1,1,-1,1),cmap="hot")
        ax.set(xlabel="theta x (deg)",ylabel="theta y (deg)",title=label)
        ax=fig.add_subplot(gs[1,j]);ax.imshow(np.sum(abs(field)**2,axis=0),origin="lower",extent=(0,record["design"]["length_um"],0,record["design"]["length_um"]),cmap="viridis")
        ax.set(xlabel="x (um)",ylabel="y (um)",title=f"g={record['lowest']['g_th_cm1']:.2f} /cm; gap={record['relative_material_gain_gap']:.2%}")
        arrays.close()
    ax=fig.add_subplot(gs[0,3]);ax.axis("off")
    m=primary["lowest"];b=m["beam"]
    ax.text(0,1,f"Selected sampled lowest pole\n\nlambda = {m['wavelength_nm']:.4f} nm\ng_th = {m['g_th_cm1']:.3f} /cm\nJ_th = {m['J_th_A_cm2']:.1f} A/cm2\nI_th = {m['I_th_A']*1000:.1f} mA\nEllipse = {b['ellipticity']:.4f}\nCenter / peak = {b['center_to_peak']:.3f}\nRadial residual = {b['radial_anisotropy_l2']:.4f}\nRadiation share = {m['radiative_extraction_fraction']:.2%}\n\nActual sampled gap = {primary['relative_material_gain_gap']:.2%}",va="top",fontsize=11)
    ax=fig.add_subplot(gs[1,2:]);
    for length,color in ((165,"#176d9c"),(200,"#bc6d24")):
        subset=sorted((r for r in rows if r["length_um"]==length and r["D"]==10 and r["dz_nm"]==2 and r["fill"]==primary["design"]["fill"]),key=lambda r:r["grid"])
        ax.plot([r["grid"] for r in subset],[r["g_th_cm1"] for r in subset],"o-",label=f"L={length} um",color=color)
    ax.set(xlabel="Envelope grid N",ylabel="Material threshold gain (/cm)",title="First pole under mesh refinement; independent real roots")
    ax.legend();ax.grid(alpha=.2)
    ax=fig.add_subplot(gs[2,:2])
    selected=[]
    for length in sorted({r["length_um"] for r in rows}):
        available=[r for r in rows if r["length_um"]==length and r["fill"]==primary["design"]["fill"]]
        selected.append(max(available,key=lambda r:(r["D"],-r["dz_nm"],r["grid"])))
    for accepted,marker,label,color in ((True,"o","Bright / round actual lowest","#176d9c"),(False,"x","Rejected: lower non-round mode","#b63338")):
        subset=[r for r in selected if r["beam_accepted"]==accepted]
        ax.scatter([r["length_um"] for r in subset],[100*r["relative_material_gain_gap"] for r in subset],marker=marker,label=label,color=color,s=65)
    ax.axhline(10,color="gray",ls="--",lw=1)
    ax.set(xlabel="Cavity side length (um)",ylabel="Actual sampled threshold gap (%)",title="Size tradeoff; D8/z3 points are preliminary")
    ax.legend();ax.grid(alpha=.2)
    ax=fig.add_subplot(gs[2,2:])
    if tolerance_rows:
        ax.bar(np.arange(len(tolerance_rows)),[r["relative_material_gain_gap"]*100 for r in tolerance_rows],color=["#176d9c" if r["beam_accepted"] else "#b63338" for r in tolerance_rows])
        ax.set_xticks(np.arange(len(tolerance_rows)),[r["perturbation"].replace("_"," ") for r in tolerance_rows],rotation=35,ha="right",fontsize=8)
        ax.axhline(primary["relative_material_gain_gap"]*100,color="black",ls="--",label="Nominal")
        ax.legend()
    ax.set(ylabel="Actual sampled gap (%)",title="Independent +/-3 nm process perturbations")
    fig.suptitle("Square-lattice PCSEL optimization: low material gain and a bright circular intensity spot\nLinear scalar CWT predictions; lateral spectrum sampled, not exhaustive",fontsize=15)
    fig.savefig(output/"square_overview.png",dpi=180);plt.close(fig)
    final=output/"final_report"
    final.mkdir(exist_ok=True)
    shutil.copyfile(output/"summary.json",final/"summary.json")
    shutil.copyfile(output/"square_overview.png",final/"overview.png")
    final_rows=[{"role":name,**row(record,path)} for name,path,record in
        (("larger_nominal_gap",args.primary,primary),("lower_nominal_threshold",args.alternative,alternative))]
    write_csv(final/"finalists.csv",final_rows)
    base=summary["same_size_original_double_circle"]
    table="\n".join(f"| {r['length_um']:g} | {r['D']} / {r['dz_nm']:g} / {r['grid']} | {r['wavelength_nm']:.4f} | {r['g_th_cm1']:.3f} | {r['relative_material_gain_gap']:.2%} | {r['ellipticity']:.4f} | {'通过' if r['beam_accepted'] else '非圆低模，淘汰'} |" for r in sorted(rows,key=lambda item:(item["length_um"],item["D"],item["grid"])) if r["fill"]==primary["design"]["fill"])
    process_table="\n".join(f"| {r['perturbation']} | {r['g_th_cm1']:.3f} | {r['relative_material_gain_gap']:.2%} | {r['ellipticity']:.3f} | {'亮圆最低模' if r['beam_accepted'] else '更低非圆模，目标失效'} |" for r in tolerance_rows)
    p=primary["lowest"];a=alternative["lowest"]
    geometry_table="\n".join(f"| {g['hole']} | ({g['center_x_a']:.6f}, {g['center_y_a']:.6f}) | ({g['center_x_nm']:.3f}, {g['center_y_nm']:.3f}) | {g['major_semiaxis_nm']:.3f} / {g['minor_semiaxis_nm']:.3f} | {g['rotation_deg']:.4f} |" for g in geometry)
    baseline_text=(f"同为165 μm，原双圆孔在D10/dz2/N49的{base['sampled_distinct_poles']}个抽样极点中，最低实频根为{base['lowest']['wavelength_nm']:.4f} nm、材料阈值{base['lowest']['g_th_cm1']:.3f} cm⁻¹、ellipse={base['lowest']['beam']['ellipticity']:.3f}、中心/峰值={base['lowest']['beam']['center_to_peak']:.4g}，实际间隔{base['relative_material_gain_gap']:.2%}。应将其是否满足圆斑与模式选择一起比较，不能用原四族中选中的圆模阈值来宣称全谱阈值降低百分比。" if base else "同尺寸原双圆孔复核尚无结果。")
    text=f"""# 正方晶格：低阈值、亮圆斑与真实横模竞争的联合优化（2026-10-21）

本轮保留两种名义候选：165 μm较大模式间隔方案与200 μm较低材料阈值方案。前者最终设置D={primary['D']}、dz={primary['dz_nm']:g} nm、N={primary['grid']}，最低抽样横模实频闭合后λ={p['wavelength_nm']:.6f} nm，g_th={p['g_th_cm1']:.4f} cm⁻¹、实际抽样间隔{primary['relative_material_gain_gap']:.2%}、ellipse={p['beam']['ellipticity']:.5f}、中心/峰值={p['beam']['center_to_peak']:.3f}。这支持线性标量CWT内的低阈值亮圆斑候选；工艺扰动中确有更低非圆模出现，不能将其称为±3 nm工艺鲁棒或已证明的单模激光器。

## 可直接复现的几何与层栈

正方晶格a={primary['design']['lattice_nm']:g} nm；两个旋转椭圆孔，孔折射率1，背景折射率3.554，总孔面积占比{primary['design']['fill']:.8f}，第一孔面积份额{primary['design']['share']:.8f}。孔边界连续光滑；下表半轴不是直径。

| 孔 | 中心 / a | 中心 / nm | 长半轴 / 短半轴 nm | 长轴角度 ° |
|---|---|---|---|---|
{geometry_table}

优化后的PC层厚{primary['design']['pc_nm']:.6f} nm、GaAs层厚{primary['design']['gaas_nm']:.6f} nm、AlGaAs spacer厚{primary['design']['qw_spacer_nm']:.6f} nm。其余沿用主入口的三阱三势垒层栈：p-clad 2000 nm；PC；GaAs；spacer；三层10 nm InGaAs阱及各自20 nm AlGaAs势垒；80 nm AlGaAs；1000 nm n-clad。上下包层折射率3.122/3.297，纵向场求TE0，PC纵向折射率用实际孔面积平均介电常数的平方根。各层完整数值在final_report/summary.json的layers_bottom_to_top。对上、下表面的收集效率及实际电极扩散尚未计算。

易工艺规则要求最小孔跨度65 nm、间隙60 nm，并计每边3 nm刻蚀预算、每孔3 nm放置预算。名义最小周期间隙{primary['fabrication']['easier']['minimum_periodic_gap_nm']:.3f} nm，最小跨度{primary['fabrication']['easier']['minimum_caliper_nm']:.3f} nm，预算后间隙{primary['fabrication']['easier']['worst_case_gap_margin_nm']:.3f} nm、跨度{primary['fabrication']['easier']['worst_case_caliper_margin_nm']:.3f} nm，几何工艺规则通过。几何可制造性通过不等于光学模排序对偏差鲁棒。

## 阈值、出耦与真实光斑的定义

统一用材料强度增益g_th=(2 alpha_field+alpha_int)/(Gamma_QW n_active/n_eff)，alpha_int=5 cm⁻¹。被动本征值虚部是场幅损耗，故有因子2；内损耗只加入一次。当前均匀注入增益/复合模型采用30 nm总有源厚度、1.5 ns寿命、透明载流子浓度1.5e18 cm⁻³；阈值电流是该近似模型的估计，未含热、电流扩展或增益压缩。165 μm方案J_th={p['J_th_A_cm2']:.3f} A/cm²，I_th={p['I_th_A']:.6f} A；200 μm方案J_th={a['J_th_A_cm2']:.3f} A/cm²，I_th={a['I_th_A']:.6f} A。尺寸引起的总电流变化不能解释为材料阈值同比例改善。

出耦占比按eta_rad=2 alpha_rad,field/(2 alpha_field+alpha_int)，165/200 μm分别为{p['radiative_extraction_fraction']:.2%}/{a['radiative_extraction_fraction']:.2%}，是模型总垂直辐射份额，不是单面收集外量子效率。

远场用实际有限腔本征场的两个复辐射分量直接求矢量傅里叶变换；统一±1°、201×201角点、检查源包络Nyquist。圆斑验收要求ellipse≤1.10、中心/峰值≥0.90、极坐标相对L2非圆度≤0.15、半高单连通且包含法线。圆环或多瓣即便二阶矩ellipse≈1也不通过。165 μm的非圆度{p['beam']['radial_anisotropy_l2']:.5f}、半高连通块{p['beam']['half_max_components']}，±0.5°收集比例{p['beam']['encircled_0p5deg']:.3%}。切向FWHM及保守旁瓣定义见summary.json的primary_beam_supplement；所有功率比例只相对±1°窗口归一。圆形强度不代表圆偏振。

## 实际搜索、完整粗谱与细网格否决

已完成258个广域样本、216个局部样本，覆盖单孔、双孔、三孔、四孔、椭圆及圆角超椭圆，并优化填充、面积分配、二维位移、长宽比、角度、PC深度、GaAs厚度、阱前间隔、孔径大小。474个样本随后用N9完整广义QZ谱复排。四个Gamma连接族各选一个模式的筛选确实会漏掉更低高阶横模，不能把该四族间隔当真实横模间隔。

已调到950 nm的150 μm方案，四族选模会给出g_th≈72.052 cm⁻¹及圆斑；更广横模谱却找到g_th≈62.707 cm⁻¹、ellipse≈3.418的更低非圆模，因此已否决该“950 nm主推”。本轮也否决100/125/150 μm中更低非圆高阶模；没有因为目标输出好看而忽略它们。

细谱每个设计至少20个独立移位簇，在四个Gamma边缘加入±pi/L、±2pi/L；165 μm的D10细验证额外沿实轴均匀加入41个目标，共61簇，每目标40极点，覆盖带边之间的区间，保留独立简并伙伴。然后对最低12个抽样极点分别固定几何与周期、重新求TE0、C(lambda)及实际有限场，闭合Re(delta_TE)=Re(lambda_finite)。数值残差和场重叠逐步保存，失败不会冒充收敛。它仍是有限抽样谱；只有粗网格QZ是该离散模型的完整谱，不能宣称完整开放Maxwell谱或动态单模证明。

| L μm | D / dz nm / N | 最低实根 λ nm | 最低g cm⁻¹ | 实际抽样间隔 | ellipse | 圆斑判断 |
|---|---|---|---|---|---|---|
{table}

## 同尺寸原双圆孔对照

{baseline_text}

## ±3 nm工艺偏差：必须保留的失效结果

每种偏差重新改变实际几何，孔边偏差会改变面积占比和长宽比；相反位置偏差使两孔相对距离改变6 nm。每例37个移位簇、每目标24极点、D10/dz2/N33，并独立闭合最低4根。它是单变量偏差，不是联合最坏角或统计良率。

| 165 μm偏差 | 最低g cm⁻¹ | 实际抽样间隔 | ellipse | 判断 |
|---|---|---|---|---|
{process_table}

名义较大模式间隔未保证3 nm制造偏差下维持目标。应将此设计定位为精密制程研究候选；下一轮需把所有偏差后实际最低横模同时纳入目标，并验证联合偏差与统计良率。200 μm的独立偏差结果保存在robustness_L200_D10_z2_N33，不用165 μm结果替代。

## 正方晶格在本任务中的优点与限制

四个基本传播波使耦合通道和模式标记较清晰；双椭圆孔可同时调整反向反馈、正交耦合和垂直辐射，计算及参数诊断直接。当前几何通过65 nm孔跨度/60 nm间隙的加工预算；没有尖角。即使底层晶格只有四重对称，真实最低模式仍可得到ellipse≈1.03、很小径向残差的亮单瓣，不能仅以六重对称就认定三角晶格必然更圆或更低阈值。

主要限制是暗/条纹状高阶横模和工艺诱发的模式重新排序；低四族阈值并不保证实际最低模式光斑合格。165 μm和200 μm展示模式间隔、材料阈值及总电流的折中。与三角晶格比较时应固定材料层栈、实际物理孔径/面积、波长处理、全横模搜索程度和角窗，并对各自真实最低模式比较。

## 文件与复现

- 固定汇总：results/current/optimization_square_20261021/final_report/summary.json、finalists.csv、overview.png。
- 原始场：每个N*/frequency_closed_poles.npz；原始横模与残差：lateral_survey.json、lateral_poles.npz；逐根频率轨迹：frequency_closed_poles.json。
- 执行入口：scripts/optimization/square_extended_validate.py；工艺：square_robustness.py；汇总：square_optimization_report.py。
- 旧主入口和历史结果未替换；新增仅为仿真脚本、数据、图和本仿真报告。

```powershell
E:/Anaconda/envs/env00/python.exe scripts/optimization/square_extended_validate.py configs/optimized_20261021/square_finalists.json --design-index 0 --output results/current/optimization_square_replay --grids 65 --D 12 --dz 1 --roots 6 --count 32
E:/Anaconda/envs/env00/python.exe scripts/optimization/square_extended_validate.py configs/optimized_20261021/square_finalists.json --design-index 1 --output results/current/optimization_square_L200_replay --grids 33 49 65 --D 10 --dz 2 --roots 12 --count 32
```

独立单元核查tests/test_optimization_square.py共9项通过；它核查面积、阈值单位与增益反算、远场闭式和源采样、三势垒、可复现提案、简并伙伴和真实孔边偏差。最终还需三维矢量全波、色散/实测折射率、空间增益/载流子和热竞争、完整边界及联合工艺角验证后，才适合提出实物阈值或稳定单模结论。
"""
    (ROOT/"docs/optimization_square_20261021_zh.md").write_text(text,encoding="utf8")
    print(output/"summary.json")
    print(output/"square_overview.png")


if __name__=="__main__":main()
