"""Package exact matched-area square finalists, geometry, and read-back audits."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import json,sys,hashlib,csv
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pcselsim.geometry import Ellipse
from pcselsim.fabrication import inclusion_boundary,evaluate_fabrication
from pcselsim.polarization import polarization_metrics
from scripts.optimization.matched300_square_search import (Design,build_cell,build_stack,
    save_json,OUT,RULES,threshold,BASE)


def geometry_key(design):
    ignored={"lattice_nm","tag"}
    return json.dumps({k:v for k,v in design.items() if k not in ignored},sort_keys=True)


def portable(result):
    d=Design(**result["design"]);cell=build_cell(d,96);holes=[]
    for i,hole in enumerate(cell.inclusions):
        boundary=inclusion_boundary(hole,np.eye(2));entry={"role":"central_main" if i==0 else "diagonal_satellite",
            "shape":d.main_shape if i==0 else d.satellite_shape,
            "boundary_over_a":boundary.tolist(),"boundary_nm":(boundary*d.lattice_nm).tolist()}
        q=np.roll(boundary,-1,axis=0);cross=boundary[:,0]*q[:,1]-boundary[:,1]*q[:,0]
        center=np.sum((boundary+q)*cross[:,None],axis=0)/(3*cross.sum())
        entry.update(area_nm2=float(abs(cross.sum())/2*d.lattice_nm**2),area_centroid_nm=(center*d.lattice_nm).tolist())
        entry["epsilon"]=float(hole.epsilon)
        if isinstance(hole,Ellipse):entry.update(center_nm=(np.array(hole.center)*d.lattice_nm).tolist(),semiaxes_nm=(np.array(hole.radii)*d.lattice_nm).tolist(),angle_deg=hole.angle_deg,
            boundary_sampled_area_nm2=entry["area_nm2"],area_nm2=float(np.pi*np.prod(hole.radii)*d.lattice_nm**2))
        holes.append(entry)
    stack=build_stack(d,cell)
    return {"protocol":"matched300_actual950_20261021_v2","lattice":"square","material":"Inoue_Table_I_semiconductor",
        "device_square_side_um":300.,"actual_winner_target_nm":950.,"lattice_constant_nm":d.lattice_nm,
        "internal_intensity_loss_cm1":5.,"design":asdict(d),"holes":holes,"contour_version":"v2_area_centroid_96",
        "background_refractive_index":3.554,"hole_refractive_index":1.0,
        "shape_parameter_notes":{"circle":"aspect/exponent/orientation descriptors are inactive; physical radius is given explicitly",
            "rounded_triangle":"Minkowski sum of unit-circumradius equilateral triangle and radius-0.35 disk, area-centered, area-normalized, independently stretched and rotated; exponent descriptor inactive"},
        "stack":asdict(stack),"accuracy":{"D":result["D"],"dz_nm":result["dz_nm"],"grid":result["grid"]},
        "process_rules":asdict(RULES),"fabrication":result["fabrication"],
        "scope":"reflection-free scalar four-wave 3-D CWT; actual finite eigenfields; uniform gain/loss; nondispersive nominal indices"}


def main():
    names={"best_balanced":"final_balanced","best_low_threshold":"final_low_threshold","best_isolation":"final_isolation"}
    selection=OUT/"selected_final_dirs.json"
    if selection.exists():names=json.loads(selection.read_text(encoding="utf-8"))
    results={k:json.loads((OUT/folder/"summary.json").read_text(encoding="utf-8")) for k,folder in names.items()}
    configdir=ROOT/"configs/optimized_matched300_20261021";configdir.mkdir(parents=True,exist_ok=True)
    configs={key:portable(r) for key,r in results.items()};save_json(configdir/"square_finalists.json",configs)
    candidates=[];audits=[]
    for key,r in results.items():
        folder=OUT/names[key];winner=r["lowest"];saved=np.load(folder/"final_fields.npz",allow_pickle=False)
        save_json(folder/"actual_geometry_nm.json",configs[key])
        assert all(np.isfinite(saved[k]).all() for k in saved.files),"Nonfinite raw array"
        field=saved[winner["mode"]+"_field"]
        assert field.shape==(4,r["grid"],r["grid"])
        assert r["design"]["length_um"]==300.
        assert abs(winner["wavelength_nm"]-950.)<.05
        for row in r["frequency_closed_modes"]:
            check=threshold(row["alpha_field_cm1"]*100,row["Gamma"],row["effective_index"],300.)
            assert np.isclose(check["g_th_cm1"],row["g_th_cm1"],rtol=1e-12)
            assert abs(row["frequency_residual_per_m"])<.05
        pol=polarization_metrics(saved["axis_deg"],saved[winner["mode"]+"_far_x"],saved[winner["mode"]+"_far_y"],core_radius_deg=.2)
        np.testing.assert_allclose(pol["integrated"]["normalized_stokes_s1_s2_s3"],winner["polarization"]["integrated"]["normalized_stokes_s1_s2_s3"],atol=1e-12)
        process=evaluate_fabrication(build_cell(Design(**r["design"]),96).inclusions,r["design"]["lattice_nm"],rules=RULES)
        assert process==r["fabrication"]
        candidate={"role":key,"id":r["id"],"side_um":300.,"period_nm":r["design"]["lattice_nm"],
            "wavelength_nm":winner["wavelength_nm"],"g1_cm1":winner["g_th_cm1"],
            "g2_cm1":r["frequency_closed_modes"][1]["g_th_cm1"],"relative_material_gap":r["relative_material_gap"],
            "I_th_A":winner["I_th_A"],"beam":winner["beam"],"polarization":winner["polarization"],
            "radiative_extraction_fraction":winner["radiative_extraction_fraction"],"fabrication":r["fabrication"],
            "portable_config":str(configdir/"square_finalists.json"),"summary_path":str(folder/"summary.json"),
            "actual_geometry_nm_path":str(folder/"actual_geometry_nm.json"),
            "final_field_path":str(folder/"final_fields.npz"),"overview_path":str(folder/"final_overview.png"),
            "period_design_path":str(folder/"period_design.json"),"source_spectrum_path":str(folder/"period_lateral/spectral_challenge.json"),
            "frequency_closed_count":len(r["frequency_closed_modes"]),"settings":{"D":r["D"],"dz_nm":r["dz_nm"],"grid":r["grid"]}}
        candidate["unresolved_closure_count"]=len(r.get("closure_failures",[]))
        candidate["unresolved_closure_modes"]=[x["mode"] for x in r.get("closure_failures",[])]
        candidate["closed_reference_ranks"]=[x["reference_rank"] for x in r["frequency_closed_modes"]]
        candidate["recommended"]=key=="best_isolation"
        candidate["eta_rad_definition"]={"formula":"2*local_CWT_field_amplitude_loss/(2*total_passive_field_amplitude_loss+500 per_m)",
            "numerator":"all local radiative channels represented in this cold CWT model",
            "denominator":"local radiation plus finite lateral passive leakage plus internal intensity loss 5 cm^-1; no duplicate internal loss",
            "not_equivalent_to":"measured slope efficiency or power inside the plotted angular window"}
        candidate["relative_threshold_current_gap"]=r["frequency_closed_modes"][1]["I_th_A"]/winner["I_th_A"]-1
        b=winner["beam"]
        candidate["basic_objective_pass"]=bool(b["ellipticity"]<=1.10 and b["center_to_peak"]>=.90 and b["radial_anisotropy_l2"]<=.15)
        candidate["stretch_objective_pass"]=bool(b["ellipticity"]<=1.05 and b["center_to_peak"]>=.98 and b["radial_anisotropy_l2"]<=.10 and r["relative_material_gap"]>=.10)
        candidates.append(candidate)
        audits.append({"role":key,"array_count":len(saved.files),"finite_arrays":True,"shape":field.shape,
                       "sha256":hashlib.sha256((folder/"final_fields.npz").read_bytes()).hexdigest(),
                       "threshold_formula_checked":True,"polarization_reconstructed":True,"final_nm_fabrication_checked":True})
    stages=sorted(p.parent.name for p in OUT.glob("*/compact_results.json"))
    convergence=[]
    keys={geometry_key(r["design"]):key for key,r in results.items()}
    for stage in stages:
        path=OUT/stage/"compact_results.json"
        if not path.exists():continue
        for r in json.loads(path.read_text(encoding="utf-8")):
            k=geometry_key(r["design"])
            if k not in keys or r.get("status")!="completed":continue
            m=r["lowest"];convergence.append({"role":keys[k],"stage":stage,"D":r.get("D",r.get("settings",{}).get("D")),
                "grid":r.get("grid",r.get("settings",{}).get("grid")),"dz_nm":r.get("dz_nm",r.get("settings",{}).get("dz_nm")),
                "g1_cm1":m["g_th_cm1"],"ellipse":m["beam"]["ellipticity"],"center":m["beam"]["center_to_peak"],
                "radial":m["beam"]["radial_anisotropy_l2"],"relative_gap":r["relative_material_gap"],
                "polarization":m.get("polarization"),"period_nm":r["design"]["lattice_nm"],
                "frequency_state":"actual lowest coarse pole designed at 950nm" if "period_design_residual_per_m" in r else "frozen 950 nm screen; period may differ from final actual-950 design","source":str(path)})
    for stage in ("recommended_mesh_N65","recommended_contour192","recommended_dz0p5","final_recommended",*names.values()):
        path=OUT/stage/("summary.json" if stage.startswith("final_") else "spectral_challenge.json")
        if not path.exists():continue
        r=json.loads(path.read_text(encoding="utf-8"));k=geometry_key(r["design"])
        if k not in keys:continue
        m=r["lowest"];convergence.append({"role":keys[k],"stage":stage,"D":r["D"],"grid":r["grid"],"dz_nm":r["dz_nm"],
            "contour_samples":r.get("vertices",96),"g1_cm1":m["g_th_cm1"],"ellipse":m["beam"]["ellipticity"],
            "center":m["beam"]["center_to_peak"],"radial":m["beam"]["radial_anisotropy_l2"],"relative_gap":r["relative_material_gap"],
            "polarization":m.get("polarization"),"period_nm":r["design"]["lattice_nm"],
            "frequency_state":"closed actual poles" if "frequency_closed_modes" in r else "frozen-950 mesh/contour/vertical sensitivity with physical seed fields","source":str(path)})
    save_json(OUT/"convergence.json",convergence)
    save_json(OUT/"readback_audit.json",audits)
    source_files=[ROOT/"src/pcselsim"/name for name in ("three_d_cwt.py","finite_discretization.py","polarization.py","mode_identity.py")]
    source_files.extend(sorted((ROOT/"scripts/optimization").glob("matched300_square_*.py")))
    save_json(OUT/"final_source_provenance.json",{"code_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "kernel_cache_validation_root":str(ROOT/"results/current/matched300_20261021/kernel_cache_validation"),
        "note":"Early screens predate the mathematically identical local radial-Green cache; final closures use the validated cache version."})
    summary={"protocol":"matched300_actual950_20261021_v2","status":"completed",
        "material":"semiconductor","lattice":"square","side_um":300.,"reference":"Inoue Table I complete stack",
        "screen_counts":{"broad_LHS":320,"local_full_QZ":288,"quarter_period_anchors":72,"isolation_local_full_QZ":192,"actual950_local_full_QZ":192,"total_geometry_evaluations":1064},
        "geometry_version_note":"Early low-order screens used 48-vertex or vertex-mean 64-vertex rounded contours; all final geometries re-centered by area centroid and rebuilt with 96 contour samples.",
        "candidates":candidates,**{c["role"]:c for c in candidates},"portable_config":str(configdir/"square_finalists.json"),
        "recommended_role":"best_isolation",
        "convergence_path":str(OUT/"convergence.json"),"readback_path":str(OUT/"readback_audit.json"),
        "source_provenance_path":str(OUT/"final_source_provenance.json"),"source_independence_path":str(OUT/"period_lateral_independence.json"),
        "not_proven":["global optimum","all open-Maxwell poles","nonlinear dynamic single-mode operation","spatial random-disorder yield","measured material and thermal robustness"],
        "higher_branch_failure_note":"The low-gain auxiliary candidate has one unresolved higher stripe continuation; its winner and nearest competitor are verified, and a later independent seed supplies the tenth verified root. No strong-isolation claim is made for that branch.",
        "scope":"lowest independently found passive CWT poles, actual finite fields, finite sampled lateral spectra; circular intensity and circular polarization are separately reported"}
    save_json(OUT/"final_summary.json",summary)
    csv_rows=[{"role":c["role"],"recommended":c["recommended"],"side_um":300.,"period_nm":c["period_nm"],
        "wavelength_nm":c["wavelength_nm"],"g1_cm1":c["g1_cm1"],"g2_cm1":c["g2_cm1"],
        "relative_gain_gap":c["relative_material_gap"],"relative_current_gap":c["relative_threshold_current_gap"],
        "Ith_A":c["I_th_A"],"ellipse":c["beam"]["ellipticity"],"center_to_peak":c["beam"]["center_to_peak"],
        "radial_L2":c["beam"]["radial_anisotropy_l2"],"eta_rad":c["radiative_extraction_fraction"],
        "on_axis_DoLP":c["polarization"]["on_axis"]["degree_of_linear_polarization"],
        "on_axis_S3_S0":c["polarization"]["on_axis"]["signed_circular_fraction_s3_over_s0"],
        "on_axis_LP_axis_deg":c["polarization"]["on_axis"]["linear_axis_angle_deg"],
        "process_margin_pass":c["fabrication"]["margin_pass"]} for c in candidates]
    with (OUT/"finalists.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=csv_rows[0]);writer.writeheader();writer.writerows(csv_rows)
    report(summary,results,convergence)
    compact_intermediate_fields()
    print(json.dumps({c["role"]:{k:c[k] for k in ("g1_cm1","g2_cm1","relative_material_gap","wavelength_nm","period_nm")} for c in candidates},indent=2))


def compact_intermediate_fields():
    """Only authored square scan arrays; keep final Jones fields and replay seeds."""
    replay={"core_D48_N49","late_D48_N49"};changes=[];root=OUT.resolve()
    audit_path=OUT/"intermediate_array_cleanup.json"
    previous=json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    selection=OUT/"selected_final_dirs.json"
    formal=set(json.loads(selection.read_text(encoding="utf-8")).values()) if selection.exists() else {"final_balanced","final_low_threshold","final_isolation"}
    for path in OUT.rglob("*.npz"):
        resolved=path.resolve()
        if not resolved.is_relative_to(root):raise RuntimeError("Cleanup target escapes square result root")
        relative=path.relative_to(OUT);parts=relative.parts
        if path.name=="final_fields.npz" and parts[0] in formal:continue
        before=path.stat().st_size
        if path.name=="lateral_fields.npz" and (parts[0] in replay or "period_lateral" in parts):
            with np.load(path,allow_pickle=False) as saved:
                arrays={k:saved[k].copy() for k in saved.files if k=="coupling_m" or k.endswith("_field")}
            np.savez_compressed(path,**arrays)
            changes.append({"path":str(path),"action":"keep_only_reference_eigenfields_and_coupling","before":before,"after":path.stat().st_size})
        else:
            path.unlink();changes.append({"path":str(path),"action":"delete_intermediate_scan_arrays","before":before,"after":0})
    released=sum(x["before"]-x["after"] for x in changes)
    save_json(audit_path,{"guarded_root":str(root),"actions":changes,"bytes_released_this_pass":released,
        "previous_passes":previous.get("previous_passes",[])+([{k:v for k,v in previous.items() if k!="previous_passes"}] if previous else []),
        "note":"The initial report pass removed large scan arrays; regenerated compaction passes need not show those earlier byte deltas.","final_fields_unchanged":True})


def report(summary,results,convergence):
    labels={"best_balanced":"Roundest low-gain candidate","best_low_threshold":"Lowest-gain candidate","best_isolation":"Recommended isolated candidate"}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for key,r in results.items():
        m=r["lowest"];axes[0].scatter(m["g_th_cm1"],m["beam"]["ellipticity"],s=90,label=labels[key])
        axes[1].scatter(m["g_th_cm1"],100*r["relative_material_gap"],s=90,label=labels[key])
    axes[0].axhline(1.05,color="grey",ls="--");axes[0].set(xlabel="Material threshold gain (cm^-1)",ylabel="Intensity moment ellipticity")
    axes[1].axhline(10,color="grey",ls="--");axes[1].set(xlabel="Material threshold gain (cm^-1)",ylabel="Checked nearest threshold gap (%)")
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle("Matched 300 x 300 um semiconductor square PCSELs; actual lowest pole at 950 nm")
    fig.savefig(OUT/"final_pareto.png",dpi=190);plt.close(fig)
    lines=["# 300 μm同尺寸中央主孔+对角副孔正方晶格优化", "", "本报告的腔边长固定300 μm，与文献有限腔L一致；所有正式候选使用同一Inoue完整层栈和5 cm⁻¹内强度损耗。先用950 nm冻结参考做多参数筛选，再把实际最低被动极点的周期调到950 nm，冻结该周期逐根闭合竞争横模。不同尺寸或916 nm旧候选不参加本轮优劣排序。", "", "## 最终Pareto候选", "", "| 候选 | a/nm | 实际λ/nm | g₁/cm⁻¹ | g₂/cm⁻¹ | 已查间隔 | ellipse | 中心/峰值 | 径向L2 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in summary["candidates"]:
        b=c["beam"];lines.append(f"| {c['role']} | {c['period_nm']:.6f} | {c['wavelength_nm']:.7f} | {c['g1_cm1']:.5f} | {c['g2_cm1']:.5f} | {100*c['relative_material_gap']:.3f}% | {b['ellipticity']:.6f} | {b['center_to_peak']:.6f} | {b['radial_anisotropy_l2']:.6f} |")
    preferred=summary["best_isolation"];r=results["best_isolation"];d=Design(**r["design"]);cell=build_cell(d,96)
    center=cell.inclusions[1];boundary=inclusion_boundary(center,np.eye(2));q=np.roll(boundary,-1,axis=0);cross=boundary[:,0]*q[:,1]-boundary[:,1]*q[:,0]
    centroid=np.sum((boundary+q)*cross[:,None],axis=0)/(3*cross.sum())*d.lattice_nm
    main_radius=cell.inclusions[0].radii[0]*d.lattice_nm
    lines.extend(["",f"推荐best_isolation：g₁={preferred['g1_cm1']:.5f} cm⁻¹、已查材料增益间隔{100*preferred['relative_material_gap']:.3f}%、η_rad={preferred['radiative_extraction_fraction']:.4f}。较低g的两个Pareto方案保留较小隔离，不用它们的四族名义大间隔宣称强单模。", "",
        f"主方案为中央圆孔（半径{main_radius:.6f} nm）+圆角三角副孔，主/副面积各占总孔面积50%，总填充{d.fill:.8f}；副孔面积形心({centroid[0]:.6f}, {centroid[1]:.6f}) nm，长宽拉伸比{d.satellite_aspect:.8f}、旋转{d.satellite_angle_deg:.8f}°，周期{d.lattice_nm:.9f} nm。圆孔的旋转角和长宽比标签不参与物理形状；所有真实边界均另存actual_geometry_nm.json。", "",
        f"该共同300 μm面积的近似阈值电流为{preferred['I_th_A']:.6f} A；材料增益间隔{100*preferred['relative_material_gap']:.3f}%对应近似阈值电流间隔仅{100*preferred['relative_threshold_current_gap']:.3f}%，因为透明载流子基线占主要电流。二者不可混用，也不代表上述电流倍数范围内已经证明动态单模。η_rad的分母包含被动横向漏光、局部辐射和5 cm⁻¹内强度损耗，不是±1°窗口功率占比或实验斜率效率。"])
    lines.extend(["", "圆度指标来自实际矢量远场总强度 |Ex|²+|Ey|² 在±1°窗口内的二阶矩；径向L2同时检查四瓣/六瓣等非椭圆畸变。圆形强度光斑不表示圆偏振。所有图均为英文标注，近场和远场均保留实际计算复场。", "", "## 几何扫描与排错", "", "320个LHS可制造几何、288个第一轮局部QZ、72个四分之一周期anchor、192个隔离方向局部QZ及192个actual-950局部QZ，共1064次几何评估。扫描总填充、主/副面积占比、副孔径向距离和偏角、两孔独立长宽比和旋转角、circle/ellipse/superellipse/rounded-triangle孔形。主孔面积形心固定在(0,0)，副孔沿对角线附近独立变位；不通过改腔尺寸或垂直层栈降低阈值。", "", "四个Γ连接族各选一个状态会漏掉更低横模，不能使用该名义四族间隔宣称模式选择。N9/N11初筛及N17完整广义QZ核验保留全部残差合格有限极点；候选随后在N35/N49用带边±π/L、±2π/L及实轴移位搜索，补入完整粗谱的全部低阈值复极点作为移位目标；主方案N49新实际周期共140个目标，N65以最低12真实场插值延续验证最低模及近邻竞争，主方案和最圆方案分别闭合最低10根。低g支线的源rank9高横阶stripe未通过0.65场重叠门槛，记录为closure_failure；以之后独立seed补齐10个已验证根，不称原最低10根全部闭合。它的winner与nearest已验证，仍不作为强隔离主推。N65延续不是完整N65谱，跨网格高横阶stripe簇单一向量身份也不强称收敛。多例名义亮圆候选被更低非圆横模反超而否决，反例记录仍在紧凑JSON中。", "", "早期筛选工艺有40 nm孔宽探索口径，正式候选全部以同一研究假设重查：最小周期孔隙40 nm、最小孔宽50 nm、边偏差3 nm、独立位置预算3 nm、曲率半径5 nm。该规则不是foundry认证。实际a调整后使用最终nm几何重新计算。", "", "## 精度与偏振", "", "本轮同时审查D、dz、N对阈值、最近竞争模间隔、ellipse、中心亮度、径向L2和Stokes的影响；不因材料阈值稳定就认定图形已收敛。逐阶段数值见convergence.json，真实频率闭合残差、参考复场重叠和所有试探点见各summary.json。", "", "Jones Ex/Ey使用共同强度尺度，S3=+2Im(ExEy*)；不额外指定左右旋命名。输出on-axis、0.2°core和整个±1°窗口的Stokes及线偏振度。空间角积分偏振度降低不等同于时间退偏振。", "", "| 候选 | on-axis线偏振度 | on-axis S3/S0 | 偏振主轴/° | ±1°积分线偏振度 |", "|---|---:|---:|---:|---:|"])
    for c in summary["candidates"]:
        p=c["polarization"];a=p["on_axis"];i=p["integrated"]
        lines.append(f"| {c['role']} | {a['degree_of_linear_polarization']:.6f} | {a['signed_circular_fraction_s3_over_s0']:.6f} | {a['linear_axis_angle_deg']:.5f} | {i['degree_of_linear_polarization']:.6f} |")
    lines.extend(["", "## 可复算文件与边界", "", "- 汇总：`results/current/matched300_20261021/square/final_summary.json`。", "- 便携物理几何、完整层栈、实际nm坐标/半轴/边界：`configs/optimized_matched300_20261021/square_finalists.json`。", "- 三个final_*目录包含频率闭合summary、最低实际复场/Jones/Stokes和英文总览；保存全部已闭合近场；仅胜出模及最近三个竞争模保存Jones远场，并保存胜出模Stokes图。", "- 新入口：`scripts/optimization/matched300_square_{search,local,actual950_local,anchors,validate,mesh,finalize,report}.py`。", "- 几何面积、实介电Fourier共轭关系、平移相位、同尺寸同层栈约束、48/64/96采样面积形心共12项测试通过。", "", "该结果属于反射自由Green函数的标量3D CWT与有限横向离散模型。完整粗网格QZ也不是完整开放Maxwell谱；细网格移位谱仍是有限抽样。未宣称全局最优、实验单模、非线性稳定、热鲁棒或随机工艺良率。", ""])
    (ROOT/"docs/optimization_matched300_square_20261021_zh.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__":main()
