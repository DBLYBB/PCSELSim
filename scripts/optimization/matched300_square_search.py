"""Matched 300-um PCSEL search: a central main hole and diagonal satellite.

The Inoue Table-I stack and 5 cm^-1 internal intensity loss are fixed. Coarse
screens are explicitly separated from accuracy checks; fields are only kept
for finalists. Every ranking uses the lowest *actually returned* passive pole.

中文：固定 300 × 300 μm 方形腔和文献层结构，只扫描单胞孔形与相对位置。
本文件做粗筛；最低“已返回”极点并不等于完整横向谱的首模，正式结论还需
独立横向谱挑战、网格检查及 fixed-period 实频闭合。
"""
from __future__ import annotations
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse, csv, hashlib, json, sys, time, warnings
from functools import lru_cache
from dataclasses import asdict, dataclass, replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import qmc
from pcselsim.geometry import Ellipse, PolygonInclusion, SquareLatticeCell
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from pcselsim.custom_analysis import FourWaveOpticalSpec, solve_finite_modes
from pcselsim.fabrication import FabricationRules, evaluate_fabrication, pareto_mask
from scripts.optimization.square_semiconductor_search import (
    BASE, build_stack, clean, save_json, threshold, direct_vector_far_field)

OUT = ROOT / "results/current/matched300_20261021/square"
RULES = FabricationRules("matched_research_50nm_span_40nm_gap", 40, 50, 3, 3, 5)


@dataclass(frozen=True)
class CachedSquareCell(SquareLatticeCell):
    """Reuse exact geometry-only Fourier values during wavelength/period roots.

    Frozen dataclass equality includes every physical inclusion and background.
    The bounded cache never contains a wavelength, stack, or Green function.
    """
    @lru_cache(maxsize=131072)
    def fourier_epsilon(self,m,n):
        return super().fourier_epsilon(m,n)

@dataclass(frozen=True)
class Design:
    """Dimensionless motif plus physical sizes / 无量纲孔结构及物理尺寸。

    fill is total hole area per cell, main_share splits that area, and distance
    is a Cartesian separation divided by the period. All angles are degrees.
    fill 为晶胞总孔面积占比；main_share 分配主孔面积；distance 为孔心的
    笛卡尔距离与周期之比，所有角度均为度。_nm/_um 为实际长度单位。
    """
    fill: float = .14
    main_share: float = .60
    distance: float = .53
    position_angle_deg: float = 45.
    main_shape: str = "ellipse"
    satellite_shape: str = "ellipse"
    main_aspect: float = 1.
    satellite_aspect: float = 1.
    main_angle_deg: float = 0.
    satellite_angle_deg: float = 0.
    main_exponent: float = 3.
    satellite_exponent: float = 3.
    lattice_nm: float = 277.
    length_um: float = 300.
    pc_nm: float = 190.
    gaas_nm: float = 110.
    qw_spacer_nm: float = 25.
    tag: str = "search"

    def __post_init__(self):
        if self.length_um != 300. or (self.pc_nm, self.gaas_nm, self.qw_spacer_nm) != (190.,110.,25.):
            raise ValueError("Matched semiconductor protocol fixes aperture and stack")
        if not (0 < self.fill < .5 and .5 <= self.main_share < 1):
            raise ValueError("Both holes need positive area; the central hole is the main hole")

    @property
    def identifier(self):
        value=asdict(self); value.pop("tag")
        return "square_"+hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:12]


def hole(center, area, kind, aspect, angle, exponent=3., vertices=48):
    """Analytic ellipses; area-normalized smooth rounded triangle/superellipse."""
    if kind in ("circle","ellipse"):
        aspect=1. if kind=="circle" else aspect
        r=np.sqrt(area/np.pi)
        return Ellipse(tuple(center),(float(r*np.sqrt(aspect)),float(r/np.sqrt(aspect))),angle)
    phase=2*np.pi*np.arange(vertices)/vertices
    if kind=="superellipse":
        points=np.column_stack((np.sign(np.cos(phase))*abs(np.cos(phase))**(2/exponent),
                                np.sign(np.sin(phase))*abs(np.sin(phase))**(2/exponent)))
    elif kind=="rounded_triangle":
        # Minkowski sum of a triangle and a disk, sampled then convex-hulled.
        triangle=np.column_stack((np.cos(np.pi/2+2*np.pi*np.arange(3)/3),
                                  np.sin(np.pi/2+2*np.pi*np.arange(3)/3)))
        circle=.35*np.column_stack((np.cos(phase),np.sin(phase)))
        samples=(triangle[:,None,:]+circle[None,:,:]).reshape(-1,2)
        points=samples[ConvexHull(samples).vertices]
    else:
        raise ValueError(kind)
    # A non-multiple-of-three sampling can bias the vertex mean. The intended
    # hole position is its physical area centroid, at every contour resolution.
    following=np.roll(points,-1,axis=0)
    cross=points[:,0]*following[:,1]-points[:,1]*following[:,0]
    centroid=np.sum((points+following)*cross[:,None],axis=0)/(3*np.sum(cross))
    points-=centroid
    points*=np.array([np.sqrt(aspect),1/np.sqrt(aspect)])
    actual=.5*abs(np.sum(points[:,0]*np.roll(points[:,1],-1)-points[:,1]*np.roll(points[:,0],-1)))
    points*=np.sqrt(area/actual)
    a=np.deg2rad(angle); rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    return PolygonInclusion(tuple(map(tuple,points@rotation.T+center)))


def build_cell(d, vertices=48, background_index=3.554):
    """Build area-preserving holes in period units / 用周期归一化坐标生成等面积孔。"""
    a=np.deg2rad(d.position_angle_deg)
    center=(d.distance*np.cos(a),d.distance*np.sin(a))
    inclusions=(hole((0.,0.),d.fill*d.main_share,d.main_shape,d.main_aspect,d.main_angle_deg,d.main_exponent,vertices),
                hole(center,d.fill*(1-d.main_share),d.satellite_shape,d.satellite_aspect,d.satellite_angle_deg,d.satellite_exponent,vertices))
    return CachedSquareCell(background_index**2,inclusions)


def build_model(d, order=4, dz=8., fixed_wavelength_nm=950., tune_period=True, vertices=48):
    """Build C(lambda); optional Bragg tuning is a screen / 构造 C(lambda)，可选粗调周期。

    The effective-index Bragg estimate is not finite-mode frequency closure.
    正式定稿需闭合有限腔极点的失谐方程，不能以有效折射率布拉格估计代替。
    """
    cell=build_cell(d,vertices);stack=build_stack(d,cell)
    model=build_geometry_coupling(cell,stack,BASE.PC_LAYER_NAME,d.lattice_nm,950.,
        ThreeDCWTSettings(order,dz),fixed_wavelength_nm=fixed_wavelength_nm)
    if tune_period:
        d=replace(d,lattice_nm=950./model.effective_index)
        model=build_geometry_coupling(cell,stack,BASE.PC_LAYER_NAME,d.lattice_nm,950.,
            ThreeDCWTSettings(order,dz),fixed_wavelength_nm=fixed_wavelength_nm)
    return d,cell,model


def acceptable(row, strict=False):
    b=row["beam"]
    return (b["ellipticity"] <= (1.05 if strict else 1.10) and
            b["center_to_peak"] >= (.98 if strict else .90) and
            b["radial_anisotropy_l2"] <= (.10 if strict else .15))


def evaluate(d, order=4, dz=8., grid=15, vertices=48, count=12,
             output=None, tune_period=True, fixed_wavelength_nm=950.):
    """Score sampled cold modes and their actual vector FFP / 评价已采样冷腔模及矢量远场。

    Shape metrics use |Ex|^2+|Ey|^2, with a shared complex Jones scale. Separate
    center brightness, second-moment axis ratio and radial anisotropy are needed.
    光斑由 |Ex|^2+|Ey|^2 得到，两分量共用归一化；中心亮度、二阶矩轴比和
    径向非圆度各有含义，不能用一个接近 1 的轴比宣称所有方向轮廓相同。
    """
    started=time.monotonic()
    d,cell,model=build_model(d,order,dz,fixed_wavelength_nm,tune_period,vertices)
    fab=evaluate_fabrication(cell.inclusions,d.lattice_nm,rules=RULES)
    spec=FourWaveOpticalSpec(model.bragg_wavelength_nm,model.effective_index,model.group_index,
        300.,5.,tuple(model.eigenvalues_m.real/100),tuple(model.eigenvalues_m.imag/100),
        grid,"staggered",count,None,1200)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        modes=solve_finite_modes(spec,model.coupling_m,model.eigenvectors,model.radiation_fields)
    rows=[]; arrays={"coupling_m":model.coupling_m,"c1d_m":model.c1d_m,
        "c2d_m":model.c2d_m,"crad_m":model.crad_m,"eigenvalues_m":model.eigenvalues_m}
    gamma=model.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME]
    for name,mode in modes.items():
        axis,fx,fy,beam=direct_vector_far_field(mode,spec.wavelength_nm,300.,129 if output is None else 201)
        field=mode.fields.reshape(4,-1)
        local=float(np.vdot(field,model.coupling_m@field).imag/np.vdot(field,field).real)
        # Amplitude loss needs factor two for power; 500 /m = 5 /cm intensity loss.
        # 振幅损耗乘 2 才是功率损耗；500 /m 对应固定的 5 /cm 内部强度损耗。
        rows.append({"mode":name,"alpha_field_cm1":mode.alpha_per_m/100,
            "delta_per_m":mode.delta_per_m,"eigen_residual":mode.eigen_residual,
            **threshold(mode.alpha_per_m,gamma,model.effective_index,300.),"beam":beam,
            "radiative_extraction_fraction":2*local/(2*mode.alpha_per_m+500)})
        if output:
            arrays.update({f"{name}_field":mode.fields,f"{name}_radiation_x":mode.radiation_x,
                f"{name}_radiation_y":mode.radiation_y,f"{name}_far_x":fx,f"{name}_far_y":fy})
    ranked=sorted(rows,key=lambda r:r["g_th_cm1"])
    result={"id":d.identifier,"status":"completed","design":asdict(d),
        "settings":{"D":order,"dz_nm":dz,"grid":grid,"vertices":vertices,"count":count},
        "wavelength_nm":model.bragg_wavelength_nm,"effective_index":model.effective_index,
        "group_index":model.group_index,"active_Gamma":gamma,"fabrication":fab,
        "passivity_correction_per_m":model.passivity_correction_m,"modes":rows,"lowest":ranked[0],
        "relative_material_gap":ranked[1]["g_th_cm1"]/ranked[0]["g_th_cm1"]-1,
        "warnings":[str(w.message) for w in caught],"elapsed_seconds":time.monotonic()-started,
        "scope":"four selected Gamma-family states; screening only until lateral/accuracy checks"}
    result["basic_beam_pass"]=acceptable(ranked[0]);result["strict_beam_pass"]=acceptable(ranked[0],True)
    if output:
        output.mkdir(parents=True,exist_ok=True)
        arrays.update(axis_deg=axis,vertical_z_um=model.vertical_mode.z_um,vertical_field=model.vertical_mode.field)
        np.savez_compressed(output/"raw_fields.npz",**arrays);save_json(output/"result.json",result)
    return result


def from_unit(p,index=0):
    kinds=("circle","ellipse","superellipse","rounded_triangle")
    # All 16 independent shape pairs recur through the Latin-hypercube sample.
    first,second=kinds[index%4],kinds[(index//4)%4]
    return Design(fill=.055+.21*p[0],main_share=.50+.35*p[1],distance=.36+.30*p[2],
        position_angle_deg=28+34*p[3],main_shape=first,satellite_shape=second,
        main_aspect=1+.8*p[4],satellite_aspect=1+.8*p[5],main_angle_deg=-90+180*p[6],
        satellite_angle_deg=-90+180*p[7],main_exponent=2.5+2*p[8],satellite_exponent=2.5+2*p[9],
        tag=f"lhs_{index}")


def proposal_designs(count=256,seed=30021021):
    """Seeded Latin-hypercube motifs after process screening / 工艺预筛后的可复现分层采样。"""
    proposals=qmc.LatinHypercube(10,seed=seed).random(count*15)
    accepted=[];rejected=[]
    for index,p in enumerate(proposals):
        d=from_unit(p,index);fab=evaluate_fabrication(build_cell(d,32).inclusions,280.,rules=RULES)
        if fab["margin_pass"]:
            accepted.append(d)
            if len(accepted)>=count:break
        else:rejected.append({"design":asdict(d),"fabrication":fab})
    if len(accepted)<count:raise RuntimeError("not enough non-overlapping feasible motifs")
    return accepted,rejected


def write_summary(records,output):
    save_json(output/"compact_results.json",records)
    good=[r for r in records if r.get("status")=="completed"]
    rows=[]
    for r in good:
        m=r["lowest"];b=m["beam"]
        rows.append({"id":r["id"],"g_th_cm1":m["g_th_cm1"],"I_th_A":m["I_th_A"],
            "mode":m["mode"],"ellipse":b["ellipticity"],"center":b["center_to_peak"],
            "radial":b["radial_anisotropy_l2"],"gap":r["relative_material_gap"],
            "eta":m["radiative_extraction_fraction"],"basic_pass":r["basic_beam_pass"],
            "strict_pass":r["strict_beam_pass"],"process_pass":r["fabrication"]["margin_pass"],
            **{k:v for k,v in r["design"].items() if k not in ("tag",)}})
    if rows:
        with (output/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    save_json(output/"run_status.json",{"completed":len(good),"total":len(records),
        "basic_pass":sum(r["basic_beam_pass"] for r in good),"strict_pass":sum(r["strict_beam_pass"] for r in good)})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=OUT/"broad")
    p.add_argument("--count",type=int,default=256);p.add_argument("--seed",type=int,default=30021021)
    p.add_argument("--D",type=int,default=4);p.add_argument("--dz",type=float,default=8.)
    p.add_argument("--grid",type=int,default=15);p.add_argument("--designs",type=Path)
    args=p.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    if args.designs:
        designs=[Design(**x) for x in json.loads(args.designs.read_text(encoding="utf-8"))]
    else:
        designs,rejected=proposal_designs(args.count,args.seed)
        save_json(args.output/"rejected_geometry.json",rejected)
    save_json(args.output/"designs.json",[asdict(d) for d in designs]);results=[]
    for i,d in enumerate(designs):
        try:r=evaluate(d,args.D,args.dz,args.grid)
        except Exception as error:r={"id":d.identifier,"status":"failed","design":asdict(d),"error":str(error)}
        results.append(r)
        if (i+1)%8==0 or i==len(designs)-1:
            write_summary(results,args.output)
            if r["status"]=="completed":
                print(f"{i+1}/{len(designs)} g={r['lowest']['g_th_cm1']:.3f} ellipse={r['lowest']['beam']['ellipticity']:.4f} center={r['lowest']['beam']['center_to_peak']:.3f} basic={sum(x.get('basic_beam_pass',False) for x in results)}",flush=True)
            else:print(i+1,r["error"],flush=True)
    write_summary(results,args.output)

if __name__=="__main__":main()
