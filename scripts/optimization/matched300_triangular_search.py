"""Matched 300-um triangular PCSEL: central hole plus two independent satellites.

All exploratory candidates use the unchanged Inoue Table-I 190-nm-PC stack.
Only the unit-cell motif changes.  Real finite lateral poles, rather than six
Gamma eigenvalues, determine threshold ordering and the beam objective.

中文：固定 300 × 300 μm 方形外边界，分别改变主孔和两颗对角副孔。
几何筛选、低阶有限模筛选与最终谱核验是不同阶段；粗筛分数不是全局最优证明。
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
import numpy as np
from scipy.stats import qmc
from pcselsim.fabrication import TRIANGULAR_DIRECT, FabricationRules, evaluate_fabrication
from pcselsim.triangular_six_wave import TriangularEllipse, TriangularPolygon
from scripts._triangular_workflow import TriangularCandidate
from scripts.optimization.triangular_semiconductor_search import (
    Motif, build_model, evaluate_finite, json_write, write_csv, threadpool_limits)


@dataclass(frozen=True)
class ThreeHole:
    """Independent motif parameters / 三孔单胞的独立参数。

    Distances are Cartesian lengths divided by the lattice period, orientations
    are degrees, and fill is relative to triangular-cell area sqrt(3)*a^2/2.
    distance 为笛卡尔距离除以晶格周期，角度单位为度；fill 相对三角晶胞
    面积 sqrt(3)*a^2/2 定义。aspect 仅对 ellipse 生效，triangle 保持正三角形。
    """
    name: str
    fill: float = .14
    main_share: float = .56
    satellite_balance: float = .5
    main_shape: str = "circle"
    satellite1_shape: str = "circle"
    satellite2_shape: str = "circle"
    main_aspect: float = 1.
    satellite1_aspect: float = 1.
    satellite2_aspect: float = 1.
    distance1: float = .39
    distance2: float = .39
    angle1_deg: float = -30.
    angle2_deg: float = 30.
    main_orientation_deg: float = 0.
    satellite1_orientation_deg: float = 0.
    satellite2_orientation_deg: float = 0.
    rounding: float = .30
    lattice_nm: float = 0.
    triangle_samples_per_corner: int = 9


RULE = FabricationRules("fine_40_50_3nm", 40, 50, 3, 3, 5)


def candidate(spec: ThreeHole) -> TriangularCandidate:
    """Map Cartesian hole centers into oblique cell coordinates / 将笛卡尔孔心转为斜晶胞坐标。

    Satellite balance divides the non-main area; it does not change total fill.
    Rounded triangles are renormalized by their actual polygon area.
    satellite_balance 只分配副孔之间的面积，不改变总孔面积；圆角三角孔
    按实际多边形面积归一化，孔心转换后仍表示原来的物理位置。
    """
    shares = (spec.main_share, (1-spec.main_share)*spec.satellite_balance,
              (1-spec.main_share)*(1-spec.satellite_balance))
    centers = [(0., 0.)] + [tuple(np.linalg.solve(TRIANGULAR_DIRECT,
        d*np.asarray((np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))))))
        for d, a in ((spec.distance1, spec.angle1_deg), (spec.distance2, spec.angle2_deg))]
    shapes = (spec.main_shape, spec.satellite1_shape, spec.satellite2_shape)
    aspects = (spec.main_aspect, spec.satellite1_aspect, spec.satellite2_aspect)
    angles = (spec.main_orientation_deg, spec.satellite1_orientation_deg, spec.satellite2_orientation_deg)
    holes = []
    for share, center, shape, aspect, angle in zip(shares, centers, shapes, aspects, angles, strict=True):
        if share <= 0: raise ValueError("All three holes must have positive area")
        if shape == "triangle":
            unit = TriangularPolygon.rounded_regular((0., 0.), 1., corner_fraction=spec.rounding, samples_per_corner=spec.triangle_samples_per_corner)
            radius = np.sqrt(spec.fill*share/unit.fill_fraction)
            hole = TriangularPolygon.rounded_regular(center, radius, angle_deg=angle,
                corner_fraction=spec.rounding, samples_per_corner=spec.triangle_samples_per_corner)
        elif shape in ("circle", "ellipse"):
            radius = np.sqrt(spec.fill*share*(np.sqrt(3)/2)/np.pi)
            ratio = aspect if shape == "ellipse" else 1.
            hole = TriangularEllipse(center, (radius*np.sqrt(ratio), radius/np.sqrt(ratio)), angle)
        else: raise ValueError(shape)
        holes.append(hole)
    return TriangularCandidate(spec.name, tuple(holes))


def base_spec(spec):
    return Motif(spec.name, fill=spec.fill, satellites=2, pc_thickness_nm=190.,
        aperture="square", radius_um=150., lattice_nm=spec.lattice_nm or 320.)


def tune_period(spec, order=6, dz=5.):
    """Initial TE0 Bragg period only / 仅计算初始 TE0 布拉格周期，尚未闭合有限模频率。"""
    # At the fixed target wavelength the stack and Bragg effective index depend
    # on total area, not the motif's in-plane arrangement.
    from pcselsim.triangular_six_wave import TriangularLatticeCell
    from pcselsim.vertical import Layer, LayerStack
    from scripts.optimization.triangular_semiconductor_search import parameters
    p = parameters(base_spec(spec)); cell = TriangularLatticeCell(p.BACKGROUND_INDEX**2, candidate(spec).inclusions)
    average = float(np.sqrt(cell.fourier_epsilon(0, 0).real))
    stack = LayerStack(tuple(Layer(l.name,l.thickness_nm,average if l.name==p.PC_LAYER_NAME else l.refractive_index)
        for l in p.LAYERS),p.TOP_CLADDING_INDEX,p.BOTTOM_CLADDING_INDEX,p.VERTICAL_PADDING_UM)
    te = stack.solve_te0(950., dz_nm=dz)
    return replace(spec,lattice_nm=float(2*950/(np.sqrt(3)*te.effective_index)))


def geometry(spec):
    return evaluate_fabrication(candidate(spec).inclusions,spec.lattice_nm,TRIANGULAR_DIRECT,RULE)


def score(row):
    """Soft screening penalties, separate from hard acceptance / 粗筛软罚函数与硬约束分开。"""
    if row.get("status") != "finite_solved": return float("inf")
    s=row.get("three_hole_spec")
    if s and s["main_share"] < (1-s["main_share"])*max(s["satellite_balance"],1-s["satellite_balance"]):
        return float("inf")
    return (row["g_th_cm"] + 180*max(row["ellipticity"]-1.05,0)
        + 400*max(.98-row["center_to_peak"],0)
        + 220*max(row["radial_anisotropy_l2"]-.10,0)
        + 120*max(.10-row["g_th_gap_fraction"],0)
        + 600*max(.01-row["eta_rad"],0))


def catalogue(count=256, seed=20261021):
    """Seeded Sobol proposals and exact anchors / 固定种子的 Sobol 提案与规则参照点。

    The historical sampler allocates 18 coordinates but uses u[0:17]; u[17]
    does not enter a geometry. Shape-specific inactive aspects remain inactive.
    历史采样器分配 18 维，实际使用前 17 维；u[17] 不影响结构。圆孔与
    正三角孔的 aspect 也不参与形状，扫描维数不能直接视为有效自由度数。
    """
    values=qmc.Sobol(18,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(count))))[:count]
    shapes=("circle","ellipse","triangle")
    out=[]
    for i,u in enumerate(values):
        # Half mirrored/equal designs, half independent controlled asymmetry.
        symmetric=i%2==0
        s=ThreeHole(f"sobol_{i:04d}", .075+.145*u[0], .38+.40*u[1], .5 if symmetric else .32+.36*u[2],
            shapes[min(2,int(3*u[3]))], shapes[min(2,int(3*u[4]))],
            shapes[min(2,int(3*u[4 if symmetric else 5]))],
            1+.6*u[6],1+.6*u[7],1+.6*u[7 if symmetric else 8],
            .29+.24*u[9],.29+.24*u[9 if symmetric else 10],
            -30+24*(u[11]-.5),30-24*(u[11]-.5) if symmetric else 30+24*(u[12]-.5),
            180*u[13],180*u[14],-180*u[14] if symmetric else 180*u[15],.18+.25*u[16])
        out.append(s)
    # Exact symmetric reference motifs span area/share/distance systematically.
    for fill in (.08,.11,.14,.17,.20):
        for share in (.4,.55,.7):
            for d in (.34,.40,.46):
                out.append(ThreeHole(f"anchor_{len(out):04d}",fill,share,distance1=d,distance2=d))
    return out


def local_catalogue(parents,count=96,seed=20261022):
    rng=np.random.default_rng(seed); out=[]
    for i in range(count):
        parent=parents[i%len(parents)]; s=ThreeHole(**parent["three_hole_spec"])
        changes={"name":f"local_{i:04d}","lattice_nm":0.}
        # Alternate one-parameter sweeps and simultaneous local variations.
        groups=(("fill",.018,.055,.24),("main_share",.07,.32,.82),
            ("satellite_balance",.07,.26,.74),("distance1",.025,.27,.56),
            ("distance2",.025,.27,.56),("angle1_deg",5.,-47.,-13.),
            ("angle2_deg",5.,13.,47.),("main_aspect",.15,1.,1.85),
            ("satellite1_aspect",.15,1.,1.85),("satellite2_aspect",.15,1.,1.85),
            ("main_orientation_deg",15.,-180.,360.),("satellite1_orientation_deg",15.,-180.,360.),
            ("satellite2_orientation_deg",15.,-180.,360.))
        selected=(groups[i%len(groups)],) if i%2==0 else groups
        for key,scale,low,high in selected:
            changes[key]=float(np.clip(getattr(s,key)+scale*rng.normal(),low,high))
            for prefix in ("main","satellite1","satellite2"):
                if key in (prefix+"_aspect",prefix+"_orientation_deg") and getattr(s,prefix+"_shape")=="circle":
                    changes[prefix+"_shape"]="ellipse"
                    if key.endswith("orientation_deg"):
                        changes[prefix+"_aspect"]=1.06
        if i%16==15:
            changes[("main","satellite1","satellite2")[(i//16)%3]+"_shape"]="triangle"
        out.append(replace(s,**changes))
    return out


def circle_grid():
    """Dense area/share/placement control separating geometry from hole shape."""
    return [ThreeHole(f"grid_{i:04d}",fill=float(fill),main_share=float(share),distance1=float(d),distance2=float(d))
        for i,(fill,share,d) in enumerate((f,s,d) for f in np.linspace(.11,.23,9)
            for s in np.linspace(.38,.74,7) for d in np.linspace(.34,.50,9))]


def evaluate(spec,cells=9,order=6,dz=5.,count=8,save=None,extra_targets=0,full_dense=False):
    """Reject invalid geometry before optical work / 先做几何筛选，再计算有限腔冷模。

    Acceptance refers to this returned spectrum and angular window. A later
    wider spectral search can reveal a lower-threshold dark competing branch.
    通过条件只针对当前返回的谱和角窗；更广的谱搜索仍可能发现阈值更低的暗模。
    """
    spec=tune_period(spec,order,dz) if spec.lattice_nm<=0 else spec
    report=geometry(spec)
    largest = spec.main_share >= (1-spec.main_share)*max(spec.satellite_balance,1-spec.satellite_balance)
    if not report["margin_pass"] or not largest:
        return {"name":spec.name,"status":"geometry_rejected","three_hole_spec":asdict(spec),"geometry_custom":report}
    row=evaluate_finite(base_spec(spec),cells,order,dz,count,save,candidate_override=candidate(spec),
        extra_targets=extra_targets,full_dense=full_dense)
    row["three_hole_spec"]=asdict(spec);row["geometry_custom"]=report
    row["central_main_has_largest_area"]=largest
    row["score"]=score(row)
    row["strict_accepted"] = bool(row["ellipticity"]<=1.05 and row["center_to_peak"]>=.98
        and row["radial_anisotropy_l2"]<=.10 and row["g_th_gap_fraction"]>=.10)
    row["basic_accepted"] = bool(row["ellipticity"]<=1.10 and row["center_to_peak"]>=.90
        and row["radial_anisotropy_l2"]<=.15)
    if save: json_write(save/"result.json",row)
    return row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("sobol","grid","local","rescreen","refine","audit"),default="sobol")
    parser.add_argument("--input",type=Path);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--count",type=int,default=256);parser.add_argument("--limit",type=int,default=6)
    parser.add_argument("--cells",type=int,default=9);parser.add_argument("--order",type=int,default=6)
    parser.add_argument("--dz",type=float,default=5.);parser.add_argument("--eigen-count",type=int,default=8)
    parser.add_argument("--extra-targets",type=int,default=0);parser.add_argument("--full-dense",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv);args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=="sobol": specs=catalogue(args.count)
    elif args.stage=="grid": specs=circle_grid()
    else:
        parents=json.loads((args.input/"candidates.json").read_text(encoding="utf-8"))
        parents=sorted([r for r in parents if r.get("status")=="finite_solved"],key=score)[:args.limit]
        if args.stage=="local": specs=local_catalogue(parents,args.count)
        else: specs=[ThreeHole(**r["three_hole_spec"]) for r in parents]
    json_write(args.output/"run_plan.json",{"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "physical_protocol":"300 x 300 um square; Inoue Table I semiconductor stack, PC190 nm; alpha_i=5/cm; Bragg950nm",
        "retention":"Exploratory scans keep scalar JSON/CSV only. Finalists retain vector complex fields.",
        "spectrum_scope":"All distinct residual-verified poles found around all six Gamma edges plus optional real shifts; not exhaustive fine-grid spectrum."})
    json_write(args.output/"input_specs.json",[asdict(s) for s in specs])
    previous=json.loads((args.output/"candidates.json").read_text(encoding="utf-8")) if args.resume and (args.output/"candidates.json").exists() else []
    names={r["name"] for r in previous};rows=list(previous)
    with threadpool_limits(limits=1):
        for index,spec in enumerate(specs):
            if spec.name in names: continue
            start=time.perf_counter()
            try:
                save=args.output/spec.name if args.stage in ("refine","audit") else None
                row=evaluate(spec,args.cells,args.order,args.dz,args.eigen_count,save,args.extra_targets,args.full_dense)
            except Exception as error: row={"name":spec.name,"status":"failed","exception":repr(error),"three_hole_spec":asdict(spec)}
            row["elapsed_seconds"]=time.perf_counter()-start;rows.append(row)
            json_write(args.output/"candidates.json",rows)
            write_csv(args.output/"candidates.csv",rows)
            solved=sorted([r for r in rows if r.get("status")=="finite_solved"],key=score)
            json_write(args.output/"leaderboard.json",solved[:20])
            print(index+1,len(specs),spec.name,row["status"],
                *(f"{row.get(k,float('nan')):.5g}" for k in ("g_th_cm","ellipticity","center_to_peak","radial_anisotropy_l2","g_th_gap_fraction")),
                f"{row['elapsed_seconds']:.2f}s",flush=True)
    json_write(args.output/"run_status.json",{"status":"completed","candidates":len(rows),
        "finite_solved":sum(r.get("status")=="finite_solved" for r in rows),"best":min(rows,key=score)})


if __name__=="__main__":main()
