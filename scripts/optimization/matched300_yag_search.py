"""Same-stack 300-um YAG square/triangular structure search and closed finals.

Use benchmark/scan/refine for exploration and close for fixed-geometry finals.
Full exploratory matrices are discarded: JSONL rows retain enough information
to reconstruct every geometry and numerical setting.

中文：同为 300 × 300 um 物理孔径、相同纵向膜系的正方/三角晶格优化入口。
粗扫仅用于筛选；正式结论需闭合实际频率、检查竞争模，再比较共享泵浦首模。
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse,json,sys,time,warnings
from dataclasses import asdict,replace,dataclass
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c,h
from scipy.optimize import brentq
from scipy.sparse.linalg import splu
from pcselsim.hybrid_vertical import HybridLayerStack,solve_guided_te_modes
from pcselsim.vertical import Layer
from pcselsim.stratified_optics import StratifiedStack,StratifiedLayer,quarter_wave_dbr,solve_stack
from pcselsim.reflector_cwt import build_reflector_coupling
from pcselsim.geometry import SquareLatticeCell,Ellipse,PolygonInclusion
from pcselsim.matched300_yag import build_triangular_reflector_coupling,binned_lateral_weights
from pcselsim.polarization import polarization_metrics,stokes_maps
from pcselsim.triangular_six_wave import TriangularLatticeCell,TriangularEllipse,TriangularPolygon,TriangularCWTSettings,plot_triangular_lattice
from pcselsim.three_d_cwt import ThreeDCWTSettings
from pcselsim.triangular_finite import (TriangularFiniteSpec,TriangularFiniteMode,
    solve_triangular_finite_modes,triangular_finite_eigenproblem,triangular_vector_far_field)
from pcselsim.custom_analysis import FourWaveOpticalSpec,LinearMode,solve_finite_modes,finite_area_eigenproblem
from pcselsim.design_objectives import beam_metrics,beam_acceptance
from pcselsim.finite_discretization import relative_residual
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.fabrication import FabricationRules,evaluate_fabrication,TRIANGULAR_DIRECT,inclusion_boundary
from pcselsim.solid_state import YbYAGMediumConfig
from pcselsim.ybyag_spatial_rates import SpatialYbYAGModel
from pcselsim.ybyag_multimode import SpatialMultimodeYbYAG,solve_multimode_spatial_ybyag
from pcselsim.ybyag_threshold import shared_inversion_pump_threshold
from pcselsim.ybyag_pump_geometry import PumpGeometryConfig
from run_ybyag_pcsel import active_weights,lateral_energy_weights
from scripts.optimization.matched300_square_search import Design as SquareDesign,build_cell,from_unit
from scripts.optimization.matched300_triangular_search import ThreeHole,candidate,catalogue
from scripts.optimization.square_semiconductor_search import direct_vector_far_field

OUT=ROOT/"results/current/matched300_20261021/yag"
PC="passive YAG photonic crystal"
RULE=FabricationRules("matched_YAG_50nm_span_40nm_gap",minimum_gap_nm=40,
    minimum_caliper_nm=50,edge_bias_budget_nm=3,placement_budget_nm=3,minimum_corner_radius_nm=5)
# Use the named fields above, not the legacy rule label, for fabrication limits.
# 工艺判据以字段为准：间隙 40 nm、最小宽度 50 nm，含边缘/位置误差预算。


@lru_cache(maxsize=200000)
def _cached_triangular_fourier(cell,m,n):
    return TriangularLatticeCell.fourier_epsilon(cell,m,n)


@dataclass(frozen=True)
class CachedYagTriangularCell(TriangularLatticeCell):
    """Bounded exact geometry-only memoization; no wavelength data cached."""
    def fourier_epsilon(self,m,n):
        return _cached_triangular_fourier(self,int(m),int(n))


def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False),encoding="utf8")


def geometry(lattice,design):
    if "_yag_explicit_holes" in design:
        classes=dict(square_ellipse=Ellipse,square_polygon=PolygonInclusion,
                     triangular_ellipse=TriangularEllipse,triangular_polygon=TriangularPolygon)
        def tuples(value):return tuple(tuples(v) for v in value) if isinstance(value,(list,tuple)) else value
        holes=tuple(classes[h["type"]](**{k:tuples(v) for k,v in h["parameters"].items()}) for h in design["_yag_explicit_holes"])
        return SquareLatticeCell(1.8154**2,holes) if lattice=="square" else CachedYagTriangularCell(1.8154**2,holes)
    if lattice=="square":
        return build_cell(SquareDesign(**{k:v for k,v in design.items() if not k.startswith("_yag_")}),vertices=96,background_index=1.8154)
    args={k:v for k,v in design.items() if not k.startswith("_yag_")}
    inclusions=candidate(ThreeHole(**args)).inclusions
    base_samples=int(args.get("triangle_samples_per_corner",9))
    samples=int(design.get("_yag_triangle_samples_per_corner",base_samples))
    if samples!=base_samples:
        resampled=[]
        for hole,key in zip(inclusions,("main","satellite1","satellite2")):
            if not isinstance(hole,TriangularPolygon):resampled.append(hole);continue
            center=tuple(np.mean(np.asarray(hole.vertices_fractional),axis=0))
            unit=TriangularPolygon.rounded_regular((0.,0.),1.,corner_fraction=design["rounding"],samples_per_corner=samples)
            radius=np.sqrt(hole.fill_fraction/unit.fill_fraction)
            resampled.append(TriangularPolygon.rounded_regular(center,radius,
                angle_deg=design[key+"_orientation_deg"],corner_fraction=design["rounding"],
                samples_per_corner=samples,epsilon=hole.epsilon))
        inclusions=tuple(resampled)
    # Legacy semiconductor triangles are equilateral rounded polygons. The
    # explicit version flag below enables a YAG-only, area-preserving local
    # aspect deformation, so old stored parameters keep their original shape.
    # 默认圆角三角孔仍为正三角形；aspect 参数只有显式开关开启后才改变三角孔。
    if design.get("_yag_triangle_aspect_enabled",False):
        changed=[]
        for hole,key in zip(inclusions,("main","satellite1","satellite2")):
            if not isinstance(hole,TriangularPolygon):changed.append(hole);continue
            points=np.asarray(hole.vertices_fractional)@TRIANGULAR_DIRECT.T
            center=points.mean(axis=0)
            angle=np.deg2rad(design[key+"_orientation_deg"]);aspect=design[key+"_aspect"]
            rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            transform=rotation@np.diag([np.sqrt(aspect),1/np.sqrt(aspect)])@rotation.T
            points=(points-center)@transform.T+center
            changed.append(replace(hole,vertices_fractional=tuple(map(tuple,np.linalg.solve(TRIANGULAR_DIRECT,points.T).T))))
        inclusions=tuple(changed)
    return CachedYagTriangularCell(1.8154**2,inclusions)


def structure(lattice,design,family=0):
    """Build the common layer stack / 两种晶格共用 500 nm 增益层和 120 nm PC 层。"""
    cell=geometry(lattice,design)
    index=np.sqrt(cell.fourier_epsilon(0,0).real)
    layers=(Layer("lower SiO2 guide spacer",2000.,1.45),
            Layer("Yb:YAG gain",500.,1.8166),Layer(PC,120.,index))
    guide=HybridLayerStack(layers,1.,1.45,2.,family)
    radiation=StratifiedStack((*quarter_wave_dbr(1030.,2.1,1.45,16,first_layer="low"),
        *(StratifiedLayer(l.name,l.thickness_nm,l.refractive_index) for l in layers)),1.8154,1.)
    return cell,guide,radiation


def optical(lattice,design,order=4,dz=5.,wavelength=1030.,period=None,family=0):
    cell,guide,radiation=structure(lattice,design,family)
    if period is None:
        n=guide.solve_te0(1030.,dz).effective_index
        period=1030./n*(1. if lattice=="square" else 2/np.sqrt(3))
    factory=build_reflector_coupling if lattice=="square" else build_triangular_reflector_coupling
    settings=ThreeDCWTSettings(order,dz) if lattice=="square" else TriangularCWTSettings(order,dz)
    model=factory(cell,guide,radiation,PC,period,1030.,settings,fixed_wavelength_nm=wavelength)
    return model,float(period),guide


def finite(lattice,model,grid=15,count=12,solve_selected=True):
    # Match physical aperture, not node count: triangular radius is the half-side.
    # 两种离散网格节点数可不同，但方形物理孔径均为 300 × 300 um。
    r=model.result
    if lattice=="square":
        spec=FourWaveOpticalSpec(r.bragg_wavelength_nm,r.effective_index,r.group_index,300.,.1,
            tuple(r.eigenvalues_m.real/100),tuple(r.eigenvalues_m.imag/100),grid,
            discretization="staggered",eigensolutions_per_target=count,
            dense_eigenproblem_limit=0,eigen_maxiter=1200)
        modes=solve_finite_modes(spec,r.coupling_m,r.eigenvectors,r.radiation_fields) if solve_selected else {}
        problem=finite_area_eigenproblem(spec,r.coupling_m)
        return spec,modes,problem,None
    spec=TriangularFiniteSpec(150.,grid,0.,count,"square","staggered")
    problem,mesh=triangular_finite_eigenproblem(r,spec)
    modes={}
    if solve_selected:
        # Search more than one pole near each Gamma edge, with a bounded
        # nonnormal eigensolve. The legacy smooth-family selector can spend
        # many thousands of iterations on an irrelevant weak-feedback branch.
        pool=seed_pool(lattice,model,spec,problem,mesh,count,0)
        if not pool:raise RuntimeError("No residual-verified Gamma-neighbour finite poles")
        modes={f"pool_{i}":as_mode(lattice,field,value,r,mesh,f"pool_{i}")
               for i,(value,field) in enumerate(pool[:12])}
        for mode in modes.values():
            mode.eigen_residual=relative_residual(problem.center_operator(),mode.fields,
                                                   complex(mode.delta_per_m,mode.alpha_per_m))
    return spec,modes,problem,mesh


def far(lattice,spec,mode,wavelength,samples=101):
    if lattice=="square":
        axis,fx,fy,metrics=direct_vector_far_field(mode,wavelength,300.,samples)
    else:
        ff=triangular_vector_far_field(mode,wavelength,1.,samples)
        axis,fx,fy=ff.angle_deg,ff.field_x,ff.field_y
        metrics=beam_metrics(axis,fx,fy,requested_view_deg=1.,source_nyquist_deg=ff.alias_free_square_view_deg)
    return axis,fx,fy,metrics


def uniform_edge_proxy(loss,qgain):
    # Constant nominal cross sections, same 300um x 500nm entrance area.
    # 仅为均匀反转侧泵排序代理；未求泵浦耗尽，不能当作共享泵浦阈值输出。
    nt=1.38e27; sa,se=.126e-20*1e-4,2.e-20*1e-4
    f=(loss/qgain/nt+sa)/(se+sa)
    denominator=(.70e-20*(1-f)-.10e-20*f)*1e-4
    return (float(f*h*c/(940e-9)/(.95e-3*denominator)*(300e-6*500e-9))
            if denominator>0 else np.inf)


def material(wavelength):
    return YbYAGMediumConfig(laser_wavelength_nm=wavelength,refractive_index=1.8166,
        dopant_density_cm3=1.38e21,spontaneous_emission_factor=0.,upper_state_lifetime_ms=.95,
        pump_wavelength_nm=940.,pump_absorption_cross_section_cm2=.70e-20,
        pump_emission_cross_section_cm2=.10e-20,laser_absorption_cross_section_cm2=.126e-20,
        laser_emission_cross_section_cm2=2.e-20,pumped_area_um2=300.**2,gain_thickness_um=.5)


def mode_row(lattice,model,guide,spec,mode,wavelength,samples=101):
    up,down=model.side_power_losses(mode.fields)
    widths,qe,qg=active_weights(model.result,guide,24)
    loss=2*mode.alpha_per_m+10.
    # alpha is field loss (/m); 10/m is the stated 0.1/cm internal POWER loss.
    # 先统一为功率损耗，再除全模增益权重；最后除 100 转为材料增益 cm^-1。
    axis,fx,fy,beam=far(lattice,spec,mode,wavelength,samples)
    row=dict(name=mode.name,delta_per_m=mode.delta_per_m,alpha_field_per_m=mode.alpha_per_m,
        total_power_loss_per_m=loss,up_power_loss_per_m=up,down_power_loss_per_m=down,
        passive_cavity_power_loss_per_m=2*mode.alpha_per_m,internal_power_loss_per_m=10.,
        boundary_transport_power_loss_per_m=float(2*mode.alpha_per_m-up-down),
        up_fraction=up/loss,gain_weight=qg.sum(),required_material_gain_cm1=loss/qg.sum()/100,
        uniform_edge_proxy_W=uniform_edge_proxy(loss,qg.sum()),beam=beam,
        beam_acceptance=beam_acceptance(beam,require_sampling=True),
        eigen_residual=mode.eigen_residual,polarization=polarization_metrics(axis,fx,fy,core_radius_deg=.2))
    return row,(axis,fx,fy,widths,qe,qg)


def export_actual_geometry(lattice,design,period,fabrication,path):
    direct=np.eye(2) if lattice=="square" else TRIANGULAR_DIRECT
    cell=geometry(lattice,design)
    kinds=([design["main_shape"],design["satellite_shape"]] if lattice=="square" else
           [design["main_shape"],design["satellite1_shape"],design["satellite2_shape"]])
    roles=["central_main","diagonal_satellite"] if lattice=="square" else ["central_main","diagonal_satellite_1","diagonal_satellite_2"]
    holes=[]
    for hole,kind,role in zip(cell.inclusions,kinds,roles):
        boundary=inclusion_boundary(hole,direct,384)*period
        x,y=boundary.T;cross=x*np.roll(y,-1)-np.roll(x,-1)*y
        center=np.array([np.sum((x+np.roll(x,-1))*cross),np.sum((y+np.roll(y,-1))*cross)])/(3*np.sum(cross))
        h=dict(role=role,shape=kind,boundary_cartesian_nm=boundary,center_cartesian_nm=center,
            area_nm2=abs(np.sum(cross))/2,epsilon=hole.epsilon,
            boundary_sampling_points=len(boundary))
        if hasattr(hole,"radii"):h.update(radii_nm=np.array(hole.radii)*period,orientation_deg=hole.angle_deg)
        if hasattr(hole,"radii_over_a"):h.update(radii_nm=np.array(hole.radii_over_a)*period,orientation_deg=hole.angle_deg)
        holes.append(h)
    save(path,dict(lattice=lattice,lattice_constant_nm=period,nominal_aperture_um=[300,300],
        direct_basis_cartesian_nm=direct*period,holes=holes,fabrication=fabrication,
        geometry_parameters=design,
        triangle_samples_per_corner=int(design.get("_yag_triangle_samples_per_corner",design.get("triangle_samples_per_corner",9))) if lattice=="triangular" else None,
        triangle_area_preserving_aspect_enabled=bool(design.get("_yag_triangle_aspect_enabled",False)),
        common_stack="500nm gain /120nm PC /2um SiO2 /16-pair DBR"))


def evaluate(lattice,design,order=4,dz=5.,grid=15,count=12):
    start=time.monotonic()
    # Reuse only the semiconductor geometry, never serialize its 190nm stack
    # fields as if they described this 120nm-PC YAG structure.
    design={k:v for k,v in design.items() if k not in
            ("lattice_nm","length_um","pc_nm","gaas_nm","qw_spacer_nm")}
    model,period,guide=optical(lattice,design,order,dz)
    fab=evaluate_fabrication(geometry(lattice,design).inclusions,period,
        None if lattice=="square" else TRIANGULAR_DIRECT,RULE)
    if not fab["margin_pass"]:
        return dict(status="geometry_rejected",lattice=lattice,design=design,period_nm=period,fabrication=fab)
    spec,modes,problem,mesh=finite(lattice,model,grid,count)
    rows=[mode_row(lattice,model,guide,spec,m,1030.,81)[0] for m in modes.values()]
    rows.sort(key=lambda r:r["required_material_gain_cm1"])
    b=rows[0]["beam"]
    gap=rows[1]["required_material_gain_cm1"]/rows[0]["required_material_gain_cm1"]-1
    score=(rows[0]["required_material_gain_cm1"]*(1+10*max(b["ellipticity"]-1.05,0)
        +20*max(.98-b["center_to_peak"],0)+10*max(b["radial_anisotropy_l2"]-.10,0)+4*max(.10-gap,0))
        +10*max(.01-rows[0]["up_fraction"],0))
    if rows[0]["required_material_gain_cm1"]>=23.93265:
        score+=1000.+100.*(rows[0]["required_material_gain_cm1"]-23.93265)
    return dict(status="frozen_screen",lattice=lattice,design=design,period_nm=period,
        settings=dict(order=order,dz_nm=dz,grid=grid,count=count),modes=rows,lowest=rows[0],
        optical_theorem_residual=model.optical_theorem_residual,fabrication=fab,
        relative_material_gap=gap,
        isolated_TE_count=len(solve_guided_te_modes(guide,1030.,dz,16)),score=score,
        seconds=time.monotonic()-start,scope="frozen geometry screen; proxy is not a pump threshold",
        pole_selector="bounded multi-pole per Gamma; lowest actual returned" if lattice=="triangular" else "four smooth Gamma-connected families")


def proposals(lattice,count=160):
    if lattice=="square":
        from scipy.stats import qmc
        for i,p in enumerate(qmc.LatinHypercube(10,seed=3001030).random(count)):
            yield asdict(replace(from_unit(p,i),lattice_nm=600.))
        # Scalar reference and balanced two-hole anchors.
        for fill in (.10,.16,.22,.25,.28):
            for distance in (.38,.46,.54,.636396):
                for share in (.50,.60,.75,.85):
                    yield asdict(SquareDesign(fill=fill,main_share=share,distance=distance,tag="anchor"))
    else:
        for d in catalogue(count,seed=3001030):
            # Keep the central main hole largest across the YAG search.
            yield asdict(replace(d,main_share=max(.50,d.main_share)))


def command_benchmark(args):
    rows=[]
    for lattice in ("square","triangular"):
        d=(asdict(SquareDesign(fill=.16,main_share=.6,distance=.53)) if lattice=="square"
           else asdict(ThreeHole("benchmark",fill=.16,main_share=.6,distance1=.4,distance2=.4)))
        for grid in ((11,) if lattice=="square" else (6,)):
            row=evaluate(lattice,d,2,10.,grid,8);rows.append(row)
            print(lattice,row["status"],row.get("seconds"),row.get("lowest",{}).get("uniform_edge_proxy_W"),flush=True)
    save(args.output/"benchmark.json",rows)


def command_scan(args):
    args.output.mkdir(parents=True,exist_ok=True)
    for lattice in args.lattices:
        target=args.output/(lattice+"_screen.jsonl")
        if target.exists():raise FileExistsError(target)
        rows=[]
        with target.open("w",encoding="utf8") as stream:
            for index,d in enumerate(proposals(lattice,args.count)):
                try:row=evaluate(lattice,d,args.order,args.dz,args.square_grid if lattice=="square" else args.tri_grid,args.roots)
                except Exception as error:row=dict(status="failed",lattice=lattice,design=d,error=repr(error))
                row["index"]=index;stream.write(json.dumps(clean(row))+"\n");stream.flush()
                if row["status"]=="frozen_screen":rows.append(row)
                if index%10==0:print(lattice,index,row["status"],row.get("score"),flush=True)
        ranked=sorted(rows,key=lambda r:r["score"])
        save(args.output/(lattice+"_screen_best.json"),ranked[:12])


def command_refine(args):
    parents=json.loads(args.input.read_text(encoding="utf8"))
    if not isinstance(parents,list):parents=[parents]
    rng=np.random.default_rng(3001031);rows=[]
    lattice=parents[0]["lattice"]
    keys_square=(("fill",.018,.055,.32),("main_share",.07,.5,.90),
        ("distance",.035,.30,.70),("position_angle_deg",6.,22.,68.),
        ("main_aspect",.2,1.,2.),("satellite_aspect",.2,1.,2.),
        ("main_angle_deg",18.,-180.,180.),("satellite_angle_deg",18.,-180.,180.),
        ("main_exponent",.5,2.,5.),("satellite_exponent",.5,2.,5.))
    keys_tri=(("fill",.018,.055,.30),("main_share",.07,.5,.88),
        ("satellite_balance",.06,.25,.75),("distance1",.03,.25,.58),("distance2",.03,.25,.58),
        ("angle1_deg",5.,-48.,-12.),("angle2_deg",5.,12.,48.),
        ("main_aspect",.2,1.,2.),("satellite1_aspect",.2,1.,2.),("satellite2_aspect",.2,1.,2.),
        ("main_orientation_deg",18.,-180.,360.),("satellite1_orientation_deg",18.,-180.,360.),
        ("satellite2_orientation_deg",18.,-180.,360.),("rounding",.05,.12,.46))
    args.output.mkdir(parents=True,exist_ok=True);target=args.output/(lattice+"_local.jsonl")
    if target.exists():raise FileExistsError(target)
    with target.open("w",encoding="utf8") as stream:
        for i in range(args.count):
            parent=parents[i%min(len(parents),6)];d=parent["design"].copy()
            keys=keys_square if lattice=="square" else keys_tri
            changed=(keys[i%len(keys)],) if i%2==0 else keys
            for key,scale,lo,hi in changed:d[key]=float(np.clip(d[key]+scale*rng.normal(),lo,hi))
            d["fill"]=max(d["fill"],getattr(args,"minimum_fill",.055))
            if lattice=="square":d["tag"]=f"local_{i:04d}"
            else:d["name"]=f"local_{i:04d}"
            try:row=evaluate(lattice,d,args.order,args.dz,args.square_grid if lattice=="square" else args.tri_grid,args.roots)
            except Exception as error:row=dict(status="failed",lattice=lattice,design=d,error=repr(error))
            row.update(index=i,parent_rank=i%min(len(parents),6),changed_parameters=[x[0] for x in changed])
            stream.write(json.dumps(clean(row))+"\n");stream.flush()
            if row["status"]=="frozen_screen":rows.append(row)
            if i%10==0:print("local",lattice,i,row["status"],row.get("score"),flush=True)
    save(args.output/(lattice+"_local_best.json"),sorted(rows+parents,key=lambda r:r["score"])[:12])


def as_mode(lattice,field,value,r,mesh,name):
    rx,ry=(r.radiation_fields(field) if lattice=="square" else r.radiation_amplitudes(field))
    if lattice=="square":
        return LinearMode(name,float(value.real),float(value.imag),field,0.,rx,ry,discretization="staggered")
    return TriangularFiniteMode(name,float(value.real),float(value.imag),field,0.,mesh,rx,ry,discretization="staggered")


def seed_pool(lattice,model,spec,problem,mesh,side_roots=16,extra_targets=2,
              coarse_qz=False,audit_path=None):
    """Retain actual low-loss poles across every Gamma cluster and lateral shifts.

    This deliberately includes the modes the usual one-fundamental-per-band
    selector discards. It is a bounded sparse census, not a full spectrum claim.
    """
    nchan=4 if lattice=="square" else 6
    shape=(4,spec.grid_points,spec.grid_points) if lattice=="square" else (6,mesh.points)
    axis=np.arange(problem.stiffness.shape[0]);v0=1+.19*np.cos(axis)+.17j*np.sin(3*axis)
    values=[];fields=[]
    operator=problem.center_operator()
    shift=np.pi/(300e-6)
    targets=[complex(v)+k*shift for v in model.result.eigenvalues_m
             for k in range(-extra_targets,extra_targets+1)]
    target_seeds=[v0]*len(targets)
    target_kinds=["Gamma_and_neighbours"]*len(targets)
    audit=dict(coarse_qz=bool(coarse_qz),gamma_target_count=len(targets),targets=[])
    if coarse_qz:
        # Coarse finite poles are much better conditioned shifts than the
        # infinite-periodic Gamma eigenvalues of an open weak-feedback cavity.
        # Put the finite poles first and bound supplementary failed searches.
        supplementary=[complex(v) for v in model.result.eigenvalues_m]
        targets=[];target_seeds=[];target_kinds=[]
        coarse_size=17 if lattice=="square" else 9
        cs,_,cp,cmesh=finite(lattice,model,coarse_size,max(8,side_roots),False)
        cv0=np.ones(cp.stiffness.shape[0],complex)
        cp.near(complex(model.result.eigenvalues_m[0]),min(40,len(cv0)-2),cv0,
                dense_limit=len(cv0)+1)
        cv,cf=cp._dense_spectrum
        valid=np.flatnonzero(cv.imag>=-1e-7)
        order=valid[np.argsort(cv[valid].imag)]
        # Keep the global lowest 40 complex poles, not a uniform real-axis
        # sample that can miss a narrow isolated low-loss spectral cluster.
        chosen=order[:40]
        audit.update(coarse_grid=coarse_size,coarse_dimension=len(cv0),
            residual_verified_full_QZ_count=len(cv),coarse_lowest40=[dict(delta_per_m=cv[i].real,
                alpha_field_per_m=cv[i].imag) for i in chosen])
        from scipy.interpolate import RegularGridInterpolator,LinearNDInterpolator
        for i in chosen:
            if lattice=="square":
                old=cf[:,i].reshape(4,coarse_size,coarse_size)
                source=((np.arange(coarse_size)+.5)/coarse_size-.5)*300
                dest=((np.arange(spec.grid_points)+.5)/spec.grid_points-.5)*300
                yy,xx=np.meshgrid(dest,dest,indexing="ij")
                interpolated=np.array([RegularGridInterpolator((source,source),f,
                    bounds_error=False,fill_value=0.)(np.stack((yy,xx),axis=-1)) for f in old]).ravel()
            else:
                old=cf[:,i].reshape(6,cmesh.points)
                xy=np.column_stack((cmesh.x_um,cmesh.y_um));dest=np.column_stack((mesh.x_um,mesh.y_um))
                interpolated=np.array([LinearNDInterpolator(xy,f,fill_value=0.)(dest) for f in old]).ravel()
            targets.append(complex(cv[i]));target_seeds.append(interpolated if np.linalg.norm(interpolated)>0 else v0)
            target_kinds.append("complete_coarse_QZ_lowest40")
        # The transverse-detuning two-wave limits are independent challenge
        # shifts. High-perpendicular initial fields help reveal stripe-like
        # clusters which even a complete coarse grid can underresolve.
        from scripts.diagnostics.audit_counterpropagating_stripes import stripe_pair
        if lattice=="square":
            coordinates=((np.arange(spec.grid_points)+.5)/spec.grid_points-.5)*300e-6
            yy,xx=np.meshgrid(coordinates,coordinates,indexing="ij")
            spacing=300e-6/spec.grid_points;directions=((1.,0.),(0.,1.))
        else:
            xx,yy=mesh.x_um*1e-6,mesh.y_um*1e-6
            spacing=mesh.spacing_m;directions=((.5,-np.sqrt(3)/2),(.5,np.sqrt(3)/2),(1.,0.))
        pair_audits=[]
        gain_weight=model.result.vertical_mode.confinement["Yb:YAG gain"]*1.8166/model.result.effective_index
        for pair,direction in enumerate(directions):
            length=300. if lattice=="square" or pair==2 else 300/np.sin(np.pi/3)
            try:
                pair_result=stripe_pair(model.result.coupling_m[2*pair:2*pair+2,2*pair:2*pair+2],
                    length,gain_weight,.1,grids=(51,101))
                pole=pair_result["independent_continuous_transfer_pole"]
                pair_audits.append(dict(pair=pair,length_um=length,**pair_result))
                if not pole["converged"]:continue
                dx,dy=direction;longitudinal=dx*xx+dy*yy;perpendicular=-dy*xx+dx*yy
                envelope=np.maximum(np.cos(np.pi*longitudinal/(length*1e-6)),0.)*np.exp(-(perpendicular/40e-6)**2)
                for q in (.35,.65):
                    stripe=envelope*np.exp(1j*q*np.pi/spacing*perpendicular)
                    initial=np.zeros(shape,complex);initial[2*pair]=stripe;initial[2*pair+1]=1j*stripe
                    initial=initial.ravel()/np.linalg.norm(initial)
                    rng=np.random.default_rng(1030300+pair*10+int(q*100))
                    initial+=.01*(rng.normal(size=initial.size)+1j*rng.normal(size=initial.size))/np.sqrt(initial.size)
                    for imag_scale in (1.,.5,.0):
                        targets.append(complex(pole["delta_per_m"],pole["alpha_field_per_m"]*imag_scale))
                        target_seeds.append(initial);target_kinds.append("analytic_stripe_high_perpendicular")
            except Exception as error:pair_audits.append(dict(pair=pair,error=repr(error)))
        audit["independent_pair_limit_challenges"]=pair_audits
        # The full common near-1030 analysis band supplies additional uniform
        # targets; their role is supplementary to the coarse complete census.
        for target in supplementary:
            targets.append(target);target_seeds.append(v0);target_kinds.append("Gamma_supplement")
        for real in np.linspace(-6e5,6e5,13):
            targets.append(complex(real,max(10.,min(model.result.eigenvalues_m.imag))))
            target_seeds.append(v0);target_kinds.append("wide_band_supplement")
    for target_index,(target,initial,kind) in enumerate(zip(targets,target_seeds,target_kinds)):
        try:
            iterations=(300 if kind=="complete_coarse_QZ_lowest40" else
                        400 if kind=="analytic_stripe_high_perpendicular" else (80 if coarse_qz else 300))
            vv,ff=problem.near(target,side_roots,initial,tol=1e-9,maxiter=iterations,dense_limit=0)
        except Exception as error:
            audit["targets"].append(dict(target_real=target.real,target_imag=target.imag,kind=kind,error=repr(error)))
            if audit_path is not None:save(audit_path,{**audit,"completed_targets":target_index+1,"planned_targets":len(targets)})
            continue
        before=len(values)
        for val,field in zip(vv,ff.T):
            if val.imag < -1e-7:continue
            residual=relative_residual(operator,field,val)
            if residual>2e-7:continue
            if not independent_eigenmode(val,field,values,fields):continue
            values.append(val);fields.append(field)
        audit["targets"].append(dict(target_real=target.real,target_imag=target.imag,
            kind=kind,returned=len(vv),new_independent=len(values)-before))
        if audit_path is not None:
            save(audit_path,{**audit,"completed_targets":target_index+1,"planned_targets":len(targets)})
    ordering=np.argsort(np.imag(values))
    audit["total_independent_returned"]=len(values)
    audit["planned_targets"]=len(targets);audit["completed_targets"]=len(audit["targets"])
    if audit_path is not None:save(audit_path,audit)
    return [(complex(values[i]),fields[i].reshape(shape)) for i in ordering]


def close_seed(lattice,design,period,reference_model,seed,spec,mesh,order,dz,family=0):
    """Adaptive local-frequency continuation, with the physical period fixed.

    A dense stripe family can rotate its field over several nanometres. Each
    accepted step follows the previous actual eigenfield; the original field
    overlap is retained as a diagnostic rather than a false global constraint.
    Every final state must solve the original finite pencil and frequency root.

    中文：本征值复虚部为振幅损耗；实部用于冷腔频率闭合。局部场重叠负责跟踪
    同一分支，最终仍须同时通过原始本征方程与频率残差检查。
    """
    value0,field0=seed;source=field0.ravel()/np.linalg.norm(field0)
    ref=reference_model.result.bragg_wavelength_nm
    trace=[];cache={}
    states={float(ref):dict(wavelength=float(ref),optical=reference_model,value=value0,
        field=field0,reference_only=True)}

    def single_step(wavelength,previous):
        model,_,guide=optical(lattice,design,order,dz,wavelength,period,family)
        if lattice=="square":
            local_spec=replace(spec,wavelength_nm=wavelength,effective_index=model.result.effective_index,
                group_index=model.result.group_index)
            problem=finite_area_eigenproblem(local_spec,model.result.coupling_m);local_mesh=None
        else:
            local_spec=spec;problem,local_mesh=triangular_finite_eigenproblem(model.result,spec)
        old=previous["field"];vector=old.ravel()/np.linalg.norm(old)
        flat=old.reshape(len(model.result.coupling_m),-1)
        change=model.result.coupling_m-previous["optical"].result.coupling_m
        predictor=previous["value"]+np.einsum("ik,ij,jk->",flat.conj(),change,flat)/np.sum(abs(flat)**2)
        overlap=0.;last_error=None;solver="shift_invert_Arnoldi"
        operator=problem.center_operator()
        # A previously converged field often needs only a few center-field
        # inverse iterations. This avoids rediscovering a dense stripe cluster
        # through an arbitrary many-vector Arnoldi basis at every tiny step.
        # Acceptance still requires the unmodified pencil's strict residual.
        q=vector.copy();z=predictor+complex(1e-5,2e-5)
        for iteration in range(12):
            try:
                links=splu((problem.stiffness-z*problem.mass).tocsc()).solve(q)
                new=problem.mass@links;new/=np.linalg.norm(new)
                applied=operator@new;candidate=complex(np.vdot(new,applied))
                residual_now=np.linalg.norm(applied-candidate*new)/max(np.linalg.norm(applied),abs(candidate),1e-30)
                overlap_now=float(abs(vector.conj()@new)**2)
                if residual_now<1e-11 and overlap_now>.75:
                    value=candidate;field=new.reshape(field0.shape);overlap=overlap_now
                    solver="physical_center_inverse_iteration";break
                q=new;z=candidate+complex(1e-9,2e-9)
            except Exception as error:last_error=error;break
        for count in (() if overlap>.75 else (6,16,32)):
            try:
                vv,ff=problem.near(predictor+complex(.5,.7),count,vector,tol=1e-10,maxiter=800,dense_limit=0)
                overlaps=abs(vector.conj()@ff)**2;chosen=int(np.argmax(overlaps));overlap=float(overlaps[chosen])
                if overlap>.75:break
            except Exception as error:last_error=error
        if overlap<.75:
            raise RuntimeError(f"Local continuation lost field at {wavelength:.9f}nm: overlap={overlap}; {last_error!r}")
        if solver=="shift_invert_Arnoldi":value=vv[chosen];field=ff[:,chosen].reshape(field0.shape)
        if value.imag < -1e-7:raise RuntimeError("Passive tracked pole has negative loss")
        beta=2*np.pi*model.result.effective_index/(wavelength*1e-9)
        beta0=(2*np.pi if lattice=="square" else 4*np.pi/np.sqrt(3))/(period*1e-9)
        residual=float((beta**2-beta0**2)/(2*beta0)-value.real)
        eigen_residual=relative_residual(operator,field,value)
        if eigen_residual>2e-7:raise RuntimeError(f"Continuation finite residual {eigen_residual}")
        mode=replace(as_mode(lattice,field,value,model.result,local_mesh,"closed"),eigen_residual=eigen_residual)
        seed_overlap=float(abs(source.conj()@field.ravel())**2/np.sum(abs(field)**2))
        record=dict(wavelength_nm=float(wavelength),frequency_residual_per_m=residual,
            delta_per_m=float(value.real),alpha_field_per_m=float(value.imag),
            overlap=overlap,local_step_overlap=overlap,original_seed_overlap=seed_overlap,
            previous_wavelength_nm=previous["wavelength"],eigen_residual=eigen_residual,
            tracking_solver=solver,
            optical_theorem_residual=model.optical_theorem_residual)
        trace.append(record)
        state=dict(wavelength=float(wavelength),optical=model,value=value,field=field)
        cache[float(wavelength)]=(residual,model,guide,local_spec,mode,overlap)
        states[float(wavelength)]=state
        return state

    def evaluate(wavelength):
        wavelength=float(wavelength)
        if wavelength in cache:return cache[wavelength]
        previous=min(states.values(),key=lambda s:abs(s["wavelength"]-wavelength))
        if previous.get("reference_only") and previous["wavelength"]==wavelength:
            single_step(wavelength,previous)
            return cache[wavelength]
        steps=0
        while wavelength not in cache:
            delta=wavelength-previous["wavelength"]
            step=np.sign(delta)*min(abs(delta),1.0)
            while True:
                trial=wavelength if abs(step)>=abs(delta)*(1-1e-12) else previous["wavelength"]+step
                try:
                    previous=single_step(float(trial),previous)
                    break
                except RuntimeError:
                    step/=2
                    if abs(step)<2e-4:raise
            steps+=1
            if steps>3000:raise RuntimeError("Frequency continuation exceeded bounded step count")
        return cache[wavelength]
    # First-order predictor includes the background mismatch of the already
    # fabricated period. It is only a bracket predictor, never the output.
    beta=2*np.pi*reference_model.result.effective_index/(ref*1e-9)
    beta0=(2*np.pi if lattice=="square" else 4*np.pi/np.sqrt(3))/(period*1e-9)
    background=(beta**2-beta0**2)/(2*beta0)
    derivative=beta/beta0*(-2*np.pi*reference_model.result.group_index/(ref*1e-9)**2)*1e-9
    estimate=ref+(value0.real-background)/derivative
    if not 1000.<=estimate<=1060.:
        raise RuntimeError(f"Tracked-pole predictor {estimate:.6f}nm outside common 1000-1060nm analysis band; no material-gain assertion outside this band")
    half=.25
    while True:
        lo,hi=max(1000.,estimate-half),min(1060.,estimate+half)
        if evaluate(lo)[0]*evaluate(hi)[0]<=0:break
        half*=2
        if (lo==1000. and hi==1060.) or half>64:raise RuntimeError("No tracked fixed-geometry frequency root in common 1000-1060nm analysis band")
    wavelength=float(brentq(lambda w:evaluate(w)[0],lo,hi,xtol=2e-8,rtol=1e-12))
    residual,model,guide,local_spec,mode,overlap=evaluate(wavelength)
    if abs(residual)>.05:raise RuntimeError("Closed root exceeds 0.05/m residual")
    trace.append(dict(wavelength_nm=wavelength,frequency_residual_per_m=residual,
        delta_per_m=mode.delta_per_m,alpha_field_per_m=mode.alpha_per_m,
        overlap=overlap,closure_complete=True,optical_theorem_residual=model.optical_theorem_residual))
    return wavelength,model,guide,local_spec,mode,trace


def distinct_closed_modes(rows,stored,wavelength_tolerance_nm=1e-7):
    """Reject repeated closed poles before giving them separate populations.

    C(lambda) differs at different frequencies. Compare subspaces only within
    one TE family and a numerically equal closed wavelength, then apply the
    strict complex-eigenvalue identity test. Distinct nonnormal poles are kept.
    """
    if len(rows)!=len(stored):raise ValueError("Closed rows and fields must match")
    retained=[];repeated=[]
    for index,row in enumerate(rows):
        peers=[i for i in retained if rows[i]["closed_TE_family"]==row["closed_TE_family"]
            and abs(rows[i]["wavelength_nm"]-row["wavelength_nm"])<=wavelength_tolerance_nm]
        value=complex(row["delta_per_m"],row["alpha_field_per_m"])
        values=[complex(rows[i]["delta_per_m"],rows[i]["alpha_field_per_m"]) for i in peers]
        if independent_eigenmode(value,stored[index][0].fields,values,
                                 [stored[i][0].fields for i in peers]):
            retained.append(index)
        else:
            repeated.append(dict(name=row["name"],compared_with=[rows[i]["name"] for i in peers]))
    return retained,dict(input_closed_modes=len(rows),independent_closed_modes=len(retained),
        repeated_records=repeated,wavelength_tolerance_nm=wavelength_tolerance_nm,
        eigenvalue_relative_tolerance=5e-9,eigenvalue_absolute_tolerance_per_m=1e-8,
        span_residual_tolerance=1e-6,
        scope="same TE family, same closed frequency, same numerical complex eigenvalue; span rank only")


def longitudinal_audit(lattice,design,period,dz=2.5):
    """Enumerate isolated TE families and audit the remote-reflector premise.

    A weakly bound isolated TE mode may couple back to the DBR through the
    2um spacer. Its scalar isolated-guide field cannot then certify a complete
    vertical spectrum. Such a case is recorded explicitly rather than silently
    labelled absent or validated by increasing only lateral resolution.
    """
    _,guide,_=structure(lattice,design)
    mirror=StratifiedStack(quarter_wave_dbr(1030.,2.1,1.45,16,first_layer="low"),1.8154,1.45)
    rows=[]
    for wavelength in (1000.,1010.,1030.,1050.,1060.):
        modes=solve_guided_te_modes(guide,wavelength,dz,32)
        for family,mode in enumerate(modes):
            k0=2*np.pi/(wavelength*1e-9);beta=k0*mode.effective_index
            kappa=np.sqrt(max(0.,beta**2-(1.45*k0)**2))
            load=solve_stack(mirror,wavelength,incident_side="top",q_parallel_per_m=beta)
            reflection_load=abs(load.r)*np.exp(-2*kappa*2000e-9)
            rows.append(dict(wavelength_nm=wavelength,family=family,effective_index=mode.effective_index,
                gain_confinement=mode.confinement["Yb:YAG gain"],
                estimated_family_Bragg_nm=(1. if lattice=="square" else np.sqrt(3)/2)*period*mode.effective_index,
                reflected_load_after_spacer=float(reflection_load),
                remote_reflector_small=bool(reflection_load<1e-3)))
    return dict(samples=rows,max_isolated_TE_count=1+max(r["family"] for r in rows),
        all_tested_families_remote_reflector_small=all(r["remote_reflector_small"] for r in rows),
        caveat="Near-cutoff families failing the remote-reflector premise require a guided-pole vertical solver; this audit is not an all-Maxwell longitudinal spectrum.")


def interpolate_envelope(lattice,field,old_spec,old_mesh,new_spec,new_mesh):
    """Interpolate physical center fields, retaining their channel convention."""
    from scipy.interpolate import RegularGridInterpolator,LinearNDInterpolator
    if lattice=="square":
        source=((np.arange(old_spec.grid_points)+.5)/old_spec.grid_points-.5)*300
        dest=((np.arange(new_spec.grid_points)+.5)/new_spec.grid_points-.5)*300
        yy,xx=np.meshgrid(dest,dest,indexing="ij")
        result=np.array([RegularGridInterpolator((source,source),f,
            bounds_error=False,fill_value=0.)(np.stack((yy,xx),axis=-1)) for f in field])
    else:
        source=np.column_stack((old_mesh.x_um,old_mesh.y_um))
        dest=np.column_stack((new_mesh.x_um,new_mesh.y_um))
        result=np.array([LinearNDInterpolator(source,f,fill_value=0.)(dest) for f in field])
    return result/np.linalg.norm(result)


def follow_reference_pole(problem,field,predictor):
    """Follow a supplied physical field on a new grid/operator, with residual QA."""
    vector=field.ravel()/np.linalg.norm(field);q=vector.copy()
    operator=problem.center_operator();z=complex(predictor)+complex(.5,.7)
    for iteration in range(16):
        try:
            links=splu((problem.stiffness-z*problem.mass).tocsc()).solve(q)
            new=problem.mass@links;new/=np.linalg.norm(new)
            applied=operator@new;value=complex(np.vdot(new,applied))
            residual=float(np.linalg.norm(applied-value*new)/max(np.linalg.norm(applied),abs(value),1e-30))
            overlap=float(abs(vector.conj()@new)**2)
            if residual<1e-11 and overlap>.3:
                return value,new.reshape(field.shape),dict(solver="physical_center_inverse_iteration",
                    iterations=iteration+1,eigen_residual=residual,interpolated_field_overlap=overlap)
            q=new;z=value+complex(1e-9,2e-9)
        except Exception:break
    vv,ff=problem.near(complex(predictor)+complex(.5,.7),16,vector,
        tol=1e-10,maxiter=500,dense_limit=0)
    overlaps=abs(vector.conj()@ff)**2;chosen=int(np.argmax(overlaps))
    value=complex(vv[chosen]);result=ff[:,chosen].reshape(field.shape)
    residual=relative_residual(operator,result,value)
    if residual>2e-7 or overlaps[chosen]<.3:raise RuntimeError("Source-to-fine pole continuation failed")
    return value,result,dict(solver="shift_invert_Arnoldi",eigen_residual=residual,
        interpolated_field_overlap=float(overlaps[chosen]))


def tune_actual_emission(lattice,design,period,order,dz,grid,count,extra_targets=1,initial_record=None):
    """Outer physical-period design, distinct from each fixed-period pole solve.

    Only this outer step is allowed to change the fabricated period. Every
    trial is independently closed first; no frozen Bragg wavelength is passed
    off as the predicted emission wavelength. The final period is then frozen
    for every competition mode and vertical family.
    """
    trace=[];previous=None
    if initial_record and "lowest" in initial_record:
        settings=initial_record["settings"]
        source_model,_,_=optical(lattice,design,settings["order"],settings["dz_nm"],
            initial_record.get("wavelength_nm",1030.),period=initial_record["period_nm"])
        source_spec,_,source_problem,source_mesh=finite(lattice,source_model,settings["grid"],count,False)
        target=complex(initial_record["lowest"]["delta_per_m"],initial_record["lowest"]["alpha_field_per_m"])
        indices=np.arange(source_problem.stiffness.shape[0]);initial=1+.19*np.cos(indices)+.17j*np.sin(3*indices)
        vv,ff=source_problem.near(target+complex(.5,.7),4,
            initial,tol=1e-10,maxiter=500,dense_limit=0)
        chosen=int(np.argmin(abs(vv-target)))
        shape=(4,source_spec.grid_points,source_spec.grid_points) if lattice=="square" else (6,source_mesh.points)
        previous=dict(model=source_model,spec=source_spec,mesh=source_mesh,
            value=complex(vv[chosen]),field=ff[:,chosen].reshape(shape))
    for iteration in range(6):
        model,_,guide=optical(lattice,design,order,dz,period=period)
        spec,modes,problem,mesh=finite(lattice,model,grid,count,False)
        if previous is None:
            pool=seed_pool(lattice,model,spec,problem,mesh,count,extra_targets)
            if not pool:raise RuntimeError("No passive pole for outer period design")
            seed=pool[0];initialization=dict(solver="bounded_Gamma_neighbour_census")
        else:
            field=interpolate_envelope(lattice,previous["field"],previous["spec"],previous["mesh"],spec,mesh)
            flat=field.reshape(len(model.result.coupling_m),-1)
            change=model.result.coupling_m-previous["model"].result.coupling_m
            predictor=previous["value"]+np.einsum("ik,ij,jk->",flat.conj(),change,flat)/np.sum(abs(flat)**2)
            value,field,initialization=follow_reference_pole(problem,field,predictor)
            seed=(value,field)
        wavelength,opt,g,closed_spec,mode,history=close_seed(lattice,design,period,model,seed,spec,mesh,order,dz)
        trace.append(dict(iteration=iteration,period_nm=period,lowest_selected_pole_nm=wavelength,
            alpha_field_per_m=mode.alpha_per_m,frequency_trace=history,initialization=initialization))
        print("outer_period",lattice,iteration,period,wavelength,flush=True)
        if abs(wavelength-1030.)<=.01:return period,trace
        previous=dict(model=opt,spec=closed_spec,mesh=mesh,
            value=complex(mode.delta_per_m,mode.alpha_per_m),field=mode.fields)
        period*=1030./wavelength
    raise RuntimeError("Actual-emission outer period design did not reach 1030 +/-0.01 nm")


def command_close(args):
    # Input is one compact accepted screen record; geometry and period fixed.
    record=json.loads(args.input.read_text(encoding="utf8"))
    if isinstance(record,list):record=record[args.index]
    lattice,design,period=record["lattice"],record["design"],record["period_nm"]
    design={k:v for k,v in design.items() if k not in
            ("lattice_nm","length_um","pc_nm","gaas_nm","qw_spacer_nm")}
    trial=getattr(args,"_pump_period_trial",0)
    if trial==0:args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/"plan.json",dict(source=str(args.input),record=record,order=args.order,dz_nm=args.dz,
        grid=args.square_grid if lattice=="square" else args.tri_grid,closed_per_TE=args.closed_roots))
    if getattr(args,"_period_override",None) is None:
        period,outer_trace=tune_actual_emission(lattice,design,period,args.order,args.dz,
            args.square_grid if lattice=="square" else args.tri_grid,args.roots,min(1,args.extra_targets),record)
    else:
        period=float(args._period_override);outer_trace=[dict(pump_aware_trial=trial,period_nm=period)]
    save(args.output/"outer_period_design.json",dict(target_nm=1030.,tolerance_nm=.01,
        frozen_period_nm=period,trace=outer_trace))
    final_fabrication=evaluate_fabrication(geometry(lattice,design).inclusions,period,
        None if lattice=="square" else TRIANGULAR_DIRECT,RULE)
    save(args.output/"final_fabrication.json",final_fabrication)
    export_actual_geometry(lattice,design,period,final_fabrication,args.output/"actual_geometry_nm.json")
    if not final_fabrication["margin_pass"]:raise RuntimeError("Actual-emission period violates fabrication constraints")
    vertical_audit=longitudinal_audit(lattice,design,period,min(args.dz,2.5))
    save(args.output/"longitudinal_applicability.json",vertical_audit)
    first,_,guide=optical(lattice,design,args.order,args.dz,period=period)
    vertical=solve_guided_te_modes(guide,1030.,args.dz,32)
    rows=[];models=[];names=[];arrays={};stored=[];failures=[];online_duplicates=[]
    for family in range(len(vertical)):
        model,_,guide=optical(lattice,design,args.order,args.dz,period=period,family=family)
        spec,modes,problem,mesh=finite(lattice,model,args.square_grid if lattice=="square" else args.tri_grid,args.roots,False)
        pool=seed_pool(lattice,model,spec,problem,mesh,args.roots,args.extra_targets,
            coarse_qz=True,audit_path=args.output/f"TE{family}_seed_identity_and_QZ_audit.json")
        target_audit=json.loads((args.output/f"TE{family}_seed_identity_and_QZ_audit.json").read_text(encoding="utf8"))
        save(args.output/f"TE{family}_frozen_census.json",dict(actual_complex_shift_targets=target_audit["planned_targets"],
            nominal_Gamma_neighbour_targets=len(model.result.eigenvalues_m)*(2*args.extra_targets+1),
            distinct_returned_poles=len(pool),poles=[dict(delta_per_m=v.real,alpha_field_per_m=v.imag) for v,_ in pool]))
        family_closed=0
        for rank,seed in enumerate(pool):
            if family_closed>=args.closed_roots:break
            name=f"TE{family}_pole{rank:02d}";start=time.monotonic()
            try:
                wavelength,opt,g,local_spec,mode,trace=close_seed(lattice,design,period,model,seed,spec,mesh,args.order,args.dz,family)
                mode=replace(mode,name=name)
                row,data=mode_row(lattice,opt,g,local_spec,mode,wavelength,201)
                axis,fx,fy,widths,qe,qg=data
                lateral=(lateral_energy_weights(mode.fields,31) if lattice=="square" else
                    binned_lateral_weights(mode.fields,mode.grid.x_um,300.,31))
                up,down=opt.side_power_losses(mode.fields)
                spatial=SpatialYbYAGModel(material(wavelength),widths,qe,2*mode.alpha_per_m+10.,up,down,
                    c/opt.result.group_index,qg,lateral)
                row.update(wavelength_nm=wavelength,frequency_trace=trace,closed_TE_family=family,
                    frequency_residual_per_m=trace[-1]["frequency_residual_per_m"],seconds=time.monotonic()-start)
                peers=[i for i,old in enumerate(rows) if old["closed_TE_family"]==family
                       and abs(old["wavelength_nm"]-wavelength)<=1e-7]
                if not independent_eigenmode(complex(mode.delta_per_m,mode.alpha_per_m),mode.fields,
                    [complex(rows[i]["delta_per_m"],rows[i]["alpha_field_per_m"]) for i in peers],
                    [stored[i][0].fields for i in peers]):
                    online_duplicates.append(dict(seed_name=name,wavelength_nm=wavelength,
                        compared_with=[rows[i]["name"] for i in peers],
                        reason="same closed wavelength/complex eigenvalue and no new span dimension; does not consume independent-mode quota"))
                    save(args.output/"online_closed_identity_rejections.json",online_duplicates)
                    continue
                rows.append(row);models.append(spatial);names.append(name);stored.append((mode,axis,fx,fy))
                family_closed+=1
                arrays.update({name+"_field":mode.fields,name+"_far_x":fx,name+"_far_y":fy,
                    name+"_radiation_x":mode.radiation_x,name+"_radiation_y":mode.radiation_y,
                    name+"_qgain":qg,name+"_qenergy":qe,name+"_lateral":lateral,
                    name+"_C":opt.result.coupling_m,name+"_Lup":opt.upward_amplitude_loss_m,
                    name+"_Ldown":opt.downward_amplitude_loss_m,name+"_widths":widths})
                if lattice=="triangular":arrays.update(x_um=mode.grid.x_um,y_um=mode.grid.y_um)
                print("closed",lattice,name,wavelength,row["uniform_edge_proxy_W"],row["beam"]["center_to_peak"],flush=True)
            except Exception as error:
                failures.append(dict(name=name,error=repr(error)));print("failed",name,repr(error),flush=True)
            save(args.output/"closed_optical_modes.json",rows);save(args.output/"closure_failures.json",failures)
    if not rows:raise RuntimeError("No roots closed")
    retained,identity_audit=distinct_closed_modes(rows,stored)
    identity_audit["online_repeated_seed_records"]=online_duplicates
    identity_audit["independent_quota_per_TE"]=args.closed_roots
    save(args.output/"closed_mode_identity_audit.json",identity_audit)
    if len(retained)!=len(rows):
        discarded={row["name"] for i,row in enumerate(rows) if i not in retained}
        arrays={key:value for key,value in arrays.items()
                if not any(key.startswith(name+"_") for name in discarded)}
        rows=[rows[i] for i in retained];models=[models[i] for i in retained]
        names=[names[i] for i in retained];stored=[stored[i] for i in retained]
        save(args.output/"closed_optical_modes.json",rows)
    arrays["angle_deg"]=axis
    checkpoint={k:v for k,v in arrays.items() if not (k.endswith("_far_x") or k.endswith("_far_y"))}
    np.savez_compressed(args.output/"closed_optical_checkpoint.npz",**checkpoint)
    system=SpatialMultimodeYbYAG(tuple(models),tuple(names));pump_results=[]
    # Pump numbers are total internal first-entry power; CW is stationary only.
    # 泵浦 W 数为增益区内部首入射总功率；后续 CW 扫描仅验证静态共享反转解。
    for architecture in ("edge_dual","top_single"):
        pump=PumpGeometryConfig(architecture,0.,300.,31)
        threshold=shared_inversion_pump_threshold(system,pump)
        scans=[]
        if threshold.first_threshold_internal_W is not None:
            for ratio in (1.05,1.25,1.5,2.,3.,4.):
                power=ratio*threshold.first_threshold_internal_W
                try:
                    result=solve_multimode_spatial_ybyag(system,replace(pump,internal_incident_power_W=power))
                    scans.append(dict(status="completed",ratio=ratio,power_internal_W=power,active_modes=result.active_modes,
                        upward_W=dict(zip(names,map(float,result.upward_output_W))),energy_budget_W=result.energy_budget_W,
                        population_residual_per_s=result.population_residual_per_s,ncp_residual=result.complementarity_residual,
                        maximum_inactive_gain_excess_per_m=result.maximum_inactive_gain_excess_per_m))
                except Exception as error:
                    scans.append(dict(status="failed",ratio=ratio,power_internal_W=power,error=repr(error)))
                save(args.output/"pump_results.json",pump_results+[dict(architecture=architecture,threshold=asdict(threshold),cw_scan_in_progress=scans)])
        pump_results.append(dict(architecture=architecture,threshold=asdict(threshold),cw=scans))
        save(args.output/"pump_results.json",pump_results)
    # The objective is the actual first shared-inversion edge-dual threshold,
    # including every supported TE family and supplied lateral competitor.
    # A different overlap or pump profile can reorder cold-loss modes. If that
    # happens, retarget that actual winner and repeat the complete competition;
    # stale trial fields are never saved as the final solution.
    # 正式赢家按共享侧泵实际首阈值选择；若其波长偏离目标，重新设计周期并全程复算。
    first=pump_results[0]["threshold"]["first_modes"]
    if first:
        winner=next(r for r in rows if r["name"]==first[0])
        if abs(winner["wavelength_nm"]-1030.)>.05:
            retarget=dict(trial=trial,period_nm=period,first_modes=first,
                actual_first_wavelength_nm=winner["wavelength_nm"],
                edge_threshold_internal_W=pump_results[0]["threshold"]["first_threshold_internal_W"],
                next_period_nm=period*1030./winner["wavelength_nm"])
            save(args.output/f"pump_emission_retarget_{trial:02d}.json",retarget)
            if trial>=5:raise RuntimeError("Actual shared-inversion winner could not be designed to 1030 +/-0.05nm")
            args._pump_period_trial=trial+1;args._period_override=retarget["next_period_nm"]
            return command_close(args)
    winners={}
    for pump_result in pump_results:
        arch=pump_result["architecture"];first_names=pump_result["threshold"]["first_modes"]
        winners[arch]=[next(r for r in rows if r["name"]==name) for name in first_names]
    far_names={r["name"] for rows_for_arch in winners.values() for r in rows_for_arch}
    arrays={k:v for k,v in arrays.items() if not
        ((k.endswith("_far_x") or k.endswith("_far_y")) and k.rsplit("_far_",1)[0] not in far_names)}
    np.savez_compressed(args.output/"closed_fields.npz",**arrays)
    save(args.output/"actual_pump_winners.json",winners)
    summary=dict(lattice=lattice,design=design,period_nm=period,side_um=300.,isolated_TE_count=len(vertical),
        included_closed_modes=len(rows),failed_roots=failures,settings=dict(order=args.order,dz_nm=args.dz,
        square_grid=args.square_grid,tri_radius_cells=args.tri_grid,extra_targets=args.extra_targets),pump_results=pump_results,
        assumptions="same 500nm Yb:YAG /120nm air PC /2um SiO2 /16-pair DBR; nominal 10at% constant cross sections",
        outer_period_design=outer_trace,fabrication=final_fabrication,pump_aware_period_trials=trial+1,
        longitudinal_applicability=vertical_audit,
        closed_mode_identity_audit=identity_audit,
        actual_winners=winners,recommended_pump_architecture="edge_dual",
        common_analysis_band_nm=[1000,1060],
        internal_intensity_loss_cm1=.1,nominal_material_pump_limited_gain_ceiling_cm1=23.93265,
        fields_file="closed_fields.npz",axis_key="angle_deg",actual_geometry_file="actual_geometry_nm.json",
        stored_far_field_modes=sorted(far_names),storage="All independent closed envelopes/radiation sheets retained; dense angular Jones maps retained for actual edge/top first modes; other scalar beam/Stokes records retained",
        scope="outer period designed at actual low-loss emission near 1030nm, then fixed-period real-frequency scalar TE CWT poles; bounded sparse lateral census; shared x-z inversion with y averaged; no TM, full-wave certification, thermal model or dynamic single-mode proof")
    save(args.output/"summary.json",summary)
    first_name=pump_results[0]["threshold"]["first_modes"]
    if first_name:
        selected=names.index(first_name[0]);mode,axis,fx,fy=stored[selected]
        power=abs(fx)**2+abs(fy)**2;power/=power.max();maps=stokes_maps(fx,fy)
        fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
        axes[0].imshow(power,origin="lower",extent=(axis[0],axis[-1],axis[0],axis[-1]),cmap="inferno")
        axes[0].set(title=f"{lattice}: {first_name[0]}",xlabel="theta x (deg)",ylabel="theta y (deg)",xlim=(-.5,.5),ylim=(-.5,.5))
        for ax,key in zip(axes[1:],("s1_over_s0","s3_over_s0")):
            masked=np.ma.array(maps[key],mask=~maps["valid_mask"])
            im=ax.imshow(masked,origin="lower",extent=(axis[0],axis[-1],axis[0],axis[-1]),cmap="coolwarm",vmin=-1,vmax=1)
            ax.set(title=key,xlabel="theta x (deg)",ylabel="theta y (deg)",xlim=(-.5,.5),ylim=(-.5,.5));fig.colorbar(im,ax=ax)
        fig.savefig(args.output/"selected_far_field_polarization.png",dpi=180);plt.close(fig)
        if lattice=="triangular":plot_triangular_lattice(geometry(lattice,design),args.output/"selected_lattice.png",5)
    print("completed",lattice,len(rows),[(p["architecture"],p["threshold"]["first_threshold_internal_W"]) for p in pump_results],flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("benchmark","scan","refine","close"))
    p.add_argument("--output",type=Path,default=OUT)
    p.add_argument("--lattices",nargs="+",default=["square","triangular"])
    p.add_argument("--count",type=int,default=160);p.add_argument("--order",type=int,default=4)
    p.add_argument("--dz",type=float,default=5.)
    p.add_argument("--square-grid",type=int,default=15);p.add_argument("--tri-grid",type=int,default=9)
    p.add_argument("--roots",type=int,default=12)
    p.add_argument("--minimum-fill",type=float,default=.055)
    p.add_argument("--closed-roots",type=int,default=16);p.add_argument("--extra-targets",type=int,default=2)
    p.add_argument("--input",type=Path);p.add_argument("--index",type=int,default=0)
    p.add_argument("--period",type=float,dest="_period_override",default=None,
        help="Explicit outer-design starting physical period; all poles still close at this fixed period and actual pump winner is checked")
    a=p.parse_args(argv)
    {"benchmark":command_benchmark,"scan":command_scan,"refine":command_refine,"close":command_close}[a.command](a)


if __name__=="__main__":main()
