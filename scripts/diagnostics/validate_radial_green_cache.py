"""Reproduce before/after radial-Green cache equality and wall-clock timing.

The archived modules are imported under a separate pcselsim subpackage, so
their relative dependencies use the archived three-/six-wave implementations.
Only compact metrics and complete input/source hashes are saved, not matrices.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np

MODULES=("three_d_cwt","triangular_six_wave","reflector_cwt","matched300_yag")
DEFAULT_OUTPUT=ROOT/"results/current/matched300_20261021/kernel_cache_validation"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,(np.integer,np.floating,np.bool_)):return value.item()
    if isinstance(value,complex):return {"real":value.real,"imag":value.imag}
    return value


def load_archived_modules(directory):
    """Isolate archived relative imports from the live pcselsim modules."""
    import pcselsim
    package_name="pcselsim._radial_cache_before"
    package=types.ModuleType(package_name)
    package.__package__=package_name
    package.__path__=[str(ROOT/"src/pcselsim")]
    sys.modules[package_name]=package
    setattr(pcselsim,"_radial_cache_before",package)
    result={}
    # Their imports resolve .three_d_cwt and .triangular_six_wave to these
    # already loaded archived versions, not to the optimized live versions.
    for name in MODULES:
        full_name=package_name+"."+name
        spec=importlib.util.spec_from_file_location(full_name,Path(directory)/(name+".py"))
        module=importlib.util.module_from_spec(spec)
        sys.modules[full_name]=module
        spec.loader.exec_module(module)
        setattr(package,name,module)
        result[name]=module
    return result


def stack_input(stack):
    keys=("top_index","bottom_index","padding_um","te_family")
    return dict(layers=[asdict(layer) for layer in stack.layers],
                **{key:getattr(stack,key) for key in keys if hasattr(stack,key)})


def validation_cases(order=8,dz=4.):
    from pcselsim.vertical import Layer,LayerStack
    from pcselsim.three_d_cwt import ThreeDCWTSettings
    from pcselsim.triangular_six_wave import TriangularLatticeCell,TriangularCWTSettings
    from scripts.optimization.matched300_square_search import Design,build_cell
    from scripts.optimization.square_semiconductor_search import BASE,build_stack
    from scripts.optimization.matched300_triangular_search import ThreeHole,candidate,base_spec
    from scripts.optimization.triangular_semiconductor_search import parameters
    from scripts.optimization.matched300_yag_search import structure,PC
    square=Design(fill=.145,main_share=.63,distance=.535,position_angle_deg=43.,
        main_aspect=1.17,satellite_aspect=1.28,main_angle_deg=21.,satellite_angle_deg=-37.,
        main_shape="ellipse",satellite_shape="rounded_triangle",lattice_nm=278.3)
    tri=ThreeHole("cache_validation",fill=.147,main_share=.58,satellite_balance=.43,
        main_shape="ellipse",satellite1_shape="triangle",satellite2_shape="ellipse",
        main_aspect=1.19,satellite2_aspect=1.31,distance1=.37,distance2=.43,
        angle1_deg=-26.,angle2_deg=34.,main_orientation_deg=17.,
        satellite1_orientation_deg=31.,satellite2_orientation_deg=-23.,lattice_nm=323.7)
    cells={"square":build_cell(square,96),
        "triangular":TriangularLatticeCell(BASE.BACKGROUND_INDEX**2,candidate(tri).inclusions)}
    p=parameters(base_spec(tri));average=np.sqrt(cells["triangular"].fourier_epsilon(0,0).real)
    tri_stack=LayerStack(tuple(Layer(l.name,l.thickness_nm,average if l.name==p.PC_LAYER_NAME
        else l.refractive_index) for l in p.LAYERS),p.TOP_CLADDING_INDEX,
        p.BOTTOM_CLADDING_INDEX,p.VERTICAL_PADDING_UM)
    cases={}
    for lattice,cell,guide,period in (
        ("square",cells["square"],build_stack(square,cells["square"]),square.lattice_nm),
        ("triangular",cells["triangular"],tri_stack,tri.lattice_nm)):
        module="three_d_cwt" if lattice=="square" else "triangular_six_wave"
        function="build_geometry_coupling" if lattice=="square" else "build_triangular_coupling"
        settings=ThreeDCWTSettings(order,dz) if lattice=="square" else TriangularCWTSettings(order,dz)
        cases["sem_"+lattice]=dict(module=module,function=function,
            args=(cell,guide,BASE.PC_LAYER_NAME,period,950.,settings),wavelength_nm=950.,
            inputs=dict(lattice=lattice,cell=asdict(cell),guided_stack=stack_input(guide),
                pc_layer=BASE.PC_LAYER_NAME,period_nm=period,wavelength_nm=950.,settings=asdict(settings)))
    for lattice,design in (("square",asdict(square)),("triangular",asdict(tri))):
        cell,guide,radiation=structure(lattice,design)
        period=604.1 if lattice=="square" else 697.2
        module="reflector_cwt" if lattice=="square" else "matched300_yag"
        function="build_reflector_coupling" if lattice=="square" else "build_triangular_reflector_coupling"
        settings=ThreeDCWTSettings(order,dz) if lattice=="square" else TriangularCWTSettings(order,dz)
        cases["yag_"+lattice]=dict(module=module,function=function,
            args=(cell,guide,radiation,PC,period,1030.,settings),wavelength_nm=1030.,
            inputs=dict(lattice=lattice,cell=asdict(cell),guided_stack=stack_input(guide),
                radiation_stack=stack_input(radiation),pc_layer=PC,period_nm=period,
                wavelength_nm=1030.,settings=asdict(settings)))
    return cases


def compare_numeric(before,after):
    left,right=np.asarray(before),np.asarray(after)
    if left.shape!=right.shape:return dict(pass_tolerance=False,shape_before=list(left.shape),shape_after=list(right.shape))
    difference=float(np.max(abs(left-right))) if left.size else 0.
    scale=max(float(np.max(abs(left))) if left.size else 0.,
              float(np.max(abs(right))) if right.size else 0.,np.finfo(float).tiny)
    relative=difference/scale
    return dict(bitwise_equal=bool(left.dtype==right.dtype and left.tobytes()==right.tobytes()),
        maximum_absolute_error=difference,relative_linf_error=relative,
        pass_tolerance=bool(relative<=1e-13))


def compare_results(before,after):
    b=getattr(before,"result",before);a=getattr(after,"result",after);items={}
    keys=("coupling_m","c1d_m","crad_m","c2d_m","cb_m","cr_m","ch_m", "eigenvalues_m",
          "eigenvectors","radiation_surface_factor","average_pc_epsilon","pc_confinement",
          "effective_index","group_index","bragg_wavelength_nm","passivity_correction_m")
    for key in keys:
        if hasattr(b,key) or hasattr(a,key):items[key]=compare_numeric(getattr(b,key),getattr(a,key))
    for key in ("z_um","field","index","effective_index"):
        items["vertical_mode."+key]=compare_numeric(getattr(b.vertical_mode,key),getattr(a.vertical_mode,key))
    for key in ("unit_cell_response_x","unit_cell_response_y"):
        old,new=getattr(b,key),getattr(a,key)
        if old.keys()!=new.keys():raise AssertionError("Changed response orders: "+key)
        for order in old:items[f"{key}[{order[0]},{order[1]}]"]=compare_numeric(old[order],new[order])
    for key in ("upward_amplitude_loss_m","downward_amplitude_loss_m","downward_surface_factor",
                "optical_theorem_residual"):
        if hasattr(before,key):items[key]=compare_numeric(getattr(before,key),getattr(after,key))
    rng=np.random.default_rng(481030)
    fields=rng.normal(size=(len(b.eigenvalues_m),7))+1j*rng.normal(size=(len(b.eigenvalues_m),7))
    function="radiation_fields" if hasattr(b,"radiation_fields") else "radiation_amplitudes"
    for index,(old,new) in enumerate(zip(getattr(b,function)(fields),getattr(a,function)(fields),strict=True)):
        items[f"outgoing_Jones_{index}"]=compare_numeric(old,new)
    if hasattr(before,"side_power_losses"):
        items["side_power_losses"]=compare_numeric(before.side_power_losses(fields),after.side_power_losses(fields))
    groups={}
    for name,metrics in items.items():
        key=name.split("[")[0]
        group=groups.setdefault(key,dict(numeric_items=0,all_bitwise_equal=True,
            maximum_absolute_error=0.,maximum_relative_linf_error=0.,all_pass_1e13=True))
        group["numeric_items"]+=1
        group["all_bitwise_equal"]&=metrics.get("bitwise_equal",False)
        group["all_pass_1e13"]&=metrics["pass_tolerance"]
        group["maximum_absolute_error"]=max(group["maximum_absolute_error"],metrics.get("maximum_absolute_error",float("inf")))
        group["maximum_relative_linf_error"]=max(group["maximum_relative_linf_error"],metrics.get("relative_linf_error",float("inf")))
    return dict(compared_numeric_items=len(items),all_bitwise_equal=all(v.get("bitwise_equal",False) for v in items.values()),
        all_pass_1e13=all(v["pass_tolerance"] for v in items.values()),
        maximum_relative_linf_error=max(v.get("relative_linf_error",float("inf")) for v in items.values()),
        item_groups=groups,nonbitwise_items={k:v for k,v in items.items() if not v.get("bitwise_equal",False)},
        itemwise_comparison_sha256=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        retention="Every item is compared; repetitive exact-zero records are summarized by group and hashed. All nonbitwise records are retained.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before",type=Path,default=DEFAULT_OUTPUT/"source_before")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"comparison.json")
    parser.add_argument("--order",type=int,default=8);parser.add_argument("--dz",type=float,default=4.)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--cases",nargs="+",choices=("sem_square","sem_triangular","yag_square","yag_triangular"))
    args=parser.parse_args(argv)
    if args.repeats<1:raise ValueError("At least one timing repeat is required")
    archived=load_archived_modules(args.before)
    live={name:importlib.import_module("pcselsim."+name) for name in MODULES}
    summary=dict(python=platform.python_version(),numpy=np.__version__,order=args.order,dz_nm=args.dz,
        repeats=args.repeats,script_sha256=sha256(__file__),source_sha256={name:dict(
            before=sha256(args.before/(name+".py")),after=sha256(ROOT/"src/pcselsim"/(name+".py"))) for name in MODULES},
        numerical_tolerance_relative=1e-13,comparison_scope="Identical inputs; all coupling terms, eigenpairs, response orders, outgoing Jones and side fluxes; no finite-mode or physical-model alteration",cases={})
    for name,case in validation_cases(args.order,args.dz).items():
        if args.cases and name not in args.cases:continue
        timings={"before":[],"after":[]};outputs={}
        for repeat in range(args.repeats):
            # Alternate timing order to reduce systematic warm-up bias.
            for label in (("before","after") if repeat%2==0 else ("after","before")):
                modules=archived if label=="before" else live
                start=time.perf_counter()
                output=getattr(modules[case["module"]],case["function"])(*case["args"],
                    fixed_wavelength_nm=case["wavelength_nm"])
                timings[label].append(time.perf_counter()-start);outputs[label]=output
        comparison=compare_results(outputs["before"],outputs["after"])
        inputs=serial(case["inputs"]);encoded=json.dumps(inputs,sort_keys=True,separators=(",",":")).encode()
        comparison.update(inputs=inputs,input_sha256=hashlib.sha256(encoded).hexdigest(),
            wall_seconds=timings,median_seconds={k:float(np.median(v)) for k,v in timings.items()},
            median_speedup=float(np.median(timings["before"])/np.median(timings["after"])))
        summary["cases"][name]=comparison
        summary["all_pass_1e13"]=all(v["all_pass_1e13"] for v in summary["cases"].values())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(serial(summary),indent=2,allow_nan=False),encoding="utf-8")
        print(name,"bitwise",comparison["all_bitwise_equal"],"relative error",comparison["maximum_relative_linf_error"],
              "speedup",comparison["median_speedup"],flush=True)
        if not comparison["all_pass_1e13"]:raise AssertionError("Radial cache changed numerical output: "+name)
    return summary


if __name__=="__main__":main()
