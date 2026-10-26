"""Compare archived scalar-edge Fourier integration to the vectorized formula.

The two functions are selected in a scoped context for both square and
triangular cells. Coupling builders, radial caches and physical inputs remain
the same. Results retain actual nonzero roundoff bounds, not large raw arrays.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_key]="1"
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
import numpy as np
from scripts.diagnostics.validate_radial_green_cache import (
    DEFAULT_OUTPUT,MODULES,compare_results,serial,sha256,validation_cases)


def load_original(path):
    name="pcselsim._polygon_scalar_before"
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module.polygon_fourier_integral


@contextmanager
def select_polygon_integral(function):
    # Triangular cells import the function directly; square cells resolve it
    # in their geometry module. Restore both bindings even after an exception.
    import pcselsim.geometry as geometry
    import pcselsim.triangular_six_wave as triangular
    bindings=(geometry.polygon_fourier_integral,triangular.polygon_fourier_integral)
    geometry.polygon_fourier_integral=triangular.polygon_fourier_integral=function
    try:yield
    finally:geometry.polygon_fourier_integral,triangular.polygon_fourier_integral=bindings


def contour_validation(before,after):
    from scripts.optimization.matched300_square_search import hole
    from pcselsim.triangular_six_wave import TriangularPolygon
    rectangle=np.array([[-.19,-.13],[.19,-.13],[.19,.13],[-.19,.13]])
    theta=np.deg2rad(1e-12);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    contours=dict(rectangle=rectangle,almost_parallel_rectangle=rectangle@rotation.T+(.173,-.217),
        dense_superellipse=np.array(hole((.127,-.219),.075,"superellipse",1.37,31.,3.7,96).vertices),
        dense_rounded_triangle=np.array(hole((.127,-.219),.075,"rounded_triangle",1.37,31.,3.7,96).vertices),
        triangular_fractional_rounded=np.array(TriangularPolygon.rounded_regular((.127,-.219),.21,
            angle_deg=31.,corner_fraction=.3,samples_per_corner=32).vertices_fractional))
    grid=(-173,-48,-13,-1,0,3,17,48,211)
    orders=[(m,n) for m in grid for n in grid]
    result={}
    for name,vertices in contours.items():
        old=[];new=[]
        for contour in (vertices,vertices[::-1],vertices+(.193,-.219)):
            for m,n in orders:
                points=tuple(map(tuple,contour))
                old.append(before(points,m,n));new.append(after(points,m,n))
        old,new=np.asarray(old),np.asarray(new);error=abs(old-new)
        area=abs(before(tuple(map(tuple,vertices)),0,0));resolved=abs(old)>area*1e-10
        result[name]=dict(comparisons=len(old),vertices=len(vertices),area=area,
            maximum_absolute_error=float(error.max()),
            maximum_relative_to_area_error=float(error.max()/area),
            maximum_relative_error_above_1e10_area=float(np.max(error[resolved]/abs(old[resolved]))),
            cancellation_floor_absolute=area*1e-10,
            below_floor_comparisons=int((~resolved).sum()),
            below_floor_maximum_absolute_error=float(error[~resolved].max()) if (~resolved).any() else 0.,
            pass_1e12=bool(error.max()<=area*1e-12))
        if not result[name]["pass_1e12"]:raise AssertionError("Contour mismatch: "+name)
    return result


def eigenvector_comparison(before,after):
    before=getattr(before,"result",before);after=getattr(after,"result",after)
    vectors=after.eigenvectors.copy()
    for i in range(vectors.shape[1]):
        product=np.vdot(before.eigenvectors[:,i],vectors[:,i])
        if abs(product)>0:vectors[:,i]*=np.exp(-1j*np.angle(product))
    separation=abs(before.eigenvalues_m[:,None]-before.eigenvalues_m[None,:])
    np.fill_diagonal(separation,np.inf)
    return dict(minimum_distinct_complex_eigenvalue_separation_per_m=float(separation.min()),
        phase_aligned_vector_relative_linf_error=float(np.max(abs(before.eigenvectors-vectors))/np.max(abs(before.eigenvectors))))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before",type=Path,default=DEFAULT_OUTPUT/"source_before/geometry.py")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT/"polygon_vectorization.json")
    parser.add_argument("--order",type=int,default=16);parser.add_argument("--dz",type=float,default=2.)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--cases",nargs="+",choices=("sem_square","sem_triangular","yag_square","yag_triangular"))
    args=parser.parse_args(argv)
    import pcselsim.geometry as geometry
    before=load_original(args.before);after=geometry.polygon_fourier_integral
    modules={name:importlib.import_module("pcselsim."+name) for name in MODULES}
    result=dict(script_sha256=sha256(__file__),geometry_sha256=dict(before=sha256(args.before),
        after=sha256(ROOT/"src/pcselsim/geometry.py")),
        builder_source_sha256={name:sha256(ROOT/"src/pcselsim"/(name+".py")) for name in MODULES},
        order=args.order,dz_nm=args.dz,repeats=args.repeats,contours=contour_validation(before,after),cases={},
        scope="Identical geometry and cached CWT builders; only scalar versus vectorized polygon-edge evaluation changes. Nonzero roundoff is retained. Timing is host/load dependent.")
    for name,case in validation_cases(args.order,args.dz).items():
        if args.cases and name not in args.cases:continue
        # Optimizer cells may memoize Fourier coefficients per geometry. Such
        # a cache must not hide the intended before/after formula change.
        if name.endswith("square"):
            cell=case["args"][0]
            case["args"]=(geometry.SquareLatticeCell(cell.background_epsilon,cell.inclusions),*case["args"][1:])
        timings={"before":[],"after":[]};outputs={}
        for repeat in range(args.repeats):
            for label in (("before","after") if repeat%2==0 else ("after","before")):
                with select_polygon_integral(before if label=="before" else after):
                    start=time.perf_counter()
                    output=getattr(modules[case["module"]],case["function"])(*case["args"],
                        fixed_wavelength_nm=case["wavelength_nm"])
                    timings[label].append(time.perf_counter()-start);outputs[label]=output
        comparison=compare_results(outputs["before"],outputs["after"])
        comparison.pop("nonbitwise_items") # all actual nonzero bounds retained by field group
        comparison["eigenvectors_phase_aligned"]=eigenvector_comparison(outputs["before"],outputs["after"])
        # Diagnostics whose exact analytic value is zero need an absolute or
        # physically scaled error; a relative error against roundoff is not a
        # meaningful tolerance. Keep their raw differences in item_groups.
        diagnostics={"passivity_correction_m","optical_theorem_residual","eigenvectors"}
        model=getattr(outputs["before"],"result",outputs["before"])
        corrections=comparison["item_groups"].get("passivity_correction_m",{})
        correction_relative_to_c=corrections.get("maximum_absolute_error",0.)/np.max(abs(model.coupling_m))
        theorem_absolute=comparison["item_groups"].get("optical_theorem_residual",{}).get("maximum_absolute_error",0.)
        comparison["near_zero_diagnostic_errors"]=dict(
            passivity_correction_difference_relative_to_coupling=float(correction_relative_to_c),
            optical_theorem_residual_absolute_difference=theorem_absolute)
        comparison["maximum_physical_relative_linf_error"]=max(v["maximum_relative_linf_error"]
            for k,v in comparison["item_groups"].items() if k not in diagnostics)
        comparison["all_pass_1e12"]=comparison["maximum_physical_relative_linf_error"]<=1e-12 and (
            comparison["eigenvectors_phase_aligned"]["phase_aligned_vector_relative_linf_error"]<=1e-12)
        comparison["all_pass_1e12"] &= correction_relative_to_c<=1e-12 and theorem_absolute<=1e-12
        inputs=serial(case["inputs"])
        comparison.update(inputs=inputs,input_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            wall_seconds=timings,median_seconds={k:float(np.median(v)) for k,v in timings.items()},
            median_speedup=float(np.median(timings["before"])/np.median(timings["after"])),
            retention="Every response/order compared; group maxima preserve the nonzero error bounds and complete itemwise metrics are hashed.")
        result["cases"][name]=comparison
        result["all_pass_1e12"]=all(c["all_pass_1e12"] for c in result["cases"].values())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(serial(result),indent=2,allow_nan=False),encoding="utf-8")
        print(name,"physical_relative_error",comparison["maximum_physical_relative_linf_error"],
              "phase_aligned_field_error",comparison["eigenvectors_phase_aligned"]["phase_aligned_vector_relative_linf_error"],
              "speedup",comparison["median_speedup"],"pass",comparison["all_pass_1e12"],flush=True)
        if not comparison["all_pass_1e12"]:raise AssertionError("Polygon vectorization changed result beyond 1e-12: "+name)
    return result


if __name__=="__main__":main()
