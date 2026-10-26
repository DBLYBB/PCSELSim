"""Local radial kernels are reused per metric, never across input calls."""
from dataclasses import replace
import importlib

import numpy as np
import pytest

from scripts.diagnostics.validate_radial_green_cache import validation_cases


def metric_count(lattice,order):
    if lattice=="square":
        return len({m*m+n*n for m in range(-order,order+1) for n in range(-order,order+1) if m*m+n*n>1})
    extent=int(np.ceil(2*order/np.sqrt(3)))
    return len({m*m-m*n+n*n for m in range(-extent,extent+1) for n in range(-extent,extent+1)
        if 1<m*m-m*n+n*n<=order**2})


@pytest.mark.parametrize("lattice",("square","triangular"))
def test_free_host_builds_each_radial_kernel_once_and_keeps_cache_call_local(monkeypatch,lattice):
    case=validation_cases(3,10.)["sem_"+lattice]
    module=importlib.import_module("pcselsim."+case["module"])
    original=module._double_green_integral;calls=[]
    def counted(*args,**kwargs):
        if not kwargs.get("radiative",False):calls.append(complex(args[3]))
        return original(*args,**kwargs)
    monkeypatch.setattr(module,"_double_green_integral",counted)
    results=[];counts=[]
    args=list(case["args"])
    for wavelength,changed_stack in ((950.,False),(951.,False),(950.,True)):
        local=list(args)
        if changed_stack:
            layers=list(args[1].layers);layers[0]=replace(layers[0],thickness_nm=layers[0].thickness_nm+13.)
            local[1]=replace(args[1],layers=tuple(layers))
        start=len(calls)
        results.append(getattr(module,case["function"])(*local,fixed_wavelength_nm=wavelength))
        counts.append(calls[start:])
    expected=metric_count(lattice,3)
    assert all(len(group)==expected==len(set(group)) for group in counts)
    assert counts[0]!=counts[1]
    assert not np.array_equal(results[0].coupling_m,results[1].coupling_m)
    assert not np.array_equal(results[0].coupling_m,results[2].coupling_m)


@pytest.mark.parametrize("lattice",("square","triangular"))
def test_layered_kernel_is_once_per_metric_and_new_frequency_or_stack_rebuilds(monkeypatch,lattice):
    case=validation_cases(3,10.)["yag_"+lattice]
    module=importlib.import_module("pcselsim."+case["module"])
    original=module.local_green_kernel;calls=[]
    def counted(stack,host,wavelength,**kwargs):
        calls.append((id(stack),float(wavelength),kwargs.get("q_parallel_per_m",0.)))
        return original(stack,host,wavelength,**kwargs)
    monkeypatch.setattr(module,"local_green_kernel",counted)
    args=list(case["args"]);results=[];groups=[]
    for wavelength,changed_stack in ((1030.,False),(1031.,False),(1030.,True)):
        local=list(args)
        if changed_stack:
            layers=list(args[2].layers)
            index=next(i for i,l in enumerate(layers) if l.name=="lower SiO2 guide spacer")
            layers[index]=replace(layers[index],thickness_nm=layers[index].thickness_nm+13.)
            local[2]=replace(args[2],layers=tuple(layers))
        start=len(calls)
        results.append(getattr(module,case["function"])(*local,fixed_wavelength_nm=wavelength))
        groups.append(calls[start:])
    expected=1+metric_count(lattice,3) # normal-radiation kernel plus evanescent shells
    assert all(len(group)==expected and len({v[2] for v in group})==expected for group in groups)
    assert {v[1] for v in groups[0]}=={1030.}
    assert {v[1] for v in groups[1]}=={1031.}
    assert groups[0][0][0]!=groups[2][0][0]
    assert not np.array_equal(results[0].result.coupling_m,results[1].result.coupling_m)
    assert not np.array_equal(results[0].result.coupling_m,results[2].result.coupling_m)
