"""Independent small algorithm/physics checks; no candidate optimisation.

中文：独立连续边值基准、离散能流恒等式和拒绝非物理解反例。自动静态扫描
只定位风险语句，不把扫描通过或测试通过误称为全部物理公式已获证明。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys

for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.optimize import root

from pcselsim.config import NumericsConfig, SimulationConfig
from pcselsim.constants import c, epsilon_0, hbar, pi
from pcselsim.custom_analysis import _upwind_1d
from pcselsim.finite_discretization import build_finite_eigenproblem, relative_residual
from pcselsim.frequency_cwt import background_detuning
from pcselsim.materials import active_gain_m, modal_gain_m
from pcselsim.mode_identity import independent_mode_indices
from pcselsim.observables import photon_density_m3, radiated_power_W
from pcselsim.solver import TimeDomainSolver
from pcselsim.triangular_finite import (
    _cartesian_point_fourier_sum, _directed_upwind, build_device_grid,
)


def static_screen():
    patterns = {
        "clipping_or_floor": r"\b(?:clip|maximum|minimum|nan_to_num)\s*\(",
        "eigenvalue_fit_or_filter": r"polyfit|extrapolat|passive_penalty|roughness|overlap",
        "analytic_or_proxy": r"analytic_mode_fields|radiation_proxy_field|target_flatness|calibrated",
        "exceptions_or_fallback": r"except\b|fallback|ArpackNoConvergence",
        "interpolation": r"Interpolat|interp\w*\(",
    }
    inventory = []
    for folder in ("src", "scripts"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            source = path.read_text(encoding="utf-8-sig")
            ast.parse(source, filename=str(path))
            hits = []
            for number, line in enumerate(source.splitlines(), 1):
                flags = [name for name, pattern in patterns.items() if re.search(pattern, line)]
                if flags:
                    hits.append({"line": number, "flags": flags, "text": line.strip()[:240]})
            inventory.append({
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "lines": len(source.splitlines()),
                "historical": "archive" in path.parts,
                "syntax": "passed",
                "scope": "static risk screen only; no full equation certification",
                "risk_hits": hits,
            })
    return {"scope": "All src/scripts Python files parsed and risk-screened; flags are not proof of bugs.",
            "file_count": len(inventory), "files": inventory}


def check_all():
    checks = []
    def record(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    # Solve the continuum DFB boundary-value problem independently with a
    # matrix exponential: R(0)=0, S(1)=0, real mutual coupling kappa=2.
    def secular(x):
        value = complex(*x)
        propagation = np.array([[-1j*value, 2j], [-2j, 1j*value]])
        last = expm(propagation)[1, 1]
        return [last.real, last.imag]
    answer = root(secular, [3.4, 1.0], tol=1e-11)
    reference = complex(*answer.x)
    record("continuum_DFB_secular_root", answer.success and np.linalg.norm(answer.fun) < 1e-10,
           eigenvalue=[reference.real, reference.imag], boundary_residual=float(np.linalg.norm(answer.fun)))
    convergence = []
    for n in (8, 16, 32, 64):
        h = 1/n
        derivative = (_upwind_1d(n, h, True), _upwind_1d(n, h, False))
        problem = build_finite_eigenproblem(np.array([[0., 2.], [2., 0.]]), derivative, h, "staggered")
        values, fields = problem.near(reference, 4, np.ones(2*n, complex))
        selected = int(np.argmin(abs(values-reference)))
        value, field = values[selected], fields[:, selected]
        budget = problem.loss_budget(field)
        error = float(abs(value-reference))
        convergence.append({"N": n, "error": error, "eigenvalue": [float(value.real), float(value.imag)]})
        residual = relative_residual(problem.center_operator(), field, value)
        record(f"DFB_N{n}_original_equation_and_flux", residual < 1e-10 and
               abs(budget['transport_amplitude_loss_per_m']-value.imag) < 1e-10,
               residual=residual, flux_loss=budget['transport_amplitude_loss_per_m'])
    ratios = [convergence[i]['error']/convergence[i+1]['error'] for i in range(3)]
    record("DFB_continuum_second_order_convergence", min(ratios) > 3.5,
           convergence=convergence, error_reduction_ratios=ratios)

    # Direct algebra, unrelated to an eigenvalue solver: characteristic center
    # averaging has boundary flux and no hidden interior upwind dissipation.
    for shape in ("circle", "hexagon", "square"):
        grid = build_device_grid(4, 30., shape)
        identity = sparse.identity(grid.points, format="csr")
        maximum = 0.
        negative = 0.
        for step, positive in zip(((1,0),(1,0),(0,1),(0,1),(1,1),(1,1)), (True,False)*3):
            d = _directed_upwind(grid, step, positive)
            t = identity-grid.spacing_m*d
            m = .5*(identity+t)
            lhs = (m.conj().T@d+d.conj().T@m).toarray()
            rhs = ((identity-t.conj().T@t)/grid.spacing_m).toarray()
            maximum = max(maximum, float(np.max(abs(lhs-rhs)))*grid.spacing_m)
            negative = min(negative, float(np.linalg.eigvalsh(rhs).min())*grid.spacing_m)
        record(f"six_wave_boundary_flux_{shape}", maximum < 1e-12 and negative > -1e-12,
               dimensionless_max_residual=maximum, minimum_boundary_flux_eigenvalue=negative)

    values = [1e5+2j, 1e5+.01+2j]
    almost_parallel = [np.array([1,0],complex), np.array([1,1e-9],complex)]
    record("resolved_nonnormal_modes_not_deleted_by_overlap",
           independent_mode_indices(values,almost_parallel) == [0,1])
    values = [2+3j]*3
    fields = [np.array([1,0],complex),np.array([0,1],complex),np.array([1,1],complex)]
    record("true_degenerate_eigenspace_keeps_dimension_not_rotated_duplicates",
           independent_mode_indices(values,fields) == [0,1])

    grid = build_device_grid(4, 30., "square")
    wave = np.array([-.7,0.,.4])*1e5
    rng = np.random.default_rng(20261025)
    fields = [rng.normal(size=grid.points)+1j*rng.normal(size=grid.points) for _ in range(2)]
    fast = _cartesian_point_fourier_sum(grid.x_um*1e-6, grid.y_um*1e-6,*fields,wave)
    error = 0.
    for component, radiation in enumerate(fields):
        expected = np.empty((3,3),complex)
        for iy, ky in enumerate(wave):
            for ix, kx in enumerate(wave):
                expected[iy,ix] = sum(r*np.exp(-1j*(kx*x+ky*y)) for r,x,y in
                    zip(radiation,grid.x_um*1e-6,grid.y_um*1e-6))
        error = max(error,float(np.max(abs(fast[component]-expected))/max(abs(expected).max(),1e-30)))
    record("far_field_fast_sum_equals_manual_complex_point_sum", error < 2e-14,
           maximum_relative_error=error)

    cfg = SimulationConfig(numerics=NumericsConfig(points=9,noise=False))
    solver = TimeDomainSolver(cfg)
    field = np.ones((4,9,9),complex)*1e6
    carrier = np.ones((9,9))*2*cfg.carrier.transparency_density_cm3*1e6
    predicted = carrier+solver.dt*solver._carrier_rhs(carrier,field,np.zeros_like(carrier))
    rejected = False
    try:
        solver._advance_carrier(carrier,field,np.zeros_like(carrier))
    except FloatingPointError:
        rejected = True
    record("negative_Heun_predictor_rejected_not_hidden",rejected and predicted.min()<0,
           unconstrained_min_predictor_m3=float(predicted.min()),physical_initial_density_m3=float(carrier.min()))
    rejected = False
    try:
        radiated_power_W(np.ones((4,3,3),complex),-1j*np.eye(4),1e-6,cfg.optical)
    except ValueError:
        rejected = True
    record("nonpassive_negative_radiation_rejected",rejected,
           previous_maximum_density_zero_would_hide_error=True)
    alpha=27.
    field=np.arange(1,37).reshape(4,3,3).astype(complex)
    dx=2e-6
    energy=2*epsilon_0*cfg.optical.effective_index*cfg.optical.group_index*np.sum(abs(field)**2)*dx**2
    expected=2*(c/cfg.optical.group_index)*alpha*energy
    observed=radiated_power_W(field,1j*alpha*np.eye(4),dx,cfg.optical)
    record("radiated_power_matches_analytic_passive_energy_decay", abs(observed/expected-1)<1e-14,
           expected_W=float(expected),actual_W=observed)

    # This is an explicitly recorded model limitation, not a test to pass by
    # editing the equations. Inoue Eq10 and A10 use different near-equal n's.
    field=np.ones((4,1,1),complex)
    n=np.full((1,1),3*cfg.carrier.transparency_density_cm3*1e6)
    vg=c/cfg.optical.group_index
    omega=2*pi*c/(cfg.optical.wavelength_nm*1e-9)
    energy=2*epsilon_0*cfg.optical.effective_index*cfg.optical.group_index*np.sum(abs(field)**2)
    field_gain_W=vg*float(modal_gain_m(n,cfg.carrier,cfg.optical)[0,0])*energy
    carrier_exchange_W=float((vg*active_gain_m(n,cfg.carrier)*photon_density_m3(field,cfg.carrier,cfg.optical)
                              *hbar*omega*cfg.carrier.active_thickness_nm*1e-9)[0,0])
    limitation={"scope":"Semiconductor equation-normalization diagnostic; not Yb:YAG rates.",
                "field_stimulated_energy_rate_over_carrier_stimulated_energy_rate":field_gain_W/carrier_exchange_W,
                "n_active_over_n_eff":cfg.optical.active_index/cfg.optical.effective_index,
                "interpretation":"The copied Inoue Eq10+A10 formula set is not an exactly energy-conserving standalone reservoir pair under this total-field-energy convention. Near-equal-index/modal-normalization approximation must be stated; equations were not silently redefined."}

    # An accurate frozen matrix eigenpair is insufficient for a frequency root.
    frozen=3.+2j
    physical_residual=background_detuning(950.,3.4,280.)-frozen.real
    record("frequency_closure_is_separate_from_matrix_eigenpair_accuracy", abs(physical_residual)>1e3,
           exact_scalar_matrix_eigen_residual=0.,physical_frequency_residual_per_m=physical_residual)
    return {"status":"passed" if all(x['passed'] for x in checks) else "failed",
            "checks":checks,"passed":sum(x['passed'] for x in checks),"total":len(checks),
            "explicit_model_limitation":limitation,
            "claim_boundary":"Independent small mathematical checks only; no full-wave, full-spectrum, experimental or all-physics certification."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/current/physics_audit_20261025/algorithms')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    static=static_screen()
    report=check_all()
    for name,data in [('all_py_static_risk_screen.json',static),('independent_checks.json',report)]:
        (args.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'checks_passed':report['passed'],'checks_total':report['total'],
                      'source_files_static_screened':static['file_count'],
                      'semiconductor_energy_exchange_ratio':report['explicit_model_limitation']['field_stimulated_energy_rate_over_carrier_stimulated_energy_rate']}))
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
