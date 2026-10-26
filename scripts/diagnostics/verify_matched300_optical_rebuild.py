"""Rebuild selected final optical models from their portable geometry inputs.

This verifies reconstruction and the saved finite eigenpairs. It does not
repeat the full parameter scan or claim an exhaustive continuum spectrum.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from pcselsim.finite_discretization import relative_residual
from pcselsim.custom_analysis import FourWaveOpticalSpec, finite_area_eigenproblem
from pcselsim.triangular_finite import TriangularFiniteSpec, triangular_finite_eigenproblem
from matched300_result_provenance import (DEFAULT_PROTOCOL, capture_case_inputs,
    validate_case_audit_binding, validate_completed_manifest)


def path(name):
    value = Path(name)
    return value if value.is_absolute() else ROOT / value


def verify(case, protocol_path=None):
    started = time.monotonic()
    summary_path = path(case["summary_path"])
    inputs = capture_case_inputs(case, protocol_path=protocol_path)
    summary = inputs["summary"]
    wavelength, order, dz = case["wavelength_nm"], case["D"], case["dz_nm"]
    if case["id"] == "semiconductor_square":
        from scripts.optimization.matched300_square_search import Design, build_model, BASE
        _, _, result = build_model(Design(**summary["design"]), order, dz,
                                    fixed_wavelength_nm=wavelength, tune_period=False, vertices=96)
        gain_weight = result.vertical_mode.confinement[BASE.ACTIVE_LAYER_NAME] * BASE.ACTIVE_INDEX / result.effective_index
        flux = None
    elif case["id"] == "semiconductor_triangular":
        from scripts.optimization.matched300_triangular_search import ThreeHole
        from scripts.diagnostics.screen_triangular_stripe_limits import fixed_model
        p, result = fixed_model(ThreeHole(**summary["three_hole_spec"]), order, dz, wavelength)
        gain_weight = result.vertical_mode.confinement[p.ACTIVE_LAYER_NAME] * p.ACTIVE_INDEX / result.effective_index
        flux = None
    else:
        from scripts.optimization.matched300_yag_search import optical
        from run_ybyag_pcsel import active_weights
        model, period, guide = optical(case["lattice"], summary["design"], order, dz,
                                      wavelength, period=summary["period_nm"], family=case.get("TE_family", 0))
        result = model.result
        _, _, qgain = active_weights(result, guide, 24)
        gain_weight = float(np.sum(qgain))
        flux = model
    internal_loss_cm = inputs["internal_intensity_loss_cm_inverse"]
    with np.load(path(case["fields_path"]), allow_pickle=False) as stored:
        saved_c = stored[case["coupling_key"]]
        field = stored[case["field_key"]]
    c_error = np.linalg.norm(result.coupling_m-saved_c) / max(np.linalg.norm(saved_c), 1e-30)
    if c_error > 2e-10:
        raise ValueError(f"Portable model reconstruction mismatch in {case['id']}: {c_error:g}")
    if case["lattice"] == "square":
        n = field.shape[1]
        spec = FourWaveOpticalSpec(wavelength, result.effective_index, result.group_index,
            300., internal_loss_cm, (0.,)*4, (0.,)*4, n, "staggered", 12, 0, 1600)
        problem = finite_area_eigenproblem(spec, result.coupling_m)
    else:
        spec = TriangularFiniteSpec(150., case["grid"], 0., 12, "square", "staggered")
        problem, _ = triangular_finite_eigenproblem(result, spec)
    eigenvalue = complex(*case["eigenvalue_per_m"])
    residual = relative_residual(problem.center_operator(), field, eigenvalue)
    budget = problem.loss_budget(field)
    # A frozen-reference eigenpair can have a perfect matrix residual while
    # failing the physical frequency equation. Test both independently.
    beta = 2*np.pi*result.effective_index/(wavelength*1e-9)
    reciprocal_factor = 2*np.pi if case["lattice"] == "square" else 4*np.pi/np.sqrt(3)
    beta0 = reciprocal_factor/(result.lattice_constant_nm*1e-9)
    frequency_residual = float((beta*beta-beta0*beta0)/(2*beta0)-eigenvalue.real)
    gain = (2*eigenvalue.imag/100 + internal_loss_cm) / gain_weight
    gain_error = abs(gain/case["g_th_cm_inverse"]-1)
    if residual > 2e-8 or gain_error > 2e-10 or abs(frequency_residual) > .05:
        raise ValueError(f"Rebuilt eigenpair, gain or frequency mismatch in {case['id']}: "
                         f"{residual:g}, {gain_error:g}, {frequency_residual:g} /m")
    audit = dict(id=case["id"], optical_coupling_relative_error=float(c_error),
        rebuilt_original_eigenpair_residual=residual, rebuilt_gain_weight=float(gain_weight),
        rebuilt_material_threshold_gain_cm_inverse=float(gain), gain_relative_difference=float(gain_error),
        rebuilt_frequency_residual_per_m=frequency_residual,
        loss_budget_per_m=budget,
        passive_loss_budget_closure_per_m=float(eigenvalue.imag-budget["local_amplitude_loss_per_m"]-budget["transport_amplitude_loss_per_m"]),
        summary_path=case["summary_path"], summary_sha256=inputs["summary_sha256"],
        fields_sha256=inputs["fields_sha256"], geometry_sha256=inputs["geometry_sha256"],
        protocol_sha256=inputs["protocol_sha256"], case_inputs=inputs["case_inputs"],
        case_fingerprint=inputs["case_fingerprint"],
        elapsed_seconds=time.monotonic()-started)
    if flux is not None:
        up, down = flux.side_power_losses(field)
        audit["rebuilt_up_down_power_loss_per_m"] = [up, down]
    validate_case_audit_binding(case, audit, protocol_path=protocol_path)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--case", default="all")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=ROOT/"results/current/matched300_20261021/comparison/optical_rebuild.json")
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    cases = manifest["cases"]
    if args.case == "all":
        validate_completed_manifest(manifest)
    if args.case != "all":
        cases = [case for case in cases if case["id"] == args.case]
        if not cases:
            raise ValueError("Unknown case id")
    reports = [verify(case, args.protocol) for case in cases]
    protocol_sha = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    if (hashlib.sha256(args.manifest.read_bytes()).hexdigest() != manifest_sha
            or any(row["protocol_sha256"] != protocol_sha for row in reports)):
        raise ValueError("Manifest or protocol changed during optical rebuild; rerun on frozen inputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status="passed", cases=reports,
        manifest_sha256=manifest_sha, protocol_sha256=protocol_sha,
        scope="Rebuild from final geometry inputs plus actual saved finite eigenpair; not an independent Maxwell solver or a repeated optimization."), indent=2), encoding="utf8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
