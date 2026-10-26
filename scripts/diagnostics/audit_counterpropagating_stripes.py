"""Audit the two-wave stripe limit of a frozen four-wave CWT matrix.

Large transverse wave numbers detune the other propagation pair in the
first-order envelope model. The remaining pair approaches a one-dimensional
open two-wave problem. This is a model diagnostic, not a two-dimensional
emitting mode or an independently certified Maxwell threshold.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy import sparse
from scipy.linalg import eig, expm
from scipy.optimize import root
from pcselsim.finite_discretization import build_finite_eigenproblem, relative_residual


def transfer_residual(value, coupling, length_m):
    """R(0)=0, S(L)=0 implies exp[-i sigma_z(delta I-C)L]22=0."""
    generator = -1j * np.diag([1., -1.]) @ (value * np.eye(2) - coupling)
    generator -= np.trace(generator) * np.eye(2) / 2
    # Removing the nonzero scalar exponential leaves its zeros unchanged.
    return complex(expm(generator * length_m)[1, 1])


def stripe_pair(coupling, length_um, gain_weight, internal_loss_cm, grids=(101, 201)):
    coupling = np.asarray(coupling, complex)
    length = length_um * 1e-6
    runs = []
    for n in grids:
        spacing = length / n
        plus = sparse.diags([-np.ones(n-1), np.ones(n)], [-1, 0], format="csr") / spacing
        minus = sparse.diags([np.ones(n), -np.ones(n-1)], [0, 1], format="csr") / spacing
        problem = build_finite_eigenproblem(coupling, (plus, minus), spacing, "staggered")
        values, links = eig(problem.stiffness.toarray(), problem.mass.toarray())
        fields = problem.mass @ links
        good = np.isfinite(values) & (values.imag >= -1e-8)
        candidates = np.flatnonzero(good)[np.argsort(values[good].imag)[:6]]
        operator = problem.center_operator()
        poles = []
        for k in candidates:
            value, field = values[k], fields[:, k]
            budget = problem.loss_budget(field)
            poles.append(dict(delta_per_m=float(value.real), alpha_field_per_m=float(value.imag),
                g_th_cm_inverse=float((2*value.imag/100 + internal_loss_cm)/gain_weight),
                eigen_residual=relative_residual(operator, field, value), **budget))
        runs.append(dict(grid=n, poles=poles))
    seed = runs[-1]["poles"][0]
    def equation(x):
        residual = transfer_residual(complex(*x), coupling, length)
        return [residual.real, residual.imag]
    solution = root(equation, [seed["delta_per_m"], seed["alpha_field_per_m"]], tol=1e-10)
    value = complex(*solution.x)
    residual = abs(transfer_residual(value, coupling, length))
    continuous = dict(converged=bool(solution.success and residual < 1e-7),
        delta_per_m=float(value.real), alpha_field_per_m=float(value.imag),
        g_th_cm_inverse=float((2*value.imag/100 + internal_loss_cm)/gain_weight),
        transfer_boundary_residual=float(residual), solver_message=str(solution.message))
    return dict(discrete_runs=runs, independent_continuous_transfer_pole=continuous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="Directory with spectral_challenge.json and lateral_fields.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.case / "spectral_challenge.json"
    fields_path = args.case / "lateral_fields.npz"
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    with np.load(fields_path, allow_pickle=False) as stored:
        coupling = stored["coupling_m"]
    first = summary["lowest"]
    gain_weight = (2*first["alpha_field_cm1"] + 5.) / first["g_th_cm1"]
    result = dict(source_case=str(args.case),
        source_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        source_fields_sha256=hashlib.sha256(fields_path.read_bytes()).hexdigest(),
        frozen_coupling_real_per_m=coupling.real.tolist(),
        frozen_coupling_imag_per_m=coupling.imag.tolist(),
        wavelength_reference_nm=950., nominal_length_um=300., gain_weight=gain_weight,
        internal_intensity_loss_cm_inverse=5., source_lowest_g_th_cm_inverse=first["g_th_cm1"],
        x_pair=stripe_pair(coupling[:2, :2], 300., gain_weight, 5.),
        y_pair=stripe_pair(coupling[2:, 2:], 300., gain_weight, 5.),
        scope="Frozen-C, first-order envelope, asymptotic two-wave diagnostic. No actual 2D Jones field, finite-frequency closure or claim that this limiting field is reached at a specified finite transverse wave number. Refining the transverse census remains necessary; no artificial volume damping or new physics was added.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({pair: result[pair]["independent_continuous_transfer_pole"]
                      for pair in ("x_pair", "y_pair")}, indent=2))


if __name__ == "__main__":
    main()
