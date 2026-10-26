"""Independent continuum benchmark and optional Liang circular-device check.

Outputs are new evidence, never replacements for archived numerical results.
Run: python -B scripts/validate_finite_discretization.py --circle --output DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project
PROJECT_ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import root

from pcselsim.custom_analysis import _upwind_1d
from pcselsim.finite_discretization import build_finite_eigenproblem, relative_residual
from pcselsim.triangular_finite import TriangularFiniteSpec, solve_triangular_finite_modes
from pcselsim.triangular_six_wave import (TriangularCWTSettings, TriangularEllipse,
    TriangularLatticeCell, build_triangular_coupling)
from pcselsim.vertical import Layer, LayerStack
from pcselsim.run_output import prepare_output, write_provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--circle", action="store_true")
    args = parser.parse_args(argv)
    output = prepare_output(args.output, PROJECT_ROOT)
    write_provenance(output, PROJECT_ROOT)

    def boundary(coordinates):
        value = complex(*coordinates)
        transfer = expm(np.asarray(((-1j*value, 2j), (-2j, 1j*value))))
        return transfer[1, 1].real, transfer[1, 1].imag
    reference_result = root(boundary, (3.4, 1.), tol=1e-11)
    if not reference_result.success or np.linalg.norm(reference_result.fun) > 1e-10:
        raise RuntimeError("Independent continuum transfer-matrix root failed")
    reference = complex(*reference_result.x)
    rows = []
    for cells in (8, 16, 32, 64):
        step = 1./cells
        derivatives = (_upwind_1d(cells, step, True), _upwind_1d(cells, step, False))
        for method in ("staggered", "legacy_upwind"):
            problem = build_finite_eigenproblem(np.asarray(((0., 2.), (2., 0.))), derivatives, step, method)
            values, fields = problem.near(reference, 4, np.ones(2*cells, dtype=complex))
            index = int(np.argmin(abs(values-reference)))
            value, field = values[index], fields[:, index]
            rows.append({"cells": cells, "method": method, "real_eigenvalue": float(value.real),
                "imaginary_eigenvalue": float(value.imag), "absolute_complex_error": float(abs(value-reference)),
                "relative_eigen_residual": relative_residual(problem.center_operator(), field, value),
                **problem.loss_budget(field)})
    # Use null for the deliberately undefined legacy volume-diffusion field.
    for row in rows:
        if not np.isfinite(row["volume_numerical_diffusion_per_m"]):
            row["volume_numerical_diffusion_per_m"] = None
    evidence = {"reference": "continuum constant-coupling 1-D DFB; exact matrix exponential, R(0)=S(1)=0, kappa=2, L=1",
        "reference_eigenvalue": [float(reference.real), float(reference.imag)],
        "staggered_source": "Liang thesis Eqs. (5.18)-(5.19); doi:10.1364/OE.21.000565",
        "scope": "Discretization validation only; not validation of material data, vertical Green approximation or a complete PCSEL",
        "rows": rows}
    (output/"01_dfb_continuum_validation.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in ("staggered", "legacy_upwind"):
        subset = [row for row in rows if row["method"] == method]
        ax.loglog([row["cells"] for row in subset], [row["absolute_complex_error"] for row in subset], "o-", label=method)
    ax.set(xlabel="Number of cells", ylabel="Absolute complex-eigenvalue error", title="Independent continuum DFB boundary-value benchmark")
    ax.grid(True, which="both", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"01_dfb_continuum_validation.png", dpi=180)
    plt.close(fig)

    if args.circle:
        # Explicitly preserve the historical equivalent Table-3.1 stack and
        # r/a=.20 assumption. Do not tune them to force Table-5.1 agreement.
        cell = TriangularLatticeCell(12.7449, (TriangularEllipse(radii_over_a=(.2, .2)),))
        pc_index = np.sqrt(cell.fourier_epsilon(0, 0).real)
        layers = (Layer("p-clad", 1500., np.sqrt(11.0224)), Layer("GaAs", 59., np.sqrt(12.7449)),
                  Layer("PC", 118., pc_index), Layer("active", 88.5, np.sqrt(12.8603)),
                  Layer("n-clad", 1500., np.sqrt(11.0224)))
        stack = LayerStack(layers, np.sqrt(11.0224), np.sqrt(11.0224), 1.)
        model = build_triangular_coupling(cell, stack, "PC", 341., 995., TriangularCWTSettings(10, 3.))
        circle_rows = []
        for cells in (7, 11, 15):
            modes = solve_triangular_finite_modes(model, TriangularFiniteSpec(30., cells))
            for name, mode in modes.items():
                circle_rows.append({"radius_cells": cells, "family": name,
                    "alpha_L": mode.alpha_l, "delta_L": mode.delta_per_m*30e-6,
                    "relative_eigen_residual": mode.eigen_residual})
        circle = {"lattice_constant_nm": 341., "radius_um": 30., "radius_over_a": .2,
            "actual_fill_fraction": cell.fill_fraction, "nominal_paper_fill_fraction": .15,
            "bragg_wavelength_nm": model.bragg_wavelength_nm,
            "equivalent_layers": [layer.__dict__ for layer in layers],
            "paper_table_5p1_alpha_L": {"A": .256, "B1_B2": .176, "C": .133, "W": .184},
            "rows": circle_rows,
            "interpretation": "Full Table 5.1 reproduction is NOT achieved. Inputs are an equivalent stack and r/a=.20; W and complete lateral spectrum are not covered. No parameters were tuned to the table."}
        (output/"02_liang_circle_comparison.json").write_text(json.dumps(circle, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    print(f"Saved independent validation to {output}")


if __name__ == "__main__":
    main()
