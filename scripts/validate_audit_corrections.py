"""Independent numerical evidence for the 2026-10-21 corrections.

Writes a new result folder. Thin-layer comparisons deliberately hold the
sampled field fixed; they are quadrature diagnostics, not new device ratings.
The DFB reference is a continuous boundary-value matrix-exponential solution.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project
ROOT = bootstrap_project()

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import root

from scripts import run_paper_reproduction as paper, run_square_double_hole as square
from scripts._four_wave_workflow import build_specs
from pcselsim.custom_analysis import _upwind_1d
from pcselsim.finite_discretization import build_finite_eigenproblem
from pcselsim.run_output import default_output, prepare_output, write_provenance
from pcselsim.vertical import Layer, LayerStack


def legacy_mask_overlap(stack, mode, name):
    cursor = -sum(layer.thickness_nm for layer in stack.layers) / 2
    total = np.trapz(abs(mode.field)**2, mode.z_um * 1e-6)
    value = 0.
    for layer in stack.layers:
        stop = cursor + layer.thickness_nm
        mask = (mode.z_um*1000 >= cursor-1e-9) & (mode.z_um*1000 <= stop+1e-9)
        if layer.name == name:
            value += np.trapz(abs(mode.field[mask])**2, mode.z_um[mask]*1e-6) / total
        cursor = stop
    return float(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output = prepare_output(args.output or default_output(ROOT, 'validation'), ROOT)
    write_provenance(output, ROOT)
    quadrature = []
    for name, entry in [('paper', paper), ('square', square)]:
        parameters = replace(entry.build_parameters(), CWT_TRUNCATION_ORDER=3)
        # Keep the pre-audit physical layer stack for a clean quadrature-only comparison.
        old_layers = tuple(layer for layer in parameters.LAYERS if layer.name != 'active AlGaAs barrier 3')
        _, _, _, layers, geometry = build_specs(replace(parameters, LAYERS=old_layers))
        stack = LayerStack(tuple(Layer(x.name, x.thickness_nm, x.refractive_index) for x in layers),
                           parameters.TOP_CLADDING_INDEX, parameters.BOTTOM_CLADDING_INDEX,
                           parameters.VERTICAL_PADDING_UM)
        for step in (1., 2., 3., 4., 5.):
            mode = stack.solve_te0(geometry.bragg_wavelength_nm, step)
            old = legacy_mask_overlap(stack, mode, parameters.ACTIVE_LAYER_NAME)
            fixed = mode.confinement[parameters.ACTIVE_LAYER_NAME]
            quadrature.append(dict(structure=name, dz_nm=step, old_mask_Gamma=old,
                                   interface_Gamma=fixed, relative_underestimate=1-old/fixed))
    with (output/'thin_layer_quadrature.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=quadrature[0]); writer.writeheader(); writer.writerows(quadrature)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, name in zip(axes, ('paper', 'square')):
        rows = [row for row in quadrature if row['structure'] == name]
        ax.plot([r['dz_nm'] for r in rows], [r['old_mask_Gamma'] for r in rows], 'o--', label='old mask integral')
        ax.plot([r['dz_nm'] for r in rows], [r['interface_Gamma'] for r in rows], 'o-', label='true interfaces included')
        ax.set(xlabel='vertical grid spacing (nm)', ylabel='active-layer overlap Gamma', title=name+' same-field quadrature')
        ax.legend(); ax.grid(alpha=.25)
    fig.savefig(output/'thin_layer_quadrature.png', dpi=180); plt.close(fig)

    def boundary(coordinates):
        value = complex(*coordinates)
        propagator = expm(np.asarray(((-1j*value, 2j), (-2j, 1j*value))))
        b = propagator[1, 1]
        return b.real, b.imag
    exact = root(boundary, (3.4, 1.), tol=1e-11)
    if not exact.success or np.linalg.norm(exact.fun) > 1e-10:
        raise RuntimeError('Independent continuous DFB reference did not converge')
    reference = complex(*exact.x)
    dfb = []
    for count in (8, 16, 32, 64):
        step = 1/count
        derivatives = (_upwind_1d(count, step, True), _upwind_1d(count, step, False))
        for method in ('legacy_upwind', 'staggered'):
            problem = build_finite_eigenproblem(np.asarray(((0., 2.), (2., 0.))), derivatives, step, method)
            values, fields = problem.near(reference, 4, np.ones(2*count, dtype=complex))
            index = int(np.argmin(abs(values-reference)))
            value = values[index]
            dfb.append(dict(grid_points=count, method=method, real=value.real, imag=value.imag,
                            absolute_error=abs(value-reference)))
    with (output/'continuous_DFB_convergence.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream, fieldnames=dfb[0]); writer.writeheader(); writer.writerows(dfb)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for method in ('legacy_upwind', 'staggered'):
        rows=[r for r in dfb if r['method']==method]
        ax.loglog([r['grid_points'] for r in rows], [r['absolute_error'] for r in rows], 'o-', label=method)
    ax.set(xlabel='grid points along cavity', ylabel='complex eigenvalue absolute error', title='Independent continuous two-wave DFB reference')
    ax.legend(); ax.grid(alpha=.25)
    fig.savefig(output/'continuous_DFB_convergence.png', dpi=180); plt.close(fig)
    (output/'validation_summary.json').write_text(json.dumps({
        'scope':'quadrature and independent linear DFB accuracy; not experimental or full-wave PCSEL certification',
        'continuous_DFB_eigenvalue':[reference.real,reference.imag],
        'thin_layer_quadrature':quadrature, 'continuous_DFB':dfb,
    }, indent=2), encoding='utf-8')
    (output/'run_status.json').write_text(json.dumps({'status':'completed'}), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
