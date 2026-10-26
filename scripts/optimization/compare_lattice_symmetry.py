"""Compare actual near-Gamma directional dispersion of C4, C6 and C2 cells.

This diagnoses a symmetry mechanism, not finite-cavity threshold, a Dirac
point, topological charge or a full-wave validation of the optimized devices.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pcselsim.geometry import Ellipse, SquareLatticeCell
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from pcselsim.triangular_six_wave import (TriangularEllipse, TriangularLatticeCell,
    TriangularCWTSettings, TRIANGULAR_DIRECTIONS, build_triangular_coupling)
from pcselsim.vertical import Layer, LayerStack
from pcselsim.run_output import default_output, prepare_output, write_provenance
from scripts import run_square_double_hole as baseline


def stack_for(cell):
    return LayerStack(tuple(Layer(layer.name, layer.thickness_nm,
        float(np.sqrt(cell.fourier_epsilon(0, 0).real))
        if layer.name == baseline.PC_LAYER_NAME else layer.refractive_index)
        for layer in baseline.LAYERS), baseline.TOP_CLADDING_INDEX,
        baseline.BOTTOM_CLADDING_INDEX, baseline.VERTICAL_PADDING_UM)


def scan(model, directions, beta0, mode_index=0):
    phi = np.linspace(0., 2*np.pi, 180, endpoint=False)
    radii = np.geomspace(2e-5, 2e-3, 21)
    reference_value = model.eigenvalues_m[mode_index]
    reference_vector = model.eigenvectors[:, mode_index]
    values = np.empty((len(radii), len(phi)), complex)
    overlaps = np.empty(values.shape, float)
    for col, angle in enumerate(phi):
        previous = reference_vector.copy()
        for row, radius in enumerate(radii):
            mismatch = directions @ (beta0 * radius * np.array([np.cos(angle), np.sin(angle)]))
            eigenvalues, eigenvectors = np.linalg.eig(model.coupling_m + np.diag(mismatch))
            eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
            projection = abs(previous.conj() @ eigenvectors)**2
            winner = int(np.argmax(projection))
            values[row, col] = eigenvalues[winner]
            overlaps[row, col] = abs(np.vdot(reference_vector, eigenvectors[:, winner]))**2
            previous = eigenvectors[:, winner]
    shift = values.real - reference_value.real
    mean = shift.mean(axis=1)
    harmonics = {str(m): 2*abs(np.mean(shift*np.exp(-1j*m*phi), axis=1)) for m in (2, 4, 6)}
    fit_orders = {}
    for m, amplitude in harmonics.items():
        valid = (amplitude > 1e-6) & (overlaps.min(axis=1) > .9)
        fit_orders[m] = float(np.polyfit(np.log(radii[valid]), np.log(amplitude[valid]), 1)[0]) if valid.sum() >= 4 else None
    return radii, phi, values, overlaps, mean, harmonics, fit_orders


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = prepare_output(args.output or default_output(ROOT, "optimization_20261021/symmetry"), ROOT)
    write_provenance(output, ROOT)
    (output/"run_status.json").write_text(json.dumps({"status": "running"}))
    fill = .15
    rows = []
    arrays = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for lattice in ("square", "triangular"):
        a = 277. if lattice == "square" else 277.*2/np.sqrt(3.)
        primitive_area = 1. if lattice == "square" else np.sqrt(3.)/2
        radius = np.sqrt(fill*primitive_area/np.pi)
        for aspect in (1., 1.2):
            radii = (float(radius*np.sqrt(aspect)), float(radius/np.sqrt(aspect)))
            if lattice == "square":
                cell = SquareLatticeCell(3.554**2, (Ellipse((0., 0.), radii),))
                model = build_geometry_coupling(cell, stack_for(cell), baseline.PC_LAYER_NAME,
                    a, 950., ThreeDCWTSettings(10, 3.))
                directions = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
                symmetry = "C4v" if aspect == 1 else "C2v"
            else:
                cell = TriangularLatticeCell(3.554**2, (TriangularEllipse(radii_over_a=radii),))
                model = build_triangular_coupling(cell, stack_for(cell), baseline.PC_LAYER_NAME,
                    a, 950., TriangularCWTSettings(10, 3.))
                directions = TRIANGULAR_DIRECTIONS
                symmetry = "C6v" if aspect == 1 else "C2v"
            name = f"{lattice}_{symmetry}"
            beta0 = 2*np.pi/(277e-9)  # equal physical reciprocal-shell magnitude
            q, phi, values, overlaps, mean, harmonics, fit_orders = scan(model, directions, beta0)
            arrays.update({name+"_q": q, name+"_phi": phi, name+"_values": values,
                           name+"_reference_overlap": overlaps, name+"_C": model.coupling_m})
            m = 4 if lattice == "square" else 6
            if aspect != 1:
                m = 2
            amplitude = harmonics[str(m)]
            axes[0].loglog(q, amplitude, "o-", label=f"{name}: m={m}")
            axes[1].loglog(q, amplitude/np.maximum(abs(mean), 1e-30), "o-", label=name)
            rows.append({"name": name, "fill": fill, "lattice_nm": a, "hole_radii_over_a": radii,
                "bragg_nm": model.bragg_wavelength_nm,
                "reference_mode": "A, a selected Gamma singlet",
                "reference_value_per_m": [float(model.eigenvalues_m[0].real), float(model.eigenvalues_m[0].imag)],
                "minimum_reference_overlap": float(overlaps.min()),
                "q_relative_to_common_beta0": q.tolist(),
                "angular_mean_shift_per_m": mean.tolist(),
                "angular_harmonic_amplitude_per_m": {key: value.tolist() for key, value in harmonics.items()},
                "empirical_log_slopes": fit_orders,
                "angular_sample_count": len(phi), "CWT_order": 10, "dz_nm": 3.,
                "stack": [asdict(layer) for layer in stack_for(cell).layers]})
            print(name, "harmonic", m, "slope", fit_orders[str(m)], "overlap", overlaps.min(), flush=True)
    axes[0].set(xlabel="|k| / common reciprocal-shell magnitude", ylabel="Directional harmonic amplitude (m$^{-1}$)",
                title="Directional frequency detuning of a Gamma singlet")
    axes[1].set(xlabel="|k| / common reciprocal-shell magnitude", ylabel="Harmonic / angular-mean frequency shift",
                title="More rotational symmetry delays allowed anisotropy")
    for ax in axes:
        ax.grid(True, which="both", alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(output/"directional_anisotropy.png", dpi=190)
    plt.close(fig)
    np.savez_compressed(output/"symmetry_arrays.npz", **arrays)
    report = {"scope": "Frozen-vertical near-Gamma CWT directional-dispersion diagnostic, not a finite-cavity or full-wave validation",
              "fixed_parameters": {"fill": fill, "common_beta0_per_m": 2*np.pi/277e-9},
              "symmetry_argument": "For an analytic non-degenerate reciprocal scalar band: C4 first allows angular degree 4, C6 degree 6; lowering to C2 permits degree 2. It does not prove lower threshold, a bright beam, or single-mode operation.",
              "rows": rows}
    (output/"symmetry_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output/"run_status.json").write_text(json.dumps({"status": "completed", "completed_at": datetime.now().isoformat(),
        "scope": report["scope"]}, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
