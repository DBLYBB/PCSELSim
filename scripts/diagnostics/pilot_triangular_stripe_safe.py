"""Fine-coupling finite pilots for stripe-safe three-circle designs."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
from dataclasses import replace, asdict
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.triangular_finite import TriangularFiniteSpec, TriangularFiniteMode, triangular_finite_eigenproblem, triangular_vector_far_field
from pcselsim.design_objectives import beam_metrics
from pcselsim.polarization import polarization_metrics
from scripts.diagnostics.screen_triangular_stripe_limits import fixed_model
from scripts.optimization.matched300_triangular_search import ThreeHole
from scripts.optimization.close_triangular_finalist_frequency import background_delta


def pilot(design, output, grid_cells=23):
    started = time.monotonic()
    p, model = fixed_model(design, 48, 1., 950.)
    spec = TriangularFiniteSpec(150., grid_cells, 0., 12, "square", "staggered")
    problem, grid = triangular_finite_eigenproblem(model, spec)
    rng = np.random.default_rng(30021021)
    initial = rng.normal(size=problem.stiffness.shape[0]) + 1j*rng.normal(size=problem.stiffness.shape[0])
    targets = list(model.eigenvalues_m + 10 + 10j)
    width = max(float(np.ptp(model.eigenvalues_m.real)), 2e4)
    targets.extend(np.linspace(model.eigenvalues_m.real.min()-.2*width,
                               model.eigenvalues_m.real.max()+.2*width, 9)+.1j)
    values, fields, failures = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    for index, target in enumerate(targets):
        try:
            found, vectors = problem.near(target, 12, initial, maxiter=1000, dense_limit=0)
            for value, vector in zip(found, vectors.T):
                if value.imag >= -1e-7 and independent_eigenmode(value, vector, values, fields):
                    values.append(value)
                    fields.append(vector)
        except Exception as error:
            failures.append(dict(target=[target.real, target.imag], error=repr(error)))
        print(design.name, f"target {index+1}/{len(targets)}", len(values), flush=True)
    gamma = model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME]
    weight = gamma * p.ACTIVE_INDEX / model.effective_index
    arrays = dict(coupling_m=model.coupling_m, x_um=grid.x_um, y_um=grid.y_um)
    rows = []
    for rank, index in enumerate(np.argsort(np.imag(values))[:4]):
        value, field = values[index], fields[index].reshape(6, -1)
        residual = background_delta(950., model.effective_index, design.lattice_nm) - value.real
        linear_wavelength = 950. + residual/(2*np.pi*model.group_index/(950e-9)**2*1e-9)
        rx, ry = model.radiation_amplitudes(field)
        mode = TriangularFiniteMode(str(rank), value.real, value.imag, field, 0., grid, rx, ry, discretization="staggered")
        far = triangular_vector_far_field(mode, linear_wavelength, 1., 201)
        beam = beam_metrics(far.angle_deg, far.field_x, far.field_y, requested_view_deg=1., source_nyquist_deg=far.alias_free_square_view_deg)
        budget = problem.loss_budget(field)
        rows.append(dict(rank=rank, delta_per_m=value.real, alpha_field_per_m=value.imag,
            g_th_cm=(2*value.imag/100+5)/weight, linear_estimated_wavelength_nm=linear_wavelength,
            frequency_residual_at_950_per_m=residual, beam=beam,
            eta_rad=2*budget["local_amplitude_loss_per_m"]/(2*value.imag+500),
            polarization=polarization_metrics(far.angle_deg, far.field_x, far.field_y, core_radius_deg=.2)))
        arrays.update({f"pole_{rank}_fields":field, f"pole_{rank}_far_x":far.field_x,
                       f"pole_{rank}_far_y":far.field_y, "angle_deg":far.angle_deg})
    np.savez_compressed(output/"fields.npz", **arrays)
    from scripts.optimization.matched300_triangular_search import geometry
    summary = dict(three_hole_spec=asdict(design), D=48, dz_nm=1., N=grid_cells, poles=rows,
        fabrication=geometry(design), sampled_node_span_um=[float(np.ptp(grid.x_um)),float(np.ptp(grid.y_um))],
        distinct_returned_poles=len(values), failures=failures, elapsed_seconds=time.monotonic()-started,
        scope="Frozen C(950nm), real finite-pole pilot with 6 Gamma plus 9 broad shifts; a linear wavelength estimate is not independent closure or exhaustive competition.")
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(design.name, [(r["g_th_cm"],r["beam"]["ellipticity"],r["beam"]["center_to_peak"]) for r in rows], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shares", type=float, nargs="*")
    parser.add_argument("--grid", type=int, default=23)
    parser.add_argument("--distance", type=float)
    args = parser.parse_args()
    source = ROOT/"results/current/matched300_20261021/triangular/highD_adaptive128/candidates.json"
    rows = json.loads(source.read_text(encoding="utf8"))
    names = ("local_0118",) if args.shares else ("local_0040", "local_0092", "local_0118")
    for name in names:
        row = next(r for r in rows if r["name"] == name)
        original = replace(ThreeHole(**row["three_hole_spec"]), lattice_nm=321.02794818855705)
        designs = ([replace(original, main_share=share, name=f"share_{share:.5f}".replace(".","p"))
                    for share in args.shares] if args.shares else [original])
        for design in designs:
            if args.distance is not None:
                design = replace(design, distance1=args.distance, distance2=args.distance,
                                 name=design.name+f"_d{args.distance:.5f}".replace(".","p"))
            if args.grid != 23:
                design = replace(design, name=design.name+f"_N{args.grid}")
            pilot(design, ROOT/"results/current/matched300_20261021/stripe_screen/finite_pilots"/design.name, args.grid)
