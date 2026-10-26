"""Reproducible exploratory six-wave semiconductor search; never edits entry defaults.

Examples (use a fresh output directory for every stage)::
  python -B scripts/optimization/triangular_semiconductor_search.py --stage screen
  python -B scripts/optimization/triangular_semiconductor_search.py --stage finite --input SCREEN
  python -B scripts/optimization/triangular_semiconductor_search.py --stage validate --input FINITE

The finite search keeps all residual-verified poles found near all six Gamma
edges, including higher transverse states and repeated independent eigenvectors.
It is a local spectral search, not an exhaustive open-cavity spectrum.  An
unresolved degeneracy is a failed single-mode criterion, not a favorable gap.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    # Environment limits above are applied before importing numpy/scipy in
    # the documented fresh-process CLI, so this diagnostic helper is optional.
    from contextlib import nullcontext
    def threadpool_limits(*args, **kwargs):
        return nullcontext()

from pcselsim.design_objectives import beam_acceptance, beam_metrics
from pcselsim.fabrication import (FabricationRules, TRIANGULAR_DIRECT,
    evaluate_fabrication, pareto_mask, perturb_triangular_inclusions)
from pcselsim.finite_discretization import relative_residual, staggered_grid_diagnostics
from pcselsim.mode_identity import independent_eigenmode
from pcselsim.run_output import default_output, prepare_output, write_provenance
from pcselsim.triangular_finite import (TriangularFiniteMode, TriangularFiniteSpec,
    triangular_finite_eigenproblem, triangular_vector_far_field, _transform_six_wave_fields)
from pcselsim.triangular_six_wave import (TRIANGULAR_MODE_NAMES, TriangularCWTSettings,
    TriangularEllipse, TriangularPolygon, plot_triangular_lattice)
from scripts._triangular_workflow import (TriangularCandidate, aperture_area_m2,
    build_model_for_candidate, threshold_audit)
from scripts.run_triangular_three_hole import build_parameters


@dataclass(frozen=True)
class Motif:
    name: str
    main_shape: str = "circle"
    fill: float = .15
    main_share: float = 1.
    satellites: int = 0
    distance: float = .4
    angle_deg: float = 0.
    main_angle_deg: float = 0.
    aspect: float = 1.
    satellite_aspect: float = 1.
    satellite_angle_offset_deg: float = 0.
    rounding: float = .30
    spread_deg: float = 0.
    asymmetry: float = 0.
    pc_thickness_nm: float = 190.
    aperture: str = "circle"
    radius_um: float = 150.
    lattice_nm: float = 0.


RULES = (FabricationRules("fine_40_50_3nm", 40, 50, 3, 3, 5),
         FabricationRules("research_60_65_3nm", 60, 65, 3, 3, 10))


def make_candidate(spec: Motif) -> TriangularCandidate:
    """Exact-area holes in Cartesian physical coordinates, then fractional basis."""
    area = np.sqrt(3)/2
    if spec.satellites == 0:
        shares, centers, angles = [1.], [(0., 0.)], [spec.main_angle_deg]
    else:
        if spec.satellites == 2:
            phases = np.asarray((-30-spec.spread_deg, 30+spec.spread_deg))+spec.angle_deg
        else:
            phases = np.arange(spec.satellites)*360/spec.satellites+spec.angle_deg
        share = np.full(spec.satellites, (1-spec.main_share)/spec.satellites)
        # Deliberately controlled C3/C6 breaking; total satellite area unchanged.
        if spec.asymmetry:
            share *= 1+spec.asymmetry*np.cos(np.deg2rad(phases-spec.angle_deg))
            share *= (1-spec.main_share)/share.sum()
        shares = [spec.main_share, *share]
        centers = [(0., 0.)]+[tuple(np.linalg.solve(TRIANGULAR_DIRECT,
            spec.distance*np.asarray((np.cos(np.deg2rad(t)), np.sin(np.deg2rad(t)))))) for t in phases]
        angles = [spec.main_angle_deg, *(phases+spec.satellite_angle_offset_deg)]
    holes = []
    for index, (share, center, angle) in enumerate(zip(shares, centers, angles, strict=True)):
        if share <= 1e-12:
            continue
        shape = spec.main_shape if index == 0 else "ellipse"
        aspect = spec.aspect if index == 0 else spec.satellite_aspect
        if shape == "triangle":
            unit = TriangularPolygon.rounded_regular((0., 0.), 1.,
                corner_fraction=spec.rounding, samples_per_corner=13)
            radius = np.sqrt(spec.fill*share/unit.fill_fraction)
            hole = TriangularPolygon.rounded_regular(center, radius, angle_deg=angle,
                corner_fraction=spec.rounding, samples_per_corner=13)
        else:
            radius = np.sqrt(spec.fill*share*area/np.pi)
            hole = TriangularEllipse(center, (radius*np.sqrt(aspect), radius/np.sqrt(aspect)), angle)
        holes.append(hole)
    return TriangularCandidate(spec.name, tuple(holes))


def parameters(spec: Motif):
    p = build_parameters()
    layers = tuple(replace(layer, thickness_nm=spec.pc_thickness_nm)
        if layer.name == p.PC_LAYER_NAME else layer for layer in p.LAYERS)
    return replace(p, LAYERS=layers, DEVICE_SHAPE=spec.aperture,
        DEVICE_HALF_SIZE_UM=spec.radius_um,
        LATTICE_CONSTANT_NM=spec.lattice_nm if spec.lattice_nm > 0 else p.LATTICE_CONSTANT_NM)


def enumerate_motifs() -> list[Motif]:
    """Deterministic balanced catalogue, with both symmetry and broken symmetry."""
    items = []
    for shape in ("circle", "ellipse", "triangle"):
        for fill in (.09, .13, .17, .21, .25, .29):
            for thickness in (120., 190., 260.):
                aspects = (1.,) if shape != "ellipse" else (1.08, 1.20, 1.40, 1.65)
                angles = (0., 30.) if shape == "triangle" else (0.,)
                for aspect in aspects:
                    for angle in angles:
                        items.append(Motif("", shape, fill, aspect=aspect,
                            main_angle_deg=angle, pc_thickness_nm=thickness))
    # Main + two satellites spans the published working branch without assuming
    # that its unresolved A/B2 ordering is favorable. Three/six rings test C3/C6.
    rng = np.random.default_rng(20261021)
    for count in (2, 3, 6):
        for index in range(96 if count < 6 else 48):
            shape = ("circle", "ellipse", "triangle")[index % 3]
            items.append(Motif("", shape, float(rng.uniform(.12, .31)),
                float(rng.uniform(.25, .65)) if count < 6 else float(rng.uniform(.15, .35)),
                count, float(rng.uniform(.31, .52)), float(rng.uniform(0, 60)),
                float(rng.uniform(0, 60)), float(rng.uniform(1, 1.5)) if shape == "ellipse" else 1.,
                float(rng.uniform(1, 1.3)), float(rng.choice((0, 30, 90))),
                float(rng.uniform(.22, .42)), float(rng.uniform(-8, 16)),
                float(rng.choice((0., 0., .08, .18, .30))),
                float(rng.choice((120., 155., 190., 225., 260.)))))
    # Exact C3/C6 references, not random near-symmetry mislabeled as symmetry.
    for count in (3, 6):
        for fill in (.16, .22, .28, .34):
            for distance in (.36, .43, .50):
                items.append(Motif("", "circle", fill, .4 if count == 3 else .25,
                    count, distance, pc_thickness_nm=190.))
    return [replace(spec, name=f"tri_{index:04d}") for index, spec in enumerate(items)]


def enumerate_symmetry_motifs() -> list[Motif]:
    """C3/C6 rings without a center hole, plus controlled symmetry breaking.

    A center inclusion consumes the narrow 320-nm cell's process budget. It
    must not be assumed that the failure of a seven-hole motif excludes a
    three-hole C3/C6-equivalent arrangement.
    """
    items = []
    for fill in (.085, .105, .125, .145, .18, .22):
        for distance in (.20, .25, .30, .35, .40):
            for aspect, asymmetry in ((1., 0.), (1., .12), (1.2, 0.), (1.2, .12)):
                for thickness in (155., 225.):
                    items.append(Motif("", "circle", fill, 0., 3, distance,
                        angle_deg=30., satellite_aspect=aspect,
                        asymmetry=asymmetry, pc_thickness_nm=thickness))
    for fill in (.17, .20, .24, .28):
        for distance in (.29, .3333333333, .37):
            for asymmetry in (0., .12):
                items.append(Motif("", "circle", fill, 0., 6, distance,
                    asymmetry=asymmetry, pc_thickness_nm=190.))
    return [replace(spec, name=f"sym_{index:04d}") for index, spec in enumerate(items)]


def enumerate_balanced_motifs() -> list[Motif]:
    """Local single-hole outcoupling trade-off around the strict-process branch."""
    items = []
    for fill in (.27, .29, .31, .33):
        for aspect in (1.06, 1.08, 1.10, 1.12):
            for thickness in (95., 120., 145.):
                items.append(Motif("", "ellipse", fill, aspect=aspect,
                    pc_thickness_nm=thickness, aperture="square", radius_um=150.,
                    lattice_nm=324.81810055428946))
    return [replace(spec, name=f"bal_{index:04d}") for index,spec in enumerate(items)]


def json_write(path, data):
    def clean(value):
        if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, np.ndarray): return clean(value.tolist())
        if isinstance(value, (np.integer, np.bool_)): return value.item()
        if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
        return value
    path.write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list, tuple))} for row in rows]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(flat)


def build_model(spec, order=8, dz=4., candidate_override=None):
    p = parameters(spec)
    candidate = make_candidate(spec) if candidate_override is None else candidate_override
    cell, layers, model = build_model_for_candidate(p, candidate, TriangularCWTSettings(order, dz))
    return p, candidate, cell, model


def fabrication(spec):
    p = parameters(spec)
    candidate = make_candidate(spec)
    reports = [evaluate_fabrication(candidate.inclusions, p.LATTICE_CONSTANT_NM,
        TRIANGULAR_DIRECT, rules) for rules in RULES]
    return reports


def screen_one(spec):
    reports = fabrication(spec)
    row = {"name": spec.name, "spec": asdict(spec), "geometry": reports,
        "fine_process_pass": reports[0]["margin_pass"],
        "research_process_pass": reports[1]["margin_pass"],
        "gap_nm": reports[0]["minimum_periodic_gap_nm"],
        "width_nm": reports[0]["minimum_caliper_nm"], "status": "geometry_rejected"}
    if not reports[0]["margin_pass"]:
        return row
    p, candidate, cell, model = build_model(spec)
    gamma = model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME]
    anti = (model.coupling_m-model.coupling_m.conj().T)/(2j)
    lower_audit = threshold_audit(p, 0., aperture_area_m2(spec.aperture, spec.radius_um), gamma, model.effective_index)
    row.update(status="screened", Gamma=gamma, effective_index=model.effective_index,
        bragg_nm=model.bragg_wavelength_nm, fill=cell.fill_fraction,
        lower_bound_gth_cm=lower_audit["required_material_gain_cm-1"],
        gamma_detuning_cm=model.eigenvalues_m.real/100,
        gamma_field_loss_cm=model.eigenvalues_m.imag/100,
        passivity_minimum_per_m=float(np.linalg.eigvalsh(anti).min()),
        passivity_projection_per_m=model.passivity_correction_m)
    return row


def collect_spectrum(model, spec, cells, count=10, dense_limit=0, extra_targets=0, full_dense=False):
    finite = TriangularFiniteSpec(spec.radius_um, cells, 0., count, spec.aperture, "staggered")
    problem, grid = triangular_finite_eigenproblem(model, finite)
    operator = problem.center_operator()
    targets = []
    for value in model.eigenvalues_m:
        if not any(abs(value-target) < 1e-7*max(abs(value), 1.) for target in targets):
            targets.append(value)
    if extra_targets:
        width = max(float(np.ptp(model.eigenvalues_m.real)), 2e4)
        targets.extend(np.linspace(model.eigenvalues_m.real.min()-.15*width,
            model.eigenvalues_m.real.max()+.15*width, extra_targets).astype(complex))
    values_out, fields_out = [], []
    def retain(value, field):
        # Resolved nonnormal eigenmodes can be nearly parallel. Compare the
        # complex eigenvalue first, using the shared numerical-identity rule.
        # 中文：已分辨的非正规模式即使场近共线也保留；只对同复极点做子空间去重。
        if not independent_eigenmode(value,field,values_out,fields_out):return
        values_out.append(value)
        fields_out.append(field)
    x = np.arange(problem.stiffness.shape[0], dtype=float)
    initial = 1+.2*np.cos(2*np.pi*x/len(x))+.17j*np.sin(6*np.pi*x/len(x))
    if full_dense:
        problem.near(targets[0], count, initial, dense_limit=problem.stiffness.shape[0])
        dense_values, dense_fields = problem._dense_spectrum
        keep = dense_values.imag >= -1e-7
        values_out = list(dense_values[keep])
        fields_out = list(dense_fields[:, keep].T)
    for target in (() if full_dense else targets):
        values, fields = problem.near(target, count, initial, maxiter=1200, dense_limit=dense_limit)
        for value, field in zip(values, fields.T, strict=True):
            if value.imag < -1e-7: continue
            # Preserve independent exact symmetry partners rather than dropping
            # the second member merely because its eigenvalue is equal.
            retain(value, field)
    # Arnoldi may recover only one vector from an exact two-dimensional pole.
    # Add symmetry partners only when the actual finite operator verifies them.
    for value, vector in (() if full_dense else list(zip(values_out, fields_out, strict=True))):
        for angle in (60., 120.):
            t = np.deg2rad(angle)
            transform = np.asarray(((np.cos(t), -np.sin(t)), (np.sin(t), np.cos(t))))
            partner = _transform_six_wave_fields(vector.reshape(6, grid.points), grid, transform)
            if partner is not None and relative_residual(operator, partner, value) < 1e-7:
                retain(value, partner.ravel()/np.linalg.norm(partner))
    order = np.argsort(np.imag(values_out), kind="stable")
    values = np.asarray(values_out)[order]
    fields = np.asarray(fields_out)[order].reshape(-1, 6, grid.points)
    records = []
    for index, (value, field) in enumerate(zip(values, fields, strict=True)):
        budget = problem.loss_budget(field)
        projection = np.sum(abs(model.eigenvectors.conj().T@field)**2, axis=1)
        family = TRIANGULAR_MODE_NAMES[int(np.argmax(projection))]
        records.append({"spectral_index": index, "family_projection_label": family,
            "delta_per_m": float(value.real), "alpha_per_m": float(value.imag),
            "family_projection": projection.tolist(),
            "relative_eigen_residual": relative_residual(operator, field, value),
            "energy_closure_per_m": float(value.imag-budget["local_amplitude_loss_per_m"]-budget["transport_amplitude_loss_per_m"]),
            **budget})
    return records, fields, grid


def evaluate_finite(spec, cells=9, order=8, dz=4., count=10, save=None, candidate_override=None,
                    extra_targets=0, full_dense=False):
    p, candidate, cell, model = build_model(spec, order, dz, candidate_override)
    records, fields, grid = collect_spectrum(model, spec, cells, count,
        extra_targets=extra_targets, full_dense=full_dense)
    if len(records) < 2: raise RuntimeError("Fewer than two independently retained poles")
    gamma = model.vertical_mode.confinement[p.ACTIVE_LAYER_NAME]
    area = aperture_area_m2(spec.aperture, spec.radius_um)
    for record in records:
        audit = threshold_audit(p, record["alpha_per_m"], area, gamma, model.effective_index)
        record.update(g_th_cm=audit["required_material_gain_cm-1"],
            J_th_A_cm2=audit["threshold_current_density_A_cm-2"], I_th_A=audit["threshold_current_A"],
            eta_rad=2*record["local_amplitude_loss_per_m"]/(2*record["alpha_per_m"]+100*p.INTERNAL_LOSS_CM))
        record["frozen_CWT_pole_wavelength_nm"] = float(1/(1/model.bragg_wavelength_nm+
            record["delta_per_m"]*1e-9/(2*np.pi*model.group_index)))
    chosen = records[0]
    ex, ey = model.radiation_amplitudes(fields[0])
    mode = TriangularFiniteMode("lowest_found", chosen["delta_per_m"], chosen["alpha_per_m"],
        fields[0], max(chosen["family_projection"]), grid, ex, ey, discretization="staggered",
        eigen_residual=chosen["relative_eigen_residual"])
    far = triangular_vector_far_field(mode, chosen["frozen_CWT_pole_wavelength_nm"], 1., 161)
    metrics = beam_metrics(far.angle_deg, far.field_x, far.field_y,
        requested_view_deg=1., source_nyquist_deg=far.alias_free_square_view_deg)
    acceptance = beam_acceptance(metrics, require_sampling=True)
    gap = records[1]["alpha_per_m"]-chosen["alpha_per_m"]
    row = {"name": spec.name, "spec": asdict(spec), "status": "finite_solved",
        "N": cells, "D": order, "dz_nm": dz, "retained_poles": len(records),
        "g_th_cm": chosen["g_th_cm"], "J_th_A_cm2": chosen["J_th_A_cm2"], "I_th_A": chosen["I_th_A"],
        "alpha_field_cm": chosen["alpha_per_m"]/100, "alpha_radiation_field_cm": chosen["local_amplitude_loss_per_m"]/100,
        "eta_rad": chosen["eta_rad"], "field_loss_gap_cm": gap/100,
        "g_th_gap_fraction": (records[1]["g_th_cm"]-chosen["g_th_cm"])/chosen["g_th_cm"],
        "Gamma": gamma, "effective_index": model.effective_index,
        "group_index": model.group_index,
        "bragg_nm": model.bragg_wavelength_nm, "beam": metrics, "beam_acceptance": acceptance,
        "frozen_CWT_pole_wavelength_nm": chosen["frozen_CWT_pole_wavelength_nm"],
        "far_field_wavelength_nm": chosen["frozen_CWT_pole_wavelength_nm"],
        "family": chosen["family_projection_label"], "spectrum": records,
        "source_sampling_view_deg": far.alias_free_square_view_deg,
        "maximum_residual": max(r["relative_eigen_residual"] for r in records),
        "maximum_energy_closure_per_m": max(abs(r["energy_closure_per_m"]) for r in records),
        "single_mode_status": "unverified_single_grid_local_spectrum",
        "spectral_search_real_targets": extra_targets,
        "complete_dense_discrete_spectrum": full_dense,
        "ellipticity": metrics["ellipticity"], "center_to_peak": metrics["center_to_peak"],
        "radial_anisotropy_l2": metrics["radial_anisotropy_l2"]}
    reports = [evaluate_fabrication(candidate.inclusions, p.LATTICE_CONSTANT_NM,
        TRIANGULAR_DIRECT, rules) for rules in RULES]
    row.update(fine_process_pass=reports[0]["margin_pass"], research_process_pass=reports[1]["margin_pass"],
        geometry=reports, gap_nm=reports[0]["minimum_periodic_gap_nm"], width_nm=reports[0]["minimum_caliper_nm"])
    if save:
        save.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save/"complex_fields.npz", eigenvalues=np.asarray([r["delta_per_m"]+1j*r["alpha_per_m"] for r in records]),
            fields=fields, x_um=grid.x_um, y_um=grid.y_um, coupling_m=model.coupling_m,
            angle_deg=far.angle_deg, far_field_x=far.field_x, far_field_y=far.field_y,
            all_radiation_x=model.radiation_amplitudes(fields.transpose(1, 0, 2))[0],
            all_radiation_y=model.radiation_amplitudes(fields.transpose(1, 0, 2))[1])
        json_write(save/"result.json", row)
        plot_triangular_lattice(cell, save/"lattice.png", cells=3)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        axes[0].tripcolor(grid.x_um, grid.y_um, mode.intensity, shading="gouraud")
        axes[0].set(aspect="equal", title=f"{spec.name}: actual N={cells}", xlabel="x (um)", ylabel="y (um)")
        axes[1].imshow(far.power, origin="lower", extent=(-far.evaluated_view_deg, far.evaluated_view_deg)*2, cmap="inferno")
        axes[1].set(title=f"ellipse={row['ellipticity']:.3f}; C/P={row['center_to_peak']:.3f}", xlabel="theta x (deg)", ylabel="theta y (deg)")
        axes[2].plot(np.arange(len(records)), [r["g_th_cm"] for r in records], "o-")
        axes[2].set(xlabel="Retained pole (loss sorted)", ylabel="g_th (cm^-1)", title=f"eta_rad={row['eta_rad']:.2%}; local spectrum")
        fig.savefig(save/"finite_diagnostics.png", dpi=180)
        plt.close(fig)
    return row


def select_screen(rows, limit):
    viable = [row for row in rows if row.get("status") == "screened"]
    # Balanced topology/thickness selection, then low gain lower bound. Never
    # use vanishing Gamma-radiation loss as an optimization reward.
    buckets = {}
    for row in viable:
        s = row["spec"]
        key = (s["main_shape"], s["satellites"], s["pc_thickness_nm"])
        buckets.setdefault(key, []).append(row)
    for bucket in buckets.values(): bucket.sort(key=lambda r: r["lower_bound_gth_cm"])
    answer = []
    while len(answer) < limit and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(answer) < limit: answer.append(buckets[key].pop(0))
    return answer


def rank_score(row):
    if row.get("status") != "finite_solved": return float("inf")
    return (row["g_th_cm"] + 200*max(row["ellipticity"]-1.10, 0)
        + 250*max(.90-row["center_to_peak"], 0)
        + 200*max(row["radial_anisotropy_l2"]-.15, 0)
        + 500*max(.01-row["eta_rad"], 0))


def local_variants(spec, start):
    items = []
    for key, values in (("fill", (spec.fill*.9, spec.fill*1.1)),
                        ("pc_thickness_nm", (max(90, spec.pc_thickness_nm-25), spec.pc_thickness_nm+25)),
                        ("aspect", (max(1, spec.aspect-.08), spec.aspect+.08)),
                        ("main_angle_deg", (spec.main_angle_deg-10, spec.main_angle_deg+10)),
                        ("aperture", ("circle", "hexagon", "square")),
                        ("radius_um", (100., 125., 175.))):
        for value in values:
            if getattr(spec, key) == value: continue
            items.append(replace(spec, name=f"local_{start+len(items):04d}", **{key: value}))
    if spec.satellites:
        for key, values in (("asymmetry", (.05, .15, .30)), ("distance", (spec.distance-.03, spec.distance+.03)),
                            ("main_share", (max(.15, spec.main_share-.08), min(.8, spec.main_share+.08)))):
            for value in values:
                items.append(replace(spec, name=f"local_{start+len(items):04d}", **{key: value}))
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "finite", "local", "validate", "robustness", "spectrum", "retune", "highmesh", "fullcoarse"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--spec-json", type=Path,
        help="Portable JSON containing a designs list of Motif fields; no historical results required")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--cells", type=int, default=9)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--dz", type=float, default=4.)
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--catalogue", choices=("broad", "symmetry", "balanced"), default="broad")
    args = parser.parse_args(argv)
    if args.spec_json is not None and args.input is not None:
        parser.error("Choose either --spec-json or --input")
    output = prepare_output(args.output or default_output(ROOT, "optimization_20261021/triangular/"+args.stage), ROOT)
    write_provenance(output, ROOT)
    (output/"executed_script.py").write_bytes(Path(__file__).read_bytes())
    json_write(output/"run_plan.json", {"args": vars(args) | {"input": str(args.input), "spec_json": str(args.spec_json), "output": str(output)},
        "thread_limit": 1, "gain_definition": "(2*alpha_field_cm+alpha_internal_cm)/(Gamma*n_active/n_eff)",
        "eta_rad_definition": "2*local_radiation_field_loss/(2*total_field_loss+internal_intensity_loss)",
        "beam_window": "same requested +/-1 degree, 161 samples, vector fields, checked against Nyquist",
        "spectrum_scope": "all distinct poles found near each Gamma edge; complete high-order lateral spectrum not certified",
        "minimum_required_radiation_fraction": .01, "balanced_radiation_fraction": .05})
    rows = []
    with threadpool_limits(limits=1):
        if args.spec_json is not None:
            if args.stage == "local":
                raise ValueError("local stage expands an evaluated population; use --input")
            data=json.loads(args.spec_json.read_text(encoding="utf-8"))
            configured=data["designs"] if isinstance(data,dict) else data
            specs=[Motif(**row) for row in configured]
            if args.names: specs=[s for s in specs if s.name in args.names]
            if args.limit>0: specs=specs[:args.limit]
            if not specs: raise ValueError("No configured designs selected")
            (output/"input_configuration.json").write_bytes(args.spec_json.read_bytes())
            json_write(output/"input_configuration_provenance.json",{
                "path":str(args.spec_json.resolve()),
                "sha256":hashlib.sha256(args.spec_json.read_bytes()).hexdigest(),
                "scope":"Motif geometry and aperture from portable configuration; tracked workflow supplies the common physical stack and gain model"})
        elif args.stage == "screen":
            specs = {"broad": enumerate_motifs, "symmetry": enumerate_symmetry_motifs,
                     "balanced": enumerate_balanced_motifs}[args.catalogue]()
            if args.limit > 0: specs = specs[:args.limit]
        else:
            if args.input is None: raise ValueError("--input required after screen")
            source = json.loads((args.input/"candidates.json").read_text(encoding="utf-8"))
            if args.names: source = [r for r in source if r["name"] in args.names]
            if args.stage == "finite":
                specs = [Motif(**r["spec"]) for r in select_screen(source, args.limit)]
            elif args.stage == "local":
                parents = sorted([r for r in source if r.get("status") == "finite_solved"], key=rank_score)[:args.limit]
                specs = []
                for parent in parents: specs.extend(local_variants(Motif(**parent["spec"]), len(specs)))
            else:
                specs = [Motif(**r["spec"]) for r in sorted(source, key=rank_score)[:args.limit]]
        json_write(output/"input_specs.json", [asdict(s) for s in specs])
        for index, spec in enumerate(specs):
            start = time.perf_counter()
            try:
                if args.stage == "screen":
                    row = screen_one(spec)
                elif args.stage == "fullcoarse":
                    row = evaluate_finite(spec, args.cells, 10, 2., 18,
                        output/spec.name, full_dense=True)
                    row["single_mode_status"] = "complete_coarse_discrete_spectrum_only_requires_fine_grid_verification"
                elif args.stage == "highmesh":
                    trials = []
                    for cells in (27, 31, 35):
                        trial = evaluate_finite(spec, cells, 10, 2., 24,
                            output/spec.name/f"N{cells}")
                        trials.append(trial)
                        json_write(output/spec.name/"highmesh_progress.json", trials)
                    row = trials[-1].copy()
                    row["highmesh_trials"] = trials[:-1]
                    sensitivities = []
                    for rank in (0, 1):
                        values = np.asarray([r["spectrum"][rank]["alpha_per_m"]/100 for r in trials])
                        _, sensitivity, status = staggered_grid_diagnostics(1/np.asarray((27,31,35)), values)
                        sensitivities.append({"pole_rank": rank, "grid_cm": sensitivity, "mesh_status": status})
                    row["lowest_two_grid_sensitivities"] = sensitivities
                    row["combined_lowest_two_grid_sensitivity_cm"] = sum(s["grid_cm"] for s in sensitivities)
                    row["single_mode_status"] = "highmesh_local_spectrum_only_combine_with_wide_spectrum_and_D_dz_validation"
                elif args.stage == "retune":
                    trials = []
                    tuned = replace(spec, name=spec.name+"_950nm")
                    for iteration in range(4):
                        trial = evaluate_finite(tuned, 15, 10, 2., 18,
                            output/tuned.name/f"retune_{iteration}")
                        trials.append(trial)
                        error = trial["frozen_CWT_pole_wavelength_nm"]-950.
                        if abs(error) < .01: break
                        period = parameters(tuned).LATTICE_CONSTANT_NM
                        tuned = replace(tuned, lattice_nm=period*950/trial["frozen_CWT_pole_wavelength_nm"])
                    row = evaluate_finite(tuned, 23, 10, 2., 24,
                        output/tuned.name/"N23", extra_targets=17)
                    row["retuning_trials"] = trials
                    row["retuning_scope"] = "Period redesigned and full coupling/vertical/finite/geometrical calculations repeated; wavelength from frozen-CWT group-index conversion, not a fixed-frequency six-wave closure"
                elif args.stage == "spectrum":
                    dense = evaluate_finite(spec, 9, 10, 2., 18,
                        output/spec.name/"full_dense_N9", full_dense=True)
                    wide = evaluate_finite(spec, args.cells, 10, 2., 24,
                        output/spec.name/f"wide_N{args.cells}", extra_targets=17)
                    row = wide
                    row["complete_coarse_spectrum"] = dense
                    row["single_mode_status"] = "full_coarse_discrete_spectrum_and_finer_wide_shift_search_not_exhaustive_fine_spectrum"
                elif args.stage == "robustness":
                    candidate = make_candidate(spec)
                    p = parameters(spec)
                    trials = []
                    for index_perturbation, (label, edge_bias, placement) in enumerate((
                        ("nominal", 0., False), ("etch_minus_3nm", -3., False),
                        ("etch_plus_3nm", 3., False), ("bounded_placement_3nm", 0., True),
                        ("etch_plus_placement_3nm", 3., True))):
                        holes = perturb_triangular_inclusions(candidate.inclusions, p.LATTICE_CONSTANT_NM,
                            np.random.default_rng(20261021), etch_bias_nm=edge_bias)
                        if placement:
                            shifted = []
                            for hole_index, hole in enumerate(holes):
                                phase = .73+2.399963229728653*hole_index
                                shift = np.linalg.solve(TRIANGULAR_DIRECT,
                                    3/p.LATTICE_CONSTANT_NM*np.asarray((np.cos(phase), np.sin(phase))))
                                if isinstance(hole, TriangularEllipse):
                                    shifted.append(replace(hole, center_fractional=tuple(np.asarray(hole.center_fractional)+shift)))
                                else:
                                    shifted.append(replace(hole, vertices_fractional=tuple(map(tuple, np.asarray(hole.vertices_fractional)+shift))))
                            holes = tuple(shifted)
                        perturbed = TriangularCandidate(spec.name+"_"+label, holes)
                        trial = evaluate_finite(spec, args.cells, args.order, args.dz, 18,
                            output/spec.name/label, candidate_override=perturbed)
                        trial["perturbation"] = label
                        trial["actual_inclusions"] = [asdict(hole) for hole in holes]
                        # Uniform internal intensity loss is external to this
                        # passive pole problem, so doubling it changes the gain
                        # requirement algebraically, not the retained eigenfield.
                        p2 = replace(p, INTERNAL_LOSS_CM=2*p.INTERNAL_LOSS_CM)
                        double = threshold_audit(p2, trial["alpha_field_cm"]*100,
                            aperture_area_m2(spec.aperture, spec.radius_um), trial["Gamma"], trial["effective_index"])
                        trial["double_internal_loss_audit"] = double
                        trial["double_internal_loss_eta_rad"] = 2*trial["alpha_radiation_field_cm"]/(2*trial["alpha_field_cm"]+p2.INTERNAL_LOSS_CM)
                        trials.append(trial)
                        json_write(output/spec.name/"robustness_progress.json", trials)
                    row = trials[0].copy()
                    row["perturbation_trials"] = trials
                    row["single_mode_status"] = "bounded_geometry_examples_not_random_device_disorder_or_global_worst_case"
                elif args.stage == "validate":
                    # Matched refinements: separate D, dz and lateral changes.
                    trials = []
                    for order, dz, cells in ((8, 3., 11), (10, 3., 11), (10, 2., 11),
                                             (10, 2., 15), (10, 2., 19), (10, 2., 23)):
                        trial = evaluate_finite(spec, cells, order, dz, 18,
                            output/spec.name/f"D{order}_dz{dz:g}_N{cells}")
                        trials.append(trial)
                        json_write(output/spec.name/"validation_progress.json", trials)
                    row = trials[-1]
                    row["validation_trials"] = trials[:-1]
                    alpha = np.asarray([r["alpha_field_cm"] for r in trials[-3:]])
                    _, mesh_sensitivity, mesh_status = staggered_grid_diagnostics(1/np.asarray((15, 19, 23)), alpha)
                    row["grid_sensitivity_cm"] = mesh_sensitivity
                    row["mesh_status"] = mesh_status
                    sensitivities = []
                    for rank in (0, 1):
                        values = np.asarray([r["spectrum"][rank]["alpha_per_m"]/100 for r in trials[-3:]])
                        _, lateral_sensitivity, _ = staggered_grid_diagnostics(1/np.asarray((15, 19, 23)), values)
                        truncation_sensitivity = abs(trials[1]["spectrum"][rank]["alpha_per_m"]-trials[0]["spectrum"][rank]["alpha_per_m"])/100
                        vertical_sensitivity = abs(trials[2]["spectrum"][rank]["alpha_per_m"]-trials[1]["spectrum"][rank]["alpha_per_m"])/100
                        sensitivities.append({"pole_rank": rank, "grid_cm": lateral_sensitivity,
                            "D8_D10_cm": truncation_sensitivity, "dz3_dz2_cm": vertical_sensitivity,
                            "combined_cm": lateral_sensitivity+truncation_sensitivity+vertical_sensitivity})
                    row["lowest_two_numerical_sensitivities"] = sensitivities
                    uncertainty = sum(s["combined_cm"] for s in sensitivities)
                    row["combined_lowest_two_sensitivity_cm"] = uncertainty
                    row["ordering_resolved_on_tested_discretizations"] = bool(row["field_loss_gap_cm"] > uncertainty)
                    row["single_mode_status"] = ("lowest_found_pole_ordering_separated_local_spectrum_only"
                        if row["ordering_resolved_on_tested_discretizations"] else "lowest_pole_ordering_unresolved")
                else:
                    if not fabrication(spec)[0]["margin_pass"]:
                        row = {"name": spec.name, "spec": asdict(spec), "status": "geometry_rejected"}
                    else:
                        row = evaluate_finite(spec, args.cells, args.order, args.dz, 10, output/spec.name)
            except Exception as error:
                row = {"name": spec.name, "spec": asdict(spec), "status": "failed", "exception": repr(error)}
            row["elapsed_seconds"] = time.perf_counter()-start
            rows.append(row)
            json_write(output/"candidates.json", rows)
            write_csv(output/"candidates.csv", rows)
            print(f"[{index+1}/{len(specs)}] {spec.name}: {row['status']}; {row.get('g_th_cm', row.get('lower_bound_gth_cm', ''))} {row['elapsed_seconds']:.2f}s", flush=True)
        solved = [r for r in rows if r.get("status") == "finite_solved"]
        if solved:
            objectives = np.asarray([[r["g_th_cm"], r["ellipticity"]-1,
                1-r["center_to_peak"], r["radial_anisotropy_l2"], -r["eta_rad"]] for r in solved])
            mask = pareto_mask(objectives)
            json_write(output/"pareto.json", [r for r, keep in zip(solved, mask, strict=True) if keep])
        json_write(output/"summary.json", {"number_attempted": len(rows),
            "number_status": {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})},
            "interpretation": "Exploratory linear six-wave model; no nonlinear single-mode proof; failed/rejected candidates retained"})
    print(output, flush=True)


if __name__ == "__main__": main()
