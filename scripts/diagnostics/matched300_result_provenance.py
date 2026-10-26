"""Check that a comparison row selects the reported shared-pump YAG winner.

An eigenpair residual cannot detect a valid but wrongly labelled mode. These
checks bind the selected field, wavelength and threshold to the same mode and
pump architecture in the source result, before plots or rebuilds are made.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs/optimized_matched300_20261021/comparison_protocol.json"


REQUIRED_CASE_IDS = {
    "semiconductor_square", "semiconductor_triangular", "YbYAG_square", "YbYAG_triangular"
}


def validate_completed_manifest(manifest):
    cases = manifest["cases"]
    if (manifest.get("status") != "completed" or len(cases) != 4
            or {case["id"] for case in cases} != REQUIRED_CASE_IDS):
        raise ValueError("A completed manifest requires all four distinct matched cavity cases")


def _same_number(label, actual, expected, *, abs_tol=1e-9, rel_tol=2e-10):
    if not math.isclose(float(actual), float(expected), abs_tol=abs_tol, rel_tol=rel_tol):
        raise ValueError(f"Selected result provenance mismatch: {label}")


def validate_geometry_period(case, summary, geometry):
    if case["id"] == "semiconductor_square":
        expected = summary["design"]["lattice_nm"]
    elif case["id"] == "semiconductor_triangular":
        expected = summary["three_hole_spec"]["lattice_nm"]
    else:
        expected = summary["period_nm"]
    _same_number("geometry versus configured physical period", geometry["lattice_constant_nm"], expected,
                 abs_tol=1e-8, rel_tol=1e-12)


def validate_case_selection(case, summary):
    """Fail if a YAG row substitutes a cold-loss mode for its pump winner.

    Semiconductor candidate summaries use their own pole-order and gain
    records. YAG needs this extra identity link because the two pump profiles
    can select different first modes even at one fixed physical geometry.
    """
    if case["id"].startswith("semiconductor_"):
        row = summary["lowest"]
        square = case["id"] == "semiconductor_square"
        name = row["mode"] if square else f"root_{row['seed_rank']}"
        keys = {"field_key": name+("_field" if square else "_fields"),
                "coupling_key": name+"_coupling_m", "far_x_key": name+"_far_x",
                "far_y_key": name+"_far_y"}
        if case.get("selected_mode", name) != name:
            raise ValueError("Selected semiconductor mode is not the source lowest-gain mode")
        for key, expected in keys.items():
            if case[key] != expected:
                raise ValueError(f"Selected semiconductor mode and saved array key disagree: {key}")
        _same_number("semiconductor wavelength", case["wavelength_nm"], row["wavelength_nm"])
        _same_number("semiconductor material gain", case["g_th_cm_inverse"], row["g_th_cm1" if square else "g_th_cm"])
        _same_number("semiconductor eigenvalue real", case["eigenvalue_per_m"][0], row["delta_per_m"])
        _same_number("semiconductor eigenvalue imaginary", case["eigenvalue_per_m"][1], 100*row["alpha_field_cm1" if square else "alpha_field_cm"])
        if case.get("threshold_current_A") is not None:
            _same_number("semiconductor threshold current", case["threshold_current_A"], row["I_th_A"])
        if case.get("relative_gain_gap") is not None:
            _same_number("semiconductor gain gap", case["relative_gain_gap"], summary["relative_material_gap" if square else "g_th_gap_fraction"])
        if case.get("relative_current_gap") is not None:
            other = (sorted(summary["frequency_closed_modes"], key=lambda r:r["g_th_cm1"])[1]
                     if square else summary["second"])
            _same_number("semiconductor current gap", case["relative_current_gap"], other["I_th_A"]/row["I_th_A"]-1)
        return dict(selected_mode=name, selected_pump_architecture=None,
                    selection_scope="Source lowest material-gain mode within its declared spectral search")
    if not case["id"].startswith("YbYAG_"):
        raise ValueError("Unknown matched cavity case id")
    name = case["selected_mode"]
    architecture = case["selected_pump_architecture"]
    if architecture != "edge_dual":
        raise ValueError("The common optimized YAG comparison uses the actual edge_dual winner")
    winners = summary["actual_winners"][architecture]
    matching = [row for row in winners if row["name"] == name]
    if len(matching) != 1:
        raise ValueError("Selected YAG mode is not a recorded first shared-pump winner")
    mode = matching[0]
    pumps = {row["architecture"]: row["threshold"] for row in summary["pump_results"]}
    for arch, threshold in pumps.items():
        names = [row["name"] for row in summary["actual_winners"][arch]]
        if set(names) != set(threshold["first_modes"]) or len(names) != len(set(names)):
            raise ValueError(f"YAG actual winner list and first_modes disagree: {arch}")
    if name not in pumps[architecture]["first_modes"]:
        raise ValueError("YAG winner and pump-threshold first_modes disagree")
    for key, suffix in (("field_key", "_field"), ("coupling_key", "_C"),
                        ("far_x_key", "_far_x"), ("far_y_key", "_far_y")):
        if case[key] != name + suffix:
            raise ValueError(f"Selected YAG mode and saved array key disagree: {key}")
    _same_number("wavelength", case["wavelength_nm"], mode["wavelength_nm"])
    _same_number("material threshold gain", case["g_th_cm_inverse"], mode["required_material_gain_cm1"])
    _same_number("eigenvalue real part", case["eigenvalue_per_m"][0], mode["delta_per_m"])
    _same_number("eigenvalue imaginary part", case["eigenvalue_per_m"][1], mode["alpha_field_per_m"])
    if case.get("TE_family", 0) != mode["closed_TE_family"]:
        raise ValueError("Selected YAG longitudinal family disagrees with saved field")
    for arch, key in (("edge_dual", "internal_edge_pump_W"), ("top_single", "internal_top_pump_W")):
        threshold = pumps[arch]["first_threshold_internal_W"]
        if threshold is None:
            if case.get(key) is not None:
                raise ValueError(f"Unreachable YAG pump threshold labelled as finite: {arch}")
        else:
            _same_number(f"{arch} first-entry internal pump", case[key], threshold)
    if case.get("relative_gain_gap") is not None:
        # A cold-gain gap needs a separately labelled reference. It must not be
        # attached automatically to the pump-selected mode, which can rank >1.
        if case.get("relative_gain_gap_reference") != "selected_mode_is_lowest_cold_material_gain":
            raise ValueError("YAG cold-gain gap requires an explicit lowest-cold-gain reference")
    return dict(selected_mode=name, selected_pump_architecture=architecture,
                first_modes_by_architecture={arch: threshold["first_modes"] for arch, threshold in pumps.items()},
                first_wavelengths_nm_by_architecture={
                    arch: [row["wavelength_nm"] for row in rows]
                    for arch, rows in summary["actual_winners"].items()},
                threshold_scope="Each pump architecture's own first shared-inversion threshold; top and edge may select different modes")


def _resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root)/path


def capture_case_inputs(case, *, root=ROOT, protocol_path=None):
    """Content-address the exact physical inputs, independently of file mtimes.

    Paths are stored for navigation; scientific identity uses file bytes and
    selected input values, so relocating identical portable files is harmless.
    """
    if case["id"] not in REQUIRED_CASE_IDS:
        raise ValueError("Unknown matched cavity case id")
    paths = {key: _resolve(case[key+"_path"], root) for key in ("summary", "fields", "geometry")}
    paths["protocol"] = _resolve(protocol_path or DEFAULT_PROTOCOL, root)
    raw = {key:path.read_bytes() for key,path in paths.items()}
    summary, geometry, protocol = (json.loads(raw[key].decode("utf8")) for key in ("summary", "geometry", "protocol"))
    selection = validate_case_selection(case, summary)
    validate_geometry_period(case, summary, geometry)
    square = case["id"].endswith("_square")
    if case["lattice"] != ("square" if square else "triangular"):
        raise ValueError("Case id and lattice disagree")
    semiconductor = case["id"].startswith("semiconductor_")
    material = protocol["semiconductor" if semiconductor else "YbYAG"]
    internal = float(material["internal_intensity_loss_cm_inverse"])
    _same_number("common internal intensity loss", internal, 5. if semiconductor else .1, abs_tol=1e-12, rel_tol=0.)
    _same_number("common nominal aperture", protocol["aperture"]["side_um"], 300.)
    if protocol["aperture"]["shape"] != "square" or case["nominal_aperture_um"] != [300, 300]:
        raise ValueError("Comparison requires nominal 300 by300um square aperture")
    if geometry.get("nominal_aperture_um", [300,300]) != [300,300]:
        raise ValueError("Stored geometry aperture differs from comparison protocol")
    target = float(material["target_Bragg_wavelength_nm"])
    _same_number("common target wavelength", case["target_wavelength_nm"], target)
    if abs(case["wavelength_nm"]-target) > .05:
        raise ValueError("Selected actual frequency misses its common target")
    if semiconductor:
        expected_settings = (summary["D"], summary["dz_nm"], summary["grid" if square else "N"])
    else:
        settings = summary["settings"]
        expected_settings = (settings["order"], settings["dz_nm"], settings["square_grid" if square else "tri_radius_cells"])
    for label, actual, expected in zip(("D", "dz_nm", "grid"),
            (case["D"], case["dz_nm"], case["grid"]), expected_settings):
        _same_number("saved numerical setting "+label, actual, expected)
    hashes = {key+"_sha256": hashlib.sha256(value).hexdigest() for key,value in raw.items()}
    physical_keys = ("id", "lattice", "D", "dz_nm", "grid", "nominal_aperture_um",
                     "wavelength_nm", "target_wavelength_nm", "eigenvalue_per_m", "g_th_cm_inverse",
                     "field_key", "coupling_key", "axis_key", "far_x_key", "far_y_key")
    inputs = {key:case[key] for key in physical_keys}
    inputs.update(hashes, actual_period_nm=float(geometry["lattice_constant_nm"]),
                  internal_intensity_loss_cm_inverse=internal,
                  selected_mode=selection["selected_mode"],
                  selected_pump_architecture=selection["selected_pump_architecture"],
                  TE_family=case.get("TE_family", 0))
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
    return dict(summary=summary, geometry=geometry, protocol=protocol, selection=selection,
                internal_intensity_loss_cm_inverse=internal, case_inputs=inputs,
                case_fingerprint=hashlib.sha256(encoded).hexdigest(), **hashes)


def validate_case_audit_binding(case, audit, *, root=ROOT, protocol_path=None):
    """Reject a valid but stale rebuild before reusing its loss decomposition."""
    current = capture_case_inputs(case, root=root, protocol_path=protocol_path)
    if (audit.get("case_fingerprint") != current["case_fingerprint"]
            or audit.get("case_inputs") != current["case_inputs"]):
        raise ValueError(f"Stale or unbound optical audit for {case['id']}; rebuild from current selected inputs")
    return current
