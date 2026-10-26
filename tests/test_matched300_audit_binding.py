"""A valid old loss ledger must not silently label a newly selected mode."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

DIAGNOSTICS = Path(__file__).resolve().parents[1]/"scripts/diagnostics"
sys.path.insert(0, str(DIAGNOSTICS))
from matched300_result_provenance import (capture_case_inputs, validate_case_audit_binding,
    validate_case_selection)
from plot_matched300_threshold_budget import build_loss_rows


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf8")


def write_fields(path, name, phase=1.):
    np.savez(path, **{name+"_field": phase*np.arange(324).reshape(4,9,9).astype(complex),
        name+"_C": (-2500+25j)*np.eye(4), name+"_far_x": phase*np.ones((3,3),complex),
        name+"_far_y": np.zeros((3,3),complex), "angle_deg": np.linspace(-1,1,3)})


@pytest.fixture
def selected_case(tmp_path):
    summary_path, fields_path, geometry_path, protocol_path = [tmp_path/name for name in (
        "summary.json", "fields.npz", "geometry.json", "protocol.json")]
    mode = dict(name="TE0_pole01", wavelength_nm=1030., required_material_gain_cm1=10.3,
                delta_per_m=-2500., alpha_field_per_m=25., closed_TE_family=0)
    summary = dict(period_nm=601.2, settings=dict(order=4, dz_nm=1., square_grid=9),
        actual_winners={"edge_dual":[mode], "top_single":[mode]},
        pump_results=[dict(architecture=arch, threshold=dict(first_modes=[mode["name"]],
            first_threshold_internal_W=power)) for arch,power in (("edge_dual", .04), ("top_single", 12.))])
    case = dict(id="YbYAG_square", lattice="square", summary_path=str(summary_path),
        fields_path=str(fields_path), geometry_path=str(geometry_path), D=4, dz_nm=1., grid=9,
        nominal_aperture_um=[300,300], target_wavelength_nm=1030., wavelength_nm=1030.,
        selected_mode=mode["name"], selected_pump_architecture="edge_dual",
        field_key=mode["name"]+"_field", coupling_key=mode["name"]+"_C",
        axis_key="angle_deg", far_x_key=mode["name"]+"_far_x", far_y_key=mode["name"]+"_far_y",
        eigenvalue_per_m=[-2500.,25.], g_th_cm_inverse=10.3,
        internal_edge_pump_W=.04, internal_top_pump_W=12.)
    write_json(summary_path, summary)
    write_json(geometry_path, dict(lattice_constant_nm=601.2, nominal_aperture_um=[300,300]))
    write_json(protocol_path, dict(aperture=dict(side_um=300., shape="square"),
        YbYAG=dict(internal_intensity_loss_cm_inverse=.1, target_Bragg_wavelength_nm=1030.)))
    write_fields(fields_path, mode["name"])
    captured = capture_case_inputs(case, protocol_path=protocol_path)
    audit = dict(id=case["id"], case_fingerprint=captured["case_fingerprint"],
        case_inputs=captured["case_inputs"], rebuilt_gain_weight=.6/10.3,
        loss_budget_per_m=dict(local_amplitude_loss_per_m=10., transport_amplitude_loss_per_m=15.),
        rebuilt_up_down_power_loss_per_m=[18.,2.])
    return case, summary, protocol_path, audit


def test_valid_ledger_remains_usable_after_identical_content_rewrite(selected_case):
    case, _, protocol, audit = selected_case
    path = Path(case["fields_path"])
    path.write_bytes(path.read_bytes())
    validate_case_audit_binding(case, audit, protocol_path=protocol)
    rows = build_loss_rows([case], dict(status="passed", cases=[audit]), protocol)
    assert rows[0]["total_material_gain_cm_inverse"] == pytest.approx(10.3)


@pytest.mark.parametrize("artifact", ["summary", "fields", "geometry", "protocol"])
def test_changed_content_rejects_an_otherwise_valid_same_gain_ledger(selected_case, artifact):
    case, _, protocol, audit = selected_case
    if artifact == "fields":
        write_fields(case["fields_path"], "TE0_pole01", 1j)
    else:
        path = protocol if artifact == "protocol" else Path(case[artifact+"_path"])
        data = json.loads(path.read_text())
        data["delivery_revision"] = "new valid input with unchanged threshold scalar"
        write_json(path, data)
    # The old g and its internal/radiative/boundary sum are still exactly valid;
    # only a content-linked audit can detect the stale scientific provenance.
    assert (.1+2*10/100+2*15/100)/audit["rebuilt_gain_weight"] == pytest.approx(case["g_th_cm_inverse"])
    with pytest.raises(ValueError, match="Stale or unbound"):
        build_loss_rows([case], dict(status="passed", cases=[audit]), protocol)


@pytest.mark.parametrize("change", ["selected_mode", "wavelength", "period", "eigenvalue"])
def test_new_consistent_mode_with_same_gain_cannot_reuse_old_audit(selected_case, change):
    case, summary, protocol, audit = selected_case
    case = copy.deepcopy(case)
    mode = summary["actual_winners"]["edge_dual"][0]
    if change == "selected_mode":
        mode["name"] = "TE0_pole02"
        case["selected_mode"] = mode["name"]
        for key,suffix in (("field_key","_field"),("coupling_key","_C"),("far_x_key","_far_x"),("far_y_key","_far_y")):
            case[key] = mode["name"]+suffix
        for pump in summary["pump_results"]:
            pump["threshold"]["first_modes"] = [mode["name"]]
        write_fields(case["fields_path"], mode["name"])
    elif change == "wavelength":
        mode["wavelength_nm"] = case["wavelength_nm"] = 1030.01
    elif change == "period":
        summary["period_nm"] = 601.3
        geometry = json.loads(Path(case["geometry_path"]).read_text())
        geometry["lattice_constant_nm"] = 601.3
        write_json(Path(case["geometry_path"]), geometry)
    else:
        mode["delta_per_m"] = -2400.
        case["eigenvalue_per_m"][0] = -2400.
    summary["actual_winners"]["top_single"] = [copy.deepcopy(mode)]
    write_json(Path(case["summary_path"]), summary)
    # Source selection remains consistent; g does not encode mode identity.
    capture_case_inputs(case, protocol_path=protocol)
    with pytest.raises(ValueError, match="Stale or unbound"):
        validate_case_audit_binding(case, audit, protocol_path=protocol)


def test_unbound_legacy_audit_is_rejected(selected_case):
    case, _, protocol, audit = selected_case
    audit.pop("case_fingerprint")
    with pytest.raises(ValueError, match="Stale or unbound"):
        build_loss_rows([case], dict(status="passed", cases=[audit]), protocol)


def test_changed_internal_loss_is_not_silently_adopted(selected_case):
    case, _, protocol, audit = selected_case
    data = json.loads(protocol.read_text())
    data["YbYAG"]["internal_intensity_loss_cm_inverse"] = .2
    write_json(protocol, data)
    with pytest.raises(ValueError, match="common internal intensity loss"):
        validate_case_audit_binding(case, audit, protocol_path=protocol)


def test_duplicate_rebuild_rows_are_not_hidden_by_id_lookup(selected_case):
    case, _, protocol, audit = selected_case
    with pytest.raises(ValueError, match="distinct cases"):
        build_loss_rows([case], dict(status="passed", cases=[audit, audit]), protocol)


def test_real_but_nonprimary_semiconductor_mode_cannot_label_primary_case():
    summary = dict(lowest=dict(seed_rank=0, wavelength_nm=950., g_th_cm=60.,
        delta_per_m=-1000., alpha_field_cm=.2, I_th_A=.4))
    case = dict(id="semiconductor_triangular", field_key="root_1_fields",
        coupling_key="root_1_coupling_m", far_x_key="root_1_far_x", far_y_key="root_1_far_y",
        wavelength_nm=950.01, g_th_cm_inverse=70., eigenvalue_per_m=[-1100.,30.])
    with pytest.raises(ValueError, match="semiconductor mode and saved array key"):
        validate_case_selection(case, summary)
