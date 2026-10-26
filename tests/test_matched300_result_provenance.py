"""Scientific result assembly must bind a pump winner to its own eigenfield."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("matched300_provenance",
    Path(__file__).resolve().parents[1] / "scripts/diagnostics/matched300_result_provenance.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example():
    # Cold-gain mode 0 and first pump mode 1 intentionally differ. The top
    # architecture also selects a different mode; its threshold remains valid.
    winner = dict(name="TE0_pole01", wavelength_nm=1030., required_material_gain_cm1=10.3,
                  delta_per_m=-2500., alpha_field_per_m=25., closed_TE_family=0)
    summary = dict(actual_winners={"edge_dual": [winner],
                   "top_single": [{**winner, "name": "TE0_pole00", "wavelength_nm": 1028.}]}, pump_results=[
        dict(architecture="edge_dual", threshold=dict(first_modes=[winner["name"]], first_threshold_internal_W=.04)),
        dict(architecture="top_single", threshold=dict(first_modes=["TE0_pole00"], first_threshold_internal_W=12.))])
    case = dict(id="YbYAG_square", selected_mode=winner["name"], selected_pump_architecture="edge_dual",
                field_key="TE0_pole01_field", coupling_key="TE0_pole01_C", far_x_key="TE0_pole01_far_x",
                far_y_key="TE0_pole01_far_y", wavelength_nm=1030., g_th_cm_inverse=10.3,
                eigenvalue_per_m=[-2500., 25.], internal_edge_pump_W=.04, internal_top_pump_W=12.)
    return case, summary


def test_pump_selected_mode_may_differ_from_lowest_cold_gain():
    case, summary = example()
    MODULE.validate_case_selection(case, summary)


@pytest.mark.parametrize("changes", [
    dict(selected_mode="TE0_pole00"), dict(field_key="TE0_pole00_field"),
    dict(wavelength_nm=1029.), dict(g_th_cm_inverse=9.),
    dict(internal_edge_pump_W=.035), dict(eigenvalue_per_m=[-2000.,25.]),
    dict(relative_gain_gap=.2), dict(selected_pump_architecture="top_single")])
def test_valid_but_mislabeled_optical_result_is_rejected(changes):
    case, summary = example()
    case.update(changes)
    with pytest.raises(ValueError):
        MODULE.validate_case_selection(case, summary)


def test_disagreement_between_winner_and_threshold_is_rejected():
    case, summary = example()
    summary["pump_results"][0]["threshold"]["first_modes"] = ["TE0_pole00"]
    with pytest.raises(ValueError):
        MODULE.validate_case_selection(case, summary)


def test_all_case_gate_rejects_duplicate_even_when_count_is_four():
    cases = [{"id": name} for name in sorted(MODULE.REQUIRED_CASE_IDS)]
    manifest = dict(status="completed", cases=cases)
    MODULE.validate_completed_manifest(manifest)
    invalid = copy.deepcopy(manifest)
    invalid["cases"][-1] = invalid["cases"][0]
    with pytest.raises(ValueError):
        MODULE.validate_completed_manifest(invalid)


def test_geometry_from_retargeted_trial_cannot_label_previous_trial_fields():
    case = dict(id="YbYAG_square")
    summary = dict(period_nm=601.2)
    MODULE.validate_geometry_period(case, summary, dict(lattice_constant_nm=601.2))
    with pytest.raises(ValueError):
        MODULE.validate_geometry_period(case, summary, dict(lattice_constant_nm=600.8))
