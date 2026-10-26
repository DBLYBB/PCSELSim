"""Guard actual-wavelength delivery and central-main hierarchy."""
from dataclasses import asdict
import json
import math
from scripts.optimization.matched300_triangular_search import ThreeHole,score
import scripts.optimization.matched300_triangular_finalize as finalize


def test_relative_output_recovery_checks_actual_lowest_wavelength(tmp_path,monkeypatch):
    monkeypatch.setattr(finalize,"ROOT",tmp_path)
    output=tmp_path/"completed";closed=output/"frequency_closed";closed.mkdir(parents=True)
    (closed/"summary.json").write_text(json.dumps({"lowest":{"wavelength_nm":950.002},"roots_closed":6}))
    (output/"period_retuning.json").write_text("[]")
    result=finalize.recover_summary(output)
    assert result["actual_target_pass"] is True
    assert result["finite_seed_path"].replace("\\","/")=="completed/final_seed/result.json"


def test_lower_off_target_competitor_is_not_certified(tmp_path,monkeypatch):
    monkeypatch.setattr(finalize,"ROOT",tmp_path)
    output=tmp_path/"rejected";closed=output/"frequency_closed";closed.mkdir(parents=True)
    (closed/"summary.json").write_text(json.dumps({"lowest":{"wavelength_nm":945.027},"roots_closed":6}))
    (output/"period_retuning.json").write_text("[]")
    assert finalize.recover_summary(output)["actual_target_pass"] is False


def test_larger_satellite_is_excluded_from_formal_ranking():
    row={"status":"finite_solved","three_hole_spec":asdict(ThreeHole("invalid-main",main_share=.38,satellite_balance=.68))}
    assert math.isinf(score(row))
