import json
from pathlib import Path

import pytest

from pcselsim.run_output import default_output, prepare_output, source_provenance
from scripts._run_controls import execute_run
from scripts import run_square_double_hole as square


def test_new_default_and_archive_protection(tmp_path):
    assert default_output(tmp_path, "pcsel/test").is_relative_to(tmp_path / "results/current/pcsel/test")
    with pytest.raises(ValueError, match="immutable"):
        prepare_output(tmp_path / "results/archive/old/new", tmp_path)
    assert not (tmp_path / "results/archive/old/new").exists()


def test_physical_json_configs_are_bound_to_source_provenance(tmp_path):
    import hashlib
    config = tmp_path/'configs'/'geometry.json'
    config.parent.mkdir()
    config.write_text('{"period_nm": 601.2}', encoding='utf-8')
    snapshot = source_provenance(tmp_path)['source_sha256']
    assert snapshot['configs/geometry.json'] == hashlib.sha256(config.read_bytes()).hexdigest()


def test_failed_run_records_failure_without_destroying_inputs(tmp_path):
    def failing(*args):
        raise RuntimeError("synthetic solver failure")
    with pytest.raises(RuntimeError):
        execute_run(square.build_parameters(), square.STEPS, "test", failing,
                    ["--output", str(tmp_path / "failed")])
    output = tmp_path / "failed"
    assert json.loads((output / "run_status.json").read_text())["status"] == "failed"
    assert (output / "run_plan.json").exists()
    assert (output / "source_provenance.json").exists()
