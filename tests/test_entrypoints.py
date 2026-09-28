import os
import sys
from pathlib import Path

from scripts._project_bootstrap import bootstrap_project


def test_bootstrap_finds_project_from_another_working_directory(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = bootstrap_project()
    assert Path.cwd() == root
    assert (root / "configs" / "inoue2019.yaml").is_file()
    assert str(root / "src") in sys.path
    assert os.path.samefile(root, Path(__file__).resolve().parents[1])

