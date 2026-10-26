"""Versioned result destinations and source provenance for new calculations."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path


def default_output(project_root: Path, family: str) -> Path:
    return project_root / "results" / "current" / family / datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def prepare_output(path: str | Path, project_root: Path) -> Path:
    """Create an empty run directory and protect the historical archive."""
    output = Path(path).resolve()
    archive = (project_root / "results" / "archive").resolve()
    if output == archive or archive in output.parents:
        raise ValueError("Historical results are immutable; choose results/current/<new-run>")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def source_provenance(project_root: Path) -> dict:
    """Hash the files actually available at launch, including uncommitted code."""
    files = [project_root / name for name in ("pyproject.toml", "requirements.txt")]
    for name in ("src", "scripts", "tests", "configs"):
        files.extend(p for p in (project_root / name).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".toml", ".json"}
                     and "__pycache__" not in p.parts)
    return {
        "created_at": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "source_sha256": {p.relative_to(project_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(files) if p.exists()},
        "result_scope": "new calculation after 2026-10-21 audit; execution success alone is not physical validation",
    }


def write_provenance(output: Path, project_root: Path) -> None:
    (output / "source_provenance.json").write_text(
        json.dumps(source_provenance(project_root), ensure_ascii=False, indent=2), encoding="utf-8")
