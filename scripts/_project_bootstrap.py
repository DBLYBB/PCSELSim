"""Make repository scripts independent of PyCharm's working directory.

This file is intentionally kept outside the installable package so the one-click
entry scripts also work before ``pip install -e .`` has been run.

中文：每个可直接点击运行的脚本都会先调用本文件，把项目根目录和 ``src`` 加入
Python 搜索路径，因此 PyCharm 工作目录设错时也不会再出现 ``No module named
pcselsim``。正式开发仍建议执行 ``pip install -e .``。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def bootstrap_project() -> Path:
    """Add ``src`` to sys.path, switch to the repository root, and return it."""
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    config_path = project_root / "configs" / "inoue2019.yaml"
    if not source_root.is_dir():
        raise RuntimeError(f"PCSELSim source directory not found: {source_root}")
    if not config_path.is_file():
        raise RuntimeError(f"PCSELSim configuration not found: {config_path}")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    os.chdir(project_root)
    return project_root

