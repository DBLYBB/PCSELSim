"""Installed command-line interface / 安装后的命令行入口。

It validates YAML and delegates to the same workflow used by PyCharm scripts;
there is no second solver hidden here. / 本文件只解析命令并调用统一工作流，不包含
另一套物理模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .reproduce import reproduce
from .run_output import default_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pcselsim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("reproduce", help="run the legacy calibrated-C Inoue comparison (not a geometry prediction)")
    run.add_argument("--config", default="configs/inoue2019.yaml")
    run.add_argument("--output", default=None)
    run.add_argument("--quick", action="store_true", help="short smoke test")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reproduce":
        config = load_config(Path(args.config))
        if args.quick:
            config = config.quick()
        project_root = Path(__file__).resolve().parents[2]
        reproduce(config, args.output or default_output(project_root, "legacy/inoue2019"))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

