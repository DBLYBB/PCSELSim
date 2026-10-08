"""Shared CLI and non-overwriting result folders / 公共运行与输出控制。"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from scripts._project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()


def execute_run(parameters, steps, family, runner, argv=None) -> None:
    """Pass one immutable input object to a runner; never patch module globals."""
    parser = argparse.ArgumentParser(description=parameters.RUN_TITLE)
    parser.add_argument("--only", nargs="*", choices=tuple(steps),
                        help="only selected outputs; their solver dependencies still run")
    parser.add_argument("--output", type=Path, help="empty/new output directory; default is timestamped")
    parser.add_argument("--list-steps", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="show all parameters and steps, without solving")
    parser.add_argument("--quick", action="store_true", help="coarse smoke test, NOT quantitative validation")
    args = parser.parse_args(argv)
    switches = {key: key in args.only for key in steps} if args.only is not None else dict(steps)
    if args.list_steps:
        for key, enabled in switches.items():
            print(f"{key}: {enabled}")
        return
    if args.quick:
        updates = {"CWT_TRUNCATION_ORDER": 3, "VERTICAL_STEP_NM": 8.0}
        if hasattr(parameters, "FINITE_EIGEN_GRIDS"):
            updates.update(FINITE_EIGEN_GRIDS=(9, 11, 13), TIME_GRID_POINTS=15,
                           END_TIME_NS=0.1, CURRENT_RATIOS=(1.4,),
                           LENGTH_SWEEP_GRID_POINTS=(9, 11, 13), LENGTH_SWEEP_UM=(150.0, 300.0))
        else:
            updates.update(FINITE_GRIDS=(5, 7, 9), BAND_POINTS=61,
                           UNIT_CELL_FIELD_POINTS=41, FAR_FIELD_SAMPLES=61)
        parameters = replace(parameters, **updates)
        print("QUICK: workflow smoke test only / 粗网格短时间窗，不是论文验证")
    root = PROJECT_ROOT / "results" / "pcsel" / family
    output = args.output.resolve() if args.output else root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    plan = {"family": family, "output": str(output), "steps": switches,
            "parameters": asdict(parameters), "quick_smoke_test": args.quick}
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty results: {output}; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    runner(parameters, switches, output)
    (output / "run_status.json").write_text(json.dumps({
        "status": "completed", "workflow_only": True,
        "note": "Execution success is not a claim of physical convergence or paper reproduction.",
        "completed_at": datetime.now().isoformat(),
    }, indent=2), encoding="utf-8")

