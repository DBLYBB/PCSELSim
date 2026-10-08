"""One-click 10-ns calibrated Inoue baseline / 10 ns 标定基准入口。

The coupling eigenvalues come from ``configs/inoue2019.yaml``.  Use this run as
a time-domain regression, not a geometry-derived proof. / 耦合本征值来自配置文件，
此结果用于时域回归，不代表从真实孔形独立预测。
"""

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

PROJECT_ROOT = bootstrap_project()

from pcselsim.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "reproduce",
                "--config",
                str(PROJECT_ROOT / "configs" / "inoue2019.yaml"),
                "--output",
                str(PROJECT_ROOT / "results" / "inoue2019"),
            ]
        )
    )

