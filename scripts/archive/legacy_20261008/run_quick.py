"""One-click smoke test / 初学者首次运行入口。

Success proves installation and data flow only; the shortened time window and
coarse grid are not a paper-validation result. / 成功只表示环境与流程正常，短时间窗和
粗网格结果不能用于论文定量对照。
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
                "--quick",
                "--config",
                str(PROJECT_ROOT / "configs" / "inoue2019.yaml"),
                "--output",
                str(PROJECT_ROOT / "results" / "quick"),
            ]
        )
    )

