"""One-click full 10-ns Inoue 2019 baseline for PyCharm users."""

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

