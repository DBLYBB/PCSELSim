"""One-click short smoke test for first-time PyCharm users."""

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

