"""PyCharm-friendly entry point for the Inoue et al. 2019 reproduction."""

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

bootstrap_project()

from pcselsim.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["reproduce", *__import__("sys").argv[1:]]))

