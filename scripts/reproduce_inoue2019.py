"""PyCharm entry for the calibrated Inoue 2019 baseline / 标定基准入口。

This is the legacy calibrated-C workflow, not the geometry-derived custom
workflow. / 本脚本不是孔形推导后端；若要修改晶格并预测 C，请运行
``run_custom_semiconductor_pcsel.py``。
"""

if __package__:
    from ._project_bootstrap import bootstrap_project
else:
    from _project_bootstrap import bootstrap_project

bootstrap_project()

from pcselsim.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["reproduce", *__import__("sys").argv[1:]]))

