"""High-level legacy Inoue reproduction workflow.

This entry uses the calibrated A/B/C/D coupling backend from ``coupling.py``.
It is retained for regression and qualitative time-domain comparison.  New
geometry-predictive work should run ``run_custom_semiconductor_pcsel.py``.

本入口使用 ``coupling.py`` 中人工给定带边本征值的标定矩阵，适合回归测试和定性
时域对照。需要从孔形预测耦合、阈值及远场时，应运行自定义半导体主程序。
"""

from __future__ import annotations

import csv
from pathlib import Path

from .config import SimulationConfig
from .io import save_result
from .plotting import plot_spatial, plot_spectra, plot_transient
from .solver import SimulationResult, TimeDomainSolver


def reproduce(config: SimulationConfig, output_dir: str | Path) -> list[SimulationResult]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    solver = TimeDomainSolver(config)
    results: list[SimulationResult] = []
    for ratio in config.reproduction.current_ratios:
        print(f"[PCSELSim] simulating I/Ith={ratio:.2f}", flush=True)
        result = solver.run(ratio)
        results.append(result)
        tag = f"I_{ratio:.2f}".replace(".", "p")
        save_result(result, output / f"{tag}.npz")
        plot_transient(result, output / f"{tag}_transient.png")
        plot_spatial(result, output / f"{tag}_spatial.png")
    plot_spectra(
        results,
        output / "figure4_spectra.png",
        config.reproduction.spectrum_window_ns,
        center_wavelength_nm=config.optical.wavelength_nm,
    )
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["I_over_Ith", "final_center_N_cm-3", "final_power_W", "max_power_W"])
        for result in results:
            writer.writerow(
                [
                    result.current_ratio,
                    result.center_carrier_cm3[-1],
                    result.power_W[-1],
                    result.power_W.max(),
                ]
            )
    return results

