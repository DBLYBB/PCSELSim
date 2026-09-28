"""High-level reproduction workflow."""

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
    plot_spectra(results, output / "figure4_spectra.png", config.reproduction.spectrum_window_ns)
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

