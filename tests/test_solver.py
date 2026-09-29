from dataclasses import replace

import numpy as np

from pcselsim.config import NumericsConfig, ReproductionConfig, SimulationConfig
from pcselsim.solver import TimeDomainSolver


def test_short_run_is_finite() -> None:
    config = SimulationConfig(
        numerics=NumericsConfig(
            points=9,
            end_time_ns=0.002,
            cfl=0.8,
            sample_interval_ps=0.2,
            noise=False,
        ),
        reproduction=ReproductionConfig(current_ratios=(1.05,)),
    )
    result = TimeDomainSolver(config).run(1.05)
    assert result.final_carrier_cm3.shape == (9, 9)
    assert np.all(np.isfinite(result.final_carrier_cm3))
    assert np.all(result.power_W >= 0.0)


def test_custom_signal_projection_is_normalized() -> None:
    config = SimulationConfig(
        numerics=NumericsConfig(points=9, end_time_ns=0.001, noise=False)
    )
    projection = np.asarray((1.0, 1.0j, 0.0, 0.0))
    original = projection.copy()
    solver = TimeDomainSolver(config, signal_projection=projection)
    assert solver.signal_projection.shape == (4,)
    assert np.isclose(np.linalg.norm(solver.signal_projection), 1.0)
    assert np.array_equal(projection, original)


def test_carrier_substeps_are_used_and_reported() -> None:
    config = SimulationConfig(
        numerics=NumericsConfig(
            points=9,
            end_time_ns=0.001,
            cfl=0.8,
            sample_interval_ps=0.2,
            carrier_substeps=3,
            noise=False,
        )
    )
    result = TimeDomainSolver(config).run(1.05)
    assert result.metadata["carrier_substeps"] == 3
    assert np.all(np.isfinite(result.final_carrier_cm3))

