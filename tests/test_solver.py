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

