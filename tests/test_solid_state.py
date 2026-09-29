import numpy as np

from pcselsim.solid_state import (
    YbYAGMediumConfig,
    gain_per_m,
    solve_ybyag_rates,
    threshold_excited_fraction,
)


def test_ybyag_gain_is_monotonic_and_has_reabsorption() -> None:
    config = YbYAGMediumConfig()
    gain = gain_per_m(np.array([0.0, 0.5, 1.0]), config)
    assert np.all(np.diff(gain) > 0.0)
    assert gain[0] < 0.0 < gain[-1]


def test_ybyag_threshold_increases_with_loss() -> None:
    config = YbYAGMediumConfig()
    threshold = threshold_excited_fraction(np.array([50.0, 150.0]), config)
    assert 0.0 < threshold[0] < threshold[1]


def test_short_ybyag_rate_solution_is_finite() -> None:
    config = YbYAGMediumConfig(end_time_ms=0.02, samples=12)
    result = solve_ybyag_rates(
        config,
        ("A", "B"),
        total_loss_per_m=np.array([65.0, 120.0]),
        output_loss_per_m=np.array([40.0, 80.0]),
    )
    assert result.photon_density_m3.shape == (2, 12)
    assert np.isfinite(result.excited_fraction).all()
    assert np.isfinite(result.output_power_W).all()
    assert np.all(result.output_power_W >= 0.0)
