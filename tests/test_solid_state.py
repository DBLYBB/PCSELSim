import numpy as np
import pytest
from dataclasses import replace

from pcselsim.solid_state import (
    YbYAGMediumConfig,
    gain_per_m,
    solve_ybyag_rates,
    threshold_excited_fraction,
    pump_limited_excited_fraction,
    pump_scan,
    threshold_pump_intensity_W_cm2,
    ybyag_energy_budget,
    ybyag_rate_rhs,
    solve_unlased_pump_propagation,
    solve_bidirectional_unlased_pump,
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


@pytest.mark.parametrize("overlap", [0.2, 0.8, 1.0])
def test_stimulated_exchange_conserves_excitation_plus_photons(overlap: float) -> None:
    config = YbYAGMediumConfig(
        confinement_factor=overlap, pump_intensity_W_cm2=0.0,
        spontaneous_emission_factor=0.0,
    )
    state = np.array([0.6, 4e18, 3e18])
    derivative = ybyag_rate_rhs(state, config, np.zeros(2))
    ion_density = config.dopant_density_cm3 * 1e6
    # In a closed cavity stimulated transitions only exchange a photon and
    # one excitation. Only the deliberately retained lifetime term remains.
    exchange = ion_density * derivative[0] + derivative[1:].sum()
    fluorescence_rate = ion_density * state[0] / (config.upper_state_lifetime_ms * 1e-3)
    np.testing.assert_allclose(exchange, -fluorescence_rate, rtol=2e-11)


def test_pump_emission_makes_high_inversion_unreachable() -> None:
    config = YbYAGMediumConfig()
    desired_fractions = np.array([0.4, 0.9])
    modal_losses = config.confinement_factor * gain_per_m(desired_fractions, config)
    intensities = threshold_pump_intensity_W_cm2(modal_losses, config)
    assert np.isfinite(intensities[0])
    assert np.isinf(intensities[1])
    np.testing.assert_allclose(pump_limited_excited_fraction(config, intensities[0]), 0.4)
    # Limit imposed by pump emission is 0.875 even for arbitrarily high pump.
    assert pump_limited_excited_fraction(config, 1e15) < 0.875


def test_steady_state_energy_budget_respects_quantum_defect() -> None:
    config = YbYAGMediumConfig(spontaneous_emission_factor=0.0, confinement_factor=0.3)
    loss = np.array([25.0])
    output = np.array([10.0])
    intensity, power, fraction, _ = pump_scan(config, loss, output, points=5)
    from pcselsim.constants import c, hbar, pi
    local_config = replace(config, pump_intensity_W_cm2=float(intensity[-1]))
    volume = config.pumped_area_um2 * 1e-12 * config.gain_thickness_um * 1e-6
    laser_energy = 2 * pi * hbar * c / (config.laser_wavelength_nm * 1e-9)
    photons = power[-1] / (laser_energy * c / config.refractive_index * output[0] * volume)
    budget = ybyag_energy_budget(np.array([fraction[-1], photons]), local_config, loss, output)
    assert abs(budget["energy_closure_residual_W"]) < 1e-9
    assert abs(budget["stored_excitation_plus_photon_energy_rate_W"]) < 1e-9
    quantum_limit = (
        config.pump_wavelength_nm / config.laser_wavelength_nm
        * budget["net_local_absorbed_pump_power_W"]
    )
    assert power[-1] <= quantum_limit


def test_output_loss_cannot_exceed_total_loss() -> None:
    with pytest.raises(ValueError, match="output_loss"):
        solve_ybyag_rates(YbYAGMediumConfig(), ("A",), np.array([10.0]), np.array([20.0]))


def test_weak_pump_propagation_recovers_beer_lambert_and_saturates() -> None:
    weak = YbYAGMediumConfig(pump_intensity_W_cm2=1e-6)
    weak_profile = solve_unlased_pump_propagation(weak, 5000.0)
    alpha = weak.dopant_density_cm3 * weak.pump_absorption_cross_section_cm2
    beer_lambert = 1.0 - np.exp(-alpha * 0.5)
    np.testing.assert_allclose(weak_profile.net_absorbed_fraction, beer_lambert, rtol=1e-6)
    strong = solve_unlased_pump_propagation(replace(weak, pump_intensity_W_cm2=2.5e4), 5000.0)
    assert 0.0 < strong.net_absorbed_fraction < weak_profile.net_absorbed_fraction
    assert np.all(np.diff(strong.intensity_W_cm2) < 0.0)
    assert np.all((strong.unlased_excited_fraction >= 0.0) & (strong.unlased_excited_fraction < 0.875))


def test_equal_power_bidirectional_pump_improves_minimum_inversion() -> None:
    config = YbYAGMediumConfig(dopant_density_cm3=1.38e21)
    single = solve_unlased_pump_propagation(config, 5000.0)
    double = solve_bidirectional_unlased_pump(config, 5000.0)
    np.testing.assert_allclose(double.intensity_W_cm2, double.intensity_W_cm2[::-1], rtol=1e-6)
    assert double.unlased_excited_fraction.min() > single.unlased_excited_fraction.min()
    assert double.intensity_W_cm2.max() <= config.pump_intensity_W_cm2
    weak = replace(config, pump_intensity_W_cm2=1e-6)
    double_weak = solve_bidirectional_unlased_pump(weak, 5000.0)
    alpha = weak.dopant_density_cm3 * weak.pump_absorption_cross_section_cm2
    np.testing.assert_allclose(double_weak.net_absorbed_fraction, 1.0 - np.exp(-alpha * 0.5), rtol=1e-6)
