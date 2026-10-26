from dataclasses import replace

import numpy as np
import pytest

from pcselsim.config import NumericsConfig, ReproductionConfig, SimulationConfig
from pcselsim.solver import TimeDomainSolver
from pcselsim.observables import photon_density_m3


def test_unresolved_stimulated_depletion_cannot_be_hidden_by_zero_clipping():
    config = SimulationConfig(numerics=NumericsConfig(points=9, noise=False))
    solver = TimeDomainSolver(config)
    carrier = np.full((9, 9), 2*config.carrier.transparency_density_cm3*1e6)
    field = np.full((4, 9, 9), 1e6, dtype=complex)
    with pytest.raises(FloatingPointError, match="Carrier Heun predictor became negative"):
        solver._advance_carrier(carrier, field, np.zeros_like(carrier))


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


def test_absorption_creates_carriers_below_transparency() -> None:
    config = SimulationConfig(numerics=NumericsConfig(points=9, noise=False))
    solver = TimeDomainSolver(config)
    carrier = np.zeros((9, 9))
    field = np.full((4, 9, 9), 1.0, dtype=complex)
    assert np.all(solver._carrier_rhs(carrier, field, np.zeros_like(carrier)) > 0.0)


def test_noise_injects_configured_photon_density_in_expectation() -> None:
    config = SimulationConfig(numerics=NumericsConfig(points=101, noise=True))
    solver = TimeDomainSolver(config)
    carrier = np.full((101, 101), 2e24)
    field = np.zeros((4, 101, 101), complex)
    solver._add_spontaneous_noise(field, carrier, np.random.default_rng(13))
    expected = config.carrier.spontaneous_emission_factor * carrier / (config.carrier.lifetime_ns * 1e-9) * solver.dt
    photons = photon_density_m3(field, config.carrier, config.optical)
    assert np.isclose(photons[1:-1, 1:-1].mean(), expected[1:-1, 1:-1].mean(), rtol=0.02)
    assert np.all(field[:, 0, :] == 0.0)
    assert np.all(field[:, -1, :] == 0.0)
    assert np.all(field[:, :, 0] == 0.0)
    assert np.all(field[:, :, -1] == 0.0)


def test_local_coupling_never_reinjects_constrained_incoming_wave():
    base = SimulationConfig(numerics=NumericsConfig(points=9, noise=False))
    config = replace(base, optical=replace(base.optical, internal_loss_cm=0.0,
                                          carrier_index_change=False, temporal_index_term=False))
    coupling = np.zeros((4, 4), complex)
    coupling[0, 1] = coupling[1, 0] = 20000.0
    solver = TimeDomainSolver(config, coupling_m=coupling)
    field = np.zeros((4, 9, 9), complex)
    field[1, 4, 0] = 1.0
    field[1, 4, 4] = 1.0
    carrier = np.full((9, 9), config.carrier.transparency_density_cm3*1e6)
    moved = solver._field_local_half_step(field, carrier, np.zeros_like(carrier))
    # At the left boundary Rx=0 for the whole local evolution, so Sx has no
    # partner to exchange energy with. Interior waves still couple normally.
    assert moved[0, 4, 0] == 0.0
    assert np.isclose(moved[1, 4, 0], 1.0)
    phase = 0.5*solver.vg*solver.dt*coupling[0, 1]
    assert np.isclose(moved[0, 4, 4], 1j*np.sin(phase))
    assert np.isclose(moved[1, 4, 4], np.cos(phase))


def test_noisy_run_retains_strict_open_boundaries_and_local_probes(tmp_path):
    config = SimulationConfig(numerics=NumericsConfig(points=9, end_time_ns=0.005, noise=True))
    result = TimeDomainSolver(config).run(1.4)
    field = result.final_field
    assert np.all(field[0, :, 0] == 0.0)
    assert np.all(field[1, :, -1] == 0.0)
    assert np.all(field[2, 0, :] == 0.0)
    assert np.all(field[3, -1, :] == 0.0)
    assert result.probe_signals.shape == (result.time_ns.size, 9, 4)
    assert result.probe_coordinates_um.shape == (9, 2)
    assert result.metadata["spectrum_probe_count"] == 9
    from pcselsim.io import save_result
    output = save_result(result, tmp_path/"probes.npz")
    with np.load(output, allow_pickle=False) as data:
        assert np.array_equal(data["probe_signals"], result.probe_signals)
        assert np.array_equal(data["probe_coordinates_um"], result.probe_coordinates_um)


def test_linear_coupled_propagation_converges_when_time_step_is_refined():
    """Compare against an independently exponentiated fixed-grid generator.

    Incoming rows are constrained and outgoing rows use the one-sided open
    stencil. This checks time-step convergence; it does not certify spatial
    convergence or claim second-order boundary accuracy.
    """
    from scipy.linalg import expm

    n = 33
    base = SimulationConfig()
    optical = replace(base.optical, internal_loss_cm=0.0, carrier_index_change=False,
                      temporal_index_term=False)
    coupling = np.zeros((4, 4), complex)
    coupling[0, 1] = coupling[1, 0] = 2000.0
    errors = []
    reference = None
    for cfl in (.4, .2, .1, .05):
        solver = TimeDomainSolver(replace(base, optical=optical,
            numerics=NumericsConfig(points=n, cfl=cfl, noise=False)), coupling)
        field = np.zeros((4, n, n), complex)
        field[0] = np.exp(-((solver.x+60e-6)/60e-6)**2)[None, :]
        field[1] = .3*np.exp(-((solver.x-40e-6)/65e-6)**2)[None, :]
        solver._apply_incoming_boundary(field)
        duration = .25*(solver.x[-1]-solver.x[0])/solver.vg
        if reference is None:
            generator = np.zeros((2*n, 2*n), complex)
            for channel, direction in ((0, 1), (1, -1)):
                for position in range(1, n-1):
                    generator[channel*n+position, channel*n+position-1] = direction*solver.vg/(2*solver.dx)
                    generator[channel*n+position, channel*n+position+1] = -direction*solver.vg/(2*solver.dx)
                outgoing = n-1 if channel == 0 else 0
                upstream = outgoing-1 if channel == 0 else 1
                generator[channel*n+outgoing, channel*n+outgoing] = -solver.vg/solver.dx
                generator[channel*n+outgoing, channel*n+upstream] = solver.vg/solver.dx
            for position in range(1, n-1):
                generator[position, n+position] = 1j*solver.vg*coupling[0, 1]
                generator[n+position, position] = 1j*solver.vg*coupling[1, 0]
            reference = expm(duration*generator) @ field[:2, n//2, :].ravel()
        carrier = np.full((n, n), base.carrier.transparency_density_cm3*1e6)
        carrier_rate = np.zeros((n, n))
        for _ in range(round(duration/solver.dt)):
            field = solver._field_local_half_step(field, carrier, carrier_rate)
            field = solver._advect(field)
            field = solver._field_local_half_step(field, carrier, carrier_rate)
        errors.append(np.linalg.norm(field[:2, n//2, :].ravel()-reference)/np.linalg.norm(reference))
    assert np.all(np.asarray(errors[1:]) < .6*np.asarray(errors[:-1]))
    assert errors[-1] < 1e-3

