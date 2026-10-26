"""Independent transfer boundary roots validate the stripe-limit diagnostic."""
import numpy as np

from scripts.diagnostics.audit_counterpropagating_stripes import stripe_pair, transfer_residual


def test_staggered_stripe_approaches_independent_continuous_transfer_root():
    coupling = np.array([[300+50j, 10000], [10000, 300+50j]])
    result = stripe_pair(coupling, 300., .1, 5., grids=(41, 81))
    exact = result["independent_continuous_transfer_pole"]
    assert exact["converged"]
    assert exact["transfer_boundary_residual"] < 1e-10
    # Mirror-symmetric +/- detuning branches have equal loss. Compare loss,
    # whose convergence is unaffected by the degenerate branch ordering.
    coarse, fine = [r["poles"][0] for r in result["discrete_runs"]]
    assert abs(fine["alpha_field_per_m"]-exact["alpha_field_per_m"]) < .3*abs(
        coarse["alpha_field_per_m"]-exact["alpha_field_per_m"])
    for record in (coarse, fine):
        assert record["eigen_residual"] < 1e-10
        assert abs(record["local_amplitude_loss_per_m"]-50.) < 1e-8
        assert abs(record["local_amplitude_loss_per_m"] +
                   record["transport_amplitude_loss_per_m"] - record["alpha_field_per_m"]) < 1e-7
        assert record["volume_numerical_diffusion_per_m"] == 0.


def test_transfer_pole_is_invariant_to_constant_basis_phase():
    coupling = np.array([[300+50j, 10000], [10000, 300+50j]])
    pole = -13528.54989714213 + 2125.383529583206j
    gauge = np.diag(np.exp(1j*np.array([.31, -1.07])))
    transformed = gauge @ coupling @ gauge.conj().T
    assert abs(transfer_residual(pole, coupling, 300e-6)) < 1e-10
    assert abs(transfer_residual(pole, transformed, 300e-6)) < 1e-10
