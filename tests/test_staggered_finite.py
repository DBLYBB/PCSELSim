"""Independent boundary-value and energy checks of the Liang staggering."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import expm
from scipy.optimize import root

from pcselsim.custom_analysis import _upwind_1d
from pcselsim.finite_discretization import (assess_mode_selection, build_finite_eigenproblem,
                                          relative_residual)
from pcselsim.triangular_finite import (_directed_upwind, build_device_grid,
                                       polarization_axis_angle, TriangularFiniteMode,
                                       triangular_vector_far_field, write_triangular_finite_table)


def _exact_uniform_dfb_eigenvalue():
    # Independent continuum two-wave problem, R(0)=S(1)=0, kappa=2.
    # Integrate the constant-coefficient ODE with its exact matrix exponential.
    def outgoing_boundary(coordinates):
        value = complex(*coordinates)
        derivative = np.asarray(((-1j*value, 2j), (-2j, 1j*value)))
        boundary = expm(derivative)[1, 1]
        return boundary.real, boundary.imag
    answer = root(outgoing_boundary, (3.4, 1.0), tol=1e-11)
    assert answer.success
    assert np.linalg.norm(answer.fun) < 1e-10
    return complex(*answer.x)


def test_staggered_converges_to_independent_continuum_dfb_without_upwind_diffusion():
    reference = _exact_uniform_dfb_eigenvalue()
    errors = {method: [] for method in ("staggered", "legacy_upwind")}
    for count in (8, 16, 32):
        step = 1.0/count
        derivatives = (_upwind_1d(count, step, True), _upwind_1d(count, step, False))
        for method in errors:
            problem = build_finite_eigenproblem(np.asarray(((0., 2.), (2., 0.))), derivatives, step, method)
            values, fields = problem.near(reference, 4, np.ones(2*count, dtype=complex))
            index = int(np.argmin(abs(values-reference)))
            value, field = values[index], fields[:, index]
            assert value.imag > 0
            assert relative_residual(problem.center_operator(), field, value) < 1e-10
            budget = problem.loss_budget(field)
            assert budget["local_amplitude_loss_per_m"] == pytest.approx(0., abs=1e-12)
            assert budget["transport_amplitude_loss_per_m"] == pytest.approx(value.imag, rel=1e-10)
            errors[method].append(abs(value-reference))
    # The analytic continuum reference verifies actual accuracy, not just
    # agreement of two implementations of the same difference equation.
    assert errors["staggered"][0]/errors["staggered"][1] > 3.5
    assert errors["staggered"][1]/errors["staggered"][2] > 3.5
    assert errors["staggered"][-1] < errors["legacy_upwind"][-1]/10


@pytest.mark.parametrize("shape", ["circle", "hexagon", "square"])
def test_six_direction_mass_transport_has_only_passive_outgoing_boundary_flux(shape):
    grid = build_device_grid(3, 30., shape)
    identity = sparse.identity(grid.points, format="csr")
    for step, positive in zip(((1, 0), (1, 0), (0, 1), (0, 1), (1, 1), (1, 1)),
                              (True, False)*3, strict=True):
        derivative = _directed_upwind(grid, step, positive)
        shift = identity-grid.spacing_m*derivative
        average = .5*(identity+shift)
        boundary = (identity-shift.conj().T@shift)/grid.spacing_m
        energy = average.conj().T@derivative+derivative.conj().T@average
        assert np.allclose(energy.toarray(), boundary.toarray(), atol=1e-8)
        # All interior rows vanish: no hidden upwind volume loss.
        assert np.allclose(boundary.toarray(), np.diag(boundary.diagonal()))
        assert np.min(boundary.diagonal()) >= -1e-8
        assert np.count_nonzero(boundary.diagonal() == 0) > grid.points//2


def test_polarization_axis_is_independent_of_common_field_normalization():
    x = np.asarray([1., 1.+2j])
    y = np.asarray([2., 3.-1j])
    axes = polarization_axis_angle(x, y)
    assert axes[0] == pytest.approx(np.arctan2(2., 1.))
    assert np.allclose(axes, polarization_axis_angle(3e-8*np.exp(.7j)*x, 3e-8*np.exp(.7j)*y))


def test_unresolved_family_order_is_not_promoted_to_lasing_claim():
    def mode(name, loss, sensitivity):
        return SimpleNamespace(name=name, alpha_per_m=loss,
                               extrapolation_uncertainty_per_m=sensitivity,
                               extrapolation_status="mesh_checked")
    uncertain = {"A": mode("A", 100., 8.), "B": mode("B", 105., 7.)}
    assert not assess_mode_selection(uncertain)["ordering_resolved_on_tested_grids"]
    uncertain["B"].alpha_per_m = 140.
    assert assess_mode_selection(uncertain)["ordering_resolved_on_tested_grids"]


def test_csv_uses_the_same_cached_far_fields_and_persists_metric_domain(tmp_path):
    import csv
    grid = build_device_grid(3, 10.)
    radiation = np.ones(grid.points, dtype=complex)
    names = ("A", "B1", "B2", "C", "D1", "D2")
    modes = {name: TriangularFiniteMode(name, 0., 1., np.ones((6, grid.points)),
              1., grid, radiation, np.zeros_like(radiation)) for name in names}
    cached = {name: triangular_vector_far_field(mode, 1000., .7, 31)
              for name, mode in modes.items()}
    path = tmp_path/"modes.csv"
    write_triangular_finite_table(modes, SimpleNamespace(bragg_wavelength_nm=1000.),
                                path, view_deg=3., samples=81, far_fields=cached)
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    for row in rows:
        far = cached[row["mode"]]
        assert float(row["evaluated_half_window_deg"]) == .7
        assert int(row["angular_samples"]) == 31
        assert float(row["full_rms_divergence_deg"]) == far.full_rms_divergence_deg
        assert row["energy_normalization"] == "evaluated_angular_window"
        assert row["moment_reference"] == "surface_normal_radial_second_moment"


def test_circular_characteristic_mask_contains_the_full_circle_and_c6_rotations():
    grid = build_device_grid(9, 30., "circle")
    # This in-circle point was previously clipped by the rectangular j<=N bound.
    assert (10, 5) in grid.index
    direct = np.asarray(((.5, .5), (-np.sqrt(3)/2, np.sqrt(3)/2)))
    rotation = np.asarray(((.5, -np.sqrt(3)/2), (np.sqrt(3)/2, .5)))
    axial_rotation = np.linalg.solve(direct, rotation@direct)
    for point in grid.index:
        rotated = tuple(np.rint(axial_rotation@point).astype(int))
        assert rotated in grid.index


def test_small_dense_generalized_spectrum_matches_sparse_shift_invert(monkeypatch):
    import pcselsim.finite_discretization as discretization
    count = 16
    step = 1/count
    derivatives = (_upwind_1d(count, step, True), _upwind_1d(count, step, False))
    coupling = np.asarray(((0., 2.), (2., 0.)))
    dense = build_finite_eigenproblem(coupling, derivatives, step, "staggered")
    reference = _exact_uniform_dfb_eigenvalue()
    values, fields = dense.near(reference, 4, np.ones(2*count, dtype=complex))
    assert getattr(dense, "_dense_spectrum", None) is not None
    assert np.isfinite(values).all() and np.min(values.imag) > 0
    for value, field in zip(values, fields.T, strict=True):
        assert relative_residual(dense.center_operator(), field, value) < 1e-11
    # Force the independent sparse algorithm on the same K,M and compare poles.
    monkeypatch.setattr(discretization, "DENSE_EIGEN_CUTOFF", 0)
    sparse_problem = build_finite_eigenproblem(coupling, derivatives, step, "staggered")
    sparse_values, _ = sparse_problem.near(reference, 4, np.ones(2*count, dtype=complex))
    assert max(min(abs(value-sparse_values)) for value in values) < 1e-9
