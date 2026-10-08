"""Independent analytic and flux checks for planar multilayer source optics."""

import numpy as np
import pytest

from pcselsim.stratified_optics import (
    StratifiedLayer, StratifiedStack, local_green_kernel, quarter_wave_dbr, solve_stack,
)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
@pytest.mark.parametrize("angle", [0.0, 23.0, 61.0])
@pytest.mark.parametrize("side", ["bottom", "top"])
def test_lossless_oblique_energy_conservation(polarization, angle, side):
    stack = StratifiedStack((StratifiedLayer("film1", 170.0, 2.1),
                             StratifiedLayer("film2", 380.0, 1.45)), 1.5, 1.0)
    result = solve_stack(stack, 1030.0, polarization=polarization, angle_deg=angle,
                         incident_side=side)
    assert result.R + result.T == pytest.approx(1.0, abs=2e-14)
    assert result.A == pytest.approx(0.0, abs=2e-14)


@pytest.mark.parametrize("side", ["bottom", "top"])
def test_interface_fresnel_normal(side):
    result = solve_stack(StratifiedStack((), 1.0, 1.5), 1000.0, incident_side=side)
    assert result.R == pytest.approx(0.04)
    assert result.T == pytest.approx(0.96)
    assert result.r == pytest.approx(-0.2 if side == "bottom" else 0.2)
    assert result.t == pytest.approx(0.8 if side == "bottom" else 1.2)


def test_tm_brewster_and_te_nonzero():
    stack = StratifiedStack((), 1.0, 1.5)
    brewster = np.rad2deg(np.arctan(1.5))
    tm = solve_stack(stack, 1000.0, angle_deg=brewster, polarization="TM")
    te = solve_stack(stack, 1000.0, angle_deg=brewster, polarization="TE")
    assert abs(tm.r) < 1e-14
    assert tm.T == pytest.approx(1.0)
    assert te.R > 0.14


def test_single_film_independent_airy_phase():
    n0, n1, n2, thickness, wavelength = 1.2, 2.0, 1.5, 217.0, 1030.0
    stack = StratifiedStack((StratifiedLayer("film", thickness, n1),), n0, n2)
    result = solve_stack(stack, wavelength)
    r01, r12 = (n0 - n1)/(n0 + n1), (n1 - n2)/(n1 + n2)
    phase = np.exp(-2j*np.pi*n1*thickness/wavelength)
    denominator = 1.0 + r01*r12*phase**2
    assert result.r == pytest.approx((r01 + r12*phase**2)/denominator, abs=1e-14)
    assert result.t == pytest.approx(4*n0*n1/(n0+n1)/(n1+n2)*phase/denominator, abs=1e-14)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_absorption_passive_convention(polarization):
    stack = StratifiedStack((StratifiedLayer("absorber", 600.0, 2.0-0.15j),), 1.0, 1.0)
    result = solve_stack(stack, 1000.0, angle_deg=33.0, polarization=polarization)
    assert result.R >= 0.0 and result.T >= 0.0
    assert 0.1 < result.A < 1.0


def test_finite_dbr_stopband_and_nonzero_bottom_transmission():
    layers = quarter_wave_dbr(1030.0, 2.1, 1.45, 10, first_layer="low")
    assert len(layers) == 20
    assert layers[-1].refractive_index == 2.1
    assert all(layer.thickness_nm*layer.refractive_index == pytest.approx(1030.0/4) for layer in layers)
    stack = StratifiedStack(layers, 1.45, 1.45)
    centre = solve_stack(stack, 1030.0, incident_side="top")
    outside = solve_stack(stack, 1500.0, incident_side="top")
    assert centre.R > 0.995
    assert centre.T > 0.0  # A finite passive DBR is not a perfect one-sided mirror.
    assert outside.R < centre.R - 0.4
    assert centre.R + centre.T == pytest.approx(1.0, abs=3e-14)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
@pytest.mark.parametrize("qfactor", [0.0, 0.8, 2.5])
def test_uniform_green_recovers_liang_and_weighted_tm(polarization, qfactor):
    n, wavelength = 1.8, 1030.0
    stack = StratifiedStack((StratifiedLayer("uniform", 500.0, n),), n, n)
    k0 = 2*np.pi/(wavelength*1e-9)
    kernel = local_green_kernel(stack, 0, wavelength, polarization=polarization,
                                q_parallel_per_m=qfactor*k0)
    z = np.linspace(0.0, 500e-9, 9)
    p = 1.0 if polarization == "TE" else 1/n**2
    k = np.sqrt((n*k0)**2-(qfactor*k0)**2+0j)
    if k.imag > 0:
        k = -k
    expected = -1j/(2*p*k)*np.exp(-1j*k*np.abs(z[:, None]-z[None, :]))
    np.testing.assert_allclose(kernel.matrix(z), expected, rtol=2e-14, atol=1e-20)
    if qfactor > n:
        assert np.max(np.abs(kernel.matrix(z).imag)) < 1e-22
        amplitudes = kernel.outgoing_amplitudes(z)
        np.testing.assert_allclose(amplitudes.top_flux, 0.0, atol=1e-22)
        np.testing.assert_allclose(amplitudes.bottom_flux, 0.0, atol=1e-22)


@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_green_reciprocity_jump_and_optical_theorem(polarization):
    layers = (*quarter_wave_dbr(1030.0, 2.1, 1.45, 5, first_layer="low"),
              StratifiedLayer("spacer", 311.0, 1.45),
              StratifiedLayer("source", 243.0, 1.8),
              StratifiedLayer("cap", 121.0, 2.0))
    stack = StratifiedStack(layers, 1.45, 1.0)
    kernel = local_green_kernel(stack, len(layers)-2, 1030.0,
                                q_parallel_per_m=0.4*2*np.pi/1030e-9,
                                polarization=polarization)
    z = np.linspace(kernel.z_start_m, kernel.z_start_m+kernel.thickness_m, 37)
    matrix = kernel.matrix(z)
    np.testing.assert_allclose(matrix, matrix.T, rtol=3e-13, atol=1e-20)
    centre = z[len(z)//2]
    h = 1e-12
    derivative_jump = kernel.p_host*(kernel.green(centre+h, centre)-2*kernel.green(centre, centre)
                                     +kernel.green(centre-h, centre))/h
    assert derivative_jump == pytest.approx(-1.0, abs=8e-5)
    source = np.exp(0.3j*np.linspace(0, 1, len(z)))*(1.0+np.linspace(0, 1, len(z)))
    weights = np.full(len(z), z[1]-z[0])
    weights[[0, -1]] *= 0.5
    audit = kernel.quadrature_flux(z, source, weights)
    assert audit["lossless_equality_expected"]
    assert audit["top_flux"] > 0.0 and audit["bottom_flux"] > 0.0
    assert abs(audit["relative_residual"]) < 2e-12
    amplitudes = kernel.outgoing_amplitudes(z)
    np.testing.assert_allclose(-np.diag(matrix).imag,
                               amplitudes.top_flux+amplitudes.bottom_flux, rtol=4e-13, atol=1e-21)


def test_evanescent_extreme_stack_is_finite_without_transfer_overflow():
    stack = StratifiedStack(tuple(StratifiedLayer(f"layer{i}", 2000.0, 1.5+i%2*0.1)
                                  for i in range(80)), 1.0, 1.0)
    q = 1e9
    scattering = solve_stack(stack, 1030.0, q_parallel_per_m=q)
    assert np.all(np.isfinite(scattering.scattering_matrix))
    assert scattering.R is None and scattering.T is None
    kernel = local_green_kernel(stack, 39, 1030.0, q_parallel_per_m=q)
    z = kernel.z_start_m + np.linspace(0, kernel.thickness_m, 5)
    assert np.all(np.isfinite(kernel.matrix(z)))
    assert np.all(np.isfinite(kernel.outgoing_amplitudes(z).top))


def test_source_absorption_difference_is_not_labeled_energy_error():
    stack = StratifiedStack((StratifiedLayer("source", 230.0, 1.8-0.02j),), 1.0, 1.0)
    kernel = local_green_kernel(stack, 0, 1030.0)
    z = np.linspace(0, kernel.thickness_m, 31)
    weights = np.full(len(z), z[1]-z[0])
    weights[[0, -1]] *= 0.5
    audit = kernel.quadrature_flux(z, np.ones_like(z), weights)
    assert not audit["lossless_equality_expected"]
    assert audit["difference_or_absorption"] > 0.0


def test_validation_and_coordinate_contract():
    with pytest.raises(ValueError, match="passive"):
        StratifiedLayer("gain", 100.0, 1.8+0.01j)
    with pytest.raises(ValueError, match="pairs"):
        quarter_wave_dbr(1030.0, 2.1, 1.45, 0)
    stack = StratifiedStack((StratifiedLayer("spacer", 100.0, 1.45),
                             StratifiedLayer("source", 200.0, 1.8)), 1.0, 1.0)
    kernel = local_green_kernel(stack, 1, 1030.0)
    assert kernel.z_start_m == pytest.approx(100e-9)
    with pytest.raises(ValueError, match="inside"):
        kernel.matrix(np.array([0.0]))
    with pytest.raises(ValueError, match="grazing"):
        solve_stack(stack, 1030.0, q_parallel_per_m=2*np.pi/1030e-9)
