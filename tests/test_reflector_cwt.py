"""Regression audits for root's independent reflection-aware CWT wrapper."""

import numpy as np
import pytest

from pcselsim.geometry import Ellipse, SquareLatticeCell
from pcselsim.hybrid_vertical import HybridLayerStack
from pcselsim.reflector_cwt import build_reflector_coupling
from pcselsim.stratified_optics import StratifiedLayer, StratifiedStack, quarter_wave_dbr, solve_stack
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from pcselsim.vertical import Layer


def _geometry():
    # An off-centred asymmetric motif exercises complex Fourier conjugates.
    cell = SquareLatticeCell(1.82**2, (Ellipse((0.13, -0.07), (0.10, 0.075),
                                               angle_deg=21.0, epsilon=1.45**2),))
    n_pc = np.sqrt(cell.fourier_epsilon(0, 0).real)
    guide = HybridLayerStack((Layer("gain", 1200.0, 1.82), Layer("PC", 160.0, n_pc)),
                             top_index=1.45, bottom_index=1.45, padding_um=2.0)
    settings = ThreeDCWTSettings(truncation_order=2, vertical_step_nm=10.0)
    return cell, guide, settings, n_pc


def test_uniform_radiation_stack_recovers_free_green_cwt():
    cell, guide, settings, n_pc = _geometry()
    uniform = StratifiedStack(tuple(StratifiedLayer(layer.name, layer.thickness_nm, n_pc)
                                     for layer in guide.layers), n_pc, n_pc)
    baseline = build_geometry_coupling(cell, guide, "PC", 600.0, 1030.0, settings)
    wrapper = build_reflector_coupling(cell, guide, uniform, "PC", 600.0, 1030.0, settings)
    for key in ("c1d_m", "crad_m", "c2d_m", "coupling_m"):
        np.testing.assert_allclose(getattr(wrapper.result, key), getattr(baseline, key),
                                   rtol=2e-10, atol=2e-8)
    assert wrapper.result.radiation_surface_factor == pytest.approx(baseline.radiation_surface_factor,
                                                                  rel=2e-10)
    for name in ("unit_cell_response_x", "unit_cell_response_y"):
        for order, response in getattr(baseline, name).items():
            np.testing.assert_allclose(getattr(wrapper.result, name)[order], response,
                                       rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(wrapper.upward_amplitude_loss_m,
                               wrapper.downward_amplitude_loss_m, rtol=3e-12, atol=1e-10)


def test_dbr_radiation_flux_closure_passivity_and_side_power_losses():
    cell, guide, settings, _ = _geometry()
    layers = (*quarter_wave_dbr(1070.0, 2.1, 1.45, 8, first_layer="low"),
              StratifiedLayer("spacer", 2000.0, 1.45),
              *(StratifiedLayer(layer.name, layer.thickness_nm, layer.refractive_index)
                for layer in guide.layers))
    radiation = StratifiedStack(layers, bottom_index=1.45, top_index=1.45)
    wrapper = build_reflector_coupling(cell, guide, radiation, "PC", 600.0, 1030.0, settings)
    coupling = wrapper.result.coupling_m
    loss = (coupling-coupling.conj().T)/(2j)
    np.testing.assert_allclose(loss, wrapper.upward_amplitude_loss_m+wrapper.downward_amplitude_loss_m,
                               rtol=2e-11, atol=2e-8)
    assert np.linalg.eigvalsh(loss).min() > -2e-8
    assert wrapper.optical_theorem_residual < 2e-12
    # The DBR is finite: a nonzero lower flux is not hidden or relabelled zero.
    assert np.trace(wrapper.downward_amplitude_loss_m).real > 0.0
    random = np.random.default_rng(784).normal(size=(4, 9)) + 1j*np.random.default_rng(83).normal(size=(4, 9))
    up, down = wrapper.side_power_losses(random)
    expected = 2*np.einsum("ik,ij,jk->", random.conj(), loss, random).real/np.sum(abs(random)**2)
    assert up + down == pytest.approx(expected, rel=3e-12)
    assert up >= 0.0 and down > 0.0
    # Optical phase/amplitude after moving DBR is physical; it is not old G*R.
    uniform = StratifiedStack(tuple(StratifiedLayer(layer.name, layer.thickness_nm,
                                                    wrapper.result.average_pc_epsilon**0.5)
                                     for layer in guide.layers),
                              wrapper.result.average_pc_epsilon**0.5, wrapper.result.average_pc_epsilon**0.5)
    no_reflections = build_reflector_coupling(cell, guide, uniform, "PC", 600.0, 1030.0, settings)
    assert not np.allclose(wrapper.result.crad_m.real, no_reflections.result.crad_m.real)


def test_two_micron_low_index_spacer_round_trip_decoupling_criterion():
    """Necessary remote-reflector criterion, not a general proof for every TE family.

    At a stated guided beta, the DBR load reflected back through a low-index
    spacer is rho_DBR exp(-2 kappa s).  Each family and actual reflector needs
    this check, especially near the lower-cladding light line or a DBR pole.
    """
    _, guide, _, _ = _geometry()
    wavelength_nm = 1070.0
    mode = guide.solve_te0(wavelength_nm, dz_nm=10.0)
    k0 = 2*np.pi/(wavelength_nm*1e-9)
    beta = k0*mode.effective_index
    n_spacer = 1.45
    kappa = np.sqrt(beta**2-(n_spacer*k0)**2)
    mirror = StratifiedStack(quarter_wave_dbr(wavelength_nm, 2.1, n_spacer, 8,
                                             first_layer="low"), n_spacer, n_spacer)
    load = solve_stack(mirror, wavelength_nm, incident_side="top", q_parallel_per_m=beta)
    decoupled_load = abs(load.r)*np.exp(-2*kappa*2000e-9)
    assert decoupled_load < 1e-8
    # Ignoring attenuation near the light line would be an unjustified shortcut.
    near_light_line_kappa = np.sqrt((1.4501*k0)**2-(n_spacer*k0)**2)
    assert np.exp(-2*near_light_line_kappa*2000e-9) > 0.1
