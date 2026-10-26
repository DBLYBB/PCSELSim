import numpy as np
import pytest

from pcselsim.design_objectives import beam_acceptance, beam_metrics


def evaluate(intensity, axis):
    return beam_metrics(axis, np.sqrt(np.maximum(intensity, 0)), np.zeros_like(intensity), source_nyquist_deg=2.)


def test_circular_and_elliptical_gaussian_have_analytic_moments():
    axis = np.linspace(-1, 1, 201)
    x, y = np.meshgrid(axis, axis)
    circular = evaluate(np.exp(-(x*x+y*y)/.05), axis)
    assert circular["ellipticity"] == pytest.approx(1., abs=1e-10)
    assert circular["radial_anisotropy_l2"] < .001
    assert beam_acceptance(circular, require_sampling=True)["accepted"]
    elliptical = evaluate(np.exp(-(x*x/.03+y*y/.06)), axis)
    assert elliptical["ellipticity"] == pytest.approx(np.sqrt(2), rel=1e-5)
    assert not beam_acceptance(elliptical)["accepted"]


def test_fourfold_pattern_cannot_pass_on_second_moment_alone():
    axis = np.linspace(-1, 1, 201)
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    intensity = np.exp(-radius**2/.08)*(1 + .7*np.cos(4*np.arctan2(y, x))*np.minimum(radius/.1, 1)**4)
    metrics = evaluate(intensity, axis)
    assert metrics["ellipticity"] == pytest.approx(1, abs=1e-10)
    assert metrics["fourfold_harmonic_l2"] > .2
    assert not beam_acceptance(metrics)["accepted"]


def test_ring_is_round_but_rejected_for_dark_center():
    axis = np.linspace(-1, 1, 161)
    x, y = np.meshgrid(axis, axis)
    metrics = evaluate((x*x+y*y)*np.exp(-(x*x+y*y)/.03), axis)
    assert metrics["ellipticity"] == pytest.approx(1)
    assert metrics["center_to_peak"] == 0
    assert not beam_acceptance(metrics)["accepted"]


def test_field_scaling_does_not_change_shape_and_sampling_is_not_padding():
    axis = np.linspace(-1, 1, 101)
    x, y = np.meshgrid(axis, axis)
    field = np.exp(-(x*x+y*y)/.05).astype(complex)
    original = beam_metrics(axis, field, field*1j, source_nyquist_deg=.6)
    scaled = beam_metrics(axis, field*1e-60j, -field*1e-60, source_nyquist_deg=.6)
    assert scaled["ellipticity"] == pytest.approx(original["ellipticity"])
    assert scaled["radial_anisotropy_l2"] == pytest.approx(original["radial_anisotropy_l2"])
    assert not beam_acceptance(scaled, require_sampling=True)["accepted"]
    with pytest.raises(ValueError, match="zero-radiation"):
        beam_metrics(axis, field*0, field*0)


def test_cropped_window_cannot_claim_requested_angle_by_padding_or_normalizing():
    axis = np.linspace(-.5, .5, 101)
    x, y = np.meshgrid(axis, axis)
    field = np.exp(-(x*x+y*y)/.02)
    metrics = beam_metrics(axis, field, field*0, requested_view_deg=1., source_nyquist_deg=.6)
    assert not metrics["requested_window_covered_by_samples"]
    assert not metrics["window_resolved_by_envelope_sampling"]
    assert not beam_acceptance(metrics, require_sampling=True)["accepted"]
