"""Exact coherent summation must preserve Jones phases on characteristic grids."""
import numpy as np
import pytest

from pcselsim.triangular_finite import _cartesian_point_fourier_sum


@pytest.mark.parametrize("cloud", ["structured", "duplicates", "nearby", "unstructured"])
def test_complex_point_sum_matches_independent_direct_definition(cloud):
    rng = np.random.default_rng(714)
    if cloud == "unstructured":
        x, y = rng.uniform(-150e-6, 150e-6, (2, 125))
    else:
        xx, yy = np.meshgrid(np.arange(-6, 7)*9e-6, np.arange(-5, 6)*7e-6)
        keep = (np.arange(xx.size) % 3 != 0)
        x, y = xx.ravel()[keep], yy.ravel()[keep]
        if cloud == "duplicates":
            x, y = np.r_[x, x[:9]], np.r_[y, y[:9]]
        elif cloud == "nearby":
            # Distinct coordinates must not be rounded to the same location.
            x = np.r_[x, x[:3]+1e-13]
            y = np.r_[y, y[:3]-2e-13]
    rx, ry = rng.normal(size=(2, len(x))) + 1j*rng.normal(size=(2, len(x)))
    wave = np.linspace(-2e5, 2e5, 19)
    kx, ky = np.meshgrid(wave, wave)
    phase = np.exp(-1j*(kx[..., None]*x + ky[..., None]*y))
    actual = _cartesian_point_fourier_sum(x, y, rx, ry, wave)
    for got, radiation in zip(actual, (rx, ry)):
        expected = np.einsum("yxn,n->yx", phase, radiation)
        assert np.max(abs(got-expected)) < 8e-13
        assert np.linalg.norm(got-expected)/np.linalg.norm(expected) < 5e-14
    # The normal sample is the coherent complex sum, separately for both
    # polarization components. Independent normalization would fail this test.
    for got, radiation in zip(actual, (rx, ry)):
        assert abs(got[len(wave)//2, len(wave)//2]-np.sum(radiation)) < 8e-13
