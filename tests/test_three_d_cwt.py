import numpy as np

from pcselsim.geometry import Ellipse, SquareLatticeCell
from pcselsim.three_d_cwt import ThreeDCWTSettings, build_geometry_coupling
from pcselsim.vertical import Layer, LayerStack


def _model(radius: float):
    cell = SquareLatticeCell(
        background_epsilon=3.52**2,
        inclusions=(Ellipse((0.0, 0.0), (radius, radius), epsilon=1.0),),
    )
    nav = np.sqrt(cell.fourier_epsilon(0, 0).real)
    stack = LayerStack(
        layers=(
            Layer("lower", 700.0, 3.30),
            Layer("PC", 120.0, nav),
            Layer("core", 180.0, 3.53),
            Layer("upper", 700.0, 3.30),
        ),
        top_index=3.30,
        bottom_index=3.30,
        padding_um=0.5,
    )
    return build_geometry_coupling(
        cell, stack, "PC", 295.0, 990.0,
        ThreeDCWTSettings(truncation_order=3, vertical_step_nm=12.0),
    )


def test_circle_has_two_symmetry_protected_band_edges() -> None:
    result = _model(np.sqrt(0.16/np.pi))
    losses = np.sort(result.eigenvalues_m.imag)
    assert np.all(np.abs(losses[:2]) < 1e-6)
    assert np.all(losses[2:] > 0.0)
    assert result.passivity_correction_m < 1e-6


def test_geometry_changes_the_coupling_matrix() -> None:
    small = _model(0.16)
    large = _model(0.23)
    assert not np.allclose(small.coupling_m, large.coupling_m)
    assert small.bragg_wavelength_nm != large.bragg_wavelength_nm


def test_unit_cell_reconstruction_contains_high_orders() -> None:
    result = _model(np.sqrt(0.16/np.pi))
    assert (2, 0) in result.unit_cell_response_x
    axis, intensity, phase = result.unit_cell_fields(result.eigenvectors[:, 0], points=41)
    assert axis.shape == (41,)
    assert intensity.shape == phase.shape == (41, 41)
    assert np.isclose(float(intensity.max()), 1.0)
    assert np.all(np.isfinite(intensity))
