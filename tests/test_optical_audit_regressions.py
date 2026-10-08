import numpy as np
import pytest

from pcselsim.custom_analysis import (
    FourWaveOpticalSpec,
    _rotate_four_wave_fields,
    coupling_from_modal_values,
    finite_area_operator,
)
from pcselsim.numerical_quality import extrapolate_grid_loss
from pcselsim.triangular_finite import (
    TriangularFiniteMode,
    build_device_grid,
    triangular_vector_far_field,
)
from pcselsim.vertical import Layer, LayerStack


def test_finite_box_cladding_state_is_not_a_guided_mode():
    stack = LayerStack((Layer("uniform", 500.0, 1.5),), 1.5, 1.5, 0.3)
    with pytest.raises(ValueError, match="no bound TE0"):
        stack.solve_te0(1000.0, dz_nm=20.0)


def test_negative_grid_intercept_is_rejected_instead_of_zero_loss():
    inverse = 1.0 / np.asarray([7.0, 9.0, 11.0])
    losses = 100.0 * inverse - 2.0
    accepted, raw, sensitivity, status = extrapolate_grid_loss(inverse, losses)
    assert raw == pytest.approx(-2.0)
    assert accepted == pytest.approx(losses[-1])
    assert status == "rejected_negative_intercept"
    assert sensitivity < 1e-12


def test_physical_c4_rotation_commutes_with_symmetric_finite_operator():
    spec = FourWaveOpticalSpec(
        1000.0, 3.4, 3.5, 70.0, 0.0,
        (-10.0, -5.0, 10.0, 10.0), (0.0, 0.0, 2.0, 2.0), 9,
    )
    operator = finite_area_operator(spec, coupling_from_modal_values(spec))
    rng = np.random.default_rng(823)
    field = rng.normal(size=(4, 9, 9)) + 1j * rng.normal(size=(4, 9, 9))
    rotated = _rotate_four_wave_fields(field)
    first = (operator @ rotated.ravel()).reshape(field.shape)
    second = _rotate_four_wave_fields((operator @ field.ravel()).reshape(field.shape))
    assert np.allclose(first, second, rtol=1e-12, atol=1e-8)


def test_six_wave_far_field_preserves_physical_tilt_direction_and_limits_aliasing():
    grid = build_device_grid(6, 60.0, "square")
    wavelength = 1000.0
    tilt = 0.4
    transverse_k = 2.0 * np.pi / (wavelength * 1e-9) * np.tan(np.deg2rad(tilt))
    radiation = np.exp(1j * transverse_k * grid.x_um * 1e-6)
    mode = TriangularFiniteMode(
        "tilted", 0.0, 1.0, np.ones((6, grid.points)), 1.0,
        grid, radiation, np.zeros_like(radiation),
    )
    far = triangular_vector_far_field(mode, wavelength, view_deg=0.8, samples=81)
    row, column = np.unravel_index(np.argmax(far.power), far.power.shape)
    assert far.angle_deg[column] == pytest.approx(tilt, abs=0.025)
    assert far.angle_deg[row] == pytest.approx(0.0, abs=0.025)
    with pytest.warns(RuntimeWarning, match="alias-free"):
        wide = triangular_vector_far_field(mode, wavelength, view_deg=8.0, samples=40)
    assert len(wide.angle_deg) % 2 == 1
    assert wide.evaluated_view_deg < wide.alias_free_square_view_deg
    assert wide.energy_normalization == "evaluated_angular_window"
