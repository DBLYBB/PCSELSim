import numpy as np

from pcselsim.triangular_six_wave import (
    TRIANGULAR_BASIC_ORDERS,
    TRIANGULAR_DIRECTIONS,
    TRIANGULAR_POLARIZATIONS,
    TriangularCWTSettings,
    TriangularEllipse,
    TriangularLatticeCell,
    TriangularPolygon,
    build_triangular_coupling,
    reciprocal_metric_squared,
    triangular_band_diagram,
)
from pcselsim.vertical import Layer, LayerStack


def test_triangular_polygon_area_and_fourier_conjugate_symmetry() -> None:
    triangle = TriangularPolygon.regular(
        center_fractional=(0.17, -0.08),
        circumradius_over_a=0.12,
        sides=3,
        angle_deg=19.0,
    )
    expected_fill = 1.5 * 0.12**2
    assert np.isclose(triangle.fill_fraction, expected_fill)
    cell = TriangularLatticeCell(12.0, (triangle,))
    assert np.isclose(
        cell.fourier_epsilon(0, 0),
        12.0 + (1.0 - 12.0) * expected_fill,
    )
    assert np.allclose(
        cell.fourier_epsilon(2, -1),
        np.conj(cell.fourier_epsilon(-2, 1)),
    )


def test_rounded_triangle_has_smaller_area_and_keeps_fourier_symmetry() -> None:
    sharp = TriangularPolygon.regular((0.0, 0.0), 0.2, sides=3, angle_deg=15.0)
    rounded = TriangularPolygon.rounded_regular(
        (0.0, 0.0),
        0.2,
        sides=3,
        angle_deg=15.0,
        corner_fraction=0.12,
        samples_per_corner=6,
    )
    assert len(rounded.vertices_fractional) == 18
    assert 0.85 * sharp.fill_fraction < rounded.fill_fraction < sharp.fill_fraction
    cell = TriangularLatticeCell(12.0, (rounded,))
    assert np.allclose(
        cell.fourier_epsilon(2, -1),
        np.conj(cell.fourier_epsilon(-2, 1)),
    )


def test_triangular_basis_has_six_unit_directions_and_transverse_polarizations() -> None:
    assert len(TRIANGULAR_BASIC_ORDERS) == 6
    assert all(reciprocal_metric_squared(*order) == 1 for order in TRIANGULAR_BASIC_ORDERS)
    assert np.allclose(np.linalg.norm(TRIANGULAR_DIRECTIONS, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(TRIANGULAR_POLARIZATIONS, axis=1), 1.0)
    assert np.allclose(
        np.sum(TRIANGULAR_DIRECTIONS * TRIANGULAR_POLARIZATIONS, axis=1), 0.0
    )


def test_centered_circle_preserves_c6_doublets_and_passivity() -> None:
    cell = TriangularLatticeCell(
        background_epsilon=3.4**2,
        inclusions=(TriangularEllipse(radii_over_a=(0.18, 0.18), epsilon=1.0),),
    )
    assert np.isclose(cell.fourier_epsilon(1, 0), cell.fourier_epsilon(0, 1))
    assert np.isclose(cell.fourier_epsilon(1, 0), cell.fourier_epsilon(1, 1))
    average_index = float(np.sqrt(cell.fourier_epsilon(0, 0).real))
    stack = LayerStack(
        layers=(Layer("PC", 180.0, average_index),),
        top_index=3.0,
        bottom_index=3.0,
        padding_um=0.35,
    )
    result = build_triangular_coupling(
        cell,
        stack,
        "PC",
        lattice_constant_nm=300.0,
        wavelength_guess_nm=850.0,
        settings=TriangularCWTSettings(truncation_order=2, vertical_step_nm=10.0),
    )
    assert result.coupling_m.shape == (6, 6)
    assert np.allclose(result.eigenvalues_m[1], result.eigenvalues_m[2], rtol=1e-8, atol=1e-5)
    assert np.allclose(result.eigenvalues_m[4], result.eigenvalues_m[5], rtol=1e-8, atol=1e-5)
    radiation = (result.coupling_m - result.coupling_m.conj().T) / (2j)
    assert np.min(np.linalg.eigvalsh(radiation)) >= -1e-7
    assert np.count_nonzero(np.maximum(result.eigenvalues_m.imag, 0.0) > 1e-6) == 2
    diagram = triangular_band_diagram(result, q_max=0.01, points=21)
    assert diagram.normalized_frequency.shape == (6, 21)
    assert (diagram.left_endpoint, diagram.right_endpoint) == ("M", "X")
    assert np.all(np.isfinite(diagram.normalized_frequency))

