import numpy as np

from pcselsim.triangular_finite import (
    TriangularFiniteMode,
    build_device_grid,
    build_hexagonal_grid,
    triangular_vector_far_field,
)


def test_hexagonal_grid_has_expected_node_count_and_radius() -> None:
    for radius in (3, 5, 7):
        grid = build_hexagonal_grid(radius, 30.0)
        assert grid.points > 1
        assert (0, 0) in grid.index
        assert all((-j, -k) in grid.index for j, k in grid.index)
        assert np.max(np.hypot(grid.x_um, grid.y_um)) <= 30.0 + 1e-12


def test_direct_far_field_of_uniform_aperture_peaks_at_center() -> None:
    grid = build_hexagonal_grid(4, 20.0)
    fields = np.ones((6, grid.points), dtype=np.complex128)
    radiation = np.ones(grid.points, dtype=np.complex128)
    mode = TriangularFiniteMode(
        name="test",
        delta_per_m=0.0,
        alpha_per_m=1.0,
        fields=fields,
        band_overlap=1.0,
        grid=grid,
        radiation_x=radiation,
        radiation_y=np.zeros_like(radiation),
    )
    far = triangular_vector_far_field(mode, wavelength_nm=1000.0, samples=41)
    assert np.isclose(far.center_to_peak, 1.0)
    assert np.isclose(far.peak_offset_deg, 0.0)
    assert far.centroid_offset_deg < 1e-12
    assert np.isclose(far.ellipticity, 1.0, rtol=0.03)
    assert 0.0 < far.encircled_power_0p5deg < far.encircled_power_1deg
    assert 0.0 < far.encircled_power_1deg < 1.0
    assert np.all(np.isfinite(far.power))
    assert far.full_rms_divergence_deg > 0.0


def test_circle_hexagon_and_square_apertures_use_same_bounding_size() -> None:
    grids = {
        shape: build_device_grid(7, 150.0, shape)
        for shape in ("circle", "hexagon", "square")
    }
    for shape, grid in grids.items():
        assert grid.aperture_shape == shape
        assert (0, 0) in grid.index
        assert np.max(np.abs(grid.x_um)) <= 150.0 + 1e-12
        assert np.max(np.abs(grid.y_um)) <= 150.0 + 1e-12
    assert grids["hexagon"].points < grids["square"].points
    assert grids["circle"].points < grids["square"].points

