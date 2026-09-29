import numpy as np

from scripts import run_custom_wang2024_triple_lattice as wang


def test_wang_structure_1_geometry_matches_main_text() -> None:
    fill = 3.0 * np.pi * wang.HOLE_RADIUS_NM**2 / wang.LATTICE_CONSTANT_NM**2
    assert np.isclose(fill, 0.08494532291191757)
    assert wang.HOLE_DIAMETER_NM == 90.0
    assert wang.LATTICE_CONSTANT_NM == 474.0


def test_wang_reported_current_density_uses_circular_window() -> None:
    area_cm2 = np.pi * (0.5 * wang.CONTACT_WINDOW_DIAMETER_UM * 1e-4) ** 2
    density_ka_cm2 = wang.REFERENCE_THRESHOLD_CURRENT_A / area_cm2 / 1e3
    assert np.isclose(density_ka_cm2, 1.66, rtol=0.01)
