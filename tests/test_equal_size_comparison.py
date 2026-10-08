import numpy as np

from scripts.archive.legacy_20261008.compare_square_triangular_equal_size import (
    INTERNAL_LOSS_CM,
    SQUARE_LATTICE_CONSTANT_NM,
    TRIANGULAR_LATTICE_CONSTANT_NM,
    aperture_area_m2,
    threshold_audit,
)


def test_reciprocal_vector_matching_lattice_constants() -> None:
    square_g = 2.0 * np.pi / SQUARE_LATTICE_CONSTANT_NM
    triangular_g = 4.0 * np.pi / (
        np.sqrt(3.0) * TRIANGULAR_LATTICE_CONSTANT_NM
    )
    assert np.isclose(square_g, triangular_g)


def test_equal_bounding_box_aperture_areas() -> None:
    circle = aperture_area_m2("circle", 150.0)
    hexagon = aperture_area_m2("hexagon", 150.0)
    square = aperture_area_m2("square", 150.0)
    assert hexagon < circle < square


def test_threshold_audit_increases_with_optical_loss() -> None:
    common = dict(area_m2=aperture_area_m2("square", 150.0),
                  confinement=0.045, effective_index=3.4)
    low = threshold_audit(200.0, **common)
    high = threshold_audit(500.0, **common)
    assert low["required_modal_gain_cm-1"] == INTERNAL_LOSS_CM + 4.0
    assert high["required_modal_gain_cm-1"] == INTERNAL_LOSS_CM + 10.0
    assert high["threshold_current_density_A_cm-2"] > low[
        "threshold_current_density_A_cm-2"
    ]
