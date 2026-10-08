import numpy as np

from scripts.archive.legacy_20261008.design_triangular_three_hole_pcsel import CANDIDATES
from scripts.archive.legacy_20261008.design_triangular_three_hole_shape_position import (
    CANDIDATES as SHAPE_CANDIDATES,
    validate_nonoverlap,
)
from scripts.archive.legacy_20261008.design_triangular_triangle_main_refinement import (
    PARAMETERS as REFINEMENT_PARAMETERS,
    candidate_from_parameters,
)
from scripts.archive.legacy_20261008.design_square_lattice_multiobjective import (
    CANDIDATES as SQUARE_CANDIDATES,
    validate_nonoverlap as validate_square_nonoverlap,
)
from scripts.archive.legacy_20261008.run_best_triangular_three_triangle_pcsel import (
    DESIGN as BEST_THREE_TRIANGLE_DESIGN,
    LATTICE_CONSTANT_NM as BEST_THREE_TRIANGLE_LATTICE_NM,
)
from scripts.archive.legacy_20261008.optimize_triangular_three_triangle_manufacturable import (
    PARAMETERS as MANUFACTURABLE_PARAMETERS,
)


def test_three_hole_candidates_keep_requested_topology() -> None:
    for candidate in CANDIDATES:
        assert len(candidate.inclusions) == 3
        main, satellite_1, satellite_2 = candidate.inclusions
        assert main.center_fractional == (0.0, 0.0)
        assert satellite_1.center_fractional[0] > 0.0
        assert satellite_1.center_fractional[1] == 0.0
        assert satellite_2.center_fractional[0] == 0.0
        assert satellite_2.center_fractional[1] > 0.0
        fill = sum(inclusion.fill_fraction for inclusion in candidate.inclusions)
        assert np.isclose(fill, 0.15, atol=0.02)


def test_shape_position_candidates_are_nonoverlapping_and_equal_fill() -> None:
    assert len(SHAPE_CANDIDATES) == 30
    assert any(candidate.family == "triangle_main" for candidate in SHAPE_CANDIDATES)
    assert any(candidate.family == "square_satellites" for candidate in SHAPE_CANDIDATES)
    for candidate in SHAPE_CANDIDATES:
        validate_nonoverlap(candidate)
        fill = sum(inclusion.fill_fraction for inclusion in candidate.inclusions)
        assert np.isclose(fill, 0.15)


def test_triangle_main_refinement_candidates_are_valid() -> None:
    assert len(REFINEMENT_PARAMETERS) == 38
    assert {item.satellite_shape for item in REFINEMENT_PARAMETERS} == {
        "ellipse", "triangle"
    }
    for parameters in REFINEMENT_PARAMETERS:
        candidate = candidate_from_parameters(parameters)
        validate_nonoverlap(candidate)
        assert len(candidate.inclusions) == 3
        assert np.isclose(
            sum(inclusion.fill_fraction for inclusion in candidate.inclusions),
            0.15,
        )


def test_square_multiobjective_candidates_are_nonoverlapping() -> None:
    assert len(SQUARE_CANDIDATES) == 24
    assert {candidate.family for candidate in SQUARE_CANDIDATES} >= {
        "single", "dimer", "triple", "triangle_main", "polygon_cluster"
    }
    for candidate in SQUARE_CANDIDATES:
        validate_square_nonoverlap(candidate)
        assert len(candidate.inclusions) >= 1


def test_best_three_triangle_full_run_keeps_optimized_geometry() -> None:
    """Protect the dedicated full-run preset from accidental parameter drift."""
    candidate = candidate_from_parameters(BEST_THREE_TRIANGLE_DESIGN)
    validate_nonoverlap(candidate)
    assert BEST_THREE_TRIANGLE_DESIGN.name == "triangle_sat_manufacturable_300um"
    assert BEST_THREE_TRIANGLE_DESIGN.satellite_shape == "triangle"
    assert BEST_THREE_TRIANGLE_DESIGN.triangle_orientation == "inward"
    assert np.isclose(BEST_THREE_TRIANGLE_DESIGN.main_radius, 0.22)
    assert np.isclose(BEST_THREE_TRIANGLE_DESIGN.main_angle_deg, 0.0)
    assert np.isclose(BEST_THREE_TRIANGLE_DESIGN.distance, 0.44)
    assert np.isclose(BEST_THREE_TRIANGLE_DESIGN.spread_deg, -6.0)
    assert np.isclose(BEST_THREE_TRIANGLE_DESIGN.main_corner_fraction, 0.10)
    assert np.isclose(BEST_THREE_TRIANGLE_DESIGN.satellite_corner_fraction, 0.10)
    assert np.isclose(BEST_THREE_TRIANGLE_LATTICE_NM, 319.8520494119969)
    assert len(candidate.inclusions) == 3
    assert np.isclose(
        sum(inclusion.fill_fraction for inclusion in candidate.inclusions),
        0.15,
    )


def test_manufacturable_scan_is_unique_and_contains_current_winner() -> None:
    assert len(MANUFACTURABLE_PARAMETERS) == 59
    names = {item.name for item in MANUFACTURABLE_PARAMETERS}
    assert len(names) == len(MANUFACTURABLE_PARAMETERS)
    assert BEST_THREE_TRIANGLE_DESIGN.name not in names
    matches = [
        item for item in MANUFACTURABLE_PARAMETERS
        if np.isclose(item.main_radius, 0.22)
        and np.isclose(item.main_angle_deg, 0.0)
        and np.isclose(item.distance, 0.44)
        and np.isclose(item.spread_deg, -6.0)
        and np.isclose(item.main_corner_fraction, 0.10)
        and np.isclose(item.satellite_corner_fraction, 0.10)
    ]
    assert len(matches) == 1
    validate_nonoverlap(candidate_from_parameters(matches[0]))
