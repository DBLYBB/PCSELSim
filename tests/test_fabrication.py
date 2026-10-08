import numpy as np

from pcselsim.fabrication import (FabricationRules, TRIANGULAR_DIRECT, evaluate_fabrication,
                                  pareto_mask, polygon_gap, perturb_triangular_inclusions)
from pcselsim.geometry import Ellipse
from pcselsim.triangular_six_wave import TriangularPolygon


def test_periodic_clearance_includes_self_neighbour_and_margin():
    value = evaluate_fabrication((Ellipse((0, 0), (0.2, 0.2)),), 300,
                                rules=FabricationRules(minimum_gap_nm=170, minimum_caliper_nm=100))
    assert np.isclose(value["minimum_periodic_gap_nm"], 180, atol=0.1)
    assert value["nominal_pass"] and not value["margin_pass"]


def test_crossing_segments_and_containment_are_zero_gap():
    rectangle = np.asarray(((-2, -0.1), (2, -0.1), (2, 0.1), (-2, 0.1)))
    assert polygon_gap(rectangle, rectangle[:, ::-1]) == 0
    assert polygon_gap(rectangle, rectangle * 0.5) == 0


def test_physical_etch_changes_fill_and_clearance():
    hole = TriangularPolygon.rounded_regular((0, 0), 0.2, corner_fraction=0.2)
    changed = perturb_triangular_inclusions((hole,), 320, np.random.default_rng(1), etch_bias_nm=3)
    before = evaluate_fabrication((hole,), 320, TRIANGULAR_DIRECT)
    after = evaluate_fabrication(changed, 320, TRIANGULAR_DIRECT)
    assert changed[0].fill_fraction > hole.fill_fraction
    assert after["minimum_periodic_gap_nm"] < before["minimum_periodic_gap_nm"]


def test_inward_etch_allows_rounded_contour_short_edges_to_disappear():
    hole = TriangularPolygon.rounded_regular((0, 0), 0.2, corner_fraction=0.1)
    shrunk = perturb_triangular_inclusions((hole,), 320, np.random.default_rng(1), etch_bias_nm=-3)[0]
    assert 0 < shrunk.fill_fraction < hole.fill_fraction


def test_pareto_keeps_tradeoffs_and_duplicates():
    assert pareto_mask(np.asarray(((1, 3), (2, 2), (3, 3), (1, 3)))).tolist() == [True, True, False, True]


def test_rounding_fraction_alone_does_not_imply_large_corner_radius():
    hole = TriangularPolygon.rounded_regular((0, 0), .22, corner_fraction=.1)
    result = evaluate_fabrication((hole,), 320, TRIANGULAR_DIRECT,
        FabricationRules(minimum_gap_nm=0, minimum_caliper_nm=0, minimum_corner_radius_nm=5))
    # This 0.22a test hole has a ~4.53 nm fillet. The historical, smaller
    # three-hole design has ~3.37 nm; rounding fraction alone fixes neither.
    assert 4 < result["minimum_sampled_corner_radius_nm"] < 5
    assert not result["nominal_pass"]
