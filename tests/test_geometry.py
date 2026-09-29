import numpy as np

from pcselsim.geometry import (
    Ellipse,
    PolygonInclusion,
    SquareLatticeCell,
    inoue_double_lattice_cell,
)


def test_fourier_conjugate_symmetry() -> None:
    cell = inoue_double_lattice_cell()
    assert np.allclose(cell.fourier_epsilon(2, -1), np.conj(cell.fourier_epsilon(-2, 1)))
    displacement = np.subtract(cell.inclusions[1].center, cell.inclusions[0].center)
    assert np.allclose(displacement, (0.25, 0.25))


def test_zero_order_is_area_average() -> None:
    inclusion = Ellipse(center=(0.0, 0.0), radii=(0.1, 0.2), epsilon=1.0)
    cell = SquareLatticeCell(background_epsilon=12.0, inclusions=(inclusion,))
    fill = np.pi * 0.1 * 0.2
    assert np.isclose(cell.fourier_epsilon(0, 0), 12.0 + (1.0 - 12.0) * fill)


def test_polygon_fourier_area_and_conjugate_symmetry() -> None:
    triangle = PolygonInclusion(
        vertices=((-0.2, -0.1), (0.2, -0.1), (0.2, 0.1)), epsilon=1.0
    )
    cell = SquareLatticeCell(background_epsilon=12.0, inclusions=(triangle,))
    expected_area = 0.5*0.4*0.2
    assert np.isclose(cell.fourier_epsilon(0, 0), 12.0+(1.0-12.0)*expected_area)
    assert np.allclose(cell.fourier_epsilon(2, -1),
                       np.conj(cell.fourier_epsilon(-2, 1)))

