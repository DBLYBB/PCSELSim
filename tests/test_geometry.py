import numpy as np

from pcselsim.geometry import Ellipse, SquareLatticeCell, inoue_double_lattice_cell


def test_fourier_conjugate_symmetry() -> None:
    cell = inoue_double_lattice_cell()
    assert np.allclose(cell.fourier_epsilon(2, -1), np.conj(cell.fourier_epsilon(-2, 1)))


def test_zero_order_is_area_average() -> None:
    inclusion = Ellipse(center=(0.0, 0.0), radii=(0.1, 0.2), epsilon=1.0)
    cell = SquareLatticeCell(background_epsilon=12.0, inclusions=(inclusion,))
    fill = np.pi * 0.1 * 0.2
    assert np.isclose(cell.fourier_epsilon(0, 0), 12.0 + (1.0 - 12.0) * fill)

