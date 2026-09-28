import numpy as np

from pcselsim.config import OpticalConfig
from pcselsim.coupling import band_edge_basis, calibrated_coupling_matrix, validate_passive_coupling


def test_band_edge_basis_is_orthonormal() -> None:
    basis = band_edge_basis()
    assert np.allclose(basis.conj().T @ basis, np.eye(4))


def test_coupling_eigenvalues_and_passivity() -> None:
    optical = OpticalConfig()
    matrix = calibrated_coupling_matrix(optical)
    validate_passive_coupling(matrix)
    expected = np.sort(np.asarray(optical.modal_radiation_loss_cm) * 100.0)
    actual = np.sort(np.linalg.eigvalsh((matrix - matrix.conj().T) / (2j)))
    assert np.allclose(actual, expected)

