"""Four-wave coupling matrices for square-lattice Gamma-point PCSELs."""

from __future__ import annotations

import numpy as np

from .config import OpticalConfig


def band_edge_basis() -> np.ndarray:
    """Return an orthonormal A/B/C/D basis in (Rx,Sx,Ry,Sy) coordinates."""
    return np.asarray(
        [
            [1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.complex128,
    ).T / np.asarray([2.0, 2.0, np.sqrt(2.0), np.sqrt(2.0)])


def calibrated_coupling_matrix(optical: OpticalConfig) -> np.ndarray:
    """Build C from calibrated band-edge eigenvalues, in m^-1.

    In Eq. (8), the coupling contribution is ``i * (c/ng) * C @ Phi``.
    Thus Im(eigenvalue(C)) is a positive amplitude-loss coefficient.
    """
    transform = band_edge_basis()
    detuning = np.asarray(optical.modal_detuning_cm, dtype=float) * 100.0
    loss = np.asarray(optical.modal_radiation_loss_cm, dtype=float) * 100.0
    eigenvalues = detuning + 1j * loss
    return transform @ np.diag(eigenvalues) @ transform.conj().T


def validate_passive_coupling(matrix: np.ndarray, atol: float = 1e-10) -> None:
    """Require the anti-Hermitian/radiation component to be passive."""
    if matrix.shape != (4, 4):
        raise ValueError("The square-lattice four-wave C matrix must be 4x4")
    radiation = (matrix - matrix.conj().T) / (2j)
    if np.min(np.linalg.eigvalsh(radiation)) < -atol:
        raise ValueError("Im(C) must be positive semidefinite for a passive cavity")

