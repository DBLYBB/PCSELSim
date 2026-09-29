"""Four-wave coupling matrices for square-lattice Gamma-point PCSELs.

``calibrated_coupling_matrix`` is the transparent legacy fallback used by the
original Inoue baseline: its A/B/C/D eigenvalues are user inputs, not geometry
predictions.  Geometry-driven studies must use :mod:`pcselsim.three_d_cwt`.

``calibrated_coupling_matrix`` 是早期 Inoue 基准使用的可追溯标定后端；A/B/C/D
本征值来自输入，并非由孔形预测。需要从晶格几何推导时应使用 ``three_d_cwt``。
"""

from __future__ import annotations

import numpy as np

from .config import OpticalConfig


def band_edge_basis() -> np.ndarray:
    """Return the orthonormal A/B/C/D basis in ``(Rx,Sx,Ry,Sy)`` coordinates.

    该固定基底用于标记 Γ 点四个带边组合；破坏对称性后，真实本征模可能是这些
    向量的混合，因此标签只按最大重叠分配。
    """
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
    注意：这是“指定带边本征值后反构造 C”，不能用于证明任意孔形的预测能力。
    """
    transform = band_edge_basis()
    detuning = np.asarray(optical.modal_detuning_cm, dtype=float) * 100.0
    loss = np.asarray(optical.modal_radiation_loss_cm, dtype=float) * 100.0
    eigenvalues = detuning + 1j * loss
    return transform @ np.diag(eigenvalues) @ transform.conj().T


def validate_passive_coupling(matrix: np.ndarray, atol: float = 1e-10) -> None:
    """Require the anti-Hermitian/radiation component to be passive.

    被动腔不能通过辐射项凭空产生能量，因此 ``Im(C)`` 必须半正定。
    """
    if matrix.shape != (4, 4):
        raise ValueError("The square-lattice four-wave C matrix must be 4x4")
    radiation = (matrix - matrix.conj().T) / (2j)
    if np.min(np.linalg.eigvalsh(radiation)) < -atol:
        raise ValueError("Im(C) must be positive semidefinite for a passive cavity")

