"""Passive characteristic staggering for finite coupled-wave eigenproblems.

Liang, thesis Eqs. (5.18)-(5.19), and Opt. Express 21, 565 (2013),
doi:10.1364/OE.21.000565, average the two end-point amplitudes at each
coefficient point.  With the incoming shift T (zero at an incoming boundary),
M=(I+T)/2, D=(I-T)/h and K=C M+i D, the problem is K u=lambda M u.
The physical coefficient-point field is phi=M u; u lives on outgoing links.

Unlike first-order upwind, this discretization has no volume dissipation:
M^H D+D^H M=(I-T^H T)/h is exactly the outgoing-boundary flux.  A staircased
aperture still has geometric boundary error and requires mesh refinement.

We implement generalized shift-invert explicitly.  scipy.eigs(A, M=M) assumes
a Hermitian positive mass matrix, which this directional average is not.

中文：未知量 u 位于出射链路，实际系数点场为 phi=M u；求解 Ku=lambda Mu。
M 为方向平均矩阵，不满足 eigs(A, M=M) 所要求的正定 Hermitian 假设，故显式
构造广义移位反演。交错格式的耗散来自出射边界；阶梯边界仍需网格收敛检查。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import eig as dense_generalized_eig
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs, splu


DISCRETIZATIONS = ("staggered", "legacy_upwind")
DENSE_EIGEN_CUTOFF = 400


@dataclass
class FiniteEigenproblem:
    """Generalized link problem and its physical fields / 广义链路问题及实际系数点场。

    Matrices use metres; Im(lambda) is field-amplitude loss per metre. near()
    returns phi=M u, so downstream intensity and flux must use those fields.
    矩阵采用米制，Im(lambda) 为振幅损耗 /m；near() 返回 phi=M u，后续近场
    强度、辐射以及能流收支均应使用该场，不能直接误用链路未知量 u。
    """
    stiffness: sparse.csc_matrix
    mass: sparse.csc_matrix
    transport: sparse.csc_matrix
    local: sparse.csc_matrix
    discretization: str

    def center_operator(self) -> LinearOperator:
        """Equivalent operator on physical center fields, for residual checks."""
        factor = splu(self.mass)
        return LinearOperator(
            self.stiffness.shape,
            matvec=lambda field: self.stiffness @ factor.solve(np.asarray(field).ravel()),
            dtype=np.complex128,
        )

    def near(self, target: complex, count: int, v0: np.ndarray,
             tol: float = 1e-9, maxiter: int = 8000, *,
             dense_limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return eigenvalues and normalized coefficient-point eigenvectors.

        中文：小矩阵使用原始广义 QZ；大矩阵对 (K-target*M)^-1 M 做 Arnoldi。
        两条路径最终都在原始物理场算符上检查残差，移位反演内部收敛不足以验收。
        """
        cutoff = DENSE_EIGEN_CUTOFF if dense_limit is None else dense_limit
        if not isinstance(cutoff, (int,np.integer)) or cutoff < 0:
            raise ValueError("dense_limit must be a nonnegative integer")
        if self.stiffness.shape[0] <= cutoff:
            # Weak-feedback transport matrices can be almost defective, making
            # a small Arnoldi window unusually slow. QZ solves this identical
            # generalized eigenproblem and retains repeated eigenvalues. Cache
            # per problem, so several band-edge targets reuse the same solve.
            spectrum = getattr(self, "_dense_spectrum", None)
            if spectrum is None:
                values, links = dense_generalized_eig(self.stiffness.toarray(), self.mass.toarray())
                fields = self.mass @ links
                norms = np.linalg.norm(fields, axis=0)
                finite = np.isfinite(values) & np.isfinite(norms) & (norms > 1e-30)
                values, fields, norms = values[finite], fields[:, finite], norms[finite]
                fields /= norms[None, :]
                operator = self.center_operator()
                valid = np.asarray([relative_residual(operator, field, value) <= 2e-6
                                    for value, field in zip(values, fields.T, strict=True)])
                self._dense_spectrum = spectrum = (values[valid], fields[:, valid])
            values, fields = spectrum
            if len(values) < count:
                raise RuntimeError("Too few finite, residual-verified dense generalized eigenpairs")
            selected = np.argsort(abs(values-target), kind="stable")[:count]
            return values[selected].copy(), fields[:, selected].copy()
        factor = splu(self.stiffness - target * self.mass)
        shifted = LinearOperator(
            self.stiffness.shape,
            matvec=lambda vector: factor.solve(self.mass @ vector),
            dtype=np.complex128,
        )
        try:
            inverse_values, links = eigs(shifted, k=count, which="LM", v0=v0,
                                         tol=tol, maxiter=maxiter)
        except ArpackNoConvergence as error:
            if error.eigenvalues is None or len(error.eigenvalues) < 4:
                raise
            inverse_values, links = error.eigenvalues, error.eigenvectors
        values = target + 1.0 / inverse_values
        fields = self.mass @ links
        fields /= np.linalg.norm(fields, axis=0, keepdims=True)
        # ARPACK's inverse residual alone can conceal errors after inversion.
        operator = self.center_operator()
        for value, field in zip(values, fields.T, strict=True):
            residual = relative_residual(operator, field, value)
            if residual > 2e-6:
                raise RuntimeError(f"Finite eigenpair residual {residual:.3g} exceeds tolerance")
        return values, fields

    def loss_budget(self, field: np.ndarray) -> dict[str, float]:
        """Exact discrete energy identity, in m^-1 of field-amplitude loss.

        中文：局域反 Hermitian 部分与边界输运相加得到振幅损耗；比较强度损耗
        或计算材料增益阈值前要乘 2。内部材料吸收由上层阈值流程单独加入。
        """
        phi = np.asarray(field).ravel()
        links = splu(self.mass).solve(phi)
        norm = float(np.vdot(phi, phi).real)
        radiation = (self.local - self.local.conj().T) / (2j)
        local_loss = float(np.vdot(phi, radiation @ phi).real / norm)
        transport_loss = float(np.vdot(phi, self.transport @ links).real / norm)
        return {"local_amplitude_loss_per_m": local_loss,
                "transport_amplitude_loss_per_m": transport_loss,
                "volume_numerical_diffusion_per_m": 0.0 if self.discretization == "staggered" else float("nan")}


def relative_residual(operator, field: np.ndarray, value: complex) -> float:
    """Scale-free residual of the original eigenpair / 原始本征方程的无量纲相对残差。"""
    vector = np.asarray(field).ravel()
    applied = operator @ vector
    denominator = max(np.linalg.norm(applied), abs(value) * np.linalg.norm(vector), 1e-30)
    return float(np.linalg.norm(applied - value * vector) / denominator)


def build_finite_eigenproblem(coupling: np.ndarray, derivatives: tuple,
                              spacing_m: float, discretization: str) -> FiniteEigenproblem:
    """Use the same staggered scheme for four- and six-wave characteristic grids."""
    if discretization not in DISCRETIZATIONS:
        raise ValueError(f"discretization must be one of {DISCRETIZATIONS}")
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    coupling = np.asarray(coupling, dtype=np.complex128)
    if coupling.shape != (len(derivatives), len(derivatives)):
        raise ValueError("one directed derivative is required for each coupled wave")
    points = derivatives[0].shape[0]
    local = sparse.kron(sparse.csc_matrix(coupling), sparse.identity(points, format="csc"), format="csc")
    transport = sparse.block_diag(derivatives, format="csc").astype(np.complex128)
    identity = sparse.identity(transport.shape[0], dtype=np.complex128, format="csc")
    mass = identity - 0.5 * spacing_m * transport if discretization == "staggered" else identity
    stiffness = (local @ mass + 1j * transport).tocsc()
    return FiniteEigenproblem(stiffness, mass.tocsc(), transport, local, discretization)


def assess_mode_selection(modes: dict) -> dict:
    """Numerical ordering is unresolved when grid sensitivity spans its gap."""
    ordered = sorted(modes.values(), key=lambda mode: mode.alpha_per_m)
    winner = ordered[0]
    gap = float(ordered[1].alpha_per_m - winner.alpha_per_m)
    uncertainty = sum(float(mode.extrapolation_uncertainty_per_m or 0.0) for mode in ordered[:2])
    invalid = any(mode.extrapolation_status in {"single_grid", "grid_sensitive", "rejected_negative_intercept"}
                  for mode in ordered[:2])
    invalid = invalid or any(getattr(mode, "discretization", "staggered") != "staggered" for mode in ordered[:2])
    resolved = not invalid and gap > uncertainty
    return {"lowest_computed_family": winner.name,
            "family_gap_per_m": gap,
            "combined_grid_sensitivity_per_m": uncertainty,
            "ordering_resolved_on_tested_grids": bool(resolved),
            "selection_interpretation": (
                "Lowest computed family; ordering separated on tested grids, not a lasing proof"
                if resolved else "Numerically unresolved family ordering; not a confirmed lasing mode"),
            "spectrum_scope": "one selected state per Gamma-connected family; higher lateral spectrum incomplete"}


def staggered_grid_diagnostics(inverse_grid: np.ndarray, values: np.ndarray) -> tuple[float, float, str]:
    """Empirical mesh sensitivity; never replace an actual eigenpair by a fit.

    Smooth rectangular grids have second-order center averaging, but masked
    circles/hexagons can have first-order geometric error. Include fits in
    both h and h^2 and the actual last mesh step in the reported sensitivity.
    """
    values = np.asarray(values, dtype=float)
    inverse_grid = np.asarray(inverse_grid, dtype=float)
    intercepts = [float(np.polyfit(inverse_grid**order, values, 1)[1]) for order in (1, 2)]
    sensitivity = max(abs(values[-1] - values[-2]), *(abs(values[-1] - value) for value in intercepts))
    status = "mesh_checked" if sensitivity <= 0.10 * max(abs(values[-1]), 1e-30) else "grid_sensitive"
    return intercepts[1], float(sensitivity), status
