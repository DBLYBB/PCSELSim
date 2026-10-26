"""Identify repeated numerical eigenvectors without inventing degenerate modes.

Only eigenvalues in the same small *complex* numerical cluster are compared.
High field overlap alone must never merge resolved distinct nonnormal modes.
Within a repeated eigenvalue, independent vectors enlarge its eigenspace;
rotating an already retained basis does not add a new physical mode.

中文：只在同一复数本征值的数值簇内比较场子空间。非正规算符的不同模式
可以高度平行，不能仅凭场重叠删除；真正简并时则保留新增的独立子空间方向。
"""
from __future__ import annotations

import numpy as np


def _unit_field(field):
    vector = np.asarray(field, dtype=complex).ravel()
    if not vector.size or not np.isfinite(vector).all():
        raise ValueError("Eigenmode fields must be nonempty and finite")
    scale = float(np.max(np.abs(vector)))
    if scale <= 0:
        raise ValueError("A zero eigenvector has no mode identity")
    vector = vector / scale
    return vector / np.linalg.norm(vector)


def independent_eigenmode(value, field, values, fields, *, eigenvalue_rtol=5e-9,
                          eigenvalue_atol=1e-8, span_tolerance=1e-6,
                          rank_tolerance=1e-8):
    """Return whether an eigenpair adds a resolved eigenvalue or span dimension.

    Fields may have any matching shape and need not be normalized. The default
    eigenvalue tolerances are in the supplied eigenvalue units (m^-1 in CWT).
    They are numerical identity tolerances, not a physical mode-separation
    criterion. A near-degenerate but resolved complex eigenvalue is retained
    even when its vector is almost parallel to another nonnormal mode.

    For frequency-dependent operators the caller must compare only poles of
    the same operator/family and the same closed frequency. Neither frequency
    matching nor spectral completeness is inferred from field overlap here.

    中文：先判断复数本征值是否数值相同，再用 SVD 判断新场是否扩大子空间。
    容差是数值身份判断，既不是物理单模间隔，也不保证已搜索完整谱。
    对频变算符，调用者须先按相同算符族与闭合频率分组。
    """
    if (not np.isfinite(value) or not np.isfinite([eigenvalue_rtol,eigenvalue_atol,
            span_tolerance,rank_tolerance]).all() or eigenvalue_rtol < 0 or eigenvalue_atol < 0
            or not 0 < span_tolerance < 1 or not 0 < rank_tolerance < 1):
        raise ValueError("Finite eigenvalues and valid identity tolerances are required")
    values, fields = list(values), list(fields)
    if len(values) != len(fields):
        raise ValueError("Previous eigenvalues and fields must have matching lengths")
    vector = _unit_field(field)
    cluster = []
    for previous, previous_field in zip(values, fields, strict=True):
        if not np.isfinite(previous):
            raise ValueError("Previous eigenvalues must be finite")
        tolerance = eigenvalue_atol + eigenvalue_rtol*max(abs(value), abs(previous), 1.)
        if abs(value-previous) <= tolerance:
            prior = _unit_field(previous_field)
            if prior.shape != vector.shape:
                raise ValueError("Fields in one eigenvalue cluster must have matching sizes")
            cluster.append(prior)
    if not cluster:
        return True
    basis, singular_values, _ = np.linalg.svd(np.asarray(cluster).T, full_matrices=False)
    basis = basis[:, singular_values > rank_tolerance*singular_values[0]]
    remainder = vector - basis @ (basis.conj().T @ vector)
    return bool(np.linalg.norm(remainder) > span_tolerance)


def independent_mode_indices(values, fields, **kwargs):
    """Return input-order indices spanning each numerical eigenvalue cluster."""
    values, fields = list(values), list(fields)
    if len(values) != len(fields):
        raise ValueError("Eigenvalues and fields must have matching lengths")
    retained = []
    for index, (value, field) in enumerate(zip(values, fields, strict=True)):
        if independent_eigenmode(value, field, [values[i] for i in retained],
                                 [fields[i] for i in retained], **kwargs):
            retained.append(index)
    return retained
