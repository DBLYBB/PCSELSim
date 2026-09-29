"""Carrier-dependent semiconductor relations used by the Inoue model.

English
-------
This module contains *material* laws only: the saturated QW gain of Inoue
Eq. (11), its conversion to modal gain, and first-order carrier-induced index
perturbations.  Keeping these relations separate from the PDE solver makes it
possible to replace the semiconductor medium without changing the optical
four-wave discretization.

中文
----
本模块只负责材料关系：Inoue 式 (11) 的量子阱饱和增益、材料增益到模增益
的换算，以及载流子引起的一阶折射率扰动。它不负责空间传播；更换增益介质时
应优先替换本模块，而不必改动四波光场求解器。
"""

from __future__ import annotations

import numpy as np

from .config import CarrierConfig, OpticalConfig


def active_gain_m(N_m3: np.ndarray, carrier: CarrierConfig) -> np.ndarray:
    """Inoue Eq. (11): saturated active-layer material gain in m^-1.

    ``N_m3`` is always SI internally.  The configuration values retain the
    paper's cm-based units and are converted exactly once here.
    ``N_m3`` 在计算内核中统一采用 m^-3，配置中的 cm 单位仅在此换算一次。
    """
    n_tr = carrier.transparency_density_cm3 * 1.0e6
    g_max = carrier.maximum_gain_cm * 100.0
    minus_g0 = -carrier.zero_carrier_gain_cm * 100.0
    denominator = N_m3 + (g_max / minus_g0) * n_tr
    denominator = np.maximum(denominator, np.finfo(float).tiny)
    return g_max * (N_m3 - n_tr) / denominator


def modal_gain_m(
    N_m3: np.ndarray, carrier: CarrierConfig, optical: OpticalConfig
) -> np.ndarray:
    """Convert active material gain to guided modal gain.

    The overlap factor ``Gamma*n_active/n_eff`` follows the projection used in
    Inoue Appendix A. / 重叠因子 ``Gamma*n_active/n_eff`` 来自附录 A 的纵向投影。
    """
    overlap = optical.confinement_factor * optical.active_index / optical.effective_index
    return overlap * active_gain_m(N_m3, carrier)


def effective_index_shift(N_m3: np.ndarray, reference_m3: float, optical: OpticalConfig) -> np.ndarray:
    """Return the first-order modal effective-index change ``Delta n_eff``.

    Under the same scalar overlap approximation as the modal gain,
    ``Delta n_eff = Gamma*(n_active/n_eff)*(dn/dN)*Delta N``.  The previous
    implementation omitted ``n_active/n_eff``; retaining it makes the phase
    term consistent with the Appendix-A projection.

    在标量纵向重叠近似下，有
    ``Delta n_eff = Gamma*(n_active/n_eff)*(dn/dN)*Delta N``。该量随后进入
    时域方程的载流子失谐项。
    """
    dn_dN_m3 = optical.dn_dN_cm3 * 1.0e-6
    overlap = optical.confinement_factor * optical.active_index / optical.effective_index
    return overlap * dn_dN_m3 * (N_m3 - reference_m3)


def temporal_index_rate_s(
    dN_dt_m3_s: np.ndarray, optical: OpticalConfig
) -> np.ndarray:
    """Return Inoue Appendix Eq. (A10) temporal-index rate ``gamma`` in s^-1.

    With a uniform active-layer perturbation the vertical integral reduces to
    ``gamma = 2*Gamma/n_active*(dn/dN)*(dN/dt)``.  This term is real in the
    envelope equation and accounts for the time-dependent modal normalization;
    it is distinct from the phase detuning caused by ``Delta n_eff``.

    若有源层内载流子扰动近似均匀，附录式 (A10) 可化为上式。这里分母是
    有源层折射率 ``n_active``，而不是有效折射率 ``n_eff``。
    """
    dn_dN_m3 = optical.dn_dN_cm3 * 1.0e-6
    return (
        2.0
        * optical.confinement_factor
        / optical.active_index
        * dn_dN_m3
        * dN_dt_m3_s
    )

