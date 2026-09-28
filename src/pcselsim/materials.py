"""Carrier-dependent semiconductor material relations from Inoue et al. (2019)."""

from __future__ import annotations

import numpy as np

from .config import CarrierConfig, OpticalConfig


def active_gain_m(N_m3: np.ndarray, carrier: CarrierConfig) -> np.ndarray:
    """Equation (11): saturated active-layer material gain in m^-1."""
    n_tr = carrier.transparency_density_cm3 * 1.0e6
    g_max = carrier.maximum_gain_cm * 100.0
    minus_g0 = -carrier.zero_carrier_gain_cm * 100.0
    denominator = N_m3 + (g_max / minus_g0) * n_tr
    denominator = np.maximum(denominator, np.finfo(float).tiny)
    return g_max * (N_m3 - n_tr) / denominator


def modal_gain_m(
    N_m3: np.ndarray, carrier: CarrierConfig, optical: OpticalConfig
) -> np.ndarray:
    """Appendix Eq. (A10): convert active material gain to guided modal gain."""
    overlap = optical.confinement_factor * optical.active_index / optical.effective_index
    return overlap * active_gain_m(N_m3, carrier)


def effective_index_shift(N_m3: np.ndarray, reference_m3: float, optical: OpticalConfig) -> np.ndarray:
    """Carrier-induced effective-index change under the confinement approximation."""
    dn_dN_m3 = optical.dn_dN_cm3 * 1.0e-6
    return optical.confinement_factor * dn_dN_m3 * (N_m3 - reference_m3)

