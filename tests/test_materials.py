import numpy as np

from pcselsim.config import CarrierConfig, OpticalConfig
from pcselsim.materials import (
    active_gain_m,
    effective_index_shift,
    modal_gain_m,
    temporal_index_rate_s,
)


def test_gain_is_zero_at_transparency() -> None:
    carrier = CarrierConfig()
    density = np.asarray([carrier.transparency_density_cm3 * 1e6])
    assert np.allclose(active_gain_m(density, carrier), 0.0)


def test_gain_saturates() -> None:
    carrier = CarrierConfig()
    optical = OpticalConfig()
    density = np.asarray([1e30])
    assert active_gain_m(density, carrier)[0] < carrier.maximum_gain_cm * 100.0
    assert modal_gain_m(density, carrier, optical)[0] > 0.0


def test_index_terms_use_appendix_a_overlap_factors() -> None:
    optical = OpticalConfig()
    reference = 2.0e24
    density = np.asarray([2.1e24])
    dn_dN_si = optical.dn_dN_cm3 * 1.0e-6
    expected_shift = (
        optical.confinement_factor
        * optical.active_index
        / optical.effective_index
        * dn_dN_si
        * (density-reference)
    )
    assert np.allclose(effective_index_shift(density, reference, optical), expected_shift)

    rate = np.asarray([3.0e32])
    expected_gamma = (
        2.0*optical.confinement_factor/optical.active_index*dn_dN_si*rate
    )
    assert np.allclose(temporal_index_rate_s(rate, optical), expected_gamma)

