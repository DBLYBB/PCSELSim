import numpy as np

from pcselsim.config import CarrierConfig, OpticalConfig
from pcselsim.materials import active_gain_m, modal_gain_m


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

