import numpy as np

from pcselsim.config import DeviceConfig
from pcselsim.injection import electrode_area_m2, electrode_profile


def test_circle_electrode_profile_is_normalized_and_radial() -> None:
    axis = np.linspace(-200e-6, 200e-6, 101)
    device = DeviceConfig(
        domain_um=400.0,
        electrode_um=200.0,
        current_spread_um=5.0,
        electrode_shape="circle",
    )
    profile = electrode_profile(axis, axis, device)
    dx = axis[1] - axis[0]
    assert np.isclose(np.sum(profile) * dx**2, 1.0)
    assert profile[50, 50] > profile[50, 80]
    assert np.isclose(profile[50, 60], profile[60, 50])


def test_nominal_electrode_areas() -> None:
    square = DeviceConfig(electrode_um=200.0, electrode_shape="square")
    circle = DeviceConfig(electrode_um=200.0, electrode_shape="circle")
    assert np.isclose(electrode_area_m2(square), (200e-6) ** 2)
    assert np.isclose(electrode_area_m2(circle), np.pi * (100e-6) ** 2)
