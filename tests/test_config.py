from pathlib import Path
from dataclasses import replace

import pytest

from pcselsim.config import SimulationConfig, load_config, validate_config


def test_inoue_config_loads() -> None:
    config = load_config(Path("configs/inoue2019.yaml"))
    assert config.optical.group_index == 3.513
    assert config.carrier.transparency_density_cm3 == 1.5e18
    assert config.numerics.points % 2 == 1


def test_nonphysical_lifetime_is_rejected() -> None:
    config = SimulationConfig()
    with pytest.raises(ValueError, match="lifetime"):
        validate_config(replace(config, carrier=replace(config.carrier, lifetime_ns=0.0)))


def test_explicit_diffusion_stability_is_checked() -> None:
    config = SimulationConfig()
    tiny_domain = replace(config, device=replace(config.device, domain_um=0.001, electrode_um=0.001))
    with pytest.raises(ValueError, match="diffusion"):
        validate_config(tiny_domain)

