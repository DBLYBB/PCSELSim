from pathlib import Path

from pcselsim.config import load_config


def test_inoue_config_loads() -> None:
    config = load_config(Path("configs/inoue2019.yaml"))
    assert config.optical.group_index == 3.513
    assert config.carrier.transparency_density_cm3 == 1.5e18
    assert config.numerics.points % 2 == 1

