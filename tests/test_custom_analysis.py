import numpy as np
import pytest

from pcselsim.custom_analysis import (
    FourWaveOpticalSpec,
    LatticeInclusionSpec,
    LatticeSpec,
    analytic_mode_fields,
    coupling_from_modal_values,
    ensure_square_four_wave,
    far_field,
    LinearMode,
    radiation_proxy_field,
)
from pcselsim.coupling import validate_passive_coupling


def _lattice(kind: str = "square") -> LatticeSpec:
    return LatticeSpec(kind, 300.0, "circle", 60.0, 60.0, 0.0, 3.4, 1.0)


def test_lattice_fill_fraction_and_square_guard() -> None:
    assert _lattice().fill_fraction == pytest.approx(np.pi*0.2**2)
    ensure_square_four_wave(_lattice())
    with pytest.raises(ValueError, match="six-wave"):
        ensure_square_four_wave(_lattice("triangular"))


def test_multi_inclusion_fill_fraction() -> None:
    motif = (
        LatticeInclusionSpec("ellipse", 30.0, 20.0, -25.0, 25.0, -35.0),
        LatticeInclusionSpec("ellipse", 20.0, 10.0, 25.0, -25.0, -35.0),
    )
    lattice = LatticeSpec(
        "square", 250.0, "double_ellipse", 0.0, 0.0, 0.0, 3.5, 1.0, motif
    )
    assert lattice.fill_fraction == pytest.approx(
        np.pi*(30.0*20.0+20.0*10.0)/250.0**2
    )


def test_custom_coupling_is_passive() -> None:
    spec = FourWaveOpticalSpec(
        wavelength_nm=980.0,
        effective_index=3.4,
        group_index=3.5,
        domain_um=200.0,
        internal_loss_cm=5.0,
        modal_detuning_cm=(-4.0, -1.0, 2.0, 5.0),
        modal_radiation_loss_cm=(0.1, 0.2, 0.3, 0.4),
    )
    validate_passive_coupling(coupling_from_modal_values(spec))


def test_far_field_proxy_is_finite_and_nonzero_for_every_mode() -> None:
    for name in ("A", "B", "C", "D"):
        mode = LinearMode(name, 0.0, 1.0, analytic_mode_fields(17, name), 1.0)
        field = radiation_proxy_field(mode)
        angle, power, divergence = far_field(field, 1030.0, 200.0, view_deg=1.2)
        assert angle.size > 3
        assert np.isfinite(power).all()
        assert power.max() == pytest.approx(1.0)
        assert divergence > 0.0
