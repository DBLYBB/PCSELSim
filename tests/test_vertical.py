from pcselsim.vertical import Layer, LayerStack


def test_symmetric_slab_te0_is_guided() -> None:
    stack = LayerStack(
        layers=(Layer("core", 500.0, 3.5),),
        top_index=3.0,
        bottom_index=3.0,
        padding_um=1.5,
    )
    mode = stack.solve_te0(wavelength_nm=1000.0, dz_nm=10.0)
    assert 3.0 < mode.effective_index < 3.5
    assert 0.0 < mode.confinement["core"] < 1.0

