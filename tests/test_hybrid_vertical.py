import numpy as np
from pcselsim.hybrid_vertical import HybridLayerStack, solve_guided_te_modes
from pcselsim.vertical import Layer, LayerStack


def test_fast_scalar_te_matches_baseline():
    args = dict(
        layers=(Layer("core", 1200, 1.82),),
        top_index=1.45,
        bottom_index=1.45,
        padding_um=2,
    )
    fast = HybridLayerStack(**args).solve_te0(1030, 10)
    old = LayerStack(**args).solve_te0(1030, 10)
    assert np.isclose(fast.effective_index, old.effective_index, atol=1e-10)


def test_explicit_multimode_family():
    stack = HybridLayerStack((Layer("core", 4000, 1.82),), 1.45, 1.45, 2)
    modes = solve_guided_te_modes(stack, 1030, 10, 6)
    assert len(modes) == 6
    assert all(
        modes[j].effective_index > modes[j + 1].effective_index for j in range(5)
    )
    second = HybridLayerStack(stack.layers, 1.45, 1.45, 2, 1).solve_te0(1030, 10)
    assert np.isclose(second.effective_index, modes[1].effective_index)
