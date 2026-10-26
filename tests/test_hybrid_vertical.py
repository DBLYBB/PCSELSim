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


def test_non_aligned_layer_partition_preserves_each_family_overlap():
    whole=HybridLayerStack((Layer('core',1200.,1.82),),1.45,1.45,2.)
    split=HybridLayerStack((Layer('left',301.3,1.82),Layer('thin',3.1,1.82),
                           Layer('right',895.6,1.82)),1.45,1.45,2.)
    for original,partitioned in zip(solve_guided_te_modes(whole,1030,7.3,2),
                                    solve_guided_te_modes(split,1030,7.3,2)):
        assert np.isclose(original.effective_index,partitioned.effective_index,atol=1e-12)
        assert np.isclose(sum(partitioned.confinement.values()),original.confinement['core'],atol=1e-12)
        assert partitioned.confinement['thin']>0
