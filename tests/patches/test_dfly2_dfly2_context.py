import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from omlx.patches.deepseek_v41_dfly2.dfly2_context import DFly2ContextConfig, build_dfly2_context, build_layer_context


def a(x):
    return np.asarray(x)


def test_sparse_plus_window_has_official_646_slots():
    cfg = DFly2ContextConfig()
    cache = mx.arange(2 * 2048 * 3, dtype=mx.float32).reshape(2, 2048, 3)
    # Deliberately disjoint picks to retain the full accounting shape.
    picks = mx.arange(512, dtype=mx.int32)[None, :]
    picks = mx.concatenate([picks, picks + 1200], axis=0)
    out = build_layer_context(cache, picks, mx.array([1000, 1000], dtype=mx.int32), cfg)
    mx.eval(out.indices, out.valid, out.latents)
    assert out.indices.shape == (2, 646)
    assert out.latents.shape == (2, 646, 3)
    assert int(a(out.valid).sum()) == 2 * 646


def test_padding_duplicates_and_out_of_range_never_read_cache():
    cfg = DFly2ContextConfig(block_size=3, context_local_window=2, index_topk=4)
    cache = mx.arange(1 * 8 * 2, dtype=mx.float32).reshape(1, 8, 2)
    picks = mx.array([[0, 0, 7, -1]], dtype=mx.int32)
    out = build_layer_context(cache, picks, mx.array([1], dtype=mx.int32), cfg)
    mx.eval(out.indices, out.valid, out.latents)
    np.testing.assert_array_equal(a(out.indices), [[0, -1, 7, -1, -1, -1, 1, 2, 3]])
    np.testing.assert_array_equal(a(out.valid), [[True, False, True, False, False, False, True, True, True]])
    np.testing.assert_array_equal(a(out.latents)[0, [1, 3, 4, 5]], np.zeros((4, 2), np.float32))


def test_three_layers_pair_independently_and_without_a_draft_cache():
    cfg = DFly2ContextConfig(block_size=2, context_local_window=1, index_topk=2)
    caches = {layer: mx.full((1, 10, 1), float(layer)) for layer in cfg.target_kv_layers}
    picks = {layer: mx.array([[layer % 5, (layer + 1) % 5]], dtype=mx.int32) for layer in cfg.target_index_layers}
    outs = build_dfly2_context(caches, picks, mx.array([5], dtype=mx.int32), cfg)
    mx.eval(*[out.latents for out in outs])
    assert len(outs) == 3
    for expected_layer, out in zip(cfg.target_kv_layers, outs):
        assert np.all(a(out.latents)[a(out.valid)] == expected_layer)
