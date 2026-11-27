"""The engine-facing runtime: shapes, layer order, and parity with the adapter."""

import mlx.core as mx
import pytest

from omlx.patches.deepseek_v41_dfly2.dfly2_context import DFly2LayerContext
from omlx.patches.deepseek_v41_dfly2.runtime import DFly2Runtime


DIM = 8


def _history(keys=2):
    return DFly2LayerContext(
        indices=mx.zeros((1, keys), dtype=mx.int32),
        valid=mx.ones((1, keys), dtype=mx.bool_),
        latents=mx.zeros((1, keys, 4)),
    )


def _embed(ids):
    return mx.ones(ids.shape + (DIM,)) * ids[..., None].astype(mx.float32)


def _runtime(layers, *, mask_token_id=99):
    return DFly2Runtime(
        layers=layers,
        fusion=lambda x, h: x + h,
        final_norm=lambda x: x,
        correct=lambda hidden, embedded: hidden + embedded,
        lm_head=lambda x: mx.zeros(x.shape[:-1] + (4,)),
        mask_token_id=mask_token_id,
    )


def test_block_hidden_shape_and_layer_order():
    calls = []

    def layer(name):
        def apply(x, start, history):
            calls.append((name, start, history))
            return x + 1.0

        return apply

    runtime = _runtime([layer("a"), layer("b"), layer("c")])
    block = runtime.block_hidden(
        _embed, mx.zeros((1, 1, DIM)), mx.array([7], dtype=mx.int32), 11, [_history() for _ in range(3)], 4
    )
    mx.eval(block)
    assert block.shape == (1, 4, DIM)
    assert [name for name, _, _ in calls] == ["a", "b", "c"]
    assert {start for _, start, _ in calls} == {11}
    assert all(history is not None for _, _, history in calls)


def test_block_hidden_masks_every_slot_after_the_bonus_token():
    seen = {}

    def spy_embed(ids):
        seen["ids"] = ids
        return mx.zeros(ids.shape + (DIM,))

    runtime = _runtime([lambda x, s, h: x])
    runtime.block_hidden(spy_embed, mx.zeros((1, 1, DIM)), mx.array([5], dtype=mx.int32), 0, [_history()], 3)
    assert seen["ids"].tolist() == [[5, 99, 99]]


def test_block_hidden_rejects_a_history_per_layer_mismatch():
    runtime = _runtime([lambda x, s, h: x, lambda x, s, h: x])
    with pytest.raises(ValueError, match="one target history"):
        runtime.block_hidden(_embed, mx.zeros((1, 1, DIM)), mx.array([1], dtype=mx.int32), 0, [_history()], 2)


def test_correct_uses_the_previous_token_embedding():
    runtime = DFly2Runtime(
        layers=[lambda x, s, h: x],
        fusion=lambda x, h: x,
        final_norm=lambda x: x,
        correct=lambda hidden, embedded: hidden + embedded,
        lm_head=lambda x: x,
        mask_token_id=0,
    )
    hidden = mx.zeros((1, DIM))
    out = runtime.correct(_embed, hidden, mx.array([3], dtype=mx.int32))
    mx.eval(out)
    assert out.tolist() == [[3.0] * DIM]


def test_runtime_matches_the_adapter_dspark_forward():
    """The adapter must return exactly what the runtime computes, slot for slot."""
    from omlx.patches.deepseek_v41.dfly2 import attach

    from omlx.patches.deepseek_v41_dfly2 import DSV41TargetBridge

    class Host:
        _config = {
            "dfly_config": {
                "block_size": 6,
                "mask_token_id": 99,
                "index_topk": 2,
                "target_kv_layers": [0],
                "target_index_layers": [0],
                "hidden_correction": {"intermediate_size": 16},
            }
        }

        embed = staticmethod(_embed)

        def __init__(self):
            self.dfly = _runtime([lambda x, s, h: x + 2.0], mask_token_id=99)
            self._dfly_cache = DSV41TargetBridge(target_layers=(0,), index_topk=2, local_window=0)
            self._dfly_cache.record(
                layer=0,
                start=0,
                window_latents=mx.zeros((1, 1, 4)),
                global_positions=mx.array([], dtype=mx.int32),
                global_latents=mx.zeros((1, 0, 4)),
                index_topk=mx.array([[[0, -1]]], dtype=mx.int32),
            )

    attach(Host)
    host = Host()
    block, state = host.dspark_forward(
        mx.zeros((1, 1, DIM)), mx.array([[4]], dtype=mx.int32), [], draft_length=3
    )
    mx.eval(block)
    assert state is None
    assert block.shape == (1, 4, DIM)
    expected = host.dfly.block_hidden(
        lambda ids: _embed(ids),
        mx.zeros((1, 1, DIM)),
        mx.array([4], dtype=mx.int32),
        1,
        [host._dfly_cache.history(layer=0, anchor_position=1)],
        4,
    )
    mx.eval(expected)
    assert block.tolist() == expected.tolist()
