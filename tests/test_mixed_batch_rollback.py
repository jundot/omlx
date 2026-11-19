# SPDX-License-Identifier: Apache-2.0
"""Vector rollback of a batch that mixes quantized and plain attention caches
(turboquant_skip_last, on by default), against the deepcopy rollback."""

import copy

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm")

from mlx_vlm.models.qwen3_5 import language as q35  # noqa: E402
from mlx_vlm.models.qwen3_5.config import TextConfig  # noqa: E402
from mlx_vlm.speculative import cache_state  # noqa: E402

from omlx.patches.mlx_lm_mtp import batch_generator as bg  # noqa: E402
from omlx.patches.mlx_lm_mtp import fused_batch  # noqa: E402
from omlx.patches.qwen35_padded_row_attention import (  # noqa: E402
    apply_qwen35_padded_row_attention_patch,
)
from omlx.patches.turboquant_attention import (  # noqa: E402
    apply_turboquant_attention_patch,
)
from omlx.patches.vlm_batch_kv_capacity import (  # noqa: E402
    apply_batch_kv_capacity_patch,
)
from omlx.turboquant_kv import quantized_cache_class  # noqa: E402

DEPTH = 3
CYCLES = ([3, 0], [1, 2], [0, 0], [2, 3], [0, 1], [3, 1])


def _model():
    apply_batch_kv_capacity_patch()
    apply_turboquant_attention_patch()
    apply_qwen35_padded_row_attention_patch()
    config = TextConfig(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_hidden_layers=4,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=100,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        head_dim=64,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
        },
        full_attention_interval=2,
    )
    language = q35.LanguageModel(config)
    mx.eval(language.parameters())
    return language


def _mixed_batch(language, lengths=(37, 30)):
    """Layers: recurrent, affine8, recurrent, plain (the skipped last layer)."""
    rows = []
    for n in lengths:
        cache = language.make_cache()
        cache[1] = quantized_cache_class("affine8")(bits=8)
        tokens = mx.random.randint(0, 100, (1, n), key=mx.random.key(n))
        mx.eval(language.model(tokens, cache=cache))
        rows.append(cache)
    return bg._merge_row_caches(rows)


def _rollback(cache, accepted, *, deferred_tail):
    """The vector rollback of _advance_group for the attention layers: the
    ragged trim of the quantized layers, the model's append-cache trim of the
    plain layer, then the mixed-batch realignment."""
    plain = cache[3]
    if deferred_tail:
        plain.__dict__["_omlx_tail_ok"] = True
    # Select by type: reading ``keys`` on an instance realigns a deferred tail.
    attention = [
        c for c in fused_batch._cache_leaves(cache) if hasattr(type(c), "keys")
    ]
    with fused_batch._ragged_trim_rows(cache, accepted, DEPTH):
        cache_state._trim_append_caches(attention, [a + 1 for a in accepted], DEPTH + 1)
    if fused_batch._ragged_mixed_with_plain_kv(cache):
        return fused_batch._align_mixed_rows_if_needed(cache)
    return False


@pytest.mark.parametrize("lengths", [(37, 30), (33, 33)])
@pytest.mark.parametrize("deferred_tail", [False, True])
def test_mixed_batch_rolls_back_like_the_deepcopy(deferred_tail, lengths):
    language = _model()
    cache = _mixed_batch(language, lengths)
    assert fused_batch._ragged_mixed_with_plain_kv(cache)
    assert fused_batch._supports_batch_rollback_all(cache)
    for cycle, accepted in enumerate(CYCLES):
        block = mx.random.randint(0, 100, (2, DEPTH + 1), key=mx.random.key(cycle))
        mx.eval(language.model(block, cache=cache))
        references = []
        for kept in accepted:
            view = copy.deepcopy(cache)
            for layer in (view[1], view[3]):
                layer.trim(DEPTH - kept)
            references.append(view)
        _rollback(cache, accepted, deferred_tail=deferred_tail)
        # Every attention layer ends each row at the same column, so the one
        # mask the model builds fits all of them. (A left-padded forward
        # replaces the cache objects, so read them here.)
        quantized, plain = cache[1], cache[3]
        assert fused_batch._mixed_rows_agree(cache)
        assert fused_batch._mixed_row_ends(quantized) == fused_batch._mixed_row_ends(
            plain
        )
        assert mx.array_equal(quantized.left_padding, plain.left_padding).item()
        for width in (1, DEPTH + 1):
            tokens = mx.random.randint(
                0, 100, (2, width), key=mx.random.key(100 + cycle)
            )
            out = language.model(tokens, cache=copy.deepcopy(cache))
            for row, view in enumerate(references):
                ref = language.model(tokens, cache=copy.deepcopy(view))
                assert mx.allclose(out[row], ref[row], atol=1e-4, rtol=1e-4).item(), (
                    cycle,
                    width,
                    row,
                )
        step = mx.random.randint(0, 100, (2, 1), key=mx.random.key(200 + cycle))
        mx.eval(language.model(step, cache=cache))


def test_plain_only_and_quantized_only_batches_are_not_mixed():
    language = _model()
    cache = _mixed_batch(language)
    assert not fused_batch._ragged_mixed_with_plain_kv([cache[0], cache[1]])
    assert not fused_batch._ragged_mixed_with_plain_kv([cache[2], cache[3]])


def test_mixed_batch_with_a_deferred_tail_moves_rows_only_when_the_plain_layer_realigns(
    monkeypatch,
):
    """With the plain layer's deferred tail the layers keep the same ragged
    ends, so a cycle realigns only when the tail grows past its bound."""
    language = _model()
    cache = _mixed_batch(language)
    calls = []
    original = fused_batch._align_mixed_rows
    monkeypatch.setattr(
        fused_batch, "_align_mixed_rows", lambda c: (calls.append(1), original(c))
    )
    moved = []
    for cycle, accepted in enumerate(CYCLES):
        block = mx.random.randint(0, 100, (2, DEPTH + 1), key=mx.random.key(cycle))
        mx.eval(language.model(block, cache=cache))
        moved.append(_rollback(cache, accepted, deferred_tail=True))
        tail = cache[3].__dict__.get("_omlx_tail")
        assert fused_batch._mixed_rows_agree(cache)
        # A realignment happens only on the cycle that cleared the plain tail.
        if moved[-1]:
            assert not tail or not any(tail)
    assert len(calls) == sum(moved)
    assert sum(moved) < len(CYCLES)


def test_a_realigned_plain_layer_realigns_the_quantized_layers():
    language = _model()
    cache = _mixed_batch(language)
    block = mx.random.randint(0, 100, (2, DEPTH + 1), key=mx.random.key(7))
    mx.eval(language.model(block, cache=cache))
    _rollback(cache, [3, 0], deferred_tail=True)
    assert cache[1]._row_ends is not None
    # An outside read of the plain bank realigns it (deferred_realign).
    mx.eval(cache[3].keys)
    assert not fused_batch._mixed_rows_agree(cache)
    assert fused_batch._align_mixed_rows_if_needed(cache)
    assert cache[1]._row_ends is None
    assert fused_batch._mixed_rows_agree(cache)
