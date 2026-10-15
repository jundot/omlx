# SPDX-License-Identifier: Apache-2.0
"""ModernBERT padding regressions for issue #3507."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_embeddings.models.modernbert import Model, ModelArgs, ModernBertModel

from omlx.patches.modernbert_attention import (
    _update_attention_mask,
    patch_modernbert_attention,
)


@pytest.mark.parametrize("length", [64, 128, 255, 256, 257, 288])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_padded_attention_is_finite_and_matches_single(length, dtype):
    mx.random.seed(3507)
    short = 5
    mask = mx.array([[1] * short + [0] * (length - short), [1] * length])
    config = SimpleNamespace(
        config=SimpleNamespace(local_attention=128),
        embeddings=SimpleNamespace(
            norm=SimpleNamespace(weight=mx.ones(64, dtype=dtype))
        ),
    )
    masks = _update_attention_mask(config, mask)
    q, k, v = [mx.random.normal((2, 1, length, 64)).astype(dtype) for _ in range(3)]
    for additive in masks:
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=0.125, mask=additive
        )
        single = mx.fast.scaled_dot_product_attention(
            q[:1, :, :short], k[:1, :, :short], v[:1, :, :short], scale=0.125
        )
        assert mx.all(mx.isfinite(output)).item()
        assert mx.allclose(output[:1, :, :short], single, atol=0.01).item()
    # Local attention excludes distant real tokens; global attention keeps them.
    if length > 65:
        assert masks[0][1, 0, 0, 65].item() == 0
        assert masks[1][1, 0, 0, 65].item() < -10000


def test_patch_is_idempotent_and_scoped(monkeypatch):
    original = ModernBertModel._update_attention_mask
    monkeypatch.setattr(ModernBertModel, "_update_attention_mask", original)
    unrelated = SimpleNamespace(model=SimpleNamespace())
    patch_modernbert_attention(unrelated)
    assert ModernBertModel._update_attention_mask is original
    model = Model(
        ModelArgs(
            model_type="modernbert",
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=1,
        )
    )
    patch_modernbert_attention(model)
    patch_modernbert_attention(model)
    assert ModernBertModel._update_attention_mask is _update_attention_mask
