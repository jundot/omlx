# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen2 text-embedding model used by mlx-embeddings."""

import mlx.core as mx

from omlx.models import qwen2_embedding


def _tiny_args(**overrides):
    args = dict(
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
    )
    args.update(overrides)
    return qwen2_embedding.ModelArgs(**args)


def test_head_dim_is_derived_from_heads():
    args = _tiny_args()
    assert args.head_dim == args.hidden_size // args.num_attention_heads == 8


def test_attention_uses_qkv_bias_without_output_bias():
    """Qwen2 keeps a bias on q/k/v projections but not on the output projection."""
    attn = qwen2_embedding.Qwen2Attention(_tiny_args())
    assert hasattr(attn.q_proj, "bias")
    assert hasattr(attn.k_proj, "bias")
    assert hasattr(attn.v_proj, "bias")
    assert not hasattr(attn.o_proj, "bias")


def test_forward_pools_last_token_and_normalizes():
    model = qwen2_embedding.Model(_tiny_args())
    mx.eval(model.parameters())

    # Second sequence is right-padded; last-token pooling must use the last
    # real token (index 2), not the padding at index 3.
    input_ids = mx.array([[1, 2, 3, 4], [5, 6, 7, 0]])
    attention_mask = mx.array([[1, 1, 1, 1], [1, 1, 1, 0]])

    out = model(input_ids, attention_mask)
    mx.eval(out.text_embeds, out.last_hidden_state)

    assert out.text_embeds.shape == (2, 32)
    assert out.last_hidden_state.shape == (2, 4, 32)

    norms = mx.sqrt((out.text_embeds * out.text_embeds).sum(axis=1)).tolist()
    for n in norms:
        assert abs(n - 1.0) < 1e-3


def test_sanitize_drops_lm_head_and_rotary_inv_freq():
    model = qwen2_embedding.Model(_tiny_args())
    weights = {
        "model.embed_tokens.weight": mx.zeros((64, 32)),
        "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((4,)),
        "lm_head.weight": mx.zeros((64, 32)),
    }
    sanitized = model.sanitize(weights)
    assert "lm_head.weight" not in sanitized
    assert all("rotary_emb.inv_freq" not in k for k in sanitized)
    assert "model.embed_tokens.weight" in sanitized
