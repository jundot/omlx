# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Llama 4 BatchKVCache offset patch."""

import mlx.core as mx


def _tiny_llama4_config():
    return {
        "model_type": "llama4",
        "text_config": {
            "attention_bias": False,
            "attention_chunk_size": 8,
            "head_dim": 8,
            "hidden_size": 32,
            "interleave_moe_layer_step": 2,
            "intermediate_size": 32,
            "intermediate_size_mlp": 32,
            "max_position_embeddings": 1000,
            "model_type": "llama4",
            "num_attention_heads": 4,
            "num_experts_per_tok": 1,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "num_local_experts": 2,
            "rms_norm_eps": 1e-4,
            "rope_scaling": None,
            "rope_theta": 1000,
            "use_qk_norm": True,
            "vocab_size": 100,
        },
        "num_hidden_layers": 4,
        "vocab_size": 100,
    }


def test_llama4_attn_scales_broadcast_scalar_and_vector_offsets():
    from omlx.patches.llama4_attention import _llama4_attn_scales

    assert _llama4_attn_scales(0, 3, 8192, 0.1).shape == (1, 1, 3, 1)
    assert _llama4_attn_scales(mx.array(0), 3, 8192, 0.1).shape == (1, 1, 3, 1)
    assert _llama4_attn_scales(mx.array([0]), 3, 8192, 0.1).shape == (1, 1, 3, 1)
    assert _llama4_attn_scales(mx.array([0, 2]), 3, 8192, 0.1).shape == (
        2,
        1,
        3,
        1,
    )


def test_llama4_attention_patch_is_idempotent():
    from omlx.patches.llama4_attention import apply_llama4_attention_patch

    first = apply_llama4_attention_patch()
    second = apply_llama4_attention_patch()

    assert first in (True, False)
    assert second is False


def test_llama4_batch_kv_cache_offset_does_not_crash():
    from mlx_lm.models import llama4
    from mlx_lm.models.cache import KVCache

    from omlx.patches.llama4_attention import apply_llama4_attention_patch

    apply_llama4_attention_patch()

    args = llama4.ModelArgs.from_dict(_tiny_llama4_config())
    model = llama4.Model(args)
    cache = [
        (
            layer_cache.merge([layer_cache])
            if type(layer_cache) is KVCache
            else layer_cache
        )
        for layer_cache in model.make_cache()
    ]

    logits = model(mx.array([[1, 2]], dtype=mx.int32), cache=cache)
    mx.eval(logits)

    assert logits.shape == (1, 2, 100)
