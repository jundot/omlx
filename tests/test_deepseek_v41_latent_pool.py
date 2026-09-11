# SPDX-License-Identifier: Apache-2.0
"""Compressor -> indexer unroped latent exactness (PoolingCache path)."""
from __future__ import annotations

import mlx.core as mx


def test_ratio2_decode_emits_unroped_latent_on_complete_window():
    from omlx.patches.deepseek_v41 import apply_deepseek_v41_patch

    apply_deepseek_v41_patch()
    import mlx_lm.models.deepseek_v41 as m
    from mlx_lm.models.cache import PoolingCache

    args = m.ModelArgs(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        head_dim=16,
        qk_rope_head_dim=8,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        compress_ratios=[0, 2, 2],
        kv_source_layer_ids=[1],
        index_source_layer_ids=[1],
        candidate_source_layer_id=-1,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
        sliding_window=8,
        engram_layer_ids=[],
        dspark_block_size=0,
        n_mtp_layers=0,
        num_nextn_predict_layers=0,
        vision_enabled=False,
    )
    model = m.Model(args)
    layer = model.layers[1]
    attn = layer.attn
    assert attn.compressor is not None and attn.indexer is not None
    cache = model.make_cache()
    layer_cache = cache[1]
    kv_pool = layer_cache[1]
    assert isinstance(kv_pool, PoolingCache)

    # Prefill 3 tokens -> one complete window of 2, remainder 1
    x = mx.random.normal((1, 3, args.hidden_size))
    out = attn(x, mask=None, cache=layer_cache, _standard_mask=True)
    assert out.shape == (1, 3, args.hidden_size)
    assert kv_pool.pooled is not None
    assert kv_pool.pooled.shape[1] == 1  # one compressed token
    assert kv_pool.remainder == 1

    # One more decode token completes the second window
    x1 = mx.random.normal((1, 1, args.hidden_size))
    # offset from window cache
    out2 = attn(x1, mask=None, cache=layer_cache, _standard_mask=False)
    assert out2.shape[-1] == args.hidden_size
    assert kv_pool.pooled.shape[1] == 2
    assert kv_pool.remainder == 0


def test_shared_pool_identity_reuse_layers():
    from omlx.patches.deepseek_v41 import apply_deepseek_v41_patch

    apply_deepseek_v41_patch()
    import mlx_lm.models.deepseek_v41 as m

    args = m.ModelArgs(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=6,
        num_attention_heads=2,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        head_dim=16,
        qk_rope_head_dim=8,
        n_routed_experts=2,
        num_experts_per_tok=1,
        compress_ratios=[0, 0, 2, 2, 2, 2],
        kv_source_layer_ids=[2],
        index_source_layer_ids=[2, 4],
        candidate_source_layer_id=-1,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
        sliding_window=8,
        engram_layer_ids=[],
        dspark_block_size=0,
        n_mtp_layers=0,
        num_nextn_predict_layers=0,
        vision_enabled=False,
    )
    model = m.Model(args)
    caches = model.make_cache()
    # layer 2 owns kv pool; layer 3 reuse; layer 4 reindex shares kv
    kv2 = caches[2][1]
    kv3 = caches[3][1]
    kv4 = caches[4][1]
    assert kv2 is kv3 is kv4

