# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import mlx.core as mx


def test_deferred_hc_attn_uses_prior_pre_mix():
    from omlx.patches.deepseek_v41 import apply_deepseek_v41_patch

    apply_deepseek_v41_patch()
    from omlx.patches.deepseek_v41.deferred_hc import (
        DeferredHyperConnection,
        hc_collapse,
        make_identity_pre_mix,
    )
    import mlx_lm.models.deepseek_v41 as m

    args = m.ModelArgs(num_hidden_layers=2, compress_ratios=[0, 0], hidden_size=64, hc_mult=4)
    hc = DeferredHyperConnection(args)
    B, L, H, D = 1, 3, args.hc_mult, args.hidden_size
    x = mx.random.normal((B, L, H, D))
    pre_mix = make_identity_pre_mix(x, args.hc_mult)
    collapsed = hc_collapse(x, pre_mix)
    assert collapsed.shape == (B, L, D)
    pre, post, comb = hc.mixes(x)
    assert pre.shape == (B, L, H)
    assert post.shape == (B, L, H)
    assert comb.shape == (B, L, H, H)


def test_shared_runtime_source_selection():
    from omlx.patches.deepseek_v41.shared_runtime import SharedAttentionRuntime

    rt = SharedAttentionRuntime()
    assert rt.kv_source_for(5, [2, 8, 14]) == 2
    assert rt.kv_source_for(8, [2, 8, 14]) == 8
    assert rt.index_source_for(30, [2, 8, 14, 20, 24, 28, 32, 36]) == 28


def test_tiny_window_only_forward():
    from omlx.patches.deepseek_v41 import apply_deepseek_v41_patch

    apply_deepseek_v41_patch()
    import mlx_lm.models.deepseek_v41 as m

    args = m.ModelArgs(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        head_dim=16,
        qk_rope_head_dim=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        compress_ratios=[0, 0],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        candidate_source_layer_id=-1,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
        sliding_window=16,
        engram_layer_ids=[],
        dspark_block_size=0,
        n_mtp_layers=0,
        num_nextn_predict_layers=0,
        vision_enabled=False,
    )
    model = m.Model(args)
    tokens = mx.array([[1, 2, 3, 4]])
    out = model(tokens)
    assert out.shape == (1, 4, args.vocab_size)
