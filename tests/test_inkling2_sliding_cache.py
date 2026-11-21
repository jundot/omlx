# SPDX-License-Identifier: Apache-2.0
"""Tests for bounded RotatingKVCache on Inkling 2 sliding-window visits."""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

from omlx.patches.inkling2.model import Model, ModelArgs, TextArgs


def _tiny_looped_model(window: int = 16) -> Model:
    text_cfg = dict(
        hidden_size=64,
        num_hidden_layers=6,
        vocab_size=128,
        unpadded_vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=16,
        sliding_window_size=window,
        rel_extent=window,
        d_rel=4,
        local_layer_ids=[0, 1, 2, 3, 4],
        n_layers_per_attn_res_block=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        intermediate_size=64,
        loop_config={
            "core_start": 2,
            "core_end": 4,
            "num_passes": 2,
            "gate_rank": 8,
            "is_varloop": False,
        },
        expert_formats=["nvfp4"] * 6,
    )
    args = ModelArgs(model_type="inkling2_loop", text_config=text_cfg, vision_config=None)
    mx.random.seed(7)
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _unbounded_cache(model: Model):
    return [CacheList(KVCache(), ArraysCache(4)) for _ in range(model.model.num_visits)]


def test_make_cache_uses_rotating_for_sliding_visits():
    model = _tiny_looped_model(window=16)
    cache = model.make_cache()
    assert len(cache) == model.model.num_visits
    for (idx, _), entry in zip(model.model._schedule, cache):
        kv = entry[0]
        if model.args.text.is_sliding(idx):
            assert isinstance(kv, RotatingKVCache)
            assert kv.max_size >= model.args.text.sliding_window_size
        else:
            assert type(kv) is KVCache


def test_sliding_cache_matches_unbounded_past_window():
    window = 16
    model = _tiny_looped_model(window=window)
    # Prompt length 40 > window + 16 so RotatingKVCache wraps around multiple times
    tokens = mx.array([[i % 120 for i in range(48)]], dtype=mx.int32)

    ref_cache = _unbounded_cache(model)
    rot_cache = model.make_cache()

    # Chunked prefill of 36 tokens
    for s in range(0, 36, 12):
        l_ref = model(tokens[:, s : s + 12], cache=ref_cache)
        l_rot = model(tokens[:, s : s + 12], cache=rot_cache)
        mx.eval(l_ref, l_rot)

    # Single-token decode for 12 more tokens across ring wrap-around
    max_diff = 0.0
    for pos in range(36, 48):
        tok = tokens[:, pos : pos + 1]
        l_ref = model(tok, cache=ref_cache)
        l_rot = model(tok, cache=rot_cache)
        mx.eval(l_ref, l_rot)
        diff = float(mx.max(mx.abs(l_ref - l_rot)))
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-4, f"RotatingKVCache diverged from unbounded KVCache: max_diff={max_diff}"
    # Verify sliding caches stayed bounded while global caches hold all 48 tokens
    for (idx, _), entry in zip(model.model._schedule, rot_cache):
        kv = entry[0]
        if model.args.text.is_sliding(idx):
            assert kv.keys.shape[2] <= kv.max_size
        else:
            assert kv.offset == 48


def test_sliding_cache_speculative_rollback_past_window():
    window = 16
    model = _tiny_looped_model(window=window)
    prefix = mx.array([[i % 120 for i in range(36)]], dtype=mx.int32)
    verify_block = mx.array([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=mx.int32)
    next_tok = mx.array([[99]], dtype=mx.int32)

    # Ground truth: prefix + first 3 tokens of verify_block (accepted=2 -> keep=3), then next_tok
    ref_cache = _unbounded_cache(model)
    mx.eval(model(prefix, cache=ref_cache))
    for i in range(3):
        mx.eval(model(verify_block[:, i : i + 1], cache=ref_cache))
    l_ref = model(next_tok, cache=ref_cache)
    mx.eval(l_ref)

    # Speculative path with RotatingKVCache: verify full 8-token block, accept 2 drafts (keep 3), rollback 5
    rot_cache = model.make_cache()
    mx.eval(model(prefix, cache=rot_cache))
    mx.eval(model(verify_block, cache=rot_cache))
    assert model.mtp_partial_rollback(rot_cache, accepted=2, num_drafts=7)
    l_rot = model(next_tok, cache=rot_cache)
    mx.eval(l_rot)

    # Ground truth 1: unbounded KVCache running the exact same verify + rollback
    ref_v_cache = _unbounded_cache(model)
    mx.eval(model(prefix, cache=ref_v_cache))
    mx.eval(model(verify_block, cache=ref_v_cache))
    assert model.mtp_partial_rollback(ref_v_cache, accepted=2, num_drafts=7)
    l_ref_v = model(next_tok, cache=ref_v_cache)
    mx.eval(l_ref_v)

    diff_exact = float(mx.max(mx.abs(l_ref_v - l_rot)))
    diff_seq = float(mx.max(mx.abs(l_ref - l_rot)))
    assert diff_exact < 1e-5, f"Rollback vs unbounded verify diverged: {diff_exact}"
    assert diff_seq < 1e-3, f"Rollback vs sequential decode diverged: {diff_seq}"
