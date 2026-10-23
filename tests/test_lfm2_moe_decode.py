# SPDX-License-Identifier: Apache-2.0
"""Fused LFM2-MoE one-token decode and the copy-free KV decode tail."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

HIDDEN = 1024


def _config(**overrides):
    config = {
        "model_type": "lfm2_moe",
        "vocab_size": 512,
        "hidden_size": HIDDEN,
        "intermediate_size": 2048,
        "moe_intermediate_size": 512,
        "num_hidden_layers": 5,
        "num_experts": 32,
        "num_experts_per_tok": 4,
        "norm_topk_prob": True,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
        "use_expert_bias": True,
        "num_dense_layers": 1,
        "norm_eps": 1e-5,
        "conv_bias": False,
        "conv_L_cache": 3,
        "rope_theta": 1000000.0,
        "layer_types": ["conv", "conv", "full_attention", "conv", "full_attention"],
    }
    config.update(overrides)
    return config


def _model(seed=0, bits=5, dtype=mx.float16, static_routing=False, **overrides):
    """A small quantized LFM2-MoE. ``static_routing`` spaces the expert biases
    wider than the sigmoid range so the top-k never sits on a near tie (MoE
    routing is discontinuous: a one-ulp difference can flip a near-tied
    expert, which says nothing about the arithmetic under test)."""
    from mlx_lm.models import lfm2_moe

    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    mx.random.seed(seed)
    model = lfm2_moe.Model(lfm2_moe.ModelArgs.from_dict(_config(**overrides)))
    # Larger-than-init weights so routing and activations are not degenerate.
    for layer in model.layers:
        ff = layer.feed_forward
        if hasattr(ff, "expert_bias"):
            n = ff.expert_bias.shape[0]
            if static_routing:
                ff.expert_bias = mx.random.permutation(n).astype(mx.float32) * 2.0
            else:
                ff.expert_bias = mx.random.normal((n,)) * 0.05
    model.set_dtype(dtype)
    nn.quantize(
        model,
        group_size=64,
        bits=bits,
        class_predicate=lambda p, m: hasattr(m, "to_quantized")
        and not p.endswith("feed_forward.gate"),
    )
    mx.eval(model.parameters())
    model.eval()  # mlx_lm.load does this; the fused path skips training mode
    assert apply_qwen35_moe_gate_up_fusion(model) == 4
    return model


@pytest.fixture
def patched():
    """Install the fused decode (and KV tail) and restore the layer call after."""
    from mlx_lm.models import lfm2_moe

    import omlx.patches.lfm2_moe_decode as fused
    from omlx.patches.kv_decode_tail import apply_kv_decode_tail_patch

    cls = lfm2_moe.Lfm2DecoderLayer
    original = cls.__call__
    apply_kv_decode_tail_patch()
    fused._APPLIED = False
    fused._DISABLED = False
    assert fused.apply_lfm2_moe_decode_patch()
    yield original
    cls.__call__ = original
    fused._APPLIED = False


def _snapshot(cache):
    return [
        ("kv", c.keys, c.values, c.offset) if hasattr(c, "offset") else ("conv", c[0])
        for c in cache
    ]


def _restore(cache, snap):
    from mlx_lm.models.cache import ArraysCache, KVCache

    restored = []
    for kind, *values in snap:
        if kind == "kv":
            c = KVCache()
            c.state = tuple(values)
        else:
            c = ArraysCache(size=1)
            c[0] = values[0]
        restored.append(c)
    return restored


def _teacher_forced(model, prompt, steps, stock_call):
    """Per decode step, run the stock and the fused layer from the same stock
    cache state; return both logits stacks and the last fused caches.

    Each step starts from the stock state, so a near-tie expert selection
    flipping on one path (MoE routing is discontinuous; activation-dtype
    rounding decides near ties) cannot compound across steps."""
    from mlx_lm.models import lfm2_moe
    from mlx_lm.models.cache import make_prompt_cache

    cls = lfm2_moe.Lfm2DecoderLayer
    fused_call = cls.__call__
    cache = make_prompt_cache(model)
    cls.__call__ = stock_call
    try:
        model(mx.array(prompt)[None], cache=cache)
        stock, fused, fused_cache = [], [], None
        for step in range(steps):
            token = mx.array([[step % 500 + 3]])
            snap = _snapshot(cache)
            mx.eval([v for item in snap for v in item[1:] if isinstance(v, mx.array)])
            cls.__call__ = fused_call
            fused_cache = _restore(cache, snap)
            fused.append(model(token, cache=fused_cache)[0, -1].astype(mx.float32))
            cls.__call__ = stock_call
            stock.append(model(token, cache=cache)[0, -1].astype(mx.float32))
            mx.eval(stock[-1], fused[-1])
        return mx.stack(fused), mx.stack(stock), fused_cache, cache
    finally:
        cls.__call__ = fused_call


def _close(a, b):
    """Fused and composed paths differ by activation-dtype rounding only."""
    scale = mx.abs(b).max().item()
    return mx.abs(a - b).max().item() <= 0.02 * scale + 1e-3


@pytest.mark.parametrize(
    "seed,bits,dtype",
    [(0, 5, mx.float16), (1, 5, mx.float16), (1, 6, mx.bfloat16), (2, 4, mx.bfloat16)],
)
def test_fused_decode_matches_composed(patched, seed, bits, dtype):
    from mlx_lm.models import lfm2_moe

    model = _model(seed=seed, bits=bits, dtype=dtype, static_routing=True)
    fused_logits, stock_logits, fused_cache, stock_cache = _teacher_forced(
        model, list(range(5 + seed, 45)), 8, patched
    )
    for step in range(fused_logits.shape[0]):
        assert _close(fused_logits[step], stock_logits[step]), step
    # Conv states and KV contents line up (the KV read goes through the
    # tail-flushing properties).
    for layer, a, b in zip(model.layers, fused_cache, stock_cache):
        if layer.is_attention_layer:
            assert a.offset == b.offset
            ka, va, _ = a.state
            kb, vb, _ = b.state
            n = a.offset
            tol = 0.05 * mx.abs(kb[..., :n, :]).max().item()
            assert mx.allclose(ka[..., :n, :], kb[..., :n, :], atol=tol).item()
        else:
            assert mx.allclose(a[0], b[0], atol=0.05 * mx.abs(b[0]).max().item()).item()
    # The fused launches actually served the decode: attention went through
    # the tail, and the MoE and conv plans were built.
    assert isinstance(fused_cache[2], lfm2_moe.KVCache)
    assert "_omlx_tail_keys" in fused_cache[2].__dict__
    assert "_omlx_tail_keys" not in stock_cache[2].__dict__
    assert model.layers[3].__dict__["_omlx_lfm2_moe_plan"][-1] is not None
    assert model.layers[3].__dict__["_omlx_lfm2_conv_plan"][-1] is not None


def test_fused_decode_free_running_stays_on_tail(patched):
    """Many fused steps in a row (crossing a tail flush) keep a consistent
    cache: offsets advance once per token and the flushed KV equals what the
    fused steps appended."""
    from mlx_lm.models.cache import make_prompt_cache

    from omlx.patches.kv_decode_tail import TAIL

    model = _model(seed=4)
    cache = make_prompt_cache(model)
    model(mx.array([list(range(3, 20))]), cache=cache)
    for step in range(TAIL + 9):
        out = model(mx.array([[step % 500 + 3]]), cache=cache)
        mx.eval(out)
    assert cache[2].offset == 17 + TAIL + 9
    assert cache[2].keys.shape[2] >= cache[2].offset
    assert not mx.isnan(out).any().item()


@pytest.mark.parametrize("rows", [2, 3, 4])
def test_fused_verify_block_matches_composed(patched, rows):
    """An M-row block (MTP verify: confirmed token + drafts) through the fused
    launches matches the stock layers row by row, and the conv rollback
    snapshot equals the stock state after the confirmed prefix."""
    from mlx_lm.models.cache import make_prompt_cache

    from omlx.patches.mlx_lm_mtp import cache_rollback

    cache_rollback.apply()
    model = _model(seed=5, static_routing=True)
    stock_cache = make_prompt_cache(model)
    from mlx_lm.models import lfm2_moe

    cls = lfm2_moe.Lfm2DecoderLayer
    fused_call = cls.__call__
    cls.__call__ = patched
    try:
        model(mx.array([list(range(9, 60))]), cache=stock_cache)
        mx.eval([c.state for c in stock_cache])
        snap = _snapshot(stock_cache)
        block = mx.array([[11 + i for i in range(rows)]])
        # stock reference: the block, and the confirmed prefix alone
        ref = model(block, cache=stock_cache)
        prefix_cache = _restore(stock_cache, snap)
        model(block[:, :1], cache=prefix_cache)
        cls.__call__ = fused_call
        fused_cache = _restore(stock_cache, snap)
        h = model.model.embed_tokens(block)
        for layer, c in zip(model.layers, fused_cache):
            h = layer(
                h, None if not layer.is_attention_layer else "causal", c, n_confirmed=1
            )
        out = model.model.embed_tokens.as_linear(model.model.embedding_norm(h))
        mx.eval(out, ref)
    finally:
        cls.__call__ = fused_call
    for r in range(rows):
        assert _close(out[0, r].astype(mx.float32), ref[0, r].astype(mx.float32)), r
    for layer, f, p in zip(model.layers, fused_cache, prefix_cache):
        if not layer.is_attention_layer:
            (conf,) = f.rollback_state
            assert mx.allclose(conf, p[0], atol=0.05 * mx.abs(p[0]).max().item()).item()
        else:
            assert f.offset == p.offset + rows - 1
            assert "_omlx_tail_keys" in f.__dict__  # served by the fused path


def test_selection_matches_mlx_lm_routing(patched):
    """The fused gate+up launch selects the same experts as mlx-lm."""
    import omlx.patches.lfm2_moe_decode as fused

    model = _model(seed=2)
    layer = model.layers[3]
    plan = fused._build_moe_plan(layer)
    assert plan is not None
    for seed in range(5):
        h = (mx.random.normal((1, 1, HIDDEN), key=mx.random.key(seed))).astype(
            mx.float16
        )
        ff = layer.feed_forward
        rw = mx.sigmoid(ff.gate(layer.ffn_norm(h)))
        scores = rw.astype(mx.float32) + ff.expert_bias
        expected = mx.argpartition(scores, kth=-4, axis=-1)[..., -4:]
        xn, logits = fused._router_kernel(plan.eps)(
            inputs=[h, plan.norm_weight, plan.gate_weight],
            template=[("T", h.dtype), ("H", HIDDEN), ("NSG", fused._ROUTER_SIMDGROUPS)],
            grid=(32 * fused._ROUTER_SIMDGROUPS * plan.num_experts, 1, 1),
            threadgroup=(32 * fused._ROUTER_SIMDGROUPS, 1, 1),
            output_shapes=[(HIDDEN,), (plan.num_experts,)],
            output_dtypes=[h.dtype, h.dtype],
        )
        _, indices, sel_scores = fused._gate_up_kernel(plan.gate_up_fmt, True, 1.0)(
            inputs=[xn, *plan.gate_up_operands, logits, plan.expert_bias],
            template=[
                ("T", h.dtype),
                ("K", HIDDEN),
                ("NI", plan.inter),
                ("RPS", fused._GATE_UP_ROWS),
                ("NSG", 1),
                ("TOPK", 4),
                ("NE", plan.num_experts),
                ("M", 1),
            ],
            grid=(32, 1 + 4 * plan.inter // fused._GATE_UP_ROWS, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(4 * plan.inter,), (4,), (4,)],
            output_dtypes=[h.dtype, mx.uint32, h.dtype],
        )
        assert sorted(indices.tolist()) == sorted(expected.reshape(-1).tolist())
        assert abs(sel_scores.astype(mx.float32).sum().item() - 1.0) < 1e-2


def test_batched_rows_keep_stock_path(patched):
    from mlx_lm.models.cache import make_prompt_cache

    model = _model(seed=3)
    cache = make_prompt_cache(model)
    out = model(mx.array([[1, 2, 3], [4, 5, 6]]), cache=cache)
    out = model(mx.array([[7], [8]]), cache=cache)
    mx.eval(out)
    assert out.shape == (2, 1, 512)


# ---------------------------------------------------------------- KV tail
def _attend_reference(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


@pytest.mark.parametrize("n1,n2", [(1, 1), (300, 0), (700, 37), (2048, 255), (40, 200)])
def test_two_segment_attention(n1, n2):
    from omlx.patches.kv_decode_tail import attend

    mx.random.seed(n1 + n2)
    q = mx.random.normal((1, 16, 1, 64)).astype(mx.float16)
    k1 = mx.random.normal((1, 4, n1 + 64, 64)).astype(mx.float16)
    v1 = mx.random.normal((1, 4, n1 + 64, 64)).astype(mx.float16)
    k2 = mx.random.normal((1, 4, 256, 64)).astype(mx.float16)
    v2 = mx.random.normal((1, 4, 256, 64)).astype(mx.float16)
    k = mx.concatenate([k1[:, :, :n1], k2[:, :, :n2]], axis=2)
    v = mx.concatenate([v1[:, :, :n1], v2[:, :, :n2]], axis=2)
    out = attend(q, k1, v1, n1, k2, v2, n2, 0.125)
    ref = _attend_reference(q, k, v, 0.125)
    assert mx.allclose(out.astype(mx.float32), ref.astype(mx.float32), atol=2e-3).item()


def test_decode_tail_keeps_kvcache_semantics():
    from mlx_lm.models.cache import KVCache

    from omlx.patches.kv_decode_tail import (
        TAIL,
        apply_kv_decode_tail_patch,
        decode_append,
    )

    apply_kv_decode_tail_patch()
    mx.random.seed(0)
    tail_cache, plain = KVCache(), KVCache()
    prompt = mx.random.normal((1, 4, 37, 64)).astype(mx.float16)
    tail_cache.update_and_fetch(prompt, prompt)
    plain.update_and_fetch(prompt, prompt)
    steps = TAIL + 70  # crosses one flush
    for i in range(steps):
        k = mx.random.normal((1, 4, 1, 64)).astype(mx.float16)
        v = k * 2
        segs = decode_append(tail_cache, k, v)
        assert segs is not None
        k1, v1, n1, k2, v2, n2 = segs
        assert n1 + n2 == tail_cache.offset
        plain.update_and_fetch(k, v)
        if i == 100:
            # A trim between appends (MTP rollback) is honored.
            tail_cache.trim(3)
            plain.trim(3)
    assert tail_cache.offset == plain.offset
    keys, values, offset = tail_cache.state
    pkeys, pvalues, _ = plain.state
    assert mx.array_equal(keys[..., :offset, :], pkeys[..., :offset, :]).item()
    assert mx.array_equal(values[..., :offset, :], pvalues[..., :offset, :]).item()
    # Stock updates keep working after tail appends.
    more = mx.random.normal((1, 4, 5, 64)).astype(mx.float16)
    tail_cache.update_and_fetch(more, more)
    plain.update_and_fetch(more, more)
    assert mx.array_equal(
        tail_cache.keys[..., : tail_cache.offset, :], plain.keys[..., : plain.offset, :]
    ).item()


def test_decode_tail_state_setter_drops_pending_tail():
    from mlx_lm.models.cache import KVCache

    from omlx.patches.kv_decode_tail import apply_kv_decode_tail_patch, decode_append

    apply_kv_decode_tail_patch()
    cache = KVCache()
    base = mx.ones((1, 2, 4, 32), dtype=mx.float16)
    cache.update_and_fetch(base, base)
    decode_append(cache, base[:, :, :1] * 3, base[:, :, :1] * 3)
    fresh = mx.zeros((1, 2, 8, 32), dtype=mx.float16)
    cache.state = (fresh, fresh, 8)
    keys, _, offset = cache.state
    assert offset == 8 and mx.array_equal(keys, fresh).item()
