# SPDX-License-Identifier: Apache-2.0
"""A short request left alone in a batch must decode as if it had never been batched.

Scenario from review of the Lightning MTP exactness change: a long and a short request share a batch, the long one
finishes first and is filtered out, and the short one keeps decoding. Its batch caches still carry the long request's
padding (pooled rows are right-padded to the batch's widest row, local rows left-padded). The stock decode path masks
that padding; the decode-consistent one-row path, enabled for every Lightning MTP load, must not attend to it either.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_map


@pytest.fixture(scope="module")
def dsv4():
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

    apply_deepseek_v4_patch()
    apply_mlx_lm_mtp_patch()
    import mlx_lm.models.deepseek_v4 as module

    return module


def _model(dsv4, decode_consistent: bool):
    config = dsv4.ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "num_hash_layers": 0,
            "q_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "head_dim": 16,
            "o_groups": 2,
            "o_lora_rank": 16,
            "index_n_heads": 2,
            "index_head_dim": 8,
            "index_topk": 4,
            "hc_mult": 4,
            "sliding_window": 8,
            # local-only, dense compressed (ratio 128: the exact path that ignored the mask) and sparse ratio 4
            "compress_ratios": [0, 128, 4],
            "num_nextn_predict_layers": 0,
        }
    )
    mx.random.seed(7)
    model = dsv4.Model(config)
    # random init is tiny-magnitude; scale it up so attention is not trivially uniform
    model.update(tree_map(lambda p: p * 4 if p.dtype in (mx.float32, mx.bfloat16) and p.ndim > 1 else p, model.parameters()))
    # run in bfloat16 like the real checkpoint: the exact kernels' bitwise contract is a bf16 one
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    for layer in model.model.layers if hasattr(model, "model") else model.layers:
        layer.attn.decode_consistent = decode_consistent
    return model


def _prefill(model, tokens):
    cache = model.make_cache()
    logits = model(mx.array([tokens]), cache=cache)
    mx.eval(logits)
    return cache, logits[:, -1]


def _merge(caches):
    from mlx_lm.models.cache import CacheList

    merged = []
    for layer_caches in zip(*caches):
        merged.append(type(layer_caches[0]).merge(list(layer_caches)))
    return merged


LONG = [(i * 7 + 3) % 64 for i in range(300)]   # 2 ratio-128 pooled rows, 75 ratio-4 rows


# 5 tokens: under the 8-token window, so local rows keep left padding; 140: one ratio-128 pooled row vs the long two
@pytest.mark.parametrize("short_len", [5, 140])
@pytest.mark.parametrize("decode_consistent", [False, True])
def test_short_request_after_long_one_leaves_batch(dsv4, short_len, decode_consistent):
    model = _model(dsv4, decode_consistent)
    short = [(i * 11 + 5) % 64 for i in range(short_len)]
    steps_together, steps_alone = 3, 12
    forced = [(i * 13 + 1) % 64 for i in range(steps_together + steps_alone)]

    # reference: the short request on its own, never batched
    ref_cache, _ = _prefill(model, short)
    reference = []
    for tok in forced:
        out = model(mx.array([[tok]]), cache=ref_cache)[:, -1]
        mx.eval(out)
        reference.append(out[0])

    # batched: prefill separately, merge, decode both rows, drop the long row, keep decoding the short one
    long_cache, _ = _prefill(model, LONG)
    short_cache, _ = _prefill(model, short)
    batch = _merge([long_cache, short_cache])
    long_forced = [(i * 17 + 2) % 64 for i in range(steps_together)]
    got = []
    for step in range(steps_together):
        out = model(mx.array([[long_forced[step]], [forced[step]]]), cache=batch)[:, -1]
        mx.eval(out)
        got.append(out[1])
    for layer_cache in batch:
        layer_cache.filter([1])
    for tok in forced[steps_together:]:
        out = model(mx.array([[tok]]), cache=batch)[:, -1]
        mx.eval(out)
        got.append(out[0])

    worst = max(float(mx.abs(g.astype(mx.float32) - r.astype(mx.float32)).max()) for g, r in zip(got, reference))
    scale = max(float(mx.abs(r.astype(mx.float32)).max()) for r in reference)
    argmax_ok = all(int(g.argmax()) == int(r.argmax()) for g, r in zip(got, reference))
    # rounding between batched and unbatched kernels is allowed; attending to padding is not
    assert worst <= 1e-2 * scale, f"short request drifted after the long one left: max |dlogit| {worst:.4f} (scale {scale:.3f})"
    assert argmax_ok


def _batched_short_logits(model, short, forced, steps_together):
    long_cache, _ = _prefill(model, LONG)
    short_cache, _ = _prefill(model, short)
    batch = _merge([long_cache, short_cache])
    long_forced = [(i * 17 + 2) % 64 for i in range(steps_together)]
    got = []
    for step in range(steps_together):
        out = model(mx.array([[long_forced[step]], [forced[step]]]), cache=batch)[:, -1]
        mx.eval(out)
        got.append(out[1])
    for layer_cache in batch:
        layer_cache.filter([1])
    for tok in forced[steps_together:]:
        out = model(mx.array([[tok]]), cache=batch)[:, -1]
        mx.eval(out)
        got.append(out[0])
    return got


@pytest.mark.parametrize("short_len", [5, 140])
def test_exact_path_is_bitwise_equal_to_stock_after_the_long_request_leaves(dsv4, short_len):
    """MTP on (decode-consistent exact path) must reproduce MTP off (stock path) bit for bit in the batch scenario:
    while padding remains both take the masked stock path; once it is gone the mask is dropped and the exact path
    matches the unmasked stock kernel."""
    short = [(i * 11 + 5) % 64 for i in range(short_len)]
    forced = [(i * 13 + 1) % 64 for i in range(3 + 20)]
    stock = _batched_short_logits(_model(dsv4, False), short, forced, 3)
    exact = _batched_short_logits(_model(dsv4, True), short, forced, 3)
    differing = [i for i, (a, b) in enumerate(zip(stock, exact)) if not mx.array_equal(a, b).item()]
    assert not differing, f"exact path differs from stock at steps {differing}"
