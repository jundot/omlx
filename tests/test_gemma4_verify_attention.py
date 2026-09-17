# SPDX-License-Identifier: Apache-2.0
"""Tests for the gemma4 decomposed small-L verify attention patch.

Uses a tiny random-init gemma4 text backbone: parity between the stock
multi-token forward and the decomposed route is checked on logits, and the
scope gates (L range, KV-sharing backbones, left padding) are exercised.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.gemma4")

from omlx.patches import gemma4_verify_attention

TINY_TEXT_CONFIG = {
    "model_type": "gemma4_text",
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "head_dim": 16,
    "global_head_dim": 16,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 2,
    "num_kv_shared_layers": 0,
    "vocab_size": 128,
    "sliding_window": 8,
    "sliding_window_pattern": 2,
    "attention_k_eq_v": True,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "final_logit_softcapping": None,
}


def _language_model(extra: dict | None = None):
    from mlx_vlm.models.gemma4.config import TextConfig
    from mlx_vlm.models.gemma4.language import LanguageModel

    params = dict(TINY_TEXT_CONFIG)
    if extra:
        params.update(extra)
    return LanguageModel(TextConfig.from_dict(params))


@pytest.mark.parametrize(
    "head_dim,query_len,gqa_factor,expected",
    [
        (256, 2, 8, True),
        (256, 4, 8, True),
        (256, 5, 8, False),
        (192, 2, 8, False),
        (96, 12, 8, True),
        (512, 2, 8, False),
    ],
)
def test_mlx0322_native_fused_shapes_are_not_intercepted(
    head_dim, query_len, gqa_factor, expected
):
    assert (
        gemma4_verify_attention._mlx0322_default_fused(
            head_dim, query_len, gqa_factor
        )
        is expected
    )


def _run(lm, prompt_len: int, step_len: int, patched: bool):
    """Prefill ``prompt_len`` tokens then run one ``step_len`` forward."""
    mx.random.seed(7)
    tokens = mx.random.randint(0, 100, (1, prompt_len + step_len))
    cache = lm.make_cache()
    out = lm(tokens[:, :prompt_len], cache=cache)
    mx.eval(out.logits)
    result = lm(tokens[:, prompt_len:], cache=cache).logits
    mx.eval(result)
    del patched
    return result


@pytest.fixture(autouse=True)
def _applied():
    assert gemma4_verify_attention.apply()
    assert gemma4_verify_attention.apply()  # idempotent
    yield


@pytest.mark.parametrize("step_len", [2, 3])
def test_decomposed_matches_stock_logits(step_len):
    # Same weights, same tokens: run the small-L forward through the
    # decomposed route and through the stock path (forced via the
    # kv-sharing gate on a config clone) — logits must agree.
    from mlx_vlm.models.gemma4 import language as g4_lang

    lm = _language_model()
    # Prompt long enough to rotate the sliding ring (window 8).
    got = _run(lm, prompt_len=24, step_len=step_len, patched=True)

    # Stock reference: bypass the route by restoring the original call.
    original = None
    for klass in type(lm.model.layers[0].self_attn).__mro__:
        if "_omlx_verify_attn_patched" in klass.__dict__:
            original = klass
            break
    assert original is g4_lang.Attention
    # Temporarily disable by widening the L gate to an impossible range.
    old_min = gemma4_verify_attention._MIN_L
    gemma4_verify_attention._MIN_L = 99
    try:
        ref = _run(lm, prompt_len=24, step_len=step_len, patched=False)
    finally:
        gemma4_verify_attention._MIN_L = old_min

    assert mx.allclose(got, ref, atol=2e-2, rtol=2e-2)
    assert (
        mx.argmax(got[0, -1]).item() == mx.argmax(ref[0, -1]).item()
    )


def _count_single_token_updates(lm, prompt_len: int, step_len: int) -> int:
    """Count 1-token ``update_and_fetch`` calls during the step forward.

    The decomposed route feeds the cache one token at a time (L calls per
    layer); the stock path updates once with the full L-token chunk. The
    probe wraps the cache instances directly — the patch's closure-bound
    sdpa symbol cannot be intercepted from outside.
    """
    mx.random.seed(7)
    tokens = mx.random.randint(0, 100, (1, prompt_len + step_len))
    cache = lm.make_cache()
    out = lm(tokens[:, :prompt_len], cache=cache)
    mx.eval(out.logits)

    single = {"n": 0}
    for c in cache:
        original = c.update_and_fetch

        def wrapper(k, v, _orig=original):
            if k.shape[2] == 1:
                single["n"] += 1
            return _orig(k, v)

        c.update_and_fetch = wrapper

    result = lm(tokens[:, prompt_len:], cache=cache).logits
    mx.eval(result)
    return single["n"]


def test_kv_sharing_backbones_stay_on_stock_path():
    # E2B/E4B-style backbones (num_kv_shared_layers > 0) must never take
    # the decomposed route: their donors feed downstream shared layers.
    lm = _language_model({"num_kv_shared_layers": 2})
    assert _count_single_token_updates(lm, prompt_len=12, step_len=2) == 0


def test_l_gate_routes_only_small_steps():
    lm = _language_model()
    # head_dim 16 keeps the fused kernel out (not lane-splittable), so
    # L=4 is out of range -> stock multi-token update.
    assert _count_single_token_updates(lm, prompt_len=12, step_len=4) == 0
    # L=2 routes: one single-token update per token per cached layer
    # (sliding layers by design; full layers as the kernel fallback since
    # head_dim 16 is not lane-splittable).
    n_layers = len(lm.make_cache())
    assert (
        _count_single_token_updates(lm, prompt_len=12, step_len=2)
        == 2 * n_layers
    )


# --- fused kernel route (head_dim % 32 == 0 -> global layers) ---------------

KERNEL_TEXT_CONFIG = dict(
    TINY_TEXT_CONFIG,
    head_dim=32,
    global_head_dim=32,
)


def _count_fused_calls(monkeypatch, lm, prompt_len: int, step_len: int) -> int:
    from omlx.patches import gemma4_verify_kernel as gvk

    gvk.is_available()  # warm the probe (it calls fused_verify_sdpa itself)
    calls = {"n": 0}
    original = gvk.fused_verify_sdpa

    def wrapper(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gvk, "fused_verify_sdpa", wrapper)
    mx.random.seed(7)
    tokens = mx.random.randint(0, 100, (1, prompt_len + step_len))
    cache = lm.make_cache()
    out = lm(tokens[:, :prompt_len], cache=cache)
    mx.eval(out.logits)
    result = lm(tokens[:, prompt_len:], cache=cache).logits
    mx.eval(result)
    return calls["n"]


def _n_full_layers(lm) -> int:
    return sum(
        1 for layer in lm.model.layers if layer.layer_type == "full_attention"
    )


def test_kernel_routes_global_layers(monkeypatch):
    pytest.importorskip("mlx.core").metal.is_available() or pytest.skip(
        "requires Metal"
    )
    lm = _language_model(KERNEL_TEXT_CONFIG)
    n_full = _n_full_layers(lm)
    assert n_full > 0
    # L=2: full layers take the fused kernel, sliding layers per-token.
    assert _count_fused_calls(monkeypatch, lm, prompt_len=24, step_len=2) == n_full
    # L=4: beyond the per-token ceiling, still fused on full layers.
    assert _count_fused_calls(monkeypatch, lm, prompt_len=24, step_len=4) == n_full
    # Past the kernel ceiling everything is stock.
    assert (
        _count_fused_calls(
            monkeypatch, lm, prompt_len=24, step_len=_KERNEL_MAX_L_PLUS_ONE
        )
        == 0
    )


_KERNEL_MAX_L_PLUS_ONE = gemma4_verify_attention._KERNEL_MAX_L + 1


@pytest.mark.parametrize("step_len", [2, 4, 5])
def test_kernel_route_matches_stock_logits(step_len):
    pytest.importorskip("mlx.core").metal.is_available() or pytest.skip(
        "requires Metal"
    )
    lm = _language_model(KERNEL_TEXT_CONFIG)
    got = _run(lm, prompt_len=24, step_len=step_len, patched=True)

    old_min = gemma4_verify_attention._MIN_L
    gemma4_verify_attention._MIN_L = 99
    try:
        ref = _run(lm, prompt_len=24, step_len=step_len, patched=False)
    finally:
        gemma4_verify_attention._MIN_L = old_min

    assert mx.allclose(got, ref, atol=2e-2, rtol=2e-2)
    assert mx.argmax(got[0, -1]).item() == mx.argmax(ref[0, -1]).item()
