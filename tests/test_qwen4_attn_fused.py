# SPDX-License-Identifier: Apache-2.0
"""Qwen4 dense-arm attention through ``attn_fused``'s kernels.

A decode row, and every row of a row-exact Lightning MTP verify window, below
the QSA block budget must keep the bits of the MLX path (grouped one-row
projections, q/k RMS norms, mlx-vlm's MRoPE kernel, MLX's vector SDPA on the
row's own plan, the sigmoid gate, ``o_proj``) and leave the same cache. Real
attention shapes (2560 hidden, 24/2 heads of 256, a 4x128 indexer, 2048-token
budget), synthetic 6-bit weights. The kernels are also checked in FP32, where
a BF16 output could hide a different summation order.
"""

from __future__ import annotations

import importlib

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
attn_fused = importlib.import_module("mlx_vlm.models.qwen4_exp.attn_fused")
qwen4_exp = importlib.import_module("mlx_vlm.models.qwen4_exp")

from omlx.patches import qwen35_verify_qmm  # noqa: E402
from omlx.patches.mlx_vlm_mtp import qwen35_verify_linear  # noqa: E402
from omlx.patches.qwen35_verify_sdpa_split import (  # noqa: E402
    apply_qwen35_verify_sdpa_split_patch,
)

qwen35_verify_qmm.apply_verify_qmm_patch()
qwen35_verify_linear.apply()
apply_qwen35_verify_sdpa_split_patch()
q35_language = importlib.import_module("mlx_vlm.models.qwen3_5.language")
q35_verifier = importlib.import_module("mlx_vlm.models.qwen3_5.speculative_verifier")

requires_kernels = pytest.mark.skipif(
    attn_fused._gpu_class() != "d", reason="MLX's vector SDPA plan is transcribed for 'd' GPUs"
)


def _float_cache_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """mlx-vlm's SDPA helper for a float KV cache."""
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask, sinks=sinks
    )


@pytest.fixture(autouse=True)
def _stock_sdpa_helper(monkeypatch):
    # Other suites install SDPA wrappers around fakes into these module
    # globals and leave them behind; pin the stock float-cache behavior.
    monkeypatch.setattr(q35_language, "scaled_dot_product_attention", _float_cache_sdpa)
    monkeypatch.setattr(
        q35_verifier, "scaled_dot_product_attention", _float_cache_sdpa, raising=False
    )


_CONFIG = qwen4_exp.TextConfig(
    model_type="qwen4_exp_text",
    hidden_size=2560,
    num_hidden_layers=4,
    num_attention_heads=24,
    num_key_value_heads=2,
    head_dim=256,
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=3,
    num_experts=4,
    num_experts_per_tok=2,
    shared_expert_intermediate_size=16,
    moe_intermediate_size=16,
    rms_norm_eps=1e-6,
    vocab_size=64,
    max_position_embeddings=262144,
    hc_count=2,
    hc_lowrank=8,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    ple_layer_ids=[],
    ple_embed_dim=32,
    ple_conv_kernel_size=3,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=17,
    make_ngram_vocab_size_divisible_by=4,
    split_ngram_parts=4,
    indexer_n_heads=4,
    indexer_kv_heads=1,
    indexer_head_dim=128,
    indexer_budget=2048,
    indexer_compress_ratio=4,
    eos_token_id=1,
    rope_parameters={
        "type": "default",
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10_000_000,
    },
)


def _attention(seed: int):
    mx.random.seed(seed)
    attn = language.Qwen4ExpAttention(_CONFIG)
    for _, module in attn.named_modules():
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            module.weight = (mx.random.normal(module.weight.shape) * fan_in**-0.5).astype(
                mx.bfloat16
            )
    for norm in (attn.q_norm, attn.k_norm, attn.indexer.q_layernorm, attn.indexer.k_layernorm):
        norm.weight = (mx.random.normal(norm.weight.shape) * 0.1).astype(mx.bfloat16)
    nn.quantize(attn, group_size=64, bits=6, class_predicate=lambda _, m: isinstance(m, nn.Linear))
    for _, module in attn.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.scales = module.scales.astype(mx.bfloat16)
            module.biases = module.biases.astype(mx.bfloat16)
    mx.eval(attn.parameters())
    return attn


def _positions(start: int, length: int) -> mx.array:
    seq = mx.arange(start, start + length, dtype=mx.int32)[None]
    return mx.broadcast_to(seq[None], (3, 1, length))


def _prefill(attn, tokens: int, seed: int):
    cache = language.QSAKVCache()
    if tokens:
        mx.random.seed(seed)
        x = mx.random.normal((1, tokens, _CONFIG.hidden_size)).astype(mx.bfloat16)
        attn(x, mask="causal", cache=cache, position_ids=_positions(0, tokens))
        mx.eval([a for a in cache.state if a is not None])
    return cache


def _step(attn, cache, x):
    """One served call: a decode row, or a row-exact verify window."""
    rows = x.shape[1]
    if rows == 1:
        out = attn(x, mask=None, cache=cache, position_ids=_positions(cache.offset, 1))
        mx.eval(out)
        return out
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
    try:
        out = attn(
            x,
            mask="causal",
            cache=cache,
            position_ids=_positions(cache.offset, rows),
            target_verify=True,
        )
        mx.eval(out)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
    return out


def _state(cache):
    arrays = [
        cache.keys[..., : cache.offset, :],
        cache.values[..., : cache.offset, :],
        cache.index_keys,
        cache.index_position_ids,
    ]
    copies = [mx.array(a) for a in arrays]
    mx.eval(copies)
    return copies


def _bit_equal(a: mx.array, b: mx.array) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


def _prefill_chunked(attn, tokens: int, seed: int, chunk: int = 4096):
    cache = language.QSAKVCache()
    mx.random.seed(seed)
    done = 0
    while done < tokens:
        size = min(chunk, tokens - done)
        x = mx.random.normal((1, size, _CONFIG.hidden_size)).astype(mx.bfloat16)
        attn(x, mask="causal", cache=cache, position_ids=_positions(done, size))
        mx.eval([a for a in cache.state if a is not None])
        done += size
    return cache


def _assert_fused_equals_mlx(monkeypatch, attn, cache, rows, seeds):
    """Each call through the kernels equals the MLX path: output and cache."""
    prologues = []
    prologue = attn._fused_prologue

    def counting(*args, **kwargs):
        prologues.append(args[0].shape[1])
        return prologue(*args, **kwargs)

    for seed in seeds:
        mx.random.seed(seed + rows)
        x = (mx.random.normal((1, rows, _CONFIG.hidden_size)) * 0.5).astype(mx.bfloat16)

        monkeypatch.setattr(attn, "_fused_prologue", counting)
        got = _step(attn, cache, x)
        got_state = _state(cache)
        cache.trim(rows)

        monkeypatch.setattr(attn_fused, "_DISABLED", True)
        want = _step(attn, cache, x)
        want_state = _state(cache)
        cache.trim(rows)
        monkeypatch.setattr(attn_fused, "_DISABLED", False)

        assert _bit_equal(got, want)
        for a, b in zip(got_state, want_state):
            assert _bit_equal(a, b)
    return prologues


# 0: the first token; 700: one-pass rows; 1016/1023: windows ending on and
# rows sitting right below MLX's two-pass switch at 1024 keys (a decode at
# 1023 sees 1024 keys); 1500: two-pass rows; 2043: the last window (eight
# rows) the indexer still leaves dense.
@requires_kernels
@pytest.mark.parametrize("context", [0, 700, 1016, 1023, 1500, 2043])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8])
def test_dense_rows_equal_mlx_path(monkeypatch, context, rows):
    attn = _attention(seed=3)
    cache = _prefill(attn, context, seed=4)
    prologues = _assert_fused_equals_mlx(monkeypatch, attn, cache, rows, (11, 12))
    # Windows whose rows share an SDPA plan take the kernels.
    if attn_fused.row_plan(context + 1, context + rows) is not None:
        assert prologues == [rows, rows]


# Past the block budget: a decode row (_masked_decode) and a row-exact verify
# window (_row_exact_masked_verify) keep the indexer's selection and the
# selected-keys SDPA around the fused projections, norms and MRoPE.
@requires_kernels
@pytest.mark.parametrize("context", [2060, 24000])
def test_masked_rows_equal_mlx_path(monkeypatch, context):
    attn = _attention(seed=7)
    cache = _prefill_chunked(attn, context, seed=8)
    for rows in (1, 2, 4, 8):
        prologues = _assert_fused_equals_mlx(monkeypatch, attn, cache, rows, (13,))
        assert prologues == [rows]


def test_fused_verify_rollback_then_decode_matches_mlx(monkeypatch):
    """Accept one draft of a four-row window, roll the rest back, decode on."""
    attn = _attention(seed=5)
    cache = _prefill(attn, 900, seed=6)
    reference = _prefill(attn, 900, seed=6)
    mx.random.seed(21)
    window = (mx.random.normal((1, 4, _CONFIG.hidden_size)) * 0.5).astype(mx.bfloat16)
    follow = (mx.random.normal((1, 3, _CONFIG.hidden_size)) * 0.5).astype(mx.bfloat16)

    verified = _step(attn, cache, window)
    cache.trim(3)  # rows 1..3 rejected
    resumed = [_step(attn, cache, follow[:, i : i + 1]) for i in range(3)]

    monkeypatch.setattr(attn_fused, "_DISABLED", True)
    expected = _step(attn, reference, window[:, :1])
    expected_resumed = [_step(attn, reference, follow[:, i : i + 1]) for i in range(3)]

    assert _bit_equal(verified[:, :1], expected)
    for got, want in zip(resumed, expected_resumed):
        assert _bit_equal(got, want)
    for got, want in zip(_state(cache), _state(reference)):
        assert _bit_equal(got, want)


def _mlx_sdpa_gate(queries, keys, values, gate, scale):
    """MLX's path: one vector SDPA per causal row, then the sigmoid gate."""
    _, heads, rows, head_dim = queries.shape
    total = keys.shape[2]
    outs = [
        mx.fast.scaled_dot_product_attention(
            queries[:, :, r : r + 1],
            keys[..., : total - rows + r + 1, :],
            values[..., : total - rows + r + 1, :],
            scale=scale,
        )
        for r in range(rows)
    ]
    out = mx.concatenate(outs, axis=2).transpose(0, 2, 1, 3).reshape(rows, -1)
    return out * mx.sigmoid(gate)


# FP32 end to end: every accumulation order shows in the output. Key counts
# cover one pass (1..1023, partial last chain groups) and two passes (1024+).
@requires_kernels
@pytest.mark.parametrize("keys", [1, 5, 33, 130, 1000, 1023, 1024, 1027, 1500, 2051])
@pytest.mark.parametrize("rows", [1, 3, 4, 8])
@pytest.mark.parametrize("spread", [1.0, 6.0])
def test_fp32_sdpa_gate_matches_mlx(keys, rows, spread):
    if keys < rows:
        pytest.skip("fewer keys than rows")
    plan = attn_fused.row_plan(keys - rows + 1, keys)
    if plan is None:
        pytest.skip("rows straddle a plan boundary")
    mx.random.seed(keys * 16 + rows)
    capacity = 2304
    key_buffer = mx.random.normal((1, 2, capacity, 256)) * spread
    value_buffer = mx.random.normal((1, 2, capacity, 256))
    queries = mx.random.normal((1, 24, rows, 256))
    gate = mx.random.normal((rows, 24 * 256)) * 4
    k = key_buffer[..., :keys, :]
    v = value_buffer[..., :keys, :]
    assert attn_fused.ready(mx.float32, 24, 2, 256, 64, 3)
    got = attn_fused.dense_sdpa_gate(queries, k, v, mx.sigmoid(gate), 0.0625, plan)
    want = _mlx_sdpa_gate(queries, k, v, gate, 0.0625)
    assert _bit_equal(got, want)


@requires_kernels
@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("pos_ndim", [2, 3])
def test_fp32_prep_matches_norms_and_mrope(rows, pos_ndim):
    attn = _attention(seed=9)
    mx.random.seed(rows + pos_ndim)
    q_proj = mx.random.normal((rows, 24 * 512)) * 4
    k_proj = mx.random.normal((rows, 2 * 256)) * 4
    start = 40000 + rows
    positions = _positions(start, rows) if pos_ndim == 3 else _positions(start, rows)[0]
    rotary = attn.rotary_emb

    queries = attn.q_norm(q_proj.reshape(1, rows, 24, 2, 256)[:, :, :, 0]).transpose(0, 2, 1, 3)
    keys = attn.k_norm(k_proj.reshape(1, rows, 2, 256)).transpose(0, 2, 1, 3)
    want_q, want_k = rotary.apply_rotary(queries, keys, positions, unsqueeze_dim=1)

    assert attn_fused.ready(mx.float32, 24, 2, 256, rotary.dim, pos_ndim)
    got_q, got_k = attn_fused.prep_qk(
        q_proj,
        k_proj,
        attn.q_norm._scale(),
        attn.k_norm._scale(),
        attn.q_norm.eps,
        positions,
        rotary.inv_freq,
        rotary.position_selector,
        heads=24,
        kv_heads=2,
        rotary_dim=rotary.dim,
    )
    assert _bit_equal(got_q, want_q)
    assert _bit_equal(got_k, want_k)
