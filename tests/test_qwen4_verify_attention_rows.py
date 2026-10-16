# SPDX-License-Identifier: Apache-2.0
"""Row-exact Lightning MTP verify rows through Qwen4 attention.

Every verify row must get the bits of the serial one-row decode step at its
position and leave the cache exactly as the serial steps would. Past the QSA
block budget (rank-three positions, below the gathered crossover) a serial
step attends through ``_masked_decode``: one-row block scores, the one-launch
selection kernel and the selected-keys SDPA. Below the budget it runs MLX's
vector SDPA over its whole prefix, on a kernel plan that depends on its key
count. Real attention shapes (2560 hidden, 24/2 heads of 256, a 4x128
indexer, 2048-token budget), synthetic 6-bit weights.
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


def _text_config():
    return qwen4_exp.TextConfig(
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


_CONFIG = _text_config()


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
        norm.weight = mx.random.normal(norm.weight.shape) * 0.1
    nn.quantize(attn, group_size=64, bits=6, class_predicate=lambda _, m: isinstance(m, nn.Linear))
    for _, module in attn.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.scales = module.scales.astype(mx.bfloat16)
            module.biases = module.biases.astype(mx.bfloat16)
    mx.eval(attn.parameters())
    return attn


def _positions(start: int, length: int) -> mx.array:
    """The served rank-three text positions below the gathered crossover."""
    seq = mx.arange(start, start + length, dtype=mx.int32)[None]
    return mx.broadcast_to(seq[None], (3, 1, length))


def _prefill(attn, tokens: int, seed: int, chunk: int = 4096):
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


def _verify(attn, cache, x):
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
    try:
        out = attn(
            x,
            mask="causal",
            cache=cache,
            position_ids=_positions(cache.offset, x.shape[1]),
            target_verify=True,
        )
        mx.eval(out)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
    return out


def _serial(attn, cache, x):
    rows = []
    for row in range(x.shape[1]):
        out = attn(
            x[:, row : row + 1],
            mask=None,
            cache=cache,
            position_ids=_positions(cache.offset, 1),
        )
        mx.eval(out)
        rows.append(out)
    return mx.concatenate(rows, axis=1)


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


def _record_selection(monkeypatch):
    """Every one-row selection (FP32 head scores, key count, token mask)."""
    calls = []
    original = language.Qwen4ExpQSAIndexer.aligned_row_mask

    def recording(self, scores, key_len):
        mask = original(self, scores, key_len)
        mx.eval(scores, mask)
        calls.append((mx.array(scores), key_len, mx.array(mask)))
        return mask

    monkeypatch.setattr(language.Qwen4ExpQSAIndexer, "aligned_row_mask", recording)
    return calls


def _bit_equal(a: mx.array, b: mx.array) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


# 2060: the window starts just past the sparse crossover; 16382: its rows
# straddle MLX's two-pass partition switch at 16384 keys (the multi-row MLX
# path gave row 0 of a two-row window the 16384-key plan); 24000: the served
# 24K case.
@pytest.mark.parametrize("context", [2060, 16382, 24000])
@pytest.mark.parametrize("rows", [2, 3, 8, 16])
def test_masked_verify_rows_equal_serial_decode_steps(monkeypatch, context, rows):
    attn = _attention(seed=3)
    cache = _prefill(attn, context, seed=4)
    selections = _record_selection(monkeypatch)
    for seed in (11, 12):
        mx.random.seed(seed + rows)
        x = (mx.random.normal((1, rows, _CONFIG.hidden_size)) * 0.5).astype(mx.bfloat16)

        verified = _verify(attn, cache, x)
        verified_state = _state(cache)
        verified_selections = selections[:]
        selections.clear()
        cache.trim(rows)

        serial = _serial(attn, cache, x)
        serial_state = _state(cache)
        serial_selections = selections[:]
        selections.clear()
        cache.trim(rows)

        assert _bit_equal(verified, serial)
        for got, want in zip(verified_state, serial_state):
            assert _bit_equal(got, want)
        # Each row's FP32 block scores and selected tokens: the multi-row
        # scores are a GEMM that rounds differently and can flip a selection
        # at the cut-off.
        assert len(verified_selections) == len(serial_selections) == rows
        for (scores, keys, mask), (want_scores, want_keys, want_mask) in zip(
            verified_selections, serial_selections
        ):
            assert keys == want_keys
            assert _bit_equal(scores, want_scores)
            assert _bit_equal(mask, want_mask)


def test_masked_verify_rollback_then_decode_matches_serial():
    """Accept one draft of a four-row window, roll the rest back, decode on."""
    attn = _attention(seed=5)
    reference = _prefill(attn, 24002, seed=6)
    cache = _prefill(attn, 24002, seed=6)
    mx.random.seed(21)
    window = (mx.random.normal((1, 4, _CONFIG.hidden_size)) * 0.5).astype(mx.bfloat16)
    follow = (mx.random.normal((1, 3, _CONFIG.hidden_size)) * 0.5).astype(mx.bfloat16)

    verified = _verify(attn, cache, window)
    cache.trim(2)  # rows 2 and 3 rejected
    resumed = _serial(attn, cache, follow)

    expected = _serial(attn, reference, window[:, :2])
    expected_resumed = _serial(attn, reference, follow)
    assert _bit_equal(verified[:, :2], expected)
    assert _bit_equal(resumed, expected_resumed)
    for got, want in zip(_state(cache), _state(reference)):
        assert _bit_equal(got, want)


# Below the block budget: 700 keys stays on MLX's one-pass vector kernel;
# windows at 1020/1022 put rows on both sides of its two-pass switch at 1024
# keys, which a shared multi-row call would hide.
@pytest.mark.parametrize("context", [700, 1020, 1022])
@pytest.mark.parametrize("rows", [2, 3, 4, 8, 16])
def test_dense_verify_rows_equal_serial_decode_steps(context, rows):
    attn = _attention(seed=8)
    cache = _prefill(attn, context, seed=9)
    for seed in (31, 32):
        mx.random.seed(seed + rows)
        x = (mx.random.normal((1, rows, _CONFIG.hidden_size)) * 0.5).astype(mx.bfloat16)
        verified = _verify(attn, cache, x)
        verified_state = _state(cache)
        cache.trim(rows)
        serial = _serial(attn, cache, x)
        serial_state = _state(cache)
        cache.trim(rows)
        assert _bit_equal(verified, serial)
        for got, want in zip(verified_state, serial_state):
            assert _bit_equal(got, want)
