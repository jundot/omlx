# SPDX-License-Identifier: Apache-2.0
"""YaRN RoPE scaling regressions for the vendored Qwen4-Exp stack.

Qwen's official long-context recipe for Qwen3.8-Flash-Next (262,144 native
→ up to 1,000,000 tokens) is static YaRN declared in
``text_config.rope_parameters`` — the same schema vLLM/SGLang/TokenSpeed
consume. The pinned mlx-vlm ``MRoPERotaryEmbedding`` builds plain
``compute_inv_freq(dim, base)`` frequencies and never dispatches on the
rope type, so ``omlx.patches.mlx_vlm_qwen4_exp_compat.yarn_rope`` applies
the correction in place.

The numeric reference below is an independent float64 NumPy transcription
of the HuggingFace Transformers ``ROPE_INIT_FUNCTIONS["yarn"]`` formulas
(which mlx-lm's ``YarnRoPE`` and the vLLM/SGLang implementations mirror).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from omlx.patches.mlx_vlm_qwen4_exp_compat import yarn_rope

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding  # noqa: E402
from mlx_vlm.models.qwen4_exp import TextConfig  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    Qwen4ExpAttention,
    Qwen4ExpMTPModule,
)
from mlx_vlm.models.rope_utils import compute_inv_freq  # noqa: E402

# Qwen's published 512k recipe (factor 2.0; the 1M recipe uses 4.0) on the
# production rotary geometry: 64 rotary dims at theta 10M, mrope [11,11,10].
# The tiny test config reaches the same 64 rotary dims with head_dim=128 and
# partial_rotary_factor=0.5 (production is 256 * 0.25).
QWEN_512K_RECIPE = {
    "type": "yarn",
    "rope_theta": 10_000_000,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.5,
    "factor": 2.0,
    "original_max_position_embeddings": 262144,
}
DEFAULT_RECIPE = {
    "type": "default",
    "rope_theta": 10_000_000,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.5,
}
ROTARY_DIM = 64
ROPE_THETA = 10_000_000.0
NATIVE_CTX = 262144


# ---------------------------------------------------------------------------
# Independent reference: HF transformers ROPE_INIT_FUNCTIONS["yarn"]
# ---------------------------------------------------------------------------


def _reference_yarn_wavelengths(
    dim: int,
    base: float,
    factor: float,
    original_max_position_embeddings: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> np.ndarray:
    def correction_dim(num_rotations: float) -> float:
        return (
            dim
            * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip(
        (np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0.0, 1.0
    )
    mask = 1.0 - ramp
    freq_extra = np.float64(base) ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    freq_inter = factor * freq_extra
    return (freq_inter * freq_extra) / (freq_inter * mask + freq_extra * (1.0 - mask))


def _reference_mscale(
    factor: float, mscale: float = 1.0, mscale_all_dim: float = 0.0
) -> float:
    def g(scale: float, m: float) -> float:
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

    return g(factor, mscale) / g(factor, mscale_all_dim)


def _reference_inv_freq(factor: float) -> np.ndarray:
    return 1.0 / _reference_yarn_wavelengths(ROTARY_DIM, ROPE_THETA, factor, NATIVE_CTX)


# ---------------------------------------------------------------------------
# Frequency-table and mscale math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_yarn_inv_freq_matches_transformers_reference(factor):
    got = np.asarray(
        yarn_rope.yarn_inv_freq(ROTARY_DIM, ROPE_THETA, factor, NATIVE_CTX).tolist(),
        dtype=np.float64,
    )
    want = _reference_inv_freq(factor)
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_yarn_inv_freq_shape_and_dtype():
    inv = yarn_rope.yarn_inv_freq(ROTARY_DIM, ROPE_THETA, 2.0, NATIVE_CTX)
    assert inv.shape == (ROTARY_DIM // 2,)
    assert inv.dtype == mx.float32


def test_attention_scaling_matches_reference():
    assert yarn_rope.yarn_attention_scaling(2.0) == pytest.approx(
        _reference_mscale(2.0)
    )
    assert yarn_rope.yarn_attention_scaling(4.0) == pytest.approx(
        0.1 * math.log(4.0) + 1.0
    )
    # mscale == mscale_all_dim cancels the temperature (DeepSeek-style off).
    assert yarn_rope.yarn_attention_scaling(2.0, mscale=1.0, mscale_all_dim=1.0) == (
        pytest.approx(1.0)
    )
    # factor <= 1 never scales.
    assert yarn_rope.yarn_attention_scaling(1.0) == 1.0
    assert yarn_rope.yarn_attention_scaling(0.5) == 1.0


# ---------------------------------------------------------------------------
# Module wiring
# ---------------------------------------------------------------------------


def _text_config(rope_parameters: dict):
    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
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
        num_key_value_heads=1,
        max_position_embeddings=NATIVE_CTX,
        hc_count=2,
        hc_lowrank=8,
        head_dim=128,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=1,
        rope_parameters=dict(rope_parameters),
    )


def _bare_rotary():
    return Qwen3_5RotaryEmbedding(
        ROTARY_DIM,
        max_position_embeddings=NATIVE_CTX,
        base=ROPE_THETA,
        mrope_section=[11, 11, 10],
    )


def test_default_rope_parameters_are_untouched():
    attn = Qwen4ExpAttention(_text_config(DEFAULT_RECIPE))
    assert mx.array_equal(
        attn.rotary_emb.inv_freq, compute_inv_freq(ROTARY_DIM, ROPE_THETA)
    )
    assert attn.rotary_emb.attention_scaling == 1.0


def test_yarn_applies_to_attention_and_shared_indexer():
    attn = Qwen4ExpAttention(_text_config(QWEN_512K_RECIPE))
    expected = yarn_rope.yarn_inv_freq(ROTARY_DIM, ROPE_THETA, 2.0, NATIVE_CTX)
    assert mx.array_equal(attn.rotary_emb.inv_freq, expected)
    # The QSA indexer must rotate block-start positions with the SAME table.
    assert attn.indexer.rotary_emb is attn.rotary_emb
    assert attn.rotary_emb.attention_scaling == pytest.approx(0.1 * math.log(2.0) + 1.0)
    # mscale != 1: the fused Metal kernel ignores attention_scaling, so the
    # module must fall back to the eager path that bakes it into cos/sin.
    assert attn.rotary_emb.fused_apply is False


def test_mtp_head_inherits_yarn_rope():
    mtp = Qwen4ExpMTPModule(_text_config(QWEN_512K_RECIPE))
    attn = mtp.layers[0].self_attn
    expected = yarn_rope.yarn_inv_freq(ROTARY_DIM, ROPE_THETA, 2.0, NATIVE_CTX)
    assert mx.array_equal(attn.rotary_emb.inv_freq, expected)
    assert attn.rotary_emb.attention_scaling == pytest.approx(0.1 * math.log(2.0) + 1.0)


def test_unit_mscale_keeps_fused_path_and_still_corrects_freqs():
    emb = _bare_rotary()
    before = emb.fused_apply
    params = dict(QWEN_512K_RECIPE, mscale=1.0, mscale_all_dim=1.0)
    assert yarn_rope.maybe_apply_yarn(emb, params) is True
    assert emb.attention_scaling == pytest.approx(1.0)
    assert emb.fused_apply == before
    assert mx.array_equal(
        emb.inv_freq, yarn_rope.yarn_inv_freq(ROTARY_DIM, ROPE_THETA, 2.0, NATIVE_CTX)
    )


def test_rope_type_alias_is_accepted():
    params = dict(QWEN_512K_RECIPE)
    del params["type"]
    params["rope_type"] = "yarn"
    emb = _bare_rotary()
    assert yarn_rope.maybe_apply_yarn(emb, params) is True
    assert mx.array_equal(
        emb.inv_freq, yarn_rope.yarn_inv_freq(ROTARY_DIM, ROPE_THETA, 2.0, NATIVE_CTX)
    )


def test_unsupported_rope_type_raises():
    emb = _bare_rotary()
    params = dict(QWEN_512K_RECIPE, type="llama3")
    with pytest.raises(ValueError, match="llama3"):
        yarn_rope.maybe_apply_yarn(emb, params)


def test_yarn_missing_required_keys_raises():
    emb = _bare_rotary()
    with pytest.raises(ValueError, match="factor"):
        yarn_rope.maybe_apply_yarn(
            emb, {"type": "yarn", "original_max_position_embeddings": NATIVE_CTX}
        )
    with pytest.raises(ValueError, match="original_max_position_embeddings"):
        yarn_rope.maybe_apply_yarn(emb, {"type": "yarn", "factor": 2.0})


def test_non_dict_rope_parameters_is_noop():
    emb = _bare_rotary()
    assert yarn_rope.maybe_apply_yarn(emb, None) is False
    assert emb.attention_scaling == 1.0


# ---------------------------------------------------------------------------
# Functional parity: eager apply_rotary vs the reference rotation
# ---------------------------------------------------------------------------


def _reference_rotate(x: np.ndarray, factor: float, start: int) -> np.ndarray:
    """Half-split rotation of the first ROTARY_DIM lanes with yarn cos/sin."""
    length = x.shape[-2]
    inv = _reference_inv_freq(factor).astype(np.float32)
    scale = np.float32(_reference_mscale(factor))
    pos = np.arange(start, start + length, dtype=np.float32)
    theta = pos[:, None] * inv[None, :]
    cos = (np.cos(theta) * scale)[None, None]
    sin = (np.sin(theta) * scale)[None, None]
    xr, xp = x[..., :ROTARY_DIM], x[..., ROTARY_DIM:]
    x1, x2 = xr[..., : ROTARY_DIM // 2], xr[..., ROTARY_DIM // 2 :]
    rotated = np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return np.concatenate([rotated, xp], axis=-1)


@pytest.mark.parametrize(
    ("start", "rtol", "atol"),
    [
        (0, 1e-5, 1e-6),
        # Beyond the native horizon: float32 position*freq products carry
        # ~1e-2 rad of argument jitter at 300k on the highest-frequency
        # lane, in production as much as here — keep the tolerance loose
        # enough for trig noise, tight enough that wrong frequencies or a
        # dropped mscale (both O(1) errors) still fail.
        (300_000, 5e-2, 5e-2),
    ],
)
def test_apply_rotary_matches_reference(start, rtol, atol):
    emb = _bare_rotary()
    assert yarn_rope.maybe_apply_yarn(emb, QWEN_512K_RECIPE) is True

    head_dim = 128  # 64 rotary + 64 passthrough, as in production geometry
    rng = np.random.default_rng(0)
    length = 8
    q_np = rng.standard_normal((1, 2, length, head_dim)).astype(np.float32)
    k_np = rng.standard_normal((1, 2, length, head_dim)).astype(np.float32)
    q, k = mx.array(q_np), mx.array(k_np)
    positions = mx.arange(start, start + length, dtype=mx.int32)[None, :]

    q_out, k_out = emb.apply_rotary(q, k, positions)

    np.testing.assert_allclose(
        np.asarray(q_out.tolist()),
        _reference_rotate(q_np, 2.0, start),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(k_out.tolist()),
        _reference_rotate(k_np, 2.0, start),
        rtol=rtol,
        atol=atol,
    )
    # Passthrough lanes are bit-exact copies.
    assert mx.array_equal(q_out[..., ROTARY_DIM:], q[..., ROTARY_DIM:])
    assert mx.array_equal(k_out[..., ROTARY_DIM:], k[..., ROTARY_DIM:])
