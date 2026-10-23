# SPDX-License-Identifier: Apache-2.0
"""Gathered (MLX-ops) sparse MLA prefill attention matches the exact reference."""

import mlx.core as mx
import pytest

from omlx.patches.glm_moe_dsa import sparse_mla


def _reference(q_latent, q_pe, kv_latent, k_pe, topk, scale, causal=True):
    """fp32 attention of every head over its query's selected keys."""
    _, H, L, DL = q_latent.shape
    K = kv_latent.shape[2]
    q = q_latent[0].astype(mx.float32)
    keys = kv_latent[0, 0].astype(mx.float32)
    if q_pe is not None:
        q = mx.concatenate([q, q_pe[0].astype(mx.float32)], axis=-1)
        keys = mx.concatenate([keys, k_pe[0, 0].astype(mx.float32)], axis=-1)
    idx = topk[0, 0].astype(mx.int32)
    valid = (idx >= 0) & (idx < K)
    if causal:
        valid = valid & (idx <= (mx.arange(L, dtype=mx.int32) + (K - L))[:, None])
    safe = mx.where(valid, idx, 0)
    kg = keys[safe]  # [L, topk, D]
    scores = mx.einsum("hld,ltd->lht", q, kg) * scale
    scores = mx.where(valid[:, None, :], scores, mx.array(-1e30))
    probs = mx.softmax(scores, axis=-1)
    out = mx.einsum("lht,ltd->lhd", probs, kg[..., :DL])
    return out.swapaxes(0, 1)[None]


def _ulp_distance(a, b):
    """Max |a-b| in units of the bf16 spacing at the reference's full scale."""
    import math

    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    full_scale = mx.max(mx.abs(b32)).item()
    fs_ulp = 2.0 ** (math.floor(math.log2(full_scale)) - 7)
    return mx.max(mx.abs(a32 - b32)).item() / fs_ulp


def _inputs(L, K, topk, H=8, DL=512, DP=64, with_pe=True, seed=0):
    mx.random.seed(seed)
    q_latent = (0.3 * mx.random.normal((1, H, L, DL))).astype(mx.bfloat16)
    kv_latent = (0.3 * mx.random.normal((1, 1, K, DL))).astype(mx.bfloat16)
    q_pe = k_pe = None
    if with_pe:
        q_pe = (0.3 * mx.random.normal((1, H, L, DP))).astype(mx.bfloat16)
        k_pe = (0.3 * mx.random.normal((1, 1, K, DP))).astype(mx.bfloat16)
    rows = []
    for i in range(L):
        avail = K - L + i + 1
        perm = mx.random.permutation(max(avail, topk))[:topk]
        rows.append(mx.sort(perm))
    idx = mx.stack(rows)[None, None].astype(mx.int32)
    return q_latent, q_pe, kv_latent, k_pe, idx


@pytest.mark.skipif(not sparse_mla._gathered_available(), reason="NAX GPU only")
@pytest.mark.parametrize("with_pe", [True, False])
def test_gathered_matches_fp32_reference(with_pe):
    L, K, topk = 200, 4096, 512
    q_latent, q_pe, kv_latent, k_pe, idx = _inputs(L, K, topk, with_pe=with_pe)
    scale = (512 + 64) ** -0.5
    ref = _reference(q_latent, q_pe, kv_latent, k_pe, idx, scale)
    out = sparse_mla.sparse_mla_attention_gathered(
        q_latent, q_pe, kv_latent, k_pe, idx, scale, q_block=64
    )
    assert out is not None
    assert out.shape == (1, 8, L, 512) and out.dtype == mx.bfloat16
    assert _ulp_distance(out, ref.astype(mx.bfloat16)) <= 2.0


@pytest.mark.skipif(not sparse_mla._gathered_available(), reason="NAX GPU only")
def test_gathered_masks_invalid_and_future_keys():
    L, K, topk = 96, 2048, 256
    q_latent, q_pe, kv_latent, k_pe, idx = _inputs(L, K, topk, with_pe=False)
    # poison: some negative slots and some keys past the query position
    idx = idx.astype(mx.int32)
    idx[..., :5] = -1
    idx[..., 5:9] = K - 1  # future for every query but the last
    scale = 512**-0.5
    ref = _reference(q_latent, None, kv_latent, None, idx, scale)
    out = sparse_mla.sparse_mla_attention_gathered(
        q_latent, None, kv_latent, None, idx, scale, q_block=32
    )
    assert _ulp_distance(out, ref.astype(mx.bfloat16)) <= 2.0
    assert mx.all(mx.isfinite(out.astype(mx.float32))).item()


@pytest.mark.skipif(not sparse_mla._gathered_available(), reason="NAX GPU only")
def test_gathered_matches_custom_kernel():
    if not hasattr(sparse_mla.glm_fast, "glm_dsa_sparse_mla_attention"):
        pytest.skip("custom kernel unavailable")
    L, K, topk = 128, 4096, 2048
    q_latent, q_pe, kv_latent, k_pe, idx = _inputs(L, K, topk, H=64, with_pe=True)
    scale = 576**-0.5
    # the production dispatcher with the gathered path switched off
    real_enabled = sparse_mla._gathered_enabled
    sparse_mla._gathered_enabled = lambda: False
    try:
        kern = sparse_mla.sparse_mla_attention(
            q_latent, q_pe, kv_latent, k_pe, idx, scale
        )
    finally:
        sparse_mla._gathered_enabled = real_enabled
    assert kern is not None
    out = sparse_mla.sparse_mla_attention_gathered(
        q_latent, q_pe, kv_latent, k_pe, idx, scale
    )
    ref = _reference(q_latent, q_pe, kv_latent, k_pe, idx, scale).astype(mx.bfloat16)
    assert _ulp_distance(out, ref) <= 2.0
    assert _ulp_distance(kern, ref) <= 2.0


def test_dispatcher_prefers_gathered_and_falls_back(monkeypatch):
    L, K, topk = 32, 4096, 2048  # a shape the custom kernel accepts too
    q_latent, q_pe, kv_latent, k_pe, idx = _inputs(L, K, topk, H=64, with_pe=False)
    calls = []

    def fake_gathered(*args, **kwargs):
        calls.append("gathered")
        return mx.zeros((1, 64, L, 512), dtype=mx.bfloat16)

    monkeypatch.setattr(sparse_mla, "_gathered_enabled", lambda: True)
    monkeypatch.setattr(sparse_mla, "sparse_mla_attention_gathered", fake_gathered)
    out = sparse_mla.sparse_mla_attention(
        q_latent, None, kv_latent, None, idx, 1.0
    )
    assert calls == ["gathered"] and out.shape == (1, 64, L, 512)
    # gathered path disabled: the custom kernel runs (pe columns zero-filled)
    calls.clear()
    monkeypatch.setattr(sparse_mla, "_gathered_enabled", lambda: False)
    if not hasattr(sparse_mla.glm_fast, "glm_dsa_sparse_mla_attention"):
        return
    out = sparse_mla.sparse_mla_attention(
        q_latent, None, kv_latent, None, idx, 1.0
    )
    assert calls == [] and out is not None and out.shape == (1, 64, L, 512)


def test_gathered_path_is_opt_in(monkeypatch):
    """The gathered path rounds scores and probabilities to bf16, so the
    dispatcher only takes it when OMLX_GLM_SPARSE_MLA_GATHERED=1."""
    import os

    if os.environ.get("OMLX_GLM_SPARSE_MLA_GATHERED"):
        pytest.skip("OMLX_GLM_SPARSE_MLA_GATHERED set in the environment")
    assert sparse_mla._GATHERED_ENABLED is False
    L, K, topk = 32, 4096, 2048
    q_latent, _, kv_latent, _, idx = _inputs(L, K, topk, H=64, with_pe=False)
    calls = []

    def fake_gathered(*args, **kwargs):
        calls.append("gathered")
        return mx.zeros((1, 64, L, 512), dtype=mx.bfloat16)

    monkeypatch.setattr(sparse_mla, "_gathered_available", lambda: True)
    monkeypatch.setattr(sparse_mla, "sparse_mla_attention_gathered", fake_gathered)
    sparse_mla.sparse_mla_attention(q_latent, None, kv_latent, None, idx, 1.0)
    assert calls == []
