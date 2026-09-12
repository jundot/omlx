# SPDX-License-Identifier: Apache-2.0
"""Qwen4 QSA indexer scores on the M5 tensor units (NAX): fused bf16 GEMM with fp32 accumulate,
ReLU, head-sum, 1/sqrt(D) and the pooled causal sentinel, matching the exact fp32 path up to
accumulation order and selecting the same top-k blocks."""
from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import qsa_fast  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (fast.is_native_available() and fast.has_symbol("qwen4_qsa_nax_indexer_scores")),
    reason="native Qwen4 NAX indexer scores not built",
)


def _reference(q, k, offset):
    """Exact fp32 path with the same pooled causal sentinel the kernels write."""
    scores = qsa_fast._portable_indexer_scores(q, k, 128)
    valid = mx.arange(k.shape[1])[None, None, :] < (offset + mx.arange(q.shape[1])[None, :, None] + 1) // 4
    return mx.where(valid, scores, mx.finfo(mx.float32).min)


def _topk_sets(scores, k):
    picks = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
    return mx.sort(picks, axis=-1)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "rows,blocks,offset",
    [
        (67, 521, 2048),        # ragged M and N (boundary tiles), mask inside the tile
        (256, 4096, 16384),     # aligned prefill chunk
        (2048, 32770, 131072 - 2048),  # 134k geometry, N two past a tile edge
        (32, 40, 100),          # every column masked for early rows
    ],
)
def test_nax_scores_match_exact_path_and_topk(dtype, rows, blocks, offset):
    mx.random.seed(rows)
    q = (mx.random.normal((1, rows, 4, 128)) * 0.3).astype(dtype)
    k = (mx.random.normal((1, blocks, 128)) * 0.3).astype(dtype)
    actual = fast.qwen4_qsa_nax_indexer_scores(
        q.transpose(0, 2, 1, 3), k[:, None], mask_ratio=4, mask_q_offset=offset
    )
    expected = _reference(q, k, offset)
    mx.eval(actual, expected)
    assert actual.shape == (1, rows, blocks) and actual.dtype == mx.float32
    sentinel = mx.finfo(mx.float32).min
    masked = expected == sentinel
    assert mx.array_equal(actual == sentinel, masked).item()          # sentinel bit-exact
    visible = mx.where(masked, 0.0, mx.abs(actual - expected))
    scale = float(mx.max(mx.where(masked, 0.0, mx.abs(expected))).item()) or 1.0
    assert float(mx.max(visible).item()) <= 2e-5 * scale               # accumulation-order noise only
    top = min(512, blocks)
    assert mx.array_equal(_topk_sets(actual, top), _topk_sets(expected, top)).item()


def test_nax_scores_symbol_is_part_of_extension_abi():
    assert "qwen4_qsa_nax_indexer_scores" in fast.NATIVE_SYMBOLS


def test_native_score_route_prefers_nax_and_kill_switch_restores_steel(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_QSA_NATIVE_SCORE_MIN_ROWS", "0")
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_SCORE_DISABLED", False)
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_SCORE_PROVEN", True)
    q = mx.zeros((1, 40, 4, 128), dtype=mx.bfloat16)
    k = mx.zeros((1, 600, 128), dtype=mx.bfloat16)
    calls = []
    monkeypatch.setattr(fast, "qwen4_qsa_nax_indexer_scores", lambda *a, **kw: calls.append("nax") or mx.zeros((1, 40, 600)))
    monkeypatch.setattr(fast, "qwen4_qsa_indexer_scores", lambda *a, **kw: calls.append("steel") or mx.zeros((1, 40, 600)))
    qsa_fast._native_indexer_scores(q, k, head_dim=128, compress_ratio=4, mask_q_offset=4000)
    monkeypatch.setenv("OMLX_QWEN4_QSA_NAX_SCORES", "0")
    qsa_fast._native_indexer_scores(q, k, head_dim=128, compress_ratio=4, mask_q_offset=4000)
    assert calls == ["nax", "steel"]
