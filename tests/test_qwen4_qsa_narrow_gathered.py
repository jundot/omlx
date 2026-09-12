"""Narrow (2..15-row) batch-one text windows take the gathered prefill arm above a context threshold."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp import language as L  # noqa: E402
QSAKVCache = L.QSAKVCache

BUDGET = 2048


def _self():
    return SimpleNamespace(
        indexer=SimpleNamespace(token_budget=BUDGET),
        _batch_one_text_position_ids=L.Qwen4ExpAttention._batch_one_text_position_ids,
    )


def _cache(offset):
    cache = QSAKVCache()
    cache.offset = offset
    return cache


def _eligible(rows, offset, target_verify=False):
    x = mx.zeros((1, rows, 64), dtype=mx.bfloat16)
    return L.Qwen4ExpAttention._gathered_text_prefill_eligible(
        _self(), x, None, _cache(offset), None, None, target_verify
    )


def test_narrow_windows_gather_above_the_context_threshold(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN4_GATHERED_NARROW_MIN_CONTEXT", raising=False)
    threshold = L._gathered_narrow_min_context()
    assert threshold == 16384
    for rows in (2, 4, 8, 15):
        assert _eligible(rows, threshold)
        assert _eligible(rows, 82_000)
        assert not _eligible(rows, threshold - 1)
        assert not _eligible(rows, BUDGET)  # below the threshold: masked path


def test_wide_windows_keep_the_budget_rule():
    assert _eligible(16, BUDGET)  # offset + rows > budget
    assert _eligible(BUDGET + 1, 0)
    assert not _eligible(BUDGET, 0)  # whole prefix fits the budget (strictly greater)
    assert not _eligible(16, 0)
    assert not _eligible(1, 82_000)  # single rows belong to the decode arm


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_GATHERED_NARROW_MIN_CONTEXT", "4096")
    assert L._gathered_narrow_min_context() == 4096
    assert _eligible(4, 4096)
    assert not _eligible(4, 4095)
    monkeypatch.setenv("OMLX_QWEN4_GATHERED_NARROW_MIN_CONTEXT", "bogus")
    assert L._gathered_narrow_min_context() == 16384


def test_verify_rows_still_excluded_from_the_prefill_arm():
    assert not _eligible(4, 82_000, target_verify=True)
