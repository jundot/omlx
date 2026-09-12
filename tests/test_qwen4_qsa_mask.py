# SPDX-License-Identifier: Apache-2.0
"""Exact QSA block-mask expansion, including sparse-budget and tail boundaries."""

import logging
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import qsa_mask  # noqa: E402


def _reference_mask(hits, counts, ends, ratio, topk, key_len):
    import mlx.core as mx

    batch, seq, blocks = hits.shape
    selected = mx.repeat(hits, ratio, axis=-1)
    if blocks * ratio < key_len:
        selected = mx.concatenate(
            [
                selected,
                mx.zeros((batch, seq, key_len - blocks * ratio), dtype=mx.bool_),
            ],
            axis=-1,
        )
    token_indices = mx.arange(key_len)
    tail = (token_indices[None, None, :] >= (counts * ratio)[None, :, None]) & (
        token_indices[None, None, :] < ends[None, :, None]
    )
    causal = token_indices[None, None, :] < ends[None, :, None]
    return mx.where((counts > topk)[None, :, None], selected | tail, causal)[:, None]


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("width", [1, 2, 4, 6, 32])
@pytest.mark.parametrize("key_len", [2048, 2051, 4096, 8193, 32771])
def test_mask_matches_general_path_exactly(batch, width, key_len):
    # Arbitrary hits intentionally include future blocks. The expansion must
    # preserve the input bitmap exactly; causal top-k filtering happens earlier.
    rng = np.random.default_rng(91226 + width)
    hits = mx.array(rng.random((batch, width, key_len // 4)) < 0.55)
    ends = key_len - width + mx.arange(width, dtype=mx.int32) + 1
    counts = ends // 4
    expected = _reference_mask(hits, counts, ends, 4, 512, key_len)
    actual = qsa_mask._launch_mask(hits, counts, ends, 4, 512, key_len)
    mx.eval(expected, actual)
    assert actual.shape == (batch, 1, width, key_len)
    assert mx.array_equal(actual, expected).item()


def _inputs():
    hits = mx.zeros((1, 4, 1024), dtype=mx.bool_)
    ends = mx.arange(4093, 4097, dtype=mx.int32)
    counts = ends // 4
    return hits, counts, ends, 4, 512, 4096


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_guarded_mask_validates_once_then_stays_lazy(monkeypatch):
    monkeypatch.setattr(qsa_mask, "_m4_max", lambda: True)
    monkeypatch.setattr(qsa_mask, "_PROVEN", False)
    monkeypatch.setattr(qsa_mask, "_FAILED", False)
    inputs = _inputs()
    mx.eval(inputs[:3])
    with monkeypatch.context() as patch:
        evaluate = Mock(wraps=mx.eval)
        patch.setattr(mx, "eval", evaluate)
        first = qsa_mask.fused_block_mask(*inputs)
        second = qsa_mask.fused_block_mask(*inputs)
        assert evaluate.call_count == 1
    mx.eval(first, second)
    assert mx.array_equal(first, second).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_mask_failure_logs_once_and_keeps_general_path(monkeypatch, caplog):
    monkeypatch.setattr(qsa_mask, "_m4_max", lambda: True)
    monkeypatch.setattr(qsa_mask, "_FAILED", False)
    launch = Mock(side_effect=RuntimeError("injected mask failure"))
    monkeypatch.setattr(qsa_mask, "_launch_mask", launch)
    with caplog.at_level(logging.WARNING, logger=qsa_mask.logger.name):
        assert qsa_mask.fused_block_mask(*_inputs()) is None
        assert qsa_mask.fused_block_mask(*_inputs()) is None
    assert launch.call_count == 1
    assert qsa_mask._FAILED
    assert len(caplog.records) == 1
    assert "injected mask failure" in caplog.text


@pytest.mark.parametrize(
    "change", ["chip", "width", "batch", "ratio", "budget", "context"]
)
def test_unmeasured_layout_stays_general(monkeypatch, change):
    monkeypatch.setattr(qsa_mask, "_m4_max", lambda: change != "chip")
    inputs = list(_inputs())
    if change == "width":
        inputs[0] = mx.zeros((1, 7, 1024), dtype=mx.bool_)
    elif change == "batch":
        inputs[0] = mx.zeros((2, 4, 1024), dtype=mx.bool_)
    elif change == "ratio":
        inputs[3] = 2
    elif change == "budget":
        inputs[4] = 256
    elif change == "context":
        inputs[5] = 65536
    assert qsa_mask.fused_block_mask(*inputs) is None
