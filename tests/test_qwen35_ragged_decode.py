# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import mlx.core as mx
import pytest

DTYPE = mx.float16
D_SIZE = 256


def _make_arrays(batch=2, q_heads=16, kv_heads=2, k_size=2048):
    mx.random.seed(42)
    q = mx.random.normal((batch, q_heads, 1, D_SIZE)).astype(DTYPE)
    k = mx.random.normal((batch, kv_heads, k_size, D_SIZE)).astype(DTYPE)
    v = mx.random.normal((batch, kv_heads, k_size, D_SIZE)).astype(DTYPE)
    mx.eval(q, k, v)
    return q, k, v


def _make_q35_module():
    mod = types.ModuleType("mlx_vlm.models.qwen3_5.language")

    def _qwen3_5_sdpa_vector_plan(seq_len, q_heads, kv_heads):
        if seq_len >= 1024:
            return ("two_pass", 1024)
        return ("one_pass", 0)

    mod._qwen3_5_sdpa_vector_plan = _qwen3_5_sdpa_vector_plan
    return mod


def test_signature_valid_shape():
    from omlx.patches.qwen35_ragged_decode import _signature

    q35 = _make_q35_module()
    q, k, v = _make_arrays()
    sig = _signature(q35, q, k, v, [0, 128])
    assert sig is not None
    assert sig[0] == "two_pass"
    assert sig[2] == str(DTYPE)
    assert sig[3] == D_SIZE


def test_signature_wrong_ndim():
    from omlx.patches.qwen35_ragged_decode import _signature

    q35 = _make_q35_module()
    q = mx.zeros((2, 16, D_SIZE)).astype(DTYPE)
    k = mx.zeros((2, 2, 2048, D_SIZE)).astype(DTYPE)
    v = mx.zeros((2, 2, 2048, D_SIZE)).astype(DTYPE)
    assert _signature(q35, q, k, v, [0, 0]) is None


def test_signature_non_decode_seq():
    from omlx.patches.qwen35_ragged_decode import _signature

    q35 = _make_q35_module()
    q = mx.zeros((2, 16, 4, D_SIZE)).astype(DTYPE)
    k = mx.zeros((2, 2, 2048, D_SIZE)).astype(DTYPE)
    v = mx.zeros((2, 2, 2048, D_SIZE)).astype(DTYPE)
    assert _signature(q35, q, k, v, [0, 0]) is None


def test_signature_diverging_plans():
    from omlx.patches.qwen35_ragged_decode import _signature

    q35 = _make_q35_module()
    q, k, v = _make_arrays(batch=2, k_size=2048)
    assert _signature(q35, q, k, v, [1023, 1025]) is None


def test_fallback_on_threadgroup_error(monkeypatch):
    from omlx.patches import qwen35_ragged_decode as mod

    monkeypatch.setattr(mod, "_PROBE_CACHE", {})
    call_count = {"n": 0}

    def failing_original(q, k, v, pads, scale):
        call_count["n"] += 1
        raise ValueError(
            "Thread group size (1024) is greater than "
            "the maximum allowed threads per threadgroup (896)."
        )

    q, k, v = _make_arrays()
    key = ("two_pass", 1024, str(DTYPE), D_SIZE, D_SIZE, 16, 2)

    result = mod._call_with_probe(failing_original, key, q, k, v, [0, 128], 1.0)
    assert result is None
    assert mod._PROBE_CACHE[key] is False
    assert call_count["n"] == 1

    result2 = mod._call_with_probe(failing_original, key, q, k, v, [0, 128], 1.0)
    assert result2 is None
    assert call_count["n"] == 1


def test_passthrough_when_supported(monkeypatch):
    from omlx.patches import qwen35_ragged_decode as mod

    monkeypatch.setattr(mod, "_PROBE_CACHE", {})
    sentinel = mx.zeros((2, 16, 1, D_SIZE)).astype(DTYPE)
    call_count = {"n": 0}

    def good_original(q, k, v, pads, scale):
        call_count["n"] += 1
        return sentinel

    q, k, v = _make_arrays()
    key = ("two_pass", 1024, str(DTYPE), D_SIZE, D_SIZE, 16, 2)

    result = mod._call_with_probe(good_original, key, q, k, v, [0, 128], 1.0)
    assert result is sentinel
    assert mod._PROBE_CACHE[key] is True
    assert call_count["n"] == 1

    result2 = mod._call_with_probe(good_original, key, q, k, v, [0, 128], 1.0)
    assert result2 is sentinel
    assert call_count["n"] == 2


def test_non_threadgroup_error_propagates(monkeypatch):
    from omlx.patches import qwen35_ragged_decode as mod

    monkeypatch.setattr(mod, "_PROBE_CACHE", {})

    def bad_original(q, k, v, pads, scale):
        raise RuntimeError("something unrelated")

    q, k, v = _make_arrays()
    key = ("two_pass", 1024, str(DTYPE), D_SIZE, D_SIZE, 16, 2)

    with pytest.raises(RuntimeError, match="unrelated"):
        mod._call_with_probe(bad_original, key, q, k, v, [0, 0], 1.0)

    assert key not in mod._PROBE_CACHE


def test_patch_install_and_idempotent(monkeypatch):
    q35 = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    from omlx.patches import qwen35_ragged_decode as mod

    monkeypatch.setattr(mod, "_PATCHED", False)
    monkeypatch.setattr(mod, "_PROBE_CACHE", {})

    def original_fn(queries, keys, values, pads, scale):
        return mx.zeros((2, 16, 1, D_SIZE)).astype(DTYPE)

    monkeypatch.setattr(q35, "_qwen3_5_ragged_decode_attention", original_fn)

    result1 = mod.apply_qwen35_ragged_decode_patch()
    assert result1 is True
    assert mod._PATCHED is True
    assert q35._qwen3_5_ragged_decode_attention is not original_fn
    assert getattr(q35._qwen3_5_ragged_decode_attention, mod._PATCH_MARKER, False)

    patched_fn = q35._qwen3_5_ragged_decode_attention
    result2 = mod.apply_qwen35_ragged_decode_patch()
    assert result2 is False
    assert q35._qwen3_5_ragged_decode_attention is patched_fn


def test_patch_returns_false_on_import_error(monkeypatch):
    from omlx.patches import qwen35_ragged_decode as mod

    monkeypatch.setattr(mod, "_PATCHED", False)
    monkeypatch.delitem(sys.modules, "mlx_vlm.models.qwen3_5.language", raising=False)
    monkeypatch.delitem(sys.modules, "mlx_vlm.models.qwen3_5", raising=False)
    monkeypatch.delitem(sys.modules, "mlx_vlm.models", raising=False)
    monkeypatch.delitem(sys.modules, "mlx_vlm", raising=False)

    with patch.dict("sys.modules", {"mlx_vlm": None}):
        result = mod.apply_qwen35_ragged_decode_patch()
    assert result is False
    assert mod._PATCHED is False
