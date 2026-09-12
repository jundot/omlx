# SPDX-License-Identifier: Apache-2.0
"""Verify-width and single-row QSA attention read selected rows in place through the
gathered decode kernel, matching the portable gather + SDPA math on the same selection."""
from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from tests.test_qwen4_qsa_decode_gather import _tiny_text_config

fast = pytest.importorskip("omlx.custom_kernels.decode_fast.fast")

pytestmark = pytest.mark.skipif(
    not fast.NATIVE_AVAILABLE or not hasattr(fast._ext, "sdpa_decode_gathered"),
    reason="gathered decode kernel not built",
)


@pytest.fixture(autouse=True)
def _vendored_qwen4(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.qsa_fast as qsa_fast

    monkeypatch.delenv("OMLX_QWEN4_QSA_GATHERED_DECODE", raising=False)
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_GATHERED_DISABLED", False, raising=False)
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_GATHERED_PROVEN", False, raising=False)


def _layer_and_prefix(seed: int = 23, prefix_tokens: int = 10):
    import mlx_vlm.models.qwen4_exp.language as language

    config = _tiny_text_config()
    config.head_dim = 64  # smallest head dim the decode kernels support
    mx.random.seed(seed)  # layer weights and inputs are then independent of test order
    attention = language.Qwen4ExpAttention(config)
    mx.eval(attention.parameters())
    prefix = mx.random.normal((1, prefix_tokens, config.hidden_size))
    caches = []
    for _ in range(2):
        cache = language.QSAKVCache()
        mx.eval(attention(prefix, mask="causal", cache=cache))
        caches.append(cache)
    return config, attention, caches


def _spy(monkeypatch):
    calls = []
    original = fast.sdpa_decode_gathered

    def tracked(q, k, v, indices, scale, **kwargs):
        out = original(q, k, v, indices, scale, **kwargs)
        # Portable gather + SDPA on the kernel's own inputs: the two caches select
        # blocks independently and a near-tie in the tiny prefix can flip a row.
        reference = original(q, k, v, indices, scale, force_fallback=True)
        calls.append((tuple(q.shape), tuple(indices.shape), out, reference))
        return out

    monkeypatch.setattr(fast, "sdpa_decode_gathered", tracked)
    return calls


@pytest.mark.parametrize("rows", [2, 4])
def test_verify_rows_use_gathered_decode_kernel_and_match_portable(monkeypatch, rows):
    config, attention, (native_cache, portable_cache) = _layer_and_prefix()
    verify = mx.random.normal((1, rows, config.hidden_size))
    calls = _spy(monkeypatch)
    actual = attention(verify, mask="causal", cache=native_cache, target_verify=True)
    mx.eval(actual)
    assert [c[0][2] for c in calls] == [rows]          # one kernel call covering every verify row
    assert calls[0][1][:2] == (1, rows)                 # one index set per row
    kernel_out, portable_out = calls[0][2], calls[0][3]
    mx.eval(kernel_out, portable_out)
    assert mx.allclose(kernel_out, portable_out, rtol=1e-4, atol=1e-4).item()
    monkeypatch.setenv("OMLX_QWEN4_QSA_GATHERED_DECODE", "0")
    expected = attention(verify, mask="causal", cache=portable_cache, target_verify=True)
    mx.eval(expected)
    assert len(calls) == 1                              # kill switch keeps the portable path
    assert expected.shape == actual.shape and mx.isfinite(expected).all().item()


def test_single_row_decode_uses_gathered_decode_kernel(monkeypatch):
    config, attention, (native_cache, portable_cache) = _layer_and_prefix()
    token = mx.random.normal((1, 1, config.hidden_size))
    calls = _spy(monkeypatch)
    actual = attention(token, mask=None, cache=native_cache)
    mx.eval(actual)
    assert [c[0][2] for c in calls] == [1]
    kernel_out, portable_out = calls[0][2], calls[0][3]
    mx.eval(kernel_out, portable_out)
    assert mx.allclose(kernel_out, portable_out, rtol=1e-4, atol=1e-4).item()
    monkeypatch.setenv("OMLX_QWEN4_QSA_GATHERED_DECODE", "0")
    expected = attention(token, mask=None, cache=portable_cache)
    mx.eval(expected)
    assert len(calls) == 1
    assert expected.shape == actual.shape and mx.isfinite(expected).all().item()
