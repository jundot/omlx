# SPDX-License-Identifier: Apache-2.0
"""Tests for the custom-kernel nanobind ABI probe (issue #2139).

An extension built with a nanobind whose ABI tag differs from the mlx
wheel's imports cleanly and lists every symbol, but rejects every mlx
array at call time. ``_verify_abi`` must catch that once at import and
disable the native symbols instead of letting each routed call raise.
"""

import pytest

from omlx.custom_kernels.bonsai import fast as bonsai_fast
from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
from omlx.custom_kernels.minimax_m3 import fast as minimax_fast
from omlx.custom_kernels.qwen35_prefill import fast as qwen35_fast

ALL_FAST = (qwen35_fast, glm_fast, minimax_fast, bonsai_fast)


class _MismatchedExt:
    """Mimics a wrong-nanobind build: symbols exist, every call raises."""

    def abi_probe(self, a):
        raise TypeError(
            "abi_probe(): incompatible function arguments. The following "
            "argument types are supported: ..."
        )


class _HealthyExt:
    def abi_probe(self, a):
        return 1


class _LegacyExt:
    """A build predating the probe symbol: assumed compatible."""


@pytest.mark.parametrize("fast", ALL_FAST, ids=lambda m: m.__name__)
def test_mismatched_build_is_disabled_with_import_error(fast):
    ext, err = fast._verify_abi(_MismatchedExt(), None)
    assert ext is None
    assert isinstance(err, TypeError)


@pytest.mark.parametrize("fast", ALL_FAST, ids=lambda m: m.__name__)
def test_healthy_build_passes_through(fast):
    ext = _HealthyExt()
    out, err = fast._verify_abi(ext, None)
    assert out is ext
    assert err is None


@pytest.mark.parametrize("fast", ALL_FAST, ids=lambda m: m.__name__)
def test_legacy_build_without_probe_passes_through(fast):
    ext = _LegacyExt()
    out, err = fast._verify_abi(ext, None)
    assert out is ext
    assert err is None


@pytest.mark.parametrize("fast", ALL_FAST, ids=lambda m: m.__name__)
def test_missing_extension_passes_through(fast):
    sentinel = ImportError("no native build")
    out, err = fast._verify_abi(None, sentinel)
    assert out is None
    assert err is sentinel


@pytest.mark.parametrize("fast", ALL_FAST, ids=lambda m: m.__name__)
def test_local_build_probe_is_healthy(fast):
    """The in-tree builds must expose abi_probe and accept mlx arrays."""
    if not fast.is_native_available():
        pytest.skip(f"{fast.__name__} native build unavailable")
    import mlx.core as mx

    assert fast._ext.abi_probe(mx.zeros((3,))) == 3


class _FoldAwareExt:
    """New build: nanobind-style doc includes the mask-fold kwargs."""

    def dsa_indexer_scores(self, *args, **kwargs):
        raise AssertionError("probe must not call the kernel")

    dsa_indexer_scores.__doc__ = (
        "dsa_indexer_scores(queries: array, keys: array, weights: array, "
        "causal: bool = True, unused_causal_prefix_topk: int = 0, "
        "skip_causal_future_store: bool = False, causal_q_offset: int = -1, "
        "mask_ratio: int = 0, mask_q_offset: int = 0, stream: None = None)"
    )


class _PreFoldExt:
    """Old build: same symbol, but without the mask-fold kwargs."""

    def dsa_indexer_scores(self, *args, **kwargs):
        raise AssertionError("probe must not call the kernel")

    dsa_indexer_scores.__doc__ = (
        "dsa_indexer_scores(queries: array, keys: array, weights: array, "
        "causal: bool = True, unused_causal_prefix_topk: int = 0, "
        "skip_causal_future_store: bool = False, causal_q_offset: int = -1, "
        "stream: None = None)"
    )


class _NoScoresExt:
    """A build without dsa_indexer_scores at all."""


def test_mask_fold_probe_detects_fold_aware_build():
    assert glm_fast._probe_mask_fold(_FoldAwareExt()) is True


def test_mask_fold_probe_rejects_pre_fold_build():
    assert glm_fast._probe_mask_fold(_PreFoldExt()) is False


def test_mask_fold_probe_handles_missing_symbol_and_ext():
    assert glm_fast._probe_mask_fold(_NoScoresExt()) is False
    assert glm_fast._probe_mask_fold(None) is False


def test_pre_fold_build_keeps_historical_call_signature(monkeypatch):
    """An old _ext must receive no mask kwargs and still get exact masking.

    Regression for the unconditional-kwargs break: GLM-5.2's native path
    raised TypeError on every call, and the V4 indexer silently fell back
    while the startup probe still reported the kernels as available.
    """
    import mlx.core as mx

    calls = []

    def old_scores(queries, keys, weights, **kwargs):
        assert "mask_ratio" not in kwargs
        assert "mask_q_offset" not in kwargs
        calls.append(kwargs)
        B, H, L, D = queries.shape
        P = keys.shape[2]
        return mx.zeros((B, H, L, P), dtype=queries.dtype)

    monkeypatch.setattr(glm_fast, "_ext", type("E", (), {"dsa_indexer_scores": staticmethod(old_scores)})())
    monkeypatch.setattr(glm_fast, "_EXT_MASK_FOLD", False)

    H, D, L, P = 64, 128, 64, 512
    q = mx.zeros((1, H, L, D), dtype=mx.bfloat16)
    keys = mx.zeros((1, 1, P, D), dtype=mx.bfloat16)
    weights = mx.zeros((1, L, H), dtype=mx.bfloat16)

    ratio, q_off = 4, 256
    out = glm_fast.dsa_indexer_scores(
        q, keys, weights, causal=False, mask_ratio=ratio, mask_q_offset=q_off
    )
    assert len(calls) == 1

    rows = mx.arange(L)[:, None]
    cols = mx.arange(P)[None, :]
    expected = mx.where(
        (cols < ((q_off + rows + 1) // ratio))[None, None],
        mx.zeros((1, H, L, P), dtype=mx.bfloat16),
        mx.finfo(mx.bfloat16).min,
    )
    mx.eval(out, expected)
    assert bool(mx.array_equal(out.view(mx.uint16), expected.view(mx.uint16)))


def test_fold_aware_build_receives_mask_kwargs(monkeypatch):
    import mlx.core as mx

    seen = {}

    def new_scores(queries, keys, weights, **kwargs):
        seen.update(kwargs)
        B, H, L, _ = queries.shape
        P = keys.shape[2]
        return mx.zeros((B, H, L, P), dtype=queries.dtype)

    monkeypatch.setattr(glm_fast, "_ext", type("E", (), {"dsa_indexer_scores": staticmethod(new_scores)})())
    monkeypatch.setattr(glm_fast, "_EXT_MASK_FOLD", True)

    H, D, L, P = 64, 128, 64, 512
    q = mx.zeros((1, H, L, D), dtype=mx.bfloat16)
    keys = mx.zeros((1, 1, P, D), dtype=mx.bfloat16)
    weights = mx.zeros((1, L, H), dtype=mx.bfloat16)

    glm_fast.dsa_indexer_scores(
        q, keys, weights, causal=False, mask_ratio=4, mask_q_offset=256
    )
    assert seen.get("mask_ratio") == 4
    assert seen.get("mask_q_offset") == 256