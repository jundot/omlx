# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.gated_delta_advance.

The patch monkey-patches mlx-lm and mlx-vlm GatedDeltaNet to call
``cache.advance(S)`` after the forward pass and to wrap the conv state
in ``mx.contiguous``. mlx-lm 0.31.3 already has both fixes upstream;
mlx-vlm e41cd25 still misses both, and Qwen3_5GatedDeltaNet is reused
by qwen3_5_moe so a single class patch covers Qwen3.5 and Qwen3.6.
"""

from __future__ import annotations

import pytest

from omlx.patches.gated_delta_advance import (
    _patched_classes,
    apply_gated_delta_advance_patch,
)


def test_apply_returns_true_when_at_least_one_target_present():
    """Patch should report success as long as one of the GatedDeltaNet
    classes is importable from the runtime."""
    assert apply_gated_delta_advance_patch() is True


def test_patch_is_idempotent():
    """Calling apply repeatedly must not double-wrap __call__."""
    apply_gated_delta_advance_patch()
    snapshot = set(_patched_classes)
    apply_gated_delta_advance_patch()
    assert _patched_classes == snapshot


def test_patch_accepts_model_arg_for_backward_compat():
    """Existing call sites pass a ``model`` argument; the new
    implementation must accept and ignore it without crashing."""
    fake_model = object()
    assert apply_gated_delta_advance_patch(fake_model) is True


def test_mlx_vlm_qwen3_5_class_is_patched():
    """The mlx-vlm class is the primary target of this patch."""
    apply_gated_delta_advance_patch()
    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet
    except ImportError:
        pytest.skip("mlx-vlm not installed in this environment")
    assert id(Qwen3_5GatedDeltaNet) in _patched_classes


def test_post_fix_failure_does_not_break_original_call():
    """If our post-processing throws (e.g. cache[0] access fails because
    mlx-vlm changed cache layout), the original forward result must
    still be returned and the error logged, not raised."""
    from omlx.patches.gated_delta_advance import _patch_class

    class _Stub:
        def __call__(self, inputs, cache=None):
            return "original_result"

    _patch_class(_Stub, "test._BadCache")

    class _BadCache:
        # Intentionally raises on every access to simulate a layout change
        def __getitem__(self, idx):
            raise RuntimeError("cache layout changed upstream")

        def advance(self, n):
            raise RuntimeError("advance not supported")

    class _FakeInputs:
        shape = (1, 5, 8)

    stub = _Stub()
    result = stub(_FakeInputs(), cache=_BadCache())
    assert result == "original_result"


def test_patched_call_forwards_extra_kwargs():
    """The wrapper must forward arbitrary positional / keyword arguments
    to the original ``__call__``. Without this mlx-vlm qwen3_5_moe
    breaks with ``gdn_sink`` kwarg.
    """
    from omlx.patches.gated_delta_advance import _patch_class

    captured = {}

    class _Stub:
        def __call__(self, inputs, *args, cache=None, **kwargs):
            captured["inputs"] = inputs
            captured["args"] = args
            captured["kwargs"] = kwargs
            captured["cache"] = cache
            return inputs

    _patch_class(_Stub, "test._Stub")

    class _FakeCache:
        def __init__(self):
            self._slot0 = None
            self._slot1 = None
            self.advance_calls: list[int] = []

        def __getitem__(self, idx):
            return self._slot0 if idx == 0 else self._slot1

        def __setitem__(self, idx, value):
            if idx == 0:
                self._slot0 = value
            else:
                self._slot1 = value

        def advance(self, n: int) -> None:
            self.advance_calls.append(n)

    class _FakeInputs:
        shape = (1, 7, 16)

    stub = _Stub()
    cache = _FakeCache()
    stub(_FakeInputs(), None, cache=cache, gdn_sink="some_extra", position_ids=42)

    # Original call received all extras
    assert captured["args"] == (None,)
    assert captured["kwargs"] == {"gdn_sink": "some_extra", "position_ids": 42}
    # advance() was called with the prefill length
    assert cache.advance_calls == [7]
