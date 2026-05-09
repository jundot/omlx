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


def test_patch_class_replaces_call():
    """_patch_class should replace __call__ with the body-replacement
    implementation and register the class id."""
    from omlx.patches.gated_delta_advance import _patch_class, _patched_classes

    class _Stub:
        def __call__(self, inputs, cache=None):
            return "original_result"

    original_call = _Stub.__call__
    _patch_class(_Stub, "test._Stub_replace")

    assert id(_Stub) in _patched_classes
    assert _Stub.__call__ is not original_call
