# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.gated_delta_advance.

The patch replaces ``Qwen3_5GatedDeltaNet.__call__`` with an mlx-lm-equivalent
body. mlx-vlm 191d7c8 (target) already includes the ``cache.advance(S)`` call
upstream, so the patch primarily carries (a) ``mx.contiguous`` wrapping on the
``cache[0]`` write to break a shared-buffer memory leak and (b) the
``cache.lengths is not None`` per-element slicing branch for ArraysCache.
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


def test_patched_call_signature_matches_mlx_vlm():
    """The replacement __call__ must accept the mlx-vlm signature
    ``(inputs, mask=None, cache=None, gdn_sink=None, target_verify=False)``.
    Any callsite that passes mlx-vlm's speculative-cache arguments must
    still work.
    """
    import inspect
    from omlx.patches.gated_delta_advance import _build_replacement_call

    sig = inspect.signature(_build_replacement_call())
    params = list(sig.parameters.keys())
    assert params == ["self", "inputs", "mask", "cache", "gdn_sink", "target_verify"]


def test_patched_call_delegates_target_verify_to_upstream():
    from omlx.patches.gated_delta_advance import _build_replacement_call

    calls = []

    def original(self, inputs, mask=None, cache=None, gdn_sink=None, target_verify=False):
        calls.append((self, inputs, mask, cache, gdn_sink, target_verify))
        return "upstream"

    replacement = _build_replacement_call(original)
    owner = object()
    sink = []
    result = replacement(
        owner,
        "inputs",
        mask="mask",
        cache="cache",
        gdn_sink=sink,
        target_verify=True,
    )

    assert result == "upstream"
    assert calls == [(owner, "inputs", "mask", "cache", sink, True)]
