# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.gated_delta_advance (no-op contract).

REVERSIBLE DECISION -- flyto soft-fork, mlx-vlm 041f889 upgrade
--------------------------------------------------------------
This file was rewritten when the gated_delta_advance patch became a no-op on
mlx-vlm 041f889. The original tests asserted that the patch REPLACED
``Qwen3_5GatedDeltaNet.__call__`` with an mlx-lm-equivalent body and checked the
old 4-arg ``(self, inputs, mask, cache, gdn_sink)`` signature.

That contract is now obsolete: on 041f889 the patch must NOT override the class
(see ``omlx/patches/gated_delta_advance.py`` docstring) because
  1. the ``mx.contiguous`` + ``cache.lengths`` fixes are upstream verbatim,
  2. the ``conv_state.shape[0] != B`` fallbacks the old body dropped are now
     load-bearing for 041f889's batched ``target_verify`` path, and
  3. upstream's GatedDeltaNet now appends a 12-element ``gdn_sink`` tuple ending
     in ``intermediate_states``; the old body's 11-tuple would break stock
     ``rollback_speculative_cache``.

If the no-op decision is rejected (i.e. you want the gated_delta patch adapted
to 041f889 instead), revert THIS file along with
``omlx/patches/gated_delta_advance.py``.
"""

from __future__ import annotations

import inspect

import pytest

from omlx.patches.gated_delta_advance import apply_gated_delta_advance_patch


def test_apply_is_noop_returns_false():
    """On mlx-vlm 041f889 the patch is a deliberate no-op and reports False
    (no class was patched)."""
    assert apply_gated_delta_advance_patch() is False


def test_apply_is_idempotent_noop():
    """Repeated calls stay a no-op and never raise."""
    assert apply_gated_delta_advance_patch() is False
    assert apply_gated_delta_advance_patch() is False


def test_apply_accepts_model_arg_for_backward_compat():
    """Existing call sites pass a ``model`` argument; the no-op must accept and
    ignore it without crashing (still returning False)."""
    fake_model = object()
    assert apply_gated_delta_advance_patch(fake_model) is False


def test_mlx_vlm_gated_delta_class_is_not_patched():
    """The no-op must leave 041f889's GatedDeltaNet.__call__ intact, including
    its ``target_verify`` parameter and the 12-tuple ``gdn_sink`` capture that
    stock ``rollback_speculative_cache`` consumes."""
    apply_gated_delta_advance_patch()
    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet
    except ImportError:
        pytest.skip("mlx-vlm not installed in this environment")

    sig = inspect.signature(Qwen3_5GatedDeltaNet.__call__)
    # Upstream 041f889 signature carries target_verify; the old flyto patch did
    # not. Its presence proves the class is the unpatched upstream one.
    assert "target_verify" in sig.parameters

    src = inspect.getsource(Qwen3_5GatedDeltaNet.__call__)
    assert "intermediate_states" in src
