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


def test_replacement_call_advances_cache_and_accepts_extra_kwargs(monkeypatch):
    """The body replacement preserves mlx-vlm extras and advances cache."""
    from omlx.patches.gated_delta_advance import _patch_class
    import mlx.core as mx

    class _Stub:
        def __init__(self):
            self.conv_kernel_size = 2
            self.conv_dim = 6
            self.key_dim = 2
            self.num_k_heads = 1
            self.num_v_heads = 1
            self.head_k_dim = 2
            self.head_v_dim = 2
            self.A_log = "A"
            self.dt_bias = "dt"
            self.training = False
            self.in_proj_qkv = lambda inputs: mx.zeros((inputs.shape[0], inputs.shape[1], 6))
            self.in_proj_z = lambda inputs: mx.zeros((inputs.shape[0], inputs.shape[1], 2))
            self.in_proj_b = lambda inputs: mx.zeros((inputs.shape[0], inputs.shape[1], 1))
            self.in_proj_a = lambda inputs: mx.zeros((inputs.shape[0], inputs.shape[1], 1))
            self.conv1d = lambda inputs: mx.zeros((inputs.shape[0], inputs.shape[1] - 1, 6))
            self.norm = lambda out, z: out
            self.out_proj = lambda out: out

    def fake_gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel):
        return v, "new_state"

    monkeypatch.setattr(
        "omlx.patches.gated_delta_advance.gated_delta_update",
        fake_gated_delta_update,
    )
    _patch_class(_Stub, "test._Replacement")

    class _FakeCache:
        def __init__(self):
            self._slot0 = None
            self._slot1 = None
            self.advance_calls: list[int] = []
            self.lengths = None

        def __getitem__(self, idx):
            return self._slot0 if idx == 0 else self._slot1

        def __setitem__(self, idx, value):
            if idx == 0:
                self._slot0 = value
            else:
                self._slot1 = value

        def advance(self, n: int) -> None:
            self.advance_calls.append(n)

    stub = _Stub()
    cache = _FakeCache()
    gdn_sink = []
    result = stub(
        mx.zeros((1, 7, 16)),
        None,
        cache=cache,
        gdn_sink=gdn_sink,
        position_ids=42,
    )

    assert result.shape == (1, 7, 2)
    assert cache._slot1 == "new_state"
    assert cache.advance_calls == [7]
    assert len(gdn_sink) == 1


def test_patched_call_signature_matches_mlx_vlm():
    """The replacement __call__ must accept the mlx-vlm signature
    ``(inputs, mask=None, cache=None, gdn_sink=None)``. Any callsite that
    passes ``gdn_sink`` (speculative-cache rollback) must still work. It also
    accepts forward-compatible extra kwargs from mlx-vlm callsites.
    """
    import inspect
    from omlx.patches.gated_delta_advance import _build_replacement_call

    sig = inspect.signature(_build_replacement_call())
    params = list(sig.parameters.keys())
    assert params[:5] == ["self", "inputs", "mask", "cache", "gdn_sink"]
    assert sig.parameters["_"].kind is inspect.Parameter.VAR_KEYWORD
