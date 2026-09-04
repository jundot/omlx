# SPDX-License-Identifier: Apache-2.0
"""Route Qwen3.5/3.6 MoE prefill weighted-sum to a native kernel.

Stock ``SwitchGLU`` sorts routed tokens for large batches, computes expert
outputs, scatters back to ``[B, T, topk, D]``, and then applies router scores.
For Qwen MoE prefill we can consume the sorted expert output directly with the
existing native weighted-sum kernel, avoiding the scatter and expanded
intermediate. Decode and target-verify paths keep the original implementation.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False


def _native_weighted_sum():
    try:
        from omlx.custom_kernels.qwen35_prefill import fast
    except Exception:
        return None
    if not fast.has_symbol("qwen35_moe_weighted_sum"):
        return None
    return fast.qwen35_moe_weighted_sum


def _target_verify_arg(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if bool(kwargs.get("target_verify", False)):
        return True
    return bool(args and isinstance(args[0], bool) and args[0])


def _should_route(self: Any, x: mx.array, target_verify: bool, min_tokens: int) -> bool:
    # Shape gates first: this runs on every MoE block call of every decode
    # step, so the common (decode) case must exit on the seq-len check
    # before touching env vars or Metal state (issue #2132).
    if x.ndim != 3 or x.shape[-2] < min_tokens:
        return False
    if target_verify:
        return False
    if x.dtype not in (mx.float16, mx.bfloat16):
        return False
    if os.environ.get("OMLX_QWEN35_MOE_WEIGHTED_SUM", "1") == "0":
        return False
    if not mx.metal.is_available():
        return False
    if getattr(self, "sharding_group", None) is not None:
        return False
    if getattr(self, "top_k", None) not in (6, 8):
        return False
    if x.shape[-2] * int(getattr(self, "top_k", 0)) < 64:
        return False
    switch_mlp = getattr(self, "switch_mlp", None)
    if switch_mlp is None or not hasattr(switch_mlp, "down_proj"):
        return False
    # Either the stock split projections or the omlx gate+up fused layout
    # (qwen35_moe_gate_up patch, issue #2238).
    return hasattr(switch_mlp, "gate_up_proj") or (
        hasattr(switch_mlp, "up_proj") and hasattr(switch_mlp, "gate_proj")
    )


def _native_switch_weighted_sum(
    switch_mlp: Any,
    x: mx.array,
    inds: mx.array,
    scores: mx.array,
    weighted_sum: Callable[..., mx.array],
) -> mx.array:
    from mlx_lm.models.switch_layers import _gather_sort

    x_sorted, idx, inv_order = _gather_sort(mx.expand_dims(x, (-2, -3)), inds)
    if switch_mlp.training:
        idx = mx.stop_gradient(idx)

    gate_up = getattr(switch_mlp, "gate_up_proj", None)
    if gate_up is not None:
        x_gate_up = gate_up(x_sorted, idx, sorted_indices=True)
        x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
    else:
        x_up = switch_mlp.up_proj(x_sorted, idx, sorted_indices=True)
        x_gate = switch_mlp.gate_proj(x_sorted, idx, sorted_indices=True)
    x_sorted = switch_mlp.down_proj(
        switch_mlp.activation(x_up, x_gate),
        idx,
        sorted_indices=True,
    )
    return weighted_sum(
        mx.contiguous(x_sorted),
        mx.contiguous(inv_order.astype(mx.uint32)),
        mx.contiguous(scores.astype(mx.float32)),
    )


def _call_shared_expert(self: Any, x: mx.array, target_verify: bool) -> mx.array:
    shared_expert = self.shared_expert
    try:
        return shared_expert(x, target_verify)
    except TypeError:
        return shared_expert(x)


def _fast_moe(self: Any, x: mx.array, target_verify: bool) -> mx.array:
    weighted_sum = _native_weighted_sum()
    if weighted_sum is None:
        raise RuntimeError("qwen35_moe_weighted_sum native kernel is unavailable")

    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if getattr(self, "norm_topk_prob", True):
        scores = scores / scores.sum(axis=-1, keepdims=True)

    y = _native_switch_weighted_sum(self.switch_mlp, x, inds, scores, weighted_sum)

    if hasattr(self, "shared_expert") and hasattr(self, "shared_expert_gate"):
        shared_y = _call_shared_expert(self, x, target_verify)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        y = y + shared_y
    return y


def _make_patched_call(orig_call: Callable[..., mx.array], min_tokens: int):
    def patched(self, x: mx.array, *args, **kwargs):
        # Decode fast path: one shape check before any argument parsing.
        if x.ndim != 3 or x.shape[-2] < min_tokens:
            return orig_call(self, x, *args, **kwargs)
        target_verify = _target_verify_arg(args, kwargs)
        if not _should_route(self, x, target_verify, min_tokens):
            return orig_call(self, x, *args, **kwargs)
        try:
            return _fast_moe(self, x, target_verify)
        except Exception:
            logger.warning("Qwen MoE weighted-sum fast path failed", exc_info=True)
            return orig_call(self, x, *args, **kwargs)

    return patched


def _patch_class(module_name: str, class_name: str, min_tokens: int) -> bool:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False
    cls = getattr(module, class_name, None)
    if cls is None:
        return False
    if getattr(cls, "_omlx_qwen_moe_weighted_sum_patched", False):
        return True

    orig = cls.__call__
    cls.__call__ = _make_patched_call(orig, min_tokens)
    cls._omlx_qwen_moe_weighted_sum_patched = True
    cls._omlx_qwen_moe_weighted_sum_original_call = orig
    return True


def apply_qwen35_moe_weighted_sum_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("OMLX_QWEN35_MOE_WEIGHTED_SUM", "1") == "0":
        return False
    if _native_weighted_sum() is None:
        logger.debug("Qwen MoE weighted-sum native kernel unavailable; patch skipped")
        return False

    min_tokens = int(
        os.environ.get("OMLX_QWEN35_MOE_WEIGHTED_SUM_MIN_TOKENS", "1024")
    )
    patched = False
    patched |= _patch_class(
        "mlx_vlm.models.qwen3_5_moe.language",
        "Qwen3_5MoeSparseMoeBlock",
        min_tokens,
    )
    patched |= _patch_class(
        "mlx_lm.models.qwen3_moe", "Qwen3MoeSparseMoeBlock", min_tokens
    )
    patched |= _patch_class(
        "mlx_lm.models.qwen3_5", "SparseMoeBlock", min_tokens
    )

    _PATCHED = patched
    if patched:
        logger.info(
            "Qwen3.5/3.6 MoE weighted-sum patch applied (min_tokens=%d)",
            min_tokens,
        )
    return patched


__all__ = ["apply_qwen35_moe_weighted_sum_patch"]
