# SPDX-License-Identifier: Apache-2.0
"""Fuse supported MoE routed gate/up projections into one gather_qmm.

At single-token decode the routed-expert path runs three tiny
``gather_qmm`` launches per MoE layer (gate, up, down). Affine
quantization packs each output row independently, so concatenating the
gate and up expert weights along the output axis and issuing one
``gather_qmm`` over ``[E, 2*inter, hidden]`` is bit-identical to the two
separate calls while removing one launch per MoE layer per token
(issue #2238). The same fused weights serve prefill and batched decode,
also bit-exact.

The fusion runs once post-load and rewrites stock ``SwitchGLU``
instances in place: gate/up weights are concatenated, ``gate_up_proj``
replaces ``gate_proj``/``up_proj`` (mirroring the vendored GLM DSA
switch layers), and the class ``__call__`` gains a fused branch.
Instances without ``gate_up_proj`` keep the original code path.

mlx-vlm verification uses the same fused projection without sorting routes.
Separate gate/up views remain available to other upstream callers.
"""

from __future__ import annotations

import copy
import logging
import os
from functools import wraps
from typing import Any

import mlx.core as mx
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)
from mlx_vlm.models.switch_layers import (
    DECODE_BLOCK_SIZE,
)
from mlx_vlm.models.switch_layers import (
    QuantizedSwitchLinear as VLMQuantizedSwitchLinear,
)
from mlx_vlm.models.switch_layers import (
    SwitchGLU as VLMSwitchGLU,
)
from mlx_vlm.models.switch_layers import (
    SwitchLinear as VLMSwitchLinear,
)

from ..scheduler import _sync_and_clear_cache

logger = logging.getLogger(__name__)

_CALL_PATCHED = False

# Loaded model classes whose module path marks a supported SwitchGLU family:
# mlx-lm Qwen3.5/3.6 and HyV3, Qwen4-Exp's inherited SwitchGLU, the oMLX
# single-checkpoint MTP wrapper, and the vendored Laguna module.
_FAMILY_TOKENS = (
    "qwen3_5",
    "qwen3_6",
    "qwen35",
    "qwen4_exp",
    "laguna",
    "hy_v3",
)


def _is_supported_family(model: Any) -> bool:
    module = type(model).__module__ or ""
    return any(token in module for token in _FAMILY_TOKENS)


def _can_fuse(switch_mlp: Any) -> bool:
    if hasattr(switch_mlp, "gate_up_proj"):
        return False
    if not (
        hasattr(switch_mlp, "gate_proj")
        and hasattr(switch_mlp, "up_proj")
        and hasattr(switch_mlp, "down_proj")
    ):
        return False
    gate, up = switch_mlp.gate_proj, switch_mlp.up_proj
    if type(gate) is not type(up):
        return False
    if isinstance(gate, (QuantizedSwitchLinear, VLMQuantizedSwitchLinear)):
        if (gate.group_size, gate.bits, gate.mode) != (
            up.group_size,
            up.bits,
            up.mode,
        ):
            return False
        if (gate.get("biases") is None) != (up.get("biases") is None):
            return False
    elif not isinstance(gate, (SwitchLinear, VLMSwitchLinear)):
        return False
    if ("bias" in gate) != ("bias" in up):
        return False
    gate_w, up_w = gate["weight"], up["weight"]
    return gate_w.shape == up_w.shape and gate_w.dtype == up_w.dtype


def _fuse_one(switch_mlp: Any) -> None:
    gate, up = switch_mlp.gate_proj, switch_mlp.up_proj
    # Concat order is [gate, up] along the output axis, matching the GLM
    # DSA fused layout and the HF gate_up_proj checkpoint convention.
    fused = {"weight": mx.concatenate([gate["weight"], up["weight"]], axis=1)}
    if isinstance(gate, (QuantizedSwitchLinear, VLMQuantizedSwitchLinear)):
        fused["scales"] = mx.concatenate([gate["scales"], up["scales"]], axis=1)
        if gate.get("biases") is not None:
            fused["biases"] = mx.concatenate([gate["biases"], up["biases"]], axis=1)
    if "bias" in gate:
        fused["bias"] = mx.concatenate([gate["bias"], up["bias"]], axis=-1)
    mx.eval(list(fused.values()))

    # Reuse the gate module as the fused container so quant params and
    # frozen state carry over; dropping gate_proj/up_proj frees the
    # original buffers.
    if type(switch_mlp) is VLMSwitchGLU:
        gate_up = copy.copy(gate)
        for name, array in fused.items():
            setattr(gate_up, name, array)
            gate[name], up[name] = mx.split(array, 2, axis=-1 if name == "bias" else 1)
        # Verify uses these views on the engine thread after loading.
        mx.eval([gate[name] for name in fused], [up[name] for name in fused])
        switch_mlp.gate_up_proj = gate_up
    else:
        for name, array in fused.items():
            setattr(gate, name, array)
        switch_mlp.gate_up_proj = gate
        del switch_mlp.gate_proj
        del switch_mlp.up_proj


def _make_patched_call(orig_call):
    def patched(self, x: mx.array, indices: mx.array) -> mx.array:
        gate_up = getattr(self, "gate_up_proj", None)
        if gate_up is None:
            return orig_call(self, x, indices)

        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_gate_up = gate_up(x, idx, sorted_indices=do_sort)
        x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    return patched


def _make_vlm_patched_call(original):
    fused_call = _make_patched_call(original)

    def patched(self, x, indices, weights=None, shared=None, residual=None):
        if getattr(self, "gate_up_proj", None) is None:
            return original(self, x, indices, weights, shared, residual)
        if not self.training and x.ndim == 3 and 1 < x.shape[1] <= DECODE_BLOCK_SIZE:
            if x.shape[0] * indices.shape[-1] >= 64:
                return original(self, x, indices, weights, shared, residual)
            # The upstream kernel would copy the strided gate/up views.
            routed = _fused_verify_switch(self, x, indices)
            return self._combine(routed, weights, shared, residual)
        routed = fused_call(self, x, indices)
        return self._combine(routed, weights, shared, residual)

    return patched


def _fused_verify_switch(switch_mlp, x, indices):
    """Keep unsorted routing for the verifier's per-position reductions."""
    batch, length, width = x.shape
    top_k = indices.shape[-1]
    flat_x = mx.expand_dims(x.reshape(batch * length, width), (-2, -3))
    flat_indices = indices.reshape(batch * length, top_k)
    gate_up = switch_mlp.gate_up_proj(flat_x, flat_indices, sorted_indices=False)
    gate, up = mx.split(gate_up, 2, axis=-1)
    out = switch_mlp.down_proj(
        switch_mlp.activation(up, gate), flat_indices, sorted_indices=False
    )
    return out.squeeze(-2).reshape(batch, length, top_k, -1)


def _ensure_vlm_verify_patch() -> None:
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    original = Qwen3_5BatchInvariantForward._switch_glu
    if getattr(original, "_omlx_gate_up_fused_verify", False):
        return

    @wraps(original)
    def fused_verify(self, switch_mlp, x, indices):
        if x.ndim == 3 and getattr(switch_mlp, "gate_up_proj", None) is not None:
            return _fused_verify_switch(switch_mlp, x, indices)
        return original(self, switch_mlp, x, indices)

    fused_verify._omlx_gate_up_fused_verify = True
    Qwen3_5BatchInvariantForward._switch_glu = fused_verify


def _ensure_call_patch() -> None:
    global _CALL_PATCHED
    for cls, wrap in (
        (SwitchGLU, _make_patched_call),
        (VLMSwitchGLU, _make_vlm_patched_call),
    ):
        if getattr(cls, "_omlx_gate_up_fused_call", False):
            continue
        original = cls.__call__
        cls.__call__ = wrap(original)
        cls._omlx_gate_up_fused_call = True
        cls._omlx_gate_up_original_call = original
    _CALL_PATCHED = True


def apply_qwen35_moe_gate_up_fusion(model: Any) -> int:
    """Fuse gate+up expert projections on a supported loaded MoE model.

    Returns the number of fused ``SwitchGLU`` instances (0 when disabled
    via ``OMLX_QWEN35_MOE_GATE_UP=0``, the model family is unsupported,
    or there is nothing to fuse).
    """
    if os.environ.get("OMLX_QWEN35_MOE_GATE_UP", "1") == "0":
        return 0
    if not _is_supported_family(model):
        return 0
    targets = [
        m
        for _, m in model.named_modules()
        if type(m) in (SwitchGLU, VLMSwitchGLU) and _can_fuse(m)
    ]
    if not targets:
        return 0
    _ensure_call_patch()
    _ensure_vlm_verify_patch()
    for switch_mlp in targets:
        _fuse_one(switch_mlp)
        # The freed gate/up buffers land in the MLX buffer pool, which the
        # server pins to total RAM (#300), so nothing releases them during
        # load and the transient grows by the whole routed gate/up set,
        # ~2/3 of the expert bytes (#2304). Drain per fused layer to bound
        # the transient to a single layer's worth.
        _sync_and_clear_cache()
    logger.info("MoE gate+up fusion applied: %d layers", len(targets))
    return len(targets)


__all__ = ["apply_qwen35_moe_gate_up_fusion"]
