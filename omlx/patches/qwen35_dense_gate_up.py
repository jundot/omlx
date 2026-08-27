# SPDX-License-Identifier: Apache-2.0
"""Fuse dense Qwen3.5-family MLP gate/up quantized projections post-load.

Affine quantization packs output rows independently. Concatenating gate and up
along the output dimension therefore preserves each output's arithmetic while
turning two qmm dispatches into one and reusing the activation tile. The loaded
gate module becomes the fused container; original gate/up buffers are dropped
layer-by-layer to bound transient memory.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from ..scheduler import _sync_and_clear_cache

logger = logging.getLogger(__name__)
_PATCHED = False


def _can_fuse(mlp: Any) -> bool:
    if hasattr(mlp, "gate_up_proj"):
        return False
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if not (
        isinstance(gate, nn.QuantizedLinear)
        and isinstance(up, nn.QuantizedLinear)
        and down is not None
        and type(gate) is type(up)
    ):
        return False
    if (gate.bits, gate.group_size, gate.mode) != (
        up.bits,
        up.group_size,
        up.mode,
    ):
        return False
    if ("bias" in gate) != ("bias" in up):
        return False
    for name in ("weight", "scales", "biases"):
        left = getattr(gate, name, None)
        right = getattr(up, name, None)
        if left is None or right is None or left.shape != right.shape:
            return False
        if left.dtype != right.dtype:
            return False
    return True


def _fuse_one(mlp: Any) -> None:
    gate, up = mlp.gate_proj, mlp.up_proj
    split = int(gate.weight.shape[0])
    arrays = {
        name: mx.concatenate([getattr(gate, name), getattr(up, name)], axis=0)
        for name in ("weight", "scales", "biases")
    }
    if "bias" in gate:
        arrays["bias"] = mx.concatenate([gate.bias, up.bias], axis=0)
    mx.eval(list(arrays.values()))
    for name, value in arrays.items():
        setattr(gate, name, value)
    mlp.gate_up_proj = gate
    del mlp.gate_proj
    del mlp.up_proj

    if os.environ.get("OMLX_QWEN35_ANE_FUSED_VIEWS") == "1":
        gate_view = copy.copy(gate)
        up_view = copy.copy(gate)
        view_arrays = []
        for name in arrays:
            value = getattr(gate, name)
            gate_value = mx.contiguous(value[:split])
            up_value = mx.contiguous(value[split:])
            setattr(gate_view, name, gate_value)
            setattr(up_view, name, up_value)
            view_arrays.extend((gate_value, up_value))
        # ANE setup and inference run on different executor threads. Fully
        # materialize the compatibility buffers before publishing the modules
        # so no lazy loader-stream slice escapes into inference.
        mx.eval(view_arrays)
        mlp.gate_proj = gate_view
        mlp.up_proj = up_view


def _ensure_call_patch() -> bool:
    global _PATCHED
    try:
        from mlx_vlm.models.qwen3_5 import language as q35

        from .qwen35_fused_swiglu_qmm import try_fast_swiglu
    except ImportError:
        return False
    cls = q35.Qwen3_5MLP
    if getattr(cls, "_omlx_dense_gate_up_patched", False):
        _PATCHED = True
        return True
    original = cls.__call__

    def patched(self, x, target_verify: bool = False):
        fused = getattr(self, "gate_up_proj", None)
        if fused is None:
            return original(self, x, target_verify=target_verify)
        activated = try_fast_swiglu(fused, x, target_verify)
        if activated is None:
            gate_up = q35._target_verify_linear(fused, x, target_verify)
            gate, up = mx.split(gate_up, 2, axis=-1)
            activated = swiglu(gate, up)
        return q35._target_verify_linear(
            self.down_proj, activated, target_verify
        )

    cls.__call__ = patched
    cls._omlx_dense_gate_up_patched = True
    cls._omlx_dense_gate_up_original_call = original
    _PATCHED = True
    return True


class _SharedProjection:
    def __init__(self, fused):
        self.fused = fused
        self.input = None
        self.output = None


class _ProjectionSlice(nn.Module):
    def __init__(self, shared, start: int, end: int, last: bool):
        super().__init__()
        self._shared = shared
        self._start = start
        self._end = end
        self._last = last

    def __call__(self, x):
        shared = self._shared
        if shared.output is None or shared.input is not x:
            shared.input = x
            shared.output = shared.fused(x)
        out = shared.output[..., self._start : self._end]
        if self._last:
            shared.input = None
            shared.output = None
        return out


def _can_fuse_mtp_qkv(attn: Any) -> bool:
    projections = [
        getattr(attn, name, None) for name in ("q_proj", "k_proj", "v_proj")
    ]
    if not all(isinstance(proj, nn.QuantizedLinear) for proj in projections):
        return False
    first = projections[0]
    for proj in projections[1:]:
        if type(proj) is not type(first):
            return False
        if (proj.bits, proj.group_size, proj.mode) != (
            first.bits, first.group_size, first.mode
        ):
            return False
        if ("bias" in proj) != ("bias" in first):
            return False
    for name in ("weight", "scales", "biases"):
        arrays = [getattr(proj, name, None) for proj in projections]
        if any(array is None for array in arrays):
            return False
        if any(array.shape[1:] != arrays[0].shape[1:] for array in arrays[1:]):
            return False
        if any(array.dtype != arrays[0].dtype for array in arrays[1:]):
            return False
    return True


def _fuse_projection_names(attn: Any, projection_names: tuple[str, ...]):
    import copy

    projections = [getattr(attn, name) for name in projection_names]
    array_names = ["weight", "scales", "biases"]
    if "bias" in projections[0]:
        array_names.append("bias")
    arrays = {
        name: mx.concatenate([getattr(proj, name) for proj in projections], axis=0)
        for name in array_names
    }
    mx.eval(list(arrays.values()))
    fused = copy.copy(projections[0])
    for name, array in arrays.items():
        setattr(fused, name, array)
    shared = _SharedProjection(fused)
    sizes = [int(proj.weight.shape[0]) for proj in projections]
    offset = 0
    for index, (name, size) in enumerate(zip(projection_names, sizes, strict=True)):
        setattr(
            attn,
            name,
            _ProjectionSlice(shared, offset, offset + size, index == len(sizes) - 1),
        )
        offset += size
    return fused


def _fuse_mtp_qkv(attn: Any) -> None:
    attn._omlx_mtp_qkv_proj = _fuse_projection_names(
        attn, ("q_proj", "k_proj", "v_proj")
    )


def apply_qwen35_attention_kv_fusion(model: Any) -> int:
    """Fuse compatible target-attention K/V projections, leaving Q intact."""
    if os.environ.get("OMLX_QWEN35_EXACT_VERIFY") == "1":
        return 0
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:
        return 0
    mtp_layer_cls = getattr(q35, "MTPDecoderLayer", None)
    mtp_attentions = {
        id(module.self_attn)
        for _, module in model.named_modules()
        if mtp_layer_cls is not None and type(module) is mtp_layer_cls
    }
    targets = [
        module
        for _, module in model.named_modules()
        if type(module) is q35.Qwen3_5Attention
        and id(module) not in mtp_attentions
        and _can_fuse_mtp_qkv(module)
    ]
    for attn in targets:
        attn._omlx_target_kv_proj = _fuse_projection_names(
            attn, ("k_proj", "v_proj")
        )
        _sync_and_clear_cache()
    logger.info("Qwen target-attention K+V fusion applied: %d layers", len(targets))
    return len(targets)


def apply_qwen35_mtp_qkv_fusion(model: Any) -> int:
    """Fuse only the MTP head's compatible Q/K/V quantized projections."""
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:
        return 0
    mtp_layer_cls = getattr(q35, "MTPDecoderLayer", None)
    if mtp_layer_cls is None:
        return 0
    targets = [
        module.self_attn
        for _, module in model.named_modules()
        if type(module) is mtp_layer_cls and _can_fuse_mtp_qkv(module.self_attn)
    ]
    for attn in targets:
        _fuse_mtp_qkv(attn)
        _sync_and_clear_cache()
    logger.info("Qwen MTP-head Q+K+V fusion applied: %d layers", len(targets))
    return len(targets)


def apply_qwen35_dense_gate_up_fusion(model: Any) -> int:
    """Rewrite loaded dense Qwen MLPs in place; return number fused."""
    if os.environ.get("OMLX_QWEN35_DENSE_GATE_UP", "1") == "0":
        return 0
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:
        return 0
    targets = [
        module
        for _, module in model.named_modules()
        if type(module) is q35.Qwen3_5MLP and _can_fuse(module)
    ]
    if not targets or not _ensure_call_patch():
        return 0
    for mlp in targets:
        _fuse_one(mlp)
        _sync_and_clear_cache()
    logger.info("Qwen dense gate+up fusion applied: %d layers", len(targets))
    return len(targets)


__all__ = [
    "apply_qwen35_attention_kv_fusion",
    "apply_qwen35_dense_gate_up_fusion",
    "apply_qwen35_mtp_qkv_fusion",
]
