# SPDX-License-Identifier: Apache-2.0
"""Fuse Qwen3.5/3.6 MoE routed gate/up projections into one gather_qmm.

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

mlx-vlm's qwen3_5_moe target-verify helper calls ``gate_proj``/
``up_proj`` directly (MTP verify), so applying the fusion also swaps
that module-level helper for a fused-aware version. Qwen3.5/3.6 MoE
MTP checkpoints route through the VLM engine, which is why the VLM
path matters even for text-only serving.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

import mlx.core as mx
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)

from ..scheduler import _sync_and_clear_cache

logger = logging.getLogger(__name__)

_CALL_PATCHED = False

# Loaded model classes whose module path marks a supported SwitchGLU family
# (mlx_lm qwen3_5 / qwen3_5_moe, Qwen4-Exp's inherited SwitchGLU, the omlx
# single-checkpoint MTP wrapper, and the vendored laguna module).
_FAMILY_TOKENS = ("qwen3_5", "qwen3_6", "qwen35", "qwen4_exp", "laguna")


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
    if isinstance(gate, QuantizedSwitchLinear):
        if (gate.group_size, gate.bits, gate.mode) != (
            up.group_size,
            up.bits,
            up.mode,
        ):
            return False
        if (gate.get("biases") is None) != (up.get("biases") is None):
            return False
    elif not isinstance(gate, SwitchLinear):
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
    if isinstance(gate, QuantizedSwitchLinear):
        fused["scales"] = mx.concatenate([gate["scales"], up["scales"]], axis=1)
        if gate.get("biases") is not None:
            fused["biases"] = mx.concatenate([gate["biases"], up["biases"]], axis=1)
    if "bias" in gate:
        fused["bias"] = mx.concatenate([gate["bias"], up["bias"]], axis=-1)
    mx.eval(list(fused.values()))

    # Reuse the gate module as the fused container so quant params and
    # frozen state carry over; dropping gate_proj/up_proj frees the
    # original buffers.
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

        use_native_q2 = _can_use_native_q2(self, x)
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64 or use_native_q2
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        native_plan = (
            _native_q2_plan(idx, gate_up.num_experts) if use_native_q2 else None
        )
        x_gate_up = _call_projection(gate_up, x, idx, do_sort, native_plan)
        x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
        x = _call_projection(
            self.down_proj,
            self.activation(x_up, x_gate),
            idx,
            do_sort,
            native_plan,
        )
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    return patched


def _is_native_q2_projection(module: Any, dtype: Any) -> bool:
    if not isinstance(module, QuantizedSwitchLinear):
        return False
    biases = module.get("biases")
    return (
        module.group_size == 32
        and module.bits == 2
        and module.mode == "affine"
        and "bias" not in module
        and module["weight"].dtype == mx.uint32
        and biases is not None
        and module["scales"].dtype == dtype
        and biases.dtype == dtype
    )


def _can_use_native_q2(switch_mlp: Any, x: mx.array) -> bool:
    # Stock gather_qmm wins at Qwen4's full 512-expert shape on M3 Ultra.
    # Keep the bit-exact native path opt-in while it remains useful for
    # smaller banks and continued kernel tuning.
    if os.environ.get("OMLX_QWEN35_MOE_Q2_NATIVE", "0") != "1":
        return False
    # Keep prompt processing on stock gather_qmm; the block-list kernel is
    # intended for the one-to-five-token decode regime.
    if x.ndim != 3 or x.shape[-2] > 5 or x.dtype not in (mx.float16, mx.bfloat16):
        return False
    gate_up = getattr(switch_mlp, "gate_up_proj", None)
    down = getattr(switch_mlp, "down_proj", None)
    if not (
        _is_native_q2_projection(gate_up, x.dtype)
        and _is_native_q2_projection(down, x.dtype)
    ):
        return False
    try:
        from ..custom_kernels.glm_moe_dsa import fast as glm_fast

        return glm_fast.has_symbol("deepseek_affine_gather_qmm_blocks")
    except Exception:
        return False


def _native_q2_plan(indices: mx.array, num_experts: int):
    from .deepseek_v4.switch_layers import _block_config, _build_mxfp4_blocks

    block_bm, variant = _block_config(indices.size, "affine")
    block_meta, block_count = _build_mxfp4_blocks(indices, num_experts, block_bm)
    return block_meta, block_count, variant


def _call_projection(module, x, indices, sorted_indices, native_plan):
    if native_plan is None:
        return module(x, indices, sorted_indices=sorted_indices)

    from ..custom_kernels.glm_moe_dsa import fast as glm_fast

    block_meta, block_count, variant = native_plan
    return glm_fast.deepseek_affine_gather_qmm_blocks(
        x,
        module["weight"],
        module["scales"],
        module["biases"],
        block_meta,
        block_count,
        module.group_size,
        module.bits,
        variant,
    )


def _make_patched_target_verify(orig_fn):
    def patched(switch_mlp, x, indices, target_verify):
        gate_up = getattr(switch_mlp, "gate_up_proj", None)
        if gate_up is None:
            return orig_fn(switch_mlp, x, indices, target_verify)
        if not (target_verify and x.ndim == 3 and x.shape[1] > 1):
            return switch_mlp(x, indices)

        B, T, D = x.shape
        k = indices.shape[-1]
        flat_x = mx.expand_dims(x.reshape(B * T, D), (-2, -3))
        flat_indices = indices.reshape(B * T, k)
        x_gate_up = gate_up(flat_x, flat_indices, sorted_indices=False)
        x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
        out = switch_mlp.down_proj(
            switch_mlp.activation(x_up, x_gate),
            flat_indices,
            sorted_indices=False,
        )
        return out.squeeze(-2).reshape(B, T, k, -1)

    return patched


def _ensure_vlm_verify_patch() -> None:
    """Make mlx-vlm's qwen3_5_moe target-verify helper fused-aware."""
    try:
        module = importlib.import_module("mlx_vlm.models.qwen3_5_moe.language")
    except Exception:
        return
    if getattr(module, "_omlx_gate_up_fused_verify", False):
        return
    orig = getattr(module, "_target_verify_switch_glu", None)
    if orig is None:
        return
    module._target_verify_switch_glu = _make_patched_target_verify(orig)
    module._omlx_gate_up_fused_verify = True


def _ensure_call_patch() -> None:
    global _CALL_PATCHED
    if _CALL_PATCHED or getattr(SwitchGLU, "_omlx_gate_up_fused_call", False):
        _CALL_PATCHED = True
        return
    orig = SwitchGLU.__call__
    SwitchGLU.__call__ = _make_patched_call(orig)
    SwitchGLU._omlx_gate_up_fused_call = True
    SwitchGLU._omlx_gate_up_original_call = orig
    _CALL_PATCHED = True


def apply_qwen35_moe_gate_up_fusion(model: Any) -> int:
    """Fuse gate+up expert projections on a loaded Qwen3.5/3.6 MoE model.

    Returns the number of fused ``SwitchGLU`` instances (0 when disabled
    via ``OMLX_QWEN35_MOE_GATE_UP=0``, the model family is unsupported,
    or there is nothing to fuse).
    """
    if os.environ.get("OMLX_QWEN35_MOE_GATE_UP", "1") == "0":
        return 0
    if not _is_supported_family(model):
        return 0
    targets = [
        m for _, m in model.named_modules() if type(m) is SwitchGLU and _can_fuse(m)
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
    logger.info("Qwen MoE gate+up fusion applied: %d layers", len(targets))
    return len(targets)


__all__ = ["apply_qwen35_moe_gate_up_fusion"]
