# SPDX-License-Identifier: Apache-2.0
"""Fuse routed gate/up projections of oMLX's own ``SwitchGLU`` variants.

MiMo V2 routes its experts through the GLM DSA package's ``SwitchGLU`` and
GLM-5.3 (``glm5_next``) through the DeepSeek V4 package's. Both run the
expert gate and up projections as two ``gather_qmm`` calls over the same
(sorted) token rows. One call over the gate and up weights concatenated
along the output axis (``[E, 2 * inter, hidden]``) does the same work with
one dispatch and one pass over the gathered activations, and twice the
output tiles per expert run, which keeps the M5 tensor units busier on the
short, ragged per-expert runs of MoE prefill (measured: MiMo 4096-token
chunk 16.1 -> 15.1 ms per layer, GLM-5.3 2048-token chunk 8.5 -> 8.0 ms).

Quantized rows are packed and scaled independently, and every output column
is its own K-reduction computed by the same kernel with the same tiling and
accumulation order, so the two halves of the fused output are bit-identical
to the separate calls (for prefill and for decode).

Like the Qwen/Laguna regroup in :mod:`omlx.patches.qwen35_moe_gate_up`, this
runs once post-load on the materialized model and rewrites instances in
place (``gate_up_proj`` replaces ``gate_proj``/``up_proj``; both classes'
forwards split the fused output as ``[gate; up]``). A module is left as is
when its gate and up differ in format, when it is not quantized, or, for the
DeepSeek V4 variant, when the format has native gate/up pair kernels (MXFP4,
affine 2/3-bit), whose tuned paths stay untouched. The engines skip it with
MoE expert offload and honor ``moe_gate_up_fusion_enabled``;
``OMLX_MOE_GATE_UP_FUSION=0`` disables it too.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx

from ..scheduler import _sync_and_clear_cache
from . import qwen35_moe_gate_up
from .deepseek_v4.switch_layers import has_native_block_kernels

logger = logging.getLogger(__name__)

_GLM_DSA_MODULE = "omlx.patches.glm_moe_dsa.switch_layers"
_DEEPSEEK_V4_MODULE = "omlx.patches.deepseek_v4.switch_layers"

# Loaded model classes whose module path marks a measured family: MiMo V2
# (mlx-lm module, the omnimodal wrapper, DFlash targets) and GLM-5.3
# (mlx-vlm glm5_next). DeepSeek V4 shares the DeepSeek V4 SwitchGLU but has
# not been measured with the fused projection, so it keeps its path.
_FAMILY_TOKENS = ("mimo_v2", "glm5_next")


def _is_supported_family(model: Any) -> bool:
    module = type(model).__module__ or ""
    return any(token in module for token in _FAMILY_TOKENS)


def _fusion_enabled() -> bool:
    value = os.environ.get("OMLX_MOE_GATE_UP_FUSION", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def _switch_family(module: Any) -> str | None:
    cls = type(module)
    if cls.__name__ != "SwitchGLU":
        return None
    if cls.__module__ == _GLM_DSA_MODULE:
        return "glm_dsa"
    if cls.__module__ == _DEEPSEEK_V4_MODULE:
        return "deepseek_v4"
    return None


def _is_quantized(linear: Any) -> bool:
    return type(linear).__name__ == "QuantizedSwitchLinear" and all(
        hasattr(linear, attr) for attr in ("group_size", "bits", "mode")
    )


def can_fuse(switch_mlp: Any) -> bool:
    """Whether ``switch_mlp`` is an oMLX ``SwitchGLU`` this pass can fuse."""
    family = _switch_family(switch_mlp)
    if family is None or "gate_up_proj" in switch_mlp:
        return False
    if not all(p in switch_mlp for p in ("gate_proj", "up_proj", "down_proj")):
        return False
    gate, up = switch_mlp.gate_proj, switch_mlp.up_proj
    if type(gate) is not type(up) or not _is_quantized(gate):
        return False
    if (gate.group_size, gate.bits, gate.mode) != (up.group_size, up.bits, up.mode):
        return False
    for field in ("weight", "scales", "biases", "bias"):
        g, u = gate.get(field), up.get(field)
        if (g is None) != (u is None):
            return False
        if g is not None and (tuple(g.shape) != tuple(u.shape) or g.dtype != u.dtype):
            return False
    # DeepSeek V4 formats with native pair kernels keep their tuned path.
    return not (family == "deepseek_v4" and has_native_block_kernels(gate))


def _fuse_one(switch_mlp: Any) -> None:
    gate, up = switch_mlp.gate_proj, switch_mlp.up_proj
    # [gate; up] along the output axis (axis 1 of the stacked [E, N, *]
    # tensors, the last axis of a per-expert [E, N] bias): the layout both
    # SwitchGLU forwards split and the GLM DSA offload adapter writes.
    fused = {}
    for field in ("weight", "scales", "biases", "bias"):
        if gate.get(field) is not None:
            fused[field] = mx.concatenate(
                [gate[field], up[field]], axis=-1 if field == "bias" else 1
            )
    mx.eval(list(fused.values()))
    # The gate module becomes the fused container: its class, quantization
    # parameters and frozen state carry over; dropping up_proj frees the
    # original buffers.
    for field, array in fused.items():
        setattr(gate, field, array)
    switch_mlp.gate_up_proj = gate
    del switch_mlp.gate_proj
    del switch_mlp.up_proj


def apply_switch_glu_gate_up_fusion(model: Any) -> int:
    """Fuse gate/up of every eligible oMLX ``SwitchGLU`` in a loaded model.

    Returns the number of fused layers (0 when disabled, for model families
    other than MiMo V2 and GLM-5.3, or when nothing is eligible).
    """
    if not _fusion_enabled() or not _is_supported_family(model):
        return 0
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return 0
    targets = [m for _, m in named_modules() if can_fuse(m)]
    if not targets:
        return 0
    for switch_mlp in targets:
        _fuse_one(switch_mlp)
        # Freed gate/up buffers land in the MLX buffer pool; drain per layer
        # so the load transient stays at one layer's gate/up (see #2304).
        _sync_and_clear_cache()
    logger.info("MoE gate+up fusion (oMLX SwitchGLU) applied: %d layers", len(targets))
    return len(targets)


def apply_moe_gate_up_fusion(model: Any) -> int:
    """Run the Qwen/Laguna regroup and the oMLX SwitchGLU fusion.

    The two passes match disjoint model families. Returns the total number
    of fused layers.
    """
    return qwen35_moe_gate_up.apply_qwen35_moe_gate_up_fusion(
        model
    ) + apply_switch_glu_gate_up_fusion(model)


__all__ = ["apply_moe_gate_up_fusion", "apply_switch_glu_gate_up_fusion", "can_fuse"]
