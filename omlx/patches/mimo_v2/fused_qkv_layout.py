# SPDX-License-Identifier: Apache-2.0
"""Fused-QKV FP8 layout for MiMo V2.5 checkpoints.

MiMo V2.5 ships attention as one fused ``qkv_proj.weight`` that is already
sharded for tensor parallelism, alongside a block-128 ``weight_scale_inv``
companion. Each shard is padded to a block boundary *individually*, so the
padding is interleaved through the tensor rather than trailing it. Applying
the scale over the fused tensor as a single block grid crosses shard
boundaries and silently yields wrong weights.

Two consumers need this layout, and both drive off this module so there is
exactly one implementation of it:

- ``mlx_lm.models.mimo_v2.Model.sanitize``, the vendored model, which splits
  the fused tensor eagerly with the whole checkpoint in memory.
- :mod:`omlx.patches.mimo_v2.virtual_qkv`, which splits it on demand for the
  streaming quantizer.

The module is owned by oMLX rather than living in the vendored model file so
that it keeps working after upstream mlx-lm ships its own ``mimo_v2``: at
that point ``mlx_lm.models.mimo_v2`` resolves to upstream's module, which
has no knowledge of these helpers.
"""

from __future__ import annotations

import mlx.core as mx

FUSED_QKV_BLOCK_SIZE = 128

_TP_CANDIDATES = (1, 2, 4, 8, 16, 32)


def fused_qkv_keys(layer_idx: int) -> tuple[str, str, str]:
    """Return ``(prefix, qkv_key, scale_key)`` for one attention layer.

    One spelling of the key template, so the eager and streaming paths
    cannot disagree about which tensors are candidates.
    """
    prefix = f"model.layers.{layer_idx}.self_attn"
    qkv_key = f"{prefix}.qkv_proj.weight"
    return prefix, qkv_key, f"{qkv_key}_scale_inv"


def layer_head_geometry(args, layer_idx: int):
    """Return ``(n_heads, n_kv_heads, head_dim, v_head_dim)`` for one layer."""
    if bool(args.hybrid_layer_pattern[layer_idx]):
        return (
            args.swa_num_attention_heads,
            args.swa_num_key_value_heads,
            args.swa_head_dim,
            args.swa_v_head_dim,
        )
    return (
        args.num_attention_heads,
        args.num_key_value_heads,
        args.head_dim,
        args.v_head_dim,
    )


def fused_qkv_part_rows(n_h, n_kv, hd, vhd, tp):
    """Per-shard row counts of the q, k and v slices.

    The single source of the row arithmetic; everything else derives from it.
    """
    return (n_h // tp) * hd, (n_kv // tp) * hd, (n_kv // tp) * vhd


def fused_qkv_shard_rows(n_h, n_kv, hd, vhd, tp):
    """Per-shard ``(actual_rows, padded_rows)`` for one fused-qkv shard."""
    rows = sum(fused_qkv_part_rows(n_h, n_kv, hd, vhd, tp))
    return rows, -(-rows // FUSED_QKV_BLOCK_SIZE) * FUSED_QKV_BLOCK_SIZE


def fused_qkv_split_shapes(n_h, n_kv, hd, vhd, tp, n_cols):
    """Output shapes of :func:`split_fused_qkv`, as ``(q, k, v)``."""
    return tuple(
        (tp * rows, n_cols)
        for rows in fused_qkv_part_rows(n_h, n_kv, hd, vhd, tp)
    )


def fused_qkv_layout_matches(n_h, n_kv, hd, vhd, tp, qkv_shape, scale_shape):
    """Whether ``tp`` reproduces the on-disk fused qkv and scale shapes.

    Shared by TP detection and by per-layer validation so the two can never
    disagree about what a well-formed layout looks like.
    """
    if n_h % tp or n_kv % tp:
        return False
    rows, padded_rows = fused_qkv_shard_rows(n_h, n_kv, hd, vhd, tp)
    return (
        rows * tp == qkv_shape[0]
        and padded_rows * tp == scale_shape[0] * FUSED_QKV_BLOCK_SIZE
    )


def detect_fused_qkv_tp(args, shape_of) -> int:
    """Resolve the tensor-parallel degree the fused qkv was sharded for.

    ``shape_of(key)`` returns that tensor's shape, or None when it is absent,
    so both the eager sanitize (a dict of arrays) and the streaming quantizer
    (a safetensors header index) can drive this without materializing data.

    Only full-attention layers are consulted: sliding-window geometry happens
    to be block-aligned, which leaves its TP degree ambiguous.
    """
    for layer_idx in range(args.num_hidden_layers):
        if bool(args.hybrid_layer_pattern[layer_idx]):
            continue
        _, qkv_key, scale_key = fused_qkv_keys(layer_idx)
        qkv_shape = shape_of(qkv_key)
        scale_shape = shape_of(scale_key)
        if qkv_shape is None or scale_shape is None:
            continue
        n_h, n_kv, hd, vhd = layer_head_geometry(args, layer_idx)
        for tp in _TP_CANDIDATES:
            if fused_qkv_layout_matches(
                n_h, n_kv, hd, vhd, tp, qkv_shape, scale_shape
            ):
                return tp
        raise ValueError(
            f"unable to determine fused-qkv TP layout from layer {layer_idx} "
            f"(weight rows={qkv_shape[0]}, scale rows={scale_shape[0]})"
        )
    # No fused qkv with a scale companion anywhere: the split path is inert
    # and TP is irrelevant.
    return 1


def split_fused_qkv(qkv_fp8, scale_inv, *, tp, n_h, n_kv, hd, vhd):
    """Dequantize a pre-sharded fused qkv FP8 tensor and split it into q/k/v.

    The block scale is applied per tensor-parallel shard, because each shard
    was padded to a block boundary on its own. Returns ``(q, k, v)`` in
    bfloat16.
    """
    block = FUSED_QKV_BLOCK_SIZE
    bf16 = mx.bfloat16
    q_pr, k_pr, v_pr = fused_qkv_part_rows(n_h, n_kv, hd, vhd, tp)
    actual_pr, padded_pr = fused_qkv_shard_rows(n_h, n_kv, hd, vhd, tp)
    n = qkv_fp8.shape[-1]

    qkv = mx.from_fp8(qkv_fp8, dtype=bf16).reshape(tp, actual_pr, n)
    if padded_pr > actual_pr:
        qkv = mx.pad(qkv, ((0, 0), (0, padded_pr - actual_pr), (0, 0)))
    n_col_blocks = scale_inv.shape[1]
    pad_side = block * n_col_blocks - n
    if pad_side > 0:
        qkv = mx.pad(qkv, ((0, 0), (0, 0), (0, pad_side)))

    blocked = qkv.reshape(tp * padded_pr // block, block, n_col_blocks, block)
    qkv = (blocked * scale_inv[:, None, :, None]).reshape(
        tp, padded_pr, n_col_blocks * block
    )[:, :actual_pr, :n]

    q = mx.contiguous(qkv[:, :q_pr, :]).reshape(tp * q_pr, n).astype(bf16)
    k = (
        mx.contiguous(qkv[:, q_pr : q_pr + k_pr, :])
        .reshape(tp * k_pr, n)
        .astype(bf16)
    )
    v = (
        mx.contiguous(qkv[:, q_pr + k_pr :, :])
        .reshape(tp * v_pr, n)
        .astype(bf16)
    )
    return q, k, v
