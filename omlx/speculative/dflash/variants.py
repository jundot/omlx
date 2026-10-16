# SPDX-License-Identifier: Apache-2.0
"""Per-checkpoint pieces of the draft forward.

DFlash2, DFlash v1, Laguna, MiMo V2 and the Muse Glimmer assistant differ in
how they project the captured context, split Q/K/V, gate or scale the
attention and rotate positions. mlx-vlm loads each class; these helpers apply the
difference inside the batched forward.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.rope_utils import initialize_rope

from ...patches import qwen35_verify_qmm


def gated_mlp(mlp, x: mx.array) -> mx.array:
    """Gated MLP; eight-row verify shapes fuse gate, up and swiglu."""
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    if gate is not None and qwen35_verify_qmm.sg8_swiglu_eligible(gate, up, x):
        batch, length, width = x.shape
        hidden = qwen35_verify_qmm.vk_swiglu_sg8(
            x.reshape(batch * length, width), gate, up
        )
        return mlp.down_proj(hidden.reshape(batch, length, -1))
    return mlp(x)


def project_context(model, hidden: mx.array) -> mx.array:
    """Turn concatenated layer captures into the drafter's context rows."""
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        # Muse Glimmer's assistant keeps fc and its norm in an encoder.
        return encoder(hidden)
    norms = getattr(model, "aux_hidden_norms", None)
    if norms:
        # Laguna normalizes each captured layer before the shared projection.
        parts = mx.split(hidden, len(norms), axis=-1)
        hidden = mx.concatenate(
            [norm(part) for norm, part in zip(norms, parts)], axis=-1
        )
    return model.hidden_norm(model.fc(hidden))


def project_qkv(attn, x: mx.array, context: mx.array) -> tuple:
    """Block queries plus keys and values for ``context`` and the block."""
    fused = getattr(attn, "qkv_proj", None)
    if fused is None:
        project_kv = getattr(attn, "_project_kv", None)
        if project_kv is None:
            return (
                attn.q_proj(x),
                attn.k_proj(context),
                attn.v_proj(context),
                attn.k_proj(x),
                attn.v_proj(x),
            )
        return (attn.q_proj(x), *project_kv(context), *project_kv(x))
    # One read of the fused weight serves the context and the block rows.
    width = int(context.shape[1])
    out = fused(mx.concatenate([context, x], axis=1))
    q_size = attn.n_heads * attn.head_dim
    kv_size = attn.n_kv_heads * attn.head_dim
    ctx, blk = out[:, :width], out[:, width:]
    return (
        blk[..., :q_size],
        ctx[..., q_size : q_size + kv_size],
        ctx[..., q_size + kv_size :],
        blk[..., q_size : q_size + kv_size],
        blk[..., q_size + kv_size :],
    )


def gate_output(attn, x: mx.array, attended: mx.array) -> mx.array:
    """Laguna's per-head softplus gate on the attention output."""
    gate_proj = getattr(attn, "g_proj", None)
    if gate_proj is None:
        return attended
    batch, length, _ = attended.shape
    gate = nn.softplus(gate_proj(x).astype(mx.float32)).astype(attended.dtype)
    attended = attended.reshape(batch, length, attn.n_heads, -1) * gate[..., None]
    return attended.reshape(batch, length, -1)


def apply_partial_rope(model: nn.Module, config: dict) -> None:
    """Rotate only the leading part of each head when the checkpoint says so.

    mlx-vlm builds the drafter RoPE over the whole head and drops
    ``partial_rotary_factor`` (Laguna, MiMo).
    """
    partial = float(config.get("partial_rotary_factor") or 1.0)
    if partial >= 1.0:
        return
    scaling = config.get("rope_parameters") or config.get("rope_scaling")
    base = config.get("rope_theta") or (scaling or {}).get("rope_theta")
    model.rope = initialize_rope(
        int(int(config["head_dim"]) * partial),
        base=float(base),
        traditional=False,
        scaling_config=scaling,
        max_position_embeddings=config.get("max_position_embeddings"),
    )


def load_mask_embedding(path: Path, hidden_size: int) -> mx.array:
    """Read MiMo's bf16 mask vector from its torch zip archive."""
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("/data/0")]
        if len(names) != 1:
            raise ValueError(f"{path} must hold exactly one tensor")
        raw = archive.read(names[0])
    if len(raw) != int(hidden_size) * 2:
        raise ValueError(f"{path} holds {len(raw)} bytes, expected {hidden_size} bf16")
    values = mx.array(np.frombuffer(raw, dtype=np.uint16).copy()).view(mx.bfloat16)
    # The engine drafts on its own executor thread; a lazy view would stay
    # bound to this loading thread's stream.
    mx.eval(values)
    return values
