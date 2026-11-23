# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 / M3.1 (minimax_m3_vl): the GPU-free part.

Only the hidden state [1, L, hidden] crosses a cut and every layer keeps its
own rows, so a stage can start at any layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ... import checkpoint

CONTRACT = 1
FIRST_STAGE_MIN = 1

# Checkpoint names in the MLX layout.
ROOT = "language_model.model."
# Routed expert banks stay packed; every other linear is unpacked to dense.
_BANKS = ("gate_proj", "up_proj", "down_proj", "gate_up_proj")


def resident(tensor: str) -> bool:
    """Whether a stage loads this tensor: the layers and the embedding."""
    return tensor.startswith((ROOT + "layers.", ROOT + "embed_tokens."))


@dataclass(frozen=True)
class Config:
    """The text settings the prefill forward reads, with the model defaults."""

    heads: int
    kv_heads: int
    head_dim: int
    rope_dims: int
    rope_base: float
    rope_scale: float
    eps: float
    gemma_norm: bool
    qk_norm: bool
    alpha: float
    limit: float
    beta: float
    top_k: int
    scoring: str
    routed_scale: float
    pack_shared: bool
    moe: tuple[bool, ...]
    sparse: tuple[bool, ...]
    index_heads: int
    index_dim: int
    block: int
    top_blocks: int
    init_blocks: int
    local_blocks: int


def _sparse_settings(text: dict[str, Any], layers: int) -> dict[str, Any]:
    """``sparse_attention_config`` with the model's defaults and aliases."""
    types = text.get("layer_types")
    freq = None if types is None else [int(t == "minimax_m3_sparse") for t in types]
    aliases = {
        "sparse_index_dim": text.get("index_head_dim"),
        "sparse_num_index_heads": text.get("index_n_heads"),
        "sparse_topk_blocks": text.get("index_topk_blocks"),
        "sparse_block_size": text.get("index_block_size"),
        "sparse_local_block": text.get("index_local_blocks"),
    }
    aliases = {name: value for name, value in aliases.items() if value is not None}
    sparse = text.get("sparse_attention_config")
    if sparse is None:
        if freq is None:
            freq = [0] * min(3, layers) + [1] * max(0, layers - 3)
        return {
            "use_sparse_attention": True,
            "sparse_index_dim": 128,
            **aliases,
            "sparse_attention_freq": freq,
        }
    sparse = {**aliases, **sparse}
    if freq is not None:
        sparse.setdefault("sparse_attention_freq", freq)
        sparse.setdefault("use_sparse_attention", True)
    disabled = sparse.get("sparse_disable_index_value")
    if sparse.get("sparse_attention_freq") is None and isinstance(disabled, list):
        sparse["sparse_attention_freq"] = disabled
        sparse.setdefault("use_sparse_attention", True)
    return sparse


def read_config(model_dir: str | Path) -> Config:
    text = checkpoint.load_config(model_dir)["text_config"]
    layers = text.get("num_hidden_layers", 60)
    hidden, heads = text.get("hidden_size", 6144), text.get("num_attention_heads", 64)
    head_dim = text.get("head_dim", 128) or hidden // heads
    rope_dims = text.get("rotary_dim")
    if rope_dims is None:
        rope_dims = int(head_dim * text.get("partial_rotary_factor", 0.5))
    scaling = text.get("rope_scaling") or {}
    rope_type = scaling.get("type") or scaling.get("rope_type", "default")
    if rope_type not in ("default", "linear"):
        raise ValueError(f"unsupported MiniMax RoPE scaling: {rope_type}")

    sparse = _sparse_settings(text, layers)
    freq = sparse.get("sparse_attention_freq")
    if not sparse.get("use_sparse_attention", False) or not isinstance(freq, list):
        freq = []
    if any(freq) and sparse.get("sparse_score_type", "max") != "max":
        raise ValueError("only max block scores are supported")
    moe = text.get("moe_layer_freq")
    if not moe:
        kinds = text.get("mlp_layer_types")
        dense = min(3, layers)
        moe = (
            [0] * dense + [1] * (layers - dense)
            if kinds is None
            else [int(kind == "sparse") for kind in kinds]
        )
    pack = text.get("pack_shared_expert")
    if pack is None:
        shared = text.get("shared_intermediate_size", 3072)
        pack = text.get("n_shared_experts", 1) == 1 and shared == text.get(
            "intermediate_size", 3072
        )
    return Config(
        heads=heads,
        kv_heads=text.get("num_key_value_heads", 4) or heads,
        head_dim=head_dim,
        rope_dims=rope_dims,
        rope_base=float(text.get("rope_theta", 5000000)),
        rope_scale=1 / scaling["factor"] if rope_type == "linear" else 1.0,
        eps=text.get("rms_norm_eps", 1e-6),
        gemma_norm=text.get("use_gemma_norm", True),
        qk_norm=text.get("use_qk_norm", True),
        alpha=text.get("swiglu_alpha", 1.702),
        limit=text.get("swiglu_limit", 7.0),
        beta=text.get("swiglu_beta", 1.0),
        top_k=text.get("num_experts_per_tok", 4),
        scoring=text.get("scoring_func", "sigmoid"),
        routed_scale=text.get("routed_scaling_factor", 2.0),
        pack_shared=bool(pack),
        moe=tuple(i >= len(moe) or bool(moe[i]) for i in range(layers)),
        sparse=tuple(i < len(freq) and bool(freq[i]) for i in range(layers)),
        index_heads=sparse.get("sparse_num_index_heads", 4),
        index_dim=sparse.get("sparse_index_dim", head_dim),
        block=sparse.get("sparse_block_size", 128),
        top_blocks=sparse.get("sparse_topk_blocks", 16),
        init_blocks=sparse.get("sparse_init_block", 0),
        local_blocks=sparse.get("sparse_local_block", 1),
    )


def is_bank(tensor: str) -> bool:
    """Whether a tensor belongs to a routed expert bank."""
    module = tensor.rsplit(".", 1)[0]
    return ".switch_mlp." in tensor and module.rsplit(".", 1)[-1] in _BANKS


def _gpu_bytes(
    name: str, tensors: dict[str, checkpoint.Tensor], config: dict[str, Any]
) -> int:
    """Bytes a tensor takes on the GPU; unpacked linears count as bfloat16."""
    module, _, leaf = name.rpartition(".")
    quantized = f"{module}.scales" in tensors
    if not quantized or is_bank(name) or module.endswith("embed_tokens"):
        return tensors[name].nbytes
    if leaf != "weight":
        return 0
    rows, words = tensors[name].shape
    bits = checkpoint.quantization(config, module)["bits"]
    return rows * (words * 32 // bits) * 2


def plan(model_dir: str | Path) -> tuple[int, list[int]]:
    """(bytes outside the layers, bytes per layer) a stage holds on its GPU."""
    tensors = checkpoint.tensors(model_dir)
    config = checkpoint.load_config(model_dir)
    shared, per_layer = 0, {}
    for name in tensors:
        if not resident(name):
            continue
        size = _gpu_bytes(name, tensors, config)
        layer = checkpoint.layer_of(name)
        if layer is None:
            shared += size
        else:
            per_layer[layer] = per_layer.get(layer, 0) + size
    count = max(per_layer) + 1 if per_layer else 0
    return shared, [per_layer.get(i, 0) for i in range(count)]
