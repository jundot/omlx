# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 checkpoint configuration, read without a GPU library."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ... import checkpoint

# Hugging Face names in config.json -> names used here.
_ALIASES = {
    "hidden_size": "dim",
    "moe_intermediate_size": "moe_inter_dim",
    "num_hidden_layers": "n_layers",
    "num_attention_heads": "n_heads",
    "num_experts_per_tok": "n_activated_experts",
    "scoring_func": "score_func",
    "routed_scaling_factor": "route_scale",
    "qk_rope_head_dim": "rope_head_dim",
    "rms_norm_eps": "norm_eps",
    "sliding_window": "window_size",
    "kv_source_layer_ids": "kv_source_layers",
    "index_source_layer_ids": "index_source_layers",
    "candidate_source_layer_id": "candidate_source_layer",
    "engram_pad_token_id": "engram_pad_id",
}


@dataclass(frozen=True)
class Config:
    """Model shape. Defaults are those of the reference implementation."""

    vocab_size: int = 129280
    dim: int = 1024
    moe_inter_dim: int = 1024
    n_layers: int = 5
    n_heads: int = 16
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    q_lora_rank: int = 256
    head_dim: int = 128
    rope_head_dim: int = 32
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 256
    window_size: int = 128
    compress_ratios: tuple[int, ...] = (0, 2, 2, 1, 1, 0)
    kv_source_layers: tuple[int, ...] = (1, 3)
    index_source_layers: tuple[int, ...] = (1, 3)
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 16
    index_head_dim: int = 64
    index_topk: int = 64
    candidate_source_layer: int = -1
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 0
    # Module path -> packed weight settings of a converted checkpoint.
    quantized: dict[str, dict[str, Any]] = field(default_factory=dict)

    def kv_source(self, layer: int) -> int:
        """The layer whose compressed KV ``layer`` attends over."""
        return max(i for i in self.kv_source_layers if i <= layer)

    def cache_ratio(self, layer: int) -> int:
        """Compress ratio of the KV a layer writes; 0 when it writes none."""
        return self.compress_ratios[layer] if layer in self.kv_source_layers else 0

    def validate(self) -> None:
        if len(self.compress_ratios) < self.n_layers:
            raise ValueError("compress_ratios must cover every layer")
        if self.n_heads % self.o_groups or self.head_dim % 32:
            raise ValueError("Invalid attention head layout")
        if self.index_head_dim % 32 or self.n_shared_experts != 1:
            raise ValueError("Expected 32-aligned index heads and one shared expert")
        if self.rope_head_dim % 2 or self.rope_head_dim > min(
            self.head_dim, self.index_head_dim
        ):
            raise ValueError("Invalid RoPE head dimension")
        source = index = None
        for i, ratio in enumerate(self.compress_ratios[: self.n_layers]):
            if i in self.kv_source_layers:
                source, index = i, None
            if i in self.index_source_layers:
                index = i
            if ratio and (
                source is None or index is None or self.compress_ratios[source] != ratio
            ):
                raise ValueError(f"Layer {i} has no compatible KV/index source")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("Engram layer/table counts differ")


def parse_config(raw: dict[str, Any]) -> Config:
    """The configuration of a converted checkpoint's ``config.json``."""
    spec = raw.get("omlx_deepseek_v41")
    if raw.get("model_type") != "deepseek_v41" or spec is None:
        raise ValueError("Expected a converted DeepSeek V4.1 checkpoint")
    if spec.get("version") != 1:
        raise ValueError("Unsupported DeepSeek V4.1 converted checkpoint version")
    text = raw.get("text_config") or raw
    values = {_ALIASES.get(k, k): v for k, v in text.items()}
    rope = text.get("rope_scaling") or {}
    values.update(
        original_seq_len=rope.get("original_max_position_embeddings", 0),
        rope_factor=rope.get("factor", 1),
        beta_fast=rope.get("beta_fast", 32),
        beta_slow=rope.get("beta_slow", 1),
        quantized=dict(spec.get("quantized_modules") or {}),
    )
    known = {f.name for f in fields(Config)}
    config = Config(
        **{
            k: tuple(v) if isinstance(v, list) else v
            for k, v in values.items()
            if k in known
        }
    )
    config.validate()
    return config


def load_config(model_dir: str | Path) -> Config:
    return parse_config(checkpoint.load_config(model_dir))


def expert_bits(spec: dict[str, Any]) -> int:
    """Bits per weight a routed expert bank is held with on the GPU.

    The expert tile kernels read 3- and 4-bit affine codes; MLX's expert kernel
    is much slower below 4 bits, so narrower codes are stored as 4-bit codes
    of the same values.
    """
    bits = spec["bits"]
    return 4 if spec["mode"] == "affine" and bits < 3 else bits
