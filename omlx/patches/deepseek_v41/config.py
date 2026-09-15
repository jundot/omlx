# SPDX-License-Identifier: MIT
"""DeepSeek V4.1 configuration; field defaults follow the official reference."""

from dataclasses import dataclass, fields
from typing import Literal


@dataclass
class ModelConfig:
    """Field names are exactly the config JSON keys. The defaults are a small model that
    `python model.py` can run, not the released shapes -- though the scale-independent
    values (norm_eps, score_func, hc_*, engram_*) do match it."""

    # runtime limits rather than model shape: they size the KV caches
    max_batch_size: int = 4
    max_seq_len: int = 4096
    temperature: float = 1
    dtype: Literal["bf16", "fp8"] = "fp8"
    expert_dtype: Literal["fp4"] | None = "fp4"
    vocab_size: int = 129280
    dim: int = 1024
    moe_inter_dim: int = 1024
    n_layers: int = 5
    n_mtp_layers: int = (
        1  # extra draft layers appended after the backbone, indices n_layers..
    )
    n_heads: int = 16
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    # attention: latent q/kv projections, plus a LoRA-factorised output projection over o_groups
    q_lora_rank: int = 256
    head_dim: int = 128
    rope_head_dim: int = 32
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 256
    # sparse attention: every layer attends over a sliding window, and may add compressed KV on top
    window_size: int = 128
    # one entry per layer, MTP layers included: 0 = sliding window only, r = KV compressed r-to-1
    compress_ratios: tuple[int, ...] = (0, 2, 2, 1, 1, 0)
    # layers sharing a ratio also share one compressed KV and one indexer, produced by the first
    kv_source_layers: tuple[int, ...] = (1, 3)
    index_source_layers: tuple[int, ...] = (1, 3)
    # rope, with YaRN extrapolation when original_seq_len > 0. Compressed KV rotates at its own
    # theta because one latent stands for compress_ratio tokens, so its positions are further apart.
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # the indexer: a small extra attention that scores compressed positions, so each query can keep
    # just `index_topk` of them. Names match DeepSeek-V3.2-Exp, where this mechanism first appeared.
    index_n_heads: int = 16
    index_head_dim: int = 64
    index_topk: int = 64
    # candidate pre-filtering: candidate_source_layer < 0 turns it off and the other two are unused
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    # hyper-connections: the residual stream is carried as hc_mult parallel copies
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # engram: n-gram hash lookups added into the residual stream at a few layers
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[
        int, ...
    ] = ()  # unpadded table rows; each rank allocates ceil(rows / world_size)
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = (
        0  # bucket size each (n-gram size, head) starts searching primes from
    )
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_id: int = (
        2  # token that fills n-gram slots with no history; matches training
    )
    # size of the compressed tokenizer vocab; every hash multiplier is derived from it
    engram_compressed_vocab_size: int = 0
    # vision (VL); vision_n_layers == 0 disables the vision path
    vision_n_layers: int = 0
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 544 * 544
    vision_max_wh_ratio: int | None = None
    # raw id of <｜deepseek_image｜>; every position of an image span carries this id in input_ids
    image_token_id: int = 129264
    # Draft weights and standalone forward are separate from server verification.
    preserve_mtp: bool = False
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 0
    dspark_n_activated_experts: int = 0

    @property
    def vision_enabled(self) -> bool:
        return self.vision_n_layers > 0

    def get_moe_config(self, layer_id: int) -> tuple[int, int]:
        """Return the routed/activated expert counts for a given layer."""
        if layer_id < self.n_layers:
            return self.n_routed_experts, self.n_activated_experts
        return (
            self.dspark_n_routed_experts or self.n_routed_experts,
            self.dspark_n_activated_experts or self.n_activated_experts,
        )

    model_type: str = "deepseek_v41"

    @classmethod
    def from_dict(cls, config):
        text = config.get("text_config") or config
        aliases = {
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
            "num_nextn_predict_layers": "n_mtp_layers",
            "dspark_num_experts_per_tok": "dspark_n_activated_experts",
            "max_position_embeddings": "max_seq_len",
        }
        values = {aliases.get(k, k): v for k, v in text.items()}
        values["preserve_mtp"] = bool(
            config.get("omlx_deepseek_v41", {}).get("preserve_mtp", False)
        )
        rope = text.get("rope_scaling") or {}
        values.update(
            {
                aliases.get(k, k): v
                for k, v in config.items()
                if k in ("image_token_id", "model_type")
            }
        )
        values.update(
            {
                "original_seq_len": rope.get("original_max_position_embeddings", 0),
                "rope_factor": rope.get("factor", 1),
                "beta_fast": rope.get("beta_fast", 32),
                "beta_slow": rope.get("beta_slow", 1),
            }
        )
        vision = config.get("vision_config")
        if vision:
            va = {
                "num_hidden_layers": "n_layers",
                "hidden_size": "dim",
                "num_attention_heads": "n_heads",
                "intermediate_size": "inter_dim",
                "max_image_tokens": "max_n_token",
            }
            values.update({"vision_" + va.get(k, k): v for k, v in vision.items()})
        fields_set = {f.name for f in fields(cls)}
        result = cls(**{k: v for k, v in values.items() if k in fields_set})
        result.validate()
        return result

    def validate(self):
        if len(self.compress_ratios) < self.n_layers:
            raise ValueError("compress_ratios must cover every backbone layer")
        if self.n_heads % self.o_groups or self.head_dim % 32:
            raise ValueError(
                "Attention heads must divide into groups; head_dim must divide by 32"
            )
        if self.rope_head_dim % 2 or self.rope_head_dim > min(
            self.head_dim, self.index_head_dim
        ):
            raise ValueError("Invalid RoPE head dimension")
        if self.index_head_dim % 32 or self.n_shared_experts != 1:
            raise ValueError("Expected 32-aligned index heads and one shared expert")
        if not 1 <= self.n_activated_experts <= self.n_routed_experts:
            raise ValueError("Invalid routed expert count")
        source = None
        index = None
        for i, ratio in enumerate(self.compress_ratios[: self.n_layers]):
            if i in self.kv_source_layers:
                source = i
                index = None
            if i in self.index_source_layers:
                index = i
            if ratio and (
                source is None or index is None or self.compress_ratios[source] != ratio
            ):
                raise ValueError(f"Layer {i} has no compatible KV/index source")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("Engram layer/table counts differ")
