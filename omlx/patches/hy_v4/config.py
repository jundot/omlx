# SPDX-License-Identifier: Apache-2.0
"""Configuration for Tencent Hy4 (``hy_v4_vl``) and Hy4-flash (``hy_v4_flash``).

Field names follow the released ``config.json`` files. Hy4 nests its text
fields under ``text_config`` (``model_type: hy_v4``) and repeats them at the top
level; Hy4-flash only nests them (``model_type: hy_v4_flash``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlx_vlm.models.base import BaseModelConfig

FLASH_TEXT_MODEL_TYPES = frozenset({"hy_v4_flash", "hy_v4_flash_text"})
MODEL_TYPES = frozenset({"hy_v4_vl", "hy_v4_flash"})

_SPARSE = frozenset({"deepseek_sparse_attention", "indexed_attention"})
_SLIDING = frozenset({"sliding_attention"})


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "hy_v4"
    vocab_size: int = 120832
    hidden_size: int = 6144
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.827
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    swiglu_limit: float = 10.0
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 1048576
    attention_bias: bool = False
    learnable_sink_init: float = 0.0
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_magnitude: float = 2.0
    enable_ihc: bool = True
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    indexer_types: list | None = None
    layer_types: list | None = None
    mlp_layer_types: list | None = None
    rope_parameters: dict | None = None
    tie_word_embeddings: bool = False
    enable_lm_head_fp32: bool = True
    pad_token_id: int | None = None
    eos_token_id: Any = None
    # Hy4-flash: sliding layers, latent experts, compressed indexer, Engram.
    sliding_window: int = 512
    swa_num_attention_heads: int | None = None
    swa_qk_nope_head_dim: int | None = None
    swa_qk_rope_head_dim: int | None = None
    swa_v_head_dim: int | None = None
    routed_expert_hidden_size: int | None = None
    compressed_indexer_compress_ratio: int | None = None
    compressed_indexer_num_attention_heads: int | None = None
    compressed_indexer_head_dim: int | None = None
    # transformers' canonical spellings of the fields above.
    index_kpool: int | None = None
    per_layer_config: dict | None = None
    engram_layer_ids: list | None = None
    engram_layer_head_vocab_sizes: dict | None = None
    engram_max_ngram_size: int = 3
    engram_n_head_per_ngram: int = 8
    engram_n_embed_per_ngram: int = 768
    engram_pad_id: int = 120002
    engram_fp8_embedding: bool = True
    engram_seed: int = 0
    engram_lookup_table_size: int = 120818
    attention_kinds: list = field(default_factory=list)

    @property
    def is_flash(self) -> bool:
        return self.model_type in FLASH_TEXT_MODEL_TYPES

    def __post_init__(self):
        n = self.num_hidden_layers
        if self.layer_types is None:
            if self.is_flash:
                self.layer_types = [
                    (
                        "deepseek_sparse_attention"
                        if i < 2 or (i - 1) % 4 == 0
                        else "sliding_attention"
                    )
                    for i in range(n)
                ]
            else:
                self.layer_types = ["deepseek_sparse_attention"] * n
        kinds = []
        for name in self.layer_types:
            if name in _SPARSE:
                kinds.append("sparse")
            elif name in _SLIDING:
                kinds.append("sliding")
            else:
                raise ValueError(f"Unsupported Hy4 attention type: {name!r}")
        self.attention_kinds = kinds
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(1, n) + ["sparse"] * max(n - 1, 0)
        if self.indexer_types is None:
            # Hy4-flash runs a compressed indexer on every sparse layer.
            self.indexer_types = ["full"] * n
        for name, schedule, allowed in (
            ("layer_types", self.layer_types, None),
            ("mlp_layer_types", self.mlp_layer_types, {"dense", "sparse"}),
            ("indexer_types", self.indexer_types, {"full", "shared"}),
        ):
            if len(schedule) != n or (allowed and set(schedule) - allowed):
                raise ValueError(f"Invalid Hy4 {name}: {schedule!r}")
        if not self.is_flash and self.indexer_types[0] != "full":
            raise ValueError("The first Hy4 sparse layer must run its own indexer")
        if not self.enable_ihc or self.hc_mult < 1:
            raise ValueError("Hy4 requires identity hyper-connections")
        for kind in set(kinds):
            params = self.rope_params(kind)
            if params.get("rope_type", "default") != "default":
                raise ValueError(f"Unsupported Hy4 RoPE type: {params['rope_type']!r}")
            if params.get("mrope_interleaved", True) is not True:
                raise ValueError("Only interleaved Hy4 M-RoPE is supported")
        if self.is_flash:
            self.compressed_indexer_compress_ratio = (
                self.compressed_indexer_compress_ratio or self.index_kpool or 4
            )
            sliding = [i for i, k in enumerate(kinds) if k == "sliding"]
            per_layer = self.per_layer_config or {}
            overrides = per_layer.get(str(sliding[0]), {}) if sliding else {}
            for key in (
                "num_attention_heads",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
            ):
                if getattr(self, "swa_" + key) is None and key in overrides:
                    setattr(self, "swa_" + key, overrides[key])
            self.swa_num_attention_heads = (
                self.swa_num_attention_heads or self.num_attention_heads
            )
            self.swa_qk_nope_head_dim = (
                self.swa_qk_nope_head_dim or self.qk_nope_head_dim
            )
            self.swa_qk_rope_head_dim = (
                self.swa_qk_rope_head_dim or self.qk_rope_head_dim
            )
            self.swa_v_head_dim = self.swa_v_head_dim or self.v_head_dim
            self.compressed_indexer_num_attention_heads = (
                self.compressed_indexer_num_attention_heads or self.index_n_heads
            )
            self.compressed_indexer_head_dim = (
                self.compressed_indexer_head_dim or self.index_head_dim
            )
            if self.index_topk % self.compressed_indexer_compress_ratio:
                raise ValueError("index_topk must be a multiple of the group size")

    @property
    def mrope_section(self) -> list | None:
        """Time, height, width channel counts of the interleaved M-RoPE."""
        sections = {
            tuple(section)
            for kind in set(self.attention_kinds)
            if (section := self.rope_params(kind).get("mrope_section")) is not None
        }
        if len(sections) > 1:
            raise ValueError("Hy4 attention kinds must share one mrope_section")
        return list(sections.pop()) if sections else None

    def rope_params(self, kind: str) -> dict:
        """RoPE parameters of one attention kind ("sparse" or "sliding")."""
        params = self.rope_parameters or {}
        if "rope_theta" in params:
            return params
        names = _SPARSE if kind == "sparse" else _SLIDING
        for name in names:
            if name in params:
                return params[name]
        theta = 1e7 if kind == "sparse" else 1e4
        return {"rope_type": "default", "rope_theta": theta}

    def rope_theta(self, kind: str) -> float:
        return float(self.rope_params(kind)["rope_theta"])

    def head_dims(self, layer: int) -> tuple[int, int, int, int]:
        """(heads, nope, rope, value) of one layer's attention."""
        if self.attention_kinds[layer] == "sliding":
            return (
                self.swa_num_attention_heads,
                self.swa_qk_nope_head_dim,
                self.swa_qk_rope_head_dim,
                self.swa_v_head_dim,
            )
        return (
            self.num_attention_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )


@dataclass
class VisionConfig(BaseModelConfig):
    """Vision tower and projector (Hy4 defaults; Hy4-flash ships its own sizes).

    The released checkpoints also carry ``rope_2d_*`` fields, but neither the
    transformers nor the vLLM implementation enables 2D RoPE for them.
    """

    model_type: str = "hy_v4_vl"
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    out_hidden_size: int = 6144
    patch_size: int = 16
    spatial_merge_size: int = 2
    num_channels: int = 3
    rms_norm_eps: float = 1e-5
    max_image_size: int = 2048
    interpolate_mode: str = "bilinear"
    projector_type: str | None = "rmsnorm_mlp_adapter"
    use_2d_rope: bool = False
    cat_extra_token: int = 0
    temporal_patch_size: int = 1

    @classmethod
    def from_dict(cls, params):
        # A checkpoint without vision_config is text-only.
        return super().from_dict(params) if params else None

    def __post_init__(self):
        if self.projector_type != "rmsnorm_mlp_adapter":
            raise ValueError(f"Unsupported Hy4 projector: {self.projector_type!r}")
        if self.use_2d_rope or self.cat_extra_token:
            raise ValueError("Hy4 vision 2D RoPE and extra image tokens are unused")
        if self.interpolate_mode != "bilinear":
            raise ValueError(
                f"Unsupported Hy4 interpolation: {self.interpolate_mode!r}"
            )
        if self.temporal_patch_size != 1:
            raise ValueError("Hy4 vision supports temporal_patch_size 1 only")
        if self.max_image_size % self.patch_size:
            raise ValueError("max_image_size must be a multiple of patch_size")


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = None
    model_type: str = "hy_v4_vl"
    image_token_id: int = 120120
    video_token_id: int = 120124
    image_start_token_id: int = 120118
    image_end_token_id: int = 120119
    eos_token_id: Any = None
    # DFly2 drafter shipped inside the checkpoint (model.dfly_layers.*).
    dfly_num_hidden_layers: int = 0
    dfly_block_size: int | None = None
    dfly_mask_token_id: int | None = None
    dfly_target_kv_layers: list | None = None
    dfly_target_index_layer: list | None = None
    dfly_target_index_topk: int = 2048
    dfly_mtp_fusion: str | None = None
    dfly_enable_hidden_correction: bool = True
    dfly_context_local_window: int | None = None
    dfly_attn_causal: bool = True
    dfly_enable_ihc: bool = False
    dfly_mlp_type: str = "dense"
    speculative_config: dict | None = None

    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        text = params.get("text_config")
        if isinstance(text, dict):
            params["text_config"] = TextConfig.from_dict(text)
        vision = params.get("vision_config")
        if isinstance(vision, dict):
            params["vision_config"] = VisionConfig.from_dict(vision)
        return super().from_dict(params)
