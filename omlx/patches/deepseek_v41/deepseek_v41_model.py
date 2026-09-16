# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 Flash MLX model (CSA2) for omlx.

Ported from HuggingFace ``inference/model.py`` to MLX, following
``omlx.patches.deepseek_v4.deepseek_v4_model`` style and reusing V4 kernels
(wsdpa, SwitchGLU, HC sinkhorn, decode_consistency) without breaking V4.
"""
from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from omlx.patches.deepseek_v4.decode_consistency import matmul as decode_matmul
from omlx.patches.deepseek_v4.indexer_dispatch import (
    disable_native_indexer,
    native_indexer_available,
    native_indexer_disabled,
    native_indexer_shape_eligible,
)
from omlx.patches.deepseek_v4.switch_layers import SwitchGLU
from omlx.patches.deepseek_v4.wsdpa_attention import wsdpa_prefill, wsdpa_topk_prefill
from omlx.patches.deepseek_v41.deferred_hc import (
    DeferredHyperConnection,
    hc_collapse,
    hc_expand,
    make_identity_pre_mix,
)
from omlx.patches.deepseek_v41.engram_fast import (
    Engram,
    EngramLayout,
    bind_engram_hash,
    engram_is_mmap,
    engram_mode,
    engram_tables_enabled,
    prefetch_engram_layer,
)
from omlx.patches.deepseek_v41.select_candidates import select_candidate_blocks
from omlx.patches.deepseek_v41.shared_runtime import SharedAttentionRuntime, shared_attn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import BatchPoolingCache, CacheList, PoolingCache, RotatingKVCache

def _is_pooling_cache(obj) -> bool:
    return isinstance(obj, (PoolingCache, BatchPoolingCache))

from .hyper_connection import HyperHead
from .mla import MultiLinear
from .pipeline import PipelineMixin

logger = logging.getLogger(__name__)

_SUPPORTED_COMPRESS = frozenset({0, 1, 2})
_V41_INDEXER_FALLBACK_WARNED = False


def _extend_mask(mask: Optional[mx.array], pool_mask: Optional[mx.array], N: int):
    """Append pooled-KV visibility onto a window mask (V4 helper port)."""
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask[None, None]
    B, H, L, S = mask.shape
    if pool_mask is None:
        pool_mask = mx.ones((B, H, L, N - S), dtype=mx.bool_)
    elif pool_mask.ndim == 2:
        pool_mask = mx.broadcast_to(pool_mask, (B, H, L, N - S))
    elif pool_mask.ndim == 3:
        pool_mask = mx.broadcast_to(pool_mask[:, None], (B, H, L, N - S))
    return mx.concatenate([mask, pool_mask], axis=-1)



def _flatten_text_config(params: dict) -> dict:
    """Merge nested HF ``text_config`` into top-level ModelArgs fields."""
    out = dict(params)
    text = out.pop("text_config", None)
    if isinstance(text, dict):
        for k, v in text.items():
            out.setdefault(k, v)
    # Drop vision_config from text ModelArgs (vision is Phase-2 stub).
    out.pop("vision_config", None)
    # Alias HF -> internal names when needed
    aliases = {
        "score_func": "scoring_func",
        "route_scale": "routed_scaling_factor",
        "n_activated_experts": "num_experts_per_tok",
        "window_size": "sliding_window",
        "norm_eps": "rms_norm_eps",
        "dim": "hidden_size",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "moe_inter_dim": "moe_intermediate_size",
        "rope_head_dim": "qk_rope_head_dim",
        "kv_source_layers": "kv_source_layer_ids",
        "index_source_layers": "index_source_layer_ids",
        "candidate_source_layer": "candidate_source_layer_id",
        "engram_pad_id": "engram_pad_token_id",
        "dspark_n_activated_experts": "dspark_num_experts_per_tok",
    }
    for src, dst in aliases.items():
        if src in out and dst not in out:
            out[dst] = out[src]
    # YaRN fields may be nested under rope_scaling already in HF config.
    if "rope_scaling" not in out and out.get("original_seq_len"):
        out["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": out.get("rope_factor", 16),
            "beta_fast": out.get("beta_fast", 32),
            "beta_slow": out.get("beta_slow", 1),
            "original_max_position_embeddings": out["original_seq_len"],
        }
    return out


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v41"
    vocab_size: int = 129280
    hidden_size: int = 5120
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2304
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    n_shared_experts: int = 1
    n_routed_experts: int = 384
    routed_scaling_factor: float = 1.5
    q_lora_rank: int = 1280
    qk_rope_head_dim: int = 64
    num_experts_per_tok: int = 6
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    rms_norm_eps: float = 1e-20
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 512
    scoring_func: str = "sqrtsoftplus"
    compress_ratios: List[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    kv_source_layer_ids: List[int] = field(default_factory=list)
    index_source_layer_ids: List[int] = field(default_factory=list)
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    engram_layer_ids: List[int] = field(default_factory=list)
    engram_num_embeddings: List[int] = field(default_factory=list)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092
    num_nextn_predict_layers: int = 3
    n_mtp_layers: int = 3
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: List[int] = field(default_factory=list)
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    tie_word_embeddings: bool = False
    topk_method: str = "noaux_tc"
    image_token_id: int = 129264
    # Vision stub markers
    vision_enabled: bool = False
    vision_config: Optional[Dict] = None

    @classmethod
    def from_dict(cls, params):
        flat = _flatten_text_config(params)
        # vision_enabled from nested vision_config
        vc = params.get("vision_config") if isinstance(params, dict) else None
        if isinstance(vc, dict) and int(vc.get("num_hidden_layers", 0) or 0) > 0:
            flat["vision_enabled"] = True
            flat["vision_config"] = vc
        sig = inspect.signature(cls)
        return cls(**{k: v for k, v in flat.items() if k in sig.parameters})

    def __post_init__(self):
        if isinstance(self.compress_ratios, tuple):
            self.compress_ratios = list(self.compress_ratios)
        if not self.compress_ratios:
            # Minimal default for tiny unit tests: all window-only
            self.compress_ratios = [0] * self.num_hidden_layers
        # Config may include MTP trailing ratios; trim or pad to backbone layers
        ratios = list(self.compress_ratios)
        if len(ratios) < self.num_hidden_layers:
            ratios = ratios + [0] * (self.num_hidden_layers - len(ratios))
        self.compress_ratios = ratios[: self.num_hidden_layers]
        bad = [r for r in self.compress_ratios if r not in _SUPPORTED_COMPRESS]
        if bad:
            raise ValueError(
                f"Unsupported DeepSeek-V4.1 compress ratios {bad}; "
                f"allowed {_SUPPORTED_COMPRESS}"
            )
        self.kv_source_layer_ids = list(self.kv_source_layer_ids or [])
        self.index_source_layer_ids = list(self.index_source_layer_ids or [])
        self.engram_layer_ids = list(self.engram_layer_ids or [])
        self.engram_num_embeddings = list(self.engram_num_embeddings or [])
        self.dspark_target_layer_ids = list(self.dspark_target_layer_ids or [])
        if self.n_mtp_layers <= 0 and self.num_nextn_predict_layers:
            self.n_mtp_layers = int(self.num_nextn_predict_layers)
        # Engram tables are ~190GB on Flash. stub/off: disable layer hooks.
        # mmap/ssd/full: keep engram_layer_ids (mmap does not allocate tables in RAM).
        if not engram_tables_enabled():
            self.engram_layer_ids = []
            self.engram_num_embeddings = []

    def get_moe_config(self, layer_id: int) -> Tuple[int, int]:
        if layer_id < self.num_hidden_layers:
            return self.n_routed_experts, self.num_experts_per_tok
        return (
            self.dspark_n_routed_experts or self.n_routed_experts,
            self.dspark_num_experts_per_tok or self.num_experts_per_tok,
        )


def make_quantization_config(model):
    """Quant notes (paper / convert): window fp8@32; compress KV fp4@16/E4M3;
    index fp4@32; weight block 32x32. Experts mxfp4; shared/attn mxfp8."""
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}

    flat_modules = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    experts = {
        k: mxfp4
        for k, _ in flat_modules
        if ".ffn.switch_mlp." in k and k.endswith("_proj")
    }
    shared_experts = {k: mxfp8 for k, _ in flat_modules if ".ffn.shared_experts." in k}
    attn = {
        k: mxfp8
        for k, _ in flat_modules
        # indexer.wk is bf16 in the official checkpoint (not fp8).
        if ".attn.w" in k or ".attn.indexer.wq" in k
    }
    mtp_projs = {
        k: mxfp8
        for k, _ in flat_modules
        if k.startswith("mtp.")
        and (k.endswith(".e_proj") or k.endswith(".h_proj") or k.endswith(".main_proj"))
    }
    engram_wkv = {k: mxfp8 for k, _ in flat_modules if k.endswith(".engram.wkv")}
    return {
        "group_size": 32,
        "bits": 8,
        "mode": "affine",
        **experts,
        **shared_experts,
        **attn,
        **mtp_projs,
        **engram_wkv,
    }


def _score_func(scores: mx.array, func: str) -> mx.array:
    if func == "softmax":
        return mx.softmax(scores, axis=-1, precise=True)
    if func == "sigmoid":
        return mx.sigmoid(scores)
    if func == "sqrtsoftplus":
        return mx.sqrt(nn.softplus(scores))
    raise ValueError(f"Unsupported scoring function: {func}")


@mx.compile
def _expert_select(
    logits: mx.array,
    e_score_correction_bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    scoring_func: str,
) -> Tuple[mx.array, mx.array]:
    """Bias selects experts; weights come from unbiased scores (official Gate)."""
    logits = logits.astype(mx.float32)
    scores = _score_func(logits, scoring_func)
    biased = scores + e_score_correction_bias
    inds = mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    if scoring_func != "softmax" and norm_topk_prob:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    weights = weights * routed_scaling_factor
    return inds, weights


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    if limit and limit > 0:
        gate = mx.minimum(gate, limit)
        up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class LimitedSwiGLU(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _limited_swiglu(gate, x, self.limit)


class DeepseekV41RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float,
        scaling_config: Optional[Dict] = None,
        max_position_embeddings: int = 1048576,
        freq_scale: int = 1,
    ):
        super().__init__()
        self.dims = dims
        self.freq_scale = freq_scale
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")
        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            original_max_position_embeddings = scaling_config[
                "original_max_position_embeddings"
            ]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return (
                    dims
                    * math.log(
                        original_max_position_embeddings / (num_rotations * 2 * math.pi)
                    )
                    / (2 * math.log(base))
                )

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001
            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported RoPE type: {rope_type}")
        self._freqs = 1.0 / inv_freq
        self._freqs_cache = {}

    def _get_freqs(self, head_dim: int, inverse: bool):
        key = (head_dim, inverse)
        if key not in self._freqs_cache:
            f = self._freqs
            if self.freq_scale != 1:
                f = f / self.freq_scale
            if inverse:
                f = -f
            nope_pairs = (head_dim - self.dims) // 2
            if nope_pairs > 0:
                f = mx.concatenate([mx.full((nope_pairs,), mx.inf), f])
            self._freqs_cache[key] = f
        return self._freqs_cache[key]

    def __call__(self, x: mx.array, offset: Any = 0, inverse: bool = False) -> mx.array:
        head_dim = x.shape[-1]
        freqs = self._get_freqs(head_dim, inverse)
        offset = offset // self.freq_scale if self.freq_scale != 1 else offset
        return mx.fast.rope(
            x,
            head_dim,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=freqs,
        )


@partial(mx.compile, shapeless=True)
def _simple_compress_kv(kv, gate, head_dim):
    """Softmax-gate pool over compress_ratio tokens. No APE (V4.1)."""
    weights = mx.softmax(gate.astype(mx.float32), axis=-2)
    weights = weights.astype(kv.dtype)
    return (kv * weights).sum(axis=-2)

class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        n_routed, top_k = config.get_moe_config(layer_idx)
        self.top_k = top_k
        self.num_experts = n_routed
        self.hidden_dim = config.hidden_size
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.num_experts, self.hidden_dim))
        self.e_score_correction_bias = mx.zeros((self.num_experts,), dtype=mx.float32)
        self.bias_vl = None
        if config.vision_enabled:
            self.bias_vl = mx.zeros((self.num_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array, image_mask: Optional[mx.array] = None):
        logits = decode_matmul(x, self.weight.T)
        bias = self.e_score_correction_bias
        if image_mask is not None and self.bias_vl is not None:
            m = image_mask
            while m.ndim < logits.ndim:
                m = m[..., None]
            bias = mx.where(m, self.bias_vl, bias)
        return _expert_select(
            logits,
            bias,
            self.top_k,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.scoring_func,
        )


class DeepseekV41MLP(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        intermediate_size: Optional[int] = None,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        hs = config.hidden_size
        mid = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hs, mid, bias=False)
        self.up_proj = nn.Linear(hs, mid, bias=False)
        self.down_proj = nn.Linear(mid, hs, bias=False)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _limited_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit)
        )


class DeepseekV41MoE(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        n_routed, _ = config.get_moe_config(layer_idx)
        self.gate = MoEGate(config, layer_idx)
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            n_routed,
            activation=LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = DeepseekV41MLP(
            config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            swiglu_limit=config.swiglu_limit,
        )

    def __call__(self, x: mx.array, image_mask: Optional[mx.array] = None) -> mx.array:
        inds, scores = self.gate(x, image_mask)
        y = self.switch_mlp(x, inds, scores=scores)
        if y.ndim == scores.ndim + 1:
            y = (y * scores[..., None].astype(y.dtype)).sum(-2)
        return y + self.shared_experts(x)


class Compressor(nn.Module):
    """Pools compress_ratio tokens. No APE / no ratio-4 overlap (V4.1)."""

    def __init__(self, config: ModelArgs, compress_ratio: int, head_dim: int):
        super().__init__()
        if compress_ratio not in (1, 2):
            raise ValueError(f"V4.1 compressor expects ratio 1 or 2, got {compress_ratio}")
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.wkv = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.wgate = (
            nn.Linear(config.hidden_size, head_dim, bias=False)
            if compress_ratio > 1
            else None
        )
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rope = DeepseekV41RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
            freq_scale=compress_ratio,
        )

    def project(self, x: mx.array):
        return self.wkv(x), (self.wgate(x) if self.wgate is not None else None)

    def consume(
        self,
        kv: mx.array,
        gate: Optional[mx.array],
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
        apply_rope: bool = True,
    ) -> mx.array:
        B = kv.shape[0]
        ratio = self.compress_ratio
        if ratio == 1:
            new_pooled = self.norm(kv)
            if apply_rope:
                new_pooled = self.rope(new_pooled[:, None], offset=offset).squeeze(1)
            if pool_cache is not None:
                new_pooled = pool_cache.update_and_fetch(new_pooled)
            return new_pooled

        if pool_cache is None:
            usable = (kv.shape[1] // ratio) * ratio
            ready_kv = kv[:, :usable]
            ready_gate = gate[:, :usable]
            pool_base = offset
        else:
            ready_kv, ready_gate, pool_base = pool_cache.accumulate_windows(
                kv, gate, offset
            )

        if ready_kv.size == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=kv.dtype)
        else:
            kv_u = mx.unflatten(ready_kv, 1, (-1, ratio))
            gate_u = mx.unflatten(ready_gate, 1, (-1, ratio))
            new_pooled = self.norm(_simple_compress_kv(kv_u, gate_u, self.head_dim))
            if apply_rope:
                new_pooled = self.rope(new_pooled[:, None], offset=pool_base).squeeze(1)

        if pool_cache is not None:
            new_pooled = pool_cache.update_and_fetch(new_pooled)
        return new_pooled

    def __call__(
        self,
        x: mx.array,
        pool_cache: Optional[PoolingCache],
        offset: Union[int, mx.array],
        apply_rope: bool = True,
    ) -> mx.array:
        return self.consume(*self.project(x), pool_cache, offset, apply_rope=apply_rope)


def _stable_topk_indices(scores: mx.array, k: int) -> mx.array:
    partition = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    selected = mx.take_along_axis(scores, partition, axis=-1)
    threshold = mx.min(selected, axis=-1, keepdims=True)
    size = scores.shape[-1]
    positions = mx.arange(size, dtype=mx.uint32)
    region = mx.where(
        scores > threshold,
        0,
        mx.where(scores == threshold, 1, 2),
    ).astype(mx.uint32)
    keys = region * size + positions
    indices = mx.argpartition(keys, kth=k - 1, axis=-1)[..., :k]
    # wsdpa_topk / native sparse require uint32 temporally-sorted indices.
    return mx.sort(indices, axis=-1).astype(mx.uint32)


class Indexer(nn.Module):
    """Indexer with wk from compressor latent; optional candidate blocks."""

    def __init__(self, config: ModelArgs, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.owns_k = layer_id in config.kv_source_layer_ids
        self.compress_ratio = config.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_id
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.index_head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.wk = None
        self.k_norm = None
        if self.owns_k:
            self.wk = nn.Linear(config.head_dim, self.index_head_dim, bias=False)
            self.k_norm = nn.RMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.rope = DeepseekV41RoPE(
            config.qk_rope_head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
            freq_scale=max(self.compress_ratio, 1),
        )

    def _resolve_index_k(
        self,
        latent: Optional[mx.array],
        index_pool: Optional[PoolingCache],
        offset: Union[int, mx.array],
        runtime: SharedAttentionRuntime,
    ) -> Optional[mx.array]:
        if self.owns_k and latent is not None and latent.shape[1] > 0:
            k = self.k_norm(self.wk(latent))
            k = self.rope(k[:, None], offset=offset).squeeze(1)
            if index_pool is not None:
                k = index_pool.update_and_fetch(k)
                runtime.index_k = index_pool
            else:
                runtime.index_k = k
            return k
        src = runtime.index_k
        if src is None:
            return None
        if _is_pooling_cache(src):
            return src.pooled
        return src

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        latent: Optional[mx.array],
        index_pool: Optional[PoolingCache],
        offset: Union[int, mx.array],
        runtime: SharedAttentionRuntime,
        k_offset: Optional[Union[int, mx.array]] = None,
    ) -> Optional[mx.array]:
        B, L, _ = x.shape
        if self.compress_ratio <= 0:
            return None
        # K RoPE uses group-start positions (pool_base); Q uses absolute tokens.
        ko = offset if k_offset is None else k_offset
        pooled_k = self._resolve_index_k(latent, index_pool, ko, runtime)
        if pooled_k is None or pooled_k.shape[1] == 0:
            return None

        q = self.wq_b(qr).reshape(B, L, self.n_heads, self.index_head_dim)
        q = self.rope(q.transpose(0, 2, 1, 3), offset)  # [B,H,L,D]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        P = pooled_k.shape[1]
        k = min(self.index_topk, P)

        if isinstance(offset, mx.array):
            # Batched decode/prefill: per-row absolute offsets [B].
            if L > 1:
                qpos = offset.astype(mx.int32)[:, None] + mx.arange(1, L + 1)[None, :]
                compress_lens = (qpos // self.compress_ratio)[:, :, None]
                compress_lens_arg = compress_lens.squeeze(-1)
            else:
                compress_lens = None
                compress_lens_arg = (offset.astype(mx.int32) + L) // self.compress_ratio
        else:
            off = int(offset)
            if L > 1:
                compress_lens = ((mx.arange(1, L + 1) + off) // self.compress_ratio)[
                    None, :, None
                ]
                compress_lens_arg = compress_lens.squeeze(-1)
            else:
                compress_lens = None
                compress_lens_arg = (off + L) // self.compress_ratio

        # Prefill: fuse score GEMM via dsa_indexer_scores when Flash shapes match
        # (H=32, D=128, topk=512). Kernel tiles are 64x64 — pad L/P then slice.
        # Decode stays on the MLX row path (+ Metal top-k) by design.
        if (
            isinstance(offset, int)
            and native_indexer_shape_eligible(
                query_tokens=L,
                pooled_tokens=P,
                n_heads=self.n_heads,
                head_dim=self.index_head_dim,
                index_topk=self.index_topk,
                dtype_supported=q.dtype in (mx.float16, mx.bfloat16),
            )
        ):
            try:
                from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

                global _V41_INDEXER_FALLBACK_WARNED
                if not native_indexer_available() and not native_indexer_disabled():
                    if not _V41_INDEXER_FALLBACK_WARNED:
                        _V41_INDEXER_FALLBACK_WARNED = True
                        logger.warning(
                            "deepseek_v41: native dsa_indexer_scores/"
                            "dsa_topk_indices unavailable; falling back to MLX"
                        )
                if native_indexer_available():
                    tile = 64
                    Lp = (L + tile - 1) // tile * tile
                    Pp = (P + tile - 1) // tile * tile
                    q_in = q
                    w_in = weights.astype(q.dtype)
                    k_in = pooled_k
                    if Lp > L:
                        q_in = mx.pad(q_in, [(0, 0), (0, 0), (0, Lp - L), (0, 0)])
                        w_in = mx.pad(w_in, [(0, 0), (0, Lp - L), (0, 0)])
                    if Pp > P:
                        k_in = mx.pad(k_in, [(0, 0), (0, Pp - P), (0, 0)])
                    # mask_ratio=0: apply compress_lens after slicing so padded
                    # pooled columns cannot leak past the real compress horizon.
                    scores4 = glm_fast.dsa_indexer_scores(
                        mx.contiguous(q_in),
                        mx.contiguous(k_in[:, None]),
                        mx.contiguous(w_in),
                        causal=False,
                        mask_ratio=0,
                        mask_q_offset=0,
                    )
                    scores = scores4[:, 0, :L, :P]
                    if L > 1:
                        pos = mx.arange(P)[None, None, :]
                        scores = mx.where(
                            pos >= compress_lens,
                            mx.array(-mx.inf, dtype=scores.dtype),
                            scores,
                        )
                    else:
                        pos = mx.arange(P)
                        scores = mx.where(
                            pos >= compress_lens_arg,
                            mx.array(-mx.inf, dtype=scores.dtype),
                            scores,
                        )

                    if self.is_candidate_source:
                        runtime.candidates = select_candidate_blocks(
                            scores,
                            compress_lens_arg,
                            self.candidate_topk_blocks,
                            self.candidate_block_size,
                        )
                    elif self.uses_candidates and runtime.candidates is not None:
                        scores = mx.where(
                            runtime.candidates,
                            scores,
                            mx.array(-mx.inf, dtype=scores.dtype),
                        )

                    scores4 = scores[:, None]
                    indices = glm_fast.dsa_topk_indices(
                        scores4,
                        self.index_topk,
                        bucketed=False,
                    )
                    if indices.ndim == 4:
                        indices = indices[:, 0]
                    return mx.sort(indices, axis=-1).astype(mx.uint32)
            except Exception:
                disable_native_indexer()
                logger.warning(
                    "DSV4.1 native indexer top-k failed; MLX fallback for "
                    "the rest of this process",
                    exc_info=True,
                )

        # Official: einsum bshd,btd->bsht then (relu * weights).sum(heads)
        # Decode: fused Metal score row when H=32 / D=128 / L=1.
        scores = None
        if L == 1 and self.n_heads == 32 and self.index_head_dim == 128:
            try:
                from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

                if glm_fast.has_symbol("dsa_decode_scores"):
                    scores4 = glm_fast.dsa_decode_scores(
                        mx.contiguous(q),
                        mx.contiguous(pooled_k[:, None]),
                        mx.contiguous(weights.astype(q.dtype)),
                        fp32_scores=True,
                    )
                    scores = scores4[:, 0] if scores4.ndim == 4 else scores4
            except Exception:
                scores = None
        if scores is None:
            scores = q @ pooled_k[:, None].swapaxes(-1, -2)  # [B,H,L,P]
            wh = weights.transpose(0, 2, 1)[:, :, :, None]  # [B,H,L,1]
            scores = (mx.maximum(scores, 0) * wh).sum(axis=1)  # [B,L,P]

        if L > 1:
            pos = mx.arange(P)[None, None, :]
            scores = mx.where(
                pos >= compress_lens,
                mx.array(-mx.inf, dtype=scores.dtype),
                scores,
            )
        else:
            pos = mx.arange(P)
            if isinstance(compress_lens_arg, mx.array) and compress_lens_arg.ndim >= 1:
                lim = compress_lens_arg.reshape(B, 1, 1)
                scores = mx.where(
                    pos.reshape(1, 1, P) >= lim,
                    mx.array(-mx.inf, dtype=scores.dtype),
                    scores,
                )
            else:
                scores = mx.where(
                    pos >= compress_lens_arg,
                    mx.array(-mx.inf, dtype=scores.dtype),
                    scores,
                )

        if self.is_candidate_source:
            runtime.candidates = select_candidate_blocks(
                scores,
                compress_lens_arg,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif self.uses_candidates and runtime.candidates is not None:
            scores = mx.where(
                runtime.candidates,
                scores,
                mx.array(-mx.inf, dtype=scores.dtype),
            )

        # Decode: Metal top-k over the materialised score row (topk fixed at 512).
        if L == 1 and k == 512 and P > k:
            try:
                from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

                if glm_fast.has_symbol("dspark_fp32_topk_indices"):
                    flat = scores.reshape(-1, P).astype(mx.float32)
                    indices = glm_fast.dspark_fp32_topk_indices(flat, 512)
                    indices = indices.reshape(B, L, 512)
                    return mx.sort(indices, axis=-1).astype(mx.uint32)
            except Exception:
                pass

        return _stable_topk_indices(scores, k)


def _project_attention_output(attn: nn.Module, out: mx.array, offset: Any) -> mx.array:
    out = attn.rope(out, offset, inverse=True)

    def prepare(row: mx.array) -> mx.array:
        batch, _, length, _ = row.shape
        row = row.reshape(batch, attn.o_groups, -1, length, attn.head_dim)
        return row.transpose(0, 1, 3, 2, 4).flatten(-2)

    def finish(row: mx.array) -> mx.array:
        return row.transpose(0, 2, 1, 3).flatten(-2)

    return attn.wo_b(finish(attn.wo_a(prepare(out))))


_V41_SPARSE_NATIVE_DISABLED = False


def _try_native_sparse_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    topk: mx.array,
    sinks: mx.array,
    scale: float,
    offset: int,
    compress_ratio: int,
    local_window: int,
) -> Optional[mx.array]:
    """Fused window+topk sparse attention (Metal). Prefill L>1; decode via wsdpa."""
    global _V41_SPARSE_NATIVE_DISABLED
    if _V41_SPARSE_NATIVE_DISABLED:
        return None
    if (
        q.shape[0] != 1
        or q.shape[1] != 64
        # Extension builds historically reject L==1; keep prefill-only until
        # deepseek_v4_sparse_attention.cpp allows decode (then drop this).
        or q.shape[2] <= 1
        or q.shape[3] != 512
        or topk.dtype != mx.uint32
        or topk.ndim != 3
        or pooled is None
        or pooled.shape[1] == 0
    ):
        return None
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

        if not glm_fast.has_symbol("deepseek_v4_sparse_attention"):
            _V41_SPARSE_NATIVE_DISABLED = True
            return None
        return glm_fast.deepseek_v4_sparse_attention(
            q,
            local_kv,
            pooled,
            topk[:, None],
            sinks,
            scale,
            int(offset),
            int(compress_ratio),
            int(local_window),
        )
    except Exception:
        _V41_SPARSE_NATIVE_DISABLED = True
        logger.warning(
            "DSV4.1 native sparse attention failed; using MLX fallback",
            exc_info=True,
        )
        return None


class Attention(nn.Module):
    """Unified CSA2 attention: window KV + optional compressed top-k.

    compress_ratio > 0 does not imply this layer owns compression — only
    ``kv_source_layer_ids`` run the compressor; others reuse SharedAttentionRuntime.
    """

    def __init__(self, config: ModelArgs, layer_id: int, runtime: SharedAttentionRuntime):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.runtime = runtime
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.compress_ratio = config.compress_ratios[layer_id]
        self.is_kv_source = layer_id in config.kv_source_layer_ids
        self.is_index_source = layer_id in config.index_source_layer_ids
        self.scale = self.head_dim**-0.5

        self.attn_sink = mx.zeros((self.n_heads,), dtype=mx.float32)
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = MultiLinear(
            (self.n_heads * self.head_dim) // self.o_groups,
            self.o_lora_rank,
            self.o_groups,
        )
        self.wo_b = nn.Linear(self.o_groups * self.o_lora_rank, config.hidden_size, bias=False)

        rope_theta = (
            config.compress_rope_theta if self.compress_ratio else config.rope_theta
        )
        rope_scaling = config.rope_scaling if self.compress_ratio else config.rope_scaling
        # Pure window layers still use YaRN from config in HF; match official:
        # ratio==0 disables YaRN (original_seq_len=0) and uses base rope_theta.
        if self.compress_ratio == 0:
            rope_scaling = None
            rope_theta = config.rope_theta
        self.rope = DeepseekV41RoPE(
            config.qk_rope_head_dim,
            rope_theta,
            rope_scaling,
            config.max_position_embeddings,
            freq_scale=1,
        )

        self.compressor = None
        self.indexer = None
        if self.is_kv_source and self.compress_ratio > 0:
            self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
        if self.is_index_source and self.compress_ratio > 0:
            self.indexer = Indexer(config, layer_id)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        *,
        _standard_mask: bool = False,
    ) -> mx.array:
        B, L, _ = x.shape
        runtime = self.runtime

        local_cache = None
        kv_pool = None
        index_pool = None
        if isinstance(cache, CacheList):
            local_cache = cache[0]
            if len(cache.caches) > 1:
                kv_pool = cache[1]
            if len(cache.caches) > 2:
                index_pool = cache[2]
        else:
            local_cache = cache

        offset = 0
        if local_cache is not None:
            offset = local_cache.offset

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).reshape(B, L, self.n_heads, self.head_dim)
        q = self.rope(q.transpose(0, 2, 1, 3), offset)

        # Window KV (MLA-style single latent head, broadcast)
        kv = self.kv_norm(self.wkv(x))
        kv = self.rope(kv[:, None], offset)  # [B,1,L,D]
        empty_v = mx.zeros((*kv.shape[:-1], 0), dtype=kv.dtype)
        if local_cache is not None:
            keys, _ = local_cache.update_and_fetch(kv, empty_v)
        else:
            keys = kv

        sinks = self.attn_sink.astype(q.dtype)
        pooled = None
        topk = None
        pool_base_for_index = offset

        if self.compress_ratio > 0:
            latent = None
            if self.is_kv_source and self.compressor is not None:
                kv_proj, gate = self.compressor.project(x)
                if self.compress_ratio == 1:
                    latent = self.compressor.norm(kv_proj)
                    roped = self.compressor.rope(latent[:, None], offset=offset).squeeze(1)
                    pooled = (
                        kv_pool.update_and_fetch(roped)
                        if kv_pool is not None
                        else roped
                    )
                    pool_base_for_index = offset
                else:
                    # Official order: accumulate complete windows -> unroped
                    # latent for indexer -> RoPE into compress KV pool.
                    ratio = self.compress_ratio
                    if kv_pool is not None:
                        ready_kv, ready_gate, pool_base = kv_pool.accumulate_windows(
                            kv_proj, gate, offset
                        )
                    else:
                        usable = (kv_proj.shape[1] // ratio) * ratio
                        ready_kv = kv_proj[:, :usable]
                        ready_gate = gate[:, :usable]
                        pool_base = offset
                    pool_base_for_index = pool_base
                    if ready_kv.size == 0:
                        latent = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
                        pooled = kv_pool.pooled if kv_pool is not None else latent
                        if pooled is None:
                            pooled = latent
                    else:
                        kv_u = mx.unflatten(ready_kv, 1, (-1, ratio))
                        gate_u = mx.unflatten(ready_gate, 1, (-1, ratio))
                        latent = self.compressor.norm(
                            _simple_compress_kv(kv_u, gate_u, self.head_dim)
                        )
                        roped = self.compressor.rope(
                            latent[:, None], offset=pool_base
                        ).squeeze(1)
                        pooled = (
                            kv_pool.update_and_fetch(roped)
                            if kv_pool is not None
                            else roped
                        )
                runtime.compress_kv = kv_pool if kv_pool is not None else pooled
            else:
                src = runtime.compress_kv
                if _is_pooling_cache(src):
                    pooled = src.pooled
                else:
                    pooled = src

            if self.is_index_source and self.indexer is not None:
                topk = self.indexer(
                    x,
                    qr,
                    latent,
                    index_pool,
                    offset,
                    runtime,
                    k_offset=pool_base_for_index,
                )
                runtime.topk_idxs = topk
            else:
                topk = runtime.topk_idxs

        # Attention: window (+ optional pooled). Always extend mask when
        # concatenating pooled rows so decode/prefill without wsdpa stay valid.
        pooled_mask = None
        if (
            pooled is not None
            and pooled.shape[1] > 0
            and kv_pool is not None
            and L > 1
        ):
            pooled_mask = kv_pool.make_mask(L, offset)

        if pooled is None or pooled.shape[1] == 0:
            out = None
            if _standard_mask and B == 1 and L > 1:
                out = wsdpa_prefill(
                    q,
                    keys,
                    None,
                    sinks,
                    self.scale,
                    offset,
                    self.config.sliding_window,
                    self.compress_ratio or 1,
                )
            if out is None:
                out = scaled_dot_product_attention(
                    q,
                    keys,
                    keys,
                    cache=None,
                    scale=self.scale,
                    mask=mask,
                    sinks=sinks,
                )
        elif topk is not None and pooled.shape[1] > self.config.index_topk:
            # Sparse top-k: native / wsdpa_topk for prefill; gather for decode.
            out = None
            if L > 1 and _standard_mask and B == 1:
                if topk.dtype != mx.uint32:
                    topk = topk.astype(mx.uint32)
                # Prefer native Metal sparse (faster on measured V4.1 ratios);
                # fall back to wsdpa_topk then gather/dense.
                out = _try_native_sparse_attention(
                    q,
                    keys,
                    pooled,
                    topk,
                    sinks,
                    self.scale,
                    offset,
                    self.compress_ratio or 1,
                    self.config.sliding_window,
                )
                if out is None:
                    out = wsdpa_topk_prefill(
                        q,
                        keys,
                        pooled,
                        topk,
                        sinks,
                        self.scale,
                        offset,
                        self.config.sliding_window,
                        self.compress_ratio or 1,
                    )
            if out is None and L == 1:
                # Gather K rows without materializing [B,1,1,P,D].
                # topk: [B,1,K] -> gathered [B,1,K,D]
                k = int(topk.shape[-1])
                idx = mx.broadcast_to(
                    topk.reshape(B, k)[:, :, None],
                    (B, k, self.head_dim),
                )
                gathered = mx.take_along_axis(pooled, idx, axis=1)
                gathered = gathered.reshape(B, 1, k, self.head_dim)
                full_kv = mx.concatenate([keys, gathered], axis=2)
                out = scaled_dot_product_attention(
                    q,
                    full_kv,
                    full_kv,
                    cache=None,
                    scale=self.scale,
                    mask=None,
                    sinks=sinks,
                )
            if out is None:
                # Last resort: dense pooled (correct but heavier than top-k).
                full_kv = mx.concatenate([keys, pooled[:, None]], axis=2)
                if _standard_mask and B == 1 and L > 1:
                    out = wsdpa_prefill(
                        q,
                        keys,
                        pooled,
                        sinks,
                        self.scale,
                        offset,
                        self.config.sliding_window,
                        self.compress_ratio or 1,
                    )
                if out is None:
                    attn_mask = _extend_mask(mask, pooled_mask, full_kv.shape[2])
                    out = scaled_dot_product_attention(
                        q,
                        full_kv,
                        full_kv,
                        cache=None,
                        scale=self.scale,
                        mask=attn_mask,
                        sinks=sinks,
                    )
        else:
            full_kv = mx.concatenate([keys, pooled[:, None]], axis=2)
            out = None
            if _standard_mask and B == 1 and L > 1:
                out = wsdpa_prefill(
                    q,
                    keys,
                    pooled,
                    sinks,
                    self.scale,
                    offset,
                    self.config.sliding_window,
                    self.compress_ratio or 1,
                )
            if out is None:
                attn_mask = _extend_mask(mask, pooled_mask, full_kv.shape[2])
                out = scaled_dot_product_attention(
                    q,
                    full_kv,
                    full_kv,
                    cache=None,
                    scale=self.scale,
                    mask=attn_mask,
                    sinks=sinks,
                )

        return _project_attention_output(self, out, offset)


class DeepseekV41Block(nn.Module):
    """Deferred mHC block: attn uses prior pre_mix; FFN uses this attn pre."""

    def __init__(
        self,
        config: ModelArgs,
        layer_idx: int,
        runtime: SharedAttentionRuntime,
        engram_layout: Optional[EngramLayout] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = Attention(config, layer_idx, runtime)
        self.ffn = DeepseekV41MoE(config, layer_idx)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeferredHyperConnection(config)
        self.ffn_hc = DeferredHyperConnection(config)
        self.engram = None
        if engram_layout is not None and layer_idx in engram_layout.layer_ids:
            self.engram = Engram(config, layer_idx, engram_layout)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        pre_mix: mx.array,
        image_mask: Optional[mx.array] = None,
        hash_ids: Optional[mx.array] = None,
        *,
        _standard_mask: bool = False,
    ):
        if self.engram is not None:
            h = self.engram(h, hash_ids, None if image_mask is None else ~image_mask)

        residual = h
        attn_pre, attn_post, attn_comb = self.attn_hc.mixes(h)
        x = hc_collapse(h, pre_mix)
        x = self.attn_norm(x)
        x = self.attn(x, mask=mask, cache=cache, _standard_mask=_standard_mask)
        h = hc_expand(x, residual, attn_post, attn_comb)

        residual = h
        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.mixes(h)
        x = hc_collapse(h, attn_pre)
        x = self.ffn_norm(x)
        x = self.ffn(x, image_mask)
        h = hc_expand(x, residual, ffn_post, ffn_comb)
        return h, ffn_pre


class DeepseekV41Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.vocab_size = config.vocab_size
        self.runtime = SharedAttentionRuntime()
        global shared_attn
        shared_attn = self.runtime

        self.engram_layout = EngramLayout.from_args(config)
        self.engram_hash = None  # bound later via bind_engram_hash / ensure_engram
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV41Block(config, idx, self.runtime, self.engram_layout)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # No HyperHead in V4.1 Flash checkpoints (uses ParallelHead/lm_head).
        self.dspark_target_layer_ids = list(config.dspark_target_layer_ids or [])
        # TODO(dspark): wire mtp.* DSpark blocks using dspark_target_layer_ids
        # [37,38,39], dspark_block_size, markov/confidence heads when speculative
        # decode is enabled in omlx. Config fields are parsed and retained.
        self.mtp = []

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        images=None,
        image_mask: Optional[mx.array] = None,
    ) -> mx.array:
        if images is not None:
            raise NotImplementedError(
                "DeepSeek-V4.1 vision is Phase-2: text path only. "
                "Do not pass images until ViT/aligner is implemented."
            )

        h = self.embed_tokens(inputs)
        h = mx.contiguous(
            mx.broadcast_to(
                h[:, :, None, :],
                (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
            )
        )

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList) else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        pre_mix = make_identity_pre_mix(h, self.args.hc_mult)
        self.runtime.reset_forward_slots()

        # Engram n-gram hashes from input_ids; start_pos from KV cache offset.
        start_pos = 0
        if mask_cache is not None and hasattr(mask_cache, 'offset'):
            try:
                start_pos = int(mask_cache.offset)
            except Exception:
                start_pos = 0
        engram_hashes = None
        if self.engram_hash is not None:
            import numpy as _np

            mx.eval(inputs)
            ids_np = _np.array(inputs, dtype=_np.int64)
            if ids_np.ndim == 1:
                ids_np = ids_np[None, :]
            tok_mask = None
            if image_mask is not None:
                mx.eval(image_mask)
                tok_mask = ~_np.array(image_mask, dtype=bool)
            engram_hashes = self.engram_hash.forward(ids_np, start_pos, tok_mask)
            # Overlap later Engram SSD gathers with earlier layer compute.
            # Prefetch every Engram layer up-front (threaded dequant); first layer
            # usually completes during embed/layer0, later ones during intervening FFN/attn.
            if engram_is_mmap() and engram_hashes is not None:
                for layer in self.pipeline_layers:
                    eng = getattr(layer, "engram", None)
                    if eng is None:
                        continue
                    layer_hash = engram_hashes[:, :, eng.layer_hash_index, :]
                    prefetch_engram_layer(self, eng.layer_id, layer_hash)

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            layer_hash = None
            if engram_hashes is not None and getattr(layer, 'engram', None) is not None:
                # Keep CPU ndarray; Engram mmap path gathers without mx roundtrip.
                layer_hash = engram_hashes[:, :, layer.engram.layer_hash_index, :]
            h, pre_mix = layer(
                h,
                mask,
                layer_cache,
                pre_mix,
                image_mask,
                layer_hash,
                _standard_mask=True,
            )

        # Official end: collapse with last FFN pre_mix, then RMSNorm.
        # Checkpoint has no HyperHead weights (ParallelHead is lm_head).
        return self.norm(hc_collapse(h, pre_mix))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV41Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if engram_tables_enabled() and self.model.engram_layout is not None:
            try:
                bind_engram_hash(self, config)
            except FileNotFoundError as e:
                logger.warning('Engram bind deferred (missing meta/token_map): %s', e)

    def ensure_engram(self):
        """Bind NgramHashState + mmap tables if not yet attached."""
        if self.model.engram_layout is None:
            return None
        if self.model.engram_hash is None or engram_is_mmap():
            return bind_engram_hash(self, self.args)
        return self.model.engram_hash

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        images=None,
        skip_lm_head: bool = False,
    ):
        out = self.model(inputs, cache, images=images)
        if skip_lm_head:
            # Chunked prefill discards per-chunk logits; first decode scores
            # the prompt's last token. Skip the full-vocab lm_head GEMM.
            return None
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return not (
                "attn_sink" in k
                or "e_score_correction_bias" in k
                or "bias_vl" in k
                or ".attn_hc." in k
                or ".ffn_hc." in k
                or ".hc_head." in k
            )

        return predicate

    def make_cache(self):
        """Per-layer caches; KV/index source layers own shared PoolingCaches."""
        runtime = self.model.runtime
        # Fresh pools per generation — register_* keeps identity within this
        # make_cache pass only. Stale B/remainder from a prior run breaks batching.
        runtime.kv_pool_by_source.clear()
        runtime.index_pool_by_source.clear()
        runtime.reset_forward_slots()
        args = self.args
        caches = []
        # Build pools in layer order so reuse layers see the same object.
        for layer in self.model.layers:
            ratio = layer.attn.compress_ratio
            lid = layer.attn.layer_id
            window = RotatingKVCache(max_size=args.sliding_window)
            if ratio == 0:
                caches.append(window)
                continue
            if layer.attn.is_kv_source:
                kv_pool = runtime.register_kv_pool(lid, PoolingCache(ratio))
                if layer.attn.is_index_source:
                    idx_pool = runtime.register_index_pool(lid, PoolingCache(ratio))
                    caches.append(CacheList(window, kv_pool, idx_pool))
                else:
                    caches.append(CacheList(window, kv_pool))
            elif layer.attn.is_index_source:
                src = runtime.kv_source_for(lid, args.kv_source_layer_ids)
                kv_pool = (
                    runtime.kv_pool_by_source.get(src) if src is not None else None
                )
                idx_pool = runtime.register_index_pool(lid, PoolingCache(ratio))
                if kv_pool is None:
                    kv_pool = PoolingCache(ratio)
                caches.append(CacheList(window, kv_pool, idx_pool))
            else:
                src = runtime.kv_source_for(lid, args.kv_source_layer_ids)
                kv_pool = (
                    runtime.kv_pool_by_source.get(src) if src is not None else None
                )
                if kv_pool is not None:
                    caches.append(CacheList(window, kv_pool))
                else:
                    # Should not happen when kv_source_layer_ids is well-formed;
                    # keep a window-only cache rather than inventing a private pool.
                    caches.append(window)
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        n_layers = self.args.num_hidden_layers
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                # Keep config-visible but skip loading until DSpark wired.
                continue
            if "engram" in k:
                # stub: skip all engram keys. mmap/ssd: skip only embed tables
                # (wkv/q_weight/k_weight load normally). full: keep everything.
                mode = engram_mode()
                if mode in ("mmap", "ssd"):
                    if "embed.weight" in k or "embed.scale" in k:
                        continue
                elif mode != "full":
                    continue
            if k.startswith("vision.") or k.startswith("aligner.") or k.startswith(
                "image_"
            ):
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    if int(parts[1]) >= n_layers:
                        continue
                except ValueError:
                    pass
            new_weights[k] = v
        weights = new_weights

        # Scale / packed weight handling (same spirit as V4)
        new_weights = {}
        for k, v in weights.items():
            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue
            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".ffn.experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[k + "s"] = v
                new_weights[wk] = weight.view(mx.uint32)
            elif weight.dtype == mx.uint8:
                # Flash UE8M0 scales are one value per 32x32 weight block:
                # shape (out/32, in/32). Expand to (out, in/32) for mlx mxfp8.
                scales = v
                if weight.ndim >= 2 and scales.ndim >= 2:
                    out_rep = weight.shape[0] // scales.shape[0]
                    in_rep = (weight.shape[-1] // 32) // scales.shape[-1]
                    if out_rep > 1:
                        scales = mx.repeat(scales, out_rep, 0)
                    if in_rep > 1:
                        scales = mx.repeat(scales, in_rep, -1)
                new_weights[k + "s"] = scales
                new_weights[wk] = weight.view(mx.uint32)

            else:
                new_weights[k] = v
        weights = new_weights

        top_remap = {
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        for old, new in top_remap.items():
            if old in weights:
                weights[new] = weights.pop(old)

        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for k, v in weights.items():
            nk = "model." + k if k.startswith("layers.") else k
            if nk.endswith(".ffn.gate.bias_vl"):
                pass  # keep MoEGate.bias_vl
            elif nk.endswith(".ffn.gate.bias"):
                nk = nk[: -len("bias")] + "e_score_correction_bias"
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
            skip = False
            for old, new in ((".hc_attn.", ".attn_hc."), (".hc_ffn.", ".ffn_hc.")):
                if old in nk:
                    candidate = nk.replace(old, new)
                    if candidate in weights or candidate in remapped:
                        skip = True
                        break
                    nk = candidate
            if skip:
                continue
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[nk] = v
        weights = remapped

        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.ffn.experts"
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                for suffix in ("weight", "scales", "biases"):
                    key0 = f"{prefix}.0.{src}.{suffix}"
                    if key0 in weights:
                        n_exp = self.args.get_moe_config(layer_idx)[0]
                        stacked = [
                            weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                            for e in range(n_exp)
                        ]
                        weights[
                            f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.{suffix}"
                        ] = mx.stack(stacked)

        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.attn.wo_a"
            for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key].reshape(
                        self.args.o_groups, self.args.o_lora_rank, -1
                    )

        return weights


# Re-export helpers used by tests
__all__ = [
    "Model",
    "ModelArgs",
    "SharedAttentionRuntime",
    "select_candidate_blocks",
    "make_quantization_config",
    "make_identity_pre_mix",
    "hc_collapse",
    "hc_expand",
    "Compressor",
    "Indexer",
    "Attention",
    "DeepseekV41Block",
]
