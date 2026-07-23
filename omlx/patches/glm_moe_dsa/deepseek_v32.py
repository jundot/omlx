# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import CacheList, KVCache
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.rope_utils import initialize_rope
from .kernels import fast as glm_fast
from .sparse_mla import (
    fused_indexer_scores,
    fused_index_score_reduce,
)
from .switch_layers import SwitchGLU


def _use_load_fused_wk_weights_proj(args) -> bool:
    if getattr(args, "model_type", None) != "glm_moe_dsa":
        return False
    quantization = getattr(args, "quantization", None)
    if not isinstance(quantization, dict):
        return False

    def resolved_spec(name):
        spec = quantization.get(name, quantization)
        return spec if isinstance(spec, dict) else {}

    def is_indexer_q8(name):
        spec = resolved_spec(name)
        return (
            spec.get("bits") == 8
            and spec.get("group_size") == 64
            and spec.get("mode", "affine") == "affine"
        )

    prefixes = []
    indexer_types = getattr(args, "indexer_types", None) or []
    num_hidden_layers = int(getattr(args, "num_hidden_layers", 0) or 0)
    for layer_idx, indexer_type in enumerate(indexer_types[:num_hidden_layers]):
        if indexer_type == "full":
            prefixes.append(f"model.layers.{layer_idx}.self_attn.indexer")

    # Preserved MTP checkpoints use runtime mtp.* paths. Include explicit MTP
    # specs when present, but do not treat their absence as the global default:
    # third-party dynamic checkpoints can omit nextn overrides and recover them
    # from tensor shapes later in sanitize (issue #2326).
    for key in quantization:
        if key.startswith("mtp.") and ".block.self_attn.indexer." in key:
            prefix, projection = key.rsplit(".", 1)
            if projection in ("wq_b", "wk", "weights_proj"):
                prefixes.append(prefix)

    prefixes = sorted(set(prefixes))
    if not prefixes:
        return False

    paths = [
        f"{prefix}.{projection}"
        for prefix in prefixes
        for projection in ("wq_b", "wk", "weights_proj")
    ]
    q8_paths = [path for path in paths if is_indexer_q8(path)]
    if not q8_paths:
        return False
    if len(q8_paths) != len(paths):
        offenders = [
            (
                path,
                resolved_spec(path).get("bits"),
                resolved_spec(path).get("group_size"),
                resolved_spec(path).get("mode", "affine"),
            )
            for path in paths
            if not is_indexer_q8(path)
        ]
        detail = ", ".join(
            f"{path}={bits}-bit/gs{group_size}/{mode}"
            for path, bits, group_size, mode in offenders[:4]
        )
        raise ValueError(
            "Invalid GLM DSA indexer quantization: Q8 fusion requires every "
            "full-indexer projection to use 8-bit affine quantization with "
            f"group_size=64; found {detail}"
        )
    return True


def _dequant_mla_proj_mode(args) -> str:
    return "0"


def _use_glm_moe_fused_gate_up(args) -> bool:
    return getattr(args, "model_type", None) == "glm_moe_dsa"


def _use_glm_shared_fused_gate_up(args) -> bool:
    return False


def _use_glm_moe_weighted_sum(args) -> bool:
    return getattr(args, "model_type", None) == "glm_moe_dsa"


def _use_glm_moe_inverse_scatter(args) -> bool:
    return getattr(args, "model_type", None) == "glm_moe_dsa"


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v32"
    vocab_size: int = 102400
    hidden_size: int = 4096
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 2048
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1407
    num_hidden_layers: int = 30
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    n_shared_experts: Optional[int] = None
    n_routed_experts: Optional[int] = None
    routed_scaling_factor: float = 1.0
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 1
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Dict = None
    attention_bias: bool = False
    indexer_rope_interleave: bool = False


# Decode-shaped multi-row forwards (MTP draft verify, L = depth clamp 8 + 1
# rows at most) must not enter the prefill-shaped fused scores kernel: its
# grid is sized for many query rows, runs ~5x slower than the plain matmul +
# fused reduce at tiny L, and the gap grows with context length (~3 ms per
# indexer layer at 128k). Below this floor the s > 1 scoring falls through to
# the q @ k.T + fused_index_score_reduce path, whose causal fill uses the
# same decode offsets. 16 mirrors the sdpa256/fa256 prefill-routing floors.
_FUSED_SCORES_MIN_S = 16


class Indexer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.load_fused_wk_weights_proj = _use_load_fused_wk_weights_proj(args)
        if self.load_fused_wk_weights_proj:
            self.wk = None
            self.weights_proj = None
            self.wk_weights_proj = nn.QuantizedLinear(
                self.dim,
                self.head_dim + self.n_heads,
                bias=False,
                group_size=64,
                bits=8,
                mode="affine",
            )
        else:
            self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
            self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
            self.wk_weights_proj = None
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.softmax_scale = self.head_dim**-0.5
        self.weight_scale = self.n_heads**-0.5 * self.softmax_scale
        self._wk_weights_proj_cache = None
        self.default_block_sparse_sdpa = args.model_type == "glm_moe_dsa"
        self.rope = initialize_rope(
            dims=args.qk_rope_head_dim,
            base=args.rope_theta,
            traditional=args.indexer_rope_interleave,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

    def _fused_wk_weights_proj(self, x: mx.array):
        if self.wk_weights_proj is not None:
            out = self.wk_weights_proj(x)
            return mx.split(out, [self.head_dim], axis=-1)

        return None

    def __call__(
        self,
        x: mx.array,
        qr: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any] = None,
    ):
        # Computes top_k indices for attention
        b, s, _ = x.shape
        q = self.wq_b(qr)
        q = q.reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
        fused_kw = self._fused_wk_weights_proj(x)
        if fused_kw is None:
            k = self.wk(x)
            weights_lh = self.weights_proj(x)
        else:
            k, weights_lh = fused_kw
        k = self.k_norm(k)
        k = mx.reshape(k, (b, 1, s, self.head_dim))

        offset = cache.offset if cache is not None else 0

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros([b, 1, s, 0]))
        if k.shape[2] <= self.index_topk:
            return None
        use_fast_topk = glm_fast.has("dsa_topk_indices")
        sort_exact_topk = True
        bucketed_topk = use_fast_topk and s > 1
        causal_valid_prefix_topk = False
        prefix_topk_rows = 0

        def causal_prefix_topk(prefix_rows: int):
            offset = k.shape[2] - s
            slots = mx.arange(self.index_topk, dtype=mx.uint32).reshape(
                1, 1, 1, self.index_topk
            )
            lengths = mx.arange(prefix_rows, dtype=mx.uint32).reshape(
                1, 1, prefix_rows, 1
            ) + mx.array(offset + 1, dtype=mx.uint32)
            prefix = mx.where(slots < lengths, slots, mx.array(0, dtype=mx.uint32))
            return mx.broadcast_to(prefix, (b, 1, prefix_rows, self.index_topk))

        def finish_topk_indices(indices, prefix_rows: int = 0):
            if indices is not None:
                indices = (
                    mx.sort(indices, axis=-1)
                    if sort_exact_topk and not bucketed_topk
                    else indices
                )
            if prefix_rows > 0:
                if (
                    indices is not None
                    and self.default_block_sparse_sdpa
                ):
                    return indices, None, prefix_rows
                prefix = causal_prefix_topk(prefix_rows)
                return (
                    prefix
                    if indices is None
                    else mx.concatenate([prefix, indices], axis=2)
                )
            return indices

        def select_topk(
            scores, prefix_rows: int = 0, causal_valid_prefix: Optional[bool] = None
        ):
            indices = None
            use_native_topk = (
                use_fast_topk
                and self.index_topk == 2048
                and scores is not None
                and scores.shape[-1] >= 2048
            )
            if use_native_topk:
                indices = glm_fast.dsa_topk_indices(
                    scores,
                    self.index_topk,
                    bucketed=bucketed_topk,
                    causal_valid_prefix=(
                        causal_valid_prefix_topk
                        if causal_valid_prefix is None
                        else causal_valid_prefix
                    ),
                )
            elif scores is not None:
                indices = mx.argpartition(scores, kth=-self.index_topk, axis=-1)[
                    ..., -self.index_topk :
                ]
            return finish_topk_indices(indices, prefix_rows)

        if s == 1:
            scores = q @ k.swapaxes(-1, -2)
            scores = mx.maximum(scores, 0)
            weights = weights_lh * self.weight_scale
            weights = weights.swapaxes(-1, -2)[..., None]
            scores = scores * weights
            scores = scores.sum(axis=1, keepdims=True)
            if mask is not None:
                scores = mx.where(mask, scores, -float("inf"))
            return select_topk(scores)
        weights_lh = weights_lh * self.weight_scale
        fuse_causal_mask = mask is not None
        causal_valid_prefix_topk = fuse_causal_mask

        scores = None
        if s >= _FUSED_SCORES_MIN_S and (mask is None or fuse_causal_mask):
            scores_feed_exact_topk = causal_valid_prefix_topk and use_fast_topk
            candidate_prefix_rows = 0
            if use_fast_topk and scores_feed_exact_topk:
                candidate_prefix_rows = min(
                    s, max(0, self.index_topk - (k.shape[2] - s))
                )
            if candidate_prefix_rows == s:
                return select_topk(None, candidate_prefix_rows)
            score_q = q[:, :, candidate_prefix_rows:, :] if candidate_prefix_rows else q
            score_weights = (
                weights_lh[:, candidate_prefix_rows:, :]
                if candidate_prefix_rows
                else weights_lh
            )
            default_chunk_mb = (
                "128" if self.default_block_sparse_sdpa and k.shape[2] <= 65536 else "0"
            )
            chunk_mb = int(default_chunk_mb)
            chunked_topk = None
            if (
                chunk_mb > 0
                and use_fast_topk
                and score_q.shape[2] > 64
            ):
                bytes_per_score = 2
                max_scores = max(1, (chunk_mb * 1024 * 1024) // bytes_per_score)
                chunk_rows = max(64, (max_scores // k.shape[2]) // 64 * 64)
                if chunk_rows < score_q.shape[2]:
                    chunk_indices = []
                    full_q_offset = k.shape[2] - s
                    for row_start in range(0, score_q.shape[2], chunk_rows):
                        row_end = min(score_q.shape[2], row_start + chunk_rows)
                        chunk_scores = fused_indexer_scores(
                            score_q[:, :, row_start:row_end, :],
                            k,
                            score_weights[:, row_start:row_end, :],
                            causal=fuse_causal_mask,
                            causal_q_offset=(
                                full_q_offset + candidate_prefix_rows + row_start
                                if fuse_causal_mask
                                else -1
                            ),
                        )
                        if chunk_scores is None:
                            chunk_indices = []
                            break
                        chunk_indices.append(
                            select_topk(
                                chunk_scores,
                                causal_valid_prefix=False,
                            )
                        )
                    if chunk_indices:
                        chunked_topk = mx.concatenate(chunk_indices, axis=2)
                        return finish_topk_indices(chunked_topk, candidate_prefix_rows)
            if chunked_topk is None:
                scores = fused_indexer_scores(
                    score_q,
                    k,
                    score_weights,
                    causal=fuse_causal_mask,
                    skip_causal_future_store=scores_feed_exact_topk,
                )
            if scores is not None:
                prefix_topk_rows = candidate_prefix_rows

        weights = weights_lh.swapaxes(-1, -2)[..., None]
        head_scores = None
        if scores is None and s > 1:
            head_scores = q @ k.swapaxes(-1, -2)
            scores = fused_index_score_reduce(
                head_scores,
                weights,
                causal=fuse_causal_mask,
            )
        if scores is None:
            if head_scores is None:
                head_scores = q @ k.swapaxes(-1, -2)
            scores = mx.maximum(head_scores, 0)
            scores = scores * weights
            scores = scores.sum(axis=1, keepdims=True)
            fuse_causal_mask = False
        if mask is not None and not fuse_causal_mask:
            scores = mx.where(mask, scores, -float("inf"))
        return select_topk(scores, prefix_topk_rows)


class DeepseekV32Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(
            self.hidden_size, self.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = self.config.rope_scaling["factor"]
                if scaling_factor > 1:
                    s = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scale = self.scale * s * s

        self.indexer = Indexer(config)
        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=self.rope_theta,
            traditional=True,
            max_position_embeddings=self.max_position_embeddings,
            scaling_config=self.config.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        if topk_indices is not None:
            if L == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask
        # Ensure the indexer cache is evaluated even if the topk_indices are unused
        # to keep the graph from getting too large
        if cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if L == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class DeepseekV32MLP(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        hidden_size: int = None,
        intermediate_size: int = None,
        fused_gate_up: bool = False,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        if fused_gate_up:
            self.gate_up_proj = nn.Linear(
                self.hidden_size, self.intermediate_size * 2, bias=False
            )
        else:
            self.gate_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=False
            )
            self.up_proj = nn.Linear(
                self.hidden_size, self.intermediate_size, bias=False
            )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def __call__(self, x):
        if hasattr(self, "gate_up_proj"):
            gate, up = mx.split(self.gate_up_proj(x), [self.intermediate_size], axis=-1)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        down_proj = self.down_proj(swiglu(gate, up))
        return down_proj


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):

    scores = mx.sigmoid(gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True)
        scores = scores / denominator
    scores = scores * routed_scaling_factor

    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))
        assert config.topk_method == "noaux_tc", "Unsupported topk method."

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class DeepseekV32MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            fused_gate_up=_use_glm_moe_fused_gate_up(config),
            inverse_scatter=_use_glm_moe_inverse_scatter(config),
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV32MLP(
                config=config,
                intermediate_size=intermediate_size,
                fused_gate_up=_use_glm_shared_fused_gate_up(config),
            )

        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        inds, scores = self.gate(x)
        use_weighted_sum = (
            _use_glm_moe_weighted_sum(self.config)
            and inds.size >= 64
            and hasattr(glm_fast, "glm_moe_weighted_sum")
        )
        y = self.switch_mlp(
            x,
            inds,
            scores=scores if use_weighted_sum else None,
            weighted_sum=use_weighted_sum,
        )
        if not use_weighted_sum:
            y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        return y


class DeepseekV32DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = DeepseekV32Attention(config)
        self.mlp = (
            DeepseekV32MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV32MLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class DeepseekV32Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV32DecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.start_idx = 0
        self.end_idx = len(self.layers)
        self.num_layers = self.end_idx

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pipeline_rank = 0
        self.pipeline_size = 1

    def pipeline(self, group):
        # Split layers in reverse so rank=0 gets the last layers and
        # rank=pipeline_size-1 gets the first
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        layers_per_rank = len(self.layers) // self.pipeline_size
        extra = len(self.layers) - layers_per_rank * self.pipeline_size
        if self.pipeline_rank < extra:
            layers_per_rank += 1
        self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
        self.end_idx = self.start_idx + layers_per_rank
        self.layers = self.layers[: self.end_idx]
        self.layers[: self.start_idx] = [None] * self.start_idx
        self.num_layers = len(self.layers) - self.start_idx

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        # Receive from the previous process in the pipeline

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for i in range(self.num_layers):
            h = self.layers[self.start_idx + i](h, mask, cache[i])

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = DeepseekV32Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        # Remove multi-token prediction layers
        mpt_layer = self.args.num_hidden_layers
        new_weights = {}
        for k, v in weights.items():
            parts = k.split(".")
            if len(parts) >= 3 and parts[1] == "layers" and int(parts[2]) >= mpt_layer:
                continue
            new_weights[k] = v
        weights = new_weights

        def dequant(weight, scale_inv):
            dtype = mx.bfloat16
            weight = mx.from_fp8(weight, dtype=mx.bfloat16)
            bs = 128  # block size
            m, n = weight.shape
            pad_bottom = (-m) % bs
            pad_side = (-n) % bs
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(dtype)

        # Dequantize
        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                weight = dequant(weight, scale_inv)
                new_weights[wk] = weight
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        if _use_load_fused_wk_weights_proj(self.args):
            fused = {}
            skip_keys = set()
            for l in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{l}.self_attn.indexer"
                wk_prefix = f"{prefix}.wk"
                wp_prefix = f"{prefix}.weights_proj"
                fused_prefix = f"{prefix}.wk_weights_proj"
                if (
                    f"{wk_prefix}.weight" not in weights
                    or f"{wp_prefix}.weight" not in weights
                ):
                    continue
                for suffix in ("weight", "scales", "biases"):
                    wk_key = f"{wk_prefix}.{suffix}"
                    wp_key = f"{wp_prefix}.{suffix}"
                    if wk_key in weights and wp_key in weights:
                        fused[f"{fused_prefix}.{suffix}"] = mx.concatenate(
                            [weights[wk_key], weights[wp_key]], axis=0
                        )
                        skip_keys.add(wk_key)
                        skip_keys.add(wp_key)
                skip_keys.add(f"{wk_prefix}.bias")
                skip_keys.add(f"{wp_prefix}.bias")
            if fused:
                weights = {
                    k: v
                    for k, v in weights.items()
                    if k not in skip_keys
                    and ".self_attn.indexer.wk." not in k
                    and ".self_attn.indexer.weights_proj." not in k
                }
                weights.update(fused)

        dequant_mla_proj = _dequant_mla_proj_mode(self.args)
        dequant_embed_q = dequant_mla_proj in {
            "1",
            "true",
            "all",
            "both",
            "embed",
            "embed_q",
        }
        dequant_unembed_out = dequant_mla_proj in {
            "1",
            "true",
            "all",
            "both",
            "unembed",
            "unembed_out",
            "out",
        }
        if (dequant_embed_q or dequant_unembed_out) and isinstance(
            getattr(self.args, "quantization", None), dict
        ):

            def dequant_mla_weight(path, input_dims):
                weight_key = f"{path}.weight"
                scales_key = f"{path}.scales"
                biases_key = f"{path}.biases"
                if weight_key not in weights:
                    weights.pop(scales_key, None)
                    weights.pop(biases_key, None)
                    return
                scales = weights.pop(scales_key, None)
                biases = weights.pop(biases_key, None)
                if scales is None:
                    return
                spec = self.args.quantization.get(path, {})
                bits = spec.get("bits")
                if bits is None:
                    bits = (weights[weight_key].shape[-1] * 32) // input_dims
                group_size = spec.get("group_size")
                if group_size is None:
                    group_size = input_dims // scales.shape[-1]
                mode = spec.get("mode", "affine")
                weights[weight_key] = mx.dequantize(
                    weights[weight_key],
                    scales,
                    biases,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                    dtype=mx.bfloat16,
                )

            for l in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{l}.self_attn"
                if dequant_embed_q:
                    dequant_mla_weight(f"{prefix}.embed_q", self.args.qk_nope_head_dim)
                    self.args.quantization.pop(f"{prefix}.embed_q", None)
                if dequant_unembed_out:
                    dequant_mla_weight(f"{prefix}.unembed_out", self.args.kv_lora_rank)
                    self.args.quantization.pop(f"{prefix}.unembed_out", None)

        # Stack experts
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
            if _use_glm_moe_fused_gate_up(self.args):
                switch_prefix = f"{prefix}.mlp.switch_mlp"
                for k in ["weight", "scales", "biases"]:
                    gate_key = f"{switch_prefix}.gate_proj.{k}"
                    up_key = f"{switch_prefix}.up_proj.{k}"
                    if gate_key in weights and up_key in weights:
                        weights[f"{switch_prefix}.gate_up_proj.{k}"] = mx.concatenate(
                            [weights.pop(gate_key), weights.pop(up_key)],
                            axis=1,
                        )
            if _use_glm_shared_fused_gate_up(self.args):
                shared_prefix = f"{prefix}.mlp.shared_experts"
                for k in ["weight", "scales", "biases"]:
                    gate_key = f"{shared_prefix}.gate_proj.{k}"
                    up_key = f"{shared_prefix}.up_proj.{k}"
                    if gate_key in weights and up_key in weights:
                        weights[f"{shared_prefix}.gate_up_proj.{k}"] = mx.concatenate(
                            [weights.pop(gate_key), weights.pop(up_key)],
                            axis=0,
                        )
            prefix = f"model.layers.{l}.self_attn"
            if f"{prefix}.kv_b_proj.weight" in weights:
                quantized = f"{prefix}.kv_b_proj.scales" in weights
                v = weights.pop(f"{prefix}.kv_b_proj.weight")
                head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim

                if quantized:
                    dims = self.args.kv_lora_rank
                    scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                    biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                    # Try to infer bits and group size
                    bits = (v.shape[-1] * 32) // dims
                    group_size = dims // scales.shape[-1]
                    v = mx.dequantize(
                        v, scales, biases, bits=bits, group_size=group_size
                    )
                num_heads = self.args.num_attention_heads
                v = v.reshape(num_heads, head_dim, -1)
                wk = mx.contiguous(
                    v[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                wv = mx.contiguous(v[:, self.args.qk_nope_head_dim :, :])
                if quantized:
                    if not dequant_embed_q:
                        wk, wk_scales, wk_biases = mx.quantize(
                            wk, bits=bits, group_size=group_size
                        )
                        weights[f"{prefix}.embed_q.scales"] = wk_scales
                        weights[f"{prefix}.embed_q.biases"] = wk_biases
                    if not dequant_unembed_out:
                        wv, wv_scales, wv_biases = mx.quantize(
                            wv, bits=bits, group_size=group_size
                        )
                        weights[f"{prefix}.unembed_out.scales"] = wv_scales
                        weights[f"{prefix}.unembed_out.biases"] = wv_biases
                weights[f"{prefix}.embed_q.weight"] = wk
                weights[f"{prefix}.unembed_out.weight"] = wv

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()
        for layer in self.model.layers:
            layer.self_attn.q_b_proj = shard_linear(
                layer.self_attn.q_b_proj, "all-to-sharded", group=group
            )

            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.num_heads //= N
            num_heads = layer.self_attn.num_heads
            sh = rank * num_heads
            eh = sh + num_heads

            def shard_heads(w):
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            # Shard the MLP
            if isinstance(layer.mlp, DeepseekV32MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # Shard the MoE. Shard in place since the MoE should be responsible
            # for aggregating the results.
            else:
                layer.mlp.sharding_group = group = group
                if hasattr(layer.mlp.shared_experts, "gate_up_proj"):
                    shard_inplace(
                        layer.mlp.shared_experts.gate_up_proj,
                        "all-to-sharded",
                        group=group,
                    )
                else:
                    shard_inplace(
                        layer.mlp.shared_experts.gate_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.shared_experts.up_proj,
                        "all-to-sharded",
                        group=group,
                    )
                shard_inplace(
                    layer.mlp.shared_experts.down_proj, "sharded-to-all", group=group
                )
                if hasattr(layer.mlp.switch_mlp, "gate_up_proj"):
                    shard_inplace(
                        layer.mlp.switch_mlp.gate_up_proj,
                        "all-to-sharded",
                        group=group,
                    )
                else:
                    shard_inplace(
                        layer.mlp.switch_mlp.gate_proj,
                        "all-to-sharded",
                        group=group,
                    )
                    shard_inplace(
                        layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                    )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )

    @property
    def layers(self):
        return self.model.layers[self.model.start_idx : self.model.end_idx]

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def make_cache(self):
        return [CacheList(KVCache(), KVCache()) for _ in self.layers]
