# SPDX-License-Identifier: MIT
# ruff: noqa
# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .pipeline import PipelineMixin
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    v_head_dim: int
    rope_theta: float
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    swa_head_dim: int
    swa_v_head_dim: int
    swa_rope_theta: float
    sliding_window_size: int
    add_full_attention_sink_bias: bool
    add_swa_attention_sink_bias: bool
    hybrid_layer_pattern: List[int]
    moe_layer_freq: List[int]
    n_routed_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    topk_method: str
    partial_rotary_factor: float
    attention_bias: bool
    layernorm_epsilon: float
    max_position_embeddings: int
    routed_scaling_factor: Optional[float] = None
    attention_value_scale: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        n = self.num_hidden_layers
        if len(self.hybrid_layer_pattern) != n:
            raise ValueError("hybrid_layer_pattern length must match num_hidden_layers")
        if len(self.moe_layer_freq) != n:
            raise ValueError("moe_layer_freq length must match num_hidden_layers")


# ── Fused-QKV FP8 layout ───────────────────────────────────────────────────
#
# MiMo V2.5 FP8 checkpoints ship a fused ``qkv_proj.weight`` that is already
# sharded for tensor parallelism, alongside a block-128 ``weight_scale_inv``
# companion. Each shard is padded to a block boundary *individually*, so the
# padding is interleaved through the tensor rather than trailing it. Treating
# the fused tensor as one block grid straddles shard boundaries and silently
# yields wrong weights, so the scale must be applied per shard before q/k/v
# are sliced apart.
#
# These helpers sit at module level (rather than nested inside ``sanitize``)
# so oMLX's streaming quantizer can reuse this exact geometry and math
# instead of reimplementing it. Second implementations of this layout have
# been the source of silent corruption; keep this the only one.
# See omlx/patches/mimo_v2/virtual_qkv.py for the streaming consumer.

FUSED_QKV_BLOCK_SIZE = 128

# Modalities this text backbone intentionally drops during sanitize. The
# streaming quantizer must skip the same prefixes: their tensors are present
# in the checkpoint but carry geometry that does not match
# ``hybrid_layer_pattern`` (the MTP head reuses the SWA shape while sitting
# at ``layers.0``), so any geometry check applied to them will misfire.
SANITIZE_SKIP_PREFIXES = (
    "model.mtp.",
    "visual.",
    "audio_encoder.",
    "speech_embeddings.",
)


def layer_head_geometry(args, layer_idx):
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


def fused_qkv_shard_rows(n_h, n_kv, hd, vhd, tp):
    """Per-shard ``(actual_rows, padded_rows)`` for one fused-qkv shard."""
    pr = (n_h // tp) * hd + (n_kv // tp) * (hd + vhd)
    return pr, -(-pr // FUSED_QKV_BLOCK_SIZE) * FUSED_QKV_BLOCK_SIZE


def fused_qkv_split_shapes(n_h, n_kv, hd, vhd, tp, n_cols):
    """Output shapes of :func:`split_fused_qkv`, as ``(q, k, v)``."""
    return (
        (tp * (n_h // tp) * hd, n_cols),
        (tp * (n_kv // tp) * hd, n_cols),
        (tp * (n_kv // tp) * vhd, n_cols),
    )


def detect_fused_qkv_tp(args, shape_of):
    """Resolve the tensor-parallel degree the fused qkv was sharded for.

    ``shape_of(key)`` returns that tensor's shape, or None when it is absent,
    so both the eager sanitize (a dict of arrays) and the streaming
    quantizer (a safetensors header index) can drive this without
    materializing any data.

    Only full-attention layers are consulted: SWA geometry happens to be
    block-aligned, which makes its TP degree ambiguous.
    """
    for layer_idx in range(args.num_hidden_layers):
        if bool(args.hybrid_layer_pattern[layer_idx]):
            continue
        qkv_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
        qkv_shape = shape_of(qkv_key)
        scale_shape = shape_of(f"{qkv_key}_scale_inv")
        if qkv_shape is None or scale_shape is None:
            continue
        actual = qkv_shape[0]
        padded = scale_shape[0] * FUSED_QKV_BLOCK_SIZE
        n_h, n_kv, hd, vhd = layer_head_geometry(args, layer_idx)
        for tp in (1, 2, 4, 8, 16, 32):
            if n_h % tp or n_kv % tp:
                continue
            pr, pr_padded = fused_qkv_shard_rows(n_h, n_kv, hd, vhd, tp)
            if pr * tp == actual and pr_padded * tp == padded:
                return tp
        raise ValueError(
            f"unable to determine fused-qkv TP layout from layer {layer_idx} "
            f"(actual={actual}, padded={padded})"
        )
    # No fused qkv with a scale companion anywhere: nothing to shard-aware
    # dequantize, so the split path is inert and TP is irrelevant.
    return 1


def split_fused_qkv(qkv_fp8, scale_inv, *, tp, n_h, n_kv, hd, vhd):
    """Dequantize a pre-sharded fused qkv FP8 tensor and split it into q/k/v.

    The block scale is applied per tensor-parallel shard, because each shard
    was padded to a block boundary on its own. Returns ``(q, k, v)`` in
    bfloat16.
    """
    BS = FUSED_QKV_BLOCK_SIZE
    bf16 = mx.bfloat16
    q_pr = (n_h // tp) * hd
    k_pr = (n_kv // tp) * hd
    v_pr = (n_kv // tp) * vhd
    actual_pr = q_pr + k_pr + v_pr
    padded_pr = -(-actual_pr // BS) * BS
    n = qkv_fp8.shape[-1]

    qkv = mx.from_fp8(qkv_fp8, dtype=bf16).reshape(tp, actual_pr, n)
    if padded_pr > actual_pr:
        qkv = mx.pad(qkv, ((0, 0), (0, padded_pr - actual_pr), (0, 0)))
    n_col_blocks = scale_inv.shape[1]
    pad_side = BS * n_col_blocks - n
    if pad_side > 0:
        qkv = mx.pad(qkv, ((0, 0), (0, 0), (0, pad_side)))

    blocked = qkv.reshape(tp * padded_pr // BS, BS, n_col_blocks, BS)
    qkv = (blocked * scale_inv[:, None, :, None]).reshape(
        tp, padded_pr, n_col_blocks * BS
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


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_sliding_window: bool):
        super().__init__()
        dim = args.hidden_size
        self.is_sliding_window = is_sliding_window
        if is_sliding_window:
            self.n_heads = args.swa_num_attention_heads
            self.n_kv_heads = args.swa_num_key_value_heads
            head_dim = args.swa_head_dim
            v_head_dim = args.swa_v_head_dim
            rope_theta = args.swa_rope_theta
            has_sinks = args.add_swa_attention_sink_bias
        else:
            self.n_heads = args.num_attention_heads
            self.n_kv_heads = args.num_key_value_heads
            head_dim = args.head_dim
            v_head_dim = args.v_head_dim
            rope_theta = args.rope_theta
            has_sinks = args.add_full_attention_sink_bias

        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.scale = head_dim**-0.5
        self.v_scale = args.attention_value_scale

        self.q_proj = nn.Linear(dim, self.n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * v_head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(self.n_heads * v_head_dim, dim, bias=False)

        self.attention_sink_bias = mx.zeros((self.n_heads,)) if has_sinks else None

        self.rope = initialize_rope(
            int(args.partial_rotary_factor * head_dim),
            base=rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = (
            self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).swapaxes(1, 2)
        )
        keys = (
            self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).swapaxes(1, 2)
        )
        values = (
            self.v_proj(x)
            .reshape(B, L, self.n_kv_heads, self.v_head_dim)
            .swapaxes(1, 2)
        )

        if self.v_scale is not None:
            values = values * self.v_scale

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.attention_sink_bias,
        )
        return self.o_proj(output.swapaxes(1, 2).reshape(B, L, -1))


class MLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        h, i = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


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

    inds = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
    scores = scores * routed_scaling_factor
    return inds, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.topk_method == "noaux_tc", "Unsupported topk method."
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor or 1.0
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))

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


class MoE(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
        self.gate = MoEGate(config)
        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, is_moe: bool, is_sliding_window: bool):
        super().__init__()
        self.self_attn = Attention(config, is_sliding_window)
        self.mlp = MoE(config) if is_moe else MLP(config)
        self.is_sliding_window = is_sliding_window
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class MiMoV2Model(PipelineMixin, nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        pattern = config.hybrid_layer_pattern
        moe_freq = config.moe_layer_freq
        self.layers = [
            DecoderLayer(
                config,
                is_moe=bool(moe_freq[idx]),
                is_sliding_window=bool(pattern[idx]),
            )
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.sliding_window_size = config.sliding_window_size

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )

        local_layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(local_layers)

        swa_local = next(
            (i for i, l in enumerate(local_layers) if l.is_sliding_window), None
        )
        ga_local = next(
            (i for i, l in enumerate(local_layers) if not l.is_sliding_window), None
        )
        full_mask = (
            create_attention_mask(h, cache[ga_local]) if ga_local is not None else None
        )
        swa_mask = (
            create_attention_mask(
                h, cache[swa_local], window_size=self.sliding_window_size
            )
            if swa_local is not None
            else None
        )

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, pipeline_rank + 1)

        for layer, c in zip(local_layers, cache):
            mask = swa_mask if layer.is_sliding_window else full_mask
            h = layer(h, mask, cache=c)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = MiMoV2Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith(SANITIZE_SKIP_PREFIXES)
        }

        BS = FUSED_QKV_BLOCK_SIZE
        bf16 = mx.bfloat16

        def shape_of(key):
            tensor = weights.get(key)
            return None if tensor is None else tensor.shape

        TP = detect_fused_qkv_tp(self.args, shape_of)

        def dequant_block(weight, scale_inv):
            weight = mx.from_fp8(weight, dtype=bf16)
            m, n = weight.shape
            pad_b, pad_r = (-m) % BS, (-n) % BS
            if pad_b or pad_r:
                weight = mx.pad(weight, ((0, pad_b), (0, pad_r)))
            weight = weight.reshape((m + pad_b) // BS, BS, (n + pad_r) // BS, BS)
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_b, n + pad_r
            )
            return weight[:m, :n].astype(bf16)

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.self_attn"
            qkv_key = f"{prefix}.qkv_proj.weight"
            scale_key = f"{qkv_key}_scale_inv"
            if qkv_key not in weights or scale_key not in weights:
                continue

            n_h, n_kv, hd, vhd = layer_head_geometry(self.args, layer_idx)

            q, k, v = split_fused_qkv(
                weights.pop(qkv_key),
                weights.pop(scale_key),
                tp=TP,
                n_h=n_h,
                n_kv=n_kv,
                hd=hd,
                vhd=vhd,
            )
            weights[f"{prefix}.q_proj.weight"] = q
            weights[f"{prefix}.k_proj.weight"] = k
            weights[f"{prefix}.v_proj.weight"] = v

        scale_keys = [k for k in weights if k.endswith("weight_scale_inv")]
        for sk in scale_keys:
            wk = sk[: -len("_scale_inv")]
            weights[wk] = dequant_block(weights[wk], weights.pop(sk))

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            for proj in ("gate_proj", "down_proj", "up_proj"):
                expert0 = f"{prefix}.experts.0.{proj}.weight"
                if expert0 not in weights:
                    continue
                weights[f"{prefix}.switch_mlp.{proj}.weight"] = mx.stack(
                    [
                        weights.pop(f"{prefix}.experts.{e}.{proj}.weight")
                        for e in range(self.args.n_routed_experts)
                    ]
                )

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        R = group.rank()
        for layer in self.model.layers:
            if layer is None:
                continue

            attn = layer.self_attn
            attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=group)
            attn.k_proj = shard_linear(attn.k_proj, "all-to-sharded", group=group)
            attn.v_proj = shard_linear(attn.v_proj, "all-to-sharded", group=group)
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.n_heads //= N
            attn.n_kv_heads //= N

            if attn.attention_sink_bias is not None:
                attn.attention_sink_bias = attn.attention_sink_bias[
                    R * attn.n_heads : (R + 1) * attn.n_heads
                ]

            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_sliding_window:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window_size))
            else:
                caches.append(KVCache())
        return caches
