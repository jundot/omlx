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
from omlx.patches.mimo_v2.fused_qkv_layout import (
    FUSED_QKV_BLOCK_SIZE,
    detect_fused_qkv_tp,
    fused_qkv_keys,
    layer_head_geometry,
    split_fused_qkv,
)


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
    layernorm_epsilon: float
    max_position_embeddings: int
    routed_scaling_factor: Optional[float] = None
    attention_value_scale: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 0
    omlx_mtp_sidecar: Optional[str] = None
    n_shared_experts: Optional[int] = None
    scoring_func: str = "sigmoid"

    def __post_init__(self):
        n = self.num_hidden_layers
        if len(self.hybrid_layer_pattern) != n:
            raise ValueError("hybrid_layer_pattern length must match num_hidden_layers")
        if len(self.moe_layer_freq) != n:
            raise ValueError("moe_layer_freq length must match num_hidden_layers")


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
        # BF16 router logits can collapse distinct expert scores into ties.
        return group_expert_select(
            x.astype(mx.float32) @ self.weight.astype(mx.float32).T,
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


class MiMoV2MTPLayer(nn.Module):
    """One MiMo next-token predictor head."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden = config.hidden_size
        self.enorm = nn.RMSNorm(hidden, eps=config.layernorm_epsilon)
        self.hnorm = nn.RMSNorm(hidden, eps=config.layernorm_epsilon)
        self.eh_proj = nn.Linear(2 * hidden, hidden, bias=False)
        self.input_layernorm = nn.RMSNorm(hidden, eps=config.layernorm_epsilon)
        self.self_attn = Attention(config, is_sliding_window=True)
        self.pre_mlp_layernorm = nn.RMSNorm(hidden, eps=config.layernorm_epsilon)
        self.mlp = MLP(config)
        self.final_layernorm = nn.RMSNorm(hidden, eps=config.layernorm_epsilon)
        self.sliding_window_size = config.sliding_window_size

    def __call__(
        self,
        hidden_states: mx.array,
        token_embeddings: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        x = self.eh_proj(
            mx.concatenate(
                [self.enorm(token_embeddings), self.hnorm(hidden_states)], axis=-1
            )
        )
        mask = create_attention_mask(
            x,
            cache,
            window_size=self.sliding_window_size,
        )
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        h = h + self.mlp(self.pre_mlp_layernorm(h))
        return self.final_layernorm(h)


class MiMoV2MultiTokenPredictor(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.layers = [
            MiMoV2MTPLayer(config)
            for _ in range(int(config.num_nextn_predict_layers or 0))
        ]

    def __call__(self, hidden, tokens, embed, cache):
        outputs = []
        for layer, layer_cache in zip(self.layers, cache):
            if tokens.shape[1] == 0:
                break
            hidden = layer(hidden, embed(tokens), layer_cache)
            outputs.append(hidden)
            hidden, tokens = hidden[:, :-1], tokens[:, 1:]
        return outputs


class _MiMoMTPCache(list):
    """Per-head caches plus the current predictor index for one draft cycle."""

    def __init__(self, values=()):
        super().__init__(values)
        self.layer_idx = 0


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
        return_hidden: bool = False,
    ) -> Any:
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

        normed = self.norm(h)
        if return_hidden:
            return normed, h
        return normed


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = MiMoV2Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        from omlx.patches.mlx_lm_mtp import get_mtp_depth, is_mtp_active

        n_mtp = int(config.num_nextn_predict_layers or 0)
        self._omlx_mtp_decode_enabled = bool(n_mtp and is_mtp_active())
        if self._omlx_mtp_decode_enabled:
            self.model.mtp = MiMoV2MultiTokenPredictor(config)
            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = min(int(get_mtp_depth()), n_mtp)
            self._omlx_mtp_head_clone = True
            self._omlx_mtp_head_prenorm = True

    @property
    def mtp(self):
        return self.model.mtp

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        del n_confirmed
        result = self.model(
            inputs,
            cache,
            input_embeddings,
            return_hidden=return_hidden,
        )
        if return_hidden:
            out, hidden = result
        else:
            out = result
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)
        if return_hidden:
            return logits, hidden
        return logits

    def mtp_begin_cycle(self, mtp_cache, depth):
        del depth
        if isinstance(mtp_cache, _MiMoMTPCache):
            mtp_cache.layer_idx = 0

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        layer_idx = getattr(mtp_cache, "layer_idx", 0) % len(self.mtp.layers)
        cache = mtp_cache[layer_idx] if mtp_cache else None
        token_embeddings = self.model.embed_tokens(next_token_ids)
        hidden = self.mtp.layers[layer_idx](hidden_states, token_embeddings, cache)
        if isinstance(mtp_cache, _MiMoMTPCache):
            mtp_cache.layer_idx = layer_idx + 1
        logits_source = hidden[:, -logits_keep:] if logits_keep else hidden
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(logits_source)
        else:
            logits = self.lm_head(logits_source)
        if return_hidden:
            return logits, hidden
        return logits

    def make_mtp_cache(self):
        if not hasattr(self.model, "mtp"):
            return _MiMoMTPCache()
        return _MiMoMTPCache(
            RotatingKVCache(max_size=self.args.sliding_window_size)
            for _ in self.mtp.layers
        )

    def mtp_partial_rollback(self, cache, accepted, num_drafts):
        rejected = num_drafts - accepted
        if rejected <= 0:
            return True
        if not all(c.is_trimmable() for c in cache):
            return False
        for c in cache:
            if c.trim(rejected) != rejected:
                raise RuntimeError("MiMo MTP cache rollback was incomplete")
        return True

    def sanitize(self, weights):
        if hasattr(self.model, "mtp") and self.args.omlx_mtp_sidecar:
            weights = {**weights, **mx.load(self.args.omlx_mtp_sidecar)}

        skip_prefixes = (
            "visual.",
            "audio_encoder.",
            "speech_embeddings.",
        )
        if not hasattr(self.model, "mtp"):
            skip_prefixes += ("model.mtp.",)
        weights = {k: v for k, v in weights.items() if not k.startswith(skip_prefixes)}

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
            prefix, qkv_key, scale_key = fused_qkv_keys(layer_idx)
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

        for layer_idx in range(int(self.args.num_nextn_predict_layers or 0)):
            prefix = f"model.mtp.layers.{layer_idx}.self_attn"
            qkv_prefix = f"{prefix}.qkv_proj"
            qkv_key = f"{qkv_prefix}.weight"
            scale_key = f"{qkv_key}_scale_inv"
            if qkv_key in weights and scale_key in weights:
                q, k, v = split_fused_qkv(
                    weights.pop(qkv_key),
                    weights.pop(scale_key),
                    tp=TP,
                    n_h=self.args.swa_num_attention_heads,
                    n_kv=self.args.swa_num_key_value_heads,
                    hd=self.args.swa_head_dim,
                    vhd=self.args.swa_v_head_dim,
                )
                weights[f"{prefix}.q_proj.weight"] = q
                weights[f"{prefix}.k_proj.weight"] = k
                weights[f"{prefix}.v_proj.weight"] = v
                continue

            # Packed MLX weights, scales, and biases share the output-row axis.
            # Split them at the same Q/K boundaries.
            if qkv_key not in weights or f"{qkv_prefix}.scales" not in weights:
                continue
            q_rows = self.args.swa_num_attention_heads * self.args.swa_head_dim
            k_rows = self.args.swa_num_key_value_heads * self.args.swa_head_dim
            boundaries = [q_rows, q_rows + k_rows]
            for suffix in ("weight", "scales", "biases"):
                fused_key = f"{qkv_prefix}.{suffix}"
                if fused_key not in weights:
                    continue
                q, k, v = mx.split(weights.pop(fused_key), boundaries, axis=0)
                weights[f"{prefix}.q_proj.{suffix}"] = q
                weights[f"{prefix}.k_proj.{suffix}"] = k
                weights[f"{prefix}.v_proj.{suffix}"] = v

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
                scale0 = expert0 + "_scale"
                if scale0 in weights:
                    packed, scales = [], []
                    for e in range(self.args.n_routed_experts):
                        key = f"{prefix}.experts.{e}.{proj}.weight"
                        weight = weights.pop(key)
                        scale = weights.pop(key + "_scale")
                        if (
                            weight.dtype != mx.uint8
                            or scale.dtype != mx.uint8
                            or scale.shape != (weight.shape[0], weight.shape[1] // 16)
                        ):
                            raise ValueError(
                                f"Invalid MiMo MXFP4 weight/scale pair: {key}"
                            )
                        packed.append(weight.view(mx.uint32))
                        scales.append(scale)
                    target = f"{prefix}.switch_mlp.{proj}"
                    weights[f"{target}.weight"] = mx.stack(packed)
                    weights[f"{target}.scales"] = mx.stack(scales)
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
