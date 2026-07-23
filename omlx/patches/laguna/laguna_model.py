# SPDX-License-Identifier: Apache-2.0
"""Laguna XS.2 MLX model — vendored from mlx-lm PR #1223 (Blaizzy).

The published configuration mixes full and sliding-window attention layers,
which may use different RoPE settings, and mixes dense and routed-MoE MLP
layers. Keep this implementation structurally aligned with the upstream patch:
the sanitizer bridges checkpoint tensor names to MLX-LM's ``SwitchGLU`` layout.
The mixed-cache method follows the proposed upstream follow-up in
``Blaizzy/mlx-lm#26`` so sliding layers retain only their usable window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = False
    attention_bias: bool = False
    gating: bool | str = True
    tie_word_embeddings: bool = False
    rope_theta: float = 500000.0
    rope_parameters: dict[str, Any] | None = None
    rope_scaling: dict[str, Any] | None = None
    partial_rotary_factor: float | None = None
    rope_style: str = "rotate-half"
    sliding_window: int | None = None
    layer_types: list[str] | None = None
    num_attention_heads_per_layer: list[int] | None = None
    swa_rope_parameters: dict[str, Any] | None = None
    swa_attention_sink_enabled: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    mlp_only_layers: list[int] = field(default_factory=lambda: [0])
    mlp_layer_types: list[str] | None = None
    gating_types: list[str] | None = None
    moe_routed_scaling_factor: float = 1.0
    moe_apply_router_weight_on_input: bool = False
    moe_router_logit_softcapping: float = 0.0
    moe_router_use_sigmoid: bool = True

    def __post_init__(self):
        if self.gating is True:
            self.gating = "per-head"

        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers.")

        # Laguna S-2.1 publishes explicit per-layer MLP and gating lists that
        # override the legacy mlp_only_layers/decoder_sparse_step cadence and
        # the single global gating mode.
        if self.mlp_layer_types is not None and (
            len(self.mlp_layer_types) != self.num_hidden_layers
        ):
            raise ValueError("mlp_layer_types must match num_hidden_layers.")
        if self.gating_types is not None:
            if len(self.gating_types) != self.num_hidden_layers:
                raise ValueError("gating_types must match num_hidden_layers.")
            self.gating_types = [
                gating_type.replace("_", "-") for gating_type in self.gating_types
            ]

        if self.num_attention_heads_per_layer is None:
            self.num_attention_heads_per_layer = [
                self.num_attention_heads
            ] * self.num_hidden_layers
        if len(self.num_attention_heads_per_layer) != self.num_hidden_layers:
            raise ValueError(
                "num_attention_heads_per_layer must match num_hidden_layers."
            )
        if any(
            h % self.num_key_value_heads for h in self.num_attention_heads_per_layer
        ):
            raise ValueError(
                "Every query-head count must be divisible by num_key_value_heads."
            )

        # Laguna groups RoPE settings by attention family in ``config.json``;
        # MLX's RoPE initializer needs one concrete mapping per layer.
        rope_parameters = (
            dict(self.rope_parameters)
            if self.rope_parameters is not None
            else (
                dict(self.rope_scaling)
                if self.rope_scaling is not None
                else {"rope_type": "default", "rope_theta": self.rope_theta}
            )
        )

        layer_types = set(self.layer_types)
        layer_rope_parameters = {
            k: v
            for k, v in rope_parameters.items()
            if k in layer_types and isinstance(v, dict)
        }
        if layer_rope_parameters:
            top_level_parameters = {
                k: v
                for k, v in rope_parameters.items()
                if k not in layer_types and not isinstance(v, dict)
            }

            def rope_parameters_for(layer_type: str) -> dict[str, Any]:
                params = dict(layer_rope_parameters.get(layer_type, {}))
                for k, v in top_level_parameters.items():
                    params.setdefault(k, v)
                return params

            default_layer_type = (
                "full_attention"
                if "full_attention" in layer_rope_parameters
                else next(iter(layer_rope_parameters))
            )
            self.rope_parameters = rope_parameters_for(default_layer_type)

            if (
                self.swa_rope_parameters is None
                and "sliding_attention" in layer_rope_parameters
            ):
                self.swa_rope_parameters = rope_parameters_for("sliding_attention")
        else:
            self.rope_parameters = rope_parameters

        if self.swa_rope_parameters is not None:
            self.swa_rope_parameters = dict(self.swa_rope_parameters)

        self.rope_parameters.setdefault("rope_type", "default")
        if self.swa_rope_parameters is not None:
            self.swa_rope_parameters.setdefault("rope_type", "default")

        if self.partial_rotary_factor is not None:
            self.rope_parameters.setdefault(
                "partial_rotary_factor", self.partial_rotary_factor
            )
            if self.swa_rope_parameters is not None:
                self.swa_rope_parameters.setdefault(
                    "partial_rotary_factor", self.partial_rotary_factor
                )


def _rope_base(args: ModelArgs, rope_config: dict[str, Any]) -> float:
    return float(rope_config.get("rope_theta", args.rope_theta))


def _rope_dims(args: ModelArgs, rope_config: dict[str, Any]) -> int:
    partial = float(rope_config.get("partial_rotary_factor", 1.0))
    return int(args.head_dim * partial)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class LagunaTopKRouter(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.use_sigmoid = args.moe_router_use_sigmoid
        self.router_logit_softcapping = args.moe_router_logit_softcapping
        self.proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_experts,))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        dtype = x.dtype
        logits = self.proj(x).astype(mx.float32)
        if self.router_logit_softcapping > 0.0:
            c = self.router_logit_softcapping
            logits = mx.tanh(logits / c) * c

        scores = mx.sigmoid(logits) if self.use_sigmoid else mx.softmax(logits, axis=-1)
        # The correction bias changes which experts are selected, but the model
        # weights selected expert outputs using the original router scores.
        corrected_scores = scores + self.e_score_correction_bias.astype(scores.dtype)

        k = self.top_k
        inds = mx.stop_gradient(
            mx.argpartition(-corrected_scores, kth=k - 1, axis=-1)[..., :k]
        )
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        return inds, weights.astype(dtype)


class LagunaSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.moe_apply_router_weight_on_input:
            raise NotImplementedError(
                "moe_apply_router_weight_on_input=True is not supported."
            )
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.gate = LagunaTopKRouter(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = mx.sum(y * scores[..., None], axis=-2)
        if self.routed_scaling_factor != 1.0:
            y = y * self.routed_scaling_factor
        return y + self.shared_expert(x)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        query_head_counts = args.num_attention_heads_per_layer
        layer_types = args.layer_types
        if query_head_counts is None or layer_types is None:
            raise ValueError(
                "Laguna attention layers require normalized model arguments."
            )

        self.n_heads = query_head_counts[layer_idx]
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        gating = (
            args.gating_types[layer_idx]
            if args.gating_types is not None
            else args.gating
        )
        self.gate_per_head = gating == "per-head"
        self.gating = bool(gating) and gating != "none"
        self.is_sliding = layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = args.sliding_window if self.is_sliding else None

        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=args.qkv_bias)
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.qkv_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.qkv_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias
        )

        if self.gating:
            gate_dim = (
                self.n_heads if self.gate_per_head else self.n_heads * self.head_dim
            )
            self.g_proj = nn.Linear(dim, gate_dim, bias=False)

        if self.is_sliding and args.swa_attention_sink_enabled:
            self.sink = mx.zeros((self.n_heads,))
        else:
            self.sink = None

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        rope_config = (
            args.swa_rope_parameters
            if self.is_sliding and args.swa_rope_parameters is not None
            else args.rope_parameters
        )
        if rope_config is None:
            raise ValueError(
                "Laguna attention layers require normalized RoPE parameters."
            )
        self.rope = initialize_rope(
            _rope_dims(args, rope_config),
            base=_rope_base(args, rope_config),
            traditional=False,
            scaling_config=rope_config,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        bsz, seq_len, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(
            queries.reshape(bsz, seq_len, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

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
            sinks=self.sink,
        )
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)

        if self.gating:
            gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
            if self.gate_per_head:
                shape = output.shape
                output = (
                    output.reshape(bsz, seq_len, self.n_heads, self.head_dim)
                    * gate[..., None]
                ).reshape(shape)
            else:
                output = output * gate

        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp: MLP | LagunaSparseMoeBlock
        # An explicit per-layer ``mlp_layer_types`` list wins; otherwise
        # ``mlp_only_layers`` preserves explicit dense layers and remaining
        # layers follow the configured sparse-MoE cadence.
        if args.mlp_layer_types is not None:
            is_sparse = args.mlp_layer_types[layer_idx] == "sparse"
        else:
            is_sparse = (layer_idx not in args.mlp_only_layers) and (
                args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
            )
        if is_sparse:
            self.mlp = LagunaSparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        layer_types = args.layer_types
        if layer_types is None:
            raise ValueError(
                "Laguna decoder layers require normalized model arguments."
            )
        self.attention_type = layer_types[layer_idx]

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LagunaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx) for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        layer_types = args.layer_types
        if layer_types is None:
            raise ValueError("Laguna models require normalized layer types.")
        # The cache type differs between full and sliding attention. Retain one
        # representative index for each family when building their masks.
        self.fa_idx = layer_types.index("full_attention")
        self.swa_idx = (
            layer_types.index("sliding_attention")
            if "sliding_attention" in layer_types
            else None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            sliding_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.args.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = (
                sliding_mask
                if layer.attention_type == "sliding_attention"
                else full_mask
            )
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LagunaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def make_cache(self) -> list[KVCache | RotatingKVCache]:
        """Bound sliding-attention KV state while retaining global history."""
        layer_types = self.args.layer_types
        if layer_types is None:
            raise ValueError("Laguna caches require normalized layer types.")

        caches: list[KVCache | RotatingKVCache] = []
        for attention_type in layer_types:
            if attention_type == "sliding_attention" and self.args.sliding_window:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            else:
                # Missing/invalid sliding-window metadata must not create a
                # RotatingKVCache with an unusable maximum size.
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        weights = self._strip_language_model_prefix(weights)
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Stack experts before the format transforms so packed/scale sidecars
        # convert as a handful of batched per-layer tensors instead of one
        # tiny op per expert.
        weights = self._remap_router_weights(weights)
        weights = self._stack_experts(weights)
        weights = self._repack_compressed_nvfp4_weights(weights)
        weights = self._unpack_compressed_tensors(weights)
        weights = self._dequantize_fp8_block_weights(weights)
        return {
            k: v
            for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k
            and not k.endswith(".self_attn.k_scale")
            and not k.endswith(".self_attn.v_scale")
            and not k.endswith(".input_global_scale")
            and not k.endswith(".weight_shape")
        }

    def _strip_language_model_prefix(self, weights):
        """Normalize VLM-tree checkpoints to the flat text-model tree.

        Conversions and oQ outputs produced through the mlx-vlm route key
        everything under ``language_model.`` (e.g.
        ``language_model.model.layers...``, ``language_model.lm_head``); the
        text model expects ``model.*`` and ``lm_head.*`` directly.
        """
        prefix = "language_model."
        if not any(k.startswith(prefix) for k in weights):
            return weights
        return {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in weights.items()
        }

    def _repack_compressed_nvfp4_weights(self, weights):
        """Convert nvfp4-pack-quantized tensors to the mlx nvfp4 layout.

        The fp4 codes reinterpret bit-exactly from uint8 pairs to the uint32
        packing mlx expects. mlx nvfp4 is single-level, so the per-tensor
        ``weight_global_scale`` is folded into the e4m3 group scales.
        """
        packed_keys = [k for k in weights if k.endswith(".weight_packed")]
        for pk in packed_keys:
            base = pk[: -len("weight_packed")]
            scale_key = f"{base}weight_scale"
            global_key = f"{base}weight_global_scale"
            if scale_key not in weights or global_key not in weights:
                continue
            global_scale = weights.pop(global_key).astype(mx.float32)
            if global_scale.ndim:
                global_scale = global_scale.reshape(*global_scale.shape[:-1], 1, 1)
            decoded = mx.from_fp8(weights.pop(scale_key), dtype=mx.float32)
            weights[f"{base}weight"] = weights.pop(pk).view(mx.uint32)
            weights[f"{base}scales"] = mx.to_fp8(decoded / global_scale)
            weights.pop(f"{base}weight_shape", None)
        return weights

    def _unpack_compressed_tensors(self, weights):
        """Convert pack-quantized int4 tensors to the mlx affine layout.

        Symmetric int4 values in [-8, 7] map exactly onto mlx affine 4-bit
        with ``biases = -8 * scales``. Handles flat Linears and stacked expert
        tensors alike. Pairs carrying a ``weight_global_scale`` sidecar are
        nvfp4-pack-quantized, and asymmetric pairs carry a zero point; both
        are left untouched here.
        """
        packed_keys = [k for k in weights if k.endswith(".weight_packed")]
        for pk in packed_keys:
            base = pk[: -len("weight_packed")]
            if (
                f"{base}weight_scale" not in weights
                or f"{base}weight_global_scale" in weights
                or f"{base}weight_zero_point" in weights
            ):
                continue
            scales = weights.pop(f"{base}weight_scale")
            weights[f"{base}weight"] = weights.pop(pk).view(mx.uint32)
            weights[f"{base}scales"] = scales
            weights[f"{base}biases"] = (-8 * scales).astype(scales.dtype)
            weights.pop(f"{base}weight_shape", None)
        return weights

    def _dequantize_fp8_block_weights(self, weights):
        """Convert compressed-tensors float-quantized (FP8) tensors.

        The checkpoint stores e4m3 weights (surfaced by ``mx.load`` as uint8)
        with float32 block scales, typically [128, 128] per block. Metal has
        no fp8 matmul, so decode the blocks and requantize to 8-bit affine;
        the R1 hadamard transform in these checkpoints is already folded into
        the stored weights and needs no runtime op.
        """
        scale_keys = [k for k in weights if k.endswith(".weight_scale")]
        for sk in scale_keys:
            base = sk[: -len("weight_scale")]
            wk = f"{base}weight"
            if wk not in weights or weights[wk].dtype != mx.uint8:
                continue
            scale = weights.pop(sk).astype(mx.float32)
            w = mx.from_fp8(weights.pop(wk), dtype=mx.float32)
            out_dim, in_dim = w.shape[-2], w.shape[-1]
            blocks_out, blocks_in = scale.shape[-2], scale.shape[-1]
            lead = w.shape[:-2]
            w = w.reshape(
                *lead,
                blocks_out,
                out_dim // blocks_out,
                blocks_in,
                in_dim // blocks_in,
            )
            w = w * scale[..., :, None, :, None]
            w = w.reshape(*lead, out_dim, in_dim).astype(mx.bfloat16)
            quantized, scales, biases = mx.quantize(w, group_size=64, bits=8)
            weights[wk] = quantized
            weights[f"{base}scales"] = scales
            weights[f"{base}biases"] = biases
        return weights

    def _remap_router_weights(self, weights):
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            gate_weight = f"{prefix}.gate.weight"
            if gate_weight in weights:
                weights[f"{prefix}.gate.proj.weight"] = weights.pop(gate_weight)

            legacy_bias = f"{prefix}.experts.e_score_correction_bias"
            if legacy_bias in weights:
                weights[f"{prefix}.gate.e_score_correction_bias"] = weights.pop(
                    legacy_bias
                )
        return weights

    def _stack_experts(self, weights):
        # Checkpoints store every expert separately, while ``SwitchGLU`` expects
        # a leading expert dimension. Stack quantization sidecars too so a
        # quantized checkpoint stays aligned with its corresponding weights.
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.mlp"
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                for suffix in [
                    "weight",
                    "scales",
                    "biases",
                    "weight_packed",
                    "weight_scale",
                    "weight_global_scale",
                ]:
                    first_key = f"{prefix}.experts.0.{proj}.{suffix}"
                    if first_key not in weights:
                        continue
                    weights[f"{prefix}.switch_mlp.{proj}.{suffix}"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{proj}.{suffix}")
                            for e in range(self.args.num_experts)
                        ]
                    )
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate.proj"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.layers
