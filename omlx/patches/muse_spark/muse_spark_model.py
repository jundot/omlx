# SPDX-License-Identifier: Apache-2.0
"""Muse Spark text model for mlx-lm.

Muse Spark (Meta) is a multimodal MoE model. This module implements the
language model only. The block structure follows the transformers reference
(``modeling_muse_spark.py``):

* gated attention with scaleless Q/K norm, a Q scale factor and a scaleless
  per-head norm on the attention output
* RoPE on sliding-window layers only; full-attention layers are NoPE
* learned residual gates: ``alpha * residual + beta * branch``
* every MoE block projects ``hidden_size`` down to ``moe_hidden_size`` before
  the routed experts and back up afterwards, and normalizes every expert
  output before the routing weights are applied

The norms reproduce the reference rounding order (scale in FP32, then one cast
to the activation dtype). ``mx.fast.rms_norm`` casts before it scales, which
changes BF16 results.

The routed experts are a ``SwitchGLU`` at ``mlp.switch_mlp``. Checkpoints in
the layout of mlx-lm's own Muse Spark model (0.32.2 and later) load as well:
``upstream_layout`` maps their names, and the norms and residual gates then
read the stored gains and coefficients as they are.

ModelOpt NVFP4 checkpoints keep the packed E2M1 codes and E4M3 block scales
as-is. Like mlx-lm, a projection takes ``global_scale`` only when the
checkpoint has one, stored as the absolute maximum the per-expert tensor
scale came from. The pinned mlx-lm cannot apply it in ``gather_qmm``, so
``ScaledSwitchGLU`` multiplies the matmul inputs by the tensor scale. This
keeps the matmul outputs at their real magnitude (unscaled outputs overflow
FP16).
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.pipeline import PipelineMixin
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)

from omlx.patches.mimo_v2.decode_fast import (
    router_gemv_supported,
    router_logits,
    router_select,
)
from omlx.patches.module_cache import cached_per_module
from omlx.patches.muse_spark.norm_kernels import (
    moe_tail,
    post_attention,
    post_moe,
    scaled_rms,
)
from omlx.patches.muse_spark.upstream_layout import (
    CENTERED_NORMS,
    RESIDUAL_GATE_EPS,
    RESIDUAL_GATES,
    gate_coefficients,
    is_upstream_layout,
    read_upstream_text,
)

# Token rows that take the router gemv kernel (decode and small batches).
_ROUTER_GEMV_MAX_ROWS = 8
_FP32_PARAMS = ("mlp.gate.", "residual_gate.", "global_scale")
# ModelOpt stores the NVFP4 tensor scale as a multiplier. mlx-lm keeps the
# absolute maximum it came from: amax = scale * 6 * 448 (E2M1 max x E4M3 max).
_NVFP4_GLOBAL_SCALE = 6.0 * 448.0


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "muse_spark"
    hidden_size: int = 8192
    num_hidden_layers: int = 62
    intermediate_size: int = 16384
    num_attention_heads: int = 128
    num_key_value_heads: int = 16
    head_dim: int = 64
    vocab_size: int = 202048
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    sliding_window: int = 2048
    qk_scale_factor: float = 5.4730064863875
    output_multiplier: float = 0.17677669529663687
    final_logit_softcapping: float = 20.0
    max_position_embeddings: int = 1048576
    rope_theta: float = 500000.0
    rope_parameters: dict[str, Any] | None = None
    layer_types: list[str] | None = None
    layer_rope_theta: list[float] | None = None
    mlp_layer_types: list[str] | None = None
    moe_hidden_size: int = 4096
    moe_intermediate_size: int = 4096
    num_local_experts: int = 256
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 1.0
    residual_gate_temperature: float = 0.3
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    quantization_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, params):
        # HF checkpoints nest the language model under "text_config".
        if "text_config" in params:
            merged = dict(params)
            merged.update(params["text_config"])
            merged["model_type"] = params.get("model_type", cls.model_type)
            params = merged
        return super().from_dict(params)

    def __post_init__(self):
        if self.rope_parameters and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.layer_types is None:
            self.layer_types = [
                (
                    "full_attention"
                    if (self.num_hidden_layers - 1 - i) % 4 == 0
                    else "sliding_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        if self.layer_rope_theta is None:
            self.layer_rope_theta = [
                0 if t == "full_attention" else self.rope_theta
                for t in self.layer_types
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["sparse"] * self.num_hidden_layers


def is_modelopt_export(quantization_config: dict | None) -> bool:
    return (quantization_config or {}).get("quant_method") == "modelopt"


def modelopt_nvfp4_expert_layers(args: ModelArgs) -> list[int]:
    """Return the layers whose routed experts a ModelOpt export packed as NVFP4."""
    config = args.quantization_config or {}
    if not is_modelopt_export(config):
        return []
    if config.get("quant_algo") != "NVFP4":
        raise ValueError(
            f"Unsupported ModelOpt quantization {config.get('quant_algo')!r} "
            "for Muse Spark."
        )
    skip = set(config.get("modules_to_not_convert") or ())
    return [
        i
        for i, kind in enumerate(args.mlp_layer_types)
        if kind == "sparse"
        and f"model.language_model.layers.{i}.mlp.experts" not in skip
    ]


@mx.compile
def _input_scaled_rms_norm(x, weight, eps):
    # The reference rounds the scaled input to the activation dtype before the norm.
    x = (x.astype(mx.float32) * weight.astype(mx.float32)).astype(x.dtype)
    return mx.fast.rms_norm(x, None, eps)


@mx.compile
def _scaled_queries(queries, eps, factor):
    queries = mx.fast.rms_norm(queries, None, eps)
    # The reference multiplies in FP32; a BF16 scalar would round the factor.
    return (queries.astype(mx.float32) * factor).astype(queries.dtype)


@mx.compile
def _gated_output(output, gate, eps):
    # Per-head norm, heads back to the hidden axis, then the sigmoid gate.
    output = mx.fast.rms_norm(output, None, eps)
    batch, heads, length, dim = output.shape
    output = output.transpose(0, 2, 1, 3).reshape(batch, length, heads * dim)
    return output * mx.sigmoid(gate)


def _build_gate_coefficients(gate):
    coefficients = gate_coefficients(gate.gate, gate.temperature, RESIDUAL_GATE_EPS)
    mx.eval(coefficients)
    return coefficients


def _shard_experts(axis):
    # Each rank keeps the whole per-expert NVFP4 tensor scale.
    def predicate(path, weight):
        return None if path.endswith("global_scale") else axis

    return predicate


class RMSNorm(nn.Module):
    """RMSNorm with a plain scale, applied in FP32."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return scaled_rms(x, self.weight, self.eps, centered=False)


class ScalelessRMSNorm(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, None, self.eps)


class CenteredRMSNorm(nn.Module):
    """RMSNorm with a zero-centered scale: norm(x) * (1 + weight).

    With ``centered`` False the weight is the gain ``T(1 + weight)`` itself.
    """

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps
        self.centered = True

    def __call__(self, x):
        return scaled_rms(x, self.weight, self.eps, centered=self.centered)


class InputScaledRMSNorm(nn.Module):
    """RMSNorm applied after the zero-centered scale: norm(x * (1 + weight)).

    With ``centered`` False the weight is the gain ``T(1 + weight)`` itself.
    """

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps
        self.centered = True

    def __call__(self, x):
        gain = 1.0 + self.weight if self.centered else self.weight
        return _input_scaled_rms_norm(x, gain, self.eps)


class ResidualGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = mx.zeros((args.hidden_size,), dtype=mx.float32)
        self.temperature = args.residual_gate_temperature

    def __call__(self, residual, hidden):
        alpha, beta = self.coefficients()
        return (alpha * residual + beta * hidden).astype(residual.dtype)

    def use_fixed_coefficients(self):
        """Take stored ``alpha`` and ``beta`` weights in place of the gate."""
        dims = self.gate.shape
        del self.gate
        self.alpha = mx.ones(dims, dtype=mx.float32)
        self.beta = mx.zeros(dims, dtype=mx.float32)

    def coefficients(self):
        """FP32 ``(alpha, beta)``; they depend on the gate weights only."""
        if "alpha" in self:
            return self.alpha, self.beta
        return cached_per_module(
            self, "_omlx_gate_coefficients", _build_gate_coefficients
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qk_scale_factor = args.qk_scale_factor
        self.eps = args.rms_norm_eps
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

        bias = args.attention_bias
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)

        theta = args.layer_rope_theta[layer_idx]
        self.rope = (
            nn.RoPE(self.head_dim, traditional=False, base=float(theta))
            if theta
            else None
        )

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.n_heads, -1)
        keys = self.k_proj(x).reshape(batch, length, self.n_kv_heads, -1)
        values = self.v_proj(x).reshape(batch, length, self.n_kv_heads, -1)

        queries = _scaled_queries(queries, self.eps, self.qk_scale_factor)
        keys = mx.fast.rms_norm(keys, None, self.eps)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return self.o_proj(_gated_output(output, self.gate_proj(x), self.eps))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, hidden = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Router(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.output_multiplier = args.output_multiplier
        self.routed_scaling_factor = args.routed_scaling_factor
        self.weight = mx.zeros(
            (args.num_local_experts, args.hidden_size), dtype=mx.float32
        )
        self.e_score_correction_bias = mx.zeros(
            (args.num_local_experts,), dtype=mx.float32
        )

    def __call__(self, x):
        n, dim = self.weight.shape
        if x.size // dim <= _ROUTER_GEMV_MAX_ROWS and router_gemv_supported(dim, n):
            # The FP32 matmul below for decode rows, without the input cast.
            logits = router_logits(x, self.weight)
        else:
            logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        # One kernel: sigmoid, top-k on the bias-corrected scores, then the
        # raw scores of the chosen experts, normalized and scaled.
        inds, weights = router_select(
            logits * self.output_multiplier,
            self.e_score_correction_bias.astype(mx.float32),
            self.top_k,
            True,
            self.routed_scaling_factor,
        )
        return mx.stop_gradient(inds), weights


@mx.compile
def _scaled_swiglu(x_gate, x_up, gate_ratio, down_scale):
    gate = x_gate.astype(mx.float32) * gate_ratio
    h = nn.silu(gate) * x_up.astype(mx.float32) * down_scale
    return h.astype(x_up.dtype)


class ScaledSwitchLinear(SwitchLinear):
    """``SwitchLinear`` whose quantized form keeps its ``global_scale``.

    oQ can requantize NVFP4 experts to affine bits. The weights then keep the
    NVFP4 units and still need the tensor scale.
    """

    def to_quantized(
        self,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        global_scale: bool = False,
    ):
        # The upstream layout marks scaled experts with global_scale in the
        # quantization map; the scale itself is already this layer's.
        ql = super().to_quantized(group_size, bits, mode)
        if "global_scale" in self:
            ql.global_scale = self.global_scale
        return ql


class NVFP4SwitchLinear(QuantizedSwitchLinear):
    """NVFP4 experts of a ModelOpt export, already in their final layer.

    ``to_quantized`` returns the layer itself, since it is quantized already.
    """

    def to_quantized(self, *args, **kwargs):
        return self


class ScaledSwitchGLU(SwitchGLU):
    """``SwitchGLU`` that applies the NVFP4 tensor scales of its projections.

    The up scale multiplies the shared matmul input, which keeps the outputs
    at their real magnitude. The gate then needs only the gate/up ratio, and
    the down scale joins the FP32 activation, so no extra BF16 rounding
    occurs. Without ``global_scale`` it runs as ``SwitchGLU``.
    """

    # oMLX's gate+up fusion may replace gate_proj/up_proj with gate_up_proj;
    # the SwitchGLU forward it patches then runs the fused projection.
    _omlx_gate_up_fusion = True

    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__(input_dims, hidden_dims, num_experts)
        self.gate_proj = ScaledSwitchLinear(input_dims, hidden_dims, num_experts, False)
        self.up_proj = ScaledSwitchLinear(input_dims, hidden_dims, num_experts, False)
        self.down_proj = ScaledSwitchLinear(hidden_dims, input_dims, num_experts, False)

    def __call__(self, x, indices):
        if "gate_up_proj" in self or not any(
            "global_scale" in p for p in (self.gate_proj, self.up_proj, self.down_proj)
        ):
            return super().__call__(x, indices)
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        # oQ folds the scale into requantized experts, so a projection may
        # have none (scale 1) while others stay NVFP4.
        def scale(proj):
            if "global_scale" not in proj:
                return 1.0
            s = proj["global_scale"][idx] * (1.0 / _NVFP4_GLOBAL_SCALE)
            return s[..., None, None]

        up_scale = scale(self.up_proj)
        if "global_scale" in self.up_proj:
            x = (x.astype(mx.float32) * up_scale).astype(x.dtype)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        h = _scaled_swiglu(
            x_gate, x_up, scale(self.gate_proj) / up_scale, scale(self.down_proj)
        )
        x = self.down_proj(h, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.moe_hidden_size
        self.gate = Router(args)
        self.pre_expert_proj = nn.Linear(args.hidden_size, dim, bias=False)
        self.pre_expert_norm = CenteredRMSNorm(dim, args.rms_norm_eps)
        self.switch_mlp = ScaledSwitchGLU(
            dim, args.moe_intermediate_size, args.num_local_experts
        )
        self.post_expert_norm = InputScaledRMSNorm(dim, args.post_norm_eps)
        self.post_expert_proj = nn.Linear(dim, args.hidden_size, bias=False)
        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        inds, weights = self.gate(x)
        h = self.pre_expert_norm(self.pre_expert_proj(x))
        y = self.switch_mlp(h, inds)
        if self.sharding_group is not None:
            # The expert norm needs the full output of every expert.
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        # post_expert_norm on every expert row, then the FP32 weighted sum.
        norm = self.post_expert_norm
        y = moe_tail(y, weights, norm.weight, norm.eps, centered=norm.centered)
        return self.post_expert_proj(y)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = (
            MoE(args) if args.mlp_layer_types[layer_idx] == "sparse" else MLP(args)
        )
        self.input_layernorm = CenteredRMSNorm(args.hidden_size, args.rms_norm_eps)
        self.pre_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.rms_norm_eps
        )
        self.post_feedforward_layernorm = InputScaledRMSNorm(
            args.hidden_size, args.post_norm_eps
        )
        self.post_attention_residual_gate = ResidualGate(args)
        self.post_feedforward_residual_gate = ResidualGate(args)
        self.post_norm_eps = args.post_norm_eps

    def __call__(self, x, mask=None, cache=None):
        h = self.self_attn(self.input_layernorm(x), mask, cache)
        # One kernel: the weightless post_attention_layernorm, the residual
        # gate and pre_feedforward_layernorm.
        alpha, beta = self.post_attention_residual_gate.coefficients()
        pre_norm = self.pre_feedforward_layernorm
        x, h = post_attention(
            x,
            h,
            alpha,
            beta,
            pre_norm.weight,
            self.post_norm_eps,
            pre_norm.eps,
            centered=pre_norm.centered,
        )

        h = self.mlp(h)
        alpha, beta = self.post_feedforward_residual_gate.coefficients()
        post_norm = self.post_feedforward_layernorm
        return post_moe(
            x,
            h,
            alpha,
            beta,
            post_norm.weight,
            post_norm.eps,
            centered=post_norm.centered,
        )


class MuseSparkModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        # The embeddings get a scaleless RMSNorm, not a sqrt(dim) scale.
        self.embed_norm = ScalelessRMSNorm(args.rms_norm_eps)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def embed(self, inputs):
        return self.embed_norm(self.embed_tokens(inputs))

    def _masks(self, h, layers, cache):
        made = {}
        for layer, c in zip(layers, cache):
            sliding = layer.self_attn.is_sliding
            if sliding not in made:
                window = self.args.sliding_window if sliding else None
                made[sliding] = create_attention_mask(h, c, window_size=window)
            if len(made) == 2:
                break
        return made

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embed(inputs)

        layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(layers)
        masks = self._masks(h, layers, cache)

        rank, size = self.pipeline_rank, self.pipeline_size
        if rank < size - 1:
            h = mx.distributed.recv_like(h, rank + 1)

        for layer, c in zip(layers, cache):
            h = layer(h, masks[layer.self_attn.is_sliding], c)

        if rank != 0:
            h = mx.distributed.send(h, (rank - 1) % size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)
        if size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


def softcap_logits(logits: mx.array, args: ModelArgs) -> mx.array:
    cap = args.final_logit_softcapping
    scaled = logits.astype(mx.float32) * (args.output_multiplier / cap)
    return (mx.tanh(scaled) * cap).astype(logits.dtype)


def sanitize_experts(weights: dict) -> dict:
    """Map the HF routed-expert tensors to the mlx-lm ``switch_mlp`` layout."""
    out = {}
    for key, value in weights.items():
        if ".mlp.experts." not in key:
            out[key] = value
            continue

        prefix, name = key.split(".mlp.experts.")
        if name.startswith("post_expert_norm."):
            out[f"{prefix}.mlp.{name}"] = value
            continue
        if name.endswith("_input_scale"):
            # Activation scales of the W4A4 export; activations stay in BF16.
            continue
        proj = "gate_up_proj" if name.startswith("gate_up_proj") else "down_proj"
        field = {
            "": "weight",
            "_weight_scale": "scales",
            "_weight_scale_2": "global_scale",
        }[name[len(proj) :]]
        if field == "weight" and value.dtype == mx.uint8:
            # Packed NVFP4 codes, two per byte.
            value = value.view(mx.uint32)
        elif field == "global_scale":
            value = value.astype(mx.float32) * _NVFP4_GLOBAL_SCALE

        switch = f"{prefix}.mlp.switch_mlp."
        if proj == "down_proj":
            out[f"{switch}down_proj.{field}"] = value
        elif field == "global_scale":
            # The fused projection has one tensor scale per half.
            out[f"{switch}gate_proj.global_scale"] = value[:, 0]
            out[f"{switch}up_proj.global_scale"] = value[:, 1]
        else:
            # HF stores the experts as (experts, 2 * hidden, dim), gate first.
            gate, up = mx.split(value, 2, axis=1)
            out[f"{switch}gate_proj.{field}"] = gate
            out[f"{switch}up_proj.{field}"] = up
    return out


def install_global_scales(model: nn.Module, weights: dict, modelopt: bool):
    """Give ``global_scale`` to the switch projections that have one in ``weights``.

    A ModelOpt export has no MLX quantization config for its packed NVFP4
    experts (scales without affine biases), so with ``modelopt`` they become
    nvfp4 layers here. Otherwise the loader quantizes every layer by config,
    and ``ScaledSwitchLinear`` keeps the scale through that.
    """
    if not isinstance(model, nn.Module):
        # oQ runs sanitize on a stand-in object only to map the names.
        return
    modules = None
    for key in weights:
        if not key.endswith(".global_scale"):
            continue
        if modules is None:
            modules = dict(model.named_modules())
        path = key[: -len(".global_scale")]
        proj = modules[path]
        if (
            modelopt
            and isinstance(proj, SwitchLinear)
            and f"{path}.scales" in weights
            and f"{path}.biases" not in weights
        ):
            parent, name = path.rsplit(".", 1)
            proj = NVFP4SwitchLinear(
                proj.input_dims,
                proj.output_dims,
                proj.num_experts,
                bias=False,
                group_size=16,
                bits=4,
                mode="nvfp4",
            )
            setattr(modules[parent], name, proj)
        proj.global_scale = mx.ones((proj.num_experts,), dtype=mx.float32)


def use_upstream_values(model: nn.Module):
    """Make the decoder layers read upstream norm gains and gate coefficients."""
    if not isinstance(model, nn.Module):
        # oQ runs sanitize on a stand-in object only to map the names.
        return
    for module in model.modules():
        if isinstance(module, DecoderLayer):
            for path, norm in module.named_modules():
                if path in CENTERED_NORMS:
                    norm.centered = False
            for gate in RESIDUAL_GATES:
                getattr(module, gate).use_fixed_coefficients()


def streamed_decoder_layer(args: ModelArgs, layer_idx: int) -> DecoderLayer:
    """Bare decoder block for streamed oQ calibration.

    ModelOpt NVFP4 experts arrive dequantized without their tensor scales, so
    the block takes ``global_scale`` for them as the full model does at load.
    """
    layer = DecoderLayer(args, layer_idx)
    if layer_idx in modelopt_nvfp4_expert_layers(args):
        switch = layer.mlp.switch_mlp
        for proj in (switch.gate_proj, switch.up_proj, switch.down_proj):
            proj.global_scale = mx.ones((args.num_local_experts,), dtype=mx.float32)
    return layer


def cast_predicate(key: str) -> bool:
    """Keep the router, the residual gates and the expert scales in FP32."""
    return not any(part in key for part in _FP32_PARAMS)


def make_cache(args: ModelArgs, layers):
    return [
        (
            RotatingKVCache(max_size=args.sliding_window, keep=0)
            if layer.self_attn.is_sliding
            else KVCache()
        )
        for layer in layers
    ]


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = MuseSparkModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return softcap_logits(out, self.args)

    def sanitize(self, weights):
        upstream = is_upstream_layout(weights)
        out = {}
        for key, value in weights.items():
            # Drop the vision and audio towers of HF and mlx-vlm checkpoints.
            if key.startswith(
                (
                    "model.vision_",
                    "model.audio_tower.",
                    "vision_",
                    "audio_tower.",
                )
            ):
                continue
            if key.startswith("model.language_model."):
                key = "model." + key[len("model.language_model.") :]
            elif key.startswith("language_model."):
                key = key[len("language_model.") :]
            out[key] = value
        if self.args.tie_word_embeddings:
            out.pop("lm_head.weight", None)
        if upstream:
            args = self.args
            out = read_upstream_text(
                out,
                "model.layers.",
                n_heads=args.num_attention_heads,
                n_kv_heads=args.num_key_value_heads,
                head_dim=args.head_dim,
            )
            use_upstream_values(self)
        else:
            out = sanitize_experts(out)
        install_global_scales(
            self, out, is_modelopt_export(self.args.quantization_config)
        )
        return out

    def shard(self, group: mx.distributed.Group | None = None):
        group = group or mx.distributed.init()
        size = group.size()
        for layer in self.model.layers:
            attn = layer.self_attn
            for name in ("q_proj", "k_proj", "v_proj", "gate_proj"):
                setattr(
                    attn,
                    name,
                    shard_linear(getattr(attn, name), "all-to-sharded", group=group),
                )
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.n_heads //= size
            attn.n_kv_heads //= size

            mlp = layer.mlp
            if isinstance(mlp, MLP):
                mlp.gate_proj = shard_linear(
                    mlp.gate_proj, "all-to-sharded", group=group
                )
                mlp.up_proj = shard_linear(mlp.up_proj, "all-to-sharded", group=group)
                mlp.down_proj = shard_linear(
                    mlp.down_proj, "sharded-to-all", group=group
                )
            else:
                # Experts shard their intermediate dimension. The router and the
                # pre/post expert projections are replicated.
                mlp.sharding_group = group
                switch = mlp.switch_mlp
                shard_inplace(switch.gate_proj, _shard_experts(1), group=group)
                shard_inplace(switch.up_proj, _shard_experts(1), group=group)
                shard_inplace(switch.down_proj, _shard_experts(-1), group=group)

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    @property
    def cast_predicate(self):
        return cast_predicate

    def make_cache(self):
        return make_cache(self.args, self.layers)
