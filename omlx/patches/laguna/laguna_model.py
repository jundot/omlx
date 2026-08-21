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

import os
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import base as mlx_lm_base
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
)

# Compiled ``mx.compile(shapeless=True)`` fusions, ported from the Swift
# challenge's ``lagunaCompiled*`` closures. Each body is the IDENTICAL
# expression tree the eager code builds (same ops, same order, no
# reassociation; compile only fuses elementwise ops, never reductions), so
# every output is bit-exact against the uncompiled path, but the elementwise
# tail is lowered once instead of rebuilt on every decode step. Set
# OMLX_LAGUNA_COMPILED_FUSIONS=0 to disable.
_COMPILED_FUSIONS = os.environ.get("OMLX_LAGUNA_COMPILED_FUSIONS", "1") != "0"


def _compiled_softplus_gate(gate: mx.array) -> mx.array:
    """Softplus output gate computed in float32, cast back to the input dtype."""
    return nn.softplus(gate.astype(mx.float32)).astype(gate.dtype)


def _swiglu(gate: mx.array, up: mx.array) -> mx.array:
    """SiLU(gate) * up (single-output, elementwise -> bit-exact when compiled)."""
    return swiglu(gate, up)


def _compiled_topk_normalize(weights: mx.array) -> mx.array:
    """Top-k mixture-weight renormalization (``weights / sum(weights)``)."""
    return weights / mx.sum(weights, axis=-1, keepdims=True)


if _COMPILED_FUSIONS:
    _compiled_softplus_gate = mx.compile(_compiled_softplus_gate, shapeless=True)
    _compiled_topk_normalize = mx.compile(_compiled_topk_normalize, shapeless=True)
    _swiglu = mx.compile(_swiglu, shapeless=True)


def _make_compiled_expert_combine(scale: float):
    """Weighted routed-expert reduction, routed scale, and shared-expert add."""

    def _combine(y: mx.array, weights: mx.array, shared: mx.array) -> mx.array:
        routed = mx.sum(y * weights[..., None], axis=-2)
        return routed * scale + shared

    return mx.compile(_combine, shapeless=True)


def _make_compiled_expert_combine_residual(scale: float):
    """Weighted reduction, scale, shared-expert add, and the decoder residual.

    ``residual + (routed * scale + shared)`` is bit-exact against the eager
    ``h + (routed*scale + shared)`` composed in the decoder layer, because IEEE
    addition is commutative (the two adds have identical operands).
    """

    def _combine(
        y: mx.array, weights: mx.array, shared: mx.array, residual: mx.array
    ) -> mx.array:
        routed = mx.sum(y * weights[..., None], axis=-2)
        return residual + (routed * scale + shared)

    return mx.compile(_combine, shapeless=True)


# One compiled weighted-expert combine per routed-scaling value, shared across
# every sparse block (the real checkpoint uses a single 2.5 factor).
_COMPILED_COMBINES: dict[float, Any] = {}
_COMPILED_COMBINES_RESIDUAL: dict[float, Any] = {}


def _compiled_combine_for(scale: float):
    combine = _COMPILED_COMBINES.get(scale)
    if combine is None:
        combine = _COMPILED_COMBINES[scale] = _make_compiled_expert_combine(scale)
    return combine


def _compiled_combine_residual_for(scale: float):
    combine = _COMPILED_COMBINES_RESIDUAL.get(scale)
    if combine is None:
        combine = _COMPILED_COMBINES_RESIDUAL[scale] = (
            _make_compiled_expert_combine_residual(scale)
        )
    return combine

# DECODE-ONLY routed gate/up fusion (Swift ``DARKBLOOM_FUSED_ROUTED_GATE_UP``):
# after load, retain a row-concatenated NVFP4 ``[gate; up]`` bank per sparse
# layer and serve single-token decode's gate/up from ONE gather-QMM instead of
# two. Bit-exact vs. the separate dispatches (each gathered output row keeps
# its own K-loop), and correct (verified against the stock path on both a
# synthetic model and the real 21.6 GB NVFP4 checkpoint), but it DEFAULTs OFF:
# on the current MLX version two 512-wide gathers beat one 1024-wide gather, so
# the fusion regresses single-token decode (~13% slower per sparse block) and
# must stay opt-in until a wider gather-QMM kernel or layout makes it a win.
_FUSED_ROUTED_GATE_UP = os.environ.get("OMLX_LAGUNA_FUSED_ROUTED_GATE_UP", "0") != "0"

# Shared-expert gate/up fusion (Swift ``DARKBLOOM_FUSED_SHARED_GATE_UP``): one
# NVFP4 quantized matmul over a row-concatenated ``[gate; up]`` bank instead of
# two. Bit-exact but "unproven in ablation" in the Swift baseline, and measured
# -2.3% decode here, so it also defaults OFF until a measured win exists.
_FUSED_SHARED_GATE_UP = os.environ.get("OMLX_LAGUNA_FUSED_SHARED_GATE_UP", "0") != "0"

# Ported mlxfast-challenge NVFP4 kernels (omlx.custom_kernels.laguna_nvfp4):
# fused gate/up QMV + in-kernel SwiGLU in ONE Metal dispatch per shared
# expert (vs the two/three stock dispatches). Default OFF: it changes the
# swiglu rounding (bf16 in-kernel, matching the challenge's reference, vs
# fp32 in the Python path), so the shared-expert outputs differ at bf16 ulp.
_LAGUNA_NVFP4_KERNELS = os.environ.get("OMLX_LAGUNA_NVFP4_KERNELS", "0") != "0"

# NVFP4 attention re-quantization (challenge lagunaNativeAffineWeight): at
# load, each bf16 attention projection is quantized to NVFP4 group-16 and a
# fused [q;k;v] bank is built so the decoder QKV + o_proj kernels can
# dispatch against it (decode-only; prefill keeps the stock bf16 params).
# Default OFF (opt-in) - changes attention numerics to the NVFP4
# approximation the challenge's ranked runtime uses.
_LAGUNA_NVFP4_ATTN = os.environ.get("OMLX_LAGUNA_NVFP4_ATTN", "0") != "0"

# Fused decode attention (challenge lagunaSlidingFusedAttention /
# lagunaFullFusedAttention): ONE Metal dispatch replaces the
# [QK-norm + RoPE] -> [K/V ring write] -> [sdpa_vector] chain for
# single-token decode. Engaged only when the NVFP4 attention bank is active
# (OMLX_LAGUNA_NVFP4_ATTN=1), the kernels are built, the layer is a sliding
# layer whose ring cache is at steady capacity (offset >= window), and the
# RoPE position atlas covers the position. Everything else falls back to the
# stock path. The kernel writes the new K/V into the ring in-place, so the
# cache clock advances without a second update_and_fetch.
_LAGUNA_FUSED_ATTN = os.environ.get("OMLX_LAGUNA_FUSED_ATTN", "1") != "0"

# Engagement counters (diagnostic; incremented by the fused decode paths).
_LAGUNA_FUSED_CALLS = [0]

# RoPE angle atlas length (challenge lagunaRoPEAngleAtlasLength). Rows are
# materialized once per attention family at first use from the family's own
# stock RoPE (probe-seed broadcast), so each row is exactly the FP32 angle
# row the stock rope would produce at that absolute position.
_LAGUNA_ROPE_ATLAS_LENGTH = 4096


class _LagunaRopeAtlas:
    """Per-family RoPE angle atlases, materialized lazily from the model's
    own stock RoPE modules (probe-seed broadcast, offset 0).

    ``rows[p]`` is exactly the FP32 angle row the family's stock rope would
    produce for absolute position p (challenge prepareRoPEAngleAtlases).
    """

    __slots__ = ("sliding", "full")

    def __init__(self, sliding: mx.array | None, full: mx.array | None):
        self.sliding = sliding
        self.full = full


_LAGUNA_ATLAS_CACHE: dict[tuple, _LagunaRopeAtlas] = {}


def _build_rope_atlas(rope: Any, dims: int, length: int, mscale: float | None) -> mx.array:
    """Materialize the FP32 angle atlas for one attention family.

    The challenge probes with a seed row whose first half is all-ones (or
    1/mscale for YaRN, since YaRN scales its rotary inputs) and second half
    zeros, broadcast along the position axis, then runs the family's own
    stock RoPE with offset 0 — row p comes back as exactly [cos(p), sin(p)].
    """
    half = dims // 2
    seed_float = 1.0 / mscale if mscale is not None else 1.0
    seed = mx.array(
        [seed_float] * half + [0.0] * half, mx.float32
    ).reshape(1, 1, 1, dims)
    bcast = mx.broadcast_to(seed, (1, 1, length, dims))
    atlas = rope(bcast, offset=0)
    mx.eval(atlas)
    return atlas


def _get_laguna_rope_atlas(rope_sliding: Any, rope_full: Any) -> _LagunaRopeAtlas:
    """Return (sliding, full) angle atlases, cached per rope identity."""
    key = (id(rope_sliding), id(rope_full))
    cached = _LAGUNA_ATLAS_CACHE.get(key)
    if cached is not None:
        return cached
    sliding = None
    full = None
    if rope_sliding is not None:
        sliding = _build_rope_atlas(
            rope_sliding, 128, _LAGUNA_ROPE_ATLAS_LENGTH, None
        )
    if rope_full is not None:
        full = _build_rope_atlas(
            rope_full, 64, _LAGUNA_ROPE_ATLAS_LENGTH, 1.3465735912322998
        )
    atlas = _LagunaRopeAtlas(sliding, full)
    _LAGUNA_ATLAS_CACHE[key] = atlas
    return atlas


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
    """Dense MLP, also used as the sparse block's shared expert.

    When gate/up are NVFP4 group-16 4-bit ``QuantizedLinear`` banks and
    ``OMLX_LAGUNA_FUSED_SHARED_GATE_UP=1``, the shared-expert projection serves
    gate and up from ONE quantized matmul over a row-concatenated ``[gate; up]``
    bank (mirrors the Swift challenge's ``DARKBLOOM_FUSED_SHARED_GATE_UP``,
    opt-in there too). Bit-exact vs the separate dispatches; the dense BF16
    layer-0 MLP never fuses.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self._fused_gateup_weight: mx.array | None = None
        self._fused_gateup_scales: mx.array | None = None
        self._fused_gateup_split: int = 0
        self._fusion_ready: bool | None = None

    def _prepare_fused_gate_up(self) -> bool:
        """Build the fused shared-expert NVFP4 ``[gate; up]`` bank."""
        if self._fusion_ready is not None:
            return self._fusion_ready
        gate = self.gate_proj
        up = self.up_proj

        def _nvfp4_pair(module: Any) -> bool:
            return (
                isinstance(module, nn.QuantizedLinear)
                and module.mode == "nvfp4"
                and module.group_size == 16
                and module.bits == 4
                and module.get("bias") is None
                and module.get("biases") is None
                and module.weight.ndim == 2
                and module.weight.dtype == mx.uint32
                and module.scales.ndim == 2
                and module.scales.dtype == mx.uint8
            )

        ready = (
            _nvfp4_pair(gate)
            and _nvfp4_pair(up)
            and gate.weight.shape == up.weight.shape
            and gate.scales.shape == up.scales.shape
            and gate.scales.shape[0] == gate.weight.shape[0]
            and gate.weight.shape[1] * 8 == gate.scales.shape[1] * 16
        )
        if ready:
            self._fused_gateup_weight = mx.concatenate([gate.weight, up.weight], axis=0)
            self._fused_gateup_scales = mx.concatenate([gate.scales, up.scales], axis=0)
            self._fused_gateup_split = gate.weight.shape[0]
        self._fusion_ready = ready
        return ready

    def __call__(self, x) -> mx.array:
        if (
            _LAGUNA_NVFP4_KERNELS
            and self._prepare_fused_gate_up()
            and x.ndim == 3
            and x.shape[-2] == 1
        ):
            from omlx.custom_kernels.laguna_nvfp4 import fast as _laguna_nvfp4

            if _laguna_nvfp4.has_native():
                # fused gate/up NVFP4 QMV + in-kernel SwiGLU, one dispatch
                act = _laguna_nvfp4.shared_nvfp4_swiglu_qmv(
                    x.reshape(-1),
                    self._fused_gateup_weight,
                    self._fused_gateup_scales,
                )  # [512] bf16
                return self.down_proj(act[None, None, :])
        if (
            _FUSED_SHARED_GATE_UP
            and self._prepare_fused_gate_up()
        ):
            gate_up = mx.quantized_matmul(
                x,
                self._fused_gateup_weight,
                self._fused_gateup_scales,
                None,
                transpose=True,
                group_size=16,
                bits=4,
                mode="nvfp4",
            )
            split = self._fused_gateup_split
            return self.down_proj(
                _swiglu(gate_up[..., :split], gate_up[..., split:])
            )
        return self.down_proj(_swiglu(self.gate_proj(x), self.up_proj(x)))


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

        # NOTE (correctness, challenge commit f8848e0): the Swift challenge
        # compiles this router tail (``lagunaCompiledRouterTail``: two outputs
        # consuming the same sigmoid intermediate) into one kernel. In Python
        # MLX 0.32 a two-output compiled function with a shared intermediate is
        # ULP-divergent, and this feeds argpartition expert selection, so it
        # stays eager (see docs/laguna-mlxfast-port-correctness.md C1). Only the
        # single-output top-k renormalization below is compiled.
        k = self.top_k
        inds = mx.stop_gradient(
            mx.argpartition(-corrected_scores, kth=k - 1, axis=-1)[..., :k]
        )
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            if _COMPILED_FUSIONS:
                weights = _compiled_topk_normalize(weights)
            else:
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
        # Fused [gate; up] NVFP4 decode bank (built lazily; see
        # ``_prepare_fused_gate_up``). Leading-underscore plain attributes so
        # Module reflection never treats them as checkpoint parameters.
        self._fused_gateup_weight: mx.array | None = None
        self._fused_gateup_scales: mx.array | None = None
        self._fused_gateup_split: int = 0
        self._routed_down_proj: Any | None = None
        self._fusion_ready: bool | None = None

    def _prepare_fused_gate_up(self) -> bool:
        """Build and retain the fused routed gate/up NVFP4 bank.

        Fuses only the exact stock sparse-layer configuration: two bias-free
        NVFP4 group-16 4-bit ``QuantizedSwitchLinear`` banks with identical
        packed shapes. On any mismatch (unquantized modules, a non-NVFP4
        mode, biases, unequal shapes) the block permanently falls back to the
        stock separate-bank path.
        """
        if self._fusion_ready is not None:
            return self._fusion_ready
        gate = self.switch_mlp.gate_proj
        up = self.switch_mlp.up_proj
        down = self.switch_mlp.down_proj

        def _nvfp4_pair(module: Any) -> bool:
            return (
                isinstance(module, QuantizedSwitchLinear)
                and module.mode == "nvfp4"
                and module.group_size == 16
                and module.bits == 4
                and module.get("bias") is None
                and module.get("biases") is None
                and module.weight.ndim == 3
                and module.weight.dtype == mx.uint32
                and module.scales.dtype == mx.uint8
            )

        ready = (
            _nvfp4_pair(gate)
            and _nvfp4_pair(up)
            and gate.weight.shape == up.weight.shape
            and gate.scales.shape == up.scales.shape
            and gate.scales.shape[0] == gate.weight.shape[0]
            and gate.scales.shape[1] == gate.weight.shape[1]
            and gate.weight.shape[2] * 8 == gate.scales.shape[2] * 16
        )
        if ready:
            self._fused_gateup_weight = mx.concatenate([gate.weight, up.weight], axis=1)
            self._fused_gateup_scales = mx.concatenate([gate.scales, up.scales], axis=1)
            self._fused_gateup_split = gate.weight.shape[1]
            self._routed_down_proj = down
            if _LAGUNA_NVFP4_KERNELS:
                # Kernel layouts: pair-interleaved [gate; up] plane (32-row
                # pairs) and the halved group-32 down scale planes.
                from omlx.custom_kernels.laguna_nvfp4 import fast as _ln

                self._kernel_plane = _ln._pair_interleave_fused(
                    gate.weight, up.weight,
                    gate.weight.shape[1])
                self._kernel_plane_scales = _ln._pair_interleave_fused(
                    gate.scales, up.scales, gate.scales.shape[1])
                self._kernel_down = down.weight
                self._kernel_down_scales = _ln.halved_group32_scale_plane(
                    down.scales, [0])
        self._fusion_ready = ready
        return ready

    def _moe_fused_forward(self, x: mx.array, inds: mx.array) -> mx.array:
        """Single-token gather-QMM over the fused [gate; up] NVFP4 bank.

        Mirrors ``SwitchGLU``'s unsorted small-batch path exactly (no
        gather-sort), but with one gather over the row-concatenated bank
        instead of two. ``down_proj`` is the stock module invoked the same way
        ``SwitchGLU`` does.
        """
        expanded = mx.expand_dims(x, (-2, -3))
        gate_up = mx.gather_qmm(
            expanded,
            self._fused_gateup_weight,
            self._fused_gateup_scales,
            None,
            rhs_indices=inds,
            transpose=True,
            group_size=16,
            bits=4,
            mode="nvfp4",
            sorted_indices=False,
        )
        split = self._fused_gateup_split
        x_gate = gate_up[..., :split]
        x_up = gate_up[..., split:]
        return self._routed_down_proj(
            _swiglu(x_gate, x_up), inds, sorted_indices=False
        ).squeeze(-2)

    def __call__(
        self, x: mx.array, residual: mx.array | None = None
    ) -> mx.array:
        inds, scores = self.gate(x)
        if (
            _LAGUNA_NVFP4_KERNELS
            and x.shape[1] == 1
            and inds.size < 64
            and self._prepare_fused_gate_up()
            and getattr(self, "_kernel_plane", None) is not None
        ):
            if os.environ.get("LAG_ROUTED_LOG"):
                print("[ROUTED] engaged", flush=True)
            from omlx.custom_kernels.laguna_nvfp4 import fast as _fn

            if _fn.has_native() and _fn.has_symbol("routed_nvfp4_swiglu_qmv"):
                inds_flat = inds.reshape(-1).astype(mx.uint32)
                scores_flat = scores.reshape(-1).astype(mx.float32)
                act = _fn.routed_nvfp4_swiglu_qmv(
                    x.reshape(-1),
                    self._kernel_plane,
                    self._kernel_plane_scales,
                    inds_flat,
                )  # [8*512]
                if (
                    _fn.has_symbol("routed_nvfp4_down_reduce")
                    and getattr(self, "_kernel_down_scales", None) is not None
                ):
                    if os.environ.get("LAG_ROUTED_LOG"):
                        print("[DOWNRED] calling", flush=True)
                    y = _fn.routed_nvfp4_down_reduce(
                        act,
                        self._kernel_down,
                        self._kernel_down_scales,
                        inds_flat,
                        scores_flat,
                    )  # [2048] — includes the kernel's x2.5
                    if self.routed_scaling_factor != 2.5:
                        y = y * mx.array(
                            self.routed_scaling_factor / 2.5, mx.float32)
                else:
                    y = self._moe_fused_forward(x, inds)
            else:
                y = self._moe_fused_forward(x, inds)
        elif (
            _FUSED_ROUTED_GATE_UP
            and x.shape[1] == 1
            and inds.size < 64
            and self._prepare_fused_gate_up()
        ):
            y = self._moe_fused_forward(x, inds)
        else:
            y = self.switch_mlp(x, inds)
        if _COMPILED_FUSIONS:
            shared = self.shared_expert(x)
            if residual is not None:
                return _compiled_combine_residual_for(self.routed_scaling_factor)(
                    y, scores, shared, residual
                )
            return _compiled_combine_for(self.routed_scaling_factor)(
                y, scores, shared
            )
        y = mx.sum(y * scores[..., None], axis=-2)
        if self.routed_scaling_factor != 1.0:
            y = y * self.routed_scaling_factor
        moe = y + self.shared_expert(x)
        return residual + moe if residual is not None else moe


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

        # NVFP4 requantized attention bank (lazy; built on first decode call
        # with the kernels enabled). mirrors lagunaNativeAffineWeight.
        self._nvfp4_bank = None

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

    def _prepare_nvfp4_bank(self) -> bool:
        """Quantize q/k/v/o to NVFP4 group-16 and build the fused QKV bank."""
        if self._nvfp4_bank is not None:
            return self._nvfp4_bank is not False
        if not _LAGUNA_NVFP4_ATTN or not _LAGUNA_NVFP4_KERNELS:
            self._nvfp4_bank = False
            return False
        try:
            from omlx.custom_kernels.laguna_nvfp4 import fast as _laguna_nvfp4

            if not _laguna_nvfp4.has_native():
                self._nvfp4_bank = False
                return False
            q = self.q_proj.weight.astype(mx.float32)
            k = self.k_proj.weight.astype(mx.float32)
            v = self.v_proj.weight.astype(mx.float32)
            o = self.o_proj.weight.astype(mx.float32)
            qc, qs = mx.quantize(q, group_size=16, bits=4, mode="nvfp4")
            kc, ks = mx.quantize(k, group_size=16, bits=4, mode="nvfp4")
            vc, vs = mx.quantize(v, group_size=16, bits=4, mode="nvfp4")
            oc, os_ = mx.quantize(o, group_size=16, bits=4, mode="nvfp4")
            # fused QKV bank: [rows=(heads+2kv)*128, hidden/8] codes, scales
            codes = mx.concatenate(
                [qc.view(mx.uint8).reshape(qc.shape[0], -1),
                 kc.view(mx.uint8).reshape(kc.shape[0], -1),
                 vc.view(mx.uint8).reshape(vc.shape[0], -1)], axis=0)
            scales = mx.concatenate([qs, ks, vs], axis=0)
            self._nvfp4_bank = {
                "qkv_codes": codes, "qkv_scales": scales,
                "o_codes": oc.view(mx.uint8).reshape(oc.shape[0], -1),
                "o_codes_raw": oc,
                "o_scales": os_,
            }
            return True
        except Exception:
            self._nvfp4_bank = False
            return False

    def _fused_attn_decode(self, x: mx.array, cache: Any | None) -> mx.array | None:
        """Fused sliding-ring decode attention (challenge
        lagunaSlidingFusedAttention). Returns the post-o_proj attended output
        for a single-token sliding-ring decode step, or None to fall back to
        the stock path.

        Engaged only when all of:
          - kernels + NVFP4 attention bank + fused-attn switch are ON,
          - the native ``sliding_fused_attn_ring`` symbol exists,
          - single token (B=1, L=1) on a sliding layer,
          - the layer cache is a RotatingKVCache at steady ring capacity
            (buffer shape [1, kv, window, head] and offset >= window),
          - the position is covered by the RoPE angle atlas.

        The kernel performs QK RMSNorm + RoPE + in-place K/V ring write +
        online-softmax attention in ONE Metal dispatch. After it returns, the
        new K/V slot the kernel wrote is mirrored onto the real cache slot and
        the clock advances exactly as the stock update_in_place would.
        """
        if os.environ.get("LAG_FUSED_LOG"):
            print(f"[g] on={_LAGUNA_NVFP4_KERNELS} fused={_LAGUNA_FUSED_ATTN} sli={self.is_sliding} ctype={type(cache).__name__ if cache is not None else None}", flush=True)
        if not (_LAGUNA_NVFP4_KERNELS and _LAGUNA_FUSED_ATTN):
            return None
        if self.sink is not None:
            return None
        if not self.is_sliding or not isinstance(cache, RotatingKVCache):
            return None
        bsz, seq_len, _ = x.shape
        if bsz != 1 or seq_len != 1:
            return None
        if os.environ.get("LAGUNA_FUSED_DBG"):
            print(f"[g2] kshape={None if cache.keys is None else cache.keys.shape} max={cache.max_size} off={cache.offset} keep={cache.keep}", flush=True)
        if cache.keys is None or cache.keys.shape[2] != cache.max_size:
            return None
        if cache.offset < cache.max_size:
            return None
        if cache.keep:
            return None
        try:
            from omlx.custom_kernels.laguna_nvfp4 import fast as _laguna_nvfp4

            if os.environ.get("LAGUNA_FUSED_DBG"):
                print("[try] sym=", _laguna_nvfp4.has_symbol("sliding_fused_attn_ring"), flush=True)
            if not _laguna_nvfp4.has_symbol("sliding_fused_attn_ring"):
                return None
            atlas = _get_laguna_rope_atlas(self.rope, None)
            if os.environ.get("LAGUNA_FUSED_DBG"):
                print("[try] atlas.sliding=", None if atlas.sliding is None else atlas.sliding.shape, flush=True)
            if atlas.sliding is None:
                return None
            pos = int(cache.offset)
            if os.environ.get("LAGUNA_FUSED_DBG"):
                print("[try] pos=", pos, "atlas_len=", None if atlas.sliding is None else atlas.sliding.shape[2], flush=True)
            if pos >= int(atlas.sliding.shape[2]):
                return None
            angles = mx.array(atlas.sliding[0, 0, pos, :], mx.float32)

            # Prefer the NVFP4 requantized QKV bank when active (matches the
            # stock qkv kernel the challenge's decode uses); otherwise the
            # bf16 projections.
            hidden = x.reshape(-1)
            if self._prepare_nvfp4_bank():
                heads4 = self.n_heads if self.n_heads in (48, 64) else 64
                projected = _laguna_nvfp4.decode_nvfp4_qkv_r1(
                    hidden, self._nvfp4_bank["qkv_codes"],
                    self._nvfp4_bank["qkv_scales"], heads4,
                )
                q_split = self.n_heads * self.head_dim
                k_end = q_split + self.n_kv_heads * self.head_dim
                raw_q = projected[:q_split].reshape(-1)
                raw_k = projected[q_split:k_end].reshape(-1)
                raw_v = projected[k_end:].reshape(-1)
            else:
                raw_q = self.q_proj(x).reshape(-1)
                raw_k = self.k_proj(x).reshape(-1)
                raw_v = self.v_proj(x).reshape(-1)
            # Pass a free reshape view of the ring backing (the cache keeps a
            # row-major [1, kv, window, head] buffer, so [kv, window, head]
            # is a zero-copy view). The fused kernel writes the new K/V slot
            # in-place directly into this shared backing — no copy, no
            # mirror needed; we only advance the clock.
            kring = cache.keys.reshape(
                self.n_kv_heads, cache.max_size, self.head_dim
            )
            vring = cache.values.reshape(
                self.n_kv_heads, cache.max_size, self.head_dim
            )
            widx = int(cache._idx)
            if widx == cache.max_size:
                widx = 0
            if os.environ.get("LAGUNA_FUSED_DBG"):
                print("[try] before kernel call", flush=True)
            y = _laguna_nvfp4.sliding_fused_attn_ring(
                raw_q, raw_k, raw_v, self.q_norm.weight, self.k_norm.weight,
                angles, kring, vring,
                mx.array([widx], mx.uint32),
                mx.array([self.scale], mx.float32),
            )
            # The kernel wrote the new K/V into the ring in place. Advance
            # the clock exactly as update_in_place(tokenCount: 1) would.
            cache.offset += 1
            cache._idx = widx + 1
            if cache._idx == cache.max_size:
                cache._idx = cache.keep
            y2 = y.reshape(bsz, seq_len, self.n_heads, self.head_dim)
            if self.gating:
                gate = self.g_proj(hidden.reshape(bsz, seq_len, -1))
                if _COMPILED_FUSIONS:
                    gate = _compiled_softplus_gate(gate)
                else:
                    gate = nn.softplus(gate.astype(mx.float32)).astype(y.dtype)
                if self.gate_per_head:
                    y2 = y2 * gate.reshape(bsz, seq_len, self.n_heads, 1)
                else:
                    y2 = y2 * gate.reshape(bsz, seq_len, -1).reshape(
                        bsz, seq_len, self.n_heads, self.head_dim
                    )
            _LAGUNA_FUSED_CALLS[0] += 1
            if os.environ.get("LAG_FUSED_LOG"):
                print(f"[FUSED+1] total={_LAGUNA_FUSED_CALLS[0]} layer={getattr(self,'is_sliding',None)}", flush=True)
            return self.o_proj(y2.reshape(bsz, seq_len, -1))
        except Exception as _e:
            if os.environ.get("LAGUNA_FUSED_DBG"):
                import traceback; traceback.print_exc()
                print("[try] EXC:", type(_e).__name__, _e, flush=True)
            return None

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        bsz, seq_len, _ = x.shape

        from omlx.custom_kernels.laguna_nvfp4 import fast as _laguna_nvfp4

        # ---- fused decode attention (opt-in) ------------------------------
        # Single-token sliding-ring decode: ONE kernel replaces the
        # [QK-norm + RoPE] -> [K/V ring write] -> [sdpa_vector] chain. The
        # kernel writes the new K/V into the ring in-place; we mirror that
        # write onto the real cache slot and advance the clock without a
        # second update_and_fetch. Engaged only at steady ring (buffer at
        # window capacity, offset >= window) with the NVFP4 bank active;
        # everything else falls back to the stock path below.
        fused_attn_out = self._fused_attn_decode(x, cache)
        if fused_attn_out is not None:
            return fused_attn_out

        # NVFP4-decode QKV path (opt-in): single-token decode dispatches the
        # fused q/k/v NVFP4 kernel against the requantized bank.
        use_nvfp4_qkv = (
            bsz == 1
            and seq_len == 1
            and self._prepare_nvfp4_bank()
            and _laguna_nvfp4.has_symbol("decode_nvfp4_qkv_r1")
        )
        if use_nvfp4_qkv:
            hidden = x.reshape(-1)
            heads4 = self.n_heads if self.n_heads in (48, 64) else 64
            projected = _laguna_nvfp4.decode_nvfp4_qkv_r1(
                hidden, self._nvfp4_bank["qkv_codes"],
                self._nvfp4_bank["qkv_scales"], heads4,
            )
            q_split = self.n_heads * self.head_dim
            k_end = q_split + self.n_kv_heads * self.head_dim
            queries = projected[:q_split].reshape(
                bsz, seq_len, self.n_heads, self.head_dim)
            keys = projected[q_split:k_end].reshape(
                bsz, seq_len, self.n_kv_heads, self.head_dim)
            values = projected[k_end:].reshape(
                bsz, seq_len, self.n_kv_heads, self.head_dim)
            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(keys).transpose(0, 2, 1, 3)
            values = values.transpose(0, 2, 1, 3)
        else:
            queries, keys, values = (
                self.q_proj(x), self.k_proj(x), self.v_proj(x))
            queries = self.q_norm(
                queries.reshape(bsz, seq_len, self.n_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                keys.reshape(bsz, seq_len, self.n_kv_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            values = values.reshape(
                bsz, seq_len, self.n_kv_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)

            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Resolved through the module, not bound at import. This module is
        # imported from maybe_apply_pre_load_patches, before the engine installs
        # the TurboQuant dispatcher, and the dispatcher's rebinding sweep only
        # covers mlx_lm/mlx_vlm model modules, so an import-time binding here
        # would never see TurboQuant at all (issue #2372).
        output = mlx_lm_base.scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.sink,
        )
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)

        # Fused gated o_proj (opt-in, challenge lagunaOProjAct). Replaces the
        # [gate softplus] * attention + o_proj tail with ONE kernel when the
        # NVFP4 bank is active, shapes match (per-head gate), and the symbol
        # exists. The kernel's bf16 per-element gate product reproduces the
        # stock decode path byte-for-byte (test_oproj_act_bit_exact) for the
        # nvfp4-quantized o_proj — which is already the numerics of this
        # opt-in bank path.
        if (use_nvfp4_qkv and self.gating and self.gate_per_head
                and self.o_proj.weight.ndim == 2):
            try:
                from omlx.custom_kernels.laguna_nvfp4 import fast as _fn

                if _fn.has_native() and _fn.has_symbol("oproj_act"):
                    in_vec = self.n_heads * self.head_dim
                    oc = self._nvfp4_bank.get("o_codes_raw")
                    os_ = self._nvfp4_bank.get("o_scales")
                    if (oc is not None and os_ is not None
                            and oc.shape == (self.o_proj.weight.shape[0], in_vec // 8)
                            and os_.shape == (self.o_proj.weight.shape[0], in_vec // 16)):
                        if _COMPILED_FUSIONS:
                            gv = _compiled_softplus_gate(self.g_proj(x))
                        else:
                            gv = nn.softplus(
                                self.g_proj(x).astype(mx.float32)
                            ).astype(output.dtype)
                        gv = gv.reshape(-1).astype(mx.bfloat16)
                        out = _fn.oproj_act(
                            output.reshape(-1).astype(mx.bfloat16),
                            gv, oc, os_, self.n_heads,
                        )
                        return out.reshape(bsz, seq_len, -1)
            except Exception:
                pass

        if self.gating:
            if _COMPILED_FUSIONS:
                gate = _compiled_softplus_gate(self.g_proj(x))
            else:
                gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(
                    output.dtype
                )
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
        mlp_input = self.post_attention_layernorm(h)
        if isinstance(self.mlp, LagunaSparseMoeBlock):
            # Fold the decoder residual add into the (compiled) sparse combine.
            return self.mlp(mlp_input, residual=h)
        return h + self.mlp(mlp_input)


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
        # LM-head int5 prune planes (challenge buildInt5Planes), built lazily
        # on the first decode call when the kernel path is enabled. None when
        # the head does not fit int5 (declines to the stock head).
        self._lm_planes = "uninit"

    def _prepare_lm_planes(self):
        """Build the int5 prune planes for the untied lm_head, once."""
        if self._lm_planes != "uninit":
            return self._lm_planes
        self._lm_planes = None
        if not (_LAGUNA_NVFP4_KERNELS and not self.args.tie_word_embeddings):
            return None
        try:
            from omlx.custom_kernels.laguna_nvfp4 import fast as _fn

            if _fn.has_native() and _fn.has_symbol("lm_head_prune"):
                lo, hi, sd = _fn.build_int5_planes(self.lm_head.weight)
                if lo is not None:
                    self._lm_planes = (lo, hi, sd)
        except Exception:
            pass
        return self._lm_planes

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        # Decode-only LM-head int5 prune: for a single output row with the
        # planes built, run the prune pipeline (coarse argmax + exact winner)
        # instead of the full 100352-wide bf16 head matmul. Args-match the
        # stock head; anything odd falls back to self.lm_head(out).
        if out.shape[0] == 1 and out.shape[1] == 1 and self._prepare_lm_planes() is not None:
            try:
                from omlx.custom_kernels.laguna_nvfp4 import fast as _fn

                lo, hi, sd = self._lm_planes
                lg = _fn.lm_head_prune(
                    out.reshape(-1), lo, hi, sd, self.lm_head.weight
                )
                return lg.reshape(1, 1, -1)
            except Exception:
                pass
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

            # Remap the router weight and all quantization sidecars from
            # ``gate.<suffix>`` to ``gate.proj.<suffix>``.  Quantized
            # checkpoints produced by oQ / mlx-vlm carry ``gate.scales`` and
            # ``gate.biases`` alongside ``gate.weight``; remapping only
            # ``.weight`` leaves the sidecars orphaned and triggers
            # ``ValueError: Received N parameters not in model`` during
            # strict weight loading.
            for suffix in ("weight", "scales", "biases"):
                old_key = f"{prefix}.gate.{suffix}"
                if old_key in weights:
                    weights[f"{prefix}.gate.proj.{suffix}"] = weights.pop(old_key)

            # ``e_score_correction_bias`` appears under two checkpoint
            # conventions: the legacy ``experts.e_score_correction_bias``
            # path (referenced in the original upstream PR) and the bare
            # ``mlp.e_score_correction_bias`` path used by the published
            # pipenetwork/Laguna-S-2.1 MLX conversions.  Map both to the
            # module-tree path the model expects.
            for bias_key in (
                f"{prefix}.experts.e_score_correction_bias",
                f"{prefix}.e_score_correction_bias",
            ):
                if bias_key in weights:
                    weights[f"{prefix}.gate.e_score_correction_bias"] = weights.pop(
                        bias_key
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
