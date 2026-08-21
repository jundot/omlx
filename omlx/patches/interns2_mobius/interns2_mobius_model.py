# Copyright © 2025 Apple Inc.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update
from .rope_utils import initialize_rope
from .switch_layers import SwitchLinear


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    num_blocks: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    full_attention_interval: int = 4
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params):
        # The checkpoint nests the language-model settings under `text_config`
        # and stores rope settings under `rope_parameters`.
        if "text_config" in params:
            params = params["text_config"]
        rope = params.get("rope_parameters", {})
        merged = dict(params)
        merged.setdefault("rope_theta", rope.get("rope_theta", 1e7))
        merged.setdefault(
            "partial_rotary_factor", rope.get("partial_rotary_factor", 1.0)
        )
        return super().from_dict(merged)


class InternS2MobiusRMSNorm(nn.Module):
    """Zero-centered RMSNorm: HF stores `weight`, applies `x * (1 + weight)`."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros(dims)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class InternS2MobiusRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        return (x * nn.silu(gate.astype(mx.float32))).astype(hidden_states.dtype)


class InternS2MobiusAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = InternS2MobiusRMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = InternS2MobiusRMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, gate = mx.split(
            self.q_proj(x).reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
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
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * mx.sigmoid(gate))


class InternS2MobiusGatedDeltaNet(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)
        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = InternS2MobiusRMSNormGated(self.head_v_dim, eps=args.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = x.shape

        mixed_qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype
            )

        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)

        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        # Match the HF kernel's `use_qk_l2norm_in_kernel=True` plus 1/sqrt(d)
        # query scaling: rms_norm(x) == l2norm(x) * sqrt(d), so these factors
        # reduce to l2norm(q)/sqrt(d) and l2norm(k).
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        entry_state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            entry_state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


def _sort_by_expert(x: mx.array, indices: mx.array):
    """Sort tokens by their expert so each expert's rows are contiguous.

    Mirrors switch_layers' own private helper: a contiguous per-expert layout
    lets `gather_qmm(sorted_indices=True)` read each expert's weights once.
    """
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _unsort_by_expert(x: mx.array, inv_order: mx.array, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class FusedSwitchGLU(nn.Module):
    """SwitchGLU with gate and up stacked into a single switch projection.

    Decode is kernel-launch bound, so the two skinny gather_qmm calls cost far
    more in dispatch than the one twice-as-wide call that replaces them.
    """

    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__()
        self.gate_up_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=False
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=False)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _sort_by_expert(x, indices)

        gate, up = mx.split(
            self.gate_up_proj(x, idx, sorted_indices=do_sort), 2, axis=-1
        )
        x = self.down_proj(swiglu(gate, up), idx, sorted_indices=do_sort)

        if do_sort:
            x = _unsort_by_expert(x, inv_order, indices.shape)

        return x.squeeze(-2)


_ROUTER_TOPK_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    const device T* row_logits = logits + row * (size_t)N;

    threadgroup float red_val[TG];
    threadgroup uint red_idx[TG];
    threadgroup float top_val[K];
    threadgroup uint top_idx[K];

    // Select the top K by K masked max-reductions. K is 8 here, so this beats
    // sorting 2560 candidates, and it needs no scratch beyond the winners.
    for (uint r = 0; r < K; ++r) {
        float best = -INFINITY;
        uint best_i = 0;
        for (uint i = tid; i < N; i += TG) {
            bool taken = false;
            for (uint j = 0; j < r; ++j) {
                taken = taken || (top_idx[j] == i);
            }
            float v = static_cast<float>(row_logits[i]);
            if (!taken && v > best) {
                best = v;
                best_i = i;
            }
        }
        red_val[tid] = best;
        red_idx[tid] = best_i;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = TG / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float o = red_val[tid + s];
                uint oi = red_idx[tid + s];
                // Lowest index wins a tie, so the result never depends on how
                // the reduction happens to be scheduled.
                if (o > red_val[tid] || (o == red_val[tid] && oi < red_idx[tid])) {
                    red_val[tid] = o;
                    red_idx[tid] = oi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            top_val[r] = red_val[0];
            top_idx[r] = red_idx[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Softmax over the K selected logits. Round 0 took the global max, so
    // top_val[0] is the shift that keeps the exponentials in range.
    if (tid == 0) {
        float total = 0.0f;
        for (uint j = 0; j < K; ++j) {
            total += metal::exp(top_val[j] - top_val[0]);
        }
        for (uint j = 0; j < K; ++j) {
            inds[row * K + j] = top_idx[j];
            scores[row * K + j] =
                static_cast<O>(metal::exp(top_val[j] - top_val[0]) / total);
        }
    }
"""

_ROUTER_TOPK_KERNEL = None
_ROUTER_TOPK_THREADS = 256


def _router_topk_kernel(logits: mx.array, k: int) -> tuple[mx.array, mx.array]:
    global _ROUTER_TOPK_KERNEL
    if _ROUTER_TOPK_KERNEL is None:
        _ROUTER_TOPK_KERNEL = mx.fast.metal_kernel(
            name="router_topk",
            input_names=["logits"],
            output_names=["inds", "scores"],
            source=_ROUTER_TOPK_SOURCE,
        )

    rows, n = logits.shape
    return _ROUTER_TOPK_KERNEL(
        inputs=[logits],
        template=[
            ("T", logits.dtype),
            ("O", logits.dtype),
            ("N", n),
            ("K", k),
            ("TG", _ROUTER_TOPK_THREADS),
        ],
        grid=(_ROUTER_TOPK_THREADS * rows, 1, 1),
        threadgroup=(_ROUTER_TOPK_THREADS, 1, 1),
        output_shapes=[(rows, k), (rows, k)],
        output_dtypes=[mx.uint32, logits.dtype],
    )


def _router_topk_ops(logits: mx.array, k: int) -> tuple[mx.array, mx.array]:
    idx = mx.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
    top = mx.take_along_axis(logits, idx, axis=-1)
    # Softmax over the selected logits equals renormalising a full softmax,
    # since softmax is monotonic and so picks the same k. Emitted in the logits'
    # dtype (see router_topk).
    scores = mx.softmax(top, axis=-1, precise=True).astype(logits.dtype)
    return idx.astype(mx.uint32), scores


def router_topk(
    logits: mx.array, k: int, use_kernel: bool = True
) -> tuple[mx.array, mx.array]:
    """Top-k experts and their softmax weights.

    Replaces an upcast, argpartition over all experts, gather and softmax --
    four kernels per layer on a decode step that is bound by kernel launches.
    Softmax over the selected logits equals renormalising a softmax over all of
    them, because softmax is monotonic and so picks the same k.

    Experts tying on the boundary logit may be picked differently than
    argpartition picks them, which is free to break ties any way it likes.
    Tied logits carry equal weight, so either choice is as correct.

    The Metal kernel is selected only under the same guard the recurrent layers
    use (`gated_delta.py`), falling back to pure MLX ops on CUDA/CPU. Scores are
    emitted in the logits' dtype (bf16 here); HF routes in fp32, a deliberate
    deviation measured token-identical. A fully `-inf` logit row would yield NaN
    scores, but gate logits are never `-inf` for this model.
    """
    assert logits.ndim == 2, "router_topk expects 2-D [tokens, experts] logits"
    if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return _router_topk_ops(logits, k)
    return _router_topk_kernel(logits, k)


class InternS2MobiusMetaMoeBlock(nn.Module):
    """Routed experts shared globally across layers (one of `num_blocks` banks).

    The bank also holds the shared expert of every layer that uses it, past the
    routed experts, so a layer's shared expert is just a top-(k+1)th expert that
    is always selected. Its sigmoid gate is the combining weight, which makes
    the fold exact: the weighted sum over k+1 experts is the routed sum plus the
    gated shared output.
    """

    def __init__(self, args: ModelArgs, num_shared: int):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.num_experts = args.num_experts
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = FusedSwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts + num_shared
        )

    def __call__(
        self, x: mx.array, shared_slot: int, shared_scale: mx.array
    ) -> mx.array:
        inds, scores = router_topk(self.gate(x), self.top_k)

        shared_ind = mx.full(
            inds.shape[:-1] + (1,), self.num_experts + shared_slot, inds.dtype
        )
        inds = mx.concatenate([inds, shared_ind], axis=-1)
        scores = mx.concatenate([scores, shared_scale], axis=-1)

        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)


class InternS2MobiusSharedExpertGate(nn.Module):
    """Sigmoid gate of a layer's shared expert.

    The expert's own weights live in the layer's routed bank, so only the gate
    is per-layer here.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.sigmoid(self.shared_expert_gate(x))


class InternS2MobiusDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = InternS2MobiusGatedDeltaNet(args)
        else:
            self.self_attn = InternS2MobiusAttention(args)
        self.shared_slot = layer_idx // args.num_blocks
        # `mlp` holds a shared-expert *gate*, not an MLP: the name matches the
        # checkpoint keys `model.layers.N.mlp.shared_expert_gate`.
        self.mlp = InternS2MobiusSharedExpertGate(args)
        self.input_layernorm = InternS2MobiusRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_attention_layernorm = InternS2MobiusRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        meta_block: InternS2MobiusMetaMoeBlock,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r

        normed = self.post_attention_layernorm(h)
        B, S, D = normed.shape
        normed_2d = normed.reshape(-1, D)
        out = meta_block(normed_2d, self.shared_slot, self.mlp(normed_2d))
        return h + out.reshape(B, S, D)


class InternS2MobiusModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_blocks = args.num_blocks
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            InternS2MobiusDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.meta_mlp = [
            InternS2MobiusMetaMoeBlock(
                args, len(range(b, args.num_hidden_layers, args.num_blocks))
            )
            for b in range(args.num_blocks)
        ]
        self.norm = InternS2MobiusRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # A representative cache index per layer type, for building the masks.
        # None when the config has no layer of that type.
        self.ssm_idx = next((i for i, l in enumerate(self.layers) if l.is_linear), None)
        self.fa_idx = next(
            (i for i, l in enumerate(self.layers) if not l.is_linear), None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = (
            create_attention_mask(h, cache[self.fa_idx])
            if self.fa_idx is not None
            else None
        )
        ssm_mask = (
            create_ssm_mask(h, cache[self.ssm_idx])
            if self.ssm_idx is not None
            else None
        )

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, self.meta_mlp[i % self.num_blocks], mask=mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = InternS2MobiusModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        hidden = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        # Rewritten in place rather than into a fresh dict: the expert weights
        # are assembled here, and the caller's dict would otherwise keep every
        # source array alive alongside its copy for the whole pass, roughly
        # doubling peak memory at load.
        for k in [k for k in weights if "visual" in k or k.startswith("mtp.")]:
            del weights[k]
        for k in [k for k in weights if k.startswith("model.language_model.")]:
            weights[k.replace("model.language_model.", "model.", 1)] = weights.pop(k)

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        def take(prefix):
            """Pop a projection's tensors as {part: array}; empty if absent."""
            return {
                part: weights.pop(f"{prefix}.{part}")
                for part in ("weight", "scales", "biases")
                if f"{prefix}.{part}" in weights
            }

        def stack_gate_up(prefix):
            """Stack gate_proj and up_proj into one gate_up_proj.

            Exact for quantized tensors too: each output row carries its own
            scales/biases groups, so concatenating rows never requantizes.
            """
            gate = take(f"{prefix}.gate_proj")
            if not gate:
                return
            up = take(f"{prefix}.up_proj")
            row_axis = gate["weight"].ndim - 2
            for part, g in gate.items():
                fused = mx.concatenate([g, up[part]], axis=row_axis)
                # Materialize each projection as it is built. Left lazy, every
                # bank's sources and results stay alive until the final eval,
                # which multiplies peak memory over the expert weights.
                mx.eval(fused)
                weights[f"{prefix}.gate_up_proj.{part}"] = fused

        # HF stores the routed gate/up already fused, which is the layout
        # FusedSwitchGLU wants, so that case is a rename; checkpoints converted
        # before this layout store the pair split and get stacked.
        for bank in range(self.args.num_blocks):
            prefix = f"model.meta_mlp.{bank}"
            for name in ("gate_up_proj", "down_proj"):
                v = weights.pop(f"{prefix}.experts.{name}", None)
                if v is not None:
                    weights[f"{prefix}.switch_mlp.{name}.weight"] = v
            stack_gate_up(f"{prefix}.switch_mlp")

        for i in range(self.args.num_hidden_layers):
            stack_gate_up(f"model.layers.{i}.mlp.shared_expert")

        # Append each layer's shared expert to its bank as an extra expert:
        # layer i takes slot i // num_blocks of bank i % num_blocks, matching
        # the indices InternS2MobiusMetaMoeBlock selects.
        for bank in range(self.args.num_blocks):
            prefix = f"model.meta_mlp.{bank}.switch_mlp"
            layers = range(bank, self.args.num_hidden_layers, self.args.num_blocks)
            for name in ("gate_up_proj", "down_proj"):
                extra = [
                    take(f"model.layers.{i}.mlp.shared_expert.{name}") for i in layers
                ]
                if not any(extra):
                    continue
                assert all(extra), f"bank {bank}: shared {name} only partly present"
                for part in extra[0]:
                    stacked = mx.concatenate(
                        [weights[f"{prefix}.{name}.{part}"]]
                        + [e[part][None] for e in extra],
                        axis=0,
                    )
                    # Materialize per projection so its sources are released
                    # before the next bank is built; leaving every bank pending
                    # until the final eval doubles peak memory over the experts.
                    mx.eval(stacked)
                    weights[f"{prefix}.{name}.{part}"] = stacked

        # The (1 + weight) offset is applied at runtime by InternS2MobiusRMSNorm,
        # not folded here, so sanitize() stays idempotent across convert + load.
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
