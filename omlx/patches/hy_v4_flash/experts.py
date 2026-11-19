# SPDX-License-Identifier: Apache-2.0
"""HY-V4-Flash dense and latent MoE sublayers (text inference only).

The HF expert banks are fused parameters. In the MLX checkpoint their paths
name modules: a BF16 bank has a weight leaf, and an oQh bank also carries
its own rotation signs and group-affine sidecars. Nothing else is quantized.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from omlx.custom_kernels.oqh import fast

from .quant_io import SUPPORTED_BITS

_BLOCK = 256
_GROUP = 128
_BITS = 8
_SPEC = {"bits": _BITS, "block": _BLOCK, "group_size": _GROUP}


def _packed_paths(args):
    meta = getattr(args, "omlx_oqh", None)
    if meta is None:
        return {}
    if (
        not isinstance(meta, dict)
        or type(meta.get("version")) is not int
        or meta["version"] != 1
    ):
        raise ValueError("HY-V4-Flash requires omlx_oqh version 1")
    entries = meta.get("tensors")
    if not isinstance(entries, dict):
        raise ValueError(
            "omlx_oqh.tensors must map routed bank paths to format entries"
        )
    allowed = {
        f"model.layers.{i}.mlp.experts.{name}"
        for i, kind in enumerate(args.mlp_layer_types)
        if kind == "sparse"
        for name in ("gate_up_proj", "down_proj")
    }
    for path, spec in entries.items():
        if path not in allowed:
            raise ValueError(f"unsupported HY-V4-Flash oQh tensor path: {path}")
        if (
            not isinstance(spec, dict)
            or set(spec) != set(_SPEC)
            or not all(type(v) is int for v in spec.values())
            or spec["bits"] not in SUPPORTED_BITS
            or {k: v for k, v in spec.items() if k != "bits"}
            != {k: v for k, v in _SPEC.items() if k != "bits"}
        ):
            raise ValueError(
                f"{path}: expected oQh {_SPEC} with bits in {SUPPORTED_BITS}, got {spec!r}"
            )
    return entries


def _placeholder(shape, value=0, dtype=mx.bfloat16):
    # The real banks have billions of elements. Even an evaluated placeholder
    # only needs one scalar until the loader replaces it with its weight.
    return mx.broadcast_to(mx.array(value, dtype=dtype), shape)


class _Linear(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weight = _placeholder((outputs, inputs))

    def __call__(self, x):
        return x @ self.weight.T


class _RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = _placeholder((width,), 1)
        self.eps = eps

    def __call__(self, x):
        # HF rounds the unweighted FP32 normalization to the input dtype
        # before multiplying by the learned scale.
        return mx.fast.rms_norm(x, None, self.eps) * self.weight


@mx.compile
def _swiglu(gate, up):
    return nn.silu(gate.astype(mx.float32)).astype(gate.dtype) * up


class _MLP(nn.Module):
    def __init__(self, args, intermediate, residual):
        super().__init__()
        width = args.hidden_size
        self.gate_proj = _Linear(width, intermediate)
        self.up_proj = _Linear(width, intermediate)
        self.down_proj = _Linear(intermediate, width)
        self.post_fc2_norm = _RMSNorm(width, args.rms_norm_eps) if residual else None

    def __call__(self, x):
        x = self.down_proj(_swiglu(self.gate_proj(x), self.up_proj(x)))
        return self.post_fc2_norm(x) if self.post_fc2_norm is not None else x


def _top_indices(scores, k):
    return mx.argpartition(scores, scores.shape[-1] - k, axis=-1)[..., -k:]


class _Router(nn.Module):
    def __init__(self, args):
        super().__init__()
        experts = args.n_routed_experts
        self.groups = args.n_group
        self.top_groups = args.topk_group
        self.top_k = args.num_experts_per_tok
        if not 1 <= self.top_groups <= self.groups <= experts or experts % self.groups:
            raise ValueError("invalid routed expert group configuration")
        if self.top_groups < self.groups and experts // self.groups < 2:
            raise ValueError("group routing requires at least two experts per group")
        if not 1 <= self.top_k <= self.top_groups * (experts // self.groups):
            raise ValueError(
                "num_experts_per_tok exceeds the available grouped experts"
            )
        self.norm_topk_prob = args.norm_topk_prob
        self.factor = args.routed_scaling_factor
        self.weight = _placeholder((experts, args.hidden_size))
        self.e_score_correction_bias = _placeholder((experts,), dtype=mx.float32)

    def __call__(self, x):
        x = x.reshape(-1, self.weight.shape[-1]).astype(mx.float32)
        logits = x @ self.weight.astype(mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.e_score_correction_bias
        if self.top_groups < self.groups:
            grouped = choice.reshape(choice.shape[0], self.groups, -1)
            best_two = mx.topk(grouped, 2, axis=-1).sum(axis=-1)
            groups = _top_indices(best_two, self.top_groups)
            keep = mx.put_along_axis(
                mx.zeros_like(best_two), groups, mx.array(1.0), axis=-1
            )
            choice = mx.where(keep[..., None], grouped, -mx.inf).reshape(choice.shape)
        # HF eager accumulates contributions in ascending expert-id order.
        indices = mx.sort(_top_indices(choice, self.top_k), axis=-1)
        weights = mx.take_along_axis(scores, indices, axis=-1)  # unbiased scores
        if self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return logits, weights * self.factor, indices


def rotate_activation(x: mx.array, signs: mx.array) -> mx.array:
    """Apply H_256 diag(signs) to each 256-wide input block, on device."""
    width = x.shape[-1]
    if width % _BLOCK or signs.shape != (width // _BLOCK, _BLOCK):
        raise ValueError(
            f"invalid oQh rotation: input width {width}, signs {signs.shape}"
        )
    if fast.has_native() and x.dtype in (mx.float16, mx.bfloat16):
        return fast.rotate(x, signs, _BLOCK)
    # Stock MLX also has a native normalized Walsh-Hadamard primitive.
    blocks = x.reshape(*x.shape[:-1], -1, _BLOCK).astype(mx.float32) * signs
    return mx.hadamard_transform(blocks).reshape(x.shape).astype(x.dtype)


class _ExpertProjection(nn.Module):
    """One [experts, outputs, inputs] matrix bank; never expand it per token."""

    def __init__(self, experts, inputs, outputs, path, packed):
        super().__init__()
        self._bits = packed["bits"] if packed else _BITS
        if packed:
            if inputs <= 0 or inputs % _BLOCK:
                raise ValueError(
                    f"{path}: oQh input width {inputs} must be a multiple of {_BLOCK}"
                )
            self.weight = _placeholder(
                (experts, outputs, inputs * self._bits // 32), dtype=mx.uint32
            )
            self.oqh_scales = _placeholder((experts, outputs, inputs // _GROUP))
            self.oqh_biases = _placeholder((experts, outputs, inputs // _GROUP))
            self.oqh_signs = _placeholder((inputs // _BLOCK, _BLOCK), 1)
        else:
            self.weight = _placeholder((experts, outputs, inputs))

    def __call__(self, x, indices):
        # Cast activations, never a multi-GiB bank, for a mixed-dtype caller.
        if "oqh_signs" not in self:
            return mx.gather_mm(
                x.astype(self.weight.dtype),
                self.weight.swapaxes(-1, -2),
                rhs_indices=indices,
            ).astype(x.dtype)
        rotated = rotate_activation(x.astype(self.oqh_scales.dtype), self.oqh_signs)
        return mx.gather_qmm(
            rotated,
            self.weight,
            self.oqh_scales,
            self.oqh_biases,
            rhs_indices=indices,
            transpose=True,
            group_size=_GROUP,
            bits=self._bits,
            mode="affine",
        ).astype(x.dtype)


@mx.compile
def _combine(current, weights):
    # HF index_add_ casts each weighted contribution to the hidden dtype
    # before adding it to the accumulator, which itself has that dtype.
    terms = (current * weights[..., None]).astype(current.dtype)
    total = terms[:, 0]
    for j in range(1, terms.shape[1]):
        total = total + terms[:, j]
    return total


class _Experts(nn.Module):
    def __init__(self, args, block, width, packed_paths):
        super().__init__()
        mid = args.moe_intermediate_size
        prefix = f"model.layers.{block}.mlp.experts"

        def bank(name, inputs, outputs):
            path = f"{prefix}.{name}"
            return _ExpertProjection(
                args.n_routed_experts, inputs, outputs, path, packed_paths.get(path)
            )

        self.gate_up_proj = bank("gate_up_proj", width, 2 * mid)
        self.down_proj = bank("down_proj", mid, width)
        self.limit = args.swiglu_limit

    def __call__(self, x, indices, weights):
        gate_up = self.gate_up_proj(x[:, None, None, :], indices)
        gate, up = mx.split(gate_up, 2, axis=-1)
        act = _swiglu(
            mx.minimum(gate, self.limit), mx.clip(up, -self.limit, self.limit)
        )
        current = self.down_proj(act, indices).squeeze(-2)
        return _combine(current, weights)


class _MoE(nn.Module):
    def __init__(self, args, block, packed_paths):
        super().__init__()
        self.gate = _Router(args)
        hidden = args.hidden_size
        width = args.routed_expert_hidden_size
        self.use_latent_experts = width is not None
        if self.use_latent_experts:
            self.routed_expert_down_proj = _Linear(hidden, width)
            self.routed_expert_up_proj = _Linear(width, hidden)
        self.experts = _Experts(
            args, block, width if self.use_latent_experts else hidden, packed_paths
        )
        self.shared_experts = _MLP(
            args, args.moe_intermediate_size * args.n_shared_experts, residual=False
        )
        self.combine_output_norm = _RMSNorm(hidden, args.rms_norm_eps)

    def __call__(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        _, weights, indices = self.gate(x)
        latent = self.routed_expert_down_proj(x) if self.use_latent_experts else x
        routed = self.experts(latent, indices, weights)
        if self.use_latent_experts:
            routed = self.routed_expert_up_proj(routed)
        return self.combine_output_norm(routed + self.shared_experts(x)).reshape(shape)


def build_mlp(args, zero_based_block: int) -> nn.Module:
    """Construct the HF-named dense or MoE MLP for one text decoder block."""
    if not 0 <= zero_based_block < len(args.mlp_layer_types):
        raise ValueError(f"MLP block index out of range: {zero_based_block}")
    if args.hidden_act != "silu":
        raise ValueError(f"unsupported HY-V4-Flash MLP activation: {args.hidden_act}")
    packed_paths = _packed_paths(args)
    kind = args.mlp_layer_types[zero_based_block]
    if kind == "dense":
        return _MLP(args, args.intermediate_size, residual=True)
    if kind == "sparse":
        return _MoE(args, zero_based_block, packed_paths)
    raise ValueError(f"unsupported MLP layer type: {kind!r}")
