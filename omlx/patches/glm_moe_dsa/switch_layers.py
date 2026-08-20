# Copyright © 2023-2024 Apple Inc.

import math
import os

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from .kernels import fast as glm_fast

# Block-list MoE GEMM dispatch, shared with DeepSeek-V4: the kernels live
# in the same native extension this file already imports, their shape
# checks are architecture-agnostic, and on pre-NAX hardware they beat
# stock mx.gather_qmm for this op (see deepseek_v4/switch_layers.py for
# the measured tuning notes). OMLX_GLM_MOE_BLOCKS=0 restores stock.
try:
    from omlx.patches.deepseek_v4.switch_layers import (
        _block_config,
        _build_mxfp4_blocks,
        _nax_prefers_stock,
        _unpack_mxfp4_block_plan,
    )

    _BLOCK_DISPATCH = os.environ.get(
        "OMLX_GLM_MOE_BLOCKS", ""
    ).strip().lower() not in ("0", "false", "off")
except Exception:  # pragma: no cover - depends on patch layout
    _BLOCK_DISPATCH = False


def _inverse_permutation(order, inverse_scatter=False):
    if inverse_scatter:
        return mx.put_along_axis(
            mx.zeros_like(order),
            order,
            mx.arange(order.size, dtype=order.dtype),
            axis=0,
        )
    return mx.argsort(order)


def _gather_sort(x, indices, inverse_scatter=False):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = _inverse_permutation(order, inverse_scatter)
    lhs_indices = order // M
    x = x.flatten(0, -3)
    return x[lhs_indices], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


class QuantizedSwitchLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
    ):
        super().__init__()

        scale = math.sqrt(1 / input_dims)
        self.weight, self.scales, *biases = mx.quantize(
            mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(num_experts, output_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.biases = biases[0] if biases else None

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Freeze this model's parameters
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def _can_use_mxfp4_blocks(self, x, sorted_indices: bool) -> bool:
        return (
            sorted_indices
            and x.ndim == 3
            and x.shape[-2] == 1
            and self.group_size == 32
            and self.bits == 4
            and self.mode == "mxfp4"
            and self.get("biases") is None
            and "bias" not in self
            and self["weight"].dtype == mx.uint32
            and self["scales"].dtype == mx.uint8
            and glm_fast.has("deepseek_mxfp4_gather_qmm_blocks")
        )

    def _can_use_affine_blocks(self, x, sorted_indices: bool, dtype=None) -> bool:
        dtype = dtype or x.dtype
        biases = self.get("biases")
        return (
            sorted_indices
            and x.ndim == 3
            and x.shape[-2] == 1
            and dtype in (mx.float16, mx.bfloat16)
            and self.group_size == 64
            and self.bits in (2, 3, 4, 8)
            and self.mode == "affine"
            and biases is not None
            and "bias" not in self
            and self["weight"].dtype == mx.uint32
            and self["scales"].dtype == dtype
            and biases.dtype == dtype
            and glm_fast.has("deepseek_affine_gather_qmm_blocks")
        )

    def _native_block_kind(self, x, sorted_indices: bool, dtype=None):
        if not _BLOCK_DISPATCH:
            return None
        if x.ndim == 3 and _nax_prefers_stock(int(x.shape[0])):
            return None
        if self._can_use_mxfp4_blocks(x, sorted_indices):
            return "mxfp4"
        if self._can_use_affine_blocks(x, sorted_indices, dtype=dtype):
            return "affine"
        return None

    def __call__(self, x, indices, sorted_indices=False, block_plan=None):
        native_kind = self._native_block_kind(x, sorted_indices)
        if native_kind is not None:
            if block_plan is None:
                block_bm, block_variant = _block_config(indices.size, native_kind)
                block_meta, block_count = _build_mxfp4_blocks(
                    indices,
                    self.num_experts,
                    block_bm,
                )
            else:
                block_meta, block_count, block_variant = _unpack_mxfp4_block_plan(
                    block_plan
                )
            if native_kind == "mxfp4":
                x = glm_fast.deepseek_mxfp4_gather_qmm_blocks(
                    x,
                    self["weight"],
                    self["scales"],
                    block_meta,
                    block_count,
                    block_variant,
                )
            else:
                x = glm_fast.deepseek_affine_gather_qmm_blocks(
                    x,
                    self["weight"],
                    self["scales"],
                    self["biases"],
                    block_meta,
                    block_count,
                    self.group_size,
                    self.bits,
                    block_variant,
                )
        else:
            x = mx.gather_qmm(
                x,
                self["weight"],
                self["scales"],
                self.get("biases"),
                rhs_indices=indices,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x


class SwitchLinear(nn.Module):
    def __init__(
        self, input_dims: int, output_dims: int, num_experts: int, bias: bool = True
    ):
        super().__init__()
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(num_experts, output_dims, input_dims),
        )

        if bias:
            self.bias = mx.zeros((num_experts, output_dims))

    @property
    def input_dims(self):
        return self.weight.shape[2]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    def __call__(self, x, indices, sorted_indices=False, block_plan=None):
        del block_plan
        x = mx.gather_mm(
            x,
            self["weight"].swapaxes(-1, -2),
            rhs_indices=indices,
            sorted_indices=sorted_indices,
        )
        if "bias" in self:
            x = x + mx.expand_dims(self["bias"][indices], -2)
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        num_experts, output_dims, input_dims = self.weight.shape
        ql = QuantizedSwitchLinear(
            input_dims,
            output_dims,
            num_experts,
            False,
            group_size,
            bits,
            mode=mode,
        )
        ql.weight, ql.scales, *biases = mx.quantize(
            self.weight, group_size, bits, mode=mode
        )
        ql.biases = biases[0] if biases else None

        if "bias" in self:
            ql.bias = self.bias
        return ql


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x, gate):
        return swiglu(gate, x)


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
        fused_gate_up: bool = False,
        inverse_scatter: bool = False,
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation
        self.inverse_scatter = inverse_scatter
        if fused_gate_up:
            self.gate_up_proj = SwitchLinear(
                input_dims, hidden_dims * 2, num_experts, bias=bias
            )
            del self.gate_proj
            del self.up_proj

    def __call__(
        self,
        x,
        indices,
        scores: mx.array | None = None,
        weighted_sum: bool = False,
    ) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(
                x, indices, inverse_scatter=self.inverse_scatter
            )
        if self.training:
            idx = mx.stop_gradient(idx)

        # One block plan shared by every projection of the layer: when all
        # of them qualify for the native block-list kernels, the expert
        # bucketing is computed once instead of per projection.
        fused = hasattr(self, "gate_up_proj")
        projections = (
            (self.gate_up_proj, self.down_proj)
            if fused
            else (self.up_proj, self.gate_proj, self.down_proj)
        )
        block_plan = None
        if (
            _BLOCK_DISPATCH
            and do_sort
            and all(isinstance(p, QuantizedSwitchLinear) for p in projections)
        ):
            native_kinds = tuple(
                p._native_block_kind(x, do_sort) for p in projections
            )
            if all(kind is not None for kind in native_kinds):
                block_kind = (
                    "mxfp4"
                    if all(kind == "mxfp4" for kind in native_kinds)
                    else "affine"
                )
                block_bm, block_variant = _block_config(idx.size, block_kind)
                block_meta, block_count = _build_mxfp4_blocks(
                    idx,
                    projections[0].num_experts,
                    block_bm,
                )
                block_plan = (block_meta, block_count, block_variant)

        if fused:
            x_gate_up = self.gate_up_proj(
                x, idx, sorted_indices=do_sort, block_plan=block_plan
            )
            x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
            x = self.down_proj(
                self.activation(x_up, x_gate),
                idx,
                sorted_indices=do_sort,
                block_plan=block_plan,
            )
        else:
            x_up = self.up_proj(
                x, idx, sorted_indices=do_sort, block_plan=block_plan
            )
            x_gate = self.gate_proj(
                x, idx, sorted_indices=do_sort, block_plan=block_plan
            )
            x = self.down_proj(
                self.activation(x_up, x_gate),
                idx,
                sorted_indices=do_sort,
                block_plan=block_plan,
            )

        if (
            weighted_sum
            and scores is not None
            and do_sort
            and hasattr(glm_fast, "glm_moe_weighted_sum")
        ):
            return glm_fast.glm_moe_weighted_sum(x, inv_order, scores)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.GELU(approx="precise"),
        bias: bool = False,
    ):
        super().__init__()

        self.fc1 = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.fc2 = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # When we have many tokens, then sort them to make sure that the access
        # of different experts is in order.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x = self.fc1(x, idx, sorted_indices=do_sort)
        x = self.activation(x)
        x = self.fc2(x, idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)
