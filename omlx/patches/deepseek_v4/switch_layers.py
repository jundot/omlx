# Copyright (c) 2026 Apple Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4 switch layers with an experimental MXFP4 block-list MoE GEMM."""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

_DEEPSEEK_MXFP4_SMALL_BLOCK_BM = 16
_DEEPSEEK_MXFP4_SMALL_BLOCK_VARIANT = 1
_DEEPSEEK_MXFP4_LARGE_BLOCK_BM = 32
_DEEPSEEK_MXFP4_LARGE_BLOCK_VARIANT = 2
_DEEPSEEK_MXFP4_LARGE_BLOCK_MIN_ROUTES = 8192


def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


@lru_cache(maxsize=None)
def _mxfp4_block_builder(num_experts: int, bm: int):
    source = r"""
        const uint expert = thread_index_in_threadgroup;

        threadgroup atomic_int local_count;
        if (expert == 0) {
            atomic_store_explicit(&local_count, 0, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (expert >= NUM_EXPERTS) {
            return;
        }

        int lo = 0;
        int hi = M;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (indices[mid] < int(expert)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        const int start = lo;

        hi = M;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (indices[mid] <= int(expert)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        const int end = lo;

        for (int row = start; row < end; row += BM) {
            const int rows = min(BM, end - row);
            const int slot = atomic_fetch_add_explicit(
                &local_count, 1, memory_order_relaxed);
            if (slot < MAX_BLOCKS) {
                block_meta[slot * 3 + 0] = row;
                block_meta[slot * 3 + 1] = int(expert);
                block_meta[slot * 3 + 2] = rows;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (expert == 0) {
            block_count[0] = atomic_load_explicit(
                &local_count, memory_order_relaxed);
        }
    """

    return mx.fast.metal_kernel(
        name=f"deepseek_v4_mxfp4_block_builder_e{num_experts}_bm{bm}",
        input_names=["indices"],
        output_names=["block_meta", "block_count"],
        source=source,
        ensure_row_contiguous=True,
    )


def _build_mxfp4_blocks(indices: mx.array, num_experts: int, bm: int):
    indices = indices.astype(mx.int32)
    max_blocks = (indices.size + bm - 1) // bm + num_experts
    builder = _mxfp4_block_builder(num_experts, bm)
    return builder(
        inputs=[indices],
        template=[
            ("NUM_EXPERTS", num_experts),
            ("BM", bm),
            ("M", indices.size),
            ("MAX_BLOCKS", max_blocks),
        ],
        grid=(num_experts, 1, 1),
        threadgroup=(num_experts, 1, 1),
        output_shapes=[(max_blocks, 3), (1,)],
        output_dtypes=[mx.int32, mx.int32],
    )


def _mxfp4_block_config(num_routes: int) -> tuple[int, int]:
    if num_routes >= _DEEPSEEK_MXFP4_LARGE_BLOCK_MIN_ROUTES:
        return (
            _DEEPSEEK_MXFP4_LARGE_BLOCK_BM,
            _DEEPSEEK_MXFP4_LARGE_BLOCK_VARIANT,
        )
    return (
        _DEEPSEEK_MXFP4_SMALL_BLOCK_BM,
        _DEEPSEEK_MXFP4_SMALL_BLOCK_VARIANT,
    )


def _unpack_mxfp4_block_plan(block_plan):
    if len(block_plan) == 3:
        return block_plan
    block_meta, block_count = block_plan
    return block_meta, block_count, _DEEPSEEK_MXFP4_SMALL_BLOCK_VARIANT


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
            and glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_blocks")
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
            and self.bits in (2, 3)
            and self.mode == "affine"
            and biases is not None
            and "bias" not in self
            and self["weight"].dtype == mx.uint32
            and self["scales"].dtype == dtype
            and biases.dtype == dtype
            and glm_fast.has_symbol("deepseek_affine_gather_qmm_blocks")
        )

    def _native_block_kind(self, x, sorted_indices: bool, dtype=None) -> str | None:
        if self._can_use_mxfp4_blocks(x, sorted_indices):
            return "mxfp4"
        if self._can_use_affine_blocks(x, sorted_indices, dtype=dtype):
            return "affine"
        return None

    def __call__(self, x, indices, sorted_indices=False, block_plan=None):
        native_kind = self._native_block_kind(x, sorted_indices)
        if native_kind is not None:
            if block_plan is None:
                block_bm, block_variant = _mxfp4_block_config(indices.size)
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
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x, indices, scores=None) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        original_dtype = x.dtype

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        block_plan = None
        native_kinds = None
        use_f16_moe = False
        projections = (self.up_proj, self.gate_proj, self.down_proj)
        if do_sort and all(isinstance(p, QuantizedSwitchLinear) for p in projections):
            native_kinds = tuple(p._native_block_kind(x, do_sort) for p in projections)
            if x.dtype == mx.bfloat16:
                f16_native_kinds = tuple(
                    p._native_block_kind(x, do_sort, dtype=mx.float16)
                    for p in projections
                )
                if all(kind == "mxfp4" for kind in f16_native_kinds) or all(
                    kind == "affine" for kind in f16_native_kinds
                ):
                    native_kinds = f16_native_kinds
                    use_f16_moe = True
            if all(kind is not None for kind in native_kinds):
                block_bm, block_variant = _mxfp4_block_config(idx.size)
                block_meta, block_count = _build_mxfp4_blocks(
                    idx,
                    self.up_proj.num_experts,
                    block_bm,
                )
                block_plan = (block_meta, block_count, block_variant)

        if use_f16_moe:
            x = x.astype(mx.float16)

        use_pair_proj = (
            block_plan is not None
            and native_kinds is not None
            and native_kinds[0] == "mxfp4"
            and native_kinds[1] == "mxfp4"
            and glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_pair_blocks")
            and self.up_proj.output_dims == self.gate_proj.output_dims
            and self.up_proj.num_experts == self.gate_proj.num_experts
        )
        use_affine_pair_proj = (
            block_plan is not None
            and native_kinds is not None
            and native_kinds[0] == "affine"
            and native_kinds[1] == "affine"
            and self.up_proj.group_size == self.gate_proj.group_size
            and self.up_proj.bits == self.gate_proj.bits
            and self.up_proj.output_dims == self.gate_proj.output_dims
            and self.up_proj.num_experts == self.gate_proj.num_experts
            and glm_fast.has_symbol("deepseek_affine_gather_qmm_pair_concat_blocks")
        )
        if use_pair_proj:
            block_meta, block_count, block_variant = _unpack_mxfp4_block_plan(
                block_plan
            )
            if glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_pair_concat_blocks"):
                x_pair = glm_fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
                    x,
                    self.up_proj["weight"],
                    self.up_proj["scales"],
                    self.gate_proj["weight"],
                    self.gate_proj["scales"],
                    block_meta,
                    block_count,
                    block_variant,
                )
                hidden_dims = self.up_proj.output_dims
                x_up = x_pair[..., :hidden_dims]
                x_gate = x_pair[..., hidden_dims:]
            else:
                x_pair = glm_fast.deepseek_mxfp4_gather_qmm_pair_blocks(
                    x,
                    self.up_proj["weight"],
                    self.up_proj["scales"],
                    self.gate_proj["weight"],
                    self.gate_proj["scales"],
                    block_meta,
                    block_count,
                    block_variant,
                )
                x_up = x_pair[0]
                x_gate = x_pair[1]
        elif use_affine_pair_proj:
            block_meta, block_count, block_variant = _unpack_mxfp4_block_plan(
                block_plan
            )
            x_pair = glm_fast.deepseek_affine_gather_qmm_pair_concat_blocks(
                x,
                self.up_proj["weight"],
                self.up_proj["scales"],
                self.up_proj["biases"],
                self.gate_proj["weight"],
                self.gate_proj["scales"],
                self.gate_proj["biases"],
                block_meta,
                block_count,
                self.up_proj.group_size,
                self.up_proj.bits,
                block_variant,
            )
            hidden_dims = self.up_proj.output_dims
            x_up = x_pair[..., :hidden_dims]
            x_gate = x_pair[..., hidden_dims:]
        else:
            x_up = self.up_proj(x, idx, sorted_indices=do_sort, block_plan=block_plan)
            x_gate = self.gate_proj(
                x, idx, sorted_indices=do_sort, block_plan=block_plan
            )
        x = self.activation(x_up, x_gate)
        if (
            block_plan is not None
            and native_kinds is not None
            and native_kinds[2] == "affine"
            and isinstance(self.down_proj, QuantizedSwitchLinear)
            and x.dtype != self.down_proj["scales"].dtype
            and self.down_proj["scales"].dtype in (mx.float16, mx.bfloat16)
        ):
            x = x.astype(self.down_proj["scales"].dtype)
        x = self.down_proj(
            x,
            idx,
            sorted_indices=do_sort,
            block_plan=block_plan,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        x = x.squeeze(-2)
        if use_f16_moe:
            x = x.astype(original_dtype)
        return x


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
