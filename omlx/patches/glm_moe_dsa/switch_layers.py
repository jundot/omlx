# Copyright © 2023-2024 Apple Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os
from functools import cache

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from .kernels import fast as glm_fast

_GLM_AFFINE_BLOCK_MIN_ROUTES = 8192
_GLM_AFFINE_BLOCK_BM = 32
_GLM_AFFINE_BLOCK_VARIANT = 2
_GLM_AFFINE_BLOCK_MODE = (
    os.environ.get(
        "OMLX_GLM_MOE_AFFINE_BLOCKS",
        "",
    )
    .strip()
    .lower()
)


def _affine_blocks_enabled() -> bool:
    return _GLM_AFFINE_BLOCK_MODE in ("1", "true", "on")


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


@cache
def _route_counting_sort_builder(num_experts: int, bm: int):
    source = r"""
        const uint expert_thread = thread_index_in_threadgroup;
        threadgroup atomic_uint counts[NUM_EXPERTS];
        threadgroup atomic_uint cursors[NUM_EXPERTS];
        threadgroup uint route_starts[NUM_EXPERTS];
        threadgroup uint block_starts[NUM_EXPERTS];

        atomic_store_explicit(
            &counts[expert_thread], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint route = expert_thread; route < M; route += NUM_EXPERTS) {
            const uint expert = uint(indices[route]);
            atomic_fetch_add_explicit(
                &counts[expert], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (expert_thread == 0) {
            uint route_offset = 0;
            uint block_offset = 0;
            for (uint expert = 0; expert < NUM_EXPERTS; ++expert) {
                const uint count = atomic_load_explicit(
                    &counts[expert], memory_order_relaxed);
                route_starts[expert] = route_offset;
                block_starts[expert] = block_offset;
                atomic_store_explicit(
                    &cursors[expert], route_offset, memory_order_relaxed);
                route_offset += count;
                block_offset += (count + BM - 1) / BM;
            }
            block_count[0] = int(block_offset);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint route = expert_thread; route < M; route += NUM_EXPERTS) {
            const uint expert = uint(indices[route]);
            const uint position = atomic_fetch_add_explicit(
                &cursors[expert], 1u, memory_order_relaxed);
            order[position] = route;
            sorted_indices[position] = expert;
            inverse[route] = position;
        }

        const uint count = atomic_load_explicit(
            &counts[expert_thread], memory_order_relaxed);
        const uint route_start = route_starts[expert_thread];
        const uint block_start = block_starts[expert_thread];
        for (uint local = 0; local < count; local += BM) {
            const uint slot = block_start + local / BM;
            block_meta[slot * 3 + 0] = int(route_start + local);
            block_meta[slot * 3 + 1] = int(expert_thread);
            block_meta[slot * 3 + 2] = int(min(uint(BM), count - local));
        }
    """

    return mx.fast.metal_kernel(
        name=f"glm_moe_dsa_route_counting_sort_e{num_experts}_bm{bm}",
        input_names=["indices"],
        output_names=[
            "order",
            "sorted_indices",
            "inverse",
            "block_meta",
            "block_count",
        ],
        source=source,
        ensure_row_contiguous=True,
    )


def _gather_counting_sort(x, indices, num_experts: int, bm: int):
    *_, routes_per_token = indices.shape
    flat_indices = indices.flatten()
    num_routes = flat_indices.size
    max_blocks = (num_routes + bm - 1) // bm + num_experts
    builder = _route_counting_sort_builder(num_experts, bm)
    order, sorted_indices, inverse, block_meta, block_count = builder(
        inputs=[flat_indices],
        template=[
            ("T", flat_indices.dtype),
            ("NUM_EXPERTS", num_experts),
            ("BM", bm),
            ("M", num_routes),
        ],
        grid=(num_experts, 1, 1),
        threadgroup=(num_experts, 1, 1),
        output_shapes=[
            (num_routes,),
            (num_routes,),
            (num_routes,),
            (max_blocks, 3),
            (1,),
        ],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32, mx.int32, mx.int32],
    )
    sorted_x = x.flatten(0, -3)[order // routes_per_token]
    return sorted_x, sorted_indices, inverse, (block_meta, block_count)


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

    def _can_use_affine_blocks(self, x, sorted_indices: bool) -> bool:
        biases = self.get("biases")
        return (
            sorted_indices
            and x.ndim == 3
            and x.shape[-2] == 1
            and x.dtype in (mx.float16, mx.bfloat16)
            and self.group_size == 64
            and self.bits == 4
            and self.mode == "affine"
            and biases is not None
            and "bias" not in self
            and self["weight"].dtype == mx.uint32
            and self["scales"].dtype == x.dtype
            and biases.dtype == x.dtype
            and glm_fast.has("deepseek_affine_gather_qmm_blocks")
        )

    def __call__(self, x, indices, sorted_indices=False, block_plan=None):
        if block_plan is not None and self._can_use_affine_blocks(
            x,
            sorted_indices,
        ):
            block_meta, block_count, block_variant = block_plan
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
        block_plan = None
        if do_sort:
            projections = (
                (self.gate_up_proj, self.down_proj)
                if hasattr(self, "gate_up_proj")
                else ()
            )
            sorted_input = x.flatten(0, -3)
            use_affine_blocks = (
                indices.size >= _GLM_AFFINE_BLOCK_MIN_ROUTES
                and _affine_blocks_enabled()
                and len(projections) == 2
                and projections[0].num_experts == 256
                and all(
                    isinstance(projection, QuantizedSwitchLinear)
                    and projection._can_use_affine_blocks(sorted_input, True)
                    for projection in projections
                )
            )
            if use_affine_blocks:
                x, idx, inv_order, raw_block_plan = _gather_counting_sort(
                    x,
                    indices,
                    projections[0].num_experts,
                    _GLM_AFFINE_BLOCK_BM,
                )
                block_plan = (*raw_block_plan, _GLM_AFFINE_BLOCK_VARIANT)
            else:
                x, idx, inv_order = _gather_sort(
                    x,
                    indices,
                    inverse_scatter=self.inverse_scatter,
                )
        if self.training:
            idx = mx.stop_gradient(idx)
        if hasattr(self, "gate_up_proj"):
            x_gate_up = self.gate_up_proj(
                x,
                idx,
                sorted_indices=do_sort,
                block_plan=block_plan,
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
                x,
                idx,
                sorted_indices=do_sort,
                block_plan=block_plan,
            )
            x_gate = self.gate_proj(
                x,
                idx,
                sorted_indices=do_sort,
                block_plan=block_plan,
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
