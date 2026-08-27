# Copyright (c) 2026 Apple Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4 switch layers with an experimental MXFP4 block-list MoE GEMM."""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
from omlx.custom_kernels.nax import is_nax_available
from omlx.patches.deepseek_v4.decode_consistency import (
    is_armed as is_dspark_verify_armed,
)

_DEEPSEEK_MXFP4_SMALL_BLOCK_BM = 16
_DEEPSEEK_MXFP4_SMALL_BLOCK_VARIANT = 1
_DEEPSEEK_MXFP4_LARGE_BLOCK_BM = 32
_DEEPSEEK_MXFP4_LARGE_BLOCK_VARIANT = 2
_DEEPSEEK_AFFINE_LARGE_BLOCK_MIN_ROUTES = 8192
# Tuned on M3 Ultra. Set this to 8192 to restore the previous crossover on
# other pre-NAX chips; M5 prefill uses the NAX fallback below.
_DEEPSEEK_MXFP4_LARGE_BLOCK_MIN_ROUTES = int(
    os.environ.get("OMLX_DEEPSEEK_MXFP4_LARGE_BLOCK_MIN_ROUTES", "16384")
)
# Real TP4/4 layer-20 ABBA: BM32 is 1.113x stock at M=1024/top6 (6144
# routes), but regresses at 3072 and 12288 routes. Keep the evidenced island
# separate from the existing large-route crossover.
_DEEPSEEK_MXFP4_MID_BLOCK_MIN_ROUTES = int(
    os.environ.get("OMLX_DEEPSEEK_MXFP4_MID_BLOCK_MIN_ROUTES", "6144")
)
_DEEPSEEK_MXFP4_MID_BLOCK_MAX_ROUTES = int(
    os.environ.get("OMLX_DEEPSEEK_MXFP4_MID_BLOCK_MAX_ROUTES", "8192")
)

# On NAX GPUs (M5 family) mx.gather_qmm dispatches to the tensor-unit
# gather_qmm_rhs_nax kernels, which beat the pre-NAX block-list kernels for
# prefill-sized route counts (same regression shape as the Qwen qmm patch:
# 4k pp 828 -> 400 tok/s on M5 Max). Decode-sized calls stay on the block
# kernels pending M5 measurements. OMLX_DEEPSEEK_MOE_NAX=0 keeps the block
# kernels everywhere, =1 routes every call to stock on NAX GPUs.
_NAX_STOCK_MODE = os.environ.get("OMLX_DEEPSEEK_MOE_NAX", "").strip().lower()
_NAX_STOCK_MIN_ROUTES = int(
    os.environ.get("OMLX_DEEPSEEK_MOE_NAX_MIN_ROUTES", "1024")
)
# Lossless full routed-MoE decode primitive.  The 3:5 TP slices are physically
# exact and isolated-kernel faster through four coalesced rows, but the live
# TP2 B=4 gate regressed aggregate decode.  Keep production at B=1.
_DEEPSEEK_MXFP4_FULL_DECODE = os.environ.get(
    "OMLX_DSV4_FULL_MOE_DECODE", "1"
).strip().lower() in ("1", "true", "on")
_DEEPSEEK_MXFP4_FULL_DECODE_MAX_TOKENS = int(
    os.environ.get("OMLX_DSV4_FULL_MOE_DECODE_MAX_TOKENS", "1")
)
_DEEPSEEK_MXFP4_FULL_DECODE_LOGGED = False
# Exact M3-family M=1024 prefill route-tail kernels. Default OFF until the
# two-host full-model gate clears. The fixed native ABI makes decode and every
# non-DS4/equal-TP2 shape fall back before any candidate operation is queued.
_DEEPSEEK_MXFP4_TAIL8 = os.environ.get(
    "OMLX_DSV4_MOE_TAIL8", "0"
).strip().lower() in ("1", "true", "on")
_DEEPSEEK_MXFP4_TAIL8_EQUAL_TP = os.environ.get(
    "OMLX_DSV4_MOE_TAIL8_EQUAL_TP", "1"
).strip().lower() in ("1", "true", "on")
_DEEPSEEK_MXFP4_TAIL8_LOGGED = False
# One rollback switch for the proven asymmetric 3:5 pair.  When enabled on
# both ranks, M3 rank 0 uses the 3/8 tail8 path and M5 rank 1 uses the
# separate expert-blocked NAX projections.  Every other model, rank, shape,
# and device continues through the existing gates below.
_DEEPSEEK_MXFP4_COMBINED = os.environ.get(
    "OMLX_DSV4_COMBINED_MOE_PREFILL", "0"
).strip().lower() in ("1", "true", "on")
# Exact M=1024 M5 Max rank-1 5/8 expert-blocked TensorOps path. The physical
# layer gate cleared every BF16 boundary at 1.513x composed, but production
# remains explicit opt-in until the full distributed cold-prefill A/B clears.
_DEEPSEEK_MXFP4_NAX_BLOCKS = os.environ.get(
    "OMLX_DSV4_NAX_MOE_BLOCKS", "0"
).strip().lower() in ("1", "true", "on")
_DEEPSEEK_MXFP4_NAX_BLOCKS_LOGGED = False
_DEEPSEEK_MXFP4_NAX_BLOCKS_REJECTION_LOGGED = False


def _nax_prefers_stock(num_routes: int) -> bool:
    if _NAX_STOCK_MODE in ("0", "false", "off"):
        return False
    if not is_nax_available():
        return False
    if _NAX_STOCK_MODE in ("1", "true", "on"):
        return True
    return num_routes >= _NAX_STOCK_MIN_ROUTES


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


def _block_config(num_routes: int, native_kind: str) -> tuple[int, int]:
    if native_kind == "mxfp4":
        if (
            _DEEPSEEK_MXFP4_MID_BLOCK_MIN_ROUTES
            <= num_routes
            < _DEEPSEEK_MXFP4_MID_BLOCK_MAX_ROUTES
        ):
            return (
                _DEEPSEEK_MXFP4_LARGE_BLOCK_BM,
                _DEEPSEEK_MXFP4_LARGE_BLOCK_VARIANT,
            )
        large_block_min_routes = _DEEPSEEK_MXFP4_LARGE_BLOCK_MIN_ROUTES
    elif native_kind == "affine":
        large_block_min_routes = _DEEPSEEK_AFFINE_LARGE_BLOCK_MIN_ROUTES
    else:
        raise ValueError(f"Unsupported native block kind: {native_kind}")

    if num_routes >= large_block_min_routes:
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

    def _has_affine_metadata_dtype(self, dtype) -> bool:
        """Return whether affine quantization metadata uses ``dtype``."""
        biases = self.get("biases")
        return (
            self.mode == "affine"
            and biases is not None
            and "bias" not in self
            and self["weight"].dtype == mx.uint32
            and self["scales"].dtype == dtype
            and biases.dtype == dtype
        )

    def _native_block_kind(self, x, sorted_indices: bool, dtype=None) -> str | None:
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
    ):
        super().__init__()

        self.gate_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.up_proj = SwitchLinear(input_dims, hidden_dims, num_experts, bias=bias)
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def _can_use_mxfp4_full_decode(self, x, indices, scores) -> bool:
        if not _DEEPSEEK_MXFP4_FULL_DECODE or self.training or scores is None:
            return False
        if x.ndim < 2 or indices.ndim < 1 or indices.shape[-1] != 6:
            return False
        tokens = indices.size // 6
        if not 1 <= tokens <= _DEEPSEEK_MXFP4_FULL_DECODE_MAX_TOKENS:
            return False
        if scores.shape != indices.shape or scores.dtype != mx.float32:
            return False
        if x.dtype not in (mx.float16, mx.bfloat16):
            return False
        projections = (self.up_proj, self.gate_proj, self.down_proj)
        if not all(isinstance(p, QuantizedSwitchLinear) for p in projections):
            return False
        if not all(
            p.group_size == 32
            and p.bits == 4
            and p.mode == "mxfp4"
            and p.get("biases") is None
            and "bias" not in p
            and p["weight"].dtype == mx.uint32
            and p["scales"].dtype == mx.uint8
            for p in projections
        ):
            return False
        up_weight = self.up_proj["weight"]
        gate_weight = self.gate_proj["weight"]
        down_weight = self.down_proj["weight"]
        intermediate = int(up_weight.shape[1])
        input_dims = int(x.shape[-1])
        try:
            device_name = str(mx.device_info().get("device_name") or "")
        except Exception:
            return False
        qualified_shape = (
            device_name == "Apple M3 Ultra" and intermediate in (768, 2048)
        ) or (
            device_name == "Apple M5 Max" and intermediate == 1280
        )
        if (
            not qualified_shape
            or input_dims <= 0
            or input_dims % 512
            or intermediate <= 0
            or intermediate % 256
            or tuple(gate_weight.shape) != tuple(up_weight.shape)
            or int(up_weight.shape[2]) * 8 != input_dims
            or int(down_weight.shape[0]) != int(up_weight.shape[0])
            or int(down_weight.shape[1]) != input_dims
            or int(down_weight.shape[2]) * 8 != intermediate
        ):
            return False
        if not glm_fast.has_symbol("deepseek_mxfp4_full_decode"):
            return False
        activation_limit = getattr(self.activation, "limit", None)
        return (
            isinstance(activation_limit, (int, float))
            and activation_limit >= 0
            and not getattr(self.activation, "fp32", False)
        )

    def _can_use_mxfp4_tail8_prefill(
        self,
        request_shape,
        indices,
        x_sorted,
        original_dtype,
        native_kinds,
        use_f16_moe,
        block_plan,
    ) -> bool:
        combined = _DEEPSEEK_MXFP4_COMBINED
        try:
            equal_default = bool(
                _DEEPSEEK_MXFP4_TAIL8_EQUAL_TP
                and not combined
                and getattr(self, "_omlx_dsv4f_moe_tp", None) == (2, 0, ())
                and mx.device_info().get("device_name") == "Apple M3 Ultra"
            )
        except Exception:
            equal_default = False
        if (
            (not combined and not _DEEPSEEK_MXFP4_TAIL8 and not equal_default)
            or self.training
            or is_nax_available()
            or (combined and is_dspark_verify_armed())
        ):
            return False
        if request_shape != (1, 1024, 4096) or indices.shape != (1, 1024, 6):
            return False
        if (
            x_sorted.shape != (6144, 1, 4096)
            or x_sorted.dtype != mx.float16
            or original_dtype != mx.bfloat16
            or not use_f16_moe
            or native_kinds != ("mxfp4", "mxfp4", "mxfp4")
            or block_plan is None
        ):
            return False
        if combined:
            try:
                exact_device = (
                    mx.device_info().get("device_name") == "Apple M3 Ultra"
                )
            except Exception:
                return False
            if (
                not exact_device
                or not getattr(self, "_omlx_dsv4f_exact_config", False)
                or getattr(self, "_omlx_dsv4f_moe_tp", None)
                != (2, 0, (3, 5))
            ):
                return False
            intermediate = 768
        else:
            intermediate = 1024
        projections = (self.up_proj, self.gate_proj, self.down_proj)
        if not all(
            isinstance(projection, QuantizedSwitchLinear)
            for projection in projections
        ):
            return False
        up, gate, down = projections
        if (
            up["weight"].shape != (256, intermediate, 512)
            or up["scales"].shape != (256, intermediate, 128)
            or gate["weight"].shape != up["weight"].shape
            or gate["scales"].shape != up["scales"].shape
            or down["weight"].shape != (256, 4096, intermediate // 8)
            or down["scales"].shape != (256, 4096, intermediate // 32)
        ):
            return False
        block_meta, block_count, block_variant = _unpack_mxfp4_block_plan(
            block_plan
        )
        if (
            block_variant != 2
            or block_meta.shape != (448, 3)
            or block_count.shape != (1,)
        ):
            return False
        activation_limit = getattr(self.activation, "limit", None)
        if activation_limit != 10.0 or getattr(self.activation, "fp32", False):
            return False
        return all(
            glm_fast.has_symbol(symbol)
            for symbol in (
                "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8",
                "deepseek_mxfp4_gather_qmm_blocks_tail8",
            )
        )

    def _can_use_mxfp4_nax_blocks_prefill(
        self,
        request_shape,
        indices,
        scores,
        x_sorted,
        original_dtype,
    ) -> bool:
        """Preflight exact DS4F M5 rank-1 4/8 or 5/8 BF16 contracts."""

        tp_contract = getattr(self, "_omlx_dsv4f_moe_tp", None)

        def reject(reason: str) -> bool:
            global _DEEPSEEK_MXFP4_NAX_BLOCKS_REJECTION_LOGGED
            if (
                _DEEPSEEK_MXFP4_NAX_BLOCKS
                and not _DEEPSEEK_MXFP4_NAX_BLOCKS_REJECTION_LOGGED
            ):
                _DEEPSEEK_MXFP4_NAX_BLOCKS_REJECTION_LOGGED = True
                logging.getLogger(__name__).warning(
                    "DeepSeek V4 M5 NAX routed-MoE prefill requested but "
                    "ineligible (%s): tp=%r request=%r indices=%r/%s "
                    "scores=%r/%s sorted=%r/%s input_dtype=%s",
                    reason,
                    tp_contract,
                    request_shape,
                    tuple(indices.shape),
                    indices.dtype,
                    tuple(scores.shape) if scores is not None else None,
                    getattr(scores, "dtype", None),
                    tuple(x_sorted.shape),
                    x_sorted.dtype,
                    original_dtype,
                )
            return False

        nax_enabled = bool(
            _DEEPSEEK_MXFP4_NAX_BLOCKS
            or (
                _DEEPSEEK_MXFP4_COMBINED
                and tp_contract == (2, 1, (3, 5))
            )
        )
        if (
            not nax_enabled
            or self.training
            or is_dspark_verify_armed()
            or not getattr(self, "_omlx_dsv4f_exact_config", False)
            or tp_contract not in {
                (2, 1, ()),
                (2, 1, (3, 5)),
                (2, 1, (4, 4)),
            }
        ):
            return reject("execution contract")
        if (
            request_shape != (1, 1024, 4096)
            or tuple(indices.shape) != (1, 1024, 6)
            or indices.dtype not in (mx.int32, mx.uint32)
            or scores is None
            or tuple(scores.shape) != tuple(indices.shape)
            or scores.dtype != mx.float32
            or tuple(x_sorted.shape) != (6144, 1, 4096)
            or x_sorted.dtype != mx.bfloat16
            or original_dtype != mx.bfloat16
        ):
            return reject("request shape or dtype")
        activation_limit = getattr(self.activation, "limit", None)
        if activation_limit != 10.0 or getattr(self.activation, "fp32", False):
            return reject("activation contract")

        projections = (self.up_proj, self.gate_proj, self.down_proj)
        if not all(
            isinstance(projection, QuantizedSwitchLinear)
            and projection.group_size == 32
            and projection.bits == 4
            and projection.mode == "mxfp4"
            and projection.get("biases") is None
            and "bias" not in projection
            and projection["weight"].dtype == mx.uint32
            and projection["scales"].dtype == mx.uint8
            for projection in projections
        ):
            return reject("quantization metadata")
        up, gate, down = projections
        local_intermediate = 1280 if tp_contract == (2, 1, (3, 5)) else 1024
        if (
            tuple(up["weight"].shape) != (256, local_intermediate, 512)
            or tuple(up["scales"].shape)
            != (256, local_intermediate, 128)
            or tuple(gate["weight"].shape) != tuple(up["weight"].shape)
            or tuple(gate["scales"].shape) != tuple(up["scales"].shape)
            or tuple(down["weight"].shape)
            != (256, 4096, local_intermediate // 8)
            or tuple(down["scales"].shape)
            != (256, 4096, local_intermediate // 32)
        ):
            return reject("projection shape")

        try:
            exact_device = mx.device_info().get("device_name") == "Apple M5 Max"
        except Exception:
            return reject("device query")
        eligible = bool(
            exact_device
            and is_nax_available()
            and glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_blocks_nax")
            and glm_fast.ds4_projection_nax_kernels_built()
            and glm_fast.ds4_projection_nax_device_available()
        )
        return eligible if eligible else reject("device or native artifact")

    def __call__(self, x, indices, scores=None) -> mx.array:
        if self._can_use_mxfp4_full_decode(x, indices, scores):
            global _DEEPSEEK_MXFP4_FULL_DECODE_LOGGED
            if not _DEEPSEEK_MXFP4_FULL_DECODE_LOGGED:
                _DEEPSEEK_MXFP4_FULL_DECODE_LOGGED = True
                logging.getLogger(__name__).info(
                    "DeepSeek V4 full routed-MoE decode primitive active "
                    "(tokens=%d, hidden=%d, local_intermediate=%d)",
                    indices.size // 6,
                    int(x.shape[-1]),
                    int(self.up_proj["weight"].shape[1]),
                )
            return glm_fast.deepseek_mxfp4_full_decode(
                x,
                self.up_proj["weight"],
                self.up_proj["scales"],
                self.gate_proj["weight"],
                self.gate_proj["scales"],
                self.down_proj["weight"],
                self.down_proj["scales"],
                indices,
                scores,
                float(self.activation.limit),
            )

        request_shape = tuple(x.shape)
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
        projections = (self.up_proj, self.gate_proj, self.down_proj)
        use_f16_moe = original_dtype == mx.bfloat16 and all(
            isinstance(p, QuantizedSwitchLinear)
            and p._has_affine_metadata_dtype(mx.float16)
            for p in projections
        )
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
                block_kind = (
                    "mxfp4"
                    if all(kind == "mxfp4" for kind in native_kinds)
                    else "affine"
                )
                block_bm, block_variant = _block_config(idx.size, block_kind)
                block_meta, block_count = _build_mxfp4_blocks(
                    idx,
                    self.up_proj.num_experts,
                    block_bm,
                )
                block_plan = (block_meta, block_count, block_variant)

        use_nax_blocks_prefill = self._can_use_mxfp4_nax_blocks_prefill(
            request_shape,
            indices,
            scores,
            x,
            original_dtype,
        )
        if use_f16_moe and not use_nax_blocks_prefill:
            x = x.astype(mx.float16)

        nax_block_plan = None
        if use_nax_blocks_prefill:
            nax_block_plan = _build_mxfp4_blocks(
                idx,
                self.up_proj.num_experts,
                _DEEPSEEK_MXFP4_LARGE_BLOCK_BM,
            )

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
        use_tail8_prefill = use_pair_proj and self._can_use_mxfp4_tail8_prefill(
            request_shape,
            indices,
            x,
            original_dtype,
            native_kinds,
            use_f16_moe,
            block_plan,
        )
        if use_nax_blocks_prefill:
            global _DEEPSEEK_MXFP4_NAX_BLOCKS_LOGGED
            if not _DEEPSEEK_MXFP4_NAX_BLOCKS_LOGGED:
                _DEEPSEEK_MXFP4_NAX_BLOCKS_LOGGED = True
                logging.getLogger(__name__).info(
                    "DeepSeek V4 exact M5 NAX BM32 routed-MoE prefill active "
                    "(rank=1, TP=%s, local_intermediate=%d, M=1024; "
                    "OMLX_DSV4_NAX_MOE_BLOCKS=0 and "
                    "OMLX_DSV4_COMBINED_MOE_PREFILL=0 disable)",
                    ":".join(map(str, self._omlx_dsv4f_moe_tp[2])),
                    int(self.up_proj["weight"].shape[1]),
                )
            block_meta, block_count = nax_block_plan
            x_up = glm_fast.deepseek_mxfp4_gather_qmm_blocks_nax(
                x,
                self.up_proj["weight"],
                self.up_proj["scales"],
                block_meta,
                block_count,
            )
            x_gate = glm_fast.deepseek_mxfp4_gather_qmm_blocks_nax(
                x,
                self.gate_proj["weight"],
                self.gate_proj["scales"],
                block_meta,
                block_count,
            )
        elif use_pair_proj:
            block_meta, block_count, block_variant = _unpack_mxfp4_block_plan(
                block_plan
            )
            if use_tail8_prefill:
                global _DEEPSEEK_MXFP4_TAIL8_LOGGED
                if not _DEEPSEEK_MXFP4_TAIL8_LOGGED:
                    _DEEPSEEK_MXFP4_TAIL8_LOGGED = True
                    logging.getLogger(__name__).info(
                        "DeepSeek V4 exact BM8 route-tail prefill kernels active"
                    )
                x = glm_fast.deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8(
                    x,
                    self.up_proj["weight"],
                    self.up_proj["scales"],
                    self.gate_proj["weight"],
                    self.gate_proj["scales"],
                    block_meta,
                    block_count,
                    float(self.activation.limit),
                    block_variant,
                )
            elif glm_fast.has_symbol(
                "deepseek_mxfp4_gather_qmm_pair_concat_blocks"
            ):
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
        if use_nax_blocks_prefill:
            x = self.activation(x_up, x_gate)
            x = glm_fast.deepseek_mxfp4_gather_qmm_blocks_nax(
                x,
                self.down_proj["weight"],
                self.down_proj["scales"],
                block_meta,
                block_count,
            )
        elif use_tail8_prefill:
            x = glm_fast.deepseek_mxfp4_gather_qmm_blocks_tail8(
                x,
                self.down_proj["weight"],
                self.down_proj["scales"],
                block_meta,
                block_count,
                block_variant,
            )
        else:
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
