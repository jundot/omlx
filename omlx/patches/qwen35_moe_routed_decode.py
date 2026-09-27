# SPDX-License-Identifier: Apache-2.0
"""Fused routed experts for one-token Qwen3.5-MoE-family decode.

After the fused router, a one-token MoE block runs five dependent launches
for its routed experts: the gate+up ``gather_qmm``, the compiled SwiGLU, the
down ``gather_qmm``, the multiply by the router scores, and the sum over the
top-k experts. At batch-one decode each of them is short and they run back
to back, so this patch runs the same arithmetic in two launches:

1. gate+up with a SwiGLU epilogue. Each simdgroup computes the gate rows and
   the matching up rows of one expert with MLX's ``qmv_fast`` lane partition
   and add order, rounds both to the activation dtype, then applies MLX's
   ``Sigmoid`` and the two multiplies of the compiled ``swiglu`` in its order.
2. down with the router-weighted sum. Simdgroup ``j`` of a threadgroup runs
   the stock ``qmv`` work (including its guarded K tail) of selected expert
   ``j`` for the threadgroup's rows. Each row is rounded, multiplied by the
   score in the activation dtype, and the ten products are summed in the
   order of MLX's ``col_reduce_small``.

The result is bit-identical to the composed path. The quantized dot products
reuse the MLX 0.32.2 transcription in ``moe_verify_gather``. MLX picks
``qmv_fast`` when K % 512 == 0 and N % 8 == 0 and ``qmv`` otherwise, so only
shapes where gate+up takes ``qmv_fast`` and down takes ``qmv`` are routed:
one bf16 token, top-k 10, 4-bit affine with group size 64, hidden % 512 == 0
and intermediate % 512 != 0 (Qwen3.8-Flash-Next: 2560 and 640). Prefill,
verify rows and every other shape keep the original body. If the first
launch fails, the patch disables itself and the block keeps its composed body.
"""

from __future__ import annotations

import logging
from functools import cache

import mlx.core as mx

from .moe_verify_gather import _HEADER as _QMV_HEADER

logger = logging.getLogger(__name__)

TOP_K = 10
BITS = 4
GROUP_SIZE = 64
_GATE_UP_ROWS = 2  # gate rows (and as many up rows) per simdgroup
_GATE_UP_SIMDGROUPS = 2
_DOWN_ROWS = 4
_DISABLED = False
_PROVEN = False

_SIGMOID = r"""
// MLX 0.32.2 Sigmoid, evaluated in T as the compiled swiglu does.
template <typename U>
inline U omlx_mlx_sigmoid(U x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

# One threadgroup per (expert slot z, block of NSG * RPS output rows). Output
# is [TOP_K, N / 2] = silu(gate) * up.
_GATE_UP_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row = int(tid.y) * (NSG * RPS) + int(simd_gid) * RPS;
    const size_t expert = size_t(rhs[tid.z]);

    const device uint8_t* ws = (const device uint8_t*)w +
        expert * N * in_vec_size_w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc = scales + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs = biases + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xp = x + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[2 * RPS] = {0};

    for (int k = 0; k < K; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < 2 * RPS; row++) {
        const int r = row < RPS ? row : N / 2 + row - RPS;
        const device uint8_t* wl = ws + r * in_vec_size_w;
        float s = sc[r * in_vec_size_g];
        float b = bs[r * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xp += BLOCK_SIZE;
    }

    for (int row = 0; row < 2 * RPS; row++) {
      result[row] = simd_sum(result[row]);
    }
    if (simd_lid == 0) {
      device T* yp = y + size_t(tid.z) * (N / 2) + out_row;
      for (int row = 0; row < RPS; row++) {
        T g = static_cast<T>(result[row]);
        T u = static_cast<T>(result[row + RPS]);
        T t = g * omlx_mlx_sigmoid<T>(g);
        yp[row] = t * u;
      }
    }
"""

# Threadgroup (32, TOP_K): simdgroup j computes RPS rows of selected expert j,
# then simdgroup 0 sums the weighted rows. Output is [N].
_DOWN_SOURCE = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    const int in_vec_size_g = K / GS;
    const int out_row = int(tid.y) * RPS;
    const int slot = int(simd_gid);
    const size_t expert = size_t(rhs[slot]);
    threadgroup T part[10 * RPS];

    const device uint8_t* ws = (const device uint8_t*)w +
        expert * N * in_vec_size_w + out_row * in_vec_size_w +
        int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* sc = scales + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* bs = biases + expert * N * in_vec_size_g +
        out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* xp = x + slot * K + int(simd_lid) * VALUES_PER_THREAD;

    float x_thread[VALUES_PER_THREAD];
    float result[RPS] = {0};

    int k = 0;
    for (; k < K - BLOCK_SIZE; k += BLOCK_SIZE) {
      float sum = load_vector<T>(xp, x_thread);
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float s = sc[row * in_vec_size_g];
        float b = bs[row * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
      }
      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      xp += BLOCK_SIZE;
    }
    const int remaining = clamp(
        int(K - k - int(simd_lid) * VALUES_PER_THREAD), 0, VALUES_PER_THREAD);
    if (remaining > 0) {
      float sum = load_vector_safe<T>(xp, x_thread, remaining);
      for (int row = 0; row < RPS; row++) {
        const device uint8_t* wl = ws + row * in_vec_size_w;
        float s = sc[row * in_vec_size_g];
        float b = bs[row * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0; row < RPS; row++) {
      result[row] = simd_sum(result[row]);
    }
    if (simd_lid == 0) {
      for (int row = 0; row < RPS; row++) {
        part[slot * RPS + row] = static_cast<T>(result[row]) * scores[slot];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0 && int(simd_lid) < RPS) {
      // col_reduce_small over 10 rows with 8 lanes: rows j and j + 8 fold
      // first, then lanes 0..7 accumulate in order, all in T.
      const int row = int(simd_lid);
      T t[8];
      for (int j = 0; j < 8; ++j) {
        t[j] = part[j * RPS + row] + T(0);
      }
      t[0] = part[8 * RPS + row] + t[0];
      t[1] = part[9 * RPS + row] + t[1];
      T acc = t[0];
      for (int j = 1; j < 8; ++j) {
        acc = t[j] + acc;
      }
      y[out_row + row] = acc;
    }
"""


def _header(fast: bool) -> str:
    return (
        _QMV_HEADER.replace("__BITS__", str(BITS))
        .replace("__GS__", str(GROUP_SIZE))
        .replace("__FAST__", "1" if fast else "0")
        + _SIGMOID
    )


@cache
def _kernels():
    gate_up = mx.fast.metal_kernel(
        name="omlx_qwen35_moe_gate_up_swiglu_decode",
        input_names=["x", "w", "scales", "biases", "rhs"],
        output_names=["y"],
        header=_header(fast=True),
        source=_GATE_UP_SOURCE,
    )
    down = mx.fast.metal_kernel(
        name="omlx_qwen35_moe_down_combine_decode",
        input_names=["x", "w", "scales", "biases", "rhs", "scores"],
        output_names=["y"],
        header=_header(fast=False),
        source=_DOWN_SOURCE,
    )
    return gate_up, down


def _quantized_ok(layer) -> bool:
    return (
        getattr(layer, "bits", None) == BITS
        and getattr(layer, "group_size", None) == GROUP_SIZE
        and getattr(layer, "mode", "affine") == "affine"
        and "biases" in layer
        and "bias" not in layer
        and layer["scales"].dtype == mx.bfloat16
    )


def routed_decode_eligible(block, x) -> bool:
    """One bf16 token whose stock launches take the replicated partitions."""
    if _DISABLED:
        return False
    hidden = x.shape[-1]
    if x.size != hidden or x.dtype != mx.bfloat16 or block.top_k != TOP_K:
        return False
    switch_mlp = getattr(block, "switch_mlp", None)
    gate_up = getattr(switch_mlp, "gate_up_proj", None)
    down = getattr(switch_mlp, "down_proj", None)
    if gate_up is None or down is None:
        return False
    if not (_quantized_ok(gate_up) and _quantized_ok(down)):
        return False
    inter = down["weight"].shape[-1] * 32 // BITS
    return (
        hidden % 512 == 0
        and inter % 512 != 0
        and inter % GROUP_SIZE == 0
        and gate_up["weight"].shape[1:] == (2 * inter, hidden * BITS // 32)
        and down["weight"].shape[1:] == (hidden, inter * BITS // 32)
    )


def routed_decode(block, x, indices, scores):
    """``(switch_mlp(x, indices) * scores[..., None]).sum(axis=-2)`` for one
    token, in two launches."""
    gate_up = block.switch_mlp.gate_up_proj
    down = block.switch_mlp.down_proj
    hidden = x.shape[-1]
    inter = down["weight"].shape[-1] * 32 // BITS
    gate_up_kernel, down_kernel = _kernels()
    dtype = x.dtype
    ids = indices.reshape(TOP_K).astype(mx.uint32)
    rows = _GATE_UP_ROWS * _GATE_UP_SIMDGROUPS
    h = gate_up_kernel(
        inputs=[
            x.reshape(hidden),
            gate_up["weight"],
            gate_up["scales"],
            gate_up["biases"],
            ids,
        ],
        template=[
            ("T", dtype),
            ("K", hidden),
            ("N", 2 * inter),
            ("RPS", _GATE_UP_ROWS),
            ("NSG", _GATE_UP_SIMDGROUPS),
        ],
        grid=(32, _GATE_UP_SIMDGROUPS * inter // rows, TOP_K),
        threadgroup=(32, _GATE_UP_SIMDGROUPS, 1),
        output_shapes=[(TOP_K, inter)],
        output_dtypes=[dtype],
    )[0]
    y = down_kernel(
        inputs=[
            h,
            down["weight"],
            down["scales"],
            down["biases"],
            ids,
            scores.reshape(TOP_K),
        ],
        template=[("T", dtype), ("K", inter), ("N", hidden), ("RPS", _DOWN_ROWS)],
        grid=(32, TOP_K * hidden // _DOWN_ROWS, 1),
        threadgroup=(32, TOP_K, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[dtype],
    )[0]
    return y.reshape(x.shape)


def apply_qwen35_moe_routed_decode_patch() -> bool:
    """Wrap the router-fused mlx-vlm ``Qwen3_5MoeSparseMoeBlock`` call.

    Needs ``qwen35_moe_router`` applied first: the fast arm reuses its fused
    routing launch, so it selects the same experts with the same scores as
    the body it replaces."""
    if not mx.metal.is_available():
        return False
    try:
        from mlx_vlm.models.qwen3_5_moe import language as vlm_moe
    except ImportError:
        return False
    from .qwen35_moe_router import fused_router_topk, router_eligible

    cls = getattr(vlm_moe, "Qwen3_5MoeSparseMoeBlock", None)
    if cls is None or not getattr(cls, "_omlx_router_fused", False):
        return False
    if getattr(cls, "_omlx_routed_decode", False):
        return True
    orig_call = cls.__call__

    def patched_call(self, x):
        global _DISABLED, _PROVEN
        if not (
            routed_decode_eligible(self, x) and router_eligible(x, self.num_experts)
        ):
            return orig_call(self, x)
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        inds, scores = fused_router_topk(gates, self.top_k)
        if scores.dtype != x.dtype:
            return orig_call(self, x)
        try:
            y = routed_decode(self, x, inds, scores)
            if not _PROVEN:
                # Surface a kernel build failure while the call can still
                # fall back, once per process.
                mx.eval(y)
                _PROVEN = True
                logger.info("Qwen MoE fused routed-expert decode engaged")
        except Exception:
            _DISABLED = True
            logger.warning(
                "fused routed-expert decode failed; composed fallback",
                exc_info=True,
            )
            return orig_call(self, x)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    patched_call._omlx_routed_decode_original = orig_call
    cls.__call__ = patched_call
    cls._omlx_routed_decode = True
    logger.info("Qwen MoE fused routed-expert decode patch applied")
    return True
