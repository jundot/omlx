# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels for the MiniMax M3 hot spots: sparse attention and routed experts.

The sources live under ``third_party`` (TensorFold: index scores, block
selection and attention on the tensor cores; mlx-serve: expert GEMMs that
dequantize an FP bank as they go). The RoPE and router kernels here form their
float32 results in the order of oMLX's Metal backend.
"""

from __future__ import annotations

import functools
from pathlib import Path

import mlx.core as mx

from . import Config

_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
# Dynamic shared memory of an expert tile: two A stages, one B tile, the table.
_EXPERT_SHARED = 2 * 64 * 128 * 2 + 128 * 128 * 2 + 1024
# Bank formats the expert kernels read, as their FMT template value.
_FORMATS = {"mxfp8": 1, "nvfp4": 3}
# Dynamic shared memory of the attention kernel: a block of keys and of values.
_ATTENTION_SHARED = 2 * 128 * 136 * 2


def available() -> bool:
    return mx.cuda.is_available()


def _source(project: str, name: str) -> str:
    return (_THIRD_PARTY / project / "minimax_m3_vl" / name).read_text()


@functools.cache
def _sparse_kernel(
    name: str, inputs: tuple[str, ...], outputs: tuple[str, ...], shared: int = 0
):
    return mx.fast.cuda_kernel(
        name=f"omlx_minimax_{name}",
        input_names=list(inputs),
        output_names=list(outputs),
        source=_source("TensorFold", f"{name}.cu"),
        header=_source("TensorFold", "msa_header.cuh"),
        shared_memory=shared,
    )


@functools.cache
def _expert_kernel(name: str, inputs: tuple[str, ...], output: str, shared: int):
    return mx.fast.cuda_kernel(
        name=f"omlx_minimax_{name}",
        input_names=list(inputs),
        output_names=[output],
        source=_source("mlx-serve", f"{name}.cu"),
        header=_source("mlx-serve", "moe_prefill_header.cu"),
        shared_memory=shared,
    )


def covers_attention(cfg: Config, dtype: mx.Dtype, index: bool) -> bool:
    """Whether the attention kernels take a layer: the M3 / M3.1 shapes."""
    heads = dtype == mx.bfloat16 and cfg.head_dim == 128
    heads = heads and cfg.heads == 16 * cfg.kv_heads
    if not index:
        return heads
    return (
        heads
        and cfg.index_dim == 128
        and cfg.block == 128
        and cfg.index_heads == cfg.kv_heads
        and cfg.init_blocks == 0
        and cfg.local_blocks == 1
        and cfg.top_blocks <= 32
    )


def select_blocks(
    index_queries: mx.array, index_keys: mx.array, start: int, top: int
) -> tuple[mx.array, mx.array]:
    """Kept blocks of each (row, index head): ids [L, heads, top] in ascending
    order (-1 past the last) and the same blocks as bitmaps [L, heads, words]."""
    heads, rows, _ = index_queries.shape
    blocks = -(-index_keys.shape[0] // 128)
    pos = mx.array([start], dtype=mx.int32)
    scores = _sparse_kernel("index_scores", ("iq", "ik", "pos"), ("scores",))(
        inputs=[index_queries, index_keys, pos],
        grid=(128 * blocks, -(-rows * heads // 64), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, heads, blocks)],
        output_dtypes=[mx.float32],
    )[0]
    ids, bits = _sparse_kernel("index_select", ("scores", "pos"), ("sel", "bits"))(
        inputs=[scores, pos],
        template=[("TOPK", top)],
        grid=(32 * heads, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, heads, top), (rows, heads, -(-blocks // 32))],
        output_dtypes=[mx.int32, mx.uint32],
    )
    return ids, bits


def attend(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    bits: mx.array,
    start: int,
    scale: float,
    partial: bool = False,
) -> mx.array:
    """Causal attention of ``queries`` [1, heads, L, 128] over the kept blocks.

    ``partial`` rounds each block's own softmax output to bfloat16 before the
    blocks combine, as the host's MSA prefill kernels do; otherwise the blocks
    share one float32 softmax.
    """
    _, heads, rows, dim = queries.shape
    names = ("q", "k", "v", "bits", "pos", "scale")
    out = _sparse_kernel("attention", names, ("o",), _ATTENTION_SHARED)(
        inputs=[
            queries[0],
            keys[0],
            values[0],
            bits,
            mx.array([start], dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[("PARTIAL", int(partial))],
        grid=(256 * -(-rows // 8), keys.shape[1], 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(heads, rows, dim)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out[None]


def attend_all(
    queries: mx.array, keys: mx.array, values: mx.array, start: int, scale: float
) -> mx.array:
    """Causal attention of ``queries`` over every key."""
    rows, heads = queries.shape[2], keys.shape[1]
    words = -(-keys.shape[2] // (128 * 32))
    bits = mx.full((rows, heads, words), 0xFFFFFFFF, dtype=mx.uint32)
    return attend(queries, keys, values, bits, start, scale)


# Each logit adds its K products in float32 FMAs, in `runs` parts of K summed
# in order, which is how Metal forms this product.
_ROUTER = """
extern __shared__ __align__(16) unsigned char smem[];
__nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
const int K = x_shape[1], outputs = w_shape[1];
const int row = blockIdx.x, e = threadIdx.x;
for (int k = threadIdx.x; k < K; k += blockDim.x) xs[k] = x[(size_t)row * K + k];
__syncthreads();
if (e >= outputs) return;
const int parts = runs[0], span = (K / 16 / parts) * 16;
float total = 0.f;
for (int p = 0; p < parts; ++p) {
  const int end = p + 1 == parts ? K : (p + 1) * span;
  float acc = 0.f;
  for (int k = p * span; k < end; ++k)
    acc = fmaf(__bfloat162float(xs[k]), w[(size_t)k * outputs + e], acc);
  total = p == 0 ? acc : total + acc;
}
out[(size_t)row * outputs + e] = total;
"""
_ROPE_HEADER = """
#include <cuda_bf16.h>
__device__ __forceinline__ void mm3_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void mm3_store(float* p, float v) { *p = v; }
"""
_ROPE = """
const int rows = x_shape[0], heads = x_shape[1], dim = x_shape[2], half = cs_shape[1];
const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
const int i = id % half, h = (id / half) % heads;
const size_t l = id / half / heads;
if (l >= (size_t)rows) return;
const float* in = x + (l * heads + h) * dim;
OUT* to = out + ((size_t)h * rows + l) * dim;
const float c = cs[l * half + i], s = sn[l * half + i];
const float x1 = in[i], x2 = in[i + half];
mm3_store(to + i, fmaf(x1, c, -__fmul_rn(x2, s)));
mm3_store(to + i + half, fmaf(x1, s, __fmul_rn(x2, c)));
for (int d = 2 * half + i; d < dim; d += half) mm3_store(to + d, in[d]);
"""


@functools.cache
def _router_kernel(shared: int):
    return mx.fast.cuda_kernel(
        name="omlx_minimax_router",
        input_names=["x", "w", "runs"],
        output_names=["out"],
        source=_ROUTER,
        shared_memory=shared,
    )


@functools.cache
def _rope_kernel():
    return mx.fast.cuda_kernel(
        name="omlx_minimax_rope",
        input_names=["x", "cs", "sn"],
        output_names=["out"],
        source=_ROPE,
        header=_ROPE_HEADER,
    )


def host_runs(rows: int, outputs: int, inputs: int) -> int:
    """Parts of K that oMLX's Metal backend sums separately in a dense
    float32 product (MLX 0.32.4 ``steel_gemm_splitk`` on a Max or Ultra GPU):
    a product with few output tiles is cut into equal runs of K."""
    small = -(-rows // 16) * -(-outputs // 16) <= 2048
    if not (small and inputs // 16 >= 8 and inputs >= max(rows, outputs)):
        return 1
    ratio = (inputs // 16) // (-(-rows // 32) * -(-outputs // 32))
    return min(max(2, 1 << max(ratio - 1, 0).bit_length()), 32)


def router(x: mx.array, columns: mx.array) -> mx.array:
    """Float32 router logits [rows, experts] of bfloat16 ``x`` [rows, K] and
    the float32 router weight as ``columns`` [K, experts], summed in the
    host's order."""
    rows, inputs = x.shape
    outputs = columns.shape[1]
    threads = -(-outputs // 32) * 32
    runs = mx.array([host_runs(rows, outputs, inputs)], dtype=mx.int32)
    return _router_kernel(inputs * 2)(
        inputs=[x, columns, runs],
        grid=(threads * rows, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, outputs)],
        output_dtypes=[mx.float32],
    )[0]


def rope(x: mx.array, cos: mx.array, sin: mx.array, dtype: mx.Dtype) -> mx.array:
    """Rotate ``x`` [L, heads, dim] float32 by ``cos`` and ``sin`` [L, pairs]
    and return it as [1, heads, L, dim] in ``dtype``."""
    rows, heads, dim = x.shape
    return _rope_kernel()(
        inputs=[x, cos, sin],
        template=[("OUT", dtype)],
        grid=(rows * heads * cos.shape[1], 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(heads, rows, dim)],
        output_dtypes=[dtype],
    )[0][None]


def covers_experts(gate, up, down, dtype: mx.Dtype) -> bool:
    """Whether the expert kernels take these banks: FP formats in 128 tiles."""
    if dtype != mx.bfloat16 or any(b.scales is None for b in (gate, up, down)):
        return False
    if gate.quant != up.quant or gate.weight.shape != up.weight.shape:
        return False
    for bank in (gate, down):
        mode = bank.quant.get("mode")
        inputs = bank.weight.shape[2] * 32 // bank.quant["bits"]
        if mode not in _FORMATS or bank.scales.dtype != mx.uint8 or inputs % 128:
            return False
    # The weighted sum gives a thread 8 of the block's output columns.
    hidden = down.weight.shape[1]
    return gate.weight.shape[1] % 64 == 0 and hidden % 128 == 0 and hidden <= 8192


def _expert_grid(tiles: int, experts: int, rows: int) -> tuple[int, int, int]:
    # A popular expert's 64-row tiles split over up to 8 blocks while the grid
    # stays near 16k blocks; more empty blocks make the launch slower.
    splits = max(1, min(8, 16384 // (tiles * experts)))
    return 256 * tiles, experts, min(splits, -(-rows // 64))


def gate_up(
    x: mx.array,
    source: mx.array,
    experts: mx.array,
    gate,
    up,
    gate_scale: mx.array,
    up_scale: mx.array,
    act: mx.array,
) -> mx.array:
    """The activation [R, I] of routed rows: row r is token ``source[r]`` of
    ``x`` [T, K] through expert ``experts[r]`` (sorted) of the gate and up
    banks, each scaled by its expert's float32 scale; ``act`` is float32
    [alpha, limit, beta]."""
    count, inter, _ = gate.weight.shape
    rows = experts.shape[0]
    names = ("x", "rows", "inds", "gw", "gs", "uw", "us", "gscale", "uscale")
    kernel = _expert_kernel(
        "moe_prefill_gate_up", (*names, "act_params"), "act", _EXPERT_SHARED
    )
    return kernel(
        inputs=[
            x,
            source.astype(mx.uint32),
            experts.astype(mx.uint32),
            gate.weight,
            gate.scales,
            up.weight,
            up.scales,
            gate_scale,
            up_scale,
            act,
        ],
        template=[
            ("T", mx.bfloat16),
            ("FMT", _FORMATS[gate.quant["mode"]]),
            ("K", x.shape[1]),
            ("I", inter),
            ("STG", 2),
            ("BDEP", 4),
        ],
        grid=_expert_grid(inter // 64, count, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, inter)],
        output_dtypes=[mx.bfloat16],
    )[0]


def gather(x: mx.array, bank, experts: mx.array) -> mx.array:
    """``x`` [R, K] rows, sorted by expert, each through its expert of ``bank``."""
    count, outputs, _ = bank.weight.shape
    rows = experts.shape[0]
    kernel = _expert_kernel(
        "moe_prefill_gather", ("x", "w", "s", "inds"), "y", _EXPERT_SHARED
    )
    return kernel(
        inputs=[x, bank.weight, bank.scales, experts.astype(mx.uint32)],
        template=[
            ("T", mx.bfloat16),
            ("FMT", _FORMATS[bank.quant["mode"]]),
            ("K", x.shape[1]),
            ("N", outputs),
            ("STG", 2),
            ("BDEP", 4),
        ],
        grid=_expert_grid(outputs // 128, count, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, outputs)],
        output_dtypes=[mx.bfloat16],
    )[0]


def weighted_sum(down: mx.array, inverse: mx.array, weights: mx.array) -> mx.array:
    """Each token's routed rows weighted and summed in float32: [T, hidden].

    ``inverse`` [T * k] is the sorted row of each token's k-th expert and
    ``weights`` [T, k] its float32 routing weight.
    """
    tokens, each = weights.shape
    hidden = down.shape[1]
    kernel = _expert_kernel("moe_prefill_sum", ("down", "inverse", "weights"), "y", 0)
    return kernel(
        inputs=[down, inverse.astype(mx.uint32), weights],
        template=[("TOPK", each), ("H", hidden)],
        grid=(tokens * (hidden // 8), 1, 1),
        threadgroup=(hidden // 8, 1, 1),
        output_shapes=[(tokens, hidden)],
        output_dtypes=[mx.bfloat16],
    )[0]
