# SPDX-License-Identifier: Apache-2.0
"""INT8-activation (A8) sorted ``gather_qmm`` for routed MoE experts on M5.

The routed-expert prefill path (``m5_gather_qmm_nax``) dequantizes the packed
Q4 weight to BF16 in threadgroup memory and multiplies it with BF16 tensor
ops. This module adds the operand path the dense oQ kernels already use
(``qwen35_oq_a8``): the packed codes are decoded straight into INT8 fragment
registers and multiplied through the INT8 x INT8 -> INT32 tensor op, with the
GS64 affine correction applied per group. Only the Gate+Up projection is
covered; the Down projection keeps the A16 path.

    checkpoint Q4 affine weight [E, N, K / 8] (uint32), GS64
    checkpoint scale / bias     [E, N, G], G = K / 64     (read in place)
            |
    Stage A: INT8 token rows, once per token row (before routing)
            |
    tile scan: the one-threadgroup pre-pass of ``m5_gather_qmm_nax``
            |
    stage this tile's 64 weight rows of scale / bias in threadgroup memory
            |
    Q4 codes -> INT8 fragments;  INT8 x INT8 -> INT32
            |
    GS64 affine correction;  SwiGLU on the accumulators
            |
    [M, N / 2] -> the existing A16 sorted down projection

Scheduling is deliberately *not* reimplemented: the tile list comes from the
same pre-pass, a threadgroup still computes one single-expert ``BM x BN``
output tile, and the activation rows are read in place through ``row_map``.
Only the operand specialization changes.

Contract (identical to the dense oQ A8 kernels; see
``tests/test_qwen35_oq_a8.py`` for the reference):

    Stage A      amax = max|x| per row; scale = amax / 127;
                 Qa = rint(x / scale) clipped to [-127, 127] as INT8;
                 Ra = the code sum of each GS64 group.
    GEMM         acc_g = sum over the group of Qa * Qw, exact in INT32.
    correction   out = Sa[m] * sum_g (Sw_g * acc_g + Bw_g * Ra_g)

Stage A runs once per token row and the kernel reads those rows through
``row_map``, so top-k replication does not repeat the quantization.

The kernel reads the checkpoint's ``[E, N, G]`` scale/bias in place. Each
threadgroup stages the metadata of its 64 weight rows as ``[group][row]`` in
threadgroup memory (2 x 64 x G x 2 bytes: 10 KiB for Flash-Next, 16 KiB at
G = 64). A larger ``G``, or one that is not a multiple of 4, uses scalar loads
of the same layout with identical bits.
"""

from __future__ import annotations

import logging
import threading

import mlx.core as mx

from . import m5_gather_qmm_nax as _nax
from .m5_gather_qmm import _swiglu_limit
from .mlx_lm_mtp.batch_generator import _mtp_language_model, _mtp_module
from .moe_expert_offload import OffloadSwitchGLU

logger = logging.getLogger(__name__)


_BN = 64
_WN = 2
_GROUP = 64
# Largest G (= K / 64) whose tile metadata is staged in threadgroup memory:
# 2 x 64 rows x 64 groups x 2 bytes = 16 KiB (see the module docstring).
_STAGE_MAX_GROUPS = 64

# Tile height. On M5 Max with Flash-Next routes (10K-328K rows) BM32 was the
# fastest or tied; the A16 planner's 64-128 rows lose up to 1.5x.
_BM = 32

_lock = threading.RLock()
_kernel = None
_kernel_failed = False

_A8_HEADER = r"""
using namespace metal;
using namespace mlx::steel;

constant constexpr int kA8FragM = 16;
constant constexpr int kA8FragN = 32;
constant constexpr int kA8FragK = 16;
constant constexpr int kA8Elems = BaseNAXFrag::kElemsPerFrag;
constant constexpr int kA8Dest = 2 * kA8Elems;
constant constexpr int kA8Group = 64;
constant constexpr int kA8Steps = kA8Group / kA8FragK;

// One single-expert BM x BN output tile of a sorted routed projection, with
// the activation operand quantized to INT8.
//
// Grid: x = column tiles * 32 lanes, y = tile index * WN, z = BM / 32.
// tid.y indexes the tile list produced by omlx_gqmm_tile_scan. The activation
// rows of a tile are sorted rows row_start .. row_start + rows - 1; the kernel
// reads the token row rmap[r] of Qa/Sa/Ra in place, so no [M, K] copy of the
// activation exists.
//
// STAGED selects how the checkpoint's [E, N, G] scale/bias are read:
//   1  the threadgroup's 64 weight rows are staged once in threadgroup memory
//      as [group][local row] (G = SG, a multiple of 4, at most 64)
//   0  scalar loads straight from the checkpoint layout
template <typename T, int BITS, int WM, int WN, int EPI, int STAGED, int SG>
METAL_FUNC void omlx_a8_gather_nax(
    const device int8_t* qa,
    const device float* sa,
    const device short* ra,
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device uint32_t* rmap,
    const device uint32_t* tiles,
    const uint32_t tile_count,
    const int N,
    const int K,
    const int n_tok,
    device T* out,
    threadgroup T* meta_s,
    threadgroup T* meta_b,
    uint3 tid,
    uint simd_gid,
    uint tl) {
  constexpr int TM = 2;
  constexpr int BM = TM * kA8FragM * WM;
  constexpr int BN = kA8FragN * WN;
  constexpr int words = (kA8Group * BITS) / 32;
  // EPI != 0: N holds each expert's gate rows followed by its up rows. A
  // 64-row weight tile pairs 32 gate with 32 up rows (the A16 epilogue's
  // pair_row): a simdgroup's two 16-column fragments are gate and up of the
  // same 16 output columns, so every lane holds both projections of its
  // elements and the SwiGLU runs on the accumulators, writing [M, N / 2].
  constexpr bool kPair = EPI != 0;
  const int half_n = N / 2;

  const int groups = K / kA8Group;

  const int tile_idx = int(tid.y);
  if (tile_idx >= int(tile_count)) {
    return;
  }
  const uint4 tile = *((const device uint4*)tiles + tile_idx);
  const int row_start = int(tile.x);
  const int expert = int(tile.y);
  const int tile_rows = int(tile.z);

  // Threadgroup layout is (32 lanes, WN column simdgroups, BM / 32 row
  // simdgroups), so the linear simdgroup index is row * WN + column -- the
  // convention gather_seg uses (tm = sgid / kWN, tn = sgid % kWN).
  const int sg_m = int(simd_gid) / WN;
  const int sg_n = int(simd_gid) % WN;
  const int row_base = sg_m * (TM * kA8FragM);
  // Plain: the simdgroup's 32 weight/output columns. Paired: its 16 output
  // columns (the weight rows are col_base + {0, half_n}).
  const int col_base = kPair ? int(tid.x) * kA8FragN + sg_n * kA8FragM
                             : int(tid.x) * BN + sg_n * kA8FragN;
  // Weight-row distance between the two fragments.
  const int frag_n = kPair ? half_n : kA8FragM;

  const short2 coord = BaseNAXFrag::get_coord();
  const int cx = int(coord.x) >> 2;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      kA8FragM,
      kA8FragN,
      kA8FragK,
      /* transpose_left = */ false,
      /* transpose_right = */ true,
      /* relaxed_precision = */ false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = mpp::tensor_ops::matmul2d_descriptor(
      kA8FragM,
      kA8FragN,
      kA8FragK,
      false,
      true,
      false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  mpp::tensor_ops::matmul2d<desc_set, metal::execution_simdgroup> op_set;

  auto ct_a =
      op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      int32_t>();

  float Cf[TM][kA8Dest];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kA8Dest; ++e) {
      Cf[i][e] = 0.0f;
    }
  }

  const int w_row = groups * words;
  const int n_lane = col_base + int(coord.y);
  const device uint32_t* wbase =
      w + size_t(expert) * size_t(N) * size_t(w_row) +
      size_t(n_lane) * size_t(w_row);
  const int w_stride8 = 8 * w_row;
  const size_t w_stride_frag = size_t(frag_n) * size_t(w_row);

  // The checkpoint layout: one expert's scale/bias are N rows of G values.
  const device T* s_exp = scales + size_t(expert) * size_t(groups) * size_t(N);
  const device T* b_exp = biases + size_t(expert) * size_t(groups) * size_t(N);

  const int n_run0 = col_base + int(coord.x);

  if (STAGED) {
    // Stage the 64 weight rows of this tile (rows are contiguous [G] runs in
    // the checkpoint layout) as [group][local row]. Local rows: paired = 32
    // gate rows then 32 up rows; plain = 64 consecutive rows.
    constexpr int kThreads = WM * WN * 32;
    constexpr int kU = 4;
    const int q4 = groups / 4;
    const int total = 64 * q4;
    const device uint2* s4 = reinterpret_cast<const device uint2*>(s_exp);
    const device uint2* b4 = reinterpret_cast<const device uint2*>(b_exp);
    for (int e0 = int(tl); e0 < total; e0 += kThreads * kU) {
      uint2 vs[kU];
      uint2 vb[kU];
      STEEL_PRAGMA_UNROLL
      for (int u = 0; u < kU; ++u) {
        const int e = e0 + u * kThreads;
        if (e < total) {
          const int lr = e / q4;
          const int c = e - lr * q4;
          const int wr = kPair
              ? (lr >= 32 ? half_n + int(tid.x) * 32 + (lr - 32)
                          : int(tid.x) * 32 + lr)
              : int(tid.x) * BN + lr;
          const size_t at = size_t(wr) * size_t(q4) + size_t(c);
          vs[u] = s4[at];
          vb[u] = b4[at];
        }
      }
      STEEL_PRAGMA_UNROLL
      for (int u = 0; u < kU; ++u) {
        const int e = e0 + u * kThreads;
        if (e < total) {
          const int lr = e / q4;
          const int c = e - lr * q4;
          const vec<T, 4> a = as_type<vec<T, 4>>(vs[u]);
          const vec<T, 4> b = as_type<vec<T, 4>>(vb[u]);
          STEEL_PRAGMA_UNROLL
          for (int k = 0; k < 4; ++k) {
            meta_s[(c * 4 + k) * 64 + lr] = a[k];
            meta_b[(c * 4 + k) * 64 + lr] = b[k];
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Local-row base of this lane's columns in the staged tile.
  const int lr0 = (kPair ? sg_n * kA8FragM : sg_n * kA8FragN) + int(coord.x);
  const int lr_h = kPair ? 32 : kA8FragM;

  // Token row of each (row fragment, row half) this lane touches: constant
  // over the K loop, so the row map is read once per tile, not per group.
  int mt_row[TM][2];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int r = 0; r < 2; ++r) {
      const int ms = min(
          row_base + i * kA8FragM + r * 8 + int(coord.y), tile_rows - 1);
      mt_row[i][r] = int(rmap[row_start + ms]);
    }
  }

  for (int g = 0; g < groups; ++g) {
    uint2 wg[4];
    STEEL_PRAGMA_UNROLL
    for (int q = 0; q < 4; ++q) {
      const device uint32_t* wr = wbase + (q & 1) * w_stride8 +
          (q >> 1) * w_stride_frag + size_t(g) * words;
      wg[q] = reinterpret_cast<const device uint2*>(wr)[cx];
    }

    STEEL_PRAGMA_UNROLL
    for (int t = 0; t < kA8Steps; ++t) {
      STEEL_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
        const int base = (q >> 1) * kA8Elems + (q & 1) * 4;
        if (BITS == 4) {
          const uint32_t word = wg[q][t >> 1];
          const char4 quad = as_type<char4>(
              ((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
          ct_b[base + 0] = quad.x;
          ct_b[base + 1] = quad.y;
          ct_b[base + 2] = quad.z;
          ct_b[base + 3] = quad.w;
        }
      }

      STEEL_PRAGMA_UNROLL
      for (int hf = 0; hf < 2; ++hf) {
        STEEL_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) {
          const int mt = mt_row[hf][r];
          const char4 quad = as_type<char4>(
              *reinterpret_cast<const device uint32_t*>(
                  qa + size_t(mt) * size_t(K) + size_t(g) * kA8Group +
                  size_t(cx) * 16 + size_t(t) * 4));
          ct_a[r * 4 + 0] = quad.x;
          ct_a[r * 4 + 1] = quad.y;
          ct_a[r * 4 + 2] = quad.z;
          ct_a[r * 4 + 3] = quad.w;
        }
        if (t == 0) {
          if (hf == 0) {
            op_set.run(ct_a, ct_b, acc0);
          } else {
            op_set.run(ct_a, ct_b, acc1);
          }
        } else {
          if (hf == 0) {
            op.run(ct_a, ct_b, acc0);
          } else {
            op.run(ct_a, ct_b, acc1);
          }
        }
      }
    }

    // GS64 scale / bias of this group for the lane's four columns, per
    // fragment half.
    vec<T, 4> sv[2];
    vec<T, 4> bv[2];
    if (STAGED) {
      STEEL_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        sv[h] = *reinterpret_cast<const threadgroup vec<T, 4>*>(
            meta_s + g * 64 + lr0 + h * lr_h);
        bv[h] = *reinterpret_cast<const threadgroup vec<T, 4>*>(
            meta_b + g * 64 + lr0 + h * lr_h);
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        const size_t n0 = size_t(n_run0 + h * frag_n);
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          sv[h][j] = s_exp[(n0 + j) * size_t(groups) + size_t(g)];
          bv[h][j] = b_exp[(n0 + j) * size_t(groups) + size_t(g)];
        }
      }
    }

    float r_g[TM][2];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int r = 0; r < 2; ++r) {
        r_g[i][r] =
            float(ra[size_t(g) * size_t(n_tok) + size_t(mt_row[i][r])]);
      }
    }

    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kA8Dest; ++e) {
      const int r = ((e & 7) >> 2);
      const float swc = float(sv[e >> 3][e & 3]);
      const float bwc = float(bv[e >> 3][e & 3]);
      Cf[0][e] =
          metal::fma(swc, float(acc0[e]), metal::fma(bwc, r_g[0][r], Cf[0][e]));
      Cf[1][e] =
          metal::fma(swc, float(acc1[e]), metal::fma(bwc, r_g[1][r], Cf[1][e]));
    }
  }

  if (kPair) {
    // Gate is element e of fragment 0 and up element e of fragment 1 (e + 8).
    // Each is rounded to T exactly as the plain store rounds it, then the
    // unfused path's activation runs op for op in T (MLX's Sigmoid and
    // Multiply, as in the A16 epilogue): silu(gate) * up.
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kA8Elems; ++e) {
        const int r = e >> 2;
        const int ms = row_base + i * kA8FragM + int(coord.y) + r * 8;
        if (ms < tile_rows) {
          const int n = col_base + int(coord.x) + (e & 3);
          const int m = row_start + ms;
          const float sc = sa[mt_row[i][r]];
          const T g = static_cast<T>(sc * Cf[i][e]);
          const T u = static_cast<T>(sc * Cf[i][e + kA8Elems]);
          out[size_t(m) * size_t(half_n) + size_t(n)] =
              Multiply()(Multiply()(g, Sigmoid()(g)), u);
        }
      }
    }
  } else {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kA8Dest; ++e) {
        const int ee = e & 7;
        const int r = ee >> 2;
        const int ms = row_base + i * kA8FragM + int(coord.y) + r * 8;
        if (ms < tile_rows) {
          const int n =
              col_base + (e >> 3) * kA8FragM + int(coord.x) + (ee & 3);
          const int m = row_start + ms;
          out[size_t(m) * size_t(N) + size_t(n)] =
              static_cast<T>(sa[mt_row[i][r]] * Cf[i][e]);
        }
      }
    }
  }
}
"""

_A8_SOURCE = """
    threadgroup T meta_s[STAGED ? SG * 64 : 1];
    threadgroup T meta_b[STAGED ? SG * 64 : 1];
    omlx_a8_gather_nax<T, BITS, WM, WN, EPI, STAGED, SG>(
        qa, sa, ra, w, scales, biases, rmap, tiles, tile_count[0],
        params[0], params[1], params[2], out, meta_s, meta_b,
        threadgroup_position_in_grid, simdgroup_index_in_threadgroup,
        thread_index_in_threadgroup);
"""


def _get_kernel():
    """Build (once) the A8 gather kernel object, or None."""
    global _kernel, _kernel_failed
    if _kernel is not None or _kernel_failed:
        return _kernel
    with _lock:
        if _kernel is not None or _kernel_failed:
            return _kernel
        mlx_src = _nax._read_mlx_headers(
            _nax._MLX_MM_HEADERS + _nax._MLX_OPS_HEADERS
        )
        if mlx_src is None:
            _kernel_failed = True
            logger.warning("mlx kernel headers not found; A8 gather disabled")
            return None
        try:
            _kernel = mx.fast.metal_kernel(
                name="omlx_a8_gather",
                input_names=[
                    "qa",
                    "sa",
                    "ra",
                    "w",
                    "scales",
                    "biases",
                    "rmap",
                    "tiles",
                    "tile_count",
                    "params",
                ],
                output_names=["out"],
                header=mlx_src + _A8_HEADER,
                source=_A8_SOURCE,
            )
        except Exception:  # noqa: BLE001
            _kernel_failed = True
            logger.warning("A8 gather kernel failed to build", exc_info=True)
            return None
        return _kernel


def supports(
    x,
    w,
    scales,
    biases,
    indices,
    group_size,
    bits,
    mode,
    row_map=None,
    tokens=None,
) -> bool:
    """Layout gate for the A8 gather.

    Accepts affine Q4 with GS64 and metadata of the activation dtype, a fused
    or plain expert weight ``[E, N, K * bits / 32]`` with ``N % 64 == 0`` and
    ``K % 64 == 0``, scale/bias in the checkpoint layout ``[E, N, K / 64]``, and
    a uint32 row map of the sorted rows onto the token rows.
    """
    if mode != "affine" or bits != 4 or group_size != _GROUP:
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if x.ndim != 3 or x.shape[1] != 1 or indices.ndim != 1:
        return False
    M, K = int(indices.shape[0]), int(x.shape[2])
    if M < 8 or indices.dtype != mx.uint32 or K % _GROUP:
        return False
    if row_map is None or row_map.ndim != 1 or int(row_map.shape[0]) != M:
        return False
    if row_map.dtype != mx.uint32:
        return False
    if tokens is not None and int(x.shape[0]) != int(tokens):
        return False
    if w.ndim != 3 or w.dtype != mx.uint32:
        return False
    E, N = int(w.shape[0]), int(w.shape[1])
    if E == 0 or N == 0 or N % _BN:
        return False
    if w.shape[2] * 32 != K * bits:
        return False
    if biases is None or scales.dtype != x.dtype or biases.dtype != x.dtype:
        return False
    return scales.shape == biases.shape and scales.shape == (E, N, K // _GROUP)


def _stage_a(x):
    from omlx.custom_kernels.qwen35_prefill import fast

    return fast.qwen35_oq_a8_stage_a_v8(x, 0)


def _staged(groups: int) -> bool:
    """Whether the tile metadata fits the staging buffers and the staging
    loads (four groups at a time) apply."""
    return groups % 4 == 0 and groups <= _STAGE_MAX_GROUPS


def _launch(
    x, w, scales, biases, indices, row_map, *, bm, swiglu, init_value=None
):
    """Stage A, the tile scan and the gather kernel; None when unavailable."""
    from omlx.custom_kernels.qwen35_prefill import fast

    if not fast.oq_a8_available():
        return None
    kernel = _get_kernel()
    scan = _nax._get_kernel("scan")
    if kernel is None or scan is None:
        return None

    T, K = int(x.shape[0]), int(x.shape[2])
    M = int(indices.shape[0])
    E, N = int(w.shape[0]), int(w.shape[1])
    wm, wn = bm // 32, _WN

    # Stage A is left lazy (no mx.eval): the kernel depends on it through the
    # graph, like the dense oQ A8 path.
    qa, sa, ra = _stage_a(x)

    # The grid is sized by the *upper bound* on tiles and the kernel returns
    # early for tile indices past tile_count, so nothing is read back to the
    # host and the launch never synchronises.
    max_tiles = (M + bm - 1) // bm + min(E, M)
    tiles, tile_count = scan(
        inputs=[indices, mx.array([M, E, max_tiles], dtype=mx.int32)],
        template=[("BM", bm), ("MAXE", _nax._MAX_EXPERTS)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(max_tiles * 4,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )

    groups = K // _GROUP
    staged = _staged(groups)
    kw = {} if init_value is None else {"init_value": init_value}
    # A 64-row weight tile is 64 output columns, or 32 once gate and up are
    # paired.
    return kernel(
        inputs=[
            qa,
            sa,
            ra,
            w,
            scales,
            biases,
            row_map,
            tiles,
            tile_count,
            mx.array([N, K, T], dtype=mx.int32),
        ],
        template=[
            ("T", x.dtype),
            ("BITS", 4),
            ("WM", wm),
            ("WN", wn),
            ("EPI", 1 if swiglu else 0),
            ("STAGED", 1 if staged else 0),
            ("SG", groups if staged else 0),
        ],
        grid=((N // _BN) * 32, max_tiles * wn, bm // 32),
        threadgroup=(32, wn, bm // 32),
        output_shapes=[(M, 1, N // 2 if swiglu else N)],
        output_dtypes=[x.dtype],
        **kw,
    )[0]


def sorted_gather_qmm_a8(
    x,
    w,
    scales,
    biases,
    indices,
    row_map,
    *,
    group_size,
    bits,
    mode="affine",
    swiglu=False,
    init_value=None,
):
    """``activation(x[row_map]) @ w[indices].T`` with an INT8 activation.

    ``x`` holds the *token* rows ``[T, 1, K]``; Stage A runs on them once and
    the kernel reads ``Qa``/``Sa``/``Ra`` through ``row_map``, so the top-k
    replication never re-quantizes an activation. ``scales`` / ``biases`` are
    the checkpoint tensors ``[E, N, K / 64]``, read in place. Returns
    ``[M, 1, N]``, or None when unsupported (the caller keeps the A16 path).

    ``swiglu=True``: ``w`` holds each expert's gate rows followed by its up
    rows (``[E, N, K]``, N = 2 * n) and the kernel returns the SwiGLU
    ``silu(gate) * up`` as ``[M, 1, n]`` (see the kernel header for the exact
    rounding contract).

    ``init_value`` pre-fills the output, so tests can detect unwritten
    elements.
    """
    if not supports(x, w, scales, biases, indices, group_size, bits, mode, row_map):
        return None
    # One canary verdict per kernel instantiation (dtype and K).
    if not _nax._checked(("a8", x.dtype, int(x.shape[2]) // _GROUP), _self_test):
        return None
    return _launch(
        x, w, scales, biases, indices, row_map,
        bm=_BM, swiglu=swiglu, init_value=init_value,
    )


# Canary: 2 experts, N = 128 (64 gate + 64 up rows), 64 sorted rows over 16
# token rows. It runs the *production* instantiation (dtype, staging, K), so a
# kernel that cannot compile or runs wrongly on this toolchain declines to the
# A16 path instead of failing a request.
_CANARY_COUNTS = (40, 24)
_CANARY_TOKENS = 16
_CANARY_TOLERANCE = 0.1  # of the reference's max, well above the A8 error (~2%)


def _self_test(key: tuple):
    """True when the fused kernel is within the A8 error of the A16 reference
    on a canary; False when it raises or misbehaves; None when it cannot be
    evaluated here (a function transformation is being traced)."""
    _, dtype, groups = key
    K = groups * _GROUP
    try:
        N = 2 * _BN
        k_w, k_x = mx.random.split(mx.random.key(0xA8), 2)
        wf = (mx.random.normal((len(_CANARY_COUNTS), N, K), key=k_w) * 0.05).astype(
            dtype
        )
        wq, scales, biases = mx.quantize(wf, group_size=_GROUP, bits=4)
        idx = mx.array(
            [e for e, n in enumerate(_CANARY_COUNTS) for _ in range(n)],
            dtype=mx.uint32,
        )
        rows = int(idx.shape[0])
        x = (mx.random.normal((_CANARY_TOKENS, 1, K), key=k_x) * 0.6).astype(dtype)
        row_map = ((mx.arange(rows, dtype=mx.uint32) * 7 + 3) % _CANARY_TOKENS).astype(
            mx.uint32
        )
        out = _launch(x, wq, scales, biases, idx, row_map, bm=_BM, swiglu=True)
        if out is None:
            return False
        gate_up = mx.gather_qmm(
            x[row_map],
            wq,
            scales,
            biases,
            rhs_indices=idx,
            transpose=True,
            group_size=_GROUP,
            bits=4,
            sorted_indices=True,
        )
        gate, up = mx.split(gate_up, 2, axis=-1)
        ref = _nax.reference_activation(up, gate).astype(mx.float32)
        err = mx.abs(out.astype(mx.float32) - ref).max().item()
        scale = mx.abs(ref).max().item()
        ok = err <= _CANARY_TOLERANCE * scale
    except Exception as e:  # noqa: BLE001
        if "transformation" in str(e):
            return None
        logger.warning("routed A8 self-test raised for K=%d: %s", K, e)
        return False
    if not ok:
        logger.warning(
            "routed A8 disabled for K=%d: canary max error %.3g (reference max %.3g)",
            K,
            err,
            scale,
        )
    return ok


# -- routed MoE integration --------------------------------------------------
#
# ``apply_qwen35_oq_a8_patch`` (the ``qwen35_oq_a8_enabled`` setting) tags the
# backbone SwitchGLU modules of one loaded model. Untagged modules, including
# the MTP draft layer, keep the A16 path.

_TAG = "_omlx_routed_a8_min_tokens"

# Model families measured for speed and NLL on this path (matched on the
# model class module path, as ``qwen35_moe_gate_up`` does).
_MEASURED_FAMILIES = ("qwen4_exp",)

# Decode and verify windows stay A16. ``qwen35_oq_a8_min_tokens`` can raise
# this floor but not lower it.
_MIN_TOKENS_FLOOR = 128


def _mtp_module_ids(model) -> set[int]:
    """Ids of every module inside the model's MTP draft head."""
    mtp = _mtp_module(_mtp_language_model(model))
    if mtp is None or not hasattr(mtp, "modules"):
        return set()
    return {id(m) for m in mtp.modules()}


def _is_measured_family(model) -> bool:
    module = type(model).__module__ or ""
    return any(token in module for token in _MEASURED_FAMILIES)


def _is_fused_routed_glu(module) -> bool:
    gate_up = getattr(module, "get", None) and module.get("gate_up_proj")
    return gate_up is not None and module.get("down_proj") is not None


def tag_routed_a8_modules(model, min_tokens: int) -> int:
    """Opt the backbone routed-expert modules of ``model`` in. Returns how many.

    The MTP draft head is excluded by identity, so the exclusion does not
    depend on how its modules are named. Never raises: on an error nothing
    stays tagged and the model keeps the A16 path.
    """
    if not _is_measured_family(model):
        return 0
    tagged = []
    try:
        skip = _mtp_module_ids(model)
        floor = max(int(min_tokens), _MIN_TOKENS_FLOOR)
        for _, module in model.named_modules():
            if id(module) in skip or not _is_fused_routed_glu(module):
                continue
            setattr(module, _TAG, floor)
            tagged.append(module)
    except Exception:  # noqa: BLE001
        logger.warning("routed A8 modules not tagged", exc_info=True)
        for module in tagged:
            setattr(module, _TAG, None)
        return 0
    if tagged:
        logger.info(
            "routed A8 gate/up enabled on %d MoE layers (min_tokens=%d)",
            len(tagged),
            floor,
        )
    return len(tagged)


def _eligible(switch_mlp, activation) -> bool:
    """Plain affine Q4 / GS64 fused Gate+Up with an unclamped SwiGLU, on a
    module that is not an expert-offload wrapper."""
    if not _is_fused_routed_glu(switch_mlp) or isinstance(switch_mlp, OffloadSwitchGLU):
        return False
    gate_up, down = switch_mlp.get("gate_up_proj"), switch_mlp.get("down_proj")
    if "bias" in gate_up or "bias" in down:
        return False
    # ``None in (...)`` would call ``mx.array.__eq__(None)`` and raise.
    if any(gate_up.get(k) is None for k in ("weight", "scales", "biases")):
        return False
    if (gate_up.bits, gate_up.group_size, gate_up.mode) != (4, _GROUP, "affine"):
        return False
    # Unknown or clamped activations are not fused; only silu(gate) * up is.
    return _swiglu_limit(activation) is None


_warned: set[str] = set()


def try_routed_a8(switch_mlp, token_rows, idx, seq_len=None):
    """Sorted rows after the A8 Gate+Up and the A16 Down (``[M, 1, K]``), or
    None to keep the A16 path.

    ``token_rows`` and ``idx`` come from ``moe_routes.sort_routes``.
    ``seq_len`` keeps a batched decode step (many sequences of one token) on
    A16. Exceptions fall back to A16 with one warning per exception type.
    """
    min_tokens = getattr(switch_mlp, _TAG, None)
    if min_tokens is None:
        return None
    try:
        x_tok, row_map = token_rows
        tokens = int(x_tok.shape[0])
        if min(tokens, tokens if seq_len is None else int(seq_len)) < min_tokens:
            return None
        if not _eligible(switch_mlp, switch_mlp.activation):
            return None
        gate_up = switch_mlp.get("gate_up_proj")
        h = sorted_gather_qmm_a8(
            x_tok,
            gate_up["weight"],
            gate_up["scales"],
            gate_up["biases"],
            idx,
            row_map,
            group_size=_GROUP,
            bits=4,
            swiglu=True,
        )
        if h is None:
            return None
        return switch_mlp.down_proj(h, idx, sorted_indices=True)
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if name not in _warned:
            _warned.add(name)
            logger.warning(
                "routed A8 gate/up raised %s and fell back to A16", name, exc_info=True
            )
        return None


__all__ = [
    "sorted_gather_qmm_a8",
    "supports",
    "tag_routed_a8_modules",
    "try_routed_a8",
]
