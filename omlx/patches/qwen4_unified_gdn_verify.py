# Metal reduction and precision structure adapted from mlx-vlm #2105.
# Copyright © 2025 Prince Canuma. Used under the MIT License.
#
# Ported from the local unified Qwen4 full-recurrence verify kernel.
# Only q/k normalization changes: this experimental component preserves the
# deployed oMLX RMS/scaling semantics. It is not a direct-L2 correction.
"""Qualified RMS-compatible GDN verify fusion with narrow opt-in routing.

Graph construction is lazy, with no import-time Metal work. Callers must
qualify exact output, recurrent state and every rollback snapshot against the
current deployed generic prework route. The opt-in hook below admits only
the separately qualified B1/S3..4 regime. Input/output projections are outside this boundary.
"""
from functools import lru_cache
import mlx.core as mx

NUM_KEY_HEADS = 16
NUM_VALUE_HEADS = 48
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
VALUE_DIM = 6144
CONV_DIM = 10240
CONV_KERNEL = 4
MAX_VERIFY_STEPS = 8  # candidate bound; exactness requires its own gate
_THREADGROUP_Y_CANDIDATES = (32, 16, 8, 4)
DONOR_BASE_SHA256 = 'e6a52ffb3c0aaeb493beaf061b2956d254a2d5056442ea1668eaca4ebb68d5aa'
DONOR_VERIFY_SHA256 = 'a30a9b35ae95119cf2432016ca54d32425f454939a58352eda1dc9b5c63a7fb0'
DONOR_KERNEL_SHA256 = '2d5d84dc1869b7d74115605f2391e6a8e0767916db4f2214df19321641c90842'
NONNORMALIZATION_SHA256 = '9c9f767cbffff7062911cf3f10bb0268bdf8e1e97b6be77a24e8250925b4d010'
NORMALIZATION_START = 3520
NORMALIZATION_END = 4941

_HEADER = r"""
#include <metal_atomic>
template <typename U>
inline U mlx_sigmoid_precise(U x) {
  U e = static_cast<U>(metal::precise::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

template <typename U>
inline U mlx_sigmoid_fast(U x) {
  U e = static_cast<U>(metal::exp(metal::abs(x)));
  U y = static_cast<U>(1) / (static_cast<U>(1) + e);
  return (x < 0) ? y : (static_cast<U>(1) - y);
}

template <typename U>
inline U mlx_log1p_fast(U x) {
  float xf = float(x);
  float xp1 = 1.0f + xf;
  float out = xp1 == 1.0f ? xf : xf * (metal::log(xp1) / (xp1 - 1.0f));
  return static_cast<U>(out);
}

template <typename U>
inline U mlx_softplus_fast(U x) {
  if (metal::isnan(x))
    return metal::numeric_limits<U>::quiet_NaN();
  constexpr U inf = metal::numeric_limits<U>::infinity();
  U zero = static_cast<U>(0);
  U hi = metal::max(x, zero);
  U lo = metal::min(x, zero);
  return (lo == -inf || hi == inf)
      ? hi
      : (hi + mlx_log1p_fast(static_cast<U>(metal::exp(lo - hi))));
}

"""

_SOURCE = r"""
  const uint hv = threadgroup_position_in_grid.z;
  const uint hk = hv / RATIO;
  const uint lane = thread_position_in_threadgroup.x;
  const uint ty = thread_position_in_threadgroup.y;
  const uint tid = thread_index_in_threadgroup;

  constexpr int NT = 32 * TY;
  constexpr int NDK = DK / 32;
  constexpr int NDV = DV / TY;
  constexpr uint KD = (uint)(HK * DK);
  constexpr uint VD = (uint)(HV * DV);
  constexpr uint CD = 2u * KD + VD;
  constexpr uint KEEP = (uint)K - 1u;
  constexpr uint SNAPS = (uint)S - 1u;

  threadgroup float sq[DK];
  threadgroup float sk[DK];
  threadgroup T sq_squared[DK];
  threadgroup T sk_squared[DK];
  threadgroup float sv[DV];
  threadgroup float sy[DV];
  threadgroup float shr[4];

  device const float* si = recurrent_state + (size_t)hv * DV * DK;
  device float* so = recurrent_state_out + (size_t)hv * DV * DK;
  float st[NDV][NDK];
  for (int j = 0; j < NDV; ++j) {
    uint dv = ty + (uint)TY * (uint)j;
    for (int i = 0; i < NDK; ++i)
      st[j][i] = si[(size_t)dv * DK + NDK * lane + i];
  }

  const bool owns_shared = (hv % RATIO) == 0u;

  // Convolution window bookkeeping is token independent: publish the final
  // window (the next conv cache) and every intermediate window the layer
  // records as a restore point.
  for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {
    uint part = idx / (uint)DK;
    uint d = idx - part * (uint)DK;
    uint c = part == 0u ? hk * DK + d
           : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);
    if (part == 2u || owns_shared) {
      for (uint tap = 0; tap < KEEP; ++tap) {
        uint row = (uint)S + tap;
        conv_state_out[(size_t)tap * CD + c] =
            row < KEEP ? conv_state[(size_t)row * CD + c]
                       : qkv[(size_t)(row - KEEP) * CD + c];
      }
      for (uint p = 1; p <= SNAPS; ++p) {
        for (uint tap = 0; tap < KEEP; ++tap) {
          uint row = p + tap;
          conv_snapshots[((size_t)(p - 1u) * KEEP + tap) * CD + c] =
              row < KEEP ? conv_state[(size_t)row * CD + c]
                         : qkv[(size_t)(row - KEEP) * CD + c];
        }
      }
    }
  }

  for (uint t = 0; t < (uint)S; ++t) {
    for (uint idx = tid; idx < (uint)(2 * DK + DV); idx += NT) {
      uint part = idx / (uint)DK;
      uint d = idx - part * (uint)DK;
      uint c = part == 0u ? hk * DK + d
             : (part == 1u ? KD + hk * DK + d : 2u * KD + hv * DV + d);
      device const T* wc = conv_weight + (size_t)c * K;
      float acc = 0.0f;
      for (uint tap = 0; tap < (uint)K; ++tap) {
        uint row = t + tap;
        T xv = row < KEEP ? conv_state[(size_t)row * CD + c]
                          : qkv[(size_t)(row - KEEP) * CD + c];
        acc += float(xv) * float(wc[tap]);
      }
      T xb = static_cast<T>(acc);
      // nn.silu is reproduced by the fast sigmoid form on every finite bf16.
      T sl = xb * mlx_sigmoid_fast(xb);
      if (part == 0u) sq[d] = float(sl);
      else if (part == 1u) sk[d] = float(sl);
      else sv[d] = float(sl);
    }

    if (tid == 0u) {
      T av = a[t * HV + hv] + dt_bias[hv];
      T sp = mlx_softplus_fast(av);
      shr[2] = metal::precise::exp(
          -metal::precise::exp(float(A_log[hv])) * float(sp));
      // mx.sigmoid on bf16 is the precise form on every finite bf16 input;
      // the fast form differs on one (x ~ -6.85), which real activations reach.
      shr[3] = float(mlx_sigmoid_precise(b[t * HV + hv]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Preserve oMLX generic RMS/scaling operation order exactly: four
    // FP32 square accumulations per SIMD lane, SIMD sum, epsilon after
    // mean, BF16 RMS materialization, then a separate BF16 scalar scale.
    if (simdgroup_index_in_threadgroup == 0u) {
      float pq = 0.0f, pk = 0.0f;
      uint base = 4u * lane;
      for (int i = 0; i < 4; ++i) {
        float qv = sq[base + i], kv = sk[base + i];
        pq += qv * qv;
        pk += kv * kv;
      }
      pq = simd_sum(pq);
      pk = simd_sum(pk);
      if (lane == 0u) {
        shr[0] = metal::precise::rsqrt(pq / float(DK) + 1.0e-6f);
        shr[1] = metal::precise::rsqrt(pk / float(DK) + 1.0e-6f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const T qscale = T(0.0078125f);
    const T kscale = T(0.08838834764831845f);
    for (uint d = tid; d < (uint)DK; d += NT) {
      const T qrms = T(1) * T(sq[d] * shr[0]);
      const T krms = T(1) * T(sk[d] * shr[1]);
      sq[d] = float(T(qscale * qrms));
      sk[d] = float(T(kscale * krms));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* state_dst =
        t < SNAPS ? state_snapshots + ((size_t)t * HV + hv) * DV * DK : so;
    for (int j = 0; j < NDV; ++j) {
      uint dv = ty + (uint)TY * (uint)j;
      float kv = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        uint s = NDK * lane + i;
        st[j][i] = st[j][i] * shr[2];
        kv += st[j][i] * sk[s];
      }
      kv = simd_sum(kv);
      float delta = (sv[dv] - kv) * shr[3];
      float out = 0.0f;
      for (int i = 0; i < NDK; ++i) {
        uint s = NDK * lane + i;
        st[j][i] = st[j][i] + sk[s] * delta;
        out += st[j][i] * sq[s];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0u)
        sy[dv] = float(static_cast<T>(out));
      for (int i = 0; i < NDK; ++i)
        state_dst[(size_t)dv * DK + NDK * lane + i] = st[j][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index_in_threadgroup == 0u) {
      float po = 0.0f;
      uint base = 4u * lane;
      for (int i = 0; i < 4; ++i) po += sy[base + i] * sy[base + i];
      po = simd_sum(po);
      if (lane == 0u)
        shr[0] = metal::precise::rsqrt(po / (float)DV + norm_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < (uint)DV; d += NT) {
      T normalized = static_cast<T>(sy[d] * shr[0]);
      normalized = norm_weight[d] * normalized;
      // float32 sigmoid of a bf16-valued gate: the precise form matches
      // mx.sigmoid on every finite bf16 input; the fast form differs on ~1%.
      float x = float(normalized) *
                mlx_sigmoid_precise<float>(float(z[t * VD + hv * DV + d]));
      output[t * VD + hv * DV + d] = static_cast<T>(x);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""

@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="omlx_qwen4_unified_gdn_verify_rms",
        input_names=[
            "qkv",
            "z",
            "b",
            "a",
            "conv_state",
            "conv_weight",
            "A_log",
            "dt_bias",
            "recurrent_state",
            "norm_weight",
            "norm_eps",
        ],
        output_names=[
            "output",
            "conv_state_out",
            "recurrent_state_out",
            "state_snapshots",
            "conv_snapshots",
        ],
        header=_HEADER,
        source=_SOURCE,
        ensure_row_contiguous=True,
    )

def qwen4_unified_gdn_verify_rms(
    qkv,
    z,
    b,
    a,
    conv_state,
    conv_weight,
    A_log,
    dt_bias,
    recurrent_state,
    norm_weight,
    norm_eps: float,
    *,
    threadgroup_y: int,
):
    """Build the fused verify graph. Callers must run structural admission first.

    Returns ``(output, conv_state_out, recurrent_state_out, state_snapshots,
    conv_snapshots)``; ``state_snapshots[:, p]`` and ``conv_snapshots[:, p]``
    are the recurrent state and convolution window after ``p + 1`` tokens for
    ``p`` in ``range(S - 1)``.
    """
    if threadgroup_y not in _THREADGROUP_Y_CANDIDATES:
        raise ValueError(
            f"unsupported threadgroup_y {threadgroup_y}; "
            f"expected one of {_THREADGROUP_Y_CANDIDATES}"
        )
    steps = int(qkv.shape[1])
    # The dispatch and the probe are bounded by what the kernel is PROVEN to
    # compute, not by what production admits: ``admit_qwen4_fused_gdn_verify``
    # is the production gate, and a bench that widens it must still be able to
    # build the graph.
    if not 3 <= steps <= MAX_VERIFY_STEPS:
        raise ValueError(
            f"unsupported verify width {steps}; "
            f"expected 3..{MAX_VERIFY_STEPS}"
        )
    outputs = _kernel()(
        inputs=[
            qkv,
            z,
            b,
            a,
            conv_state,
            conv_weight,
            A_log,
            dt_bias,
            recurrent_state,
            norm_weight,
            float(norm_eps),
        ],
        template=[
            ("T", qkv.dtype),
            ("HK", NUM_KEY_HEADS),
            ("HV", NUM_VALUE_HEADS),
            ("DK", KEY_HEAD_DIM),
            ("DV", VALUE_HEAD_DIM),
            ("K", CONV_KERNEL),
            ("S", steps),
            ("TY", threadgroup_y),
            ("RATIO", NUM_VALUE_HEADS // NUM_KEY_HEADS),
        ],
        grid=(32, threadgroup_y, NUM_VALUE_HEADS),
        threadgroup=(32, threadgroup_y, 1),
        output_shapes=[
            (1, steps, VALUE_DIM),
            (1, CONV_KERNEL - 1, CONV_DIM),
            (1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
            (1, steps - 1, NUM_VALUE_HEADS, VALUE_HEAD_DIM, KEY_HEAD_DIM),
            (1, steps - 1, CONV_KERNEL - 1, CONV_DIM),
        ],
        output_dtypes=[qkv.dtype, qkv.dtype, mx.float32, mx.float32, qkv.dtype],
    )
    return tuple(outputs)


# Qualified deployment hook; the caller retains an explicit environment opt-out. The component is qualified only at these widths;
# wider blocks retain the existing route even though the raw kernel can build.
_RECEIPT = {"calls": 0, "declines": {}, "failures": 0}
_DISABLED = False


def reset_receipt():
    _RECEIPT.update(calls=0, declines={}, failures=0)


def receipt():
    return {"calls": _RECEIPT["calls"], "declines": dict(_RECEIPT["declines"]),
            "failures": _RECEIPT["failures"], "exception_fuse": _DISABLED}


def _decline(reason):
    counts = _RECEIPT["declines"]
    if reason not in counts and len(counts) >= 32:
        reason = "other"
    counts[reason] = counts.get(reason, 0) + 1
    return None


def _admission(module, inputs, mask, cache, gdn_sink):
    if type(module).__name__ != "Qwen4ExpGatedDeltaNet":
        return "not Qwen4ExpGatedDeltaNet"
    if module.training or getattr(module, "sharding_group", None) is not None:
        return "training or sharding"
    if tuple(inputs.shape) not in ((1, 3, 2560), (1, 4, 2560)) or inputs.dtype != mx.bfloat16:
        return "input not BF16 B1/S3..4/D2560"
    if mask is not None or gdn_sink is None or cache is None:
        return "mask or missing rollback sink/cache"
    from mlx_lm.models.cache import ArraysCache as LMArrayCache
    from mlx_vlm.models.qwen4_exp.cache import ArraysCache as Qwen4ArrayCache
    from omlx.cache.type_handlers import SizedArraysCache
    qualified_inner = (LMArrayCache, Qwen4ArrayCache)
    # Prefix restoration adds this concrete size-tracking wrapper. Admit only
    # one exact wrapper over an already qualified concrete cache, and retain
    # the outer object for all reads, commits and metadata advancement below.
    # Subclasses, unknown inners and nested wrappers keep the existing route.
    if not (type(cache) in qualified_inner or
            (type(cache) is SizedArraysCache and type(cache._inner) in qualified_inner)):
        return "cache is outside qualified concrete ArraysCache ABI"
    if getattr(cache, "lengths", None) is not None or getattr(cache, "left_padding", None) is not None:
        return "padded cache"
    if tuple(getattr(module, name, None) for name in
             ("num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim", "conv_kernel_size")) != (16, 48, 128, 128, 4):
        return "unsupported head/convolution geometry"
    if module.norm.activation != "sigmoid" or module.norm.eps != 1e-6:
        return "unqualified norm activation/epsilon"
    if getattr(module.conv1d, "bias", None) is not None:
        return "convolution bias"
    values = ((cache[0], (1, 3, 10240), mx.bfloat16),
              (cache[1], (1, 48, 128, 128), mx.float32),
              (module.conv1d.weight, (10240, 4, 1), mx.bfloat16),
              (module.A_log, (48,), mx.bfloat16), (module.dt_bias, (48,), mx.bfloat16),
              (module.norm.weight, (128,), mx.bfloat16))
    if any(tuple(getattr(value, "shape", ())) != shape or getattr(value, "dtype", None) != dtype
           for value, shape, dtype in values):
        return "state/parameter shape or dtype"
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return "Metal GPU unavailable"
    return None


def try_fused_rms(module, inputs, mask, cache, gdn_sink):
    """Return the candidate output or decline without changing cache ownership.

    The legacy rollback sink remains fully populated: its q/k/v graph is lazy
    and is only needed by older replay consumers. This deliberately preserves
    the existing fallback contract instead of adding a new receipt variant.
    """
    global _DISABLED
    if _DISABLED:
        return _decline("exception fuse")
    reason = _admission(module, inputs, mask, cache, gdn_sink)
    if reason:
        return _decline(reason)
    from mlx_vlm.models.qwen3_5 import language as q35
    from . import qwen35_gdn_prework as generic
    import logging
    steps = inputs.shape[1]
    conv_state, recurrent_state = cache[0], cache[1]
    try:
        qkv, z, b, a = q35._target_verify_linears(
            (module.in_proj_qkv, module.in_proj_z, module.in_proj_b, module.in_proj_a), inputs, True)
        if any(tuple(value.shape) != shape or value.dtype != mx.bfloat16 for value, shape in
               ((qkv, (1, steps, 10240)), (z, (1, steps, 6144)),
                (a, (1, steps, 48)), (b, (1, steps, 48)))):
            return _decline("projected shape or dtype")
        flat, next_conv, next_state, snapshots, _conv_snapshots = qwen4_unified_gdn_verify_rms(
            qkv, z, b, a, conv_state, module.conv1d.weight, module.A_log,
            module.dt_bias, recurrent_state, module.norm.weight, module.norm.eps,
            threadgroup_y=32)
        # Preserve all legacy replay operands without evaluating their graph.
        q_scale, k_scale = generic._qwen4_scales(module)
        q, k, v, _ = generic.gdn_prework_fused(qkv, conv_state, module.conv1d.weight,
            q_scale, k_scale, 16, 48, 128, 128)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        result = q35._target_verify_linear(module.out_proj, flat, True)
    except Exception:
        _DISABLED = True
        _RECEIPT["failures"] += 1
        logging.getLogger(__name__).warning("RMS-compatible Qwen4 GDN candidate failed; retaining existing route", exc_info=True)
        return _decline("graph construction exception")
    # Commit lies outside the recoverable graph-construction catch. If a
    # custom sink/cache setter fails here, fail-stop the request; falling
    # through after partial ownership transfer would process the slab twice.
    gdn_sink.append((q, k, v, a, b, module.A_log, module.dt_bias, recurrent_state,
                     None, conv_input, 4, snapshots))
    cache[0], cache[1] = next_conv, next_state
    # Match the existing route's commit order, including its cache metadata.
    # These concrete ArraysCache methods are no-ops for the admitted unpadded
    # lane, but still call the ABI. Any failure propagates without fallback.
    cache.advance(steps)
    q35._qwen3_5_advance_left_padding_info(cache, steps)
    q35._qwen3_5_advance_lengths_info(cache, steps)
    _RECEIPT["calls"] += 1
    return result
