# SPDX-License-Identifier: Apache-2.0
"""Fused Metal kernel for a GatedDeltaNet T==1 decode step.

Profiling on M2 Ultra (Ornith-1.0-35B-4bit, B=1 decode) found the 30
linear-attention (GatedDeltaNet) layers cost ~6.28ms/step, of which only
~1.9ms is the recurrent delta-rule scan (``mlx_lm.models.gated_delta.
gated_delta_kernel``, itself a hand-written ``mx.fast.metal_kernel``). The
remaining ~3.75ms is ~15 tiny dispatches per layer: the causal depthwise
conv1d + SiLU, the q/k/v split, the q/k RMS-normalization, and the beta/g
gating prep (``sigmoid`` + ``compute_g``'s exp/softplus/exp chain).

First attempt (kept for the record, see git history / task report): fusing
just that "glue" into its own single ``mx.fast.metal_kernel`` and keeping
the existing scan kernel as a second, unmodified dispatch. Measured
honestly, this did NOT hit the perf gate -- MLX's own native ops (conv1d,
``mx.fast.rms_norm``, the ``mx.compile``d ``compute_g``) already batch
cheaply inside one lazy Mode-A graph, so a hand-written custom kernel
covering the same ops was a wash (sometimes slightly slower) once its own
per-call Python/argument-marshaling overhead is counted. The dominant real
cost at B=1 turned out to be the *fixed per-custom-kernel-dispatch overhead*
itself (empirically ~20-50us/call on this hardware), not the glue's op
count -- so two custom kernels (glue + scan) cost about the same as zero
custom kernels for the glue (native ops) plus one for the scan.

This module instead fuses conv+silu+split+q/k-RMSNorm+beta/g-gating
*directly into the scan kernel's own dispatch*, eliminating the second
custom-kernel call entirely (down to ONE dispatch total for everything
except the ``nn.Linear`` matmuls and the final output ``RMSNormGated``).
It reuses the scan kernel's exact grid/threadgroup layout and inner-loop
math verbatim; the new work is computed per-thread as registers (never
materialized as intermediate q/k/v/beta/g tensors), which also removes the
HBM round-trip the two-kernel design paid for those six tensors.

Grid layout (unchanged from ``mlx_lm.models.gated_delta.gated_delta_kernel``):
``grid=(32, Dv, B*Hv)``, ``threadgroup=(32, 4, 1)`` -- one thread per
``(dk-lane, dv, b*Hv+hv)``. Per thread:

  - q/k: ``head_k_dim`` is covered by the 32 dk-lanes (``n_per_t =
    head_k_dim // 32`` values per lane, exactly mirroring the scan's own
    ``s_idx`` indexing). Each lane computes conv1d+SiLU for its own
    ``n_per_t`` channels of its head, then a single ``simd_sum`` across the
    32 lanes (one simdgroup, no cross-simdgroup reduction or
    threadgroup_barrier needed) gives the full RMS-norm denominator for
    that head. This work is redundant across the ``Dv`` different
    threadgroups in the grid's y-dimension that all share the same
    ``(b, hv)`` (q/k don't depend on ``dv``) and across the ``Hv/Hk``
    v-heads that share a k/q head -- cheap ALU work traded for zero
    synchronization, which is the right trade on a GPU that is
    launch/bandwidth-bound at this tensor size, not compute-bound.
  - v: a single conv1d+SiLU value for this thread's own ``(b, hv, dv)`` --
    no reduction needed (v has no norm).
  - beta/g: recomputed redundantly per thread from the tiny per-head
    scalars -- five cheap float ops, far below the cost of any
    synchronization that would be needed to share it instead.
  - conv cache update: exactly one designated thread per output channel
    writes ``cache[0]``'s shifted window (guarded so the ``Hk*2 + Hv``
    channel groups are each written by exactly one thread, not the
    redundant copies above).
  - scan: the existing kernel's inner-loop body verbatim (decay, delta,
    output, state write-back), just fed q/k/v/beta/g from registers
    instead of global-memory reads.

The output RMSNormGated (+ z-gate) stays OUTSIDE this kernel and runs as
the native ``mx.fast.rms_norm`` op: it needs a reduction over the full 128
``head_v_dim`` values per ``(b, hv)``, which live in ``Dv/4 = 32`` different
threadgroups in this grid (each only owns a 4-wide slice of ``dv``) --
folding that reduction in would need either a threadgroup of 32*128=4096
threads (over the ~1024 Metal per-threadgroup limit) or a second pass, so
it remains the native op (itself already a fused/optimized primitive, not
naive multi-op) rather than a third custom dispatch.

Specialized to conv_kernel_size == 4 (3 history steps + 1 new activation,
unrolled) -- the only value this model family ships.
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = """
    auto n = thread_position_in_grid.z;        // b * Hv + hv_idx
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    constexpr int hv_per_hk = Hv / Hk;
    auto hk_idx = hv_idx / hv_per_hk;
    constexpr int n_per_t = Dk / 32;

    auto dk_lane = thread_position_in_threadgroup.x;   // 0 .. 31
    auto dv_idx  = thread_position_in_grid.y;          // 0 .. Dv-1

    constexpr uint KeyDim   = (uint)Hk * (uint)Dk;
    constexpr uint ValueDim = (uint)Hv * (uint)Dv;
    constexpr uint ConvDim  = 2u * KeyDim + ValueDim;

    // Exactly one (dv_idx, hv_idx-representative) thread group writes each
    // q/k conv-cache channel, even though every thread recomputes q/k.
    bool is_conv_writer = (dv_idx == 0) && (hv_idx % (uint)hv_per_hk == 0);

    // ---- q/k: conv1d + SiLU for this lane's n_per_t channels, then a
    // single simd_sum (one simdgroup = all 32 dk-lanes) for the RMS-norm
    // denominator over the full head_k_dim. ----
    float q_local[n_per_t];
    float k_local[n_per_t];
    float q_sumsq = 0.0f;
    float k_sumsq = 0.0f;

    for (int i = 0; i < n_per_t; ++i) {
        uint local_dk = (uint)n_per_t * dk_lane + (uint)i;
        uint qchan = hk_idx * (uint)Dk + local_dk;
        uint kchan = KeyDim + hk_idx * (uint)Dk + local_dk;

        auto qw  = conv1d_weight + qchan * 4u;
        auto qcs = conv_state_in + b_idx * 3u * ConvDim + qchan;
        float qacc = float(qw[0]) * float(qcs[0 * ConvDim])
                   + float(qw[1]) * float(qcs[1 * ConvDim])
                   + float(qw[2]) * float(qcs[2 * ConvDim]);
        float q_new = float(qkv[b_idx * ConvDim + qchan]);
        qacc += float(qw[3]) * q_new;
        float q_conv = qacc * (1.0f / (1.0f + metal::fast::exp(-qacc)));
        q_local[i] = q_conv;
        q_sumsq += q_conv * q_conv;

        auto kw  = conv1d_weight + kchan * 4u;
        auto kcs = conv_state_in + b_idx * 3u * ConvDim + kchan;
        float kacc = float(kw[0]) * float(kcs[0 * ConvDim])
                   + float(kw[1]) * float(kcs[1 * ConvDim])
                   + float(kw[2]) * float(kcs[2 * ConvDim]);
        float k_new = float(qkv[b_idx * ConvDim + kchan]);
        kacc += float(kw[3]) * k_new;
        float k_conv = kacc * (1.0f / (1.0f + metal::fast::exp(-kacc)));
        k_local[i] = k_conv;
        k_sumsq += k_conv * k_conv;

        if (is_conv_writer) {
            conv_state_out[(b_idx * 3u + 0u) * ConvDim + qchan] = qcs[1 * ConvDim];
            conv_state_out[(b_idx * 3u + 1u) * ConvDim + qchan] = qcs[2 * ConvDim];
            conv_state_out[(b_idx * 3u + 2u) * ConvDim + qchan] = static_cast<InT>(q_new);
            conv_state_out[(b_idx * 3u + 0u) * ConvDim + kchan] = kcs[1 * ConvDim];
            conv_state_out[(b_idx * 3u + 1u) * ConvDim + kchan] = kcs[2 * ConvDim];
            conv_state_out[(b_idx * 3u + 2u) * ConvDim + kchan] = static_cast<InT>(k_new);
        }
    }
    q_sumsq = simd_sum(q_sumsq);
    k_sumsq = simd_sum(k_sumsq);
    float q_rms_inv = metal::rsqrt(q_sumsq / float(Dk) + 1e-6f);
    float k_rms_inv = metal::rsqrt(k_sumsq / float(Dk) + 1e-6f);
    float inv_scale = metal::rsqrt(float(Dk));
    float q_scale = q_rms_inv * (1.0f / float(Dk));   // inv_scale^2 = 1/Dk
    float k_scale = k_rms_inv * inv_scale;

    // ---- v: single conv1d + SiLU value for this thread's (b, hv, dv) ----
    uint vchan = 2u * KeyDim + hv_idx * (uint)Dv + dv_idx;
    auto vw  = conv1d_weight + vchan * 4u;
    auto vcs = conv_state_in + b_idx * 3u * ConvDim + vchan;
    float vacc = float(vw[0]) * float(vcs[0 * ConvDim])
               + float(vw[1]) * float(vcs[1 * ConvDim])
               + float(vw[2]) * float(vcs[2 * ConvDim]);
    float v_new = float(qkv[b_idx * ConvDim + vchan]);
    vacc += float(vw[3]) * v_new;
    float v_val = vacc * (1.0f / (1.0f + metal::fast::exp(-vacc)));
    if (dk_lane == 0) {
        conv_state_out[(b_idx * 3u + 0u) * ConvDim + vchan] = vcs[1 * ConvDim];
        conv_state_out[(b_idx * 3u + 1u) * ConvDim + vchan] = vcs[2 * ConvDim];
        conv_state_out[(b_idx * 3u + 2u) * ConvDim + vchan] = static_cast<InT>(v_new);
    }

    // ---- gating: beta = sigmoid(b), g = exp(-exp(A_log) * softplus(a+dt_bias)) ----
    float b_val = float(b_lin[b_idx * (uint)Hv + hv_idx]);
    float beta_val = 1.0f / (1.0f + metal::fast::exp(-b_val));
    float xg = float(a_lin[b_idx * (uint)Hv + hv_idx]) + float(dt_bias[hv_idx]);
    float sp = metal::max(xg, 0.0f)
             + metal::log(1.0f + metal::fast::exp(-metal::fabs(xg)));
    float alog = float(A_log[hv_idx]);
    float g_val = metal::fast::exp(-metal::fast::exp(alog) * sp);

    // ---- scan: mlx_lm.models.gated_delta.gated_delta_kernel's inner-loop
    // body verbatim (T==1: the per-timestep loop collapses to one pass),
    // fed from the registers above instead of global-memory reads. ----
    auto i_state = delta_state_in  + (n * (uint)Dv + dv_idx) * (uint)Dk;
    auto o_state = delta_state_out + (n * (uint)Dv + dv_idx) * (uint)Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
        uint s_idx = (uint)n_per_t * dk_lane + (uint)i;
        state[i] = static_cast<float>(i_state[s_idx]);
    }

    float kv_mem = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] * g_val;
        kv_mem += state[i] * (k_local[i] * k_scale);
    }
    kv_mem = simd_sum(kv_mem);

    float delta = (v_val - kv_mem) * beta_val;

    float out = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] + (k_local[i] * k_scale) * delta;
        out += state[i] * (q_local[i] * q_scale);
    }
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
        y[n * (uint)Dv + dv_idx] = static_cast<InT>(out);
    }
    for (int i = 0; i < n_per_t; ++i) {
        uint s_idx = (uint)n_per_t * dk_lane + (uint)i;
        o_state[s_idx] = static_cast<StT>(state[i]);
    }
"""

_kernel = None
_kernel_build_attempted = False


def _get_kernel():
    global _kernel, _kernel_build_attempted
    if _kernel_build_attempted:
        return _kernel
    _kernel_build_attempted = True
    if not mx.metal.is_available():
        return None
    _kernel = mx.fast.metal_kernel(
        name="qwen3_5_gdn_decode_step",
        input_names=[
            "qkv",
            "conv_state_in",
            "conv1d_weight",
            "b_lin",
            "a_lin",
            "dt_bias",
            "A_log",
            "delta_state_in",
        ],
        output_names=["y", "conv_state_out", "delta_state_out"],
        source=_SOURCE,
        ensure_row_contiguous=True,
    )
    return _kernel


def is_available() -> bool:
    return (
        mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and _get_kernel() is not None
    )


def fused_gdn_decode_step(
    qkv: mx.array,
    conv_state_in: mx.array,
    conv1d_weight: mx.array,
    b_lin: mx.array,
    a_lin: mx.array,
    dt_bias: mx.array,
    A_log: mx.array,
    delta_state_in: mx.array,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
):
    """Fused conv1d+SiLU+split+q/k-RMSNorm+gating+delta-scan for T==1 decode.

    Args (all T==1, pre-flattened over the trivial sequence axis internally):
      qkv:            (B, 1, conv_dim)  -- in_proj_qkv(inputs)
      conv_state_in:  (B, 3, conv_dim)  -- cache[0] (conv_kernel_size-1 == 3)
      conv1d_weight:  (conv_dim, 4, 1)  -- GatedDeltaNet.conv1d.weight
      b_lin, a_lin:   (B, 1, num_v_heads) -- in_proj_b/a(inputs)
      dt_bias:        (num_v_heads,)
      A_log:          (num_v_heads,) float32
      delta_state_in: (B, num_v_heads, head_v_dim, head_k_dim) -- cache[1]

    Returns (y, conv_state_out, delta_state_out):
      y:              (B, 1, num_v_heads, head_v_dim)  dtype == qkv.dtype
                       (ready for ``GatedDeltaNet.norm(y, z)`` then out_proj)
      conv_state_out: (B, 3, conv_dim)                 dtype == qkv.dtype
      delta_state_out:(B, num_v_heads, head_v_dim, head_k_dim) dtype == delta_state_in.dtype

    Specialized to conv_kernel_size == 4 and num_v_heads % num_k_heads == 0.
    """
    kernel = _get_kernel()
    assert kernel is not None, "fused_gdn_decode_step requires Metal"
    assert conv1d_weight.shape[1] == 4, "kernel is specialized to conv_kernel_size=4"
    assert conv_state_in.shape[1] == 3, "kernel is specialized to conv_kernel_size=4"
    assert head_k_dim % 32 == 0, "kernel assumes a 32-wide simdgroup divides head_k_dim"
    assert num_v_heads % num_k_heads == 0

    B = qkv.shape[0]
    Hk, Hv = num_k_heads, num_v_heads
    Dk, Dv = head_k_dim, head_v_dim
    key_dim = Hk * Dk
    value_dim = Hv * Dv
    conv_dim = 2 * key_dim + value_dim
    assert conv1d_weight.shape[0] == conv_dim

    y, conv_state_out, delta_state_out = kernel(
        inputs=[
            qkv,
            conv_state_in,
            conv1d_weight,
            b_lin,
            a_lin,
            dt_bias,
            A_log,
            delta_state_in,
        ],
        template=[
            ("InT", qkv.dtype),
            ("StT", delta_state_in.dtype),
            ("Hk", Hk),
            ("Hv", Hv),
            ("Dk", Dk),
            ("Dv", Dv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[
            (B, 1, Hv, Dv),
            (B, 3, conv_dim),
            (B, Hv, Dv, Dk),
        ],
        output_dtypes=[qkv.dtype, qkv.dtype, delta_state_in.dtype],
    )
    return y, conv_state_out, delta_state_out


__all__ = ["fused_gdn_decode_step", "is_available"]
