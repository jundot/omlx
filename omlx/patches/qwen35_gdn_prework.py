# SPDX-License-Identifier: Apache-2.0
#
# Kernel adapted from mlx-serve (src/transformer.zig, GDN_PREWORK_SOURCE),
# itself a port of the mlxfast-challenge qwen35_packed_gdn_prework kernel.
"""Fused GDN prework for Qwen3.5/3.6 MTP verify widths (S in 3..9).

The composed target-verify prework in mlx-vlm's ``Qwen3_5GatedDeltaNet`` —
conv-state concat + depthwise conv1d + SiLU + q/k/v split + reshapes + two
ones-weight RMS norms + two scalar scales + the next conv-state slice — is
~10 small dispatches per GDN layer per verify cycle. On the 27B that is 48
layers x ~0.29 ms (measured sync-mode at S=4 on M3 Ultra), the largest
remaining verify-cycle cost after the attention split.

One Metal launch replaces the whole chain. Numerics notes carried from the
donor kernel: the in-kernel sigmoid uses MLX's own unary formula
(exp-of-abs), which the challenge swept bit-exact over all finite bf16
inputs; the RMS applies the ones-weight rounding then the separate scalar
multiply's rounding — the composed chain's two casts.

S >= 3 is a HARD gate: the next conv state is copied from qkv rows only,
which is wrong when a state row would still come from the OLD conv state.
Only the target-verify arm routes here; decode (S=1) and prefill keep the
stock path.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_ENGAGED_LOGGED = False
_KERNEL = None

_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint logical_head = threadgroup_position_in_grid.z;
    constexpr uint q_heads = uint(HK);
    constexpr uint k_head_base = uint(HK);
    constexpr uint v_head_base = 2 * uint(HK);
    bool is_q = logical_head < q_heads;
    bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
    uint head = is_q ? logical_head
               : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
    uint channel_base = is_q ? head * uint(DK)
                       : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                               : 2 * uint(HK) * uint(DK) + head * uint(DV));
    T activated[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint channel = channel_base + lane * 4 + i;
        float acc = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            uint input_row = row + tap;
            const T xv = input_row < uint(NKEEP)
                ? conv_state[input_row * uint(C) + channel]
                : qkv[(input_row - uint(NKEEP)) * uint(C) + channel];
            acc += float(xv) * float(conv_w[channel * 4 + tap]);
        }
        const T conv = T(acc);
        T sy = T(1) / (T(1) + metal::exp(metal::abs(conv)));
        const T act = conv * ((conv < T(0)) ? sy : T(1) - sy);
        activated[i] = act;
        float value = float(act);
        sumsq += value * value;
    }
    if (is_q || is_k) {
        sumsq = simd_sum(sumsq);
        float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
        const T scale = is_q ? q_scale : k_scale;
        uint out_base = (row * uint(HK) + head) * uint(DK) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            const T rms = T(1) * T(float(activated[i]) * inv);
            const T value = scale * rms;
        if (is_q) {
            q_out[out_base + i] = value;
        } else {
            k_out[out_base + i] = value;
        }
        }
    } else {
        uint out_base = (row * uint(HV) + head) * uint(DV) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            v_out[out_base + i] = activated[i];
        }
    }
    if (row + uint(NKEEP) >= uint(S)) {
        uint state_row = row + uint(NKEEP) - uint(S);
        uint raw_base = row * uint(C) + channel_base + lane * 4;
        uint state_base = state_row * uint(C) + channel_base + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            conv_out[state_base + i] = qkv[raw_base + i];
        }
    }
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen35_gdn_prework",
            input_names=["qkv", "conv_state", "conv_w", "q_scale", "k_scale"],
            output_names=["q_out", "k_out", "v_out", "conv_out"],
            source=_SOURCE,
        )
    return _KERNEL


def gdn_prework_fused(qkv, conv_state, conv_w, q_scale, k_scale, hk, hv, dk, dv):
    """One fused dispatch. qkv [1,S,C], conv_state [1,3,C], conv_w [C,4,1]."""
    s_len = qkv.shape[1]
    c_dim = qkv.shape[2]
    outs = _kernel()(
        inputs=[qkv, conv_state, conv_w, q_scale, k_scale],
        template=[
            ("T", qkv.dtype),
            ("HK", hk),
            ("HV", hv),
            ("DK", dk),
            ("DV", dv),
            ("NKEEP", 3),
            ("C", c_dim),
            ("S", s_len),
        ],
        grid=(32, s_len, 2 * hk + hv),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (1, s_len, hk, dk),
            (1, s_len, hk, dk),
            (1, s_len, hv, dv),
            (1, 3, c_dim),
        ],
        output_dtypes=[qkv.dtype] * 4,
    )
    return outs


def apply_qwen35_gdn_prework_patch() -> bool:
    """Route the batch-1 target-verify GDN prework through the fused kernel.

    Wraps ``Qwen3_5GatedDeltaNet.__call__``: the fused arm re-implements the
    verify forward using the module's own weights and the SAME module-level
    helpers (recurrence, sink capture, cache advance); every other shape
    falls through to the original untouched. Any failure inside the fused
    arm falls back to the original call for that layer permanently.
    """
    global _PATCHED
    if _PATCHED:
        return True
    if not mx.metal.is_available():
        return False

    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:
        return False

    needed = (
        "Qwen3_5GatedDeltaNet",
        "_target_verify_linears",
        "_target_verify_linear",
        "_gated_delta_update_verify_decode",
        "_qwen3_5_advance_left_padding_info",
        "_qwen3_5_advance_lengths_info",
    )
    if not all(hasattr(q35, n) for n in needed):
        logger.debug("gdn prework: upstream seams missing; patch skipped")
        return False

    cls = q35.Qwen3_5GatedDeltaNet
    if getattr(cls, "_omlx_gdn_prework_patched", False):
        _PATCHED = True
        return True

    orig_call = cls.__call__
    disabled = {"flag": False}

    def _eligible(self, inputs, mask, cache, gdn_sink, s_len):
        if gdn_sink is None or cache is None:
            return False
        if inputs.shape[0] != 1 or not (3 <= s_len <= 9):
            return False
        if mask is not None:
            return False
        if inputs.dtype != mx.bfloat16:
            return False
        if getattr(self, "conv_kernel_size", 0) != 4:
            return False
        if self.head_k_dim != 128 or self.head_v_dim != 128:
            return False
        if getattr(cache, "lengths", None) is not None:
            return False
        conv_state = cache[0]
        if conv_state is None or conv_state.shape[0] != 1:
            return False
        if conv_state.dtype != mx.bfloat16:
            return False
        if self.conv1d.weight.dtype != mx.bfloat16:
            return False
        return getattr(self.conv1d, "bias", None) is None

    def patched_call(self, inputs, mask=None, cache=None, gdn_sink=None,
                     target_verify=False):
        S = inputs.shape[1]
        if disabled["flag"] or not _eligible(self, inputs, mask, cache, gdn_sink, S):
            return orig_call(self, inputs, mask=mask, cache=cache,
                             gdn_sink=gdn_sink, target_verify=target_verify)
        # Captured before the try block (and before any fallible op) so the
        # except branch always has the pre-mutation state to restore, even
        # if the failure happens before this point is normally reached.
        conv_state = cache[0]
        recurrent_state = cache[1]
        sink_len = len(gdn_sink) if gdn_sink is not None else 0
        try:
            B = 1
            mixed_qkv, z, b, a = q35._target_verify_linears(
                (self.in_proj_qkv, self.in_proj_z, self.in_proj_b,
                 self.in_proj_a),
                inputs,
                True,
            )
            z = z.reshape(B, S, -1, self.head_v_dim)

            if not hasattr(self, "_omlx_gdn_scales"):
                inv = self.head_k_dim ** -0.5
                self._omlx_gdn_scales = (
                    mx.array(inv * inv, dtype=mx.bfloat16),
                    mx.array(inv, dtype=mx.bfloat16),
                )
            q_scale, k_scale = self._omlx_gdn_scales

            q, k, v, new_conv_state = gdn_prework_fused(
                mixed_qkv,
                conv_state,
                self.conv1d.weight,
                q_scale,
                k_scale,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
            cache[0] = new_conv_state

            state = recurrent_state
            if state is not None and state.shape[0] != B:
                state = None
            initial_state = state
            out, state, intermediate_states = (
                q35._gated_delta_update_verify_decode(
                    q, k, v, a, b, self.A_log, self.dt_bias, state, None,
                    use_kernel=not self.training,
                )
            )
            # The rollback capture wants the conv INPUT window; build it
            # lazily — it is only evaluated on a partial accept.
            conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
            gdn_sink.append(
                (
                    q, k, v, a, b, self.A_log, self.dt_bias, initial_state,
                    None, conv_input, self.conv_kernel_size,
                    intermediate_states,
                )
            )

            cache[1] = state

            global _ENGAGED_LOGGED
            if not _ENGAGED_LOGGED:
                _ENGAGED_LOGGED = True
                logger.info(
                    "[gdn-prework] fused verify prework engaged (S=%d)", S
                )

            out = self.norm(out, z)
            result = q35._target_verify_linear(
                self.out_proj, out.reshape(B, S, -1), True
            )
            # Deferred to the very end, after every fallible step has
            # succeeded: advancing here and then hitting an exception below
            # would double-advance once the except branch below falls back
            # to orig_call, which advances the cache itself.
            if hasattr(cache, "advance"):
                cache.advance(S)
                q35._qwen3_5_advance_left_padding_info(cache, S)
                q35._qwen3_5_advance_lengths_info(cache, S)
            return result
        except Exception:
            disabled["flag"] = True
            # cache[0] may already hold the fused kernel's post-update conv
            # state (set above, before the delta update that can raise);
            # orig_call's stock path recomputes conv from scratch and
            # expects the pre-call state, so restore it before falling back
            # -- otherwise it silently double-applies the conv step.
            cache[0] = conv_state
            # A late failure can also happen after the recurrent delta state
            # has been committed to cache[1]. The stock fallback consumes the
            # same tokens again, so it must start from the original recurrent
            # state as well or silently double-apply the delta update.
            cache[1] = recurrent_state
            # A failure after the sink append (norm/out_proj) would leave
            # the fused entry in place while orig_call appends the stock
            # one -- two entries for one layer call shifts every later
            # layer's rollback capture. Drop anything this call appended.
            if gdn_sink is not None:
                del gdn_sink[sink_len:]
            logger.warning(
                "gdn prework fused arm failed; reverting to stock for the "
                "rest of the process",
                exc_info=True,
            )
            return orig_call(self, inputs, mask=mask, cache=cache,
                             gdn_sink=gdn_sink, target_verify=target_verify)

    cls.__call__ = patched_call
    cls._omlx_gdn_prework_patched = True
    _PATCHED = True
    logger.info("Qwen3.5/3.6 fused GDN verify prework patch applied")
    return True
