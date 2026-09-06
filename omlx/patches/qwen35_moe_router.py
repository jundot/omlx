# SPDX-License-Identifier: Apache-2.0
"""Fused MoE router top-k for Qwen3.5/3.6 (and qwen3-next family) decode.

The routing chain after the gate linear — full softmax over the expert
count, ``argpartition`` top-k, ``take_along_axis``, top-k renormalize —
is five tiny-tensor ops per MoE layer per token. On the 256-expert
Qwen3.6-35B-A3B that chain measures ~51 us per layer at decode against a
~5 us fused launch: x40 layers it is ~2 ms of a ~9 ms decode step.

The gate linear and the softmax stay composed (the softmax output IS the
selection key the composed ``argpartition`` reads, so reusing it makes
the selected SET match bit-for-bit); one launch per row then selects
top-k over the rounded probabilities — ties resolve to the HIGHEST
index, mlx argpartition's empirically pinned behavior at this shape —
and renormalizes over the selected k in fp32 with one final rounding.
Scores can differ from the composed sum/divide chain by reduction-order
ulp; the routed expert set never differs.

Only short rows route here (decode and spec-verify widths); prefill keeps
the composed chain, whose cost amortizes over the chunk.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_KERNEL = None
_ENGAGED_LOGGED = False
_MAX_ROWS = 8

_SOURCE = """
    // One threadgroup (a single simdgroup) per row; each lane owns
    // NE/32 experts. Input is the composed chain's OWN softmax output, so
    // the selection keys are bit-identical to what argpartition reads —
    // the selected set matches exactly: ties on equal rounded
    // probabilities resolve to the HIGHEST index, the empirically pinned
    // mlx argpartition behavior at this shape (see the parity test).
    // Renormalization
    // runs in fp32 with one rounding at the end (score values may differ
    // from the composed sum/divide by reduction-order ulp).
    constexpr uint PER = uint(NE) / 32;
    uint lane = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    const device T* g = probs + row * uint(NE);

    float vals[PER];
    for (uint i = 0; i < PER; ++i) {
        vals[i] = float(g[lane * PER + i]);
    }

    bool taken[PER];
    for (uint i = 0; i < PER; ++i) {
        taken[i] = false;
    }

    float sel_p[K];
    uint sel_i[K];
    for (uint j = 0; j < K; ++j) {
        float best = -INFINITY;
        uint best_i = uint(NE);
        for (uint i = 0; i < PER; ++i) {
            if (!taken[i] && vals[i] >= best) {
                best = vals[i];
                best_i = lane * PER + i;
            }
        }
        float gbest = simd_max(best);
        uint cand = (best == gbest) ? best_i : 0u;
        uint gbest_i = simd_max(cand);
        if (best == gbest && best_i == gbest_i) {
            taken[best_i - lane * PER] = true;
        }
        sel_p[j] = gbest;
        sel_i[j] = gbest_i;
    }

    if (lane == 0) {
        float total = 0.0f;
        for (uint j = 0; j < K; ++j) {
            total += sel_p[j];
        }
        float inv = 1.0f / total;
        for (uint j = 0; j < K; ++j) {
            indices[row * uint(K) + j] = sel_i[j];
            scores[row * uint(K) + j] = T(sel_p[j] * inv);
        }
    }
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen35_moe_router_topk",
            input_names=["probs"],
            output_names=["indices", "scores"],
            source=_SOURCE,
        )
    return _KERNEL


def fused_router_topk(probs, top_k: int):
    """top-k + renormalize over softmaxed gate probabilities.

    ``probs`` is [..., NE] (the composed chain's own softmax output);
    returns (indices [..., k] uint32, scores [..., k] in the input
    dtype), selection descending."""
    gates = probs
    shape = gates.shape
    rows = 1
    for d in shape[:-1]:
        rows *= d
    ne = shape[-1]
    inds, scores = _kernel()(
        inputs=[gates],
        template=[
            ("T", gates.dtype),
            ("NE", ne),
            ("K", top_k),
        ],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*shape[:-1], top_k), (*shape[:-1], top_k)],
        output_dtypes=[mx.uint32, gates.dtype],
    )
    return inds, scores


def router_eligible(x, num_experts: int) -> bool:
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    return (
        rows <= _MAX_ROWS
        and num_experts % 32 == 0
        and x.dtype in (mx.bfloat16, mx.float16)
    )


def apply_qwen35_moe_router_patch() -> bool:
    """Route short-row MoE gating through the fused top-k launch.

    Wraps ``Qwen3NextSparseMoeBlock.__call__`` (shared by qwen3_5_moe and
    qwen3-next in mlx-lm): the fast arm reuses the module's own gate,
    experts, and shared expert; prefill rows and sharded runs keep the
    original body untouched."""
    global _ENGAGED_LOGGED
    if not mx.metal.is_available():
        return False
    try:
        from mlx_lm.models import qwen3_next as q3n
    except ImportError:
        return False
    cls = getattr(q3n, "Qwen3NextSparseMoeBlock", None)
    if cls is None or getattr(cls, "_omlx_router_fused", False):
        return cls is not None

    orig_call = cls.__call__

    def patched_call(self, x):
        if (
            self.sharding_group is not None
            or not self.norm_topk_prob
            or not router_eligible(x, self.num_experts)
        ):
            return orig_call(self, x)
        try:
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            inds, scores = fused_router_topk(gates, self.top_k)

            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)

            shared_y = self.shared_expert(x)
            shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
            return y + shared_y
        except Exception:
            logger.warning(
                "fused MoE router failed; composed fallback", exc_info=True
            )
            return orig_call(self, x)

    cls.__call__ = patched_call
    cls._omlx_router_fused = True

    try:
        from mlx_vlm.models.qwen3_5_moe import language as vlm_moe
    except ImportError:
        vlm_moe = None
    vcls = getattr(vlm_moe, "Qwen3_5MoeSparseMoeBlock", None) if vlm_moe else None
    if vcls is not None and not getattr(vcls, "_omlx_router_fused", False):
        vlm_orig = vcls.__call__

        def vlm_patched_call(self, x, target_verify=False):
            if not router_eligible(x, self.num_experts):
                return vlm_orig(self, x, target_verify=target_verify)
            try:
                gates = vlm_moe._target_verify_linear(
                    self.gate, x, target_verify
                )
                gates = mx.softmax(gates, axis=-1, precise=True)
                inds, scores = fused_router_topk(gates, self.top_k)

                if getattr(self.switch_mlp, "_omlx_expert_streaming", False):
                    # The streaming pool chunks wide MTP verification routes
                    # as a whole GLU so each expert group is loaded once.
                    y = self.switch_mlp(x, inds)
                else:
                    y = vlm_moe._target_verify_switch_glu(
                        self.switch_mlp, x, inds, target_verify
                    )
                y = (y * scores[..., None]).sum(axis=-2)

                shared_y = self.shared_expert(x, target_verify)
                shared_y = (
                    mx.sigmoid(
                        vlm_moe._target_verify_linear(
                            self.shared_expert_gate, x, target_verify
                        )
                    )
                    * shared_y
                )
                return y + shared_y
            except Exception:
                logger.warning(
                    "fused MoE router (vlm) failed; composed fallback",
                    exc_info=True,
                )
                return vlm_orig(self, x, target_verify=target_verify)

        vcls.__call__ = vlm_patched_call
        vcls._omlx_router_fused = True

    if not _ENGAGED_LOGGED:
        _ENGAGED_LOGGED = True
        logger.info("Qwen MoE fused router top-k patch applied")
    return True
