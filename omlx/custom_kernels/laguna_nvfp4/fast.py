"""NVFP4 (E4M3) decode kernels for oMLX, ported from the
Layr-Labs/mlxfast-challenge Laguna runtime.

Public API
----------
has_native()            -> bool     native C++ extension available

shared_nvfp4_swiglu_qmv(x, w, scales, stream=None) -> mx.array
    Shared-expert fused gate/up NVFP4 QMV with in-kernel SwiGLU: one kernel
    computes silu(gate) * up over the fused [gate; up] NVFP4 plane. Falls
    back to stock mx.quantized_matmul(mode="nvfp4") + swiglu when the native
    extension is unavailable.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _detach_import_error(exc: Exception) -> Exception:
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None
    return exc


try:
    _ext = importlib.import_module("omlx.custom_kernels.laguna_nvfp4._ext")
except Exception as exc:  # pragma: no cover - depends on local native build
    _ext = None
    _IMPORT_ERROR: Exception | None = _detach_import_error(exc)
else:
    _IMPORT_ERROR = None


def _verify_abi(ext, import_error):
    """Disable native symbols when the nanobind ABI tag does not match mlx."""
    if ext is None:
        return ext, import_error
    probe = getattr(ext, "abi_probe", None)
    if probe is None:
        return ext, import_error
    try:
        probe(mx.zeros((1,)))
    except TypeError as exc:
        logger.warning(
            "%s: native kernels disabled — nanobind ABI mismatch "
            "(rebuild against the installed mlx).",
            __name__,
        )
        return None, _detach_import_error(exc)
    return ext, import_error


_ext, _IMPORT_ERROR = _verify_abi(_ext, _IMPORT_ERROR)


def has_native() -> bool:
    """True when the compiled C++ extension is loaded and ABI-verified."""
    return _ext is not None


def is_native_available() -> bool:
    """Alias for has_native() — matches omlx kernel package convention."""
    return has_native()


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def has_symbol(name: str) -> bool:
    return _ext is not None and hasattr(_ext, name)


def _swiglu(gate: mx.array, up: mx.array) -> mx.array:
    gate32 = gate.astype(mx.float32)
    return (gate32 * mx.sigmoid(gate32) * up.astype(mx.float32)).astype(
        gate.dtype
    )


def shared_nvfp4_swiglu_qmv(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    stream=None,
) -> mx.array:
    """Shared-expert fused gate/up NVFP4 QMV with in-kernel SwiGLU.

    Parameters
    ----------
    x      : [K] bf16                — shared-expert input (K = 2048)
    w      : [2N, K*4/32] uint8      — fused [gate; up] NVFP4 codes (N = 512)
    scales : [2N, K/16] uint8        — E4M3 group-16 scales

    Returns [N] bf16 silu(gate) * up.
    """
    if _ext is not None and has_symbol("shared_nvfp4_swiglu_qmv"):
        return _ext.shared_nvfp4_swiglu_qmv(x, w, scales, stream=stream)
    # Fallback: stock mlx nvfp4 matmul over the fused gate/up plane + swiglu.
    gate_up = mx.quantized_matmul(
        x[None, :],
        w,
        scales=scales,
        transpose=True,
        group_size=16,
        bits=4,
        mode="nvfp4",
        stream=stream,
    )  # [1, 2N]
    split = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :split], gate_up[..., split:]
    return _swiglu(gate, up).squeeze(0)


def _pair_interleave_fused(gate, up, rows=512):
    """Build the routed-expert pair-interleaved fused plane from stacked
    gate/up weight tensors (each [E, rows, K/8] uint32 or [E, rows, K/16]
    uint8 scales): per expert, 32-row [gate; up] pairs —
    plane[p*64 + r] = gate[p*32 + r], plane[p*64 + 32 + r] = up[p*32 + r].
    Mirrors the challenge's fused routed plane layout.
    """
    E = gate.shape[0]
    pairs = rows // 32
    flat = mx.reshape(gate, (E, pairs, 32) + gate.shape[2:])
    flat_u = mx.reshape(up, (E, pairs, 32) + up.shape[2:])
    inter = mx.stack([flat, flat_u], axis=2)  # [E, pairs, 2, 32, ...]
    return mx.reshape(inter, (E, rows * 2) + gate.shape[2:])


def routed_nvfp4_swiglu_qmv(
    input_x: mx.array,
    fused_weight: mx.array,
    fused_scales: mx.array,
    indices: mx.array,
    stream=None,
) -> mx.array:
    """Routed-expert fused gate/up NVFP4 QMV with in-kernel SwiGLU
    (challenge lagunaRoutedSwiGLUQMVKernel).

    Parameters
    ----------
    input_x      : [K] bf16                — routed-expert input (K = 2048)
    fused_weight : [E, 2N, K/8] uint32     — per-expert pair-interleaved
                                             [gate; up] planes (N = 512)
    fused_scales : [E, 2N, K/16] uint8     — E4M3 group-16 scales
    indices      : [R] uint32              — top-R routed expert ids (R = 8)

    Returns [R*N] bf16 per-slot silu(gate) * up.
    """
    if _ext is not None and has_symbol("routed_nvfp4_swiglu_qmv"):
        return _ext.routed_nvfp4_swiglu_qmv(
            input_x, fused_weight, fused_scales, indices, stream=stream)
    # Fallback: un-interleave the pair plane into [gate; up] rows and use
    # the stock nvfp4 matmul + swiglu per routed expert.
    N = fused_weight.shape[1] // 2
    pairs = N // 32
    fw = fused_weight.reshape(indices.shape[0] * 0 + fused_weight.shape[0]
                              if False else fused_weight.shape[0],
                              pairs, 2, 32, fused_weight.shape[-1])
    fs = fused_scales.reshape(fused_scales.shape[0], pairs, 2, 32,
                              fused_scales.shape[-1])
    gate_w = mx.concatenate([fw[:, p, 0] for p in range(pairs)], axis=1)
    up_w = mx.concatenate([fw[:, p, 1] for p in range(pairs)], axis=1)
    gate_s = mx.concatenate([fs[:, p, 0] for p in range(pairs)], axis=1)
    up_s = mx.concatenate([fs[:, p, 1] for p in range(pairs)], axis=1)
    outs = []
    for slot in range(indices.shape[0]):
        e = int(indices[slot])
        g = mx.quantized_matmul(
            input_x[None, :], gate_w[e], scales=gate_s[e], transpose=True,
            group_size=16, bits=4, mode="nvfp4", stream=stream)
        u = mx.quantized_matmul(
            input_x[None, :], up_w[e], scales=up_s[e], transpose=True,
            group_size=16, bits=4, mode="nvfp4", stream=stream)
        outs.append(_swiglu(g, u)[0])
    return mx.concatenate(outs)




def halved_group32_scale_plane(
    scales: mx.array, allowed_flat_pairs: list[int] | None = None
) -> mx.array | None:
    """Build the halved group-32 scale plane (challenge
    lagunaHalvedGroup32ScalePlane): one byte per 32 weights, taken as the
    even group-16 bytes in flat order, prefixed by a 128-byte patch header
    holding the allowed odd-byte exceptions. Returns None when any
    non-allowed pair's even/odd bytes differ (the certificate fails).
    """
    allowed = allowed_flat_pairs or []
    flat = scales.astype(mx.uint8).reshape(-1)
    pair_count = flat.size // 2
    pairs = flat.reshape(pair_count, 2)
    even = pairs[:, 0]
    odd = pairs[:, 1]
    mismatch = (even != odd).astype(mx.int32)
    violations = int(mx.sum(mismatch))
    for idx in allowed:
        violations -= int(mismatch[idx])
    if violations != 0:
        return None
    header = mx.zeros((128,), mx.uint8)
    for slot, idx in enumerate(allowed):
        if slot >= 128:
            break
        header = mx.where(
            mx.arange(128) == slot,
            odd[idx].astype(mx.uint8),
            header,
        )
    return mx.concatenate([header, even.astype(mx.uint8)])

def routed_nvfp4_down_reduce(
    activated: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    indices: mx.array,
    router_weights: mx.array,
    stream=None,
) -> mx.array:
    """Routed-expert down_proj + weighted reduction fused in one kernel
    (challenge lagunaRoutedDownReduceKernel).

    Parameters
    ----------
    activated      : [R*K2] bf16          — per-slot swiglu outputs
    down_weight    : [E, N, K2/8] uint32  — per-expert NVFP4 down planes
    down_scales    : [128 + E*N*16] uint8 — halved group-32 scale planes
    indices        : [R] uint32           — routed expert ids
    router_weights : [R] float32          — routed scores

    Returns [N] bf16 sum_slots(act * w) * 2.5.
    """
    if _ext is not None and has_symbol("routed_nvfp4_down_reduce"):
        return _ext.routed_nvfp4_down_reduce(
            activated, down_weight, down_scales, indices, router_weights,
            stream=stream,
        )
    # Fallback: de-halve the group-32 plane back to group-16 and run the
    # stock nvfp4 matmul per routed expert, then the weighted reduction.
    N, K2 = 2048, 512
    E = down_weight.shape[0]
    R = indices.shape[0]
    halved = mx.reshape(down_scales[128:], (E, N, 16))
    g16 = mx.repeat(halved, 2, axis=-1)  # [E, N, 32]
    # pair-0 exception: row 0 groups 0/1 — the odd byte is the header value
    h0 = down_scales[0].astype(mx.uint8)
    g16 = mx.where(
        mx.reshape(mx.arange(N * 32, dtype=mx.uint32), (1, N, 32)) == 1,
        mx.broadcast_to(h0[None, None, None], g16.shape),
        g16,
    )
    total = None
    for slot in range(R):
        e = int(indices[slot])
        x_slot = activated[slot * K2 : (slot + 1) * K2]
        down = mx.quantized_matmul(
            x_slot[None, :], down_weight[e], scales=g16[e], transpose=True,
            group_size=16, bits=4, mode="nvfp4", stream=stream,
        ).squeeze(0)
        term = down * router_weights[slot]
        total = term if total is None else total + term
    return (total * mx.array(2.5, mx.float32)).astype(mx.bfloat16)


def _qk_norm_rope_fallback(
    raw_queries, raw_keys, query_weight, key_weight, angles,
    q_heads, k_heads, rotary_pairs, mscale=None,
):
    """Stock-op fallback for the fused QK-norm+RoPE kernels: rms_norm + the
    pair rotation. Approximate (few-ulp) vs the kernel's stepwise bf16."""
    dim = 128
    norm_q = mx.fast.rms_norm(
        raw_queries.reshape(q_heads, dim), query_weight, 1e-6)
    norm_k = mx.fast.rms_norm(
        raw_keys.reshape(k_heads, dim), key_weight, 1e-6)
    cos = angles[:rotary_pairs]
    sin = angles[rotary_pairs : 2 * rotary_pairs]

    def rotate(n):
        if mscale is not None:
            # the kernel scales only the rotated pairs (rounded ms in bf16);
            # the unrotated tail (dims 2*rp..) stays un-scaled
            rot = (n[..., : 2 * rotary_pairs].astype(mx.float32)
                   * mx.array(mscale, mx.bfloat16).astype(mx.float32)
                   ).astype(mx.bfloat16)
            n = mx.concatenate([rot, n[..., 2 * rotary_pairs :]], axis=-1)
        first = n[..., :rotary_pairs].astype(mx.float32)
        second = n[..., rotary_pairs : 2 * rotary_pairs].astype(mx.float32)
        out1 = (first * cos - second * sin).astype(mx.bfloat16)
        out2 = (first * sin + second * cos).astype(mx.bfloat16)
        tail = n[..., 2 * rotary_pairs :]
        return mx.concatenate([out1, out2, tail], axis=-1)

    return rotate(norm_q).reshape(-1), rotate(norm_k).reshape(-1)


def full_qk_norm_yarn(
    raw_queries, raw_keys, query_weight, key_weight, angles, stream=None
):
    """Fused Q/K RMSNorm + partial-RoPE with the YaRN mscale (48 q + 8 k
    heads, rotary 64). Returns (queries [6144], keys [1024]) bf16."""
    if _ext is not None and has_symbol("full_qk_norm_yarn"):
        return _ext.full_qk_norm_yarn(
            raw_queries, raw_keys, query_weight, key_weight, angles,
            stream=stream,
        )
    return _qk_norm_rope_fallback(
        raw_queries, raw_keys, query_weight, key_weight, angles,
        48, 8, 32, mscale=1.3465735912322998,
    )


def sliding_qk_norm_rope(
    raw_queries, raw_keys, query_weight, key_weight, angles, stream=None
):
    """Fused Q/K RMSNorm + full RoPE (64 q + 8 k heads, rotary 128).
    Returns (queries [8192], keys [1024]) bf16."""
    if _ext is not None and has_symbol("sliding_qk_norm_rope"):
        return _ext.sliding_qk_norm_rope(
            raw_queries, raw_keys, query_weight, key_weight, angles,
            stream=stream,
        )
    return _qk_norm_rope_fallback(
        raw_queries, raw_keys, query_weight, key_weight, angles,
        64, 8, 64, mscale=None,
    )


def decode_nvfp4_qkv_r1(
    normalized: mx.array,
    weight_codes: mx.array,
    weight_scales: mx.array,
    heads: int,
    stream=None,
) -> mx.array:
    """Fused Q/K/V NVFP4 projection for one decode token (R1, one row per
    simdgroup) — verbatim from lagunaDecodeNVFP4QKVR1Source (tail header at
    the default fold/defer config).

    Parameters
    ----------
    normalized    : [2048] bf16        — attention input
    weight_codes  : [rows, 1024] uint8 — fused Q/K/V plane
    weight_scales : [rows, 128] uint8  — E4M3 group-16 scales
    heads         : int                — query head count (48 full / 64 sliding)

    Returns [rows] bf16 (rows = (heads + 16)*128).
    """
    if _ext is not None and has_symbol("decode_nvfp4_qkv_r1"):
        return _ext.decode_nvfp4_qkv_r1(
            normalized, weight_codes, weight_scales, heads, stream=stream)
    # Fallback: stock nvfp4 matmul over the fused plane (reinterpret the
    # kernel's byte plane as the uint32 layout mx expects — same bytes).
    rows = weight_codes.shape[0]
    codes = weight_codes.view(mx.uint32).reshape(rows, -1)
    return mx.quantized_matmul(
        normalized[None, :], codes, scales=weight_scales,
        transpose=True, group_size=16, bits=4, mode="nvfp4", stream=stream,
    ).squeeze(0).astype(mx.bfloat16)


def oproj_act(
    attention_output: mx.array,
    gate_values: mx.array,
    weight_codes: mx.array,
    weight_scales: mx.array,
    heads: int,
    stream=None,
) -> mx.array:
    """Gated affine o_proj with a pre-activated per-head gate, fused in one
    kernel (verbatim from lagunaGatedAffineOProjNVFP4Source with
    preActivatedGate, default flag config).

    Parameters
    ----------
    attention_output : [heads*128] bf16
    gate_values      : [heads] bf16       — pre-activated per-head gate
    weight_codes     : [2048, heads*16] uint32
    weight_scales    : [2048, heads*8] uint8

    Returns [2048] bf16.
    """
    if _ext is not None and has_symbol("oproj_act"):
        return _ext.oproj_act(
            attention_output, gate_values, weight_codes, weight_scales,
            heads, stream=stream,
        )
    in_vec = heads * 128
    # the kernel rounds the per-element gate product to bf16 before the qdot
    gated = (attention_output.reshape(heads, 128).astype(mx.float32)
             * gate_values[:, None].astype(mx.float32)).astype(mx.bfloat16)
    return mx.quantized_matmul(
        gated.reshape(1, in_vec), weight_codes, scales=weight_scales,
        transpose=True, group_size=16, bits=4, mode="nvfp4", stream=stream,
    ).squeeze(0).astype(mx.bfloat16)


def residual_rms(
    residual: mx.array,
    branch: mx.array,
    weight: mx.array,
    stream=None,
):
    """Fused residual add + RMSNorm (verbatim from
    lagunaResidualRMSNormKernel). Returns (summed, normalized) [2048] bf16.
    """
    if _ext is not None and has_symbol("residual_rms"):
        return _ext.residual_rms(residual, branch, weight, stream=stream)
    summed = (residual + branch).astype(mx.bfloat16)
    normalized = mx.fast.rms_norm(summed, weight, 1e-6)
    return summed, normalized


def decode_router_top8(
    logits: mx.array,
    correction_bias: mx.array,
    normalizing: bool = False,
    stream=None,
):
    """Decode router top-8 (256-lane bitonic tournament; verbatim from
    lagunaDecodeRouterTop8KernelSource). Returns (indices, scores) [8].
    """
    if _ext is not None and has_symbol("decode_router_top8"):
        return _ext.decode_router_top8(
            logits, correction_bias, normalizing, stream=stream)
    # Fallback: sigmoid scores; the correction bias orders the sort key only
    # (the kernel emits the raw sigmoid as the score, as per the challenge).
    x = logits.astype(mx.float32)
    y = 1.0 / (1.0 + mx.exp(mx.abs(x)))
    score = mx.where(x < 0, y, 1.0 - y)
    key = score + correction_bias.astype(mx.float32)
    ids = mx.argsort(-key, axis=0)[:8].astype(mx.uint32)
    vals = mx.take(score, ids)
    if normalizing:
        vals = vals / mx.maximum(mx.sum(vals), 1e-6)
    return ids, vals.astype(mx.bfloat16)


def sliding_fused_attn_ring(
    raw_queries, raw_keys, raw_values, query_weight, key_weight, angles,
    k_cache, v_cache, params, scale_arr, stream=None,
):
    """Fused sliding-attention decode (steady ring regime; verbatim from
    lagunaSlidingFusedAttentionKernel). Returns attended [64*128] bf16.
    """
    if _ext is not None and has_symbol("sliding_fused_attn_ring"):
        return _ext.sliding_fused_attn_ring(
            raw_queries, raw_keys, raw_values, query_weight, key_weight,
            angles, k_cache, v_cache, params, scale_arr, stream=stream)
    raise RuntimeError(
        "laguna_nvfp4 sliding_fused_attn_ring requires the native extension "
        "(the fused ring path has no pure-mlx equivalent).")


def residual_rms_router(
    residual, branch, weight, router_weight, correction_bias, stream=None,
):
    """Fused residual add + RMSNorm + MoE router GEMV (verbatim from
    lagunaResidualRMSNormRouterSource, rowsPerGroup 8, precomputed ordinal
    keys). Returns (summed, normalized, router_logits, router_keys)."""
    if _ext is not None and has_symbol("residual_rms_router"):
        return _ext.residual_rms_router(
            residual, branch, weight, router_weight, correction_bias,
            stream=stream,
        )
    summed = (residual + branch).astype(mx.bfloat16)
    normalized = mx.fast.rms_norm(summed, weight, 1e-6)
    logits = (normalized[None, :] @ router_weight.T).squeeze(0).astype(
        mx.bfloat16)
    x = logits.astype(mx.float32)
    y = 1.0 / (1.0 + mx.exp(mx.abs(x)))
    score = mx.where(x < 0, y, 1.0 - y)
    key = -(score + correction_bias.astype(mx.float32))
    bits = key.view(mx.uint32)
    magnitude = bits & 0x7FFFFFFF
    keys = mx.where(
        magnitude > 0x7F800000,
        mx.full_like(bits, 0xFFFFFFFF),
        mx.where(
            magnitude == 0,
            mx.full_like(bits, 0x80000000),
            mx.where(
                (bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000,
            ),
        ),
    )
    return summed, normalized, logits, keys


def prefill_moe_tail(
    expert_outputs, router_weights, shared_output, residual, stream=None,
):
    """Prefill MoE tail (verbatim from lagunaPrefillMoETailKernel): the
    weighted 8-expert combine (x2.5) + shared + residual. Returns
    [rows*2048] bf16."""
    if _ext is not None and has_symbol("prefill_moe_tail"):
        return _ext.prefill_moe_tail(
            expert_outputs, router_weights, shared_output, residual,
            stream=stream,
        )
    w = router_weights.astype(mx.bfloat16)  # [1, rows, 8]
    y = mx.sum(expert_outputs * w[..., None], axis=2) * mx.array(
        2.5, mx.float32)  # [1, rows, 2048]
    y = (y + shared_output + residual).astype(mx.bfloat16)
    return y.reshape(-1)

def shared_nvfp4_down_residual(
    activated: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    routed: mx.array,
    residual: mx.array,
    stream=None,
) -> mx.array:
    """Shared-expert down_proj with routed + residual adds fused in one
    kernel (challenge lagunaSharedDownResidualKernel).

    Parameters
    ----------
    activated   : [K2] bf16              — swiglu output (K2 = 512)
    down_weight : [N, K2/2] uint8        — NVFP4 down_proj codes (N = 2048)
    down_scales : [N, K2/16] uint8       — E4M3 group-16 scales
    routed      : [N] bf16               — routed-expert output
    residual    : [N] bf16               — decoder residual

    Returns [N] bf16 residual + (routed + shared).
    """
    if _ext is not None and has_symbol("shared_nvfp4_down_residual"):
        return _ext.shared_nvfp4_down_residual(
            activated, down_weight, down_scales, routed, residual,
            stream=stream,
        )
    # Fallback: stock nvfp4 matmul + the two adds.
    shared = mx.quantized_matmul(
        activated[None, :], down_weight, scales=down_scales,
        transpose=True, group_size=16, bits=4, mode="nvfp4",
        stream=stream,
    ).squeeze(0)  # [N]
    return (residual + (routed + shared)).astype(mx.bfloat16)
