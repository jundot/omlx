# SPDX-License-Identifier: Apache-2.0
"""Three Metal kernels for small Qwen4 hyper-connection inputs.

Fuses per-stream RMS norm, down/inject projections with activation, and the up
projection with stream mixing. Supports at most 16 BF16 rows, four streams,
and affine group-size-64 projections with 4/5/6/8-bit weights. The small block
injection projection may also use its original BF16 weights. FP32 epilogues
can round differently from the canonical BF16 operations.

Each kernel specialization is evaluated once inside the failure handler to
catch lazy compilation errors. Later calls stay lazy; errors during their
external evaluation propagate to the caller. Disable with OMLX_QWEN4_HC_FUSED=0.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx
import mlx.nn as nn

from .hc_projection import _HEADER

logger = logging.getLogger(__name__)

MAX_ROWS = 16
_GROUP_SIZE = 64
_SUPPORTED_BITS = (4, 5, 6, 8)
_DISABLED = os.environ.get("OMLX_QWEN4_HC_FUSED", "1").strip().lower() in {
    "0",
    "false",
    "no",
    "off",
}
_KERNELS: dict[str, object] = {}
_VALIDATED: set[tuple] = set()
_RUNTIME_FAILED = False
_FAILURE_LOGGED = False
_INELIGIBLE_LOGGED = False

_N_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint s = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[8];
    constexpr int PER = (H + 255) / 256;
    const device T* xp = x + (size_t)row * K + (size_t)s * H;
    const device T* wp = w + (size_t)s * H;
    device T* op = xn + (size_t)row * K + (size_t)s * H;
    float v[PER];
    float ss = 0.0f;
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = (k < H) ? float(xp[k]) : 0.0f;
        ss += v[i] * v[i];
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = metal::rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) op[k] = T(v[i] * inv * (1.0f + float(wp[k])));
    }
"""

_D_SOURCE = r"""
    const uint tg = threadgroup_position_in_grid.y;
    const uint row = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    constexpr int KS = 4;
    constexpr int SLICE = K / KS;
    constexpr int GROUPS = K / 64;
    threadgroup float part[KS][8];
    const device T* x = xn + (size_t)row * K;
    const int rg = int(sg & 1);
    const int ks = int(sg >> 1);
    if (tg < R / 8) {
        constexpr int PF = hc_pack_factor<BITS_D>();
        constexpr int BP = hc_bytes_per_pack<BITS_D>();
        constexpr int ROW_BYTES = K * BP / PF;
        constexpr int PPT = 2;
        constexpr int VPT = PF * PPT;
        constexpr int BLOCK = VPT * 32;
        constexpr int SCALE_STEP = 64 / VPT;
        const int out_row = int(tg) * 8 + rg * 4;
        const device uint8_t* wp = (const device uint8_t*)down_w
            + out_row * ROW_BYTES + ks * (SLICE * BP / PF) + int(lane) * PPT * BP;
        const device T* sp = down_s + out_row * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const device T* bp = down_b + out_row * GROUPS + ks * (SLICE / 64)
            + int(lane) / SCALE_STEP;
        const device T* xp = x + ks * SLICE + int(lane) * VPT;
        float result[4] = {0.0f};
        float xv[VPT];
        constexpr int TAIL = SLICE % BLOCK;
        for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
            float sum = hc_load_vector<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < 4; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
            wp += BLOCK * BP / PF;
            sp += BLOCK / 64;
            bp += BLOCK / 64;
            xp += BLOCK;
        }
        // Load only lanes inside the tail; all lanes must join simd_sum.
        if (TAIL > 0 && int(lane) * VPT < TAIL) {
            float sum = hc_load_vector<T, VPT, BITS_D>(xp, xv);
            for (int r = 0; r < 4; ++r) {
                result[r] += hc_qdot<VPT, BITS_D>(
                    wp + r * ROW_BYTES, xv,
                    float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
            }
        }
        for (int r = 0; r < 4; ++r) {
            float v = simd_sum(result[r]);
            if (lane == 0) part[ks][rg * 4 + r] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < 2 && lane < 4) {
            const int r = int(lane);
            float v = part[0][sg * 4 + r] + part[1][sg * 4 + r]
                + part[2][sg * 4 + r] + part[3][sg * 4 + r];
            v = v / float(HC);
            act[(size_t)row * R + int(tg) * 8 + int(sg) * 4 + r]
                = T(v / (1.0f + metal::exp(-v)));
        }
        return;
    }
    if (INJ == 0 || rg != 0) return;
    {
        float result[HC] = {0.0f};
        if constexpr (DENSE_I) {
            // Four output rows are small enough to share the down projection's
            // dispatch without quantizing precision-sensitive injection weights.
            const device T* wp = (const device T*)inject_w;
            for (int k = int(lane); k < SLICE; k += 32) {
                const int col = ks * SLICE + k;
                const float xv = float(x[col]);
                for (int r = 0; r < HC; ++r) {
                    result[r] += xv * float(wp[r * K + col]);
                }
            }
        } else {
            constexpr int PF = hc_pack_factor<BITS_I>();
            constexpr int BP = hc_bytes_per_pack<BITS_I>();
            constexpr int ROW_BYTES = K * BP / PF;
            constexpr int PPT = 1;
            constexpr int VPT = PF;
            constexpr int BLOCK = VPT * 32;
            constexpr int SCALE_STEP = 64 / VPT;
            const device uint8_t* wp = (const device uint8_t*)inject_w
                + ks * (SLICE * BP / PF) + int(lane) * BP;
            const device T* sp = inject_s + ks * (SLICE / 64) + int(lane) / SCALE_STEP;
            const device T* bp = inject_b + ks * (SLICE / 64) + int(lane) / SCALE_STEP;
            const device T* xp = x + ks * SLICE + int(lane) * VPT;
            float xv[VPT];
            constexpr int TAIL = SLICE % BLOCK;
            for (int k = 0; k < SLICE - TAIL; k += BLOCK) {
                float sum = hc_load_vector<T, VPT, BITS_I>(xp, xv);
                for (int r = 0; r < HC; ++r) {
                    result[r] += hc_qdot<VPT, BITS_I>(
                        wp + r * ROW_BYTES, xv,
                        float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
                }
                wp += BLOCK * BP / PF;
                sp += BLOCK / 64;
                bp += BLOCK / 64;
                xp += BLOCK;
            }
            if (TAIL > 0 && int(lane) * VPT < TAIL) {
                float sum = hc_load_vector<T, VPT, BITS_I>(xp, xv);
                for (int r = 0; r < HC; ++r) {
                    result[r] += hc_qdot<VPT, BITS_I>(
                        wp + r * ROW_BYTES, xv,
                        float(sp[r * GROUPS]), float(bp[r * GROUPS]), sum);
                }
            }
        }
        for (int r = 0; r < HC; ++r) {
            float v = simd_sum(result[r]);
            if (lane == 0) part[ks][r] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0 && lane < HC) {
            const int r = int(lane);
            float v = part[0][r] + part[1][r] + part[2][r] + part[3][r];
            v = v / float(HC);
            inj[(size_t)row * HC + r] = T(2.0f / (1.0f + metal::exp(-v)));
        }
    }
"""

_U_SOURCE = r"""
    const uint g = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const int h = int(g) * 64 + int(t >> 2);
    const int s = int(t & 3);
    const int n = s * H + h;
    constexpr int PF = hc_pack_factor<BITS_U>();
    constexpr int BP = hc_bytes_per_pack<BITS_U>();
    constexpr int ROW_BYTES = R * BP / PF;
    constexpr int GROUPS_R = R / 64;
    constexpr int CH = 2 * PF;
    constexpr int CPG = 64 / CH;
    const device uint8_t* w = (const device uint8_t*)up_w + (size_t)n * ROW_BYTES;
    const device T* sp = up_s + (size_t)n * GROUPS_R;
    const device T* bp = up_b + (size_t)n * GROUPS_R;
    float xv[CH];
    for (int r = 0; r < S; ++r) {
        const device T* a = act + (size_t)r * R;
        float acc = 0.0f;
        for (int gq = 0; gq < GROUPS_R; ++gq) {
            const float sc = float(sp[gq]);
            const float bi = float(bp[gq]);
            for (int c = 0; c < CPG; ++c) {
                const int e = gq * 64 + c * CH;
                float sum = hc_load_vector<T, CH, BITS_U>(a + e, xv);
                acc += hc_qdot<CH, BITS_U>(w + e * BP / PF, xv, sc, bi, sum);
            }
        }
        float gate = 1.0f / (1.0f + metal::exp(-acc));
        float v = gate * float(xn[(size_t)r * K + n]);
        v += simd_shuffle_down(v, 1);
        v += simd_shuffle_down(v, 2);
        if (s == 0) mixed[(size_t)r * H + h] = T(v / float(HC));
    }
"""


def _kernel(
    name: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
    header: str = "",
):
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            header=header,
            source=source,
            ensure_row_contiguous=True,
        )
        _KERNELS[name] = kernel
    return kernel


def enabled() -> bool:
    return not _DISABLED and not _RUNTIME_FAILED


def _quantized_ok(projection) -> bool:
    return (
        type(projection) is nn.QuantizedLinear
        and getattr(projection, "group_size", None) == _GROUP_SIZE
        and getattr(projection, "bits", None) in _SUPPORTED_BITS
        and getattr(projection, "mode", "affine") == "affine"
        and "bias" not in projection
        and isinstance(getattr(projection, "weight", None), mx.array)
        and projection.weight.dtype == mx.uint32
        and isinstance(getattr(projection, "scales", None), mx.array)
        and isinstance(getattr(projection, "biases", None), mx.array)
        and projection.scales.dtype == mx.bfloat16
        and projection.biases.dtype == mx.bfloat16
    )


def _dense_inject_ok(projection, hc: int, width: int) -> bool:
    return (
        type(projection) is nn.Linear
        and "bias" not in projection
        and isinstance(getattr(projection, "weight", None), mx.array)
        and projection.weight.dtype == mx.bfloat16
        and projection.weight.shape == (hc, width)
    )


def _ineligible(reason: str) -> bool:
    """Log the first unsupported model layout per process."""
    global _INELIGIBLE_LOGGED
    if not _INELIGIBLE_LOGGED:
        _INELIGIBLE_LOGGED = True
        logger.info(
            "Qwen4 fused hyper-connection kernels not used for this model (%s); "
            "the canonical path stays in effect",
            reason,
        )
    return False


def _rows_of(hyper_input) -> int | None:
    if not (
        isinstance(hyper_input, mx.array)
        and hyper_input.ndim == 3
        and hyper_input.dtype == mx.bfloat16
    ):
        return None
    return hyper_input.shape[0] * hyper_input.shape[1]


def compatible(module, hyper_input) -> bool:
    """Whether ``module`` (a Qwen4ExpGatedResidual) can take the fused path for ``hyper_input``."""
    rows = _rows_of(hyper_input)
    return (
        enabled()
        and rows is not None
        and 1 <= rows <= MAX_ROWS
        and _layout_compatible(module, hyper_input)
    )


def prefill_compatible(module, hyper_input) -> bool:
    """Whether rows above MAX_ROWS can use the compiled prefill mean."""
    rows = _rows_of(hyper_input)
    return (
        enabled()
        and rows is not None
        and rows > MAX_ROWS
        and _layout_compatible(module, hyper_input)
    )


def _layout_compatible(module, hyper_input) -> bool:
    if hasattr(module, "input_inject_weight"):
        return _ineligible("combined input projection layout")
    hc_count = getattr(module, "hc_count", None)
    hidden = getattr(module, "hidden_size", None)
    lowrank = getattr(module, "hc_lowrank", None)
    if not (
        isinstance(hc_count, int)
        and isinstance(hidden, int)
        and isinstance(lowrank, int)
        and hc_count == 4
        # The up grid and quantization groups require 64-element alignment.
        and hidden % 64 == 0
        and lowrank % 64 == 0
        and hyper_input.shape[2] == hc_count * hidden
    ):
        return _ineligible(
            f"geometry hidden_size={hidden} hc_lowrank={lowrank} hc_count={hc_count}"
        )
    norm = getattr(module, "hc_norm", None)
    if not (
        norm is not None
        and getattr(norm, "group_size", None) == hidden
        and isinstance(getattr(norm, "weight", None), mx.array)
        and norm.weight.dtype == mx.bfloat16
        and norm.weight.shape == (hc_count * hidden,)
    ):
        return _ineligible("hc_norm layout")
    if not (
        _quantized_ok(getattr(module, "input_mix_weight_down", None))
        and _quantized_ok(getattr(module, "input_mix_weight_up", None))
    ):
        return _ineligible(
            "projection quantisation (need affine group-size-64 4/5/6/8-bit with bf16 scales)"
        )
    if "block_inject_weight" in module:
        inject = module.block_inject_weight
        if not (
            _quantized_ok(inject)
            or _dense_inject_ok(inject, hc_count, hc_count * hidden)
        ):
            return _ineligible("block_inject_weight layout (need affine or dense BF16)")
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _eps_array(module) -> mx.array:
    eps = getattr(module, "_omlx_hc_fused_eps", None)
    if eps is None:
        eps = mx.array([float(module.hc_norm.eps)], dtype=mx.float32)
        mx.eval(eps)
        module._omlx_hc_fused_eps = eps
    return eps


def _kernel_norm(module, flat, rows, hc, hidden, dtype):
    width = hc * hidden
    return _kernel("omlx_qwen4_hc_fused_norm", ["x", "w", "eps"], ["xn"], _N_SOURCE)(
        inputs=[flat, module.hc_norm.weight, _eps_array(module)],
        template=[("T", dtype), ("K", width), ("H", hidden)],
        grid=(256, hc, rows),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[dtype],
    )[0]


_TAILS: dict[tuple[int, int], object] = {}


def _tail(hc: int, hidden: int):
    """Compiled mean over the streams as slice products: one fused pass instead of a strided reduce."""
    fn = _TAILS.get((hc, hidden))
    if fn is None:

        def tail(up, normed):
            gate = mx.sigmoid(up)
            acc = gate[..., :hidden] * normed[..., :hidden]
            for g in range(1, hc):
                lo = g * hidden
                acc = acc + gate[..., lo : lo + hidden] * normed[..., lo : lo + hidden]
            return acc * (1.0 / hc)

        fn = mx.compile(tail)
        _TAILS[(hc, hidden)] = fn
    return fn


def prefill_forward(module, hyper_input):
    """Prefill with canonical normalization and a compiled mean; None on failure."""
    global _RUNTIME_FAILED, _FAILURE_LOGGED
    try:
        hc, hidden = module.hc_count, module.hidden_size
        dtype = hyper_input.dtype
        normed = module.hc_norm(hyper_input)
        mix = nn.silu(module.input_mix_weight_down(normed) / hc)
        mixed = _tail(hc, hidden)(module.input_mix_weight_up(mix), normed)
        inject = module.block_inject_weight if "block_inject_weight" in module else None
        injection = None if inject is None else 2 * mx.sigmoid(inject(normed) / hc)
        signature = (
            "prefill",
            dtype,
            hc,
            hidden,
            module.hc_lowrank,
            module.input_mix_weight_down.bits,
        )
        if signature not in _VALIDATED:
            mx.eval(mixed) if injection is None else mx.eval(mixed, injection)
            _VALIDATED.add(signature)
        if injection is None:
            return mixed
        return mixed, hyper_input, injection
    except Exception as exc:  # noqa: BLE001 - optional native path
        _RUNTIME_FAILED = True
        if not _FAILURE_LOGGED:
            _FAILURE_LOGGED = True
            logger.warning(
                "Qwen4 fused hyper-connection kernels failed closed; using the "
                "canonical path: %s",
                exc,
            )
        return None


def fused_forward(module, hyper_input):
    """Return fused outputs, or None on construction or first-evaluation failure."""
    global _RUNTIME_FAILED, _FAILURE_LOGGED
    try:
        hc, hidden, lowrank = module.hc_count, module.hidden_size, module.hc_lowrank
        width = hc * hidden
        batch, seq, _ = hyper_input.shape
        rows = batch * seq
        down, up = module.input_mix_weight_down, module.input_mix_weight_up
        inject = module.block_inject_weight if "block_inject_weight" in module else None
        dense_inject = inject is not None and type(inject) is nn.Linear
        inject_bits = (
            16 if dense_inject else (inject.bits if inject is not None else None)
        )
        flat = hyper_input.reshape(rows, width)
        dtype = hyper_input.dtype
        normed = _kernel_norm(module, flat, rows, hc, hidden, dtype)
        if dense_inject:
            # Scale/bias arguments are unused in the dense specialization.
            inject_tensors = (inject.weight, down.scales, down.biases)
        elif inject is not None:
            inject_tensors = (inject.weight, inject.scales, inject.biases)
        else:
            inject_tensors = (down.weight, down.scales, down.biases)
        act, injection = _kernel(
            "omlx_qwen4_hc_fused_down",
            ["xn", "down_w", "down_s", "down_b", "inject_w", "inject_s", "inject_b"],
            ["act", "inj"],
            _D_SOURCE,
            header=_HEADER,
        )(
            inputs=[normed, down.weight, down.scales, down.biases, *inject_tensors],
            template=[
                ("T", dtype),
                ("BITS_D", down.bits),
                (
                    "BITS_I",
                    (
                        inject_bits
                        if inject is not None and not dense_inject
                        else down.bits
                    ),
                ),
                ("DENSE_I", dense_inject),
                ("K", width),
                ("R", lowrank),
                ("HC", hc),
                ("INJ", 1 if inject is not None else 0),
            ],
            grid=(32, 8 * (lowrank // 8 + 1), rows),
            threadgroup=(32, 8, 1),
            output_shapes=[(rows, lowrank), (rows, hc)],
            output_dtypes=[dtype, dtype],
        )
        mixed = _kernel(
            "omlx_qwen4_hc_fused_up",
            ["xn", "act", "up_w", "up_s", "up_b"],
            ["mixed"],
            _U_SOURCE,
            header=_HEADER,
        )(
            inputs=[normed, act, up.weight, up.scales, up.biases],
            template=[
                ("T", dtype),
                ("BITS_U", up.bits),
                ("K", width),
                ("R", lowrank),
                ("HC", hc),
                ("H", hidden),
                ("S", rows),
            ],
            grid=(256, hidden // 64, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, hidden)],
            output_dtypes=[dtype],
        )[
            0
        ]
        signature = (
            dtype,
            hc,
            hidden,
            lowrank,
            rows,
            down.bits,
            up.bits,
            inject_bits,
        )
        if signature not in _VALIDATED:
            # Metal compilation is lazy. Validate once, without synchronizing
            # subsequent layers or decode steps using the same specialization.
            if inject is None:
                mx.eval(mixed)
            else:
                mx.eval(mixed, injection)
            _VALIDATED.add(signature)
        mixed = mixed.reshape(batch, seq, hidden)
        if inject is None:
            return mixed
        return mixed, hyper_input, injection.reshape(batch, seq, hc)
    except Exception as exc:  # noqa: BLE001 - optional native path
        _RUNTIME_FAILED = True
        if not _FAILURE_LOGGED:
            _FAILURE_LOGGED = True
            logger.warning(
                "Qwen4 fused hyper-connection kernels failed closed; using the "
                "canonical path: %s",
                exc,
            )
        return None


__all__ = [
    "MAX_ROWS",
    "compatible",
    "enabled",
    "fused_forward",
    "prefill_compatible",
    "prefill_forward",
]
