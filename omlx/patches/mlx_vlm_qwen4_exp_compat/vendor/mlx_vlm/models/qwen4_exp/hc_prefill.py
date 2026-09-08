# SPDX-License-Identifier: Apache-2.0

"""Fused RMSNorm for Qwen4-Exp's hyper-connection module during prefill
(S in the hundreds-to-thousands range, not the single-token decode step).

``Qwen4ExpGatedResidual._forward`` computes its RMSNorm (``self.hc_norm``)
as a plain MLX composition -- reshape, square, mean, rsqrt, multiply,
multiply -- dispatched once per layer per token. Layer-level profiling on a
real ``Qwen3.8-Flash-Next-oQ4e-mtp`` checkpoint found this norm to be the
dominant cost inside the module, ahead of its two read projections and its
up-projection, which all run through MLX's own (already reasonably fast)
``qmm``. This module replaces just the norm with one fused Metal dispatch;
the read/up projections are left exactly as they are -- ``nn.Linear`` or
``nn.QuantizedLinear``, quantized or not, whatever the checkpoint has.

A tiled-matmul kernel for the projections was also built and measured (see
the PR this shipped in). It did produce a further improvement on top of this
norm fusion, but a smaller and less consistent one -- roughly +7-11% over
this norm-only path at S in {2048, 4096}, and a small *regression* at
S=1024 -- for two entire additional hand-written Metal kernels' worth of
review surface and quantization-format coupling. That tradeoff did not seem
worth it, so this PR ships only the norm fusion, which is simpler, has no
quantization-scheme dependency at all, and captures most of the available
win on its own:

============  ============  ==========  ==========
S             canonical     this path   speedup
============  ============  ==========  ==========
1024          266.61 ms     179.34 ms   1.49x
2048          514.97 ms     374.39 ms   1.38x
4096          1062.68 ms    799.52 ms   1.33x
============  ============  ==========  ==========

(Forced-accumulation timing, every one of 96 simulated hyper-connection
calls actually executed per measurement, interleaved rounds, GPU confirmed
exclusive -- see the PR description for the full methodology.)

Set ``OMLX_QWEN4_HC_PREFILL=1`` to enable; it is opt-in (default off) since
this is new, independently designed Metal exercised so far only on one Mac
generation (M4 Max). Fails closed to the canonical path on any shape/dtype
mismatch, or any runtime exception the first time the kernel actually runs
-- turning it on cannot make a checkpoint this was not tuned against
produce wrong output, only leave it on the unmodified path.

The kernel itself is not tied to Qwen4-Exp's specific ``hc_count=4,
hidden_size=2560`` -- the Metal source takes both as compile-time template
parameters, and the only real constraint is ``hidden_size % 32 == 0`` (each
of the 32 lanes in a simdgroup owns ``hidden_size/32`` elements; a
non-multiple would need boundary handling this kernel doesn't have). Each
distinct ``(hidden_size, hc_count)`` pair gets its own compiled kernel,
cached after first use, so this costs nothing at the validated shape --
confirmed by benchmark, see the PR description.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_RUNTIME_FAILED = False
_FAILURE_LOGGED = False


def _enabled() -> bool:
    return os.environ.get("OMLX_QWEN4_HC_PREFILL", "").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


@dataclass
class _Prepared:
    """Per-module state needed to run the fused norm and the rest of the
    canonical forward pass with the module's own (untouched) projections."""

    gain: mx.array
    eps: float
    has_inject: bool
    hidden: int
    lanes: int
    stream: int


def _prepare(module) -> "_Prepared | None":
    """Validate one hyper-connection module. Fails closed: anything that
    does not match a shape this kernel can run returns ``None`` and the
    caller falls back to ``module._forward``.

    Unlike a projection-shaped kernel, this only constrains the norm's own
    shape -- the projections underneath (``input_mix_weight_down``,
    ``input_mix_weight_up``, ``block_inject_weight``) are used exactly as
    the module already has them, quantized or not, at whatever bit width.
    ``hidden_size`` and ``hc_count`` are read from the module, not fixed to
    Qwen4-Exp's specific values (see the module docstring) -- the only
    hard requirement is ``hidden_size`` being a multiple of 32.
    """
    hidden = getattr(module, "hidden_size", None)
    if not isinstance(hidden, int) or hidden <= 0 or hidden % 32 != 0:
        return None
    lanes = getattr(module, "hc_count", None)
    if not isinstance(lanes, int) or lanes <= 0:
        return None
    stream = lanes * hidden

    norm = getattr(module, "hc_norm", None)
    gain = getattr(norm, "weight", None)
    if not isinstance(gain, mx.array) or gain.shape != (stream,):
        return None
    if getattr(norm, "group_size", None) != hidden:
        return None
    eps = getattr(norm, "eps", None)
    if not isinstance(eps, float):
        return None

    for name in ("input_mix_weight_down", "input_mix_weight_up"):
        if not callable(getattr(module, name, None)):
            return None

    return _Prepared(
        gain=gain, eps=float(eps), has_inject="block_inject_weight" in module,
        hidden=hidden, lanes=lanes, stream=stream,
    )


_R1_HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"

# One threadgroup per (token, lane): grid.x = 32 * S * HC_COUNT. A thread's
# own HIDDEN/32 values live in registers for the whole call -- nothing but
# this thread ever reads them again, so there is nothing to stage in
# threadgroup memory.
_R1_SOURCE = """
    constexpr int HID = {hidden}, PER = {hidden} / 32;
    uint tid = thread_position_in_threadgroup.x;
    uint grp = threadgroup_position_in_grid.x;      // 0 .. S*{lanes}-1
    uint base = grp * HID + tid * PER;

    float xv[PER];
    float sq = 0.0f;
    for (int i = 0; i < PER; ++i) {{
        xv[i] = float(x[base + i]);
        sq += xv[i] * xv[i];
    }}
    sq = simd_sum(sq);                              // broadcasts to all 32 lanes, no barrier
    float scale = rsqrt(sq / float(HID) + eps[0]);

    uint gbase = (grp % {lanes}) * HID + tid * PER;  // gain is [{lanes},HID], independent of token
    for (int i = 0; i < PER; ++i) {{
        float g = float(gain[gbase + i]);
        n[base + i] = bfloat16_t(xv[i] * scale * (1.0f + g));
    }}
"""


@lru_cache(maxsize=None)
def _r1_kernel(hidden: int, lanes: int):
    return mx.fast.metal_kernel(
        name=f"omlx_qwen4_hc_prefill_r1_rmsnorm_h{hidden}_l{lanes}",
        input_names=["x", "gain", "eps"],
        output_names=["n"],
        source=_R1_SOURCE.format(hidden=hidden, lanes=lanes),
        header=_R1_HEADER,
    )


def _r1_rmsnorm(x: mx.array, gain: mx.array, eps: float, hidden: int, lanes: int, stream: int) -> mx.array:
    """Grouped RMSNorm, one dispatch, ``x`` never leaves registers.

    ``x``    [S, stream] bf16, ``stream == lanes * hidden``
    ``gain`` [stream] bf16 (zero-centered; applied as ``1 + gain``)
    returns  [S, stream] bf16

    ``(hidden, lanes)`` select (and, on first use, compile) the kernel
    variant for that shape -- see ``_r1_kernel``'s cache. The validated
    Qwen4-Exp shape (2560, 4) pays the same cost as before generalization:
    one dict lookup keyed by a pair of ints, not a runtime branch inside
    the kernel itself.
    """
    lead = x.shape[:-1]
    s = 1
    for d in lead:
        s *= d
    x2 = x.reshape(s, stream)
    eps_arr = mx.array([eps], dtype=mx.float32)

    (n,) = _r1_kernel(hidden, lanes)(
        inputs=[x2, gain, eps_arr],
        grid=(32 * s * lanes, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(s, stream)],
        output_dtypes=[mx.bfloat16],
    )
    return n.reshape(*lead, stream)


def prefill_read(module, hyper_input: mx.array):
    """Entry point called from ``Qwen4ExpGatedResidual.__call__``.

    Runs the fused norm above, then the rest of ``_forward``'s logic
    verbatim (silu, sigmoid, the 4-lane mean, the injection sigmoid),
    through the module's own, unmodified projection layers -- only the
    norm dispatch is replaced.

    Returns ``None`` (canonical path) whenever ``OMLX_QWEN4_HC_PREFILL`` is
    not set, the module's shape does not match what this kernel can run
    (see ``_prepare``), or the kernel raises the first time it actually
    runs -- in the last case it disables itself for the rest of the
    process rather than retrying every call.
    """
    global _RUNTIME_FAILED, _FAILURE_LOGGED
    if _RUNTIME_FAILED or not _enabled():
        return None
    if not (
        isinstance(hyper_input, mx.array)
        and hyper_input.dtype == mx.bfloat16
        and hyper_input.ndim >= 2
        and mx.default_device() == mx.gpu
    ):
        return None

    prepared = getattr(module, "_omlx_hc_prefill", False)
    if prepared is False:
        prepared = _prepare(module)
        module._omlx_hc_prefill = prepared
    if prepared is None:
        return None
    if hyper_input.shape[-1] != prepared.stream:
        return None

    try:
        normed = _r1_rmsnorm(
            hyper_input, prepared.gain, prepared.eps,
            prepared.hidden, prepared.lanes, prepared.stream,
        )
        mix = module.input_mix_weight_down(normed)
        block_injection = (
            module.block_inject_weight(normed) if prepared.has_inject else None
        )
        mix = nn.silu(mix / module.hc_count)
        mix = mx.sigmoid(module.input_mix_weight_up(mix))
        mix = mix.reshape(*mix.shape[:-1], module.hc_count, module.hidden_size)
        streams = normed.reshape(*normed.shape[:-1], module.hc_count, module.hidden_size)
        mixed_input = mx.mean(mix * streams, axis=-2)
    except Exception as exc:  # noqa: BLE001 - optional native path, fail closed
        _RUNTIME_FAILED = True
        if not _FAILURE_LOGGED:
            _FAILURE_LOGGED = True
            logger.warning(
                "Qwen4 prefill hyper-connection norm kernel failed closed; "
                "using canonical path for the rest of this process: %s",
                exc,
            )
        return None

    if block_injection is None:
        return mixed_input
    injection_weights = 2 * mx.sigmoid(block_injection / module.hc_count)
    return mixed_input, hyper_input, injection_weights


__all__ = ["prefill_read"]
