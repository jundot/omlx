# SPDX-License-Identifier: Apache-2.0
"""Fused 2x2 Sinkhorn kernel for SenseNova-Loop checkpoints (jundot/omlx#5415).

The SenseNova-Loop checkpoints (``netprojects/SenseNova-Loop-*``) ship their own
``sensenova_loop.py`` through ``config.json:model_file``. Every mHC router runs a
log-space Sinkhorn balancing (``mhc.iterations``, default 20) over a 2x2 score
matrix: 40 tiny ``logsumexp`` reductions plus an ``exp`` per router call, and a
layer pass issues 96 of those calls. ``mx.compile`` removes the Python overhead
but not the kernel launches, so the routers dominate decode latency on
high-bandwidth hosts.

This patch swaps the module-global ``sinkhorn`` for a single ``mx.fast.metal_kernel``
that keeps the 2x2 matrix in registers and runs the same row/column
log-normalisation in place. Measured on an M5 / 24 GiB host: per-router numerics
agree to 1.8e-7, decode goes 5.05 -> 6.11 tok/s (+21%, ~34 ms/token saved), and a
128-token greedy completion is token-identical; the saving scales with how large a
share the routers have of the step (about 60% on an M3 Ultra, 15% on an M5).

Set ``OMLX_SENSENOVA_SINKHORN=0`` to keep the reference implementation.

Note on logits: a ~1e-7 router perturbation is not bounded here -- for some prompts the reference
implementation with the same magnitude injected moves end-to-end logits by 5e-1, while for others
the delta is absorbed entirely (bitwise-equal logits). Logits are therefore not a sound bitwise
criterion between two non-identical Sinkhorn implementations; greedy token output is. 512/1024-token
chunked-prefill checks gave identical greedy tokens with logits deltas of 0.53 / 0.125.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

# One thread per 2x2 matrix; the four entries live in registers for the whole loop.
_SOURCE = r"""
uint i = thread_position_in_grid.x;
auto p = logits + i * 4;
float a = p[0], b = p[1], c = p[2], d = p[3];
for (int it = 0; it < ITERATIONS; ++it) {
    float m0 = fmax(a, b); float s0 = m0 + log(exp(a - m0) + exp(b - m0)); a -= s0; b -= s0;
    float m1 = fmax(c, d); float s1 = m1 + log(exp(c - m1) + exp(d - m1)); c -= s1; d -= s1;
    float m2 = fmax(a, c); float s2 = m2 + log(exp(a - m2) + exp(c - m2)); a -= s2; c -= s2;
    float m3 = fmax(b, d); float s3 = m3 + log(exp(b - m3) + exp(d - m3)); b -= s3; d -= s3;
}
auto o = mix + i * 4;
o[0] = exp(a); o[1] = exp(b); o[2] = exp(c); o[3] = exp(d);
"""

_KERNELS: dict[int, object] = {}


def _kernel(iterations: int):
    kernel = _KERNELS.get(iterations)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"omlx_sensenova_sinkhorn2x2_{iterations}",
            input_names=["logits"],
            output_names=["mix"],
            source=_SOURCE.replace("ITERATIONS", str(iterations)),
        )
        _KERNELS[iterations] = kernel
    return kernel


def apply(model) -> bool:
    """Replace the checkpoint module's ``sinkhorn`` with the fused kernel, in place.

    The custom model is loaded with ``spec_from_file_location`` and never inserted
    into ``sys.modules``, so its namespace is reached through the class that was
    created from it. Returns True when the patch is installed.
    """
    if os.environ.get("OMLX_SENSENOVA_SINKHORN", "1") == "0":
        return False
    if not mx.metal.is_available():
        return False

    namespace = getattr(getattr(type(model), "__init__", None), "__globals__", None)
    if (
        not isinstance(namespace, dict)
        or "sinkhorn" not in namespace
        or "_ROUTE_CACHE" not in namespace
    ):
        return False
    if namespace.get("_OMLX_SINKHORN_PATCHED"):
        return True

    original = namespace["sinkhorn"]

    def sinkhorn(logits, iterations):
        if (
            logits.shape[-2:] != (2, 2)
            or logits.size == 0
            or not mx.metal.is_available()
        ):
            return original(logits, iterations)
        scores = logits.astype(mx.float32)
        n = scores.size // 4
        out = _kernel(int(iterations))(
            inputs=[mx.contiguous(scores.reshape(n, 4))],
            grid=(n, 1, 1),
            threadgroup=(min(n, 256), 1, 1),
            output_shapes=[(n, 4)],
            output_dtypes=[mx.float32],
        )[0]
        return out.reshape(scores.shape)

    namespace["sinkhorn"] = sinkhorn
    namespace["_OMLX_SINKHORN_PATCHED"] = True
    # The compiled route/merge graphs are built lazily; drop them so the first
    # forward traces against the patched function.
    for name in ("_ROUTE_CACHE", "_MERGE"):
        cached = namespace.get(name)
        if hasattr(cached, "clear"):
            cached.clear()
    logger.info("sensenova_loop: fused 2x2 Sinkhorn kernel enabled")
    return True
