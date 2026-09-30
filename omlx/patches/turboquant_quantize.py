# SPDX-License-Identifier: Apache-2.0
"""Compile the existing RHT MSE codec for one-token cache appends.

Graph fusion removes intermediate pointwise dispatches while keeping the
codec's normalization, Hadamard transform, comparisons and packing. The
rotation signs and midpoints stay explicit inputs, so compiled graphs may
be shared by codecs with different seeds without retaining codec objects.
"""

import logging
import os
from functools import cache, wraps
from types import SimpleNamespace

import mlx.core as mx

logger = logging.getLogger(__name__)
# Fusion improves warmed decode on the pinned MLX, but can increase short-
# prompt TTFT and regress other builds. Keep it opt-in until broader validation.
_ENABLED = os.environ.get("OMLX_TURBOQUANT_COMPILE_QUANTIZE", "0") != "0"
_DISABLED = set()


@cache
def _compiled_quantizer(original, dim, bits):
    from mlx_vlm import turboquant as tq

    def quantize(vectors, signs, midpoints):
        # The original codec reads only these fields on its RHT MSE arm.
        # Build this view during tracing, rather than capture a live codec.
        codec = SimpleNamespace(
            dim=dim,
            bits=bits,
            _midpoints=midpoints,
            _rotate_forward=lambda x: tq._rht_forward(x, signs),
        )
        state = original(codec, vectors)
        return state.norms, state.indices

    return mx.compile(quantize)


def apply_turboquant_quantize_patch():
    from mlx_vlm import turboquant as tq

    cls = tq._TurboQuantMSECodec
    original = cls.quantize
    if not _ENABLED or not mx.metal.is_available():
        return False
    if getattr(original, "_omlx_compiled_mse_quantize", False):
        return True

    @wraps(original)
    def quantize(self, vectors):
        if (
            not _ENABLED
            or type(self) is not cls
            or vectors.ndim != 4
            or vectors.shape[-2] != 1
            or vectors.dtype not in (mx.float16, mx.bfloat16, mx.float32)
            or mx.default_device() != mx.gpu
            or not self.use_rht
            or self.dim not in (64, 128, 256)
            or vectors.shape[-1] != self.dim
            or self.bits not in (1, 2, 3, 4)
            or "_rotate_forward" in self.__dict__
        ):
            return original(self, vectors)
        key = (self.dim, self.bits, vectors.dtype)
        if key in _DISABLED:
            return original(self, vectors)
        try:
            norms, indices = _compiled_quantizer(original, self.dim, self.bits)(
                vectors, self.signs, self._midpoints
            )
            return tq.TurboQuantMSEState(norms, indices)
        except Exception:
            _DISABLED.add(key)
            logger.warning(
                "Compiled TurboQuant quantizer failed; stock fallback", exc_info=True
            )
            return original(self, vectors)

    quantize._omlx_compiled_mse_quantize = True
    cls.quantize = quantize
    return True
