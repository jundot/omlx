"""Guard mlx-vlm Qwen3.5 ragged decode SDPA kernels.

mlx-vlm's Qwen3.5/Qwen3.6 language runtime uses custom Metal kernels for
ragged left-padded decode.  The one-pass kernel and the second phase of the
2-pass kernel are launched with a fixed ``threadgroup=(1024, 1, 1)``.  On some
Apple GPU pipeline specialisations Metal reports a lower
``maxTotalThreadsPerThreadgroup`` (for example 896 on M2 Ultra with the
D_SIZE=256 two-pass-2 variant).  Because MLX submits lazily, the validation
error surfaces later from ``mx.async_eval`` in ``mlx_lm.generate`` and tears
down the whole request.

This patch probes each compiled ragged-SDPA variant once with a tiny eager
launch.  Variants that cannot be launched on the current device are disabled so
mlx-vlm's existing per-left-padding reference path handles the row group.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_APPLIED = False
_ORIGINAL_RAGGED_DECODE_ATTENTION: Any | None = None
_RUNTIME_VERIFIED_VARIANTS: set[tuple[Any, ...]] = set()
_RUNTIME_DISABLED_VARIANTS: set[tuple[Any, ...]] = set()


def _is_threadgroup_limit_error(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "Thread group size" in message
        and "maximum allowed threads per threadgroup" in message
    )


def _dtype_name(dtype: Any) -> str:
    if dtype is mx.bfloat16:
        return "bf16"
    if dtype is mx.float16:
        return "fp16"
    return str(dtype)


def _cache_key(
    dtype: Any,
    d_size: int,
    v_size: int,
    q_heads: int,
    kv_heads: int,
    batch: int,
    k_size: int,
    mode: str,
    blocks: int,
) -> tuple[str, int, int, int, int, int, int, str, int]:
    # Exact long-context probes can allocate too much scratch; cap the probe
    # length while still preserving the shape bucket that triggers the
    # problematic two-pass pipelines.
    probe_k_size = min(max(1, k_size), 4096)
    return (
        _dtype_name(dtype),
        d_size,
        v_size,
        q_heads,
        kv_heads,
        batch,
        probe_k_size,
        mode,
        blocks,
    )


@lru_cache(maxsize=256)
def _variant_supported_cached(
    dtype_name: str,
    d_size: int,
    v_size: int,
    q_heads: int,
    kv_heads: int,
    batch: int,
    probe_k_size: int,
    mode: str,
    blocks: int,
) -> bool:
    # Reconstruct the dtype inside the cached function so the lru key stays
    # stable even if MLX dtype objects change identity across versions.
    dtype = mx.bfloat16 if dtype_name == "bf16" else mx.float16
    gqa_factor = q_heads // kv_heads

    try:
        lang = importlib.import_module("mlx_vlm.models.qwen3_5.language")

        if mode == "one_pass":
            queries = mx.zeros((batch, q_heads, 1, d_size), dtype=dtype)
            keys = mx.zeros((batch, kv_heads, probe_k_size, d_size), dtype=dtype)
            values = mx.zeros((batch, kv_heads, probe_k_size, v_size), dtype=dtype)
            pads = mx.array((0,) * batch, dtype=mx.int32)
            scale = mx.array((1.0,), dtype=mx.float32)
            k_size = mx.array((probe_k_size,), dtype=mx.int32)
            kernel = lang._qwen3_5_ragged_sdpa_one_pass_kernel(dtype, d_size, v_size)
            (out,) = kernel(
                inputs=[queries, keys, values, pads, scale, k_size],
                template=[
                    ("T", dtype),
                    ("D_SIZE", int(d_size)),
                    ("V_SIZE", int(v_size)),
                    ("NUM_Q_HEADS", int(q_heads)),
                    ("NUM_KV_HEADS", int(kv_heads)),
                    ("GQA_FACTOR", int(gqa_factor)),
                ],
                grid=(1024, batch * q_heads, 1),
                threadgroup=(1024, 1, 1),
                output_shapes=[(batch, q_heads, 1, v_size)],
                output_dtypes=[dtype],
            )
            mx.eval(out)
            return True

        # Probe the whole two-pass graph, not just phase 2. MLX metal kernels
        # can be shape-specialised beyond the explicit template constants, and
        # the failing validation is deferred until the complete graph is eval'd.
        queries = mx.zeros((batch, q_heads, 1, d_size), dtype=dtype)
        keys = mx.zeros((batch, kv_heads, probe_k_size, d_size), dtype=dtype)
        values = mx.zeros((batch, kv_heads, probe_k_size, v_size), dtype=dtype)
        pads = mx.array((0,) * batch, dtype=mx.int32)
        scale = mx.array((1.0,), dtype=mx.float32)
        k_size = mx.array((probe_k_size,), dtype=mx.int32)
        template = [
            ("T", dtype),
            ("D_SIZE", int(d_size)),
            ("V_SIZE", int(v_size)),
            ("NUM_Q_HEADS", int(q_heads)),
            ("NUM_KV_HEADS", int(kv_heads)),
            ("GQA_FACTOR", int(gqa_factor)),
        ]
        kernel_1 = lang._qwen3_5_ragged_sdpa_two_pass_1_kernel(
            dtype, d_size, v_size, blocks
        )
        partials, sums, maxs = kernel_1(
            inputs=[queries, keys, values, pads, scale, k_size],
            template=[*template, ("BLOCKS", int(blocks))],
            grid=(32 * kv_heads, gqa_factor * batch, blocks),
            threadgroup=(32, gqa_factor, 1),
            output_shapes=[
                (batch, q_heads, 1, blocks, v_size),
                (batch, q_heads, 1, blocks),
                (batch, q_heads, 1, blocks),
            ],
            output_dtypes=[dtype, mx.float32, mx.float32],
        )
        kernel_2 = lang._qwen3_5_ragged_sdpa_two_pass_2_kernel(dtype, v_size, blocks)
        (out,) = kernel_2(
            inputs=[partials, sums, maxs],
            template=[
                ("T", dtype),
                ("D_SIZE", int(v_size)),
                ("BLOCKS", int(blocks)),
            ],
            grid=(1024, batch * q_heads, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(batch, q_heads, 1, v_size)],
            output_dtypes=[dtype],
        )
        mx.eval(out)
        return True
    except (ValueError, RuntimeError) as exc:
        if _is_threadgroup_limit_error(exc):
            logger.warning(
                "Disabling mlx-vlm Qwen3.5 ragged SDPA %s variant "
                "(dtype=%s, d=%d, v=%d, q_heads=%d, kv_heads=%d, "
                "batch=%d, k_size=%d, blocks=%d): %s",
                mode,
                dtype_name,
                d_size,
                v_size,
                q_heads,
                kv_heads,
                batch,
                probe_k_size,
                blocks,
                exc,
            )
        else:
            logger.debug(
                "Qwen3.5 ragged SDPA %s probe failed; falling back to reference path",
                mode,
                exc_info=True,
            )
        return False


def _variant_supported(
    dtype: Any,
    d_size: int,
    v_size: int,
    q_heads: int,
    kv_heads: int,
    batch: int,
    k_size: int,
    mode: str,
    blocks: int,
) -> bool:
    return _variant_supported_cached(
        *_cache_key(dtype, d_size, v_size, q_heads, kv_heads, batch, k_size, mode, blocks)
    )


def apply_qwen35_ragged_sdpa_guard_patch() -> bool:
    """Install the ragged-SDPA probe guard on mlx-vlm's Qwen3.5 module."""
    global _APPLIED, _ORIGINAL_RAGGED_DECODE_ATTENTION

    try:
        lang = importlib.import_module("mlx_vlm.models.qwen3_5.language")
    except Exception as exc:
        logger.debug("mlx-vlm Qwen3.5 language module unavailable: %s", exc)
        return False

    current = getattr(lang, "_qwen3_5_ragged_decode_attention", None)
    if current is None:
        return False
    if getattr(current, "_omlx_ragged_sdpa_guard", False):
        _APPLIED = True
        return True

    _ORIGINAL_RAGGED_DECODE_ATTENTION = current

    def guarded_ragged_decode_attention(
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        pads: list[int],
        scale: float,
    ) -> mx.array | None:
        # Mirror mlx-vlm's fast-path eligibility checks before probing so the
        # wrapper remains passthrough-safe for unsupported shapes/backends.
        if not mx.metal.is_available():
            return None
        if (
            queries.ndim != 4
            or keys.ndim != 4
            or values.ndim != 4
            or queries.shape[2] != 1
            or queries.dtype not in (mx.bfloat16, mx.float16)
            or keys.dtype != queries.dtype
            or values.dtype != queries.dtype
        ):
            return None

        batch, q_heads, _, d_size = queries.shape
        pads_tuple = tuple(int(p) for p in pads)
        if len(pads_tuple) != batch or any(p < 0 for p in pads_tuple):
            return None
        kv_heads = keys.shape[1]
        k_size = keys.shape[2]
        v_size = values.shape[-1]
        if (
            q_heads % kv_heads != 0
            or d_size != v_size
            or d_size not in (64, 96, 128, 256)
            or any(p >= k_size for p in pads_tuple)
        ):
            return None

        plans = [
            lang._qwen3_5_sdpa_vector_plan(k_size - pad, q_heads, kv_heads)
            for pad in pads_tuple
        ]
        if len(set(plans)) != 1:
            return None
        mode, blocks = plans[0]

        variant_key = _cache_key(
            queries.dtype,
            int(d_size),
            int(v_size),
            int(q_heads),
            int(kv_heads),
            int(batch),
            int(k_size),
            str(mode),
            int(blocks),
        )
        if variant_key in _RUNTIME_DISABLED_VARIANTS:
            return None
        if not _variant_supported(
            queries.dtype,
            int(d_size),
            int(v_size),
            int(q_heads),
            int(kv_heads),
            int(batch),
            int(k_size),
            str(mode),
            int(blocks),
        ):
            _RUNTIME_DISABLED_VARIANTS.add(variant_key)
            return None

        output = _ORIGINAL_RAGGED_DECODE_ATTENTION(
            queries, keys, values, list(pads_tuple), scale
        )
        if variant_key not in _RUNTIME_VERIFIED_VARIANTS:
            try:
                # The probe above uses tiny synthetic inputs. Do one eager
                # validation with the real graph for this shape bucket so lazy
                # Metal launch errors are caught here instead of later in
                # mlx_lm.generate's mx.async_eval.
                mx.eval(output)
            except (ValueError, RuntimeError) as exc:
                if _is_threadgroup_limit_error(exc):
                    logger.warning(
                        "Disabling mlx-vlm Qwen3.5 ragged SDPA %s variant "
                        "after real-shape validation failed: %s",
                        mode,
                        exc,
                    )
                    _RUNTIME_DISABLED_VARIANTS.add(variant_key)
                    return None
                logger.debug(
                    "Qwen3.5 ragged SDPA real-shape validation failed; "
                    "falling back to reference path",
                    exc_info=True,
                )
                _RUNTIME_DISABLED_VARIANTS.add(variant_key)
                return None
            _RUNTIME_VERIFIED_VARIANTS.add(variant_key)
        return output

    guarded_ragged_decode_attention._omlx_ragged_sdpa_guard = True
    lang._qwen3_5_ragged_decode_attention = guarded_ragged_decode_attention
    _APPLIED = True
    logger.info("mlx-vlm Qwen3.5 ragged SDPA guard patch applied")
    return True


__all__ = [
    "apply_qwen35_ragged_sdpa_guard_patch",
    "_is_threadgroup_limit_error",
]
