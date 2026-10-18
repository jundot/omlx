"""Sparse-attention index kernels: indexer scores and top-k selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from .._loader import load_extension
from .._loader import native_stream_kwargs as _native_stream_kwargs

_ext, _IMPORT_ERROR = load_extension(__package__, Path(__file__).parent)


def is_native_available() -> bool:
    return _ext is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def has_symbol(name: str) -> bool:
    return hasattr(_ext, name)


def _probe_mask_fold(ext) -> bool:
    """True iff the built extension accepts the mask-fold kwargs.

    ``dsa_indexer_scores`` grew ``mask_ratio``/``mask_q_offset`` after the
    first native builds shipped. An older ``_ext`` parses no such kwargs, so
    passing them unconditionally raises ``TypeError`` for every caller —
    including GLM-5.2's historical unmasked path. Nanobind renders named
    args into ``__doc__``, so probe the signature once at import; callers
    on older builds keep the historical call signature and the mask is
    applied in a second pass with identical sentinel semantics.
    """
    fn = getattr(ext, "dsa_indexer_scores", None)
    if fn is None:
        return False
    doc = getattr(fn, "__doc__", None) or ""
    return "mask_ratio" in doc and "mask_q_offset" in doc


_EXT_MASK_FOLD = _probe_mask_fold(_ext)


def _probe_mma_score(ext) -> bool:
    """True iff the built extension exposes the v25 M2 MMA score kernel."""
    return getattr(ext, "dsa_indexer_scores_mma", None) is not None


_EXT_MMA_SCORE = _probe_mma_score(_ext)


def dsa_indexer_scores_mma(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    mask_ratio: int = 0,
    mask_q_offset: int = 0,
    *,
    stream=None,
) -> mx.array:
    """v25 M2 from-scratch MMA indexer scores (~1.37x over the Steel kernel).

    Serves ONLY bf16, H=64, D=128, weights rank 3 ([B, L, H]), non-causal;
    the extension raises on anything else — callers gate and fall back to
    ``dsa_indexer_scores``. Same fused pooled-ratio mask semantics
    (``mask_ratio``/``mask_q_offset``) and bit-exact output vs the Steel
    kernel. No slow-path fallback: requires a local extension build that
    exposes the symbol (probe with ``_EXT_MMA_SCORE``).
    """
    if not (_ext is not None and _EXT_MMA_SCORE):
        raise RuntimeError(
            "dsa_indexer_scores_mma requires a local extension build that "
            "exposes the v25 MMA score kernel"
        )
    return _ext.dsa_indexer_scores_mma(
        queries,
        keys,
        weights,
        mask_ratio=mask_ratio,
        mask_q_offset=mask_q_offset,
        **_native_stream_kwargs(stream),
    )


def dsa_indexer_scores(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    causal: bool = True,
    unused_causal_prefix_topk: int = 0,
    skip_causal_future_store: bool = False,
    causal_q_offset: int = -1,
    mask_ratio: int = 0,
    mask_q_offset: int = 0,
    *,
    stream=None,
) -> mx.array:
    """Head-summed DSA indexer scores.

    ``mask_ratio > 0`` folds the pooled-ratio causal mask into the kernel
    epilogue: pooled column ``c`` is masked for query row ``r`` iff
    ``c >= (mask_q_offset + r + 1) // mask_ratio`` and receives the
    ``finfo(dtype).min`` sentinel — bit-identical to applying
    ``mx.where(mask, scores, finfo.min)`` in a second pass. ``mask_ratio=0``
    (default) is the historical unmasked behavior. On extension builds
    predating the fold kwargs, the historical call signature is kept and
    the same mask is applied in a second pass with identical semantics.
    """
    if _ext is not None and _EXT_MASK_FOLD:
        return _ext.dsa_indexer_scores(
            queries,
            keys,
            weights,
            causal=causal,
            unused_causal_prefix_topk=unused_causal_prefix_topk,
            skip_causal_future_store=skip_causal_future_store,
            causal_q_offset=causal_q_offset,
            mask_ratio=mask_ratio,
            mask_q_offset=mask_q_offset,
            **_native_stream_kwargs(stream),
        )
    if _ext is None:
        raise RuntimeError("dsa_indexer_scores native kernel is unavailable")
    # Older build without the mask-fold kwargs: keep the historical
    # call signature; the mask is applied in a second pass below.
    scores = _ext.dsa_indexer_scores(
        queries,
        keys,
        weights,
        causal=causal,
        unused_causal_prefix_topk=unused_causal_prefix_topk,
        skip_causal_future_store=skip_causal_future_store,
        causal_q_offset=causal_q_offset,
        **_native_stream_kwargs(stream),
    )
    if mask_ratio > 0:
        # Preserve the fused kernel's exact sentinel semantics on the
        # non-fused paths (same validity rule, same finfo.min sentinel).
        L = queries.shape[2]
        P = keys.shape[2]
        pool_idx = mx.arange(P)
        query_idx = mx.arange(mask_q_offset + 1, mask_q_offset + L + 1)
        mask = pool_idx < query_idx[:, None] // mask_ratio
        scores = mx.where(
            mask[None, None], scores, mx.finfo(scores.dtype).min
        )
    return scores


def dsa_decode_scores(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    fp32_scores: bool = False,
    *,
    stream=None,
) -> mx.array:
    if _ext is None:
        raise RuntimeError(
            "dsa_decode_scores requires the native sparse_index extension"
        )
    return _ext.dsa_decode_scores(
        queries,
        keys,
        weights,
        fp32_scores=fp32_scores,
        **_native_stream_kwargs(stream),
    )


def dsa_topk_indices(
    scores: mx.array,
    topk: int,
    bucketed: bool = False,
    causal_valid_prefix: bool = False,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None:
        return _ext.dsa_topk_indices(
            scores,
            topk,
            bucketed=bucketed,
            causal_valid_prefix=causal_valid_prefix,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("dsa_topk_indices native kernel is unavailable")


def fp32_topk_indices(
    scores: mx.array,
    topk: int = 512,
    *,
    stream=None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "fp32_topk_indices"):
        raise RuntimeError("fp32_topk_indices native kernel is unavailable")
    return _ext.fp32_topk_indices(
        scores,
        topk,
        **_native_stream_kwargs(stream),
    )


def qsa_indexer_scores(
    queries: mx.array,
    pooled_keys: mx.array,
    mask_ratio: int = 4,
    mask_q_offset: int = 0,
    *,
    stream=None,
) -> mx.array:
    """Fused, fp32 Qwen4 QSA block scores for the M3 prefill geometry.

    ``queries`` is ``[1, 4, M, 128]`` and ``pooled_keys`` is
    ``[1, 1, N, 128]`` in matching bf16/fp16. The result is fp32
    ``[1, M, N]`` after head-summed ReLU, ``1/sqrt(128)`` scaling, and the
    pooled-causal mask. The dedicated ABI intentionally has no implicit
    fallback; ``qsa_fast`` owns the portable float32 fallback.
    """
    if _ext is None or not hasattr(_ext, "qsa_indexer_scores"):
        raise RuntimeError(
            "qsa_indexer_scores requires a local extension build that "
            "exposes the Qwen4 QSA score kernel"
        )
    return _ext.qsa_indexer_scores(
        queries,
        pooled_keys,
        mask_ratio=mask_ratio,
        mask_q_offset=mask_q_offset,
        **_native_stream_kwargs(stream),
    )


def qsa_topk_indices(
    scores: mx.array,
    topk: int = 512,
    *,
    stream=None,
) -> mx.array:
    """Select Qwen4's top 512 block indices from fp32 ``[1, M, N]`` scores.

    The native radix path returns only ``[1, M, 512]`` uint32 indices, avoiding
    the full-width index allocation made by ``mx.argpartition``. Exact cutoff
    ties retain their highest-index members, matching the set selected by the
    portable QSA expression. The ABI is deliberately fixed to ``topk=512`` and
    fails closed when the rebuilt extension or production geometry is absent.
    """
    if _ext is None or not hasattr(_ext, "qsa_topk_indices"):
        raise RuntimeError("qsa_topk_indices native kernel is unavailable")
    return _ext.qsa_topk_indices(
        scores,
        topk,
        **_native_stream_kwargs(stream),
    )


def msa_block_topk(
    idx_queries: mx.array,
    idx_keys: mx.array,
    *,
    q_start: int,
    scale: float,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    stream=None,
) -> mx.array:
    if _ext is not None:
        return _ext.msa_block_topk(
            idx_queries,
            idx_keys,
            q_start,
            scale,
            block_size,
            topk,
            init_blocks,
            local_blocks,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("msa_block_topk native kernel is unavailable")


def __getattr__(name: str) -> Any:
    if _ext is not None and hasattr(_ext, name):
        return getattr(_ext, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    if _ext is not None:
        names.update(dir(_ext))
    return sorted(names)
