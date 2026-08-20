"""Fast GLM kernels with a fallback to patched ``mlx.core.fast`` symbols."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


def _detach_import_error(exc: Exception) -> Exception:
    """Keep the diagnostic message without retaining import caller frames."""
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None
    return exc


try:
    from . import _ext
except Exception as exc:  # pragma: no cover - depends on local native build
    _ext = None
    _IMPORT_ERROR = _detach_import_error(exc)
    # Default installs ship no extension; warn only when a built _ext fails
    # to load (e.g. unresolved @rpath/libmlx.dylib, issue #2233) so the
    # silent-slow-path fallback leaves a trace in the server log.
    if any(Path(__file__).parent.glob("_ext*.so")):
        logger.warning(
            "%s: native extension is present but failed to load; falling "
            "back to the slow path: %s",
            __name__,
            _IMPORT_ERROR,
        )
else:
    _IMPORT_ERROR = None


def _verify_abi(ext, import_error):
    """Disable the native symbols when the extension rejects mlx arrays.

    An extension built with a nanobind whose ABI tag differs from the mlx
    wheel's imports cleanly and lists every symbol, but its type casters
    live in an isolated NB_DOMAIN, so every call raises ``TypeError:
    incompatible function arguments`` (issue #2139). Probe once at import
    and degrade with a single warning instead of failing per call; builds
    predating the ``abi_probe`` binding are assumed compatible.
    """
    if ext is None:
        return ext, import_error
    probe = getattr(ext, "abi_probe", None)
    if probe is None:
        return ext, import_error
    try:
        probe(mx.zeros((1,)))
    except TypeError as exc:
        logger.warning(
            "%s: native kernels disabled — the extension was built with a "
            "nanobind ABI that does not match this mlx wheel; rebuild it "
            "against the installed mlx (see pyproject build-system pins).",
            __name__,
        )
        return None, _detach_import_error(exc)
    return ext, import_error


_ext, _IMPORT_ERROR = _verify_abi(_ext, _IMPORT_ERROR)


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


NATIVE_SYMBOLS = (
    "dsa_decode_scores",
    "dsa_indexer_scores",
    "dsa_topk_indices",
    "dspark_fp32_topk_indices",
    "dspark_exact_mxfp8_qmv_pair",
    "glm_dsa_sparse_mla_attention",
    "glm_dsa_exact_block_attention",
    "deepseek_v4_sparse_attention",
    "dspark_ring_gemm",
    "dspark_rowwise_gemm",
    "glm_dsa_q8_vup_flat",
    "glm_moe_weighted_sum",
    "deepseek_mxfp4_gather_qmm_blocks",
    "deepseek_mxfp4_gather_qmm_pair_blocks",
    "deepseek_mxfp4_gather_qmm_pair_concat_blocks",
    "deepseek_mxfp4_gather_qmm_expert",
    "deepseek_affine_gather_qmm_blocks",
    "deepseek_affine_gather_qmm_pair_concat_blocks",
)


def is_native_available() -> bool:
    return _ext is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def has_symbol(name: str) -> bool:
    return hasattr(_ext, name) or hasattr(mx.fast, name)


def native_symbols() -> tuple[str, ...]:
    if _ext is None:
        return ()
    return tuple(name for name in NATIVE_SYMBOLS if hasattr(_ext, name))


def missing_symbols(required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not has_symbol(name)]


def _native_stream_kwargs(stream) -> dict[str, object]:
    """Accept the same stream shorthand that mlx.fast kernels accept."""
    if isinstance(stream, mx.DeviceType):
        stream = None
    return {"stream": stream}


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
    if _ext is not None:
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
    else:
        scores = mx.fast.dsa_indexer_scores(
            queries,
            keys,
            weights,
            causal=causal,
            unused_causal_prefix_topk=unused_causal_prefix_topk,
            skip_causal_future_store=skip_causal_future_store,
            causal_q_offset=causal_q_offset,
            stream=stream or mx.gpu,
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
            "dsa_decode_scores requires the native glm_moe_dsa extension"
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
    return mx.fast.dsa_topk_indices(
        scores,
        topk,
        bucketed=bucketed,
        causal_valid_prefix=causal_valid_prefix,
        stream=stream or mx.gpu,
    )


def dspark_fp32_topk_indices(
    scores: mx.array,
    topk: int = 512,
    *,
    stream=None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "dspark_fp32_topk_indices"):
        raise RuntimeError("DSpark FP32 top-k kernel is unavailable")
    return _ext.dspark_fp32_topk_indices(
        scores,
        topk,
        **_native_stream_kwargs(stream),
    )


def glm_dsa_sparse_mla_attention(
    q_latent: mx.array,
    q_pe: mx.array,
    kv_latent: mx.array,
    k_pe: mx.array,
    topk_indices: mx.array,
    scale: float,
    causal: bool = True,
    topk_valid_prefix: bool = False,
    causal_prefix_indices: bool = False,
    topk_length: mx.array | None = None,
    causal_prefix_rows: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None:
        return _ext.glm_dsa_sparse_mla_attention(
            q_latent,
            q_pe,
            kv_latent,
            k_pe,
            topk_indices,
            scale,
            causal=causal,
            topk_valid_prefix=topk_valid_prefix,
            causal_prefix_indices=causal_prefix_indices,
            topk_length=topk_length,
            causal_prefix_rows=causal_prefix_rows,
            **_native_stream_kwargs(stream),
        )
    return mx.fast.glm_dsa_sparse_mla_attention(
        q_latent,
        q_pe,
        kv_latent,
        k_pe,
        topk_indices,
        scale,
        causal=causal,
        topk_valid_prefix=topk_valid_prefix,
        causal_prefix_indices=causal_prefix_indices,
        topk_length=topk_length,
        causal_prefix_rows=causal_prefix_rows,
        stream=stream or mx.gpu,
    )


def glm_dsa_exact_block_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    block_mask: mx.array,
    block_token_mask: mx.array,
    scale: float,
    causal: bool = True,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "glm_dsa_exact_block_attention"):
        return _ext.glm_dsa_exact_block_attention(
            q,
            k,
            v,
            block_mask,
            block_token_mask,
            scale,
            causal=causal,
            **_native_stream_kwargs(stream),
        )
    return mx.fast.glm_dsa_exact_block_attention(
        q,
        k,
        v,
        block_mask,
        block_token_mask,
        scale,
        causal=causal,
        stream=stream or mx.gpu,
    )


def dspark_rowwise_gemm(
    lhs: mx.array,
    rhs: mx.array,
    transpose_rhs: bool,
    *,
    stream=None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "dspark_rowwise_gemm"):
        raise RuntimeError("DSpark rowwise NAX GEMM is unavailable")
    return _ext.dspark_rowwise_gemm(
        lhs,
        rhs,
        transpose_rhs,
        **_native_stream_kwargs(stream),
    )


def dspark_ring_gemm(
    lhs: mx.array,
    source: mx.array,
    indices: mx.array,
    transpose_rhs: bool,
    *,
    stream=None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "dspark_ring_gemm"):
        raise RuntimeError("DSpark physical-ring GEMM is unavailable")
    return _ext.dspark_ring_gemm(
        lhs,
        source,
        indices,
        transpose_rhs,
        **_native_stream_kwargs(stream),
    )


def dspark_exact_mxfp8_qmv_pair(
    input: mx.array,
    weight_a: mx.array,
    scales_a: mx.array,
    weight_b: mx.array,
    scales_b: mx.array,
    *,
    stream=None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "dspark_exact_mxfp8_qmv_pair"):
        raise RuntimeError("DSpark exact MXFP8 QMV pair kernel is unavailable")
    return _ext.dspark_exact_mxfp8_qmv_pair(
        input,
        weight_a,
        scales_a,
        weight_b,
        scales_b,
        **_native_stream_kwargs(stream),
    )


def deepseek_v4_sparse_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: mx.array,
    topk_indices: mx.array,
    sinks: mx.array,
    scale: float,
    q_offset: int,
    compress_ratio: int,
    local_window: int,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "deepseek_v4_sparse_attention"):
        return _ext.deepseek_v4_sparse_attention(
            q,
            local_kv,
            pooled,
            topk_indices,
            sinks,
            scale,
            q_offset,
            compress_ratio,
            local_window,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("deepseek_v4_sparse_attention native kernel is unavailable")


def glm_dsa_q8_vup_flat(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "glm_dsa_q8_vup_flat"):
        return _ext.glm_dsa_q8_vup_flat(
            x,
            weight,
            scales,
            biases,
            **_native_stream_kwargs(stream),
        )
    return mx.fast.glm_dsa_q8_vup_flat(
        x,
        weight,
        scales,
        biases,
        stream=stream or mx.gpu,
    )


def glm_moe_weighted_sum(
    x_sorted: mx.array,
    inv_order: mx.array,
    scores: mx.array,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "glm_moe_weighted_sum"):
        return _ext.glm_moe_weighted_sum(
            x_sorted,
            inv_order,
            scores,
            **_native_stream_kwargs(stream),
        )
    return mx.fast.glm_moe_weighted_sum(
        x_sorted,
        inv_order,
        scores,
        stream=stream or mx.gpu,
    )


def deepseek_mxfp4_gather_qmm_blocks(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    block_meta: mx.array,
    block_count: mx.array,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "deepseek_mxfp4_gather_qmm_blocks"):
        return _ext.deepseek_mxfp4_gather_qmm_blocks(
            x,
            weight,
            scales,
            block_meta,
            block_count,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("deepseek_mxfp4_gather_qmm_blocks native kernel is unavailable")


def deepseek_mxfp4_gather_qmm_pair_blocks(
    x: mx.array,
    weight0: mx.array,
    scales0: mx.array,
    weight1: mx.array,
    scales1: mx.array,
    block_meta: mx.array,
    block_count: mx.array,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "deepseek_mxfp4_gather_qmm_pair_blocks"):
        return _ext.deepseek_mxfp4_gather_qmm_pair_blocks(
            x,
            weight0,
            scales0,
            weight1,
            scales1,
            block_meta,
            block_count,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError(
        "deepseek_mxfp4_gather_qmm_pair_blocks native kernel is unavailable"
    )


def deepseek_mxfp4_gather_qmm_pair_concat_blocks(
    x: mx.array,
    weight0: mx.array,
    scales0: mx.array,
    weight1: mx.array,
    scales1: mx.array,
    block_meta: mx.array,
    block_count: mx.array,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(
        _ext, "deepseek_mxfp4_gather_qmm_pair_concat_blocks"
    ):
        return _ext.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
            x,
            weight0,
            scales0,
            weight1,
            scales1,
            block_meta,
            block_count,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError(
        "deepseek_mxfp4_gather_qmm_pair_concat_blocks native kernel is unavailable"
    )


def deepseek_mxfp4_gather_qmm_expert(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    indices: mx.array,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "deepseek_mxfp4_gather_qmm_expert"):
        return _ext.deepseek_mxfp4_gather_qmm_expert(
            x,
            weight,
            scales,
            indices,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("deepseek_mxfp4_gather_qmm_expert native kernel is unavailable")


def deepseek_affine_gather_qmm_blocks(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    block_meta: mx.array,
    block_count: mx.array,
    group_size: int,
    bits: int,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "deepseek_affine_gather_qmm_blocks"):
        return _ext.deepseek_affine_gather_qmm_blocks(
            x,
            weight,
            scales,
            biases,
            block_meta,
            block_count,
            group_size,
            bits,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("deepseek_affine_gather_qmm_blocks native kernel is unavailable")


def deepseek_affine_gather_qmm_pair_concat_blocks(
    x: mx.array,
    weight0: mx.array,
    scales0: mx.array,
    biases0: mx.array,
    weight1: mx.array,
    scales1: mx.array,
    biases1: mx.array,
    block_meta: mx.array,
    block_count: mx.array,
    group_size: int,
    bits: int,
    variant: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(
        _ext, "deepseek_affine_gather_qmm_pair_concat_blocks"
    ):
        return _ext.deepseek_affine_gather_qmm_pair_concat_blocks(
            x,
            weight0,
            scales0,
            biases0,
            weight1,
            scales1,
            biases1,
            block_meta,
            block_count,
            group_size,
            bits,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError(
        "deepseek_affine_gather_qmm_pair_concat_blocks native kernel is unavailable"
    )


def __getattr__(name: str) -> Any:
    if _ext is not None and hasattr(_ext, name):
        return getattr(_ext, name)
    return getattr(mx.fast, name)


def __dir__() -> list[str]:
    names = set(globals())
    names.update(NATIVE_SYMBOLS)
    names.update(dir(mx.fast))
    if _ext is not None:
        names.update(dir(_ext))
    return sorted(names)
