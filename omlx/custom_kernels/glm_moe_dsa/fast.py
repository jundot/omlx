"""Fast GLM kernels with a fallback to patched ``mlx.core.fast`` symbols."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

try:
    from . import _ext
except Exception as exc:  # pragma: no cover - depends on local native build
    _ext = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


NATIVE_SYMBOLS = (
    "dsa_indexer_scores",
    "dsa_topk_indices",
    "glm_dsa_sparse_mla_attention",
    "glm_dsa_exact_block_attention",
    "glm_dsa_q8_vup_flat",
    "glm_moe_weighted_sum",
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


def dsa_indexer_scores(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    causal: bool = True,
    unused_causal_prefix_topk: int = 0,
    skip_causal_future_store: bool = False,
    causal_q_offset: int = -1,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None:
        return _ext.dsa_indexer_scores(
            queries,
            keys,
            weights,
            causal=causal,
            unused_causal_prefix_topk=unused_causal_prefix_topk,
            skip_causal_future_store=skip_causal_future_store,
            causal_q_offset=causal_q_offset,
            **_native_stream_kwargs(stream),
        )
    return mx.fast.dsa_indexer_scores(
        queries,
        keys,
        weights,
        causal=causal,
        unused_causal_prefix_topk=unused_causal_prefix_topk,
        skip_causal_future_store=skip_causal_future_store,
        causal_q_offset=causal_q_offset,
        stream=stream or mx.gpu,
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
