# SPDX-License-Identifier: Apache-2.0
"""SpecPrefill: Attention-based sparse prefill for MLX.

Reduces TTFT on long prompts by using a small draft model to identify
important tokens, then prefilling only those tokens on the target model
while preserving original positional encoding via manual RoPE.

Based on arxiv.org/abs/2502.02789 and waybarrios/vllm-mlx PR #180.

Pipeline:
  1. score_tokens()  — draft model scores token importance via attention
  2. select_chunks() — chunk-based top-K% selection
  3. sparse_prefill() — target prefill with manual RoPE at original positions
  4. cleanup_rope()  — restore original RoPE after generation

Design notes:
  - RoPE is relative: Q_m @ K_p^T depends only on (m - p). Selected keys
    stored contiguously in cache with correct RoPE angles produce correct
    attention during decode.
  - After sparse prefill of N tokens from M total, cache.offset = N but
    decode needs position M. _OffsetAdjustedRoPE adds (M - N) to each offset.
"""

from __future__ import annotations

import inspect
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)


# ===========================================================================
# Token importance scoring (draft model)
# ===========================================================================


class _AttentionCapture:
    """Wraps attention to capture post-RoPE query vectors during lookahead.

    Delegates to the original attention module while recording queries
    for importance scoring.
    """

    def __init__(self, original, buf_idx, query_buffer, query_extractor):
        self._original = original
        self._buf_idx = buf_idx
        self._query_buffer = query_buffer
        self._query_extractor = query_extractor

    def __call__(self, x, mask=None, cache=None, **kwargs):
        if kwargs and _accepts_extractor_kwargs(self._query_extractor, kwargs):
            queries = self._query_extractor(self._original, x, cache, **kwargs)
        else:
            # Backward compatibility for custom extractors with the older
            # (attn, x, cache) signature.
            queries = self._query_extractor(self._original, x, cache)
        self._query_buffer[self._buf_idx].append(queries)
        return self._original(x, mask=mask, cache=cache, **kwargs)

    def __getattr__(self, name):
        return getattr(self._original, name)


# ---------------------------------------------------------------------------
# Query extractors (architecture-specific scoring only)
# ---------------------------------------------------------------------------


def _accepts_extractor_kwargs(extractor, kwargs) -> bool:
    """Whether a query extractor accepts the supplied keyword arguments."""
    try:
        params = inspect.signature(extractor).parameters.values()
    except (TypeError, ValueError):
        return True

    for param in params:
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True

    accepted = {
        param.name
        for param in params
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return set(kwargs).issubset(accepted)


def _qwen35_extract_queries(attn, x, cache=None, **kwargs):
    """Qwen3.5: gate split + q_norm + RoPE."""
    B, L, D = x.shape
    q_out = attn.q_proj(x)
    queries, _gate = mx.split(
        q_out.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
    )
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _qwen36_extract_queries(attn, x, cache=None, **kwargs):
    """Qwen3.6 MoE / non-gated q_norm models: q_proj + q_norm + RoPE."""
    B, L, _ = x.shape
    n_heads = getattr(
        attn,
        "num_attention_heads",
        getattr(attn, "n_heads", getattr(attn, "num_heads", None)),
    )
    queries = attn.q_proj(x).reshape(B, L, n_heads, -1)
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _llama_extract_queries(attn, x, cache=None, **kwargs):
    """Standard transformer: q_proj + reshape + RoPE."""
    B, L, D = x.shape
    n_heads = getattr(
        attn,
        "num_attention_heads",
        getattr(attn, "n_heads", getattr(attn, "num_heads", None)),
    )
    queries = attn.q_proj(x)
    queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _gemma4_extract_queries(attn, x, cache=None, offset=None, **kwargs):
    """Gemma 4: q_proj + per-head q_norm + RoPE with shared-KV offset support."""
    B, L, D = x.shape
    n_heads = getattr(attn, "n_heads", getattr(attn, "num_attention_heads", None))
    queries = attn.q_proj(x).reshape(B, L, n_heads, -1)
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    rope_offset = (
        offset if offset is not None else (cache.offset if cache is not None else 0)
    )
    queries = attn.rope(queries, offset=rope_offset)
    return queries


def _nemotron_h_extract_queries(attn, x, cache=None, **kwargs):
    """Nemotron-H: q_proj only, no RoPE (content-based attention)."""
    B, L, D = x.shape
    queries = attn.q_proj(x).reshape(B, L, attn.num_heads, -1).transpose(0, 2, 1, 3)
    return queries


# ---------------------------------------------------------------------------
# Model topology helpers
# ---------------------------------------------------------------------------


def _find_attention_layers(model) -> List[Tuple[int, Any]]:
    """Find all full-attention layers across architectures.

    Supports self_attn (standard) and block_type=="*" (Nemotron-H).
    """
    results = []
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "self_attn"):
            results.append((idx, layer))
        elif getattr(layer, "block_type", None) == "*":
            results.append((idx, layer))
    return results


def _get_attn_module(layer):
    """Get attention module from a layer (self_attn or mixer)."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if getattr(layer, "block_type", None) == "*":
        return layer.mixer
    return None


def _set_attn_module(layer, module):
    """Set attention module on a layer."""
    if hasattr(layer, "self_attn"):
        layer.self_attn = module
    elif getattr(layer, "block_type", None) == "*":
        layer.mixer = module


def _build_layer_to_cache_map(model) -> Dict[int, int]:
    """Build layer_idx → cache_idx mapping.

    Standard models: identity. Nemotron-H: compacted (only M/* layers).
    Gemma 4: shared-KV layers reuse earlier caches via previous_kvs, which
    lives on Gemma4TextModel — reachable at ``.model`` for the text
    model or ``.language_model.model`` for the VLM wrapper.
    """
    for root in (
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(model, "model", None),
        model,
    ):
        previous_kvs = getattr(root, "previous_kvs", None)
        if previous_kvs is not None:
            return dict(enumerate(previous_kvs))

    has_block_type = any(hasattr(layer, "block_type") for layer in model.layers)
    if not has_block_type:
        return {i: i for i in range(len(model.layers))}

    layer_to_cache = {}
    cache_idx = 0
    for layer_idx, layer in enumerate(model.layers):
        bt = getattr(layer, "block_type", None)
        if bt in ("M", "*"):
            layer_to_cache[layer_idx] = cache_idx
            cache_idx += 1
    return layer_to_cache


def _linear_output_dims(linear) -> Optional[int]:
    """Best-effort output dimension lookup for Linear / QuantizedLinear layers."""
    if linear is None:
        return None
    if hasattr(linear, "out_features"):
        return getattr(linear, "out_features")
    if hasattr(linear, "output_dims"):
        return getattr(linear, "output_dims")
    weight = getattr(linear, "weight", None)
    if weight is not None and getattr(weight, "shape", None):
        return weight.shape[0]
    return None


def _uses_gated_q_proj(attn_obj) -> bool:
    """Detect Qwen-style gated q_proj layout from projection dimensions."""
    n_heads = getattr(
        attn_obj,
        "num_attention_heads",
        getattr(attn_obj, "n_heads", getattr(attn_obj, "num_heads", None)),
    )
    head_dim = getattr(attn_obj, "head_dim", None)
    q_out = _linear_output_dims(getattr(attn_obj, "q_proj", None))
    if None in (n_heads, head_dim, q_out):
        return False
    return q_out == 2 * n_heads * head_dim


def _uses_non_gated_q_norm(attn_obj) -> bool:
    """Detect Qwen3.6-style attention: per-head RMSNorm on q, no gate split.

    Matches when q_norm has weight shape (head_dim,) and q_proj outputs
    n_heads * head_dim. Filters out olmo-style flat q_norm and cohere-style
    LayerNorm2D, whose q_norm.weight.shape[0] does not equal head_dim.
    """
    q_norm = getattr(attn_obj, "q_norm", None)
    if q_norm is None:
        return False
    q_norm_weight = getattr(q_norm, "weight", None)
    if q_norm_weight is None:
        return False
    shape = getattr(q_norm_weight, "shape", None)
    if not shape:
        return False
    q_norm_dim = shape[0]

    n_heads = getattr(
        attn_obj,
        "num_attention_heads",
        getattr(attn_obj, "n_heads", getattr(attn_obj, "num_heads", None)),
    )
    q_out = _linear_output_dims(getattr(attn_obj, "q_proj", None))
    if not n_heads or q_out is None:
        return False
    return q_out == n_heads * q_norm_dim


def _is_gemma4_attention(attn_obj) -> bool:
    """Detect Gemma 4 attention by its shared-KV / offset call contract."""
    if not hasattr(attn_obj, "q_norm") or not hasattr(attn_obj, "rope"):
        return False
    call = getattr(attn_obj, "__call__", None)
    if call is None:
        return False
    try:
        params = inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False
    return "shared_kv" in params and "offset" in params


def _detect_query_extractor(attn_obj) -> Callable:
    """Auto-detect the appropriate query extractor for the model architecture."""
    if _uses_gated_q_proj(attn_obj):
        return _qwen35_extract_queries
    elif _is_gemma4_attention(attn_obj):
        return _gemma4_extract_queries
    elif not hasattr(attn_obj, "rope"):
        return _nemotron_h_extract_queries
    elif _uses_non_gated_q_norm(attn_obj):
        return _qwen36_extract_queries
    else:
        return _llama_extract_queries


# ---------------------------------------------------------------------------
# Scoring pipeline internals
# ---------------------------------------------------------------------------


def _patch_attention_for_capture(model, query_buffer, query_extractor):
    """Replace attention modules with capture wrappers.

    Returns (originals, attn_layer_indices) for cleanup.
    """
    originals = []
    attn_indices = []
    for layer_idx, layer in _find_attention_layers(model):
        buf_idx = len(attn_indices)
        attn_indices.append(layer_idx)
        orig = _get_attn_module(layer)
        _set_attn_module(
            layer,
            _AttentionCapture(orig, buf_idx, query_buffer, query_extractor),
        )
        originals.append((layer_idx, orig))
    return originals, attn_indices


def _unpatch_attention_capture(model, originals):
    """Restore original attention modules after capture."""
    for layer_idx, orig in originals:
        _set_attn_module(model.layers[layer_idx], orig)


def _prefill_draft(model, tokens, cache, step_size=2048):
    """Prefill draft model with all prompt tokens. Returns last logits."""
    prompt = mx.array(tokens) if not isinstance(tokens, mx.array) else tokens
    n = len(tokens)
    processed = 0
    while n - processed > 1:
        chunk = min(step_size, n - processed - 1)
        model(prompt[processed : processed + chunk][None], cache=cache)
        mx.eval([c.state for c in cache])
        processed += chunk
        mx.clear_cache()
    logits = model(prompt[processed:][None], cache=cache)
    mx.eval(logits)
    return logits


def _lookahead_decode(model, first_logits, cache, n_steps, temp=0.6, top_p=0.95):
    """Run n_steps autoregressive decode, capturing queries via patched attention."""
    from ..utils.sampling import make_sampler

    sampler = make_sampler(temp=temp, top_p=top_p)
    y = sampler(first_logits[:, -1, :])
    mx.eval(y)
    generated = [y.item()]
    for _ in range(n_steps):
        logits = model(y.reshape(1, -1), cache=cache)
        y = sampler(logits[:, -1, :])
        mx.eval(y)
        generated.append(y.item())
    return generated


def _avg_pool1d(x, kernel_size):
    """1D average pooling along last axis via prefix-sum."""
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    padded = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)])
    zeros = mx.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    prefix = mx.concatenate([zeros, mx.cumsum(padded, axis=-1)], axis=-1)
    return (prefix[..., kernel_size:] - prefix[..., :-kernel_size]) / kernel_size


def _compute_importance(
    query_buffer, attn_caches, n_prompt, n_attn_heads, n_kv_heads, pool_kernel=13
):
    """Compute per-token importance from captured queries and cached keys.

    Aggregation (SpecPrefill paper):
      1. softmax(Q @ K^T / sqrt(d)) per head, per layer, per lookahead token
      2. avg_pool1d smoothing
      3. max across (layers x heads)
      4. mean across lookahead tokens
    """
    heads_per_group = n_attn_heads // n_kv_heads
    all_scores = []

    for layer_i, captures in enumerate(query_buffer):
        if not captures:
            continue
        cache = attn_caches[layer_i]
        prompt_keys = cache.keys[..., :n_prompt, :]
        # Skip windowed/rotating caches that don't span the full prompt
        if prompt_keys.shape[-2] < n_prompt:
            continue
        head_dim = prompt_keys.shape[-1]
        q_stack = mx.concatenate(captures, axis=2)
        if heads_per_group > 1:
            expanded_keys = mx.repeat(prompt_keys, heads_per_group, axis=1)
        else:
            expanded_keys = prompt_keys
        scale = head_dim**-0.5
        scores = (q_stack @ expanded_keys.transpose(0, 1, 3, 2)) * scale
        weights = mx.softmax(scores.astype(mx.float32), axis=-1)
        all_scores.append(weights.squeeze(0))

    if not all_scores:
        raise RuntimeError("No attention scores captured — check model/patching")

    combined = mx.concatenate(all_scores, axis=0)
    if pool_kernel and pool_kernel > 1:
        combined = _avg_pool1d(combined, pool_kernel)
    max_scores = mx.max(combined, axis=0)
    importance = mx.mean(max_scores, axis=0)
    return importance


# ===========================================================================
# Public API — Token scoring
# ===========================================================================


def score_tokens(
    model,
    tokens,
    n_lookahead: int = 8,
    pool_kernel: int = 13,
    temp: float = 0.6,
    top_p: float = 0.95,
    prefill_step_size: int = 2048,
    query_extractor: Optional[Callable] = None,
    existing_cache: Optional[List[Any]] = None,
) -> Tuple[mx.array, Any]:
    """Score token importance using attention patterns on a draft model.

    Pipeline:
      1. Prefill draft with all tokens (or suffix if existing_cache provided)
      2. N lookahead decode steps, capturing query vectors
      3. Importance = Q_lookahead @ K_prompt^T, aggregated across heads/layers

    Args:
        model: Draft model (small, fast — e.g. 4B)
        tokens: list or mx.array of token IDs
        n_lookahead: decode steps for query capture (default 8)
        pool_kernel: smoothing kernel size (default 13, 0=disable)
        temp: sampling temperature for lookahead
        top_p: top-p for lookahead
        prefill_step_size: chunk size for draft prefill
        query_extractor: custom function(attn, x, cache) -> queries
        existing_cache: pre-populated cache from paged cache restore.
            If provided, only the suffix beyond cached tokens is prefilled.

    Returns:
        (importance, cache) — importance scores (M,) and the draft cache
        for storage in paged cache.
    """
    from mlx_lm.models.cache import make_prompt_cache

    if isinstance(tokens, mx.array):
        tokens = tokens.tolist()
    n_prompt = len(tokens)

    # Model topology
    attn_layers = _find_attention_layers(model)
    n_attn_layers = len(attn_layers)
    attn_obj = _get_attn_module(attn_layers[0][1])
    n_attn_heads = getattr(
        attn_obj,
        "num_attention_heads",
        getattr(attn_obj, "n_heads", getattr(attn_obj, "num_heads", None)),
    )
    n_kv_heads = getattr(
        attn_obj, "num_key_value_heads", getattr(attn_obj, "n_kv_heads", None)
    )

    if query_extractor is None:
        query_extractor = _detect_query_extractor(attn_obj)

    # Phase 1: Prefill (full or suffix-only if cache provided)
    if existing_cache is not None:
        cache = existing_cache
        cached_len = cache[0].offset if hasattr(cache[0], "offset") else 0
        suffix = tokens[cached_len:]
        if suffix:
            logits = _prefill_draft(model, suffix, cache, step_size=prefill_step_size)
        else:
            # Exact cache hit — run last token to get logits
            logits = model(mx.array([tokens[-1]])[None], cache=cache)
            mx.eval(logits)
    else:
        cache = make_prompt_cache(model)
        logits = _prefill_draft(model, tokens, cache, step_size=prefill_step_size)

    # Record cache offset before lookahead so we can trim afterwards.
    # Lookahead decode appends n_lookahead+1 tokens to the cache which
    # must NOT be persisted when the caller stores the cache to SSD.
    pre_lookahead_offset = cache[0].offset if hasattr(cache[0], "offset") else n_prompt

    # Phase 2: Lookahead decode with query capture
    query_buffer = [[] for _ in range(n_attn_layers)]
    patches, attn_indices = _patch_attention_for_capture(
        model, query_buffer, query_extractor
    )
    try:
        _lookahead_decode(model, logits, cache, n_lookahead, temp=temp, top_p=top_p)
        mx.eval(query_buffer)
    finally:
        _unpatch_attention_capture(model, patches)

    # Phase 3: Compute importance
    layer_to_cache = _build_layer_to_cache_map(model)
    attn_caches = [cache[layer_to_cache[i]] for i in attn_indices]
    importance = _compute_importance(
        query_buffer,
        attn_caches,
        n_prompt,
        n_attn_heads,
        n_kv_heads,
        pool_kernel=pool_kernel if pool_kernel > 0 else None,
    )
    mx.eval(importance)

    # Trim lookahead tokens from cache before returning.
    # KVCache stores keys/values as contiguous tensors; slicing back
    # to pre_lookahead_offset removes the lookahead-generated entries.
    for c in cache:
        if hasattr(c, "offset") and c.offset > pre_lookahead_offset:
            trim = c.offset - pre_lookahead_offset
            if hasattr(c, "keys") and c.keys is not None:
                c.keys = c.keys[..., :pre_lookahead_offset, :]
                c.values = c.values[..., :pre_lookahead_offset, :]
            c.offset = pre_lookahead_offset

    del logits, query_buffer, attn_caches
    mx.clear_cache()

    return importance, cache


def select_chunks(
    importance: mx.array, keep_pct: float = 0.3, chunk_size: int = 32
) -> mx.array:
    """Select top-K% token chunks by average importance.

    Args:
        importance: (M,) per-token importance scores
        keep_pct: fraction of chunks to keep (default 0.3)
        chunk_size: tokens per chunk (default 32)

    Returns:
        sorted mx.array of kept token indices
    """
    M = importance.shape[0]
    if keep_pct >= 1.0:
        return mx.arange(M)

    n_chunks = math.ceil(M / chunk_size)
    keep_n = max(1, math.ceil(n_chunks * keep_pct))

    chunk_scores = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, M)
        chunk_scores.append(mx.mean(importance[start:end]).item())

    top_chunks = sorted(range(n_chunks), key=lambda i: chunk_scores[i], reverse=True)[
        :keep_n
    ]
    top_chunks.sort()

    indices = []
    for ci in top_chunks:
        start = ci * chunk_size
        end = min(start + chunk_size, M)
        indices.extend(range(start, end))

    return mx.array(indices)


# ===========================================================================
# Manual RoPE at arbitrary positions
# ===========================================================================


def manual_rope(x, positions, dims, base=10000.0, scale=1.0):
    """Apply RoPE at arbitrary (non-contiguous) positions.

    Args:
        x: (B, n_heads, L, head_dim)
        positions: (L,) position indices
        dims: number of dimensions to rotate
        base: RoPE base frequency
        scale: position scale divisor
    """
    half = dims // 2
    inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    scaled_pos = positions.astype(mx.float32) / scale
    angles = scaled_pos[:, None] * inv_freq[None, :]
    cos_a = mx.cos(angles)[None, None, :, :]
    sin_a = mx.sin(angles)[None, None, :, :]
    x_rot, x_pass = x[..., :dims], x[..., dims:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate(
        [x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1
    )
    return mx.concatenate([rotated, x_pass], axis=-1)


def manual_rope_with_freqs(x, positions, dims, freqs, pre_scale=1.0):
    """Apply RoPE at arbitrary positions using pre-computed frequencies.

    For custom RoPE variants (Llama3, Yarn, SuScaled) that store _freqs.
    """
    half = dims // 2
    inv_freq = (1.0 / freqs).astype(mx.float32)
    angles = positions[:, None].astype(mx.float32) * inv_freq[None, :]
    cos_a = mx.cos(angles)[None, None, :, :]
    sin_a = mx.sin(angles)[None, None, :, :]
    x_rot, x_pass = x[..., :dims], x[..., dims:]
    if pre_scale != 1.0:
        x_rot = pre_scale * x_rot
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate(
        [x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1
    )
    return mx.concatenate([rotated, x_pass], axis=-1)


# ---------------------------------------------------------------------------
# RoPE wrappers for sparse prefill / decode
# ---------------------------------------------------------------------------


def _scalar_offset(offset) -> int:
    """Coerce scalar RoPE offset to int (Gemma4 wraps cache.offset in mx.array)."""
    if isinstance(offset, mx.array):
        return int(offset.item())
    return int(offset)


class _PositionMappedRoPE:
    """Applies RoPE at non-contiguous positions during sparse prefill.

    Maps cache offset to position array index:
        positions = all_positions[(offset - cache_start) : (offset - cache_start) + L]
    """

    def __init__(self, original_rope, all_positions, cache_start=0):
        self._original = original_rope
        self._all_positions = all_positions
        self._cache_start = _scalar_offset(cache_start)
        self._has_custom_freqs = hasattr(original_rope, "_freqs")

        if self._has_custom_freqs:
            self._freqs = original_rope._freqs
            self._dims = _get_dims(original_rope)
            self._pre_scale = _get_pre_scale(original_rope)
        else:
            self._dims = original_rope.dims
            self._base = original_rope.base
            self._scale = original_rope.scale

    def __call__(self, x, offset=0):
        L = x.shape[2]
        idx = _scalar_offset(offset) - self._cache_start
        positions = self._all_positions[idx : idx + L]
        if self._has_custom_freqs:
            return manual_rope_with_freqs(
                x, positions, self._dims, self._freqs, pre_scale=self._pre_scale
            )
        return manual_rope(x, positions, self._dims, base=self._base, scale=self._scale)


class _OffsetAdjustedRoPE:
    """Adds constant offset to RoPE positions for decode after sparse prefill.

    After sparse prefill of N tokens from M total:
      cache.offset = N + i, desired position = M + i
      adjustment = M - N
    """

    def __init__(self, original_rope, adjustment):
        self._original = original_rope
        self._adjustment = adjustment

    def __call__(self, x, offset=0):
        return self._original(x, offset=offset + self._adjustment)


def _get_dims(rope_module):
    """Extract rotary dimensions from any RoPE variant."""
    for attr in ("_dims", "dim", "dims"):
        if hasattr(rope_module, attr):
            return getattr(rope_module, attr)
    raise ValueError(f"Cannot determine dims from {type(rope_module)}")


def _get_pre_scale(rope_module):
    """Extract pre-scale factor from custom RoPE variants."""
    if hasattr(rope_module, "mscale"):
        return rope_module.mscale
    if hasattr(rope_module, "_scale") and hasattr(rope_module, "dim"):
        return rope_module._scale
    return 1.0


# ===========================================================================
# Public API — Sparse prefill
# ===========================================================================


def sparse_prefill(
    model,
    tokens,
    selected_indices,
    cache,
    step_size: int = 2048,
    position_offset: int = 0,
) -> mx.array:
    """Prefill model cache with selected tokens at their original positions.

    Runs the model on only selected tokens while preserving positional
    encoding via manual RoPE. After this call, cache contains KV entries
    with correct RoPE positions and attention layers have _OffsetAdjustedRoPE
    installed for correct decode positioning.

    Args:
        model: Target model with .layers property
        tokens: (M,) all prompt token IDs
        selected_indices: (N,) sorted indices into tokens to keep
        cache: list of KVCache from make_prompt_cache()
        step_size: chunk size for processing
        position_offset: added to positions for RoPE (e.g. system prompt cache)

    Returns:
        logits from the last selected token

    Side effects:
        - Populates cache with KV for selected tokens
        - Installs _OffsetAdjustedRoPE on attention layers
        - Call cleanup_rope(model) after generation to restore
    """
    if not isinstance(tokens, mx.array):
        tokens = mx.array(tokens)
    if not isinstance(selected_indices, mx.array):
        selected_indices = mx.array(selected_indices)

    M = tokens.shape[0]

    # Ensure tail tokens for RotatingKVCache (sliding window) layers
    max_rotating_size = 0
    for c in cache:
        if type(c).__name__ == "RotatingKVCache":
            max_rotating_size = max(max_rotating_size, getattr(c, "max_size", 0))
    if max_rotating_size > 0:
        tail_start = max(0, M - max_rotating_size)
        tail_indices = set(range(tail_start, M))
        existing = set(selected_indices.tolist())
        merged = sorted(existing | tail_indices)
        selected_indices = mx.array(merged)

    # Build positions with offset
    selected_positions = selected_indices.astype(mx.int32) + position_offset
    selected_tokens = tokens[selected_indices]
    N = selected_tokens.shape[0]

    # Detect initial cache offset (non-zero when system KV is restored)
    attn_layers = _find_attention_layers(model)
    layer_to_cache = _build_layer_to_cache_map(model)
    first_attn_layer_idx = attn_layers[0][0]
    first_attn_cache_idx = layer_to_cache[first_attn_layer_idx]
    cache_start = (
        cache[first_attn_cache_idx].offset
        if hasattr(cache[first_attn_cache_idx], "offset")
        else 0
    )

    # Check if model has RoPE (Nemotron-H doesn't)
    first_attn = _get_attn_module(attn_layers[0][1])
    has_rope = hasattr(first_attn, "rope")

    # Patch RoPE for position-mapped prefill
    original_ropes = {}
    if has_rope:
        for layer_idx, layer in attn_layers:
            attn = _get_attn_module(layer)
            original_ropes[layer_idx] = attn.rope
            attn.rope = _PositionMappedRoPE(
                attn.rope, selected_positions, cache_start=cache_start
            )

    try:
        prompt = selected_tokens
        n = int(N)
        processed = 0

        while n - processed > 1:
            chunk = min(step_size, n - processed - 1)
            model(prompt[processed : processed + chunk][None], cache=cache)
            mx.eval([c.state for c in cache])
            processed += chunk
            mx.clear_cache()

        # Last token -> logits
        logits = model(prompt[processed:][None], cache=cache)
        mx.eval(logits)

    finally:
        # Replace position-mapped RoPE with offset-adjusted RoPE for decode
        if has_rope:
            total_prompt_len = position_offset + M
            final_cache_offset = cache_start + N
            adjustment = int(total_prompt_len) - int(final_cache_offset)
            for layer_idx, layer in attn_layers:
                attn = _get_attn_module(layer)
                original = original_ropes[layer_idx]
                if adjustment > 0:
                    attn.rope = _OffsetAdjustedRoPE(original, adjustment)
                else:
                    attn.rope = original

    return logits


def cleanup_rope(model):
    """Restore original RoPE on all attention layers.

    Call after generation to remove _OffsetAdjustedRoPE wrappers.
    No-op for architectures without RoPE (e.g. Nemotron-H).
    """
    for _, layer in _find_attention_layers(model):
        attn = _get_attn_module(layer)
        if attn is None or not hasattr(attn, "rope"):
            continue
        rope = attn.rope
        if isinstance(rope, (_OffsetAdjustedRoPE, _PositionMappedRoPE)):
            attn.rope = rope._original


# ===========================================================================
# Keep rate presets
# ===========================================================================

KEEP_RATE_PRESETS = {
    0.10: "Aggressive (~5-7x, some quality loss)",
    0.20: "Balanced (~3x, recommended)",
    0.25: "Conservative+ (~2.5x)",
    0.30: "Conservative (~2.2x)",
    0.40: "Mild (~1.8x)",
    0.50: "Minimal (~1.5x)",
}

DEFAULT_KEEP_RATE = 0.20
DEFAULT_THRESHOLD = 8192
DEFAULT_MAX_TOKENS = 65536
