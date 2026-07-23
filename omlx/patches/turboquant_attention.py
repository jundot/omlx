# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to support TurboQuantKVCache.

When TurboQuantKVCache is detected, routes attention to:
  - Decode (L=1): cache.decode_attention() — Metal kernel, no dequant
  - Decode-shaped multi-row (1 < L <= 15, causal; MTP verify): a fused
    2-pass kernel that unpacks each KV token once and scores all L rows
    against it (MSE codecs, issue #2215); outside its envelope the L rows
    are folded into the GQA repeat dimension so the codecs' decode kernels
    apply, with the causal tail mask injected between key scoring and the
    value weighted sum — one lazy pass over the KV, no dequantize
  - Prefill (L>1): cache.prefill_attention() fast path, fallback to
    dequantize + mx.fast.scaled_dot_product_attention
"""

import logging
from functools import cache
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_LONG_PREFILL_QUANTIZED_THRESHOLD = 8192
_LONG_PREFILL_QUERY_BLOCK_SIZE = 256
_LONG_PREFILL_KEY_CHUNK_SIZE = 16384
# MTP verify is a decode-shaped multi-row call (q_len = 1 + draft depth <= 9).
# Above this floor a multi-row call is genuine (chunked) prefill.
_DECODE_MULTIROW_MAX_Q_LEN = 15
# The repeat kernels unroll per-repeat register arrays, so folding is only a
# win while n_repeats * q_len stays under the register-pressure knee
# (measured: 24 fine, 30+ loses to single-chunk quantized_attention).
_MAX_FOLDED_REPEATS = 24
# Softmax-denominator floor, matching turboquant's quantized_attention.
_STATS_EPS = 1e-6
# Fused multi-row verify kernel envelope. Register arrays scale with QRows
# (q + o accumulators per row), so cap the rows; the adaptive MTP controller
# tops out at depth 3 (L=4). Below the token floor the fold path is already
# sub-0.2ms and the 2-pass block split has too few tokens per block.
_FUSED_MULTIROW_MAX_Q_ROWS = 4
_FUSED_MULTIROW_MIN_TOKENS = 2048


@cache
def _fused_mse_multirow_2pass1_kernel(key_bits: int, val_bits: int, dim: int):
    """Pass 1 of the fused multi-row MSE verify attention.

    Derived from turboquant's ``_fused_mse_decode_2pass_1_kernel`` with one
    structural change: each simdgroup unpacks a token's K/V codebook entries
    once and reuses them across all QRows query rows (per-row online softmax
    stats, causal tail applied inline). The upstream decode kernels re-unpack
    the KV per query row, so MTP verify paid the unpack ALU L times over
    (issue #2215).
    """
    from mlx_vlm import turboquant as _tq

    if not _tq._metal_available() or key_bits <= 0 or val_bits <= 0:
        return None
    if dim < 32 or dim % 32 != 0:
        return None

    elems_per_lane = dim // 32
    k_misaligned = (elems_per_lane * key_bits) % 8 != 0
    v_misaligned = (elems_per_lane * val_bits) % 8 != 0
    k_exprs = _tq._gen_unrolled_extract(
        key_bits, elems_per_lane, "key_codebook", "k_bit_off" if k_misaligned else ""
    )
    v_exprs = _tq._gen_unrolled_extract(
        val_bits, elems_per_lane, "val_codebook", "v_bit_off" if v_misaligned else ""
    )
    v_exprs = [e.replace("kb[", "vb[") for e in v_exprs]
    k_lines = "\n            ".join(
        f"k_el[{i}] = {expr};" for i, expr in enumerate(k_exprs)
    )
    v_lines = "\n            ".join(
        f"v_el[{i}] = {expr};" for i, expr in enumerate(v_exprs)
    )

    source = f"""
        constexpr int BD = 32;
        constexpr int qk_per_thread = Dim / BD;
        constexpr int v_per_thread = Dim / BD;
        typedef float U;

        // Thread identity — matches turboquant's mse_sdpa_2pass_1 layout
        auto kv_head_idx = threadgroup_position_in_grid.x;
        auto batch_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto simd_lid = thread_index_in_simdgroup;
        auto gqa_idx = thread_position_in_threadgroup.y;

        auto token_count = key_norms_shape[2];
        auto kv_heads = key_norms_shape[1];
        auto bh = batch_idx * kv_heads + kv_head_idx;
        auto bqh = batch_idx * kv_heads * RepeatCount
            + kv_head_idx * RepeatCount + gqa_idx;

        auto k_nm = key_norms + bh * token_count;
        auto k_pk = key_packed + bh * token_count * KPackedWidth;
        auto v_nm = val_norms + bh * token_count;
        auto v_pk = val_packed + bh * token_count * VPackedWidth;

        // All QRows pre-rotated queries for this (kv_head, repeat) pair
        thread U q[QRows][qk_per_thread];
        for (int r = 0; r < QRows; r++) {{
            auto qr = queries + (bqh * QRows + r) * Dim
                + simd_lid * qk_per_thread;
            for (int i = 0; i < qk_per_thread; i++)
                q[r][i] = static_cast<U>(qr[i]);
        }}

        thread U o[QRows][v_per_thread] = {{}};
        U max_score[QRows];
        U sum_exp_score[QRows];
        for (int r = 0; r < QRows; r++) {{
            max_score[r] = -INFINITY;
            sum_exp_score[r] = 0;
        }}

        // Byte/bit offset for this lane's first element
        int k_bit_start = simd_lid * qk_per_thread * {key_bits};
        int v_bit_start = simd_lid * v_per_thread * {val_bits};
        int k_byte_base = k_bit_start >> 3;
        int v_byte_base = v_bit_start >> 3;
        {"int k_bit_off = k_bit_start & 7;" if k_misaligned else ""}
        {"int v_bit_off = v_bit_start & 7;" if v_misaligned else ""}

        // KV loop: unpack each token once, score all QRows rows against it
        for (int t = block_idx; t < (int)token_count; t += Blocks) {{
            U kn = static_cast<U>(k_nm[t]);
            auto kb = (const device uint8_t*)(k_pk + t * KPackedWidth)
                + k_byte_base;
            U k_el[qk_per_thread];
            {k_lines}

            auto vb = (const device uint8_t*)(v_pk + t * VPackedWidth)
                + v_byte_base;
            U vn = static_cast<U>(v_nm[t]);
            U v_el[v_per_thread];
            {v_lines}

            // Row r sits at global position token_count - QRows + r; token
            // t is invisible to rows r < t - (token_count - QRows).
            int first_row = t - (int)token_count + QRows;
            for (int r = 0; r < QRows; r++) {{
                U dot = 0;
                for (int i = 0; i < qk_per_thread; i++)
                    dot += q[r][i] * k_el[i];
                U score = simd_sum(dot) * kn;
                if (r >= first_row) {{
                    U new_max = max(max_score[r], score);
                    U factor = fast::exp(max_score[r] - new_max);
                    U exp_score = fast::exp(score - new_max);
                    max_score[r] = new_max;
                    sum_exp_score[r] = sum_exp_score[r] * factor + exp_score;
                    for (int i = 0; i < v_per_thread; i++)
                        o[r][i] = o[r][i] * factor + exp_score * v_el[i] * vn;
                }}
            }}
        }}

        // Write per-row partial results for this block
        for (int r = 0; r < QRows; r++) {{
            auto row_out = bqh * QRows + r;
            if (simd_lid == 0) {{
                out_sums[row_out * Blocks + block_idx] = sum_exp_score[r];
                out_maxs[row_out * Blocks + block_idx] = max_score[r];
            }}
            for (int i = 0; i < v_per_thread; i++)
                out_acc[(row_out * Blocks + block_idx) * Dim
                    + simd_lid * v_per_thread + i] = static_cast<U>(o[r][i]);
        }}
    """

    return mx.fast.metal_kernel(
        name=f"omlx_tq_mse_multirow_2pass1_k{key_bits}_v{val_bits}_d{dim}",
        input_names=[
            "queries",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
        ],
        output_names=["out_acc", "out_sums", "out_maxs"],
        source=source,
    )


def _fused_multirow_mse_attention(
    real_cache, queries, keys_state, values_state, scale, total
):
    """Run MTP verify attention through the fused multi-row kernel.

    Returns None when the states/codecs are outside the kernel envelope
    (non-MSE codecs, fractional bits, mismatched dims); the caller falls
    back to the fold / one-shot paths.
    """
    from mlx_vlm import turboquant as _tq

    key_codec = getattr(real_cache, "key_codec", None)
    value_codec = getattr(real_cache, "value_codec", None)
    if not (
        isinstance(key_codec, _tq._TurboQuantMSECodec)
        and isinstance(value_codec, _tq._TurboQuantMSECodec)
    ):
        return None
    if not (
        isinstance(keys_state, _tq.TurboQuantMSEState)
        and isinstance(values_state, _tq.TurboQuantMSEState)
    ):
        return None
    if key_codec.bits != int(key_codec.bits) or value_codec.bits != int(
        value_codec.bits
    ):
        return None

    B, n_q_heads, L, D = queries.shape
    if key_codec.dim != D or value_codec.dim != D:
        return None
    if keys_state.norms.shape[0] != B:
        return None
    n_kv_heads = keys_state.norms.shape[1]
    n_repeats = n_q_heads // n_kv_heads

    pass1 = _fused_mse_multirow_2pass1_kernel(
        int(key_codec.bits), int(value_codec.bits), D
    )
    pass2 = _tq._fused_mse_decode_2pass_2_kernel()
    if pass1 is None or pass2 is None:
        return None

    grouped = (queries * scale).reshape(B, n_kv_heads, n_repeats, L, D)
    q_rot = key_codec.prepare_queries(grouped)
    q_rot_flat = q_rot.reshape(B * n_kv_heads * n_repeats * L, D)

    # Same block split table as turboquant's 2-pass decode dispatch.
    if total <= 8192:
        num_blocks = 64
    elif total <= 32768:
        num_blocks = 128
    elif total <= 65536:
        num_blocks = 256
    else:
        num_blocks = 512

    n_rows = B * n_q_heads * L
    out_acc, out_sums, out_maxs = pass1(
        inputs=[
            q_rot_flat,
            keys_state.norms,
            keys_state.indices,
            key_codec.codebook,
            values_state.norms,
            values_state.indices,
            value_codec.codebook,
        ],
        template=[
            ("Dim", D),
            ("RepeatCount", n_repeats),
            ("QRows", L),
            ("Blocks", num_blocks),
            ("KPackedWidth", keys_state.indices.shape[-1]),
            ("VPackedWidth", values_state.indices.shape[-1]),
        ],
        grid=(n_kv_heads * 32, B * n_repeats, num_blocks),
        threadgroup=(32, n_repeats, 1),
        output_shapes=[
            (n_rows * num_blocks, D),
            (n_rows * num_blocks,),
            (n_rows * num_blocks,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = pass2(
        inputs=[out_acc, out_sums, out_maxs],
        template=[("Dim", D), ("Blocks", num_blocks)],
        grid=(n_rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(n_rows, D)],
        output_dtypes=[mx.float32],
    )[0]

    out_rotated = out.reshape(B, n_kv_heads, n_repeats, L, D)
    output = value_codec._rotate_inverse(out_rotated)
    return output.reshape(B, n_q_heads, L, D).astype(queries.dtype)


def _decode_multirow_quantized_attention(real_cache, queries, keys, values, scale):
    """Wider verify rows: one-shot quantized_attention over the whole KV.

    A single query block and a single key chunk turn quantized_attention's
    chunked online softmax into one pass — its einsum path amortizes the
    key unpack across rows, staying flat in q_len where the folded decode
    kernels hit register spill.
    """
    if not hasattr(real_cache, "quantized_attention"):
        return None
    old_query_block_size = getattr(real_cache, "prefill_query_block_size", None)
    old_key_chunk_size = getattr(real_cache, "prefill_key_chunk_size", None)
    try:
        real_cache.prefill_query_block_size = queries.shape[-2]
        real_cache.prefill_key_chunk_size = real_cache.decode_key_chunk_size
        return real_cache.quantized_attention(
            queries,
            keys_state=keys,
            values_state=values,
            scale=scale,
            mask="causal",
        )
    finally:
        if old_query_block_size is not None:
            real_cache.prefill_query_block_size = old_query_block_size
        if old_key_chunk_size is not None:
            real_cache.prefill_key_chunk_size = old_key_chunk_size


def _decode_multirow_attention(real_cache, queries, keys, values, scale):
    """Causal multi-row attention over TurboQuant states in one lazy pass.

    MTP verify would otherwise fall into the prefill fallbacks, which
    re-dequantize or chunk-scan the whole cache with per-chunk eval syncs
    on every verify cycle (issue #2127 class). MSE-codec states take the
    fused multi-row kernel (one KV unpack shared across the L rows, issue
    #2215); other codecs fold the L rows into the repeat dimension so the
    L==1 decode kernels stay applicable (repeat count is a kernel template
    parameter), with the causal tail mask applied on the raw scores before
    the value weighted sum. Returns None when the states don't fit; the
    caller falls back to the generic paths.
    """
    from mlx_vlm.turboquant import TurboQuantSplitState

    from ..turboquant_kv import _state_length

    keys_state = real_cache._unwrap(keys)
    values_state = real_cache._unwrap(values)
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = (
        keys_state.low.norms.shape[1]
        if isinstance(keys_state, TurboQuantSplitState)
        else keys_state.norms.shape[1]
    )
    n_repeats = n_q_heads // n_kv_heads
    total = _state_length(keys_state)
    if total < L:
        return None
    if L <= _FUSED_MULTIROW_MAX_Q_ROWS and total > _FUSED_MULTIROW_MIN_TOKENS:
        try:
            result = _fused_multirow_mse_attention(
                real_cache, queries, keys_state, values_state, scale, total
            )
        except Exception:
            logger.debug(
                "TurboQuant fused multi-row kernel failed; using fold path",
                exc_info=True,
            )
            result = None
        if result is not None:
            return result
    if n_repeats * L > _MAX_FOLDED_REPEATS:
        return _decode_multirow_quantized_attention(
            real_cache, queries, keys, values, scale
        )

    folded = (queries * scale).reshape(B, n_kv_heads, n_repeats * L, 1, D)
    prepared = real_cache.key_codec.prepare_queries(folded)
    scores = real_cache.key_codec.score_prepared(prepared, keys_state)

    # (B, H, R*L, 1, T): fold index r*L + i is the row at global position
    # total - L + i; mask the keys after it.
    scores = scores.reshape(B, n_kv_heads, n_repeats, L, total)
    q_pos = mx.arange(total - L, total)
    causal = mx.arange(total)[None, :] <= q_pos[:, None]
    scores = mx.where(causal, scores, mx.finfo(scores.dtype).min)
    scores = scores.reshape(B, n_kv_heads, n_repeats * L, 1, total)

    out, denom, _ = real_cache.value_codec.weighted_sum_stats_from_scores(
        scores, values_state
    )
    out = out / mx.maximum(denom[..., None], _STATS_EPS)
    out = out.reshape(B, n_q_heads, L, real_cache.value_codec.dim)
    return out.astype(queries.dtype)


def _patch_update_eval_policy() -> None:
    """Skip the per-layer eval for decode-shaped multi-row cache appends.

    Upstream ``update_and_fetch`` forces ``mx.eval`` whenever more than one
    token is appended — a graph-bounding measure sized for prefill chunks.
    MTP verify appends 2..9 rows per layer, so that policy serializes every
    layer of every verify cycle (~15 forced syncs/cycle). Raise the eval
    floor to prefill-sized appends; verify rows stay lazy and materialize
    at the cycle's sampling sync like the rest of the forward.
    """
    from mlx_vlm import turboquant as _tq

    cls = _tq.TurboQuantKVCache
    if getattr(cls, "_omlx_multirow_eval_patched", False):
        return

    def update_and_fetch(self, keys, values):
        # Mirror of upstream TurboQuantKVCache.update_and_fetch; the only
        # change is the eval gate (n_new > 1 -> prefill-sized appends).
        self._ensure_codecs(keys, values)

        new_keys, new_values = self._try_fused_kv_quantize(keys, values)
        if new_keys is None:
            new_keys = self.key_codec.quantize(keys)
            new_values = self.value_codec.quantize(values)

        new_end = self.offset + keys.shape[2]
        if self.keys is None:
            self.keys = _tq._allocate_state_like(new_keys, new_end)
            self.values = _tq._allocate_state_like(new_values, new_end)
        else:
            self.keys = _tq._reserve_state_capacity(
                self.keys, self.offset, new_end, self.cache_step
            )
            self.values = _tq._reserve_state_capacity(
                self.values, self.offset, new_end, self.cache_step
            )

        _tq._write_state(self.keys, new_keys, self.offset)
        _tq._write_state(self.values, new_values, self.offset)

        n_heads = keys.shape[1]
        n_new = keys.shape[2]

        self.offset = new_end
        self._cached_state = None
        self._cached_state_offset = -1
        if n_new > _DECODE_MULTIROW_MAX_Q_LEN or (self.offset % 50 == 0):
            mx.eval(self.keys, self.values)
        ks, vs = self.state
        return (
            _tq._QuantizedStateProxy(ks, self.offset, n_heads),
            _tq._QuantizedStateProxy(vs, self.offset, n_heads),
        )

    cls.update_and_fetch = update_and_fetch
    cls._omlx_multirow_eval_patched = True


def _patch_vlm_target_verify_attention() -> None:
    """Make mlx-vlm's qwen3_5 MTP verify attention TurboQuant-safe.

    The upstream verify path slices ``keys[:, :, : prefix + i + 1, :]`` per
    draft row before calling SDPA. With TurboQuant the fetched keys/values
    are packed ``_QuantizedStateProxy`` objects that are not subscriptable,
    so every verify forward crashes (issue #2139). Route TurboQuant caches
    through one causal SDPA call instead — the TurboQuant-patched dispatcher
    handles decode-shaped multi-row natively with identical semantics (row i
    attends the first ``prefix + i + 1`` positions).
    """
    try:
        from mlx_vlm.models.qwen3_5 import language as q35_lang
    except ImportError:
        return
    if getattr(q35_lang, "_omlx_tq_target_verify_patched", False):
        return
    original = getattr(q35_lang, "_target_verify_left_padded_attention", None)
    if original is None:
        return

    def patched(queries, keys, values, *, cache, scale, mask):
        from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache

        from ..turboquant_kv import BatchTurboQuantKVCache

        real_cache = cache
        if hasattr(cache, "_cache") and not isinstance(
            cache, (_TQCache, BatchTurboQuantKVCache)
        ):
            real_cache = cache._cache
        if not isinstance(real_cache, (_TQCache, BatchTurboQuantKVCache)):
            return original(queries, keys, values, cache=cache, scale=scale, mask=mask)

        sdpa = q35_lang.scaled_dot_product_attention
        if queries.shape[0] == 1 and not isinstance(mask, mx.array):
            return sdpa(
                queries, keys, values, cache=cache, scale=scale, mask="causal"
            )
        # Left-padded batches / explicit array masks: dequantize once and
        # replicate the caller's per-row causal slicing on dense arrays.
        dk, dv = real_cache.dequantize(keys_state=keys, values_state=values)
        dk = dk.astype(queries.dtype)
        dv = dv.astype(queries.dtype)
        L = queries.shape[2]
        prefix_len = dk.shape[-2] - L
        return mx.concatenate(
            [
                sdpa(
                    queries[:, :, i : i + 1, :],
                    dk[:, :, : prefix_len + i + 1, :],
                    dv[:, :, : prefix_len + i + 1, :],
                    cache=None,
                    scale=scale,
                    mask=(
                        mask[..., i : i + 1, : prefix_len + i + 1]
                        if isinstance(mask, mx.array) and mask.ndim >= 4
                        else None
                    ),
                )
                for i in range(L)
            ],
            axis=2,
        )

    q35_lang._target_verify_left_padded_attention = patched
    q35_lang._omlx_tq_target_verify_original = original
    q35_lang._omlx_tq_target_verify_patched = True


def apply_turboquant_attention_patch() -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for TurboQuant."""
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    try:
        _patch_update_eval_policy()
    except Exception:
        logger.debug("TurboQuant update eval-policy patch skipped", exc_info=True)

    try:
        _patch_vlm_target_verify_attention()
    except Exception:
        logger.debug(
            "TurboQuant VLM target-verify attention patch skipped", exc_info=True
        )

    original_sdpa = mlx_base.scaled_dot_product_attention

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: Optional[mx.array],
        sinks: Optional[mx.array] = None,
    ) -> mx.array:
        from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache

        from ..turboquant_kv import BatchTurboQuantKVCache, _state_length

        # Detect underlying TQ cache (may be wrapped by proxy objects)
        real_cache = cache
        if hasattr(cache, "_cache") and not isinstance(
            cache, (_TQCache, BatchTurboQuantKVCache)
        ):
            real_cache = cache._cache

        if isinstance(real_cache, (_TQCache, BatchTurboQuantKVCache)):
            if sinks is not None:
                # TurboQuant's quantized kernels do not implement attention
                # sinks. Preserve correctness by falling back to MLX's
                # sink-aware SDPA over dequantized states.
                dequantized_keys, dequantized_values = real_cache.dequantize(
                    keys_state=keys,
                    values_state=values,
                )
                return mx.fast.scaled_dot_product_attention(
                    queries,
                    dequantized_keys.astype(queries.dtype),
                    dequantized_values.astype(queries.dtype),
                    scale=scale,
                    mask=mask,
                    sinks=sinks,
                )
            if queries.shape[-2] == 1:
                # Decode (B=1 and B>1). Continuous-batching decode passes a
                # per-request left-padding array mask; the masked decode_attention
                # path runs the quantized kernels directly (no full-batch
                # dequantize per step). The RHT masked-decode fix landed upstream
                # in mlx-vlm (Blaizzy/mlx-vlm#1244, in the pinned commit).
                return real_cache.decode_attention(
                    queries,
                    keys_state=keys,
                    values_state=values,
                    scale=scale,
                    mask=mask,
                )
            if (
                queries.shape[-2] <= _DECODE_MULTIROW_MAX_Q_LEN
                and isinstance(mask, str)
                and mask == "causal"
            ):
                # Decode-shaped multi-row (MTP verify) — see helper docstring.
                try:
                    result = _decode_multirow_attention(
                        real_cache, queries, keys, values, scale
                    )
                    if result is not None:
                        return result
                except Exception:
                    logger.debug(
                        "TurboQuant multi-row decode attention failed; "
                        "falling back to prefill paths",
                        exc_info=True,
                    )
            # Prefill: try quantized fast path, fallback to dequantize+SDPA
            result = real_cache.prefill_attention(
                queries,
                keys_state=keys,
                values_state=values,
                scale=scale,
                mask=mask,
            )
            if result is not None:
                return result
            keys_state = getattr(keys, "_state", keys)
            try:
                total_tokens = _state_length(keys_state)
            except Exception:
                total_tokens = 0
            if (
                total_tokens > _LONG_PREFILL_QUANTIZED_THRESHOLD
                and hasattr(real_cache, "quantized_attention")
            ):
                old_query_block_size = getattr(
                    real_cache, "prefill_query_block_size", None
                )
                old_key_chunk_size = getattr(
                    real_cache, "prefill_key_chunk_size", None
                )
                try:
                    real_cache.prefill_query_block_size = (
                        _LONG_PREFILL_QUERY_BLOCK_SIZE
                    )
                    real_cache.prefill_key_chunk_size = _LONG_PREFILL_KEY_CHUNK_SIZE
                    return real_cache.quantized_attention(
                        queries,
                        keys_state=keys,
                        values_state=values,
                        scale=scale,
                        mask=mask,
                    )
                except Exception:
                    logger.debug(
                        "TurboQuant quantized prefill attention failed; "
                        "falling back to dequantize+SDPA",
                        exc_info=True,
                    )
                finally:
                    if old_query_block_size is not None:
                        real_cache.prefill_query_block_size = old_query_block_size
                    if old_key_chunk_size is not None:
                        real_cache.prefill_key_chunk_size = old_key_chunk_size
            dequantized_keys, dequantized_values = real_cache.dequantize()
            return mx.fast.scaled_dot_product_attention(
                queries,
                dequantized_keys.astype(queries.dtype),
                dequantized_values.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )

        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Patch the module attribute
    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Also patch any model modules that already imported it locally
    # Covers both mlx_lm (LLM) and mlx_vlm (VLM) model modules
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models.")):
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            func = getattr(mod, "scaled_dot_product_attention")
            if func is original_sdpa or func is not patched_sdpa:
                setattr(mod, "scaled_dot_product_attention", patched_sdpa)

    # Also patch mlx_vlm.models.base if loaded
    try:
        from mlx_vlm.models import base as vlm_base
        if hasattr(vlm_base, "scaled_dot_product_attention"):
            vlm_base.scaled_dot_product_attention = patched_sdpa
    except ImportError:
        pass

    _PATCHED = True
    logger.info("TurboQuant attention patch applied")
    return True
