# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to fix head_dim=256 long-context prefill.

MLX's fused SDPA kernel supports head_dim in {64, 80, 128} only, so head_dim=256
(e.g. Qwen3.6-27B) multi-token prefill falls back to an unfused path that
materializes the full ``[n_q, query_len, kv_len]`` score matrix -> O(L^2) memory,
OOMing / tripping the prefill guard far below the context window. Decode
(query_len == 1) is unaffected (MLX has a fused vector kernel for 256).

This routes head_dim=256 causal prefill through three possible paths, picked
per-call by ``_route_decision``:

- **unfused** (stock MLX fallback): the fast path wherever its
  ``[n_q, query_len, kv_len]`` score matrix fits under live guard headroom.
- **q-split** (``_unfused_qsplit_sdpa``): when the full unfused call doesn't
  fit, split the query axis into sub-tiles (keys/values narrowed to each
  sub-tile's causal end) and run the SAME fast stock kernel per sub-tile —
  smaller transient, still the fast kernel, no accuracy cost.
- **tiled** (``_flash_sdpa256``): a flash-style online-softmax pass in pure
  MLX array ops (tiled over KV; running max/sum/accumulator) that never
  materializes the score matrix -> true O(L) peak, at the cost of one
  ``mx.eval`` per KV tile (real measured overhead, not "on par with the
  fallback" — thousands of small CPU<->GPU syncs at long context). This is
  now the last resort once even a minimum-size q-split slice wouldn't fit,
  or when headroom info is unavailable (memory-safe #2025 default).
  ``register_tiled_prefill_head_dim`` flips the prefill-guard estimator to
  O(L) in lockstep so it prices this route instead of the O(L^2) unfused
  formula (else the guard keeps rejecting requests the tiled route could
  actually serve).

The route is memory-aware (issue #2204, extended for q-split): unfused is
faster everywhere its score matrix fits (on NAX GPUs its big GEMMs run on
the tensor units), so when the Scheduler has registered a headroom provider
the route only narrows to q-split, then finally tiled, as live headroom
shrinks. Without a provider (no Scheduler, ceiling not propagated yet) the
route keeps the memory-safe default: always tiled past the kv_len threshold.
``OMLX_SDPA256_TILED=1`` forces the tiled pass whenever the shape gates
match (pre-#2204 behavior, also skips q-split); ``OMLX_SDPA256_TILED=0``
never engages tiled OR q-split (restores the O(L^2) memory wall —
benchmarking only). ``OMLX_SDPA256_QSPLIT=0`` disables q-split specifically,
falling straight to tiled once unfused doesn't fit (pre-q-split behavior,
for rollback without touching the tiled/unfused override).

Install mechanics mirror turboquant_attention.py (patch the module attr + rebind
already-imported model modules). The route is strictly gated (see _should_route);
everything else passes through to the original SDPA unchanged.
"""

import logging
import os
import weakref

import mlx.core as mx

from omlx.memory_monitor import estimate_unfused_sdpa_call_bytes

logger = logging.getLogger(__name__)

_PATCHED = False

HEAD_DIM = 256
# Engage the tiled kernel only once the context is long enough that the unfused
# fallback's O(L^2) score matrix becomes a memory problem. Below this, the
# fused-GEMM fallback is faster and fits comfortably. Tunable.
_SDPA256_MIN_KV_LEN = 8192
# Decode-shaped multi-row calls (MTP verify: q_len = 1 + draft depth <= 9)
# must not take this route: the per-KV-tile eval sync only amortizes over
# prefill-sized q tiles, and at tiny q_len it costs O(kv_len/tile) sequential
# dispatches per call — 8-22x slower than stock SDPA, collapsing long-context
# MTP throughput (issue #2127). Below this floor the stock path's score
# matrix is at most n_q * 15 * kv_len, which is never a memory problem.
_SDPA256_MIN_Q_LEN = 16
# Tile sizes for the online-softmax kernel (tuned on M2 Max).
_Q_TILE = 512
_KV_TILE = 1024

_NEG_INF = -1e30  # fp32 sentinel for masked logits (exp -> 0)

# Live guard-headroom provider for memory-aware routing (issue #2204).
# Registered by Scheduler.__init__ as a bound method returning the bytes left
# under the adaptive-prefill-throttle target (hard ceiling x headroom safety,
# clamped by the abort cap), or a negative value when no ceiling is active.
# Held as a WeakMethod so a torn-down Scheduler auto-unregisters and the route
# falls back to the memory-safe always-tiled default.
_HEADROOM_PROVIDER: "weakref.WeakMethod | None" = None
# OMLX_SDPA256_TILED override, parsed at apply time: True = always tiled,
# False = never tiled, None = memory-aware auto.
_FORCE_TILED: bool | None = None
# OMLX_SDPA256_QSPLIT override, parsed at apply time: False disables the
# q-split route (falls straight to tiled once unfused doesn't fit, restoring
# pre-q-split behavior for rollback); True/None (default) leaves it enabled.
_QSPLIT_ENABLED: bool = True
# Minimum query rows per q-split sub-call. Below this the per-call overhead
# stops paying for itself and genuinely tight headroom is better served by
# the tiled path's true O(L) floor.
_QSPLIT_MIN_Q = 128
# Route decisions round-trip through here so callers branch on one value.
_ROUTE_UNFUSED = "unfused"
_ROUTE_QSPLIT = "qsplit"
_ROUTE_TILED = "tiled"
# Last route decision, so we log every *transition* (not just the
# first-ever engagement) at INFO — cheap, and lets a live server log show a
# request flapping between routes as guard headroom rises and falls
# mid-prefill (see the docstring on
# omlx.scheduler.Scheduler._sdpa256_unfused_headroom for why that happens).
# The tiled/qsplit pass trades substantial prefill throughput at long
# kv_len for safer memory, and nothing surfaced the route decision before
# (issue #2283 took an A/B repro to diagnose), so this stays at INFO rather
# than DEBUG.
_LAST_ROUTE_DECISION: str | None = None


def _note_route(decision: str, detail) -> None:
    """``detail`` may be a plain string or a zero-arg callable. This runs on
    every head-dim-256 prefill SDPA call (thousands per long-context
    request), and the vast majority hit the decision == last-decision
    no-op below -- callers with a formatted (f-string) detail should pass a
    lambda so the string is only built on an actual transition."""
    global _LAST_ROUTE_DECISION
    if decision == _LAST_ROUTE_DECISION:
        return
    _LAST_ROUTE_DECISION = decision
    if callable(detail):
        detail = detail()
    logger.info(
        "sdpa256: route -> %s (%s). OMLX_SDPA256_TILED=1/0 forces "
        "unfused/tiled; OMLX_SDPA256_QSPLIT=0 disables the q-split route.",
        decision,
        detail,
    )


def _total_qsplit_bytes(
    n_q_heads: int,
    q_len: int,
    kv_len: int,
    causal: bool,
    q_sub: int,
    head_dim: int,
    score_dtype_size: float,
) -> int:
    """Sum of every q-split sub-tile's transient, not just one sub-tile's.

    Causal: each sub-tile's keys/values are narrowed to that sub-tile's own
    causal end (``_unfused_qsplit_sdpa``), so kv width grows sub-tile to
    sub-tile -- every sub-tile is a *different* size, and Metal's buffer
    pool cannot reuse a differently-sized buffer for the next one. They
    accumulate as retained (not just active) memory across the whole loop:
    ``mx.eval`` between sub-tiles bounds what is *live* to one sub-tile,
    but says nothing about what the pool has *retained* from the ones
    before it (confirmed live -- see ``_unfused_qsplit_sdpa``'s docstring).

    Non-causal: every sub-tile is ``q_sub`` rows against the full, constant
    ``kv_len`` (no narrowing) -- same size as every other sub-tile, so the
    pool genuinely reuses one buffer across the loop and only ONE
    sub-tile's transient (not the whole q_len's) is ever retained at a
    time. Summing here would overcharge a case that was never broken.
    """

    if not causal:
        return estimate_unfused_sdpa_call_bytes(
            n_q_heads, min(q_sub, q_len), kv_len, head_dim,
            score_dtype_size=score_dtype_size,
        )
    kv_off = kv_len - q_len
    total = 0
    for qi0 in range(0, q_len, q_sub):
        qi1 = min(qi0 + q_sub, q_len)
        total += estimate_unfused_sdpa_call_bytes(
            n_q_heads,
            qi1 - qi0,
            kv_off + qi1,
            head_dim,
            score_dtype_size=score_dtype_size,
        )
    return total


def _max_q_sub_for_headroom(
    n_q_heads: int,
    q_len: int,
    kv_len: int,
    causal: bool,
    head_dim: int,
    score_dtype_size: float,
    headroom: int,
) -> int:
    """Largest (128-aligned) query-row sub-tile count whose q-split
    transient fits ``headroom`` -- the *total* retained across every
    sub-tile in the split for a causal call (see ``_total_qsplit_bytes``),
    not the naive "one sub-tile at a time" a per-call eval only bounds the
    *active* set to. Non-causal degrades to the original single-transient
    inversion, since every sub-tile there is the same size and genuinely
    reusable.

    A closed-form inversion of the causal sum is a quadratic in ``q_sub``;
    a bounded linear search from a safe starting point (this call's
    single-sub-tile estimate, an upper bound since it ignores accumulation
    and can therefore only be too large) is simpler to keep correct than
    an inverted formula that has to be re-derived by hand every time this
    function's cost model changes.
    """

    if headroom <= 0:
        return 0
    per_row = n_q_heads * (kv_len * score_dtype_size + head_dim * 4)
    if per_row <= 0:
        return 0
    q_sub = min(q_len, int(headroom // per_row))
    q_sub = (q_sub // 128) * 128
    while q_sub >= 128:
        total = _total_qsplit_bytes(
            n_q_heads, q_len, kv_len, causal, q_sub, head_dim, score_dtype_size
        )
        if total <= headroom:
            return q_sub
        q_sub -= 128
    return 0


def set_unfused_headroom_provider(method) -> None:
    """Register a bound method(kv_len, q_len) -> live headroom in bytes
    (negative when no ceiling is active). Lets ``_should_route`` prefer the
    faster unfused fallback whenever its O(L^2) transient fits. ``kv_len``
    and ``q_len`` together let the provider tell a genuine same-shape
    repeat call (safe to credit pooled memory against) from the first call
    at a new, larger kv_len -- or a same-kv_len call with a different
    q_len, e.g. a different chunk of the same request (not safe -- see
    Scheduler._sdpa256_unfused_headroom)."""
    global _HEADROOM_PROVIDER
    _HEADROOM_PROVIDER = weakref.WeakMethod(method)


def _parse_force_tiled_env() -> bool | None:
    value = os.environ.get("OMLX_SDPA256_TILED", "").strip()
    if value == "1":
        return True
    if value == "0":
        return False
    return None


def _parse_qsplit_env() -> bool:
    return os.environ.get("OMLX_SDPA256_QSPLIT", "").strip() != "0"


def _route_decision(
    queries, keys, mask, q_sub_ceiling: int | None = None
) -> tuple[str, int]:
    """Decide unfused / q-split / tiled for a shape-matched prefill call.

    Returns ``(route, q_sub)`` — ``q_sub`` is only meaningful for
    ``_ROUTE_QSPLIT`` (query rows per sub-call), 0 otherwise.

    The stock unfused fallback is faster wherever its transient fits
    (issues #2155 / #2204): take it whole when the full call fits, split
    the query axis into sub-calls that individually fit when the full call
    doesn't (still the fast kernel, just narrower), and fall back to the
    true O(L) tiled pass only once even a minimum-size q-split slice
    wouldn't fit, or when headroom info is unavailable (memory-safe
    #2025 default).

    ``q_sub_ceiling`` is this request's hysteresis floor (set by
    ``_should_route`` once a call has ever needed a smaller transient than
    the full call): caps how large a transient this call may use, not just
    which route label it gets. kv_len only grows within a request, so a
    transient shed earlier reflects real, non-relaxing pressure -- and
    capping only the *route* (skip the "fits" fast path but still let
    q_sub float back up to q_len when headroom looks momentarily generous)
    was verified live to reproduce byte-for-byte the same full-size
    transient as plain unfused, just labeled qsplit, moments before the
    reactive guard aborted a request. ``q_sub_ceiling == 0`` means a
    previous call already needed tiled -- never try qsplit or unfused
    again this request."""
    if _FORCE_TILED is not None:
        if _FORCE_TILED:
            _note_route(_ROUTE_TILED, "forced by OMLX_SDPA256_TILED=1")
            return _ROUTE_TILED, 0
        _note_route(_ROUTE_UNFUSED, "forced by OMLX_SDPA256_TILED=0")
        return _ROUTE_UNFUSED, 0
    try:
        if q_sub_ceiling == 0:
            _note_route(
                _ROUTE_TILED,
                "held at tiled by this request's hysteresis floor",
            )
            return _ROUTE_TILED, 0
        provider = _HEADROOM_PROVIDER() if _HEADROOM_PROVIDER is not None else None
        if provider is None:
            _note_route(
                _ROUTE_TILED,
                "no guard headroom provider registered "
                "(engine without a scheduler, or scheduler gone)",
            )
            return _ROUTE_TILED, 0
        batch, n_q, q_len, _ = queries.shape
        kv_len = keys.shape[-2]
        n_q_heads = batch * n_q
        dtype_size = queries.dtype.size
        causal = mask == "causal"
        headroom = provider(kv_len, q_len)
        if headroom is None or headroom < 0:
            _note_route(
                _ROUTE_TILED,
                "memory ceiling not available (enforcer state not yet "
                "propagated)",
            )
            return _ROUTE_TILED, 0
        transient = estimate_unfused_sdpa_call_bytes(
            n_q_heads, q_len, kv_len, HEAD_DIM, score_dtype_size=dtype_size
        )
        if transient <= headroom and q_sub_ceiling is None:
            _note_route(
                _ROUTE_UNFUSED,
                lambda: f"full call ~{transient / 2**20:.0f}MiB fits "
                f"~{headroom / 2**20:.0f}MiB headroom at kv_len={kv_len}",
            )
            return _ROUTE_UNFUSED, 0
        if _QSPLIT_ENABLED:
            q_sub = _max_q_sub_for_headroom(
                n_q_heads, q_len, kv_len, causal, HEAD_DIM, dtype_size, headroom
            )
            if q_sub_ceiling is not None:
                q_sub = min(q_sub, q_sub_ceiling)
            if q_sub >= _QSPLIT_MIN_Q:
                q_sub = min(q_sub, q_len)
                _note_route(
                    _ROUTE_QSPLIT,
                    lambda: f"q_sub={q_sub} of q_len={q_len} fits "
                    f"~{headroom / 2**20:.0f}MiB headroom at kv_len={kv_len} "
                    f"(full-call transient ~{transient / 2**20:.0f}MiB)",
                )
                return _ROUTE_QSPLIT, q_sub
        _note_route(
            _ROUTE_TILED,
            lambda: f"unfused transient ~{transient / 2**20:.0f}MiB exceeds "
            f"~{headroom / 2**20:.0f}MiB headroom at kv_len={kv_len} even "
            f"at the q-split floor ({_QSPLIT_MIN_Q} rows)",
        )
        return _ROUTE_TILED, 0
    except Exception:
        logger.debug("sdpa256 headroom probe failed", exc_info=True)
        _note_route(_ROUTE_TILED, "guard headroom probe failed")
        return _ROUTE_TILED, 0  # headroom info unavailable -> memory-safe default


def _flash_sdpa256(queries, keys, values, scale, mask):
    """Flash-style online-softmax attention for head_dim=256 prefill.

    queries: [batch, n_q, q_len, head_dim]
    keys/values: [batch, n_kv, k_len, head_dim]   (n_q % n_kv == 0)
    mask: "causal" or None. Returns [batch, n_q, q_len, head_dim] in
    queries.dtype.

    Tiles over Q and KV, keeping a running (max m, sum denom, accumulator acc) per
    query row so the [q x full_kv] score matrix is never materialized. fp32
    accumulators; output cast back to the input dtype. GQA via reshape+broadcast.

    MLX is lazy: without forcing materialization the whole tiled graph would stay
    live until eval (peak dominated by graph buildup, not the O(L) working set),
    so the running carry is eval'd per KV step / per finished Q tile to bound the
    live graph to ~one tile -> true O(L) peak.
    """
    batch, n_q, q_len, head_dim = queries.shape
    _, n_kv, k_len, _ = keys.shape
    group_size = n_q // n_kv
    causal = mask == "causal"

    qr = queries.reshape(batch, n_kv, group_size, q_len, head_dim)
    kr = keys.reshape(batch, n_kv, 1, k_len, head_dim)
    vr = values.reshape(batch, n_kv, 1, k_len, head_dim)

    # MLX 'causal' aligns queries to the END of the key axis: with a cached
    # prefix (k_len > q_len, chunked prefill) local query i is global position
    # i + offset and attends keys 0..(i + offset). offset == 0 for square.
    offset = k_len - q_len

    out_q_tiles = []
    for qi0 in range(0, q_len, _Q_TILE):
        qi1 = min(qi0 + _Q_TILE, q_len)
        qb = qr[:, :, :, qi0:qi1, :].astype(mx.float32)
        qt = qi1 - qi0
        q_pos = mx.arange(qi0 + offset, qi1 + offset).reshape(1, 1, 1, qt, 1)

        m = mx.full((batch, n_kv, group_size, qt, 1), _NEG_INF, dtype=mx.float32)
        denom = mx.zeros((batch, n_kv, group_size, qt, 1), dtype=mx.float32)
        acc = mx.zeros((batch, n_kv, group_size, qt, head_dim), dtype=mx.float32)

        kv_end = min(qi1 + offset, k_len) if causal else k_len
        for kj0 in range(0, kv_end, _KV_TILE):
            kj1 = min(kj0 + _KV_TILE, kv_end)
            kb = kr[:, :, :, kj0:kj1, :].astype(mx.float32)
            vb = vr[:, :, :, kj0:kj1, :].astype(mx.float32)
            kt = kj1 - kj0

            s = (qb @ mx.swapaxes(kb, -1, -2)) * scale
            if causal:
                k_pos = mx.arange(kj0, kj1).reshape(1, 1, 1, 1, kt)
                s = mx.where(k_pos > q_pos, _NEG_INF, s)

            m_tile = mx.max(s, axis=-1, keepdims=True)
            m_new = mx.maximum(m, m_tile)
            p = mx.exp(s - m_new)
            corr = mx.exp(m - m_new)
            denom = denom * corr + mx.sum(p, axis=-1, keepdims=True)
            acc = acc * corr + (p @ vb)
            m = m_new
            mx.eval(m, denom, acc)  # bound the live graph -> O(L) peak

        out_tile = (acc / denom).astype(queries.dtype)
        mx.eval(out_tile)
        out_q_tiles.append(out_tile)

    out = mx.concatenate(out_q_tiles, axis=3)
    return out.reshape(batch, n_q, q_len, head_dim)


def _unfused_qsplit_sdpa(
    queries, keys, values, cache, scale, mask, sinks, original_sdpa, q_sub
):
    """Run the fast stock (unfused) SDPA kernel over query sub-tiles instead
    of the whole chunk, each with keys/values narrowed to that sub-tile's
    causal end -- same kernel as the full-call fast path, just a smaller
    per-call transient so it fits under tighter headroom than a single call
    would.

    Correctness: MLX's ``mask="causal"`` right-aligns queries to the tail of
    the key axis (see ``_flash_sdpa256``'s comment on the same convention).
    For sub-tile [qi0:qi1) with global offset ``kv_off = k_len - q_len``,
    narrowing keys/values to [0:kv_off+qi1) makes MLX infer the offset
    ``(kv_off+qi1) - (qi1-qi0) = kv_off+qi0`` for this call -- exactly the
    sub-tile's true global offset -- so no manual position masking is
    needed. This also does less wasted compute than a single full call:
    each sub-tile's GEMM only covers the KV prefix its own causal window
    can see, not the full kv_len every time.

    Memory: each sub-tile's keys/values window grows with ``qi1`` (a later
    sub-tile sees a wider causal prefix than an earlier one), so without
    forcing evaluation between sub-tiles MLX's laziness lets every sub-call's
    graph -- including its own score-matrix transient -- stay unmaterialized
    and pile up simultaneously, rather than bounding the live set to one
    sub-tile at a time the way ``q_sub``'s sizing assumes. ``mx.eval`` here
    mirrors ``_flash_sdpa256``'s own per-tile eval for the same reason: it
    converts "smaller instantaneous peak" from an aspiration into something
    actually true of the live Metal graph (confirmed necessary, not just
    defensive, by a live run where q-split engaged but didn't prevent the
    memory trip it was sized to avoid).
    """
    q_len = queries.shape[-2]
    causal = mask == "causal"
    kv_off = keys.shape[-2] - q_len if causal else 0
    out_tiles = []
    for qi0 in range(0, q_len, q_sub):
        qi1 = min(qi0 + q_sub, q_len)
        q_slice = queries[..., qi0:qi1, :]
        if causal:
            k_slice = keys[..., : kv_off + qi1, :]
            v_slice = values[..., : kv_off + qi1, :]
        else:
            k_slice, v_slice = keys, values
        out_tile = original_sdpa(q_slice, k_slice, v_slice, cache, scale, mask, sinks)
        mx.eval(out_tile)
        out_tiles.append(out_tile)
    return mx.concatenate(out_tiles, axis=-2)


def _should_route(queries, keys, cache, mask, sinks) -> tuple[str, int]:
    # Never raise: any unexpected input must fall through to the original SDPA,
    # never break a request. Worst case we decline to engage.
    # Shape gates first: this wrapper is installed unconditionally and runs
    # on every SDPA call of every decode step, so the common (decode / MTP
    # verify) case must exit on the q_len check alone (issue #2132).
    try:
        if queries.shape[-2] < _SDPA256_MIN_Q_LEN:  # decode / MTP verify
            return _ROUTE_UNFUSED, 0
        if queries.shape[-1] != HEAD_DIM:
            return _ROUTE_UNFUSED, 0
        if keys.shape[-2] < _SDPA256_MIN_KV_LEN:
            return _ROUTE_UNFUSED, 0
        if sinks is not None:
            return _ROUTE_UNFUSED, 0
        # Quantized KV cache (TurboQuant etc.): keys/values are packed state,
        # not plain [.., kv, hd] arrays. MLX's own dispatcher detects this via
        # hasattr(cache, "bits"); let the quant-aware path handle it.
        if cache is not None and hasattr(cache, "bits"):
            return _ROUTE_UNFUSED, 0
        if not (mask is None or (isinstance(mask, str) and mask == "causal")):
            return _ROUTE_UNFUSED, 0
        n_q = queries.shape[-3]
        n_kv = keys.shape[-3]
        if n_kv <= 0 or n_q % n_kv != 0:
            return _ROUTE_UNFUSED, 0
        # Hysteresis floor: once this request's cache has needed a smaller
        # transient than the full call, never let a later chunk's estimate
        # push the transient back up -- kv_len is monotone within a
        # request, so the pressure that forced the downgrade cannot have
        # relaxed by the next chunk. Ratchets on transient SIZE (q_sub),
        # not just the route label: capping only the label and letting
        # q_sub float back up to q_len when headroom looks momentarily
        # generous was verified live to reproduce the identical full-size
        # transient labeled qsplit instead of unfused. Stashed on ``cache``
        # since that is the one object stable across every chunk/layer of a
        # single request but never shared across requests.
        ceiling = getattr(cache, "_sdpa256_q_sub_ceiling", None)
        route, q_sub = _route_decision(queries, keys, mask, q_sub_ceiling=ceiling)
        if cache is not None:
            try:
                if route == _ROUTE_TILED:
                    cache._sdpa256_q_sub_ceiling = 0
                elif route == _ROUTE_QSPLIT:
                    cache._sdpa256_q_sub_ceiling = (
                        q_sub if ceiling is None else min(ceiling, q_sub)
                    )
            except Exception:
                pass
        return route, q_sub
    except Exception:
        return _ROUTE_UNFUSED, 0


def apply_sdpa256_attention_patch(min_kv_len: int = _SDPA256_MIN_KV_LEN) -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for head_dim=256
    long-context prefill, and register the O(L) cost with the memory monitor."""
    global _PATCHED, _SDPA256_MIN_KV_LEN, _FORCE_TILED, _QSPLIT_ENABLED
    if _PATCHED:
        return False
    _SDPA256_MIN_KV_LEN = min_kv_len
    _FORCE_TILED = _parse_force_tiled_env()
    _QSPLIT_ENABLED = _parse_qsplit_env()

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    original_sdpa = mlx_base.scaled_dot_product_attention

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: mx.array | None,
        sinks: mx.array | None = None,
    ) -> mx.array:
        route, q_sub = _should_route(queries, keys, cache, mask, sinks)
        try:
            if route == _ROUTE_QSPLIT:
                return _unfused_qsplit_sdpa(
                    queries, keys, values, cache, scale, mask, sinks,
                    original_sdpa, q_sub,
                )
            if route == _ROUTE_TILED:
                return _flash_sdpa256(queries, keys, values, scale, mask)
        except Exception:
            logger.warning(
                "sdpa256 prefill kernel failed; falling back to MLX SDPA",
                exc_info=True,
            )
        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Rebind already-imported model modules that did
    # `from .base import scaled_dot_product_attention` at import time. Only
    # rebind modules whose attribute IS the base function we wrapped — a model
    # that defined its own SDPA keeps it untouched (don't silently redirect a
    # model we never intended to patch).
    import sys

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("mlx_lm.models."):
            continue
        if getattr(mod, "scaled_dot_product_attention", None) is original_sdpa:
            mod.scaled_dot_product_attention = patched_sdpa

    # mlx-vlm carries its own base SDPA (a distinct function, TurboQuant-aware
    # cache handling included), and model modules like qwen3_5.language copy
    # the reference at import time. It needs its own capture + wrapper +
    # submodule rebind, mirroring qwen35_fa256_attention: checking mlx-vlm
    # modules against the mlx-lm original can never match, which left the VLM
    # engine on the unfused O(L^2) path and — because this patch installs
    # first — polluted the fa256 patch's "original" capture so its rebind
    # missed the VLM submodules too.
    try:
        from mlx_vlm.models import base as vlm_base
    except ImportError:
        vlm_base = None

    if vlm_base is not None:
        original_vlm_sdpa = getattr(vlm_base, "scaled_dot_product_attention", None)
        if original_vlm_sdpa is not None:

            def patched_vlm_sdpa(
                queries,
                keys,
                values,
                cache,
                scale: float,
                mask=None,
                sinks=None,
            ) -> mx.array:
                route, q_sub = _should_route(queries, keys, cache, mask, sinks)
                try:
                    if route == _ROUTE_QSPLIT:
                        return _unfused_qsplit_sdpa(
                            queries, keys, values, cache, scale, mask, sinks,
                            original_vlm_sdpa, q_sub,
                        )
                    if route == _ROUTE_TILED:
                        return _flash_sdpa256(queries, keys, values, scale, mask)
                except Exception:
                    logger.warning(
                        "sdpa256 prefill kernel failed; falling back to "
                        "MLX SDPA",
                        exc_info=True,
                    )
                return original_vlm_sdpa(
                    queries, keys, values, cache, scale, mask, sinks
                )

            vlm_base.scaled_dot_product_attention = patched_vlm_sdpa
            for mod_name, mod in list(sys.modules.items()):
                if mod is None or not mod_name.startswith("mlx_vlm.models."):
                    continue
                if (
                    getattr(mod, "scaled_dot_product_attention", None)
                    is original_vlm_sdpa
                ):
                    mod.scaled_dot_product_attention = patched_vlm_sdpa

    # Keep the prefill memory guard in lockstep: tell the monitor head_dim 256
    # prefill is now O(L), so it stops charging the O(L^2) score matrix.
    try:
        from .. import memory_monitor

        memory_monitor.register_tiled_prefill_head_dim(
            HEAD_DIM, min_kv_len=min_kv_len, kv_tile=_KV_TILE
        )
    except Exception:
        logger.debug("could not register sdpa256 with memory_monitor", exc_info=True)

    _PATCHED = True
    if _FORCE_TILED is None:
        qsplit_note = "q-split then " if _QSPLIT_ENABLED else ""
        routing = f"{qsplit_note}tiled only when unfused exceeds guard headroom"
    elif _FORCE_TILED:
        routing = "always tiled (OMLX_SDPA256_TILED=1)"
    else:
        routing = "never tiled or q-split (OMLX_SDPA256_TILED=0)"
    logger.info(
        "sdpa256 attention patch applied (head_dim=256 prefill, kv_len>=%d, %s)",
        min_kv_len,
        routing,
    )
    return True
