# SPDX-License-Identifier: Apache-2.0
"""Keep head-dim-256 long-context prefill bounded on MLX 0.32.2.

MLX 0.32.2 ships a fused full-attention kernel for head dimensions 192 and 256,
but deliberately keeps the faster unfused path as the default on pre-NAX GPUs.
That default materializes the full ``[n_q, query_len, kv_len]`` score matrix and
can still exceed oMLX's memory-guard ceiling.

When the unfused transient fits, this patch preserves MLX's default routing.
When it doesn't, the route narrows in two stages as live guard headroom shrinks
(``_route_decision``):

- **q-split** (``_unfused_qsplit_sdpa``): split the query axis into sub-tiles
  (keys/values narrowed to each sub-tile's causal end) and run the SAME fast
  stock kernel per sub-tile -- a smaller transient, still the fast kernel, no
  accuracy cost. ``OMLX_SDPA256_QSPLIT=0`` disables this stage.
- **bounded/tiled** (``_flash_sdpa256``): the last resort once even a
  minimum-size q-split slice wouldn't fit (or no guard ceiling is available).
  On Metal it calls MLX 0.32.2 with ``force_fused=True`` (the native fused
  kernel); explicit array masks and sinks take the portable array-tiled
  fallback, which the native fused path can silently unfuse. This replaces
  oMLX's old pure-array-only bounded implementation. On NAX, MLX's default
  already selects its fast split-D head-dim-256 kernel for causal prefills
  with at least 1024 queries.

``OMLX_SDPA256_TILED=1/0`` remains accepted for compatibility and now forces or
disables the bounded route. Metal uses the native fused kernel; CUDA retains
the prior array-tiled implementation because MLX 0.32.2's CUDA fused kernel
does not support head_dim 256. The default is memory-aware.

Install mechanics mirror turboquant_attention.py (patch the module attr + rebind
already-imported model modules). The route is strictly gated (see _should_route);
everything else passes through to the original SDPA unchanged.
"""

import logging
import os
import threading
import weakref

import mlx.core as mx

from omlx.memory_monitor import (
    SDPA256_UNFUSED_SCORE_DTYPE_SIZE,
    estimate_unfused_sdpa_call_bytes,
)

logger = logging.getLogger(__name__)

_PATCHED = False

HEAD_DIM = 256
# Force the bounded kernel only once the context is long enough that the
# default unfused route's O(L^2) score matrix becomes a memory problem.
_SDPA256_MIN_KV_LEN = 8192
# Decode-shaped multi-row calls (MTP verify: q_len = 1 + draft depth <= 9)
# do not need the forced full-attention route. Below this floor the stock path's
# score matrix is at most n_q * 15 * kv_len and is not a memory problem.
_SDPA256_MIN_Q_LEN = 16
_Q_TILE = 512
# A deliberately conservative score-tile width used only by the admission
# estimate. MLX's fused kernel keeps a smaller on-chip block, so this does not
# understate the bounded route's score working set.
_KV_TILE = 1024
_NEG_INF = -1e30

# Live guard-headroom provider for memory-aware routing (issue #2204).
# Scheduler.step registers the active Scheduler on its execution thread. Each
# engine uses its own worker, so thread-local storage keeps concurrent engines
# from replacing one another's provider. The bound method is weakly held so a
# torn-down Scheduler leaves that worker on the memory-bounded native fused
# default.
_HEADROOM_PROVIDER_LOCAL = threading.local()
# Backward-compatible override: True = always force fused, False = never force,
# None = memory-aware auto.
_FORCE_TILED: bool | None = None
# OMLX_SDPA256_QSPLIT override, parsed at apply time: False disables the
# q-split route (falls straight to tiled once unfused doesn't fit, restoring
# pre-q-split behavior for rollback); True/None (default) leaves it enabled.
_QSPLIT_ENABLED: bool = True
# Minimum query rows per q-split sub-call. Below this the per-call overhead
# stops paying for itself and genuinely tight headroom is better served by the
# tiled path's true O(L) floor.
_QSPLIT_MIN_Q = 128
# Route decisions round-trip through here so callers branch on one value.
_ROUTE_UNFUSED = "unfused"
_ROUTE_QSPLIT = "qsplit"
_ROUTE_TILED = "tiled"
# Last route decision, so we log every *transition* (not just the first-ever
# engagement) at INFO -- cheap, and lets a live server log show a request
# flapping between routes as guard headroom rises and falls mid-prefill. The
# tiled/qsplit pass trades prefill throughput for safer memory, and nothing
# surfaced the route decision before (issue #2283 took an A/B repro to
# diagnose), so this stays at INFO rather than DEBUG.
_LAST_ROUTE_DECISION: "str | None" = None


def _note_route(decision: str, detail) -> None:
    """``detail`` may be a plain string or a zero-arg callable. This runs on
    every head-dim-256 prefill SDPA call (thousands per long-context request),
    and the vast majority hit the decision == last-decision no-op below --
    callers with a formatted (f-string) detail should pass a lambda so the
    string is only built on an actual transition."""
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
    sub-tile -- every sub-tile is a *different* size, and Metal's buffer pool
    cannot reuse a differently-sized buffer for the next one. They accumulate
    as retained (not just active) memory across the whole loop: ``mx.eval``
    between sub-tiles bounds what is *live* to one sub-tile, but says nothing
    about what the pool has *retained* from the ones before it.

    Non-causal: every sub-tile is ``q_sub`` rows against the full, constant
    ``kv_len`` (no narrowing) -- same size as every other sub-tile, so the pool
    genuinely reuses one buffer across the loop and only ONE sub-tile's
    transient (not the whole q_len's) is ever retained at a time. Summing here
    would overcharge a case that was never broken.
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
    """Largest (128-aligned) query-row sub-tile count whose q-split transient
    fits ``headroom`` -- the *total* retained across every sub-tile in the
    split for a causal call (see ``_total_qsplit_bytes``), not the naive "one
    sub-tile at a time" a per-call eval only bounds the *active* set to.
    Non-causal degrades to the original single-transient inversion, since every
    sub-tile there is the same size and genuinely reusable.

    A closed-form inversion of the causal sum is a quadratic in ``q_sub``; a
    bounded linear search from a safe starting point (this call's
    single-sub-tile estimate, an upper bound since it ignores accumulation and
    can therefore only be too large) is simpler to keep correct than an
    inverted formula that has to be re-derived by hand every time this
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
    """Bind the active Scheduler's headroom provider to this worker thread."""
    ref = getattr(_HEADROOM_PROVIDER_LOCAL, "ref", None)
    current = ref() if ref is not None else None
    if (
        current is not None
        and current.__self__ is method.__self__
        and current.__func__ is method.__func__
    ):
        return
    _HEADROOM_PROVIDER_LOCAL.ref = weakref.WeakMethod(method)


def _get_unfused_headroom_provider():
    ref = getattr(_HEADROOM_PROVIDER_LOCAL, "ref", None)
    return ref() if ref is not None else None


def _parse_force_tiled_env() -> bool | None:
    value = os.environ.get("OMLX_SDPA256_TILED", "").strip()
    if value == "1":
        return True
    if value == "0":
        return False
    return None


def _parse_qsplit_env() -> bool:
    return os.environ.get("OMLX_SDPA256_QSPLIT", "").strip() != "0"


def _notify_bounded_route(provider, active: bool) -> None:
    """Let the scheduler retire measurements from the previous route."""
    try:
        owner = getattr(provider, "__self__", None)
        callback = getattr(owner, "_sdpa256_bounded_route_changed", None)
        if callable(callback):
            callback(active)
    except Exception:
        logger.debug("sdpa256 route notification failed", exc_info=True)


def _route_decision(
    queries, keys, mask, sinks, q_sub_ceiling: "int | None" = None
) -> "tuple[str, int]":
    """Decide unfused / q-split / tiled for a shape-matched prefill call.

    Returns ``(route, q_sub)`` -- ``q_sub`` is only meaningful for
    ``_ROUTE_QSPLIT`` (query rows per sub-call), 0 otherwise.

    The stock unfused fallback is faster wherever its transient fits (issues
    #2155 / #2204): take it whole when the full call fits, split the query axis
    into sub-calls that individually fit when the full call doesn't (still the
    fast kernel, just narrower), and fall back to the true O(L) tiled pass only
    once even a minimum-size q-split slice wouldn't fit, or when headroom info
    is unavailable (memory-safe #2025 default).

    Reconciliation notes (new vs the pre-rebase PR #2991 code -- flagged for
    delta review):
      * Pricing: the unfused fallback materializes fp32 scores even for bf16
        inputs (#3461). All three qsplit sizing sites are priced with the
        shared ``SDPA256_UNFUSED_SCORE_DTYPE_SIZE``, not ``queries.dtype.size``
        -- pricing at the query dtype halves the predicted transient for bf16
        and admits OOM. This is the same constant the admission guard uses.
      * Route-flip notification: ``_notify_bounded_route`` (#3461) tells the
        scheduler to retire EWMA measurements when the memory regime changes.
        The scheduler protocol is boolean bounded/unbounded; q-split IS
        memory-bounded (a narrower transient than the full unfused call), so it
        notifies ``active=True`` exactly like tiled. Only the whole-call
        unfused route is ``active=False``. Collapsing qsplit and tiled onto the
        same boolean means a qsplit<->tiled flip does NOT retire history -- and
        that is safe *because* the transient tracker maxes its observations:
        carrying qsplit's (larger) transients into a tiled regime only
        over-predicts, the conservative direction. Do not "fix" this into a
        3-way enum.
      * Array masks / sinks never take q-split: q-split narrows KV to each
        sub-tile's causal end (only valid for causal), and the array-tiled
        bounded kernel (#3461) is the route that actually slices an array mask
        / applies sinks correctly. They fall through to tiled here, matching
        main's array-mask-capable bounded routing (Qwen4's profile trusts it).

    ``q_sub_ceiling`` is this request's hysteresis floor (set by
    ``_should_route`` once a call has ever needed a smaller transient than the
    full call): caps how large a transient this call may use, not just which
    route label it gets. kv_len only grows within a request, so a transient
    shed earlier reflects real, non-relaxing pressure. ``q_sub_ceiling == 0``
    means a previous call already needed tiled -- never try qsplit or unfused
    again this request."""
    provider = _get_unfused_headroom_provider()
    if _FORCE_TILED is not None:
        if _FORCE_TILED:
            _note_route(_ROUTE_TILED, "forced by OMLX_SDPA256_TILED=1")
            _notify_bounded_route(provider, True)
            return _ROUTE_TILED, 0
        _note_route(_ROUTE_UNFUSED, "forced by OMLX_SDPA256_TILED=0")
        _notify_bounded_route(provider, False)
        return _ROUTE_UNFUSED, 0
    try:
        if q_sub_ceiling == 0:
            _note_route(
                _ROUTE_TILED,
                "held at tiled by this request's hysteresis floor",
            )
            _notify_bounded_route(provider, True)
            return _ROUTE_TILED, 0
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
        # Price every qsplit sizing site at the fp32 score dtype the unfused
        # fallback actually materializes (#3461), never the query dtype.
        score_dtype_size = SDPA256_UNFUSED_SCORE_DTYPE_SIZE
        causal = isinstance(mask, str) and mask == "causal"
        # q-split is only valid for the string masks whose per-sub-tile KV
        # narrowing preserves the result: causal (narrow to the causal end)
        # and no-mask (full KV every sub-tile). Explicit array masks and sinks
        # must reach the bounded tiled kernel, never a narrowed stock call.
        allow_qsplit = (
            _QSPLIT_ENABLED
            and sinks is None
            and not isinstance(mask, mx.array)
        )
        headroom = provider(kv_len, q_len)
        if headroom is None or headroom < 0:
            _note_route(
                _ROUTE_TILED,
                "memory ceiling not available (enforcer state not yet "
                "propagated)",
            )
            _notify_bounded_route(provider, True)
            return _ROUTE_TILED, 0
        transient = estimate_unfused_sdpa_call_bytes(
            n_q_heads, q_len, kv_len, HEAD_DIM, score_dtype_size=score_dtype_size
        )
        if transient <= headroom and q_sub_ceiling is None:
            _note_route(
                _ROUTE_UNFUSED,
                lambda: f"full call ~{transient / 2**20:.0f}MiB fits "
                f"~{headroom / 2**20:.0f}MiB headroom at kv_len={kv_len}",
            )
            _notify_bounded_route(provider, False)
            return _ROUTE_UNFUSED, 0
        if allow_qsplit:
            q_sub = _max_q_sub_for_headroom(
                n_q_heads, q_len, kv_len, causal, HEAD_DIM,
                score_dtype_size, headroom,
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
                _notify_bounded_route(provider, True)
                return _ROUTE_QSPLIT, q_sub
        _note_route(
            _ROUTE_TILED,
            lambda: f"unfused transient ~{transient / 2**20:.0f}MiB exceeds "
            f"~{headroom / 2**20:.0f}MiB headroom at kv_len={kv_len} even "
            f"at the q-split floor ({_QSPLIT_MIN_Q} rows)",
        )
        _notify_bounded_route(provider, True)
        return _ROUTE_TILED, 0
    except Exception:
        logger.debug("sdpa256 headroom probe failed", exc_info=True)
        _note_route(_ROUTE_TILED, "guard headroom probe failed")
        return _ROUTE_TILED, 0  # headroom info unavailable -> memory-safe default


def _broadcast_mask_5d(mask, batch, n_kv, group_size, q_len, k_len):
    """Reshape an array mask for the tiled GQA attention layout."""
    if mask.ndim == 4:
        pass
    elif mask.ndim == 3:
        # Preserve mlx-lm's convention: [batch, query, key].
        mask = mask[:, None, :, :]
    elif mask.ndim == 2:
        mask = mask[None, None, :, :]
    elif mask.ndim == 1:
        mask = mask[None, None, None, :]
    else:
        raise ValueError(f"unsupported attention mask ndim: {mask.ndim}")
    n_q = n_kv * group_size
    mask = mx.broadcast_to(mask, (batch, n_q, q_len, k_len))
    return mask.reshape(batch, n_kv, group_size, q_len, k_len)


def _array_tiled_sdpa256(queries, keys, values, scale, mask, sinks=None):
    """Portable bounded fallback for shapes without a native fused kernel."""
    batch, n_q, q_len, head_dim = queries.shape
    _, n_kv, k_len, _ = keys.shape
    value_dim = values.shape[-1]
    group_size = n_q // n_kv
    causal = isinstance(mask, str) and mask == "causal"
    array_mask = None
    if isinstance(mask, mx.array):
        array_mask = _broadcast_mask_5d(mask, batch, n_kv, group_size, q_len, k_len)

    qr = queries.reshape(batch, n_kv, group_size, q_len, head_dim)
    kr = keys.reshape(batch, n_kv, 1, k_len, head_dim)
    vr = values.reshape(batch, n_kv, 1, k_len, value_dim)
    offset = k_len - q_len

    out_q_tiles = []
    for qi0 in range(0, q_len, _Q_TILE):
        qi1 = min(qi0 + _Q_TILE, q_len)
        qb = qr[:, :, :, qi0:qi1, :].astype(mx.float32)
        qt = qi1 - qi0
        q_pos = mx.arange(qi0 + offset, qi1 + offset).reshape(1, 1, 1, qt, 1)

        state_shape = (batch, n_kv, group_size, qt, 1)
        if sinks is None:
            m = mx.full(state_shape, _NEG_INF, dtype=mx.float32)
            denom = mx.zeros(state_shape, dtype=mx.float32)
        else:
            sink_logits = sinks.astype(mx.float32).reshape(1, n_kv, group_size, 1, 1)
            m = mx.broadcast_to(sink_logits, state_shape)
            denom = mx.ones(state_shape, dtype=mx.float32)
        acc = mx.zeros((batch, n_kv, group_size, qt, value_dim), dtype=mx.float32)

        kv_end = min(qi1 + offset, k_len) if causal else k_len
        for kj0 in range(0, kv_end, _KV_TILE):
            kj1 = min(kj0 + _KV_TILE, kv_end)
            kb = kr[:, :, :, kj0:kj1, :].astype(mx.float32)
            vb = vr[:, :, :, kj0:kj1, :].astype(mx.float32)
            kt = kj1 - kj0

            scores = (qb @ mx.swapaxes(kb, -1, -2)) * scale
            if causal:
                k_pos = mx.arange(kj0, kj1).reshape(1, 1, 1, 1, kt)
                scores = mx.where(k_pos > q_pos, _NEG_INF, scores)
            elif array_mask is not None:
                tile_mask = array_mask[..., qi0:qi1, kj0:kj1]
                if tile_mask.dtype == mx.bool_:
                    scores = mx.where(tile_mask, scores, _NEG_INF)
                else:
                    scores = scores + tile_mask.astype(mx.float32)

            tile_max = mx.max(scores, axis=-1, keepdims=True)
            new_max = mx.maximum(m, tile_max)
            probabilities = mx.exp(scores - new_max)
            correction = mx.exp(m - new_max)
            denom = denom * correction + mx.sum(probabilities, axis=-1, keepdims=True)
            acc = acc * correction + (probabilities @ vb)
            m = new_max
            mx.eval(m, denom, acc)

        out_tile = (acc / denom).astype(queries.dtype)
        mx.eval(out_tile)
        out_q_tiles.append(out_tile)

    out = mx.concatenate(out_q_tiles, axis=3)
    return out.reshape(batch, n_q, q_len, value_dim)


# ``force_fused=`` arrived in MLX 0.32.2. On an older runtime the keyword is a
# TypeError, and retrying without it is not a safe substitute -- MLX would then
# be free to pick the unfused fp32 score matrix, which is the O(L^2) spike this
# patch exists to bound. Such a runtime routes to the array-tiled path instead,
# which is bounded by construction.
#
# Probed by use rather than by signature: MLX's nanobind functions report
# ``(*args, **kwargs)``, so the keyword is only visible in the docstring, and
# ``python -OO`` strips that. One TypeError on the first call is cheaper than a
# fragile capability check, and the answer is latched.
_NATIVE_FORCE_FUSED = True


def _flash_sdpa256(queries, keys, values, scale, mask, sinks=None):
    """Use MLX 0.32.2 native fused SDPA on Metal, portable tiling elsewhere.

    Explicit array masks never take the native fused call: MLX's fused
    array-mask support is unproven and may silently fall back to the
    unfused fp32 score matrix, which is exactly the O(L^2) spike this patch
    exists to bound. The array-tiled implementation already handles bool and
    additive masks, so route them there directly."""
    global _NATIVE_FORCE_FUSED

    if isinstance(mask, mx.array):
        return _array_tiled_sdpa256(queries, keys, values, scale, mask, sinks)
    native_shape = values.shape[-1] == HEAD_DIM and not (
        isinstance(mask, str)
        and mask == "causal"
        and queries.shape[-2] > keys.shape[-2]
    )
    if mx.metal.is_available() and native_shape and _NATIVE_FORCE_FUSED:
        try:
            return mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=scale,
                mask=mask,
                sinks=sinks,
                force_fused=True,
            )
        except TypeError:
            # Falling through to the tiled path rather than re-raising: it is a
            # correct implementation of the same op, so a genuinely malformed
            # call still fails there rather than being swallowed here.
            _NATIVE_FORCE_FUSED = False
            logger.warning(
                "sdpa256: mlx %s has no force_fused= (0.32.2+); using the "
                "array-tiled bounded route instead of the native fused kernel",
                getattr(mx, "__version__", "?"),
            )
    return _array_tiled_sdpa256(queries, keys, values, scale, mask, sinks)


def _unfused_qsplit_sdpa(
    queries, keys, values, cache, scale, mask, sinks, original_sdpa, q_sub
):
    """Run the fast stock (unfused) SDPA kernel over query sub-tiles instead of
    the whole chunk, each with keys/values narrowed to that sub-tile's causal
    end -- same kernel as the full-call fast path, just a smaller per-call
    transient so it fits under tighter headroom than a single call would.

    Correctness: MLX's ``mask="causal"`` right-aligns queries to the tail of
    the key axis (see ``_flash_sdpa256``'s comment on the same convention).
    For sub-tile [qi0:qi1) with global offset ``kv_off = k_len - q_len``,
    narrowing keys/values to [0:kv_off+qi1) makes MLX infer the offset
    ``(kv_off+qi1) - (qi1-qi0) = kv_off+qi0`` for this call -- exactly the
    sub-tile's true global offset -- so no manual position masking is needed.
    This also does less wasted compute than a single full call: each sub-tile's
    GEMM only covers the KV prefix its own causal window can see, not the full
    kv_len every time. For ``mask=None`` (full bidirectional attention) every
    sub-tile sees the full, constant KV -- also correct, no narrowing.

    Memory: each causal sub-tile's keys/values window grows with ``qi1`` (a
    later sub-tile sees a wider causal prefix than an earlier one), so without
    forcing evaluation between sub-tiles MLX's laziness lets every sub-call's
    graph -- including its own score-matrix transient -- stay unmaterialized
    and pile up simultaneously, rather than bounding the live set to one
    sub-tile at a time the way ``q_sub``'s sizing assumes. ``mx.eval`` here
    mirrors ``_flash_sdpa256``'s own per-tile eval for the same reason:
    confirmed necessary, not just defensive, by a live run where q-split
    engaged but didn't prevent the memory trip it was sized to avoid."""
    q_len = queries.shape[-2]
    causal = isinstance(mask, str) and mask == "causal"
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


def _should_route(queries, keys, cache, mask, sinks) -> "tuple[str, int]":
    # Never raise: any unexpected input must fall through to the original SDPA,
    # never break a request. Worst case we decline to engage (unfused).
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
        # Quantized KV cache (TurboQuant etc.): keys/values are packed state,
        # not plain [.., kv, hd] arrays. MLX's own dispatcher detects this via
        # hasattr(cache, "bits"); let the quant-aware path handle it.
        if cache is not None and hasattr(cache, "bits"):
            return _ROUTE_UNFUSED, 0
        # Array masks and sinks stay eligible for the bounded route (#3461):
        # the tiled kernel slices array masks and applies sinks correctly, and
        # Qwen4's profile trusts only an array-mask-capable bounded route. They
        # never reach q-split (see _route_decision's allow_qsplit).
        if not (
            mask is None
            or (isinstance(mask, str) and mask == "causal")
            or (isinstance(mask, mx.array) and 1 <= mask.ndim <= 4)
        ):
            return _ROUTE_UNFUSED, 0
        n_q = queries.shape[-3]
        n_kv = keys.shape[-3]
        if n_kv <= 0 or n_q % n_kv != 0:
            return _ROUTE_UNFUSED, 0
        # Hysteresis floor: once this request's cache has needed a smaller
        # transient than the full call, never let a later chunk's estimate push
        # the transient back up -- kv_len is monotone within a request, so the
        # pressure that forced the downgrade cannot have relaxed by the next
        # chunk. Ratchets on transient SIZE (q_sub), not just the route label:
        # capping only the label and letting q_sub float back up to q_len when
        # headroom looks momentarily generous was verified live to reproduce
        # the identical full-size transient labeled qsplit instead of unfused.
        # Stashed on ``cache`` -- the one object stable across every chunk/layer
        # of a single request but never shared across requests.
        ceiling = getattr(cache, "_sdpa256_q_sub_ceiling", None)
        route, q_sub = _route_decision(
            queries, keys, mask, sinks, q_sub_ceiling=ceiling
        )
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


def _register_bounded_route(min_kv_len: int) -> bool:
    """Publish only a runtime guarantee that is actually enabled."""
    if _FORCE_TILED is False:
        return False
    try:
        from .. import memory_monitor

        memory_monitor.register_tiled_prefill_head_dim(
            HEAD_DIM,
            min_query_len=_SDPA256_MIN_Q_LEN,
            min_kv_len=min_kv_len,
            kv_tile=_KV_TILE,
            supports_array_mask=True,
        )
    except Exception:
        logger.debug("could not register sdpa256 with memory_monitor", exc_info=True)
        return False
    return True


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
                return _flash_sdpa256(queries, keys, values, scale, mask, sinks)
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
                        return _flash_sdpa256(
                            queries, keys, values, scale, mask, sinks
                        )
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
    # prefill is now O(L), so it stops charging the O(L^2) score matrix. The
    # explicit benchmark override disables this guarantee, so registering it
    # in that mode would under-estimate the same unfused path the user forced.
    _register_bounded_route(min_kv_len)

    _PATCHED = True
    if _FORCE_TILED is None:
        qsplit_note = "q-split then " if _QSPLIT_ENABLED else ""
        routing = (
            f"{qsplit_note}bounded only when unfused exceeds guard headroom"
        )
    elif _FORCE_TILED:
        routing = "always force bounded (OMLX_SDPA256_TILED=1)"
    else:
        routing = "never force bounded or q-split (OMLX_SDPA256_TILED=0)"
    logger.info(
        "sdpa256 attention patch applied (head_dim=256 prefill, kv_len>=%d, %s)",
        min_kv_len,
        routing,
    )
    return True
