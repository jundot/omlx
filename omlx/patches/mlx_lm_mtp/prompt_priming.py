# SPDX-License-Identifier: Apache-2.0
"""MTP-head prompt priming: fold the prompt into the head cache during prefill.

Without priming the MTP head starts generation with an empty KV cache — its
first drafts see none of the prompt and acceptance starts context-starved,
recovering only as committed generation tokens accumulate (MTPLX measured
committed-history priming at 0.90 acceptance vs 0.26 unprimed on depth-1
real-code prompts). This module rides the existing prefill forwards: every
backbone chunk forward already computes the trunk-normed hidden for all chunk
positions, so the (hidden[t], token[t+1]) pairs the head history needs are
available for free. Each chunk is folded into a head cache immediately and
the chunk hidden is discarded — only a single (1, 1, H) pending row carries
across chunks.

Transport: scheduler requests own their contexts until insertion assigns a
BatchGenerator UID. Decode capture uses the exact UID order of the active
GenerationBatch, including ordinary decode used for cost calibration. The
host slot is only a cursor while a request or row runs. Direct singleton
callers retain the original slot interface. Every capture also verifies
offset contiguity; request identity and offset are both required.

Capture sites (each calls :func:`maybe_capture` after the backbone forward):

- mlx-lm qwen3_5 text path: the patched ``TextModel.__call__``
  (``qwen35_model``), which computes the trunk-normed hidden inline.
- mlx-vlm qwen3_5 path: a wrap on the inner ``Qwen3_5Model.__call__``
  (``qwen35_vlm_runtime``), whose return value *is* the trunk-normed
  hidden; the MoE inner model inherits it. The outer ``LanguageModel`` is
  reached via a weakref stamped at init.
- DeepSeek-V4 (``deepseek_v4_model``): the patched ``Model.__call__``
  requests ``return_raw_hidden`` and passes the raw 4D Hyper-stream hidden
  (the head input variant; no trunk norm).
- GLM-5.2 (``glm_moe_dsa_model``): the patched ``Model.__call__`` passes
  the post-final-norm hidden it already computes.

All sites skip ``return_hidden=True`` forwards (MTP verify cycles and the
activation forward in ``_post_init_mtp``); the final (hidden[prompt[-1]],
main_tok) pair is folded by :func:`take_primed` at activation instead.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# The MTP head is fed the trunk's *post-norm* hidden and chains on its own
# post-norm output. Measured on Qwen3.6-27B this accepts a few points higher
# than PR 990's pre-norm at every depth. Draft-side only, so output identity
# is unaffected regardless. Priming folds must use the same variant as the
# decode-time history folds in batch_generator, hence the single definition
# here.
HEAD_HIDDEN_POST_NORM = True

_CTX_ATTR = "_omlx_mtp_prime_ctx"
_PLAN_ATTR = "_omlx_mtp_prime_plan"

_SUPPRESS = threading.local()


def priming_enabled() -> bool:
    """Prompt priming is on by default for MTP-enabled models."""
    return os.environ.get("OMLX_MTP_PROMPT_PRIMING", "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


_PRIME_WINDOW_CONFIG = 0


def set_prime_window_config(window: int) -> None:
    """Persist the per-model bound (ModelSettings.mtp_prime_window).

    Called by the model loader on every load; the env var remains an
    explicit override. Unbounded priming on huge prompts pushes the
    head-cache activation seam and its prefix-snapshot publish past
    the memory guard (the 750k ladder's decode-start abort), so the
    bound must survive reboots without launchctl session env.
    """
    global _PRIME_WINDOW_CONFIG
    try:
        _PRIME_WINDOW_CONFIG = max(0, int(window))
    except (TypeError, ValueError):
        _PRIME_WINDOW_CONFIG = 0


def prime_window() -> int:
    """Max tokens to fold into one prime context; 0 = unlimited.

    Escape hatch for the head-cache memory cost of priming (one
    full-attention layer of KV over the folded span). The cap is measured
    against the span actually folded this request — with a warm prefix cache
    that is only the boundary remainder, not the full prompt — so a
    long-context request with a small remainder still primes. A remainder
    larger than the window runs unprimed.

    Precedence: OMLX_MTP_PRIME_WINDOW env (explicit override) > the
    configured per-model bound > unlimited.
    """
    raw = os.environ.get("OMLX_MTP_PRIME_WINDOW", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return _PRIME_WINDOW_CONFIG


@contextmanager
def suppress_capture():
    """Disable capture on this thread for the duration of the block."""
    _SUPPRESS.value = True
    try:
        yield
    finally:
        _SUPPRESS.value = False


def _suppressed() -> bool:
    return bool(getattr(_SUPPRESS, "value", False))


@dataclass
class _PrimeCtx:
    """Streaming priming state in the host model's single slot."""

    mtp_cache: List[Any] = field(default_factory=list)
    # Head-input hidden of the newest seen token, (1, 1, ..., H) — pairs
    # with the first token of the next chunk (or main_tok at activation).
    pending_hidden: Optional[Any] = None
    # Folded (hidden, next_token) pairs == head-cache offset.
    folded: int = 0
    # Anchor cache offset observed after the last captured forward. The next
    # capture requires offset_now - S == expected_offset (contiguity).
    expected_offset: int = 0
    valid: bool = True
    # The current contiguous timeline exceeded OMLX_MTP_PRIME_WINDOW. Keep a
    # lightweight marker so later small chunks cannot restart priming.
    window_exceeded: bool = False
    # Absolute MTP history is ``folded``; this counter is only the work folded
    # by the current request.  A warm prefix restore starts at a nonzero
    # absolute history but must still apply OMLX_MTP_PRIME_WINDOW to the small
    # uncached suffix, preserving the option's documented meaning.
    folded_this_request: int = 0
    # Committed pairs collected during ordinary decoding. None means
    # ordinary prompt priming; a list resumes an already active head.
    deferred_pairs: Optional[List[Any]] = None
    # Request/prefix-cache metadata used to publish and restore one exact
    # full-block MTP boundary snapshot.  The cache itself remains generic and
    # treats the snapshot as an opaque sidecar.
    request_id: Optional[str] = None
    prompt_tokens: Optional[tuple[int, ...]] = None
    block_size: int = 0
    prefix_cache: Any = None
    extra_keys: Optional[tuple[Any, ...]] = None
    extra_key_token_start: Optional[int] = None
    extra_key_ranges: Optional[list[tuple[int, tuple[Any, ...]]]] = None
    snapshot_candidate: Any = None


@dataclass
class _PrimePlan:
    """Scheduler-owned metadata for the next singleton prompt timeline."""

    request_id: str
    prompt_tokens: tuple[int, ...]
    block_size: int
    prefix_cache: Any
    extra_keys: Optional[tuple[Any, ...]] = None
    extra_key_token_start: Optional[int] = None
    extra_key_ranges: Optional[list[tuple[int, tuple[Any, ...]]]] = None


@dataclass
class _MtpPrefixSnapshot:
    """Detached MTP-head state at a backbone full-block boundary."""

    boundary_tokens: int
    mtp_cache: List[Any]
    pending_hidden: Any


@dataclass
class _MtpBoundaryCandidate:
    """Cheap boundary marker retained until activation publishes a snapshot."""

    boundary_tokens: int
    pending_hidden: Any


def _read_offset(entry: Any) -> Optional[int]:
    """``entry.offset`` as a plain int, unwrapping size-1 array offsets.

    Batch caches (``BatchKVCache`` / ``BatchRotatingKVCache``) hold their
    offset as a 1-element ``mx.array``. Reading it costs one sync, so
    callers do it once per forward at most.
    """
    offset = getattr(entry, "offset", None)
    if type(offset) is int:
        return offset
    if offset is not None and getattr(offset, "size", 0) == 1:
        try:
            return int(offset.reshape(()).item())
        except Exception:
            return None
    return None


def _offset_readable(entry: Any) -> bool:
    """Whether :func:`_read_offset` can serve this entry — no sync."""
    offset = getattr(entry, "offset", None)
    return type(offset) is int or (
        offset is not None and getattr(offset, "size", 0) == 1
    )


class _IntOffsetAnchor:
    """Anchor view exposing a scalar-or-size-1-array offset as an int.

    Under ``BatchGenerator`` every request's caches are merged into
    ``Batch*`` entries at ``PromptProcessingBatch.__init__``, whose
    ``offset`` is a 1-element ``mx.array`` **even for a single request**
    (B==1). The plain-int probe this replaces therefore found no anchor on
    any batch-engine prefill, so ``maybe_capture`` bailed silently and
    priming never activated there (#3079).

    The unwrap is unambiguous because :func:`maybe_capture` only captures
    ``(1, S)`` forwards — a singleton timeline. It does cost one ``int()``
    sync per captured forward, which is what the contiguity invariant is
    built on; ``BatchRotatingKVCache._offset`` would be sync-free but
    counts buffer slots rather than tokens.
    """

    __slots__ = ("_cache",)

    def __init__(self, cache: Any) -> None:
        self._cache = cache

    @property
    def offset(self) -> Optional[int]:
        return _read_offset(self._cache)


def _anchor(cache: Optional[List[Any]]) -> Optional[Any]:
    """First cache entry whose offset can be read as an int, as a view.

    Container layers (``CacheList``-style, exposing ``.caches`` — DeepSeek-V4
    and GLM-5.2 backbones) are searched one level deep: the container itself
    has no offset but its first sub-cache (RotatingKVCache / KVCache) does.
    """
    if not cache:
        return None
    for c in cache:
        if _offset_readable(c):
            return _IntOffsetAnchor(c)
        for sub in getattr(c, "caches", ()) or ():
            if _offset_readable(sub):
                return _IntOffsetAnchor(sub)
    return None


def _activation_offset(cache: Optional[List[Any]]) -> Optional[int]:
    """Attention-layer offset at MTP activation, tolerant of batch caches.

    Between the last capture and activation, ``insert()`` runs mlx-lm's
    cache merge: scalar ``KVCache`` entries without singleton passthrough
    become batch caches whose ``offset`` is a 1-element ``mx.array``.
    """
    if not cache:
        return None
    for c in cache:
        got = _read_offset(c)
        if got is not None:
            return got
        for sub in getattr(c, "caches", ()) or ():
            got = _read_offset(sub)
            if got is not None:
                return got
    return None


def _host_candidates(model: Any):
    """The model itself plus the wrapped language model, if any.

    Mirrors ``batch_generator._resolve_mtp_chain_depth``: the host that
    carries the slot is the patched language-model instance — the outer
    adapter / VLM wrapper for qwen paths, the Model itself for DeepSeek/GLM.
    """
    yield model
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            yield inner


def _find_ctx(model: Any) -> Optional[_PrimeCtx]:
    for host in _host_candidates(model):
        ctx = getattr(host, _CTX_ATTR, None)
        if ctx is not None:
            return ctx
    return None


def _find_plan(model: Any) -> Optional[_PrimePlan]:
    for host in _host_candidates(model):
        plan = getattr(host, _PLAN_ATTR, None)
        if isinstance(plan, _PrimePlan):
            return plan
    return None


def drop_ctx(model: Any) -> None:
    """Remove any priming context/plan from the model's host slots."""
    if model is None:
        return
    for host in _host_candidates(model):
        for attr in (_CTX_ATTR, _PLAN_ATTR):
            if getattr(host, attr, None) is not None:
                try:
                    delattr(host, attr)
                except AttributeError:
                    pass


def _host_eligible(host: Any) -> bool:
    get_mtp = getattr(host, "get_mtp_module", None)
    mtp = get_mtp() if callable(get_mtp) else getattr(host, "mtp", None)
    return (
        getattr(host, "_omlx_mtp_decode_enabled", False) is True
        and getattr(host, "_omlx_mtp_chain", False) is True
        and mtp is not None
    )


def _eligible_host(model: Any) -> Any | None:
    for host in _host_candidates(model):
        if _host_eligible(host):
            return host
    return None


def _clone_mtp_cache(cache: List[Any]) -> List[Any]:
    """Detach an MTP cache so later decode writes cannot mutate a snapshot."""
    import copy

    import mlx.core as mx

    def clone_one(entry: Any) -> Any:
        if entry is None:
            return None
        subs = getattr(entry, "caches", None)
        if subs is not None:
            return type(entry)(*[clone_one(sub) for sub in subs])
        clone = copy.copy(entry)
        for attr, value in vars(entry).items():
            if isinstance(value, mx.array):
                setattr(clone, attr, value + 0)
            elif isinstance(value, list):
                setattr(clone, attr, list(value))
        return clone

    return [clone_one(entry) for entry in cache]


def _flat_cache_entries(cache: List[Any]):
    for entry in cache:
        subs = getattr(entry, "caches", None)
        if subs is None:
            yield entry
        else:
            yield from subs


def _cache_at_offset(cache: List[Any], target: int) -> Optional[List[Any]]:
    """Return a detached, exactly trimmed MTP cache or fail closed."""
    if target < 0 or not cache:
        return None
    cloned = _clone_mtp_cache(cache)
    saw_offset = False
    for entry in _flat_cache_entries(cloned):
        current = _read_offset(entry)
        if current is None:
            continue
        saw_offset = True
        if current < target:
            return None
        extra = current - target
        if extra:
            trim = getattr(entry, "trim", None)
            if not callable(trim) or int(trim(extra)) != extra:
                return None
        if _read_offset(entry) != target:
            return None
    return cloned if saw_offset else None


def _snapshot_arrays(snapshot: _MtpPrefixSnapshot) -> list[Any]:
    """Arrays that must be materialized to sever the live prefill graph."""
    import mlx.core as mx

    arrays: list[Any] = []
    if isinstance(snapshot.pending_hidden, mx.array):
        arrays.append(snapshot.pending_hidden)
    for entry in _flat_cache_entries(snapshot.mtp_cache):
        for value in vars(entry).values():
            if isinstance(value, mx.array):
                arrays.append(value)
    return arrays


def capture_eligible(host: Any, cache: Optional[List[Any]]) -> bool:
    """Cheap pre-check for capture sites that must decide the forward shape.

    The DeepSeek/GLM backbones only expose the head-input hidden when asked
    (``return_raw_hidden``), so their patched ``__call__`` consults this
    before choosing the call form. Everything here is re-checked inside
    :func:`maybe_capture`; this exists purely to keep the ineligible path
    identical to stock.
    """
    return (
        not _suppressed()
        and priming_enabled()
        and cache is not None
        and _host_eligible(host)
    )


@dataclass
class _OwnedPriming:
    requests: dict = field(default_factory=dict)
    uids: dict = field(default_factory=dict)


_OWNED_ATTR = "_omlx_mtp_owned_priming"
_DECODE_SCOPE = ContextVar("omlx_mtp_priming_decode", default=None)
_PREFILL_SCOPE = ContextVar("omlx_mtp_priming_prefill", default=None)


def _owned(model, create=False):
    host = _eligible_host(model)
    if host is None or getattr(host, "_omlx_dspark_decode_enabled", False):
        return None, None
    state = getattr(host, _OWNED_ATTR, None)
    if not isinstance(state, _OwnedPriming):
        state = _OwnedPriming() if create else None
        if state is not None:
            setattr(host, _OWNED_ATTR, state)
    return host, state


def _slot(host):
    return (getattr(host, _CTX_ATTR, None), getattr(host, _PLAN_ATTR, None))


def _restore_slot(host, record):
    for attr, value in zip((_CTX_ATTR, _PLAN_ATTR), record):
        if value is None:
            if getattr(host, attr, None) is not None:
                delattr(host, attr)
        else:
            setattr(host, attr, value)


def activate_request(model, request_id):
    """Select the request before each externally scheduled prefill chunk."""
    host, state = _owned(model)
    if state is not None:
        _restore_slot(host, state.requests.get(request_id, (None, None)))


def bind_uid(model, request_id, uid):
    """Move a prepared request's history to its assigned generator UID."""
    host, state = _owned(model)
    if state is None:
        return
    record = state.requests.pop(request_id, None)
    if record is not None:
        if uid in state.uids:
            raise RuntimeError("Lightning MTP priming UID already owned")
        state.uids[uid] = record
    current = _find_plan(host)
    if current is not None and current.request_id == request_id:
        drop_ctx(host)


def release_uids(model, uids):
    _, state = _owned(model)
    if state is not None:
        for uid in uids:
            state.uids.pop(uid, None)


def release_request(model, request_id):
    host, state = _owned(model)
    if state is None:
        return
    state.requests.pop(request_id, None)
    for uid, (_, plan) in list(state.uids.items()):
        if plan is not None and plan.request_id == request_id:
            del state.uids[uid]
    plan = _find_plan(host)
    if plan is not None and plan.request_id == request_id:
        drop_ctx(host)


def clear_owned(model):
    host, state = _owned(model)
    if state is not None:
        state.requests.clear()
        state.uids.clear()
        drop_ctx(host)


@contextmanager
def prefill_scope(model, uids, tokens, cache):
    """Identify each prompt row before the upstream loop adds right padding."""
    if len(uids) > 1 and not any(
        getattr(host, "_omlx_mtp_multi_request", False) is True
        for host in _host_candidates(model)
    ):
        yield
        return
    host, state = _owned(model, create=bool(tokens) and priming_enabled())
    if state is None or not tokens or len(uids) != len(tokens):
        yield
        return
    offsets = _row_offsets(cache, len(uids))
    if offsets is None:
        logger.debug("MTP prefill priming discarded: unknown per-row offsets")
        release_uids(model, uids)
        yield
        return
    for uid in uids:
        state.uids.setdefault(uid, (None, None))
    scope = dict(
        host=host,
        uids=tuple(uids),
        lengths=[len(t) for t in tokens],
        offsets=offsets,
        consumed=0,
    )
    token = _PREFILL_SCOPE.set(scope)
    try:
        yield
    finally:
        _PREFILL_SCOPE.reset(token)


@contextmanager
def decode_scope(model, uids):
    host, state = _owned(model)
    token = _DECODE_SCOPE.set((host, tuple(uids))) if state is not None else None
    try:
        yield
    finally:
        if token is not None:
            _DECODE_SCOPE.reset(token)


def prepare_prefix_context(model, *, request_id, **kwargs):
    host, state = _owned(model, create=priming_enabled())
    if state is None:
        return _prepare_prefix_context(model, request_id=request_id, **kwargs)
    activate_request(model, request_id)
    result = _prepare_prefix_context(model, request_id=request_id, **kwargs)
    record = _slot(host)
    if isinstance(record[1], _PrimePlan):
        state.requests[request_id] = record
    return result


def _prepare_prefix_context(
    model: Any,
    *,
    request_id: str,
    prompt_tokens: list[int],
    cached_tokens: int,
    prefix_cache: Any,
    extra_keys: Optional[tuple[Any, ...]] = None,
    extra_key_token_start: Optional[int] = None,
    extra_key_ranges: Optional[list[tuple[int, tuple[Any, ...]]]] = None,
) -> bool:
    """Prepare exact MTP priming for one scheduler-owned prompt timeline.

    ``cached_tokens`` is the final reconstructed backbone offset (after any
    exact-hit trim).  A matching sidecar restores the MTP-head KV at
    ``cached_tokens - 1`` plus the pending trunk hidden at the boundary, so
    the uncached suffix can continue folding without replaying the trunk.
    Missing, stale, VLM-range-keyed, or shape-incompatible snapshots fail
    closed to the existing unprimed path.

    Returns True only when a warm sidecar was restored.  Repeating the call
    for the same request is idempotent and never double-primes a live suffix.
    """
    host = _eligible_host(model)
    if host is not None and getattr(host, "_omlx_dspark_decode_enabled", False):
        # DSpark owns a different context in the shared priming slot. Another
        # request may still need it at activation; generic sidecar preparation
        # must neither interpret it nor replace it with a generic plan.
        return False
    if host is None or not priming_enabled():
        drop_ctx(model)
        return False

    tokens = tuple(int(token) for token in prompt_tokens)
    cached_tokens = max(0, int(cached_tokens))
    existing = _find_ctx(model)
    plan = _find_plan(model)
    if (existing is not None and existing.request_id == request_id) or (
        plan is not None
        and plan.request_id == request_id
        and plan.prompt_tokens == tokens
    ):
        return existing is not None and existing.expected_offset >= cached_tokens

    drop_ctx(model)
    plan = _PrimePlan(
        request_id=request_id,
        prompt_tokens=tokens,
        block_size=max(0, int(getattr(prefix_cache, "block_size", 0) or 0)),
        prefix_cache=prefix_cache,
        extra_keys=extra_keys,
        extra_key_token_start=extra_key_token_start,
        extra_key_ranges=(
            list(extra_key_ranges) if extra_key_ranges is not None else None
        ),
    )
    setattr(host, _PLAN_ATTR, plan)
    if cached_tokens <= 0:
        return False

    restore = getattr(prefix_cache, "restore_mtp_prefix_snapshot", None)
    if not callable(restore):
        return False
    try:
        snapshot = restore(
            list(tokens),
            cached_tokens,
            extra_keys=extra_keys,
            extra_key_token_start=extra_key_token_start,
            extra_key_ranges=extra_key_ranges,
        )
    except Exception as exc:
        logger.debug("MTP prefix sidecar lookup failed closed: %s", exc)
        return False
    if not isinstance(snapshot, _MtpPrefixSnapshot):
        return False
    if snapshot.boundary_tokens != cached_tokens or cached_tokens < 2:
        return False

    target_offset = cached_tokens - 1
    try:
        restored_cache = _cache_at_offset(snapshot.mtp_cache, target_offset)
        if restored_cache is None or snapshot.pending_hidden is None:
            return False

        import mlx.core as mx

        pending_hidden = snapshot.pending_hidden + 0
    except Exception as exc:
        logger.debug("MTP prefix sidecar restore failed closed: %s", exc)
        return False
    ctx = _PrimeCtx(
        mtp_cache=restored_cache,
        pending_hidden=pending_hidden,
        folded=target_offset,
        expected_offset=cached_tokens,
        request_id=request_id,
        prompt_tokens=tokens,
        block_size=plan.block_size,
        prefix_cache=prefix_cache,
        extra_keys=extra_keys,
        extra_key_token_start=extra_key_token_start,
        extra_key_ranges=plan.extra_key_ranges,
    )
    setattr(host, _CTX_ATTR, ctx)
    try:
        arrays = [pending_hidden, *_snapshot_arrays(snapshot)]
        if arrays:
            mx.async_eval(arrays)
    except Exception as exc:
        drop_ctx(model)
        logger.debug("MTP prefix sidecar materialization failed closed: %s", exc)
        return False
    logger.debug(
        "MTP prompt history restored at %d cached tokens for %s",
        cached_tokens,
        request_id,
    )
    return True


def _capture_boundary_candidate(
    ctx: _PrimeCtx,
    normed: Any,
    *,
    seq_start: int,
    seq_end: int,
) -> None:
    """Detach the newest full-block MTP boundary crossed by this chunk."""
    block = int(ctx.block_size or 0)
    if (
        block <= 0
        or ctx.prefix_cache is None
        or not ctx.prompt_tokens
        or seq_end < block
    ):
        return
    boundary = (seq_end // block) * block
    if boundary <= seq_start or boundary > len(ctx.prompt_tokens):
        return
    previous = ctx.snapshot_candidate
    if (
        isinstance(previous, _MtpBoundaryCandidate)
        and previous.boundary_tokens >= boundary
    ):
        return

    # A backbone boundary at C tokens needs MTP pairs through C-1 and keeps
    # hidden(token[C-1]) pending for the next token.  ``normed`` spans
    # [seq_start, seq_end), so the boundary row is available without replay.
    if boundary <= 1 or ctx.folded < boundary - 1:
        return
    row = boundary - seq_start - 1
    if row < 0 or row >= int(normed.shape[1]):
        return
    try:
        import mlx.core as mx

        candidate = _MtpBoundaryCandidate(
            boundary_tokens=boundary,
            pending_hidden=normed[:, row : row + 1] + 0,
        )
        mx.async_eval(candidate.pending_hidden)
    except Exception as exc:
        logger.debug("MTP prefix boundary capture failed closed: %s", exc)
        return
    ctx.snapshot_candidate = candidate


def _publish_boundary_candidate(ctx: _PrimeCtx) -> None:
    candidate = ctx.snapshot_candidate
    store = getattr(ctx.prefix_cache, "store_mtp_prefix_snapshot", None)
    if not isinstance(candidate, _MtpBoundaryCandidate) or not callable(store):
        return
    try:
        snapshot_cache = _cache_at_offset(ctx.mtp_cache, candidate.boundary_tokens - 1)
        if snapshot_cache is None:
            return
        snapshot = _MtpPrefixSnapshot(
            boundary_tokens=candidate.boundary_tokens,
            mtp_cache=snapshot_cache,
            pending_hidden=candidate.pending_hidden,
        )
        arrays = _snapshot_arrays(snapshot)
        if arrays:
            import mlx.core as mx

            mx.async_eval(arrays)
        stored = store(
            list(ctx.prompt_tokens or ()),
            snapshot.boundary_tokens,
            snapshot,
            extra_keys=ctx.extra_keys,
            extra_key_token_start=ctx.extra_key_token_start,
            extra_key_ranges=ctx.extra_key_ranges,
        )
    except Exception as exc:
        logger.debug("MTP prefix sidecar publish failed closed: %s", exc)
        return
    if stored:
        logger.debug(
            "MTP prompt history cached at %d-token boundary for %s",
            snapshot.boundary_tokens,
            ctx.request_id or "anonymous request",
        )


def _row_offsets(cache, size):
    import mlx.core as mx

    for entry in cache or ():
        for part in (entry, *(getattr(entry, "caches", ()) or ())):
            offset = getattr(part, "offset", None)
            if isinstance(offset, mx.array) and offset.ndim == 1:
                if offset.size == size:
                    return offset.tolist()
            elif size == 1 and isinstance(offset, int):
                return [offset]
    return None


def maybe_capture(host, inputs, normed, cache):
    if _suppressed() or not priming_enabled():
        return
    _, state = _owned(host)
    prefill = _PREFILL_SCOPE.get()
    prefill = prefill if prefill is not None and prefill["host"] is host else None
    scope = _DECODE_SCOPE.get()
    uids = None
    if state is not None:
        if prefill is not None:
            uids = prefill["uids"]
        elif scope is not None and scope[0] is host:
            uids = scope[1]
    if uids is not None:
        if inputs is None or inputs.ndim != 2 or len(uids) != inputs.shape[0]:
            raise RuntimeError("Lightning MTP priming UID scope mismatch")
        owned_uids = [uid for uid in uids if uid in state.uids]
        if not owned_uids:
            return
        offsets = (
            prefill["offsets"]
            if prefill is not None
            else _row_offsets(cache, len(uids))
        )
        if offsets is None:
            logger.debug("MTP priming discarded: unknown batched offsets")
            release_uids(host, owned_uids)
            return
        previous = _slot(host)
        count = int(inputs.shape[1])
        try:
            for row, uid in enumerate(uids):
                record = state.uids.get(uid)
                if record is None:
                    continue
                valid = count
                offset = int(offsets[row])
                if prefill is not None:
                    valid = min(
                        count, max(0, prefill["lengths"][row] - prefill["consumed"])
                    )
                    offset += prefill["consumed"] + valid
                if not valid:
                    continue
                _restore_slot(host, record)
                _capture_single(
                    host,
                    inputs[row : row + 1, :valid],
                    normed[row : row + 1, :valid],
                    [SimpleNamespace(offset=offset)],
                )
                state.uids[uid] = _slot(host)
        finally:
            _restore_slot(host, previous)
        if prefill is not None:
            prefill["consumed"] += count
        return
    _capture_single(host, inputs, normed, cache)
    if state is not None:
        plan = _find_plan(host)
        if plan is not None and plan.request_id in state.requests:
            state.requests[plan.request_id] = _slot(host)


def retain_batch_head_history(batch, owner):
    """Keep committed head history at a drained handoff to ordinary decode."""
    if not priming_enabled():
        return
    host, registry = _owned(batch.model, create=True)
    if registry is None:
        # Models with their own priming transport retain their existing path.
        logger.debug("MTP head history retention: model-owned priming transport")
        return
    states = [owner.states.get(uid) for uid in batch.uids]
    if any(
        state is None or not state.chain or state.queue or state.next_main is None
        for state in states
    ):
        raise RuntimeError("Cannot retain a non-drained batch head frontier")
    retained_uids = {
        uid for uid, state in zip(batch.uids, states) if state.head_history_primed
    }
    if not retained_uids:
        return
    offsets = _row_offsets(batch.prompt_cache, len(batch.uids))
    if offsets is None:
        raise RuntimeError("Cannot retain head history without target offsets")
    if any(uid in registry.uids for uid in retained_uids):
        raise RuntimeError("Handoff would overwrite owned head history")
    from . import batched_head
    from .batch_generator import _mtp_head_trim_to

    batched_head.flush(owner)
    for uid, state, offset in zip(batch.uids, states, offsets):
        if uid not in retained_uids:
            continue
        if not state.head_clone:
            _mtp_head_trim_to(state.mtp_cache, state.hist_offset)
        ctx = _PrimeCtx(
            mtp_cache=state.mtp_cache,
            folded=state.hist_offset,
            expected_offset=offset,
            deferred_pairs=[],
        )
        registry.uids[uid] = (ctx, None)


def _capture_deferred_history(host, inputs, hidden, cache):
    import mlx.core as mx

    ctx = _find_ctx(host)
    if not isinstance(ctx, _PrimeCtx) or ctx.deferred_pairs is None:
        return False
    anchor = _anchor(cache)
    count = int(inputs.shape[1])
    after = anchor.offset if anchor is not None else None
    if (
        inputs.shape[0] != 1
        or not ctx.valid
        or after is None
        or ctx.expected_offset != after - count
    ):
        raise RuntimeError("Deferred head history lost its request timeline")
    if ctx.pending_hidden is None:
        paired_hidden, paired_tokens = hidden[:, :-1], inputs[:, 1:]
    else:
        paired_hidden = mx.concatenate([ctx.pending_hidden, hidden[:, :-1]], axis=1)
        paired_tokens = inputs
    if paired_tokens.shape[1]:
        ctx.deferred_pairs.append((paired_hidden, paired_tokens))
    ctx.pending_hidden = hidden[:, -1:]
    ctx.expected_offset = after
    return True


def _flush_deferred_history(model, ctx, chunk_size=512):
    import mlx.core as mx

    pairs = ctx.deferred_pairs
    if not pairs:
        return
    hidden = mx.concatenate([h for h, _ in pairs], axis=1)
    tokens = mx.concatenate([t for _, t in pairs], axis=1)
    count = int(tokens.shape[1])
    if hidden.shape[:2] != tokens.shape:
        raise RuntimeError("Deferred hidden/token history has different lengths")
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        model.mtp_forward(
            hidden[:, start:end], tokens[:, start:end], ctx.mtp_cache, logits_keep=1
        )
        # Match prompt priming's asynchronous materialization: no CPU/GPU
        # barrier, and the catch-up work belongs to MTP reactivation cost.
        values = []
        for layer in ctx.mtp_cache:
            for part in getattr(layer, "caches", (layer,)):
                values.extend(
                    v
                    for v in (
                        getattr(part, "keys", None),
                        getattr(part, "values", None),
                    )
                    if v is not None
                )
        mx.async_eval(values)
    ctx.folded += count
    ctx.folded_this_request += count
    ctx.deferred_pairs = []


def _capture_single(
    host: Any, inputs: Any, normed: Any, cache: Optional[List[Any]]
) -> None:
    """Fold this forward's (hidden, next_token) pairs into the priming cache.

    ``host`` is the patched language model (mlx-lm ``TextModel`` or mlx-vlm
    ``LanguageModel``) exposing ``mtp`` / ``model.embed_tokens`` /
    ``make_mtp_cache``. ``normed`` is the trunk-normed hidden for all
    positions of ``inputs`` (1, S, H). Host-side bookkeeping only — the head
    forward is dispatched lazily and no GPU sync happens here.

    Call sites guard the cheap negatives (return_hidden / n_confirmed /
    inputs_embeds) before calling; everything here re-checks what is
    load-bearing and bails silently, so a miss degrades to unprimed.
    """
    if _suppressed() or not priming_enabled():
        return
    if cache is None or not _host_eligible(host):
        return
    if inputs is None or getattr(inputs, "ndim", 0) != 2:
        return
    if inputs.shape[0] != 1:
        # A B>1 forward advances the anchor invisibly to capture, so a later
        # singleton chunk could look contiguous with a timeline it never
        # belonged to (chunk boundaries are aligned across requests). Drop
        # the slot rather than risk a wrong history.
        drop_ctx(host)
        return
    anchor = _anchor(cache)
    if anchor is None:
        return

    if _capture_deferred_history(host, inputs, normed, cache):
        return

    import mlx.core as mx

    seq_len = int(inputs.shape[1])
    offset_after = anchor.offset  # forward already ran; offset includes S
    if offset_after is None:
        return

    ctx = getattr(host, _CTX_ATTR, None)
    if ctx is not None and (
        not ctx.valid or ctx.expected_offset != offset_after - seq_len
    ):
        # Rewind / trim / request switch / unknown path: never guess.
        plan = _find_plan(host)
        drop_ctx(host)
        if plan is not None:
            setattr(host, _PLAN_ATTR, plan)
        ctx = None
    if ctx is not None and ctx.window_exceeded:
        ctx.expected_offset = offset_after
        return
    window = prime_window()
    if window:
        # Cap by the primed span (the head-KV the window exists to bound),
        # not the absolute prompt offset: on a warm prefix cache only the
        # boundary remainder is ever folded, so a long-context request with a
        # small remainder is exactly the cheap case priming is for (#2909).
        folded = ctx.folded_this_request if ctx is not None else 0
        if folded + seq_len > window:
            setattr(
                host,
                _CTX_ATTR,
                _PrimeCtx(
                    expected_offset=offset_after,
                    window_exceeded=True,
                ),
            )
            return
    if ctx is None:
        if seq_len <= 1:
            # A lone decode step cannot start a prompt timeline.
            return
        plan = _find_plan(host)
        ctx = _PrimeCtx(
            mtp_cache=host.make_mtp_cache(),
            request_id=plan.request_id if plan is not None else None,
            prompt_tokens=plan.prompt_tokens if plan is not None else None,
            block_size=plan.block_size if plan is not None else 0,
            prefix_cache=plan.prefix_cache if plan is not None else None,
            extra_keys=plan.extra_keys if plan is not None else None,
            extra_key_token_start=(
                plan.extra_key_token_start if plan is not None else None
            ),
            extra_key_ranges=(plan.extra_key_ranges if plan is not None else None),
        )
        if not ctx.mtp_cache:
            return
        setattr(host, _CTX_ATTR, ctx)

    if ctx.pending_hidden is not None:
        if seq_len > 1:
            pairs_hidden = mx.concatenate([ctx.pending_hidden, normed[:, :-1]], axis=1)
        else:
            pairs_hidden = ctx.pending_hidden
        pairs_tokens = inputs
    else:
        if seq_len <= 1:
            ctx.pending_hidden = normed[:, -1:]
            ctx.expected_offset = offset_after
            return
        pairs_hidden = normed[:, :-1]
        pairs_tokens = inputs[:, 1:]

    # Fold through the public mtp_forward so every family's head layout
    # (module, block list, CacheList head caches) is handled by its own
    # model patch. The returned logits are never evaluated — nothing pulls
    # on them, so the lm_head tail costs nothing.
    host.mtp_forward(pairs_hidden, pairs_tokens, ctx.mtp_cache, logits_keep=1)
    ctx.folded += int(pairs_tokens.shape[1])
    ctx.folded_this_request += int(pairs_tokens.shape[1])
    ctx.pending_hidden = normed[:, -1:]
    ctx.expected_offset = offset_after
    _capture_boundary_candidate(
        ctx,
        normed,
        seq_start=offset_after - seq_len,
        seq_end=offset_after,
    )
    # Materialize the head-cache buffers per chunk so the fold graph never
    # accumulates across a long prefill; the (1,1,H) pending row is evaluated
    # alongside so the chunk's full hidden can be freed.
    evals = [ctx.pending_hidden]
    flat = []
    for c in ctx.mtp_cache:
        subs = getattr(c, "caches", None)
        flat.extend(subs if subs else (c,))
    for c in flat:
        keys = getattr(c, "keys", None)
        values = getattr(c, "values", None)
        if keys is not None:
            evals.append(keys)
        if values is not None:
            evals.append(values)
    mx.async_eval(evals)


def take_primed(model, cache, main_tok, *, uid=None, cache_offset=None):
    host, state = _owned(model)
    if uid is None or state is None:
        return _take_primed(model, cache, main_tok, cache_offset=cache_offset)
    record = state.uids.pop(uid, None)
    previous = _slot(host)
    if record is None:
        # Never consume another scheduler request's cursor at activation.
        ctx, plan = previous
        if plan is not None or (ctx is not None and ctx.request_id is not None):
            return None
        return _take_primed(model, cache, main_tok, cache_offset=cache_offset)
    try:
        _restore_slot(host, record)
        return _take_primed(model, cache, main_tok, cache_offset=cache_offset)
    finally:
        _restore_slot(host, previous)


def _take_primed(
    model: Any,
    cache: Optional[List[Any]],
    main_tok: Any,
    *,
    cache_offset=None,
) -> Optional[tuple]:
    """Pop the priming context at MTP activation and finish the seam.

    Called from ``_post_init_mtp`` after its 1-token backbone forward at
    ``main_tok`` (which capture skipped — it runs with return_hidden=True).
    Validates that the context is contiguous up to exactly that forward,
    folds the final (hidden[prompt[-1]], main_tok) pair through the public
    ``mtp_forward`` (adapter/outer-model level), and returns
    ``(mtp_cache, hist_offset)`` — or None, in which case the caller keeps
    the current unprimed behaviour.
    """
    # Hosts with their own priming shape (inkling's sliding-window
    # multi-block fold) own the whole activation seam.
    for host in _host_candidates(model):
        hook = getattr(host, "mtp_take_primed", None)
        if callable(hook):
            primed = hook(cache, main_tok)
            if primed is not None:
                return primed
            # None means the hook declined ownership, not "no priming": the
            # DeepSeek-V4 patch registers ``mtp_take_primed`` on the class
            # but only DSpark builds answer it, so legacy single-head MTP
            # models could never reach the generic seam below and priming
            # was structurally dead for them (#3079). Every hook pops its
            # own context before declining, so the fallthrough cannot adopt
            # a foreign timeline.
            break
    ctx = _find_ctx(model)
    if not isinstance(ctx, _PrimeCtx):
        # No context, or a host-owned one sharing the slot (inkling's) whose
        # hook declined without popping it — not ours to consume.
        return None
    drop_ctx(model)
    if not (ctx.valid and ctx.folded > 0 and ctx.pending_hidden is not None):
        if ctx.deferred_pairs is not None:
            raise RuntimeError("Deferred head history activation seam is invalid")
        return None
    offset = _activation_offset(cache) if cache_offset is None else cache_offset
    if offset is None or ctx.expected_offset != offset - 1:
        if ctx.deferred_pairs is not None:
            raise RuntimeError("Deferred head history activation seam is invalid")
        logger.debug(
            "MTP priming discarded: seam offset mismatch (ctx=%s cache=%s)",
            ctx.expected_offset,
            offset,
        )
        return None
    if ctx.deferred_pairs is not None:
        _flush_deferred_history(model, ctx)
    try:
        model.mtp_forward(
            ctx.pending_hidden,
            main_tok.reshape(1, 1),
            ctx.mtp_cache,
            logits_keep=1,
        )
    except Exception as exc:
        logger.debug("MTP priming discarded: seam fold failed: %s", exc)
        return None
    _publish_boundary_candidate(ctx)
    return ctx.mtp_cache, ctx.folded + 1


def prime_ctx_stats(model: Any) -> Optional[int]:
    """Folded pair count of a live context (introspection / tests)."""
    ctx = _find_ctx(model)
    return ctx.folded if ctx is not None and not ctx.window_exceeded else None


__all__ = [
    "HEAD_HIDDEN_POST_NORM",
    "priming_enabled",
    "prime_window",
    "prepare_prefix_context",
    "suppress_capture",
    "maybe_capture",
    "take_primed",
    "drop_ctx",
    "prime_ctx_stats",
]
