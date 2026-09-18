# SPDX-License-Identifier: Apache-2.0
"""Streaming-backing holder traversal + engine lifecycle helpers.

Everything rooted in finding the ``_expert_streaming_backing`` on a
model/engine wrapper chain lives here, plus the shared load/teardown
sequence (``streaming_offload_load`` -> ``post_load_offload_pipeline`` ->
``teardown_expert_streaming``) and the per-request health summary/log.
Leaf of the expert_streaming package: it imports nothing from the package
root at module level — converter entry points are reached through
deferred ``from . import ...`` lookups inside the functions that need
them, so package attributes (and their monkeypatches) keep resolving at
call time.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def expert_streaming_summary(cache: Any, backing: Any | None = None) -> dict:
    """One-line request/bench summary of streaming health.

    Aggregates the counters the implementation already keeps (LRU hits,
    advisor, ctx fallbacks) into a single
    dict for per-request logs and the admin payload. Never raises;
    missing pieces report as None/0.
    """
    out: dict = {}
    try:
        stats = getattr(cache, "stats", None)
        hits = int(getattr(stats, "hits", 0) or 0)
        misses = int(getattr(stats, "misses", 0) or 0)
        out["lru_hit_rate"] = hits / (hits + misses) if (hits + misses) else 0.0
        out["lru_hits"] = hits
        out["lru_misses"] = misses
        out["lru_evictions"] = int(getattr(stats, "evictions", 0) or 0)
        out["lru_size"] = int(getattr(cache, "size", 0) or 0)
        out["lru_capacity"] = int(getattr(cache, "capacity", 0) or 0)
    except Exception:
        pass
    try:
        # Partitioned backends (V4.1 dedicated adapter, legacy
        # fetch-on-miss) have no app-level LRU cache — the backing's own
        # summary carries the counters, folded into the lru_* slots the
        # request log reads so the line stops reporting a permanent 0.
        if cache is None and backing is not None:
            bsum = getattr(backing, "summary", None)
            if callable(bsum):
                b = bsum()
                if isinstance(b, dict):
                    out["backing"] = b
                    hits = int(b.get("hits", 0) or 0)
                    misses = int(b.get("misses", 0) or 0)
                    out["lru_hit_rate"] = (
                        hits / (hits + misses) if (hits + misses) else 0.0
                    )
                    out["lru_hits"] = hits
                    out["lru_misses"] = misses
                    out["lru_evictions"] = int(b.get("evictions", 0) or 0)
                    out["lru_size"] = int(b.get("resident", 0) or 0)
                    out["lru_capacity"] = int(
                        b.get("capacity_per_layer", 0) or 0
                    ) * int(b.get("layers", 0) or 0)
    except Exception:
        pass
    try:
        out["ctx_fallbacks"] = dict(getattr(cache, "ctx_fallback_stats", lambda: {})())
    except Exception:
        out["ctx_fallbacks"] = {}
    try:
        gov = getattr(backing, "governor", None) if backing is not None else None
        if gov is not None:
            out["governor"] = gov.summary()
    except Exception:
        pass
    try:
        out["cache_policy"] = getattr(cache, "policy", "lru")
    except Exception:
        pass
    try:
        # Speculation state: advisor counter + learned transition table.
        spec = getattr(cache, "spec_state", None)
        if spec is not None:
            sstats = getattr(spec, "stats", None)
            if isinstance(sstats, dict):
                out["advised"] = int(sstats.get("advised", 0) or 0)
                if sstats.get("trans_overfetch"):
                    out["trans_overfetch"] = int(sstats["trans_overfetch"])
            out["trans_updates"] = int(getattr(spec, "trans_updates", 0) or 0)
            out["trans_sources"] = len(getattr(spec, "trans", {}) or {})
    except Exception:
        pass
    return out


# Wrapper hops a ``_expert_streaming_backing`` can hide behind, on either
# side of the engine/model boundary: the model-side descent
# (``language_model``/``model``/``_vlm_model``/``_language_model`` — the
# chain ``scheduler._streaming_backing_of`` walks) plus the engine-side
# holders (``engine``/``_model``/``_vlm_model`` — the chain the old
# ``_engine_holders`` walked flat). One walk covers both spellings so no
# consumer needs a second chain.
#
# The sets stay separate on purpose: a walk that STARTS at a model must
# not hop into ``engine``/``_model`` — a model's ``.engine`` back-ref
# would surface a sibling model's backing (and mock-based callers
# fabricate those attributes).
_MODEL_BACKING_HOPS = (
    "language_model",
    "model",
    "_vlm_model",
    "_language_model",
)
_ENGINE_BACKING_HOPS = ("engine", "_model")
_BACKING_HOPS = _MODEL_BACKING_HOPS + _ENGINE_BACKING_HOPS

# Depth cap bounds adapter property loops (a hop that returns ``self``)
# and pathological wrapper nesting; both legacy walks used 5.
_BACKING_WALK_DEPTH = 5


def _backing_holders(root: Any, hops: tuple[str, ...] = _BACKING_HOPS):
    """Yield *root*, then every object reachable via *hops*.

    Breadth-first so an engine's ``_model`` and ``_vlm_model`` are both
    visited (the flat engine-holder order) before descending into their
    model-side wrappers; a visited set plus the depth cap bound cycles
    and property loops.
    """
    seen: set[int] = set()
    frontier = [root]
    for _ in range(_BACKING_WALK_DEPTH):
        if not frontier:
            return
        nxt: list = []
        for obj in frontier:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            yield obj
            for attr in hops:
                try:
                    child = getattr(obj, attr, None)
                except Exception:
                    continue
                if child is not None:
                    nxt.append(child)
        frontier = nxt


def find_streaming_backing(root: Any) -> Any | None:
    """The ``_expert_streaming_backing`` anywhere on *root*'s wrapper chain.

    Works from either side: pass a model (covers the
    ``language_model``/``model``/``_vlm_model``/``_language_model``
    descent the scheduler's ``_streaming_backing_of`` performs) or an
    engine (covers its ``_model``/``_vlm_model`` holders). First match in
    breadth-first hop order wins.
    """
    for holder in _backing_holders(root):
        backing = getattr(holder, "_expert_streaming_backing", None)
        if backing is not None:
            return backing
    return None


def find_model_streaming_backing(model: Any) -> Any | None:
    """Model-side variant of ``find_streaming_backing``.

    Uses only ``_MODEL_BACKING_HOPS`` — a walk rooted at a model never
    hops into ``engine``/``_model``, where it could surface a different
    model's backing (and mock-based callers fabricate those attributes).
    """
    for holder in _backing_holders(model, _MODEL_BACKING_HOPS):
        backing = getattr(holder, "_expert_streaming_backing", None)
        if backing is not None:
            return backing
    return None


def _engine_holders(engine: Any):
    """Yield the objects that can carry streaming state for an engine:
    the engine itself, then its ``_model`` / ``_vlm_model`` wrappers and
    whatever those wrap (same walk ``find_streaming_backing`` uses)."""
    yield from _backing_holders(engine)


def resolve_streaming_backing(engine: Any) -> Any | None:
    """Return the engine's streaming backing, walking the holder chain.

    The unified converter stamps the backing on engine and model; the
    legacy adapter (and the alias' streaming fallback) stamps its store
    on the model only. Used by the engine stop paths so legacy fds/mmaps
    release on teardown instead of whenever GC gets around to it.
    """
    return find_streaming_backing(engine)


def streaming_summary_backing(engine: Any) -> Any | None:
    """The object owning governor + summary for this engine.

    The legacy adapter aggregates its per-layer caches under a
    governor-facing state stamped as ``_moe_offload_legacy_state`` (same
    duck-type as the V4.1 backing); when present it overrides the plain
    backing for reporting.
    """
    for holder in _engine_holders(engine):
        state = getattr(holder, "_moe_offload_legacy_state", None)
        if state is not None:
            return state
    return resolve_streaming_backing(engine)


def streaming_cache_of(backing: Any) -> Any | None:
    """The app-level expert cache a backing exposes, if any.

    The converter stamps the shared LRU/policy cache on the backing as
    ``_streaming_cache``; keep the private-name knowledge here so the
    scheduler and engine pool stop getattr-mining it.
    """
    return getattr(backing, "_streaming_cache", None)


def log_expert_streaming_summary(
    engine: Any, *, prompt_tokens: int = 0, completion_tokens: int = 0
) -> None:
    """One-line MoE streaming health log per completed request.

    No-op unless expert streaming is active. Also drives the dynamic
    governor's per-request observation. Never raises.
    """
    try:
        backing = streaming_summary_backing(engine)
        if backing is None:
            return
        # Dynamic residency: revisit cache capacity from free memory
        # once per request boundary (opt-in; never raises).
        governor = getattr(backing, "governor", None)
        if governor is not None:
            try:
                action = governor.observe()
            except Exception:
                logger.debug("governor observe failed", exc_info=True)
            else:
                logger.info("expert_streaming governor: %s", action)
        cache = streaming_cache_of(backing)
        summary = expert_streaming_summary(cache, backing)
        if not summary:
            return
        logger.info(
            "expert_streaming req prompt=%d completion=%d lru_hit=%.3f "
            "(h=%d m=%d evict=%d size=%d/%d) advised=%d "
            "ctx_fallbacks=%s",
            prompt_tokens,
            completion_tokens,
            summary.get("lru_hit_rate", 0.0),
            summary.get("lru_hits", 0),
            summary.get("lru_misses", 0),
            summary.get("lru_evictions", 0),
            summary.get("lru_size", 0),
            summary.get("lru_capacity", 0),
            summary.get("advised", 0),
            summary.get("ctx_fallbacks", {}),
        )
    except Exception:
        pass


def ensure_streaming_backing_or_raise(
    model: Any,
    backing: Any | None,
    *,
    requested: bool,
    model_name: str,
) -> None:
    """Fail a lazy load whose streaming conversion produced nothing.

    A model lazy-loaded on the promise of SSD streaming must not reach
    materialize_lazy_state with every expert bank resident — that is the
    silent-OOM path the conversion exists to prevent. Raises when the
    checkpoint structurally supports streaming but neither the unified
    converter nor the legacy adapter claimed any MoE layer. No-op when the
    intent was off, the estimate declines the checkpoint, or real
    conversion happened — backing presence alone is not evidence of
    conversion.
    """
    if not requested:
        return
    if backing is not None and int(getattr(backing, "streaming_converted", 0) or 0) > 0:
        return
    try:
        from .residency import expert_streaming_estimate

        estimate = expert_streaming_estimate(str(model_name))
    except Exception:
        return
    if not estimate.supported:
        return
    # A legacy OffloadSwitchGLU wrap also satisfies the intent (resident
    # fraction by design). Class-name match avoids a circular import.
    try:
        for _, mod in model.named_modules():
            cls_name = mod.__class__.__name__
            if cls_name in ("StreamingSwitchGLU", "OffloadSwitchGLU"):
                return
    except Exception:
        pass
    raise RuntimeError(
        f"Expert streaming: {model_name} supports streaming and was "
        "lazy-loaded for it, but no MoE layer was converted — refusing to "
        "materialize every expert bank in RAM"
    )


async def streaming_offload_load(
    model: Any,
    model_name: str,
    settings: Any,
    *,
    label: str = "model",
) -> tuple[Any | None, int]:
    """The load-time offload sequence shared by the engine paths.

    Streaming conversion -> legacy MoE offload -> zero-conversion guard.
    Must run BEFORE ``materialize_lazy_state`` and gate/up fusion: the
    checkpoint stayed lazy on this path and dropping the stock MoE modules
    is what keeps non-resident experts from ever materializing. All MLX
    work runs on the MLX executor because it allocates the resident slot
    tensors (#1304).

    Returns ``(backing, wrapped)`` — the streaming backing for the caller
    to keep alive, and the legacy-offload wrapped-layer count (0 when the
    unified path served or offload was not requested). The caller stamps
    ``backing`` onto the engine itself.
    """
    import asyncio

    from ...engine_core import get_mlx_executor
    from ...model_settings import moe_offload_requested

    loop = asyncio.get_running_loop()
    offload_requested = moe_offload_requested(settings)
    backing = None
    if getattr(settings, "expert_streaming_enabled", False):
        try:

            def _do_streaming():
                # Resolved through the package namespace so monkeypatches
                # of the re-exported converter keep working.
                from . import convert_model_to_streaming

                _, b = convert_model_to_streaming(model, model_name, settings)
                # keep backing alive on the model
                if b is not None:
                    model._expert_streaming_backing = b  # type: ignore[attr-defined]
                return b

            backing = await loop.run_in_executor(get_mlx_executor(), _do_streaming)
            logger.info("Expert streaming enabled for %s %s", label, model_name)
        except Exception as e:
            # Fail clean: streaming was explicitly enabled, so a backing
            # failure must fail the load — continuing to materialize
            # would retain all expert banks in RAM (OOM).
            logger.error(
                "Expert streaming conversion failed for %s %s: %s",
                label,
                model_name,
                e,
                exc_info=True,
            )
            raise RuntimeError(
                f"Expert streaming conversion failed for {label} {model_name}: {e}"
            ) from e

    # MoE expert offload: replace covered SwitchGLU layers with a
    # fetch-on-miss LRU cache streaming experts from the checkpoint's
    # own safetensors. The caches' slot maps and resident slots live on
    # plain attributes outside the module tree, so the lazy-state
    # materialization never reaches them; left lazy they stay bound to
    # the loader stream and the first request from an inference thread
    # dies with "There is no Stream(gpu, N) in current thread" — the
    # post-apply materialize fixes that on the same executor.
    wrapped = 0
    if offload_requested:
        from ..moe_expert_offload import (
            apply_moe_expert_offload,
            materialize_offload_state,
        )

        fraction = float(
            getattr(settings, "moe_expert_offload_resident_fraction", 0.25)
        )
        wrapped = await loop.run_in_executor(
            get_mlx_executor(),
            apply_moe_expert_offload,
            model,
            model_name,
            fraction,
            settings,
        )
        if wrapped:
            await loop.run_in_executor(
                get_mlx_executor(), materialize_offload_state, model
            )

    # Fail before materializing: a lazy-loaded streaming checkpoint with
    # zero converted layers would evaluate every expert bank in RAM.
    if offload_requested:
        ensure_streaming_backing_or_raise(
            model, backing, requested=True, model_name=model_name
        )
    return backing, wrapped


async def post_load_offload_pipeline(
    model: Any,
    model_name: str,
    settings: Any,
    *,
    label: str = "model",
    holder: Any | None = None,
) -> tuple[Any | None, int]:
    """The post-load offload sequence every engine path repeats.

    ``streaming_offload_load`` (conversion / legacy wrap / the
    zero-conversion guard) then ``materialize_lazy_state`` — all MLX work
    on the MLX executor. Conversion must precede materialize so the
    dropped expert banks never evaluate (OOM); materialize must precede
    gate/up fusion, which ``gate_up_fusion_blocked`` gates for the
    caller. When *holder* is given (engines pass ``self``) the streaming
    backing is stamped on it — the model-side stamp already happened
    inside ``streaming_offload_load``.

    Returns ``(backing, wrapped)`` exactly like ``streaming_offload_load``
    so callers keep their fusion-skip logging context.
    """
    import asyncio

    from ...engine_core import get_mlx_executor
    from ...utils.model_loading import materialize_lazy_state

    loop = asyncio.get_running_loop()
    backing, wrapped = await streaming_offload_load(
        model, model_name, settings, label=label
    )
    if backing is not None and holder is not None:
        try:
            holder._expert_streaming_backing = backing  # type: ignore[attr-defined]
        except Exception:
            pass
    # Materialize lazy buffers on the loader thread so per-engine
    # inference threads can read them (#1304). Post-conversion the MoE
    # banks are gone, so this stays bounded.
    await loop.run_in_executor(get_mlx_executor(), materialize_lazy_state, model)
    return backing, wrapped


def gate_up_fusion_blocked(settings: Any, wrapped: int) -> bool:
    """True when the post-load MoE gate/up fusion must be skipped.

    Fusion concatenates the stock SwitchGLU gate/up weights in place, so
    it cannot run when either offload system is active: legacy-wrapped
    layers (``wrapped``) were never materialized and are not stock
    SwitchGLU anyway, and a streaming conversion already owns the
    projection layout. An explicit ``moe_gate_up_fusion_enabled=False``
    also blocks it. Mirrors the predicate both engines spell inline.
    """
    if wrapped:
        return True
    return not (
        getattr(settings, "moe_gate_up_fusion_enabled", True) is not False
        and not getattr(settings, "expert_streaming_enabled", False)
    )


def shutdown_expert_streaming(backing: Any) -> None:
    """Release MoE streaming resources held by *backing*.

    Persists the transition table, then closes shard fds/mmaps.
    Idempotent; safe to call with None or a RAM-dict backing. Engines
    call this in stop() and before replacing the model on reload so
    threads/fds never leak across model lifetimes.
    """
    if backing is None or isinstance(backing, dict):
        return
    # Persist the learned transition table before threads/fds die.
    try:
        from . import save_transition_profile

        save_transition_profile(backing)
    except Exception:
        pass
    try:
        close = getattr(backing, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def save_expert_pin_profile(engine: Any) -> None:
    """Persist the learned pin profile of a streaming engine, if any.

    Called from the engine ``stop()`` paths while the backing store (and the
    PinController attached to it) is still reachable — before teardown drops
    the references. Never raises: a failed save only costs the learned hot
    set, never correctness.
    """
    for holder in _engine_holders(engine):
        backing = getattr(holder, "_expert_streaming_backing", None)
        pinner = getattr(backing, "_pin_controller", None)
        if pinner is not None:
            try:
                pinner.save_profile()
            except Exception:
                logger.debug("Expert streaming: pin profile save failed", exc_info=True)
            return


def teardown_expert_streaming(engine: Any) -> None:
    """Engine-stop teardown shared by the batched and VLM wrappers.

    Persists the learned pin profile while the backing is still
    reachable, then shuts the backing down (transition profile +
    fds/mmaps) and clears the engine-side reference so a later
    ``resolve_streaming_backing`` cannot hand back a closed store. Each
    piece is already never-raise.
    """
    save_expert_pin_profile(engine)
    shutdown_expert_streaming(resolve_streaming_backing(engine))
    try:
        engine._expert_streaming_backing = None  # type: ignore[attr-defined]
    except Exception:
        pass
