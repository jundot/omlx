# SPDX-License-Identifier: Apache-2.0
"""Cross-layer speculation state (per conversion).

Owns the routing history, converted-linears registry, advise stats and
the transition table that feed the next-layer F_RDADVISE advisor. Lives
in its own module so ``streaming_switch`` keeps the demand path; nothing
here imports the switch — this is a leaf like ``slot_cache``.

The speculation state is PER CONVERSION (one instance per backing/cache
pair), never module-global. Two engines (different checkpoints, same
tensor keys) can never share routing history or linears registries: the
state dies with the owning store, and close() drains the speculation
workers the same way ExpertBackingStore.close drains its readers.

Transition-table overfetch: the (layer, expert) -> next-token expert
distribution (EWMA, temporal: same layer, token t-1 -> t) feeds the RA
advisor with one extra candidate per demanded expert (k+1 overfetch).
Hints only (F_RDADVISE), never changes output. 0 disables.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Dict, Tuple

from ._env import env_bool, env_int
from .staging import PrevTokenPredictor, StagedReads


_TRANSITION_ENV = env_bool("OMLX_EXPERT_STREAMING_TRANSITION", True)
_TRANSITION_TOP = 8  # entries kept per (layer, expert) source
_TRANSITION_OVERFETCH = 1  # extra candidates per demanded expert

# Staged decode prefetch — real reads into NumPy rows ahead of demand.
# The predictor is prev_uniq_by_layer (previous token's routing for the
# next layer) plus transition-table candidates; a per-layer EWMA recall
# gate stops prefetching layers whose routing diverges, so a bad predictor
# costs a bounded burst and then shuts itself off.
_STAGED_ENV = env_bool("OMLX_EXPERT_STREAMING_STAGED", True)
# Bound on staged rows + in-flight keys (rows are ~0.9 MB each here).
_STAGED_MAX = env_int("OMLX_EXPERT_STREAMING_STAGED_MAX", 512, lo=16)
# Predicted set cap per (layer call, projection).
_STAGED_MAX_IDS = env_int("OMLX_EXPERT_STREAMING_STAGED_IDS", 48, lo=1)
# Per-layer recall gate: stage only when the prev-token prediction has
# covered at least this share of observed demand recently (EWMA).
_STAGED_MIN_RECALL = 0.25
_STAGED_RECALL_DECAY = 0.9

# Advise hygiene: the decode advisory would otherwise re-issue F_RDADVISE
# for nearly the same prev-set every token (~1.6x demand bytes in hints).
# _ADVISE_TTL dedupes per (layer, expert) across a window of layer-calls
# (~2 tokens at 48 layers). _ADVISE_TRANS_MIN_PRECISION gates the
# transition-table extras by their measured precision (the share of last
# token's extras that appeared in this token's demand). 0 disables.
_ADVISE_TTL = 96
_ADVISE_TRANS_MIN_PRECISION = 0.05


class SpeculationState:
    """Per-conversion speculation state.

    Owns the routing history used by the next-layer advisor, the
    converted-linears registry, the advise stats, and the transition
    table. Hangs off ``cache.spec_state``
    (and off ``backing.spec_state`` when the backing is an object) so the
    demand path, the advisor, and backing.close() all share one instance.
    All mutations are lock-guarded. A closed state stops advising so a
    drained engine never serves another conversion's speculation.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.closed = False
        # Per-conversion contract field stamped by conversion from the
        # expert_streaming_readahead setting: None means "no opinion" —
        # the OMLX_EXPERT_STREAMING_RA env default applies.
        self.readahead_enabled: bool | None = None
        # Prev-token prediction + recall gate ride the shared primitive;
        # the public dict attrs stay as aliases into it (external readers
        # — the advisor, the summary — see live state either way).
        self._predictor = PrevTokenPredictor(
            decay=_STAGED_RECALL_DECAY, min_recall=_STAGED_MIN_RECALL
        )
        self.prev_uniq_by_layer: Dict[int, list[int]] = (
            self._predictor.prev_uniq_by_layer
        )
        self.recall_ewma: Dict[int, float] = self._predictor.recall_ewma
        self.linears_by_layer: Dict[int, list[Any]] = {}
        # Counter: bump() and the stage_resolve hit/miss tallies just
        # auto-vivify; the two keys the streaming summary reads
        # (__init__.py: "advised", "trans_overfetch") start visible at 0.
        self.stats: Counter = Counter({"advised": 0, "trans_overfetch": 0})
        # Transition table (layer, expert) -> {next_expert: weight}.
        # Temporal only (same layer, token t-1 -> t); cross-layer same-token
        # transitions are not observable without a model-loop hook.
        # Bounded: TOP entries per source, pruned on write.
        self.trans: Dict[Tuple[int, int], Dict[int, float]] = {}
        self.trans_updates = 0
        # Staged prefetch state. `staged` holds completed NumPy rows
        # keyed by the TARGET linear's bundle_key; `staged_futs` maps the
        # same keys to (future, ids, key_of) while the set read is in
        # flight — demand joins the in-flight read instead of re-issuing
        # it. Both are aliases into the shared StagedReads registry; the
        # registry's own (leaf) lock serializes its mutations.
        self._staged_reads = StagedReads(_STAGED_MAX)
        self.staged: Dict[Any, Any] = self._staged_reads.rows
        self.staged_futs: Dict[Any, Tuple[Any, list, Any]] = (
            self._staged_reads.futs
        )
        # Advise TTL dedup + transition-overfetch precision. The clock
        # advances one step per record_prev call (one MoE layer-call), so
        # TTL units are layer-calls (≈ num_layers per decode token).
        self._advise_seen: Dict[Tuple[int, int], int] = {}
        self._advise_clock = 0
        self._trans_pending: Dict[int, set] = {}
        self._trans_hits = 0
        self._trans_total = 0

    # -- registry / history ----------------------------------------------

    def register_linears(self, layer_idx: int, linears: list[Any]) -> None:
        """Record one MoE layer's converted quantized linears (convert-time)."""
        kept = [lin for lin in linears if lin is not None]
        if not kept:
            return
        with self.lock:
            if self.closed:
                return
            self.linears_by_layer[int(layer_idx)] = kept

    def record_prev(
        self, layer_idx: int, ids: list[int], *, positions: int | None = None
    ) -> None:
        """Remember this layer's routing for the next token's speculation."""
        now = [int(e) for e in ids]
        # Called once per MoE layer-call from StreamingSwitchGLU.__call__ —
        # the projections share the plan's uniq_list, so a per-projection
        # record would triple-count every layer.
        with self.lock:
            if self.closed:
                return
            self._advise_clock += 1
            # Transition-overfetch precision: score last call's extras
            # for this layer against what this call actually demanded.
            pend = self._trans_pending.pop(int(layer_idx), None)
            if pend:
                self._trans_total += len(pend)
                self._trans_hits += len(set(now) & pend)
            # Prev-set store + recall EWMA fold-in (the predictor scores
            # the PREVIOUS prediction against this call before storing).
            prev = self._predictor.record(int(layer_idx), now)
            if _TRANSITION_ENV and prev and now:
                # Credit temporal transitions prev -> now (EWMA without a
                # global decay pass: w = 1 + 0.9*w, normalized on read).
                for ep in prev:
                    row = self.trans.setdefault((int(layer_idx), int(ep)), {})
                    for en in now:
                        row[int(en)] = 1.0 + 0.9 * float(row.get(int(en), 0.0))
                    if len(row) > _TRANSITION_TOP:
                        for k in sorted(row, key=row.get)[: len(row) - _TRANSITION_TOP]:  # type: ignore[arg-type]
                            del row[k]
                self.trans_updates += 1

    def to_payload(self) -> dict:
        """Serialize the transition table (fingerprint filled by caller)."""
        try:
            with self.lock:
                layers: dict[str, dict[str, dict[str, float]]] = {}
                for (layer, expert), row in self.trans.items():
                    layers.setdefault(str(int(layer)), {})[str(int(expert))] = {
                        str(int(k)): float(v) for k, v in row.items()
                    }
                return {"version": 1, "updates": int(self.trans_updates), "trans": layers}
        except Exception:
            return {"version": 1, "updates": 0, "trans": {}}

    def load_payload(self, payload: dict) -> int:
        """Load a transition payload; returns sources restored."""
        try:
            raw = (payload or {}).get("trans") or {}
            n = 0
            with self.lock:
                if self.closed:
                    return 0
                for layer_s, experts in raw.items():
                    for expert_s, row in (experts or {}).items():
                        clean = {
                            int(k): float(v)
                            for k, v in (row or {}).items()
                        }
                        if clean:
                            self.trans[(int(layer_s), int(expert_s))] = dict(
                                sorted(clean.items(), key=lambda kv: kv[1], reverse=True)[:_TRANSITION_TOP]
                            )
                            n += 1
                self.trans_updates += int((payload or {}).get("updates", 0) or 0)
            return n
        except Exception:
            return 0

    def predict_next(self, layer_idx: int, ids: list[int], k: int = _TRANSITION_OVERFETCH) -> list[int]:
        """Top-k next-token candidates for this layer's demand set.

        Scores sum the transition rows of the demanded experts; the demanded
        ids themselves are excluded (the demand path already loads them).
        Returns at most k ids, possibly fewer (cold table). Lock-guarded.
        """
        if not _TRANSITION_ENV or k <= 0 or not ids:
            return []
        try:
            scores: Dict[int, float] = {}
            now = {int(e) for e in ids}
            with self.lock:
                if self.closed:
                    return []
                for e in now:
                    row = self.trans.get((int(layer_idx), int(e)))
                    if not row:
                        continue
                    for cand, w in row.items():
                        if cand not in now:
                            scores[int(cand)] = scores.get(int(cand), 0.0) + float(w)
            ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
            return [int(c) for c in ranked[: max(1, int(k))]]
        except Exception:
            return []

    def bump(self, key: str, amount: int = 1) -> None:
        with self.lock:
            if self.closed:
                return
            self.stats[key] += amount

    # -- staged decode prefetch ---------------------------------------------

    def stage_enabled(self) -> bool:
        return _STAGED_ENV
    def stage_gate(self, layer_idx: int) -> bool:
        """True when this layer's prediction earns its reads — the
        prev-token EWMA gates staging."""
        if not (_STAGED_ENV and not self.is_closed()):
            return False
        li = int(layer_idx)
        with self.lock:
            return self._predictor.gate(li)

    def stage_room(self) -> bool:
        return not self.is_closed() and self._staged_reads.room()

    def stage_pending(self, key: Any) -> bool:
        """True while a staged read covers *key* (done or in flight).

        Non-consuming probe for speculative prefetch membership: unlike
        ``stage_resolve`` it never pops a row or parks on a future, so a
        prefetch can skip experts whose read is already committed without
        racing the demand join.
        """
        if not _STAGED_ENV:
            return False
        return self._staged_reads.pending(key)

    def stage_register(self, fut: Any, ids: list, linear: Any) -> int:
        """Map a submitted set-read future under each expert's bundle_key.

        Returns the number of NEW keys registered (ids already staged or in
        flight are skipped). On demand, ``stage_resolve`` awaits the future
        once and fans its rows out into ``staged`` for the siblings.
        """
        if self.is_closed():
            return 0
        return self._staged_reads.register(fut, ids, linear.bundle_key)

    def stage_resolve(self, key: Any, wait: bool = True) -> Any:
        """Return a staged NumPy row for *key*, or None.

        Consumed rows are popped (the caller admits them to the LRU — the
        demand touch is the admission signal). An in-flight future is
        awaited once; its rows fan out to ``staged`` so sibling experts of
        the same set resolve without touching the pool again.
        ``wait=False`` only serves completed rows — used by the rolling
        prefetch's speculative split, which must never block the
        inference thread on a just-submitted read.
        """
        if not _STAGED_ENV:
            return None
        if self.is_closed():
            return None
        row = self._staged_reads.resolve(key, wait=wait)
        # Mirror the registry's resolve counters into the stats Counter —
        # assignment, not +=, so the mirror is always the true total.
        with self.lock:
            self.stats["staged_hits"] = self._staged_reads.hits
            self.stats["staged_misses"] = self._staged_reads.misses
        return row

    def stage_drop(self, key: Any) -> None:
        """Remove *key* from staging; cancel its in-flight read when held.

        The set-read future is shared by every expert id it covers — it
        is cancelled only when no other staged_futs entry still references
        it, so dropping one key cannot kill a sibling's join.
        """
        self._staged_reads.drop(key)

    def staged_keys(self) -> list:
        """Locked snapshot of every staged key (done rows + in-flight)."""
        return self._staged_reads.keys()

    # -- advise hygiene ----------------------------------------------------

    def advise_fresh(self, layer_idx: int, eids: list) -> list:
        """Drop experts advised within the TTL window; mark the survivors."""
        if _ADVISE_TTL <= 0:
            return list(eids)
        with self.lock:
            if self.closed:
                return list(eids)
            now = self._advise_clock
            keep = []
            for e in eids:
                k = (int(layer_idx), int(e))
                if now - self._advise_seen.get(k, -10**9) >= _ADVISE_TTL:
                    keep.append(int(e))
                    self._advise_seen[k] = now
            if len(self._advise_seen) > 1 << 17:
                self._advise_seen = {
                    k: v
                    for k, v in self._advise_seen.items()
                    if now - v < _ADVISE_TTL
                }
            return keep

    def trans_overfetch_ok(self) -> bool:
        """True while the transition extras earn their hints.

        Precision = share of last call's extras that this call actually
        demanded. Until enough evidence accumulates the extras stay on
        (optimistic); below the floor they stop being added.
        """
        with self.lock:
            if self._trans_total < 512:
                return True
            return (
                self._trans_hits / self._trans_total
                >= _ADVISE_TRANS_MIN_PRECISION
            )

    def note_trans_extras(self, layer_idx: int, extras: list) -> None:
        """Record the extras added to this call's advisory for scoring."""
        if not extras:
            return
        with self.lock:
            if self.closed:
                return
            self._trans_pending[int(layer_idx)] = set(int(e) for e in extras)

    def is_closed(self) -> bool:
        with self.lock:
            return self.closed

    def close(self) -> None:
        """Stop speculation: mark closed, cancel staged reads, clear state.

        Idempotent. In-flight staged futures are collected (closed +
        cleared) under the lock and cancelled outside it — a pending set
        read never starts; a running one finishes into the void, and a
        worker cannot fan rows out into a cleared ``staged`` after close:
        ``StagedReads.resolve``'s join re-checks its own ``closed`` under
        its lock for the same reason. The routing history and registry
        are cleared so a closed state can never serve another
        conversion's speculation.
        """
        with self.lock:
            if self.closed:
                return
            self.closed = True
            # Latch + drop under the registry's leaf lock (nested inside
            # ours — the only spec->staged nesting); cancels run after
            # release so a done callback can never fire under the lock.
            futs = self._staged_reads.clear()
            self.prev_uniq_by_layer.clear()
            self.linears_by_layer.clear()
            self._advise_seen.clear()
            self._trans_pending.clear()
            self.recall_ewma.clear()
        for fut in futs:
            try:
                fut.cancel()
            except Exception:
                pass
