# SPDX-License-Identifier: Apache-2.0
"""Shared speculative-staging primitives.

Factored out of ``speculation.SpeculationState`` so other staged-prefetch
paths (V4.1 span staging, future decoders) adopt the same machinery
instead of growing a third copy:

- ``ewma`` — the recall/precision decay update.
- ``PrevTokenPredictor`` — per-layer last-demand sets plus the EWMA
  recall gate that stops prefetching layers whose routing diverged.
- ``StagedReads`` — the keyed registry of completed speculative rows and
  in-flight set reads (register / resolve-or-join / drop / clear).
- ``stage_headroom`` — the governor-pressure suppress (at floor or inside
  the desperate-free band -> no speculative reads).

Leaf module: imports nothing from omlx, so both the generic streaming
switch and the per-family backings can share it without a cycle.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Tuple
from collections.abc import Callable


def ewma(prev: float, new: float, decay: float) -> float:
    """One EWMA update: ``decay * prev + (1 - decay) * new``."""
    return decay * prev + (1.0 - decay) * new


class PrevTokenPredictor:
    """Prev-token routing predictor with a per-layer recall gate.

    ``record`` stores a layer's last demanded set and folds the previous
    prediction's observed recall (|prev ∩ now| / |now|) into an EWMA;
    ``gate`` is True while that EWMA clears ``min_recall`` — a layer whose
    routing diverges stops earning speculative reads after a bounded
    burst instead of taxing every token's I/O.

    NOT thread-safe: the owning state serializes under its own lock.
    """

    def __init__(self, decay: float = 0.9, min_recall: float = 0.25) -> None:
        self.decay = float(decay)
        self.min_recall = float(min_recall)
        self.prev_uniq_by_layer: dict[int, list[int]] = {}
        self.recall_ewma: dict[int, float] = {}

    def record(self, layer_idx: int, ids) -> list[int] | None:
        """Store *ids* as the layer's last demand; return the previous set.

        The recall observation folds in BEFORE the store so it scores the
        prediction that was live for this call, not the one being
        written.
        """
        li = int(layer_idx)
        now = [int(e) for e in ids]
        prev = self.prev_uniq_by_layer.get(li)
        if prev and now:
            obs = len(set(prev) & set(now)) / max(1, len(now))
            self.recall_ewma[li] = ewma(
                self.recall_ewma.get(li, 0.0), obs, self.decay
            )
        self.prev_uniq_by_layer[li] = now
        return prev

    def predict(self, layer_idx: int) -> list[int]:
        """The stored last demand for *layer_idx* (empty when unseen)."""
        return self.prev_uniq_by_layer.get(int(layer_idx)) or []

    def recall(self, layer_idx: int) -> float:
        """The live recall EWMA for *layer_idx* (0.0 when unseen)."""
        return self.recall_ewma.get(int(layer_idx), 0.0)

    def gate(self, layer_idx: int) -> bool:
        """True while this layer's predictions earn their staged reads."""
        return self.recall(layer_idx) >= self.min_recall

    def clear(self) -> None:
        self.prev_uniq_by_layer.clear()
        self.recall_ewma.clear()


class StagedReads:
    """Keyed registry of speculative rows and in-flight set reads.

    ``rows`` holds completed payloads keyed by the consumer's bundle key;
    ``futs`` maps the same keys to ``(future, ids, key_of)`` while the
    set read is in flight — a demand resolve JOINS the in-flight read
    instead of re-issuing it, and the joined rows fan out to ``rows`` so
    sibling keys resolve without touching the pool again. ``hits`` /
    ``misses`` count resolve outcomes (a ``wait=False`` early exit and a
    never-staged key count neither).

    Internally locked — leaf order: the lock is held for registry
    mutation only, never across a future join, and no method calls back
    into an owner, so the registry may nest under an outer lock safely.
    """

    def __init__(self, max_keys: int) -> None:
        self.max_keys = max(1, int(max_keys))
        self.rows: dict[Any, Any] = {}
        self.futs: dict[Any, tuple[Any, list, Callable[[int], Any]]] = {}
        self.hits = 0
        self.misses = 0
        self.closed = False
        self.lock = threading.Lock()

    def room(self) -> bool:
        """True while another staged key fits under the bound."""
        with self.lock:
            return (
                not self.closed
                and len(self.rows) + len(self.futs) < self.max_keys
            )

    def pending(self, key: Any) -> bool:
        """Non-consuming membership probe (done or in flight)."""
        with self.lock:
            return key in self.rows or key in self.futs

    def keys(self) -> list:
        """Locked snapshot of every staged key (done rows + in-flight).

        For callers that sweep a subset (e.g. drop last token's staged
        set for one layer) — iterating the dicts unlocked would race a
        resolve's fan-out writes.
        """
        with self.lock:
            return list(self.rows) + list(self.futs)

    def register(
        self, fut: Any, ids: list, key_of: Callable[[int], Any]
    ) -> int:
        """Map a submitted set-read future under each id's key.

        Returns the number of NEW keys registered — ids already staged or
        in flight are skipped, and ``max_keys`` bounds the map. On demand,
        ``resolve`` awaits the future once and fans its rows out for the
        siblings.
        """
        registered = 0
        with self.lock:
            if self.closed:
                return 0
            for eid in ids:
                key = key_of(int(eid))
                if key in self.rows or key in self.futs:
                    continue
                if len(self.rows) + len(self.futs) >= self.max_keys:
                    break
                self.futs[key] = (fut, ids, key_of)
                registered += 1
        return registered

    def resolve(self, key: Any, wait: bool = True) -> Any:
        """Return the staged row for *key*, or None.

        Consumed rows are popped — the caller admits them elsewhere (the
        demand touch is the admission signal). An in-flight future is
        awaited once, OUTSIDE the lock so a blocking read never
        serializes the registry behind I/O; its rows then fan out to
        ``rows`` for the siblings of the same set. ``wait=False`` serves
        only completed rows — a non-consuming probe for paths that must
        never park on a just-submitted read.
        """
        with self.lock:
            if self.closed:
                return None
            row = self.rows.pop(key, None)
            if row is not None:
                self.hits += 1
                return row
            entry = self.futs.get(key)
            if entry is not None and not wait and not entry[0].done():
                return None
        if entry is None:
            return None
        fut, ids, key_of = entry
        try:
            got = fut.result()
        except Exception:
            got = None
        rows = got[1] if isinstance(got, tuple) else got
        with self.lock:
            if self.closed:
                # clear() landed mid-join: serve THIS key's row only —
                # fanning out into cleared dicts would resurrect state.
                if rows:
                    for eid, r in zip(ids, rows):
                        if r is not None and key_of(int(eid)) == key:
                            return r
                return None
            if rows:
                for eid, r in zip(ids, rows):
                    if r is None:
                        continue
                    k = key_of(int(eid))
                    self.futs.pop(k, None)
                    self.rows.setdefault(k, r)
            for eid in ids:
                self.futs.pop(key_of(int(eid)), None)
            row = self.rows.pop(key, None)
            if row is not None:
                self.hits += 1
            else:
                self.misses += 1
            return row

    def drop(self, key: Any) -> None:
        """Remove *key*; cancel its read when no sibling references it.

        A set-read future is shared by every id it covers — it is
        cancelled only when no other ``futs`` entry still references it,
        so dropping one key cannot kill a sibling's join.
        """
        with self.lock:
            self.rows.pop(key, None)
            entry = self.futs.pop(key, None)
            fut = entry[0] if entry is not None else None
            if fut is not None and any(
                e[0] is fut for e in self.futs.values()
            ):
                fut = None
        if fut is not None:
            try:
                fut.cancel()
            except Exception:
                pass

    def clear(self) -> list:
        """Mark closed and drop every entry; returns futures to cancel.

        The caller cancels them AFTER releasing its own lock — a Future's
        cancel fires done callbacks on the calling thread, so the
        registry never cancels under a foreign lock.
        """
        with self.lock:
            if self.closed:
                return []
            self.closed = True
            futs = [e[0] for e in self.futs.values()]
            self.rows.clear()
            self.futs.clear()
            return futs

    def close(self) -> None:
        """``clear()`` + cancel — standalone teardown. Idempotent."""
        for fut in self.clear():
            try:
                fut.cancel()
            except Exception:
                pass


def stage_headroom(governor: Any) -> bool:
    """True when speculative staged reads have residency slack.

    Suppresses staging when the shared cache capacity sits at the
    governor's floor or the last observed free memory was inside the
    desperate-clear band — under memory pressure staged rows get dropped
    before they are ever hit. A missing or raising governor (static
    caches) reports slack so staging stays on.
    """
    if governor is None:
        return True
    try:
        at_floor = getattr(governor, "at_floor", None)
        if callable(at_floor) and at_floor():
            return False
        desperate = getattr(governor, "in_desperate_band", None)
        if callable(desperate) and desperate():
            return False
    except Exception:
        return True
    return True
