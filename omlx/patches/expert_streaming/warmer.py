# SPDX-License-Identifier: Apache-2.0
"""F_RDADVISE readahead and mlock pinning for streamed experts.

Both mechanisms exploit the page-cache-only streaming default (no LRU):
the OS file cache holds recently read expert pages, and reuse is served
from RAM at memory bandwidth instead of the NVMe.

PageCacheWarmer
    During decode, right before a MoE layer loads its experts, submit
    F_RDADVISE kernel readahead hints for the PREVIOUS token's experts of
    the NEXT layer. Independent per-layer routing repeats ~35% of experts
    across adjacent tokens (measured on FlashNext-class checkpoints); the
    kernel prefetches those pages so the next layer's demand reads hit
    RAM. Hints only — nothing is stored, no heap, no LRU, no userspace
    copy.

PinController
    Observe routed experts for the first N decode calls, then mlock the
    file-cache pages of the most frequent experts per layer within a byte
    budget. Locked pages are the file pages themselves (zero-copy) but
    become wired memory — they cannot be evicted. This substitutes a hot
    set for the LRU at a fraction of the accounting cost.

Readahead and seeding default on (OMLX_EXPERT_STREAMING_RA/_SEED=0
disables); pinning is opt-in via _PIN=1. All are decode-only (gated on
routing row count).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Decode rows are top_k * batch (8 * B); prefill chunks are much larger.
# A small prompt (<= this many rows) may warm needlessly once — bounded waste.
_MAX_WARM_ROWS = 64

PIN_ENABLED = os.environ.get("OMLX_EXPERT_STREAMING_PIN", "0") == "1"
# F_RDADVISE readahead (Fase G): same prediction flow as the read-warmer,
# but the submitted jobs are kernel readahead hints instead of discarded
# reads — no userspace copy, near-zero cost, so it defaults ON (disable
# with OMLX_EXPERT_STREAMING_RA=0).
RA_ENABLED = os.environ.get("OMLX_EXPERT_STREAMING_RA", "1") != "0"
# Prefill-hotness seeding (Fase G): after a streaming prefill, replace the
# expert cache contents with the prompt's hot experts (ds4's cache seeding).
SEED_ENABLED = os.environ.get("OMLX_EXPERT_STREAMING_SEED", "1") != "0"
SEED_BYTES = int(
    float(os.environ.get("OMLX_EXPERT_STREAMING_SEED_GIB", "2.0")) * 1024**3
)
# Fase L: start the pin budget small (256 MiB) — the L2 matrix tests
# 256 MiB / 512 MiB / 1.25 GiB before any default is promoted.
PIN_BUDGET_BYTES = int(
    float(os.environ.get("OMLX_EXPERT_STREAMING_PIN_GIB", "0.25")) * 1024**3
)
PIN_OBSERVE_CALLS = max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_PIN_TOKENS", "8")))
# Learned pin store (colibri-style): persist observed per-layer frequencies
# to this JSON and reload them on the next load, skipping the observation
# window so the hot set is wired from token 1.
PIN_PROFILE_PATH = os.environ.get("OMLX_EXPERT_STREAMING_PIN_PROFILE", "") or None
# Fase I6: the profile feeds the HOBBIT top-fraction split, so it must cover
# the model's full expert width (GLM: 288 experts/layer) — 64 truncated it
# and made the fraction's denominator meaningless. Env-tunable, JSON format
# unchanged (a list of [expert, count] pairs per layer).
_PIN_PROFILE_KEEP = max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_PIN_KEEP", "512")))

# Fase L: profile format version. v2 splits the learned frequencies into
# per-regime counters (decode vs prefill) and refuses to apply a profile
# whose model fingerprint does not match the loaded model. v1 (merged freq)
# migrates to the decode regime on load. Both regimes also persist the
# legacy top-level "freq" (= decode regime) so older consumers keep working.
PROFILE_VERSION = 2
# Fase L: which regime drives the pin selection. Env override for the bench
# matrix (arm E: prefill profile applied to decode).
PIN_REGIME = os.environ.get("OMLX_EXPERT_STREAMING_PIN_REGIME", "decode")
# Fase L: pin synchronously at engine load when 1. Bench arms need the
# mlock pass finished before the first request; the server default stays
# async (pins must never delay the request path).
PIN_SYNC_ENABLED = os.environ.get("OMLX_EXPERT_STREAMING_PIN_SYNC", "") == "1"

_PAGE_SIZE = os.sysconf("SC_PAGESIZE") if hasattr(os, "sysconf") else 4096

_WARM_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="omlx-expert-warm")


def _proj_keys(linear: Any) -> list[str]:
    keys = []
    w = getattr(linear, "stacked_weight_key", None) or getattr(linear, "stacked_key", None)
    if w:
        keys.append(w)
    s = getattr(linear, "stacked_scales_key", None)
    if s and hasattr(linear.backing, "load_expert_slice"):
        keys.append(s)
    b = getattr(linear, "stacked_biases_key", None)
    if b:
        keys.append(b)
    return keys


class PageCacheWarmer:
    """F_RDADVISE kernel readahead for the previous token's next-layer
    experts, grouped per contiguous expert run: same prediction flow as
    the removed warm-only discarded-read arm, zero data copied into
    userspace.
    """

    def __init__(self, linears_by_layer: Dict[int, list]):
        self.linears_by_layer = linears_by_layer
        self.keys_by_layer: Dict[int, dict[int, list[str]]] = {
            layer: {id(lin): _proj_keys(lin) for lin in linears}
            for layer, linears in linears_by_layer.items()
        }
        self.last_uniq: Dict[int, list[int]] = {}
        self.advised = 0
        self.advised_bytes = 0
        self.advise_failures = 0

    def on_layer_start(self, layer_idx: int, positions: int) -> None:
        """Fire the next layer's previous-token readahead hints before this
        layer's demand loads (maximum overlap with GPU compute)."""
        if positions > _MAX_WARM_ROWS:
            return
        nxt = layer_idx + 1
        prev = self.last_uniq.get(nxt)
        if prev:
            linears = self.linears_by_layer.get(nxt)
            if linears:
                self._submit(prev, linears, nxt)

    def on_layer_plan(self, layer_idx: int, uniq_list: list[int], positions: int) -> None:
        """Record this token's expert set for the next token's advisory."""
        self.last_uniq[layer_idx] = [] if positions > _MAX_WARM_ROWS else list(uniq_list)

    def _submit(self, eids: list[int], linears: list, layer_idx: int) -> None:
        backing = getattr(linears[0], "backing", None)
        if backing is None:
            return
        if not hasattr(backing, "advise_expert_run"):
            return
        jobs = [
            (key, eids)
            for lin in linears
            for key in self.keys_by_layer.get(layer_idx, {}).get(id(lin), [])
        ]
        if not jobs:
            return

        def _run():
            for key, ids in jobs:
                # ids come from np.unique (ascending): group contiguous runs.
                start = None
                prev_id = None
                for eid in ids:
                    if start is None:
                        start = prev_id = eid
                        continue
                    if eid == prev_id + 1:
                        prev_id = eid
                        continue
                    self._advise_one(backing, key, start, prev_id - start + 1)
                    start = prev_id = eid
                if start is not None:
                    self._advise_one(backing, key, start, prev_id - start + 1)

        _WARM_POOL.submit(_run)

    def _advise_one(self, backing: Any, key: str, first_id: int, count: int) -> None:
        try:
            if backing.advise_expert_run(key, first_id, count):
                self.advised += 1
            else:
                self.advise_failures += 1
        except Exception:
            self.advise_failures += 1


class PinController:
    """Observe routing per regime, then mlock the hot experts per layer.

    Fase L: the learned profile is version 2 and regime-split — decode rows
    (<= _MAX_WARM_ROWS positions, the union fast-path shape) accrue under
    regimes["decode"] and prefill rows under regimes["prefill"].
    The pin selection reads one regime (pin_regime, default decode), the
    budget is distributed proportionally to each layer's usage mass with a
    minimum of one expert per layer, and the unique page ranges of the
    chosen experts are deduped before the budget is enforced.
    """

    def __init__(
        self,
        linears_by_layer: Dict[int, list],
        backing: Any,
        *,
        budget_bytes: int = PIN_BUDGET_BYTES,
        observe_calls: int = PIN_OBSERVE_CALLS,
        per_expert_bytes: int = 0,
        profile_path: str | None = None,
        num_experts: int = 0,
        model_fingerprint: Dict[str, Any] | None = None,
        packing: str | None = None,
        pin_regime: str | None = None,
        pin_sync: bool | None = None,
    ):
        self.linears_by_layer = linears_by_layer
        self.backing = backing
        self.budget_bytes = budget_bytes
        self.observe_calls = observe_calls
        self.per_expert_bytes = per_expert_bytes
        # Fase I6: expert width of the routed layers — lets on_layer_plan
        # size per-token bincounts (and validate counts payloads) without
        # another call. 0 = unknown (tests / legacy wiring); the counts
        # payload then defines its own width.
        self.num_experts = int(num_experts)
        # Fase L: per-regime learned frequencies. decode = routing calls at
        # or below _MAX_WARM_ROWS positions; prefill = larger calls.
        self.regimes: Dict[str, Dict[int, Counter]] = {"decode": {}, "prefill": {}}
        # Fase M1: explicit wiring wins; the env constant is the fallback
        # when the caller passes None (server defaults, tests).
        if pin_regime is None:
            pin_regime = PIN_REGIME
        if pin_sync is None:
            pin_sync = PIN_SYNC_ENABLED
        self.pin_regime = pin_regime if pin_regime in ("decode", "prefill") else "decode"
        self.pin_sync = bool(pin_sync)
        self.model_fingerprint = (dict(model_fingerprint) if model_fingerprint else None)
        self.packing = packing
        # None = no fingerprint to verify; True/False after a v2 load.
        self.fingerprint_match: bool | None = None
        self.profile_regime = self.pin_regime
        self.calls = 0
        self.pinned = False
        self.pin_jobs = 0
        self.pin_load_time_ms = 0.0
        self.pinned_pages_estimate = 0
        # Server wiring passes a per-model path (<model>/.omlx/...); the env
        # path stays the explicit bench/override opt-in and wins when set.
        self.profile_path = PIN_PROFILE_PATH or profile_path
        if self.profile_path and self._load_profile():
            # Learned hot set available: pin immediately, no observation.
            # Sync only when the effective wiring says so — the server
            # default keeps pins off the request path.
            self._pin_all(sync=self.pin_sync)
        # Fase M1: truthful load-time flags for the bench JSON — a profile
        # was loaded at engine load, and the wired pins were applied before
        # the first request when the effective sync is on.
        self.pins_applied_at_load = self.pinned

    @property
    def freq(self) -> Dict[int, Counter]:
        """The active regime's frequencies (legacy name kept for tests)."""
        return self.regimes[self.pin_regime]

    @freq.setter
    def freq(self, value: Dict[int, Counter]) -> None:
        self.regimes[self.pin_regime] = value

    def _load_profile(self) -> bool:
        try:
            import json

            data = json.loads(open(self.profile_path).read())
            version = int(data.get("version", 1))
            if version >= 2:
                fp = data.get("model_fingerprint")
                if self.model_fingerprint and fp != self.model_fingerprint:
                    logger.warning(
                        "Expert streaming: pin profile %s fingerprint mismatch "
                        "(loaded %s, model %s) — profile ignored",
                        self.profile_path,
                        fp,
                        self.model_fingerprint,
                    )
                    self.fingerprint_match = False
                    return False
                self.fingerprint_match = True
                regimes = data.get("regimes") or {}
                for regime in ("decode", "prefill"):
                    freq = (regimes.get(regime) or {}).get("freq") or {}
                    self.regimes[regime] = {
                        int(layer): Counter({int(e): int(c) for e, c in pairs})
                        for layer, pairs in freq.items()
                    }
                self.packing = data.get("packing") or self.packing
            else:
                # v1 migration (documented): the merged freq was decode-driven
                # (the pin pass fired only from decode rows), so it becomes
                # the decode regime; prefill starts empty.
                freq = data.get("freq") or {}
                if not freq:
                    return False
                self.regimes["decode"] = {
                    int(layer): Counter({int(e): int(c) for e, c in pairs})
                    for layer, pairs in freq.items()
                }
                self.regimes["prefill"] = {}
                self.packing = data.get("packing") or self.packing
            if data.get("per_expert_bytes"):
                self.per_expert_bytes = int(data["per_expert_bytes"])
            logger.info(
                "Expert streaming: loaded learned pin profile (%d layers, "
                "regime %s) from %s",
                len(self.regimes[self.pin_regime]),
                self.pin_regime,
                self.profile_path,
            )
            return True
        except Exception as e:
            logger.debug("Failed to load pin profile %s: %s", self.profile_path, e)
            return False

    def save_profile(self) -> None:
        if not self.profile_path:
            return
        try:
            import json
            from pathlib import Path

            if not (self.regimes["decode"] or self.regimes["prefill"]):
                return
            Path(self.profile_path).parent.mkdir(parents=True, exist_ok=True)
            # Plan format: regimes.<regime>.freq.<layer> -> [[expert, count]].
            regimes = {
                regime: {
                    "freq": {
                        str(layer): counter.most_common(_PIN_PROFILE_KEEP)
                        for layer, counter in sorted(counters.items())
                    }
                }
                for regime, counters in self.regimes.items()
            }
            data = {
                "version": PROFILE_VERSION,
                "model_fingerprint": self.model_fingerprint,
                "packing": self.packing,
                "per_expert_bytes": self.per_expert_bytes,
                "regimes": regimes,
                # Legacy top-level freq = the decode regime, so v1 consumers
                # (older tooling reading only "freq") keep working.
                "freq": regimes["decode"].get("freq", {})
                if isinstance(regimes["decode"], dict)
                else {},
            }
            tmp = self.profile_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.profile_path)
            logger.info("Expert streaming: saved pin profile to %s", self.profile_path)
        except Exception as e:
            logger.debug("Failed to save pin profile %s: %s", self.profile_path, e)

    @staticmethod
    def _counts_payload(counts, num_experts: int) -> Dict[int, int] | None:
        # Coerce a per-token usage histogram into {expert: count}. counts
        # is either the np.bincount(plan.flat_np, minlength=E) vector from
        # the streaming switch (Fase I6's hotness signal: usage per token,
        # not presence per plan) or a plain mapping expert -> usage. None
        # when the payload is unusable so callers fall back to the
        # presence-based uniq_list signal (old wiring / tests).
        if counts is None:
            return None
        try:
            if hasattr(counts, "tolist"):  # np.ndarray / mx.array-like
                counts = counts.tolist()
            if isinstance(counts, dict):
                return {int(e): int(c) for e, c in counts.items()}
            if isinstance(counts, (list, tuple)):
                if num_experts > 0 and len(counts) > num_experts:
                    counts = counts[:num_experts]
                return {i: int(c) for i, c in enumerate(counts) if int(c) > 0}
        except Exception:
            return None
        return None

    def on_layer_plan(
        self,
        layer_idx: int,
        uniq_list: list[int],
        positions: int,
        counts=None,
    ) -> None:
        # Fase L: keep recording the regime this call belongs to even after
        # the pin pass, so the profile refreshes during the session and the
        # stop() save reflects the latest usage. The mlock pass itself still
        # fires only from the decode observe window or at load time.
        usage = self._counts_payload(counts, self.num_experts)
        regime = "decode" if positions <= _MAX_WARM_ROWS else "prefill"
        if usage is not None:
            self.regimes[regime].setdefault(layer_idx, Counter()).update(usage)
        else:
            self.regimes[regime].setdefault(layer_idx, Counter()).update(
                int(e) for e in uniq_list
            )
        if regime != "decode" or self.pinned:
            return
        # One decode token = one plan per layer; pin after the window.
        self.calls += 1
        if self.calls >= self.observe_calls * max(len(self.linears_by_layer), 1):
            self._pin_all()

    def _plan_page_ranges(self, jobs: list[tuple[str, int]]) -> int:
        """Unique page count across the chosen experts (page-aligned dedupe).

        Experts are contiguous in the stacked bank, so consecutive ids share
        the boundary page; the budget must count unique pages only.
        """
        intervals: Dict[tuple[str, str], list[tuple[int, int]]] = {}
        for key, eid in jobs:
            try:
                reader = self.backing._reader_for_key(key, int(eid))
                off, end = reader.expert_byte_range(key, int(eid))
            except Exception:
                continue
            sp = off // _PAGE_SIZE
            ep = (end + _PAGE_SIZE - 1) // _PAGE_SIZE
            if ep > sp:
                intervals.setdefault((str(reader.path), key), []).append((sp, ep))
        total = 0
        for ivs in intervals.values():
            ivs.sort()
            cur_s, cur_e = ivs[0]
            for s, e in ivs[1:]:
                if s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    total += cur_e - cur_s
                    cur_s, cur_e = s, e
            total += cur_e - cur_s
        return total

    def _pin_all(self, sync: bool = False) -> None:
        self.pinned = True
        per_expert = self.per_expert_bytes
        if per_expert <= 0:
            per_expert = max(
                (getattr(l, "_per_expert_hint", 0) for ls in self.linears_by_layer.values() for l in ls),
                default=0,
            )
        freq = self.regimes[self.pin_regime]
        if not freq or per_expert <= 0:
            return
        budget = self.budget_bytes
        # Fase L: per-layer budget proportional to usage mass, minimum one
        # expert per layer with a valid frequency.
        total_mass = sum(c.total() for c in freq.values()) or 1
        ranked: list[tuple[float, str, int]] = []
        for layer_idx, counter in freq.items():
            if counter.total() <= 0:
                continue
            layer_budget = budget * counter.total() / total_mass
            n = max(1, int(layer_budget // per_expert))
            linears = self.linears_by_layer.get(layer_idx) or []
            keys = [k for lin in linears for k in _proj_keys(lin)]
            if not keys:
                continue
            for eid, count in counter.most_common(n):
                for key in keys:
                    ranked.append((float(count), key, int(eid)))

        def _run():
            t0 = time.perf_counter()
            jobs = sorted(set(ranked), key=lambda j: -j[0])
            total_pages = self._plan_page_ranges([(k, e) for _, k, e in jobs])
            # Trim least-hot until the unique pinned pages fit the budget.
            while total_pages * _PAGE_SIZE > budget and len(jobs) > 1:
                jobs.pop()
                total_pages = self._plan_page_ranges([(k, e) for _, k, e in jobs])
            self.pinned_pages_estimate = total_pages
            for _count, key, eid in jobs:
                try:
                    self.backing.pin_expert(key, eid)
                except Exception:
                    pass
            self.pin_jobs = len(jobs)
            self.pin_load_time_ms = (time.perf_counter() - t0) * 1000.0
            self.save_profile()
            logger.info(
                "Expert streaming: pinned %d expert slices (%.2f GiB unique "
                "pages, regime %s) in %.1fms",
                self.backing.pinned_count,
                (self.pinned_pages_estimate * _PAGE_SIZE) / 1024**3,
                self.pin_regime,
                self.pin_load_time_ms,
            )

        if sync:
            _run()
        else:
            _WARM_POOL.submit(_run)


def _infer_per_expert_bytes(linears_by_layer: Dict[int, list], backing: Any) -> int:
    """Per-expert byte size from the first stacked weight key's header."""
    for linears in linears_by_layer.values():
        for lin in linears:
            key = getattr(lin, "stacked_weight_key", None)
            if key and hasattr(backing, "expert_bytes"):
                try:
                    n = int(backing.expert_bytes(key))
                    if n > 0:
                        return n
                except Exception:
                    pass
    return 0


class PrefillHotnessRecorder:
    """Seed the expert caches from prefill routing hotness (Fase G).

    The prefill demand path fills the LRU with whatever the *last* chunks
    read, so decode starts with a nearly useless cache (F2: hit_rate 0.002
    at budget 4 GiB). This recorder accumulates per-layer expert frequency
    over the prefill, then — on the first decode-sized call — swaps the
    cache to the prompt's hot set: LRU retain + missing-hot loads when a
    budget exists, a bounded page-cache seed burst otherwise (budget 0 =
    page-cache-only default). One-shot per prefill.
    """

    def __init__(
        self,
        linears_by_layer: Dict[int, list],
        backing: Any,
        cache: Any = None,
        *,
        per_expert_bytes: int = 0,
        seed_bytes: int = SEED_BYTES,
    ):
        self.linears_by_layer = linears_by_layer
        self.keys_by_layer: Dict[int, dict[int, list[str]]] = {
            layer: {id(lin): _proj_keys(lin) for lin in linears}
            for layer, linears in linears_by_layer.items()
        }
        self.backing = backing
        self.cache = cache
        if self.cache is not None and getattr(self.cache, "capacity", 0) > 0:
            # C4: avoid filling the LRU with the final prefill chunk; C5's
            # hotness seed repopulates it after routing frequencies are known.
            setattr(self.cache, "prefill_bypass", True)
        self.per_expert_bytes = per_expert_bytes
        self.seed_bytes = seed_bytes
        self.freq: Dict[int, Counter] = {}
        self.saw_prefill = False
        self.seeded = False
        self.seeded_experts = 0
        self.seeded_s = 0.0
        self.seed_done = threading.Event()

    def on_layer_plan(
        self,
        layer_idx: int,
        uniq_list: list[int],
        positions: int,
        counts=None,
    ) -> None:
        if self.seeded:
            return
        if positions > _MAX_WARM_ROWS:
            # Prefill-sized call: accumulate frequency (decode rows are
            # top_k * batch and would bias toward the first token). With a
            # per-token counts payload (I6) the accumulation is true usage;
            # the uniq_list fallback is presence-per-plan (legacy wiring).
            self.saw_prefill = True
            usage = PinController._counts_payload(counts, 0)
            if usage is not None:
                self.freq.setdefault(layer_idx, Counter()).update(usage)
            else:
                self.freq.setdefault(layer_idx, Counter()).update(
                    int(e) for e in uniq_list
                )

    def maybe_seed(self, layer_idx: int, positions: int) -> None:
        """Fire once, at the first decode-sized call after a prefill."""
        if self.seeded or not self.saw_prefill or positions > _MAX_WARM_ROWS:
            return
        self.seeded = True
        if self.cache is not None:
            setattr(self.cache, "prefill_bypass", False)
        if not self.freq:
            self.seed_done.set()
            return
        t0 = time.perf_counter()
        try:
            if self.cache is not None and self.cache.capacity > 0:
                n = self._seed_lru()
            else:
                n = self._seed_page_cache()
        except Exception as e:
            logger.debug("Expert streaming: hotness seed failed: %s", e)
            return
        self.seeded_experts = n
        self.seeded_s = time.perf_counter() - t0
        logger.info(
            "Expert streaming: seeded %d hot expert slices from prefill "
            "routing in %.2fs",
            n,
            self.seeded_s,
        )

    def _hot_top(self, experts_per_layer: int) -> Dict[int, list[int]]:
        return {
            layer: [e for e, _ in counter.most_common(experts_per_layer)]
            for layer, counter in self.freq.items()
        }

    def _seed_lru(self) -> int:
        per_layer_cap = getattr(self.cache, "_per_layer_cap", 0) or 0
        if per_layer_cap <= 0:
            return 0
        # Retain the known hot entries synchronously so the first decode call
        # never evicts useful prompt-wide entries. Missing bundles are read on
        # the warm pool; the C3 cache lock makes worker-side raw bundle puts
        # safe, and quantized linears promote them on the inference thread.
        hot = self._hot_top(max(1, per_layer_cap // 3))
        hot_pairs = {(layer, eid) for layer, eids in hot.items() for eid in eids}
        retain = getattr(self.cache, "retain_hot", None)
        if callable(retain):
            retain(hot_pairs)
        jobs: list[tuple[int, Any, Any, int]] = []
        ready = 0
        for layer, eids in hot.items():
            for lin in self.linears_by_layer.get(layer) or []:
                loader = getattr(lin, "_load_expert_bundle", None)
                for eid in eids:
                    key = (layer, eid, getattr(lin, "stacked_weight_key", None))
                    if self.cache.get(key) is not None:
                        ready += 1
                    elif loader is not None:
                        jobs.append((layer, lin, loader, eid))
        if not jobs:
            self.seed_done.set()
            return ready

        def _run() -> None:
            warmed = 0
            try:
                for layer, lin, loader, eid in jobs:
                    try:
                        # Quantized streamers expose raw slice keys. Read plain
                        # NumPy buffers off-thread; never allocate MLX arrays here.
                        weight_key = getattr(lin, "stacked_weight_key", None)
                        scales_key = getattr(lin, "stacked_scales_key", None)
                        if (
                            weight_key
                            and scales_key
                            and hasattr(self.backing, "load_expert_slice")
                        ):
                            w = self.backing.load_expert_slice(weight_key, eid)
                            s = self.backing.load_expert_slice(scales_key, eid)
                            b = None
                            bias_key = getattr(lin, "stacked_biases_key", None)
                            if bias_key:
                                try:
                                    b = self.backing.load_expert_slice(bias_key, eid)
                                except Exception:
                                    b = None
                            self.cache.put((layer, eid, weight_key), (w, s, b))
                        else:
                            # Non-quantized test/backing path: retain correctness
                            # with its existing loader contract.
                            loader(eid)
                        warmed += 1
                    except Exception:
                        continue
            finally:
                self.seed_done.set()
            logger.debug(
                "Expert streaming: async LRU seed warmed %d/%d bundles",
                warmed,
                len(jobs),
            )

        _WARM_POOL.submit(_run)
        return ready + len(jobs)

    def _seed_page_cache(self) -> int:
        """Budget-0: discarded reads of the hot set into the page cache.

        Async on the warm pool (no LRU writes from worker threads), capped
        at seed_bytes across the whole model.
        """
        num_layers = max(len(self.freq), 1)
        per_expert = self.per_expert_bytes
        if per_expert <= 0:
            per_expert = _infer_per_expert_bytes(self.linears_by_layer, self.backing)
        if per_expert <= 0:
            return 0
        experts_per_layer = max(1, min(64, self.seed_bytes // (num_layers * per_expert)))
        hot = self._hot_top(experts_per_layer)

        def _run():
            t0 = time.perf_counter()
            n = 0
            try:
                for layer, eids in hot.items():
                    sorted_ids = sorted(eids)
                    runs: list[tuple[int, int]] = []
                    for eid in sorted_ids:
                        if runs and eid == runs[-1][0] + runs[-1][1]:
                            first, count = runs[-1]
                            runs[-1] = (first, count + 1)
                        else:
                            runs.append((eid, 1))
                    for lin in self.linears_by_layer.get(layer) or []:
                        b = getattr(lin, "backing", None)
                        keys = self.keys_by_layer.get(layer, {}).get(id(lin), []) or _proj_keys(lin)
                        for key in keys:
                            try:
                                if hasattr(b, "load_expert_run"):
                                    for first, count in runs:
                                        b.load_expert_run(key, first, count)
                                        n += count
                                else:
                                    for eid in sorted_ids:
                                        b.load_expert_slice(key, eid)
                                        n += 1
                            except Exception:
                                pass
            finally:
                self.seeded_s = time.perf_counter() - t0
                self.seed_done.set()
            logger.info(
                "Expert streaming: page-cache seed burst done: %d slices in %.2fs",
                n,
                self.seeded_s,
            )

        _WARM_POOL.submit(_run)
        return sum(len(eids) for eids in hot.values())


class WarmPinHook:
    """Attachment point for StreamingSwitchGLU (attribute `_warm_pins`)."""

    def __init__(
        self,
        warmer: PageCacheWarmer | None,
        pinner: PinController | None,
        recorder: "PrefillHotnessRecorder | None" = None,
    ):
        self.warmer = warmer
        self.pinner = pinner
        self.recorder = recorder

    @property
    def wants_usage_counts(self) -> bool:
        # Fase I6: only the pin/recorder consumers use the per-token usage
        # histogram, so the switch only pays the np.bincount when at least
        # one of them is attached. The readahead warmer keeps the plain
        # uniq_list contract (contiguous-run F_RDADVISE grouping).
        return self.pinner is not None or self.recorder is not None

    def on_layer_start(self, layer_idx: int, positions: int) -> None:
        if self.recorder is not None:
            self.recorder.maybe_seed(layer_idx, positions)
        if self.warmer is not None:
            self.warmer.on_layer_start(layer_idx, positions)

    def on_layer_plan(
        self,
        layer_idx: int,
        uniq_list: list[int],
        positions: int,
        counts=None,
    ) -> None:
        if self.warmer is not None:
            # Warmer (readahead/discarded reads) keeps the uniq-list signal:
            # its predictions are set-based (contiguous-run grouping), and a
            # histogram adds nothing there.
            self.warmer.on_layer_plan(layer_idx, uniq_list, positions)
        if self.pinner is not None:
            self.pinner.on_layer_plan(layer_idx, uniq_list, positions, counts)
        if self.recorder is not None:
            self.recorder.on_layer_plan(layer_idx, uniq_list, positions, counts)


