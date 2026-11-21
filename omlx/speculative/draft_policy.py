# SPDX-License-Identifier: Apache-2.0
"""Per verify step: which draft source, and how many of its drafts to verify.

Each step the policy maximizes ``E[tokens committed] / T(step)`` over

    none (k = 0) | suffix lookup, k <= its proposal | the drafter, k <= its block

from running estimates of acceptance and a measured cost of a ``k + 1`` row
verify pass. It only chooses which drafts are verified: the target checks every
one, so the output is unchanged.

The design is Basalt's (``basalt/spec/controller`` and ``draft_policy``):

- Lookup: ``E(k) = 1 + q + ... + q^k``, ``q`` the acceptance rate of lookup
  drafts whose match was about as long (four buckets of match length, decayed
  counts over a prior). A long window costs much more to verify than the
  drafter's usual few rows, so a short noisy match must not win just because
  it proposes more: expected tokens per millisecond decide.
- Drafter without stated chances: position ``i``'s conditional acceptance
  ``p_i`` (an EMA), ``E(k) = 1 + p_0 + p_0 p_1 + ...``.
- Drafter with a stated chance per draft (a confidence head): the survival
  product of the calibrated chances, so the window follows this block's own
  drafts instead of an average.
- Cost: a round costs ``scale x base(rows)``. The scale follows every round
  fast (the context grows, the machine gets busy), each size's base follows
  its own rounds slowly, and sizes not measured lately are inferred from the
  ones that were, or from a prior shape. A drafter round also pays the
  drafter; that cost is learned as the difference to draft-free rounds of the
  same size, which is what a lookup round saves.
- ``min_gain``: drafting must beat plain decoding by this much to start, or
  the step runs plain; a drafter already running keeps running while it beats
  plain at all (the estimates are noisy, and a parked drafter learns nothing).
  ``margin``: a lookup window must beat the drafter's by this much.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

# Largest verify window (the pending token plus its drafts) the cost model tracks.
MAX_ROWS = 16
K_MAX = MAX_ROWS - 1

# The shape of a round's cost by window size, relative to one row, used until
# sizes are measured (Basalt's, between its measured curves).
_SHAPE = (0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6)
_SHAPE = _SHAPE + tuple(3.6 + 0.35 * i for i in range(1, MAX_ROWS - 7))
_SCALE_ALPHA = 0.2
_COST_ALPHA = 0.1
# A size's first rounds start from what the other sizes imply, worth this many
# rounds, so one cold round cannot rule a size out.
_BASE_PRIOR = 3.0
# Rounds after which a size's cost is from another time.
_STALE = 64
# A window size is tried this often before its guessed cost can veto it.
_PROBES = 3
_PROBE_RATE = 0.85
# Plain rounds after which one draft is verified again: nothing is learned
# about the drafter while no draft is tested, so a dip would park it for good.
_REPROBE = 48
_DRAFT_ALPHA = 0.1

_LOOKUP_DECAY = 0.97
_LOOKUP_PRIOR = (0.35, 0.6, 0.8, 0.92)
_LOOKUP_PRIOR_N = 4.0

_POSITION_ALPHA = 0.05
# Inkling 2 Mini's DSpark on real weights (1,564 cycles): 0.68, 0.69, 0.60, then 0.63.
_POSITION_PRIOR = (0.68, 0.69, 0.60, 0.63)

# Stated chances: the bins' upper edges, finest near 1 where long windows are decided.
_CHANCE_EDGES = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995)
_CHANCE_DECAY = 0.98
_CHANCE_PRIOR_N = 10.0


class Source(enum.IntEnum):
    NONE = 0
    LOOKUP = 1
    DRAFTER = 2


@dataclass(frozen=True)
class Choice:
    source: Source = Source.NONE
    k: int = 0
    expected_tokens: float = 1.0
    tokens_per_ms: float = 0.0
    # The lookup match behind a LOOKUP choice.
    match: int = 0
    # Taken to measure a window size, not because its estimate wins.
    probe: bool = False


def lookup_bucket(match: int) -> int:
    """Match lengths 3-4, 5-7, 8-15 and 16 or more."""
    return 3 if match >= 16 else 2 if match >= 8 else 1 if match >= 5 else 0


def _chance_bin(chance: float) -> int:
    b = 0
    while b < len(_CHANCE_EDGES) and chance >= _CHANCE_EDGES[b]:
        b += 1
    return b


class _Costs:
    """Round time by window size: ``scale x base(rows)``, plus the drafter's."""

    def __init__(self, shape: Sequence[float], draft_ratio: float, timed: bool):
        self.shape = tuple(shape)
        self.draft_ratio = float(draft_ratio)
        self.timed = timed
        self.scale = 0.0
        self.base = [0.0] * (MAX_ROWS + 1)
        self.base_n = [0.0] * (MAX_ROWS + 1)
        self.seen = [0] * (MAX_ROWS + 1)
        # The round a size was last measured without a draft in it.
        self.pure = [0] * (MAX_ROWS + 1)
        self.round = 0
        self._draft: Optional[float] = None

    def recent(self, rows: int) -> bool:
        return self.base_n[rows] > 0 and self.round - self.seen[rows] <= _STALE

    def unmeasured(self, rows: int) -> bool:
        return self.base_n[rows] < _BASE_PRIOR + _PROBES or not self.recent(rows)

    def _base(self, rows: int) -> float:
        if self.recent(rows):
            return self.base[rows]
        for lately in (True, False):
            b = self._base_from(rows, lately)
            if b > 0:
                return b
        return self.shape[rows]

    def _base_from(self, rows: int, lately: bool) -> float:
        # The measured sizes next to `rows`: between two, the line through
        # them; above the largest, the line on from the two largest (the cost
        # of a row here); below the smallest, or with one, the prior shape.
        lo = hi = below = 0
        for u in range(1, MAX_ROWS + 1):
            if u == rows or self.base_n[u] <= 0 or (lately and not self.recent(u)):
                continue
            if u < rows:
                below, lo = lo, u
            elif hi == 0:
                hi = u
        base, shape = self.base, self.shape
        if lo and hi:
            return base[lo] + (base[hi] - base[lo]) * (rows - lo) / (hi - lo)
        if lo and below:
            slope = max(0.0, (base[lo] - base[below]) / (lo - below))
            return base[lo] + slope * (rows - lo)
        if lo:
            return base[lo] * shape[rows] / shape[lo]
        if hi:
            return base[hi] * shape[rows] / shape[hi]
        return -1.0

    def verify_ms(self, rows: int) -> float:
        scale = self.scale if self.scale > 0 else 1.0
        if rows <= MAX_ROWS:
            return scale * self._base(max(1, rows))
        # Past the tracked sizes (several streams' rows): the cost of a row on.
        row = self._base(MAX_ROWS) - self._base(MAX_ROWS - 1)
        return scale * (self._base(MAX_ROWS) + max(0.0, row) * (rows - MAX_ROWS))

    def draft_ms(self) -> float:
        if self._draft is not None:
            return self._draft
        return self.draft_ratio * self.verify_ms(1)

    def observe(self, rows: int, ms: Optional[float], drafted: bool) -> None:
        rows = max(1, min(int(rows), MAX_ROWS))
        self.round += 1
        if not self.timed:
            ms = self.shape[rows] + (
                self.draft_ratio * self.shape[1] if drafted else 0.0
            )
        if ms is None or ms <= 0:
            return
        if drafted:
            if self.pure[rows] and self.round - self.pure[rows] <= _STALE:
                sample = max(0.0, ms - self.verify_ms(rows))
                self._draft = self.draft_ms() + _DRAFT_ALPHA * (
                    sample - self.draft_ms()
                )
            ms = max(ms - self.draft_ms(), 0.1 * ms)
        else:
            self.pure[rows] = self.round
        if self.scale <= 0:
            self.scale = ms / self.shape[rows]
        if not self.recent(rows):
            self.base[rows] = self._base(rows)
            self.base_n[rows] = _BASE_PRIOR
        self.scale += _SCALE_ALPHA * (ms / self.base[rows] - self.scale)
        w = max(_COST_ALPHA, 1.0 / (self.base_n[rows] + 1.0))
        self.base[rows] = (1.0 - w) * self.base[rows] + w * ms / self.scale
        self.base_n[rows] += 1.0
        self.seen[rows] = self.round


class DraftPolicy:
    """Online choice of draft source and window for one decoding stream."""

    def __init__(
        self,
        max_drafts: int = 7,
        *,
        min_gain: float = 0.05,
        margin: float = 0.03,
        timed: bool = True,
        cost_shape: Optional[Sequence[float]] = None,
        draft_ratio: float = 0.15,
        position_prior: Optional[Sequence[float]] = None,
    ) -> None:
        """``timed`` False: round times are not read and every round costs the
        prior shape of its size, so the same tokens give the same windows on
        every run. ``cost_shape[rows]`` is that prior (index 0 unused) and
        ``draft_ratio`` the drafter's prior cost, both relative to one row."""
        self.max_drafts = max(0, min(int(max_drafts), K_MAX))
        self.min_gain = float(min_gain)
        self.margin = float(margin)
        shape = list(cost_shape) if cost_shape is not None else list(_SHAPE)
        while len(shape) <= MAX_ROWS:
            shape.append(shape[-1] + (shape[-1] - shape[-2]))
        self._costs = _Costs(shape[: MAX_ROWS + 1], draft_ratio, timed)
        prior = tuple(position_prior or _POSITION_PRIOR)
        self._position = [prior[min(i, len(prior) - 1)] for i in range(K_MAX)]
        self._lookup_ok = [0.0] * len(_LOOKUP_PRIOR)
        self._lookup_bad = [0.0] * len(_LOOKUP_PRIOR)
        self._chance_ok = [0.0] * (len(_CHANCE_EDGES) + 1)
        self._chance_n = [0.0] * (len(_CHANCE_EDGES) + 1)
        self._plain_rounds = 0
        self._drafting = False

    # --- estimates -------------------------------------------------------

    def lookup_rate(self, match: int) -> float:
        """Acceptance rate of a lookup draft behind a match this long."""
        b = lookup_bucket(match)
        ok, bad = self._lookup_ok[b], self._lookup_bad[b]
        return (ok + _LOOKUP_PRIOR_N * _LOOKUP_PRIOR[b]) / (ok + bad + _LOOKUP_PRIOR_N)

    def position_rate(self, position: int) -> float:
        """Acceptance rate of the drafter's draft ``position`` given its prefix."""
        return self._position[min(position, K_MAX - 1)]

    def chance_rate(self, chance: float) -> float:
        """Acceptance rate of a draft whose stated chance was ``chance``."""
        b = _chance_bin(chance)
        c = min(1.0, max(0.0, float(chance)))
        return (self._chance_ok[b] + _CHANCE_PRIOR_N * c) / (
            self._chance_n[b] + _CHANCE_PRIOR_N
        )

    def cost_ms(self, rows: int, drafted: bool = False) -> float:
        """Time of a round verifying ``rows`` tokens, with or without a draft."""
        ms = self._costs.verify_ms(int(rows))
        return ms + self._costs.draft_ms() if drafted else ms

    def lookup_survival(self, k: int, match: int) -> List[float]:
        """Chance that lookup drafts ``0..j`` are all accepted, for ``j < k``."""
        q = self.lookup_rate(match)
        out, run = [], 1.0
        for _ in range(k):
            run *= q
            out.append(run)
        return out

    def drafter_survival(
        self, k: int, conf: Optional[Sequence[float]] = None
    ) -> List[float]:
        """The same for the drafter's block: from the stated chances when the
        drafter gives them, else from the per-position rates."""
        out, run = [], 1.0
        for i in range(k):
            if conf is not None and i < len(conf):
                run *= self.chance_rate(conf[i])
            else:
                run *= self.position_rate(i)
            out.append(run)
        return out

    # --- the choice ------------------------------------------------------

    def choose(
        self,
        *,
        lookup_k: int = 0,
        match: int = 0,
        drafter_k: int = 0,
        conf: Optional[Sequence[float]] = None,
        drafter_window: Optional[int] = None,
        drafted: bool = False,
    ) -> Choice:
        """The step with the most expected tokens per millisecond.

        ``lookup_k`` tokens are proposed by a lookup behind a match of
        ``match``; the drafter can give ``drafter_k`` drafts, with stated
        chances ``conf`` if it has them. ``drafter_window``: the host has
        already settled how many drafter drafts to verify (a depth controller
        owns that), so only whether a lookup beats that window is decided.
        ``drafted``: the drafter's block already exists, so its cost is paid
        whichever window is taken.
        """
        costs = self._costs
        sunk = costs.draft_ms() if drafted else 0.0
        baseline = 1.0 / (costs.verify_ms(1) + sunk)
        incumbent = Choice(tokens_per_ms=baseline)

        drafter_k = max(0, min(int(drafter_k), self.max_drafts))
        if drafter_k > 0:
            draft = costs.draft_ms()
            survival = self.drafter_survival(drafter_k, conf)
            fixed = (
                None
                if drafter_window is None
                else max(1, min(int(drafter_window), drafter_k))
            )
            best = None
            expected = 1.0
            for k in range(1, drafter_k + 1):
                expected += survival[k - 1]
                if fixed is not None and k != fixed:
                    continue
                rate = expected / (costs.verify_ms(k + 1) + draft)
                if best is None or rate > best.tokens_per_ms:
                    best = Choice(Source.DRAFTER, k, expected, rate)
            if (
                fixed is None
                and conf is not None
                and best.k < drafter_k
                and costs.unmeasured(best.k + 2)
                and survival[best.k] >= _PROBE_RATE
            ):
                # A size whose cost is only guessed is tried a few times when
                # its drafts are likely accepted, or the window never grows.
                k = best.k + 1
                best = Choice(
                    Source.DRAFTER,
                    k,
                    best.expected_tokens + survival[k - 1],
                    best.tokens_per_ms,
                    probe=True,
                )
            # The gain needed to start drafting; none to keep on.
            gain = 0.0 if self._drafting and conf is None else self.min_gain
            if (
                fixed is not None
                or best.probe
                or best.tokens_per_ms >= baseline * (1.0 + gain)
            ):
                incumbent = best
            elif self._plain_rounds >= _REPROBE:
                incumbent = Choice(
                    Source.DRAFTER,
                    1,
                    1.0 + survival[0],
                    (1.0 + survival[0]) / (costs.verify_ms(2) + draft),
                    probe=True,
                )

        lookup_k = max(0, min(int(lookup_k), self.max_drafts))
        if lookup_k > 0:
            q = self.lookup_rate(match)
            best = None
            expected, run = 1.0, 1.0
            for k in range(1, lookup_k + 1):
                run *= q
                expected += run
                rate = expected / (costs.verify_ms(k + 1) + sunk)
                if best is None or rate > best.tokens_per_ms:
                    best = Choice(Source.LOOKUP, k, expected, rate, match=match)
            bar = (
                incumbent.tokens_per_ms * (1.0 + self.margin)
                if incumbent.source is Source.DRAFTER
                else baseline * (1.0 + self.min_gain)
            )
            if best.tokens_per_ms > bar:
                return best
            if (
                lookup_k > incumbent.k
                and costs.unmeasured(lookup_k + 1)
                and q >= _PROBE_RATE
            ):
                return Choice(
                    Source.LOOKUP,
                    lookup_k,
                    expected,
                    expected / (costs.verify_ms(lookup_k + 1) + sunk),
                    match=match,
                    probe=True,
                )
        return incumbent

    def window(self, conf: Sequence[float], n: Optional[int] = None) -> int:
        """Rows to verify (1 + drafts) of a block drafted with stated chances."""
        n = len(conf) if n is None else min(int(n), len(conf))
        return 1 + self.choose(drafter_k=n, conf=conf, drafted=True).k

    # --- learning --------------------------------------------------------

    def observe(
        self,
        choice: Choice,
        accepted: int,
        round_ms: Optional[float] = None,
        *,
        conf: Optional[Sequence[float]] = None,
        drafted: Optional[bool] = None,
    ) -> None:
        """After the round: ``accepted`` of the ``choice.k`` drafts matched (a
        prefix), in ``round_ms`` (None: not a clean sample of this window).
        ``conf``: the stated chances of a drafter window's drafts. ``drafted``:
        whether a draft ran in the round (default: a drafter round)."""
        k = max(0, int(choice.k))
        accepted = max(0, min(int(accepted), k))
        if drafted is None:
            drafted = choice.source is Source.DRAFTER
        self._costs.observe(k + 1, round_ms, bool(drafted))
        if k == 0 or choice.source is Source.NONE:
            self._plain_rounds += 1
            self._drafting = False
            return
        self._plain_rounds = 0
        if choice.source is Source.DRAFTER:
            self._drafting = True
        if choice.source is Source.LOOKUP:
            b = lookup_bucket(choice.match)
            self._lookup_ok[b] = _LOOKUP_DECAY * self._lookup_ok[b] + accepted
            self._lookup_bad[b] = _LOOKUP_DECAY * self._lookup_bad[b] + (
                1.0 if accepted < k else 0.0
            )
            return
        # Drafts 0..accepted-1 were accepted given their prefix and draft
        # `accepted` (if the window had it) was not; the rest were never tested.
        tested = min(k, accepted + 1)
        position = self._position
        for i in range(tested):
            hit = 1.0 if i < accepted else 0.0
            position[i] += _POSITION_ALPHA * (hit - position[i])
            if conf is not None and i < len(conf):
                b = _chance_bin(conf[i])
                self._chance_ok[b] = _CHANCE_DECAY * self._chance_ok[b] + hit
                self._chance_n[b] = _CHANCE_DECAY * self._chance_n[b] + 1.0
        if accepted == k:
            # The positions past the window were never tested: without this
            # they keep their prior and the window can never grow.
            for i in range(k, K_MAX):
                position[i] += _POSITION_ALPHA * (position[k - 1] - position[i])


def allocate_batch_verify_budget(
    survivals: Sequence[Sequence[float]],
    cost_ms: Callable[[int], float],
    *,
    rectangular: bool = False,
    fair: bool = False,
    min_gain: float = 0.0,
) -> List[int]:
    """Drafts to verify per stream when several streams share one verify pass.

    ``survivals[s][j]`` is the chance that stream ``s``'s drafts ``0..j`` are
    all accepted (``lookup_survival`` / ``drafter_survival``) and
    ``cost_ms(rows)`` the time of a pass over ``rows`` rows in total. The
    result maximizes the streams' expected tokens per millisecond; every
    stream verifies its pending token whatever it gets.

    Each further row adds its survival to the expected tokens, and survivals
    only fall along a stream, so the best set of ``R`` rows is the ``R``
    largest survivals; ``R`` is then the count with the best rate.
    ``rectangular``: every stream must verify the same number of drafts (a
    verify pass that takes one window size).

    ``fair``: the most tokens in total can starve a stream: every row another
    stream verifies lengthens the pass for all, so a stream with weak drafts
    next to one quoting its context falls below its plain-decoding rate. With
    ``fair`` the product of the streams' rates is maximized instead (each
    stream's share of the gain is proportional), among the allocations that
    leave no stream slower than plain decoding.
    """
    streams = len(survivals)
    if not streams:
        return []
    plain = streams / cost_ms(streams)
    if rectangular:
        width = min(len(s) for s in survivals)
        best_w, best_rate = 0, plain
        expected = float(streams)
        for w in range(1, width + 1):
            expected += sum(s[w - 1] for s in survivals)
            rate = expected / cost_ms(streams * (w + 1))
            if rate > best_rate:
                best_w, best_rate = w, rate
        if best_rate < plain * (1.0 + min_gain):
            best_w = 0
        return [best_w] * streams
    if fair:
        return _fair_budget(survivals, cost_ms, min_gain)
    rows = sorted(
        ((s[j], j, i) for i, s in enumerate(survivals) for j in range(len(s))),
        key=lambda row: (-row[0], row[1], row[2]),
    )
    best_r, best_rate = 0, plain
    expected = float(streams)
    for r, (gain, _, _) in enumerate(rows, start=1):
        expected += gain
        rate = expected / cost_ms(streams + r)
        if rate > best_rate:
            best_r, best_rate = r, rate
    counts = [0] * streams
    if best_rate >= plain * (1.0 + min_gain):
        for _, _, i in rows[:best_r]:
            counts[i] += 1
    return counts


def _fair_budget(survivals, cost_ms, min_gain: float) -> List[int]:
    """Row counts maximizing ``sum(log(expected_i)) - streams * log(cost)``.

    A stream's log expected tokens is concave in its rows, so rows are added
    where the next one raises it most, and the best total is kept.
    """
    streams = len(survivals)
    counts = [0] * streams
    expected = [1.0] * streams
    plain_ms = cost_ms(streams)
    best_score = streams * (math.log1p(min_gain) - math.log(plain_ms))
    best_counts = list(counts)
    for total in range(streams + 1, streams + sum(len(s) for s in survivals) + 1):
        _, i = max(
            (math.log1p(s[counts[i]] / expected[i]), -i)
            for i, s in enumerate(survivals)
            if counts[i] < len(s)
        )
        i = -i
        expected[i] += survivals[i][counts[i]]
        counts[i] += 1
        ms = cost_ms(total)
        if min(expected) * plain_ms < ms:
            continue  # a stream would commit slower than it does undrafted
        score = sum(math.log(e) for e in expected) - streams * math.log(ms)
        if score > best_score:
            best_counts, best_score = list(counts), score
    return best_counts
