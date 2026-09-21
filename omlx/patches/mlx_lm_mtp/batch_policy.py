# SPDX-License-Identifier: Apache-2.0
"""Choose draft depth from whole-batch work and measured ordinary decode."""

from collections import deque
from statistics import median


class BatchPolicy:
    """A policy belongs to one UID cohort, never to a model or single row.

    Costs include the complete batch call. Acceptance is pooled by conditional
    prefix reach, so rows with different accepted lengths contribute once each.
    Ordinary decode supplies its own baseline; depth-zero MTP is not a proxy.
    """

    def __init__(self, uids, max_depth, *, fixed=False):
        self.uids = tuple(uids)
        self.max_depth = max(1, max_depth)
        self.cur = self.max_depth
        # A fixed policy drafts max_depth every cycle: no ordinary-decode
        # calibration, no depth search and no parking. Block drafters use it
        # because their block is trained, not chosen.
        self.fixed = bool(fixed)
        self.standard = deque(maxlen=8)
        self.standard_warmup = 2
        self.costs = {}
        self.acceptance = [0.6] * self.max_depth
        self.losing = 0
        self.cooldown = 128
        self.remaining = 0
        self.elapsed_ms = 0.0
        self.last_probe_ms = 0.0
        self.last_seen = {}
        self.finished_at = None
        self.last_mode = None
        self._timing_interrupted = False

    def interrupt_timing(self):
        self._timing_interrupted = True

    def cycle_time_ms(self, mode, started, finished):
        # Between calls, the scheduler handles responses while the previously
        # dispatched MLX work may still run. Include that interval exactly
        # once, rather than timing only the host's wait inside next().
        begin = self.finished_at if self.last_mode == mode else started
        self.finished_at = finished
        self.last_mode = mode
        if self._timing_interrupted:
            # Prefill may have consumed pending decode work as well as wall time.
            self._timing_interrupted = False
            return None
        return max(0.0, (finished - begin) * 1000)

    def needs_standard(self):
        if self.fixed:
            return False
        return self.standard_warmup > 0 or len(self.standard) < 3 or self.remaining > 0

    def observe_standard(self, milliseconds):
        if milliseconds is not None:
            if self.standard_warmup:
                self.standard_warmup -= 1
            else:
                self.standard.append(max(1e-6, milliseconds))
        if self.remaining:
            self.remaining -= 1
            if not self.remaining:
                self.costs.clear()
                self.last_seen.clear()
                self.acceptance = [0.6] * self.max_depth
                self.cur = self.max_depth
                self.losing = 0

    def score(self, depth):
        if depth == 0:
            return len(self.uids) / median(self.standard) if self.standard else 0.0
        costs = self.costs.get(depth)
        if not costs:
            return 0.0
        expected = reach = 1.0
        for probability in self.acceptance[:depth]:
            reach *= probability
            expected += reach
        return len(self.uids) * expected / median(costs)

    def observe_mtp(self, depth, accepted, milliseconds, *, stable):
        for position in range(depth):
            reached = sum(count >= position for count in accepted)
            if not reached:
                break
            rate = sum(count > position for count in accepted) / reached
            self.acceptance[position] += 0.08 * (rate - self.acceptance[position])
        if milliseconds is None or self.fixed:
            return
        self.elapsed_ms += milliseconds
        # A depth transition combines the previous asynchronous head and a new
        # CPU dispatch shape. Wait for a complete cycle at the same depth.
        if not stable:
            return
        self.costs.setdefault(depth, deque(maxlen=8)).append(max(1e-6, milliseconds))
        self.last_seen[depth] = self.elapsed_ms
        pending = [
            d for d in range(self.max_depth, 0, -1) if len(self.costs.get(d, ())) < 3
        ]
        if pending:
            self.cur = pending[0]
            return
        best = max(range(1, self.max_depth + 1), key=self.score)
        if self.score(best) > self.score(self.cur) * 1.03:
            self.cur = best
        self.losing = self.losing + 1 if self.score(best) <= self.score(0) * 1.03 else 0
        # Refresh stale alternatives without probing on every decision.
        if self.elapsed_ms - self.last_probe_ms >= 1000:
            rival = min(self.last_seen, key=self.last_seen.get)
            stale = self.elapsed_ms - self.last_seen[rival]
            if rival != self.cur and (
                self.score(rival) * 1.15 >= self.score(best) or stale >= 5000
            ):
                self.cur = rival
                self.last_probe_ms = self.elapsed_ms

    def should_park(self):
        return not self.fixed and self.losing >= 16

    def park(self):
        self.remaining = self.cooldown
        self.cooldown = min(4096, self.cooldown * 2)
        self.standard_warmup = 2
        self.losing = 0
