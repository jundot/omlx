"""Shape-aware forward-time estimates without scaling fixed tail costs."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from math import isfinite

_SAMPLES_PER_BUCKET = 8
_MAX_SAMPLE_AGE_SECONDS = 30.0


@dataclass
class PrefillTiming:
    """Interpolate a monotone curve of materialized forward observations.

    Each power-of-two token bucket retains at most eight recent observations,
    expiring after 30 seconds even when all new batches are declined. Its
    smallest token count and largest duration form a conservative point.
    Prefix maxima make interpolation monotone, as the
    planner requires. Short forwards affect the short end of the curve rather
    than imposing their low tokens/second on every larger chunk. Outside the
    observed range, estimates scale proportionally from the nearest endpoint.

    Observations exclude row handoff and scheduler cleanup. Those operations
    still accrue actual decode debt; a forward estimate is not a bound on the
    entire scheduling turn. Storage is logarithmic in the largest observation.
    """

    _samples: dict[int, deque[tuple[int, float, float]]] = field(default_factory=dict)

    def observe(
        self, token_count: int, forward_seconds: float, *, now: float | None = None
    ) -> None:
        if (
            type(token_count) is not int
            or token_count <= 0
            or isinstance(forward_seconds, bool)
            or not isinstance(forward_seconds, (int, float))
            or not isfinite(forward_seconds)
            or forward_seconds <= 0
        ):
            return
        bucket = token_count.bit_length()
        observed_at = time.perf_counter() if now is None else now
        self._samples.setdefault(bucket, deque(maxlen=_SAMPLES_PER_BUCKET)).append(
            (token_count, forward_seconds, observed_at)
        )

    def estimate(self, token_count: int, *, now: float | None = None) -> float | None:
        if type(token_count) is not int or token_count <= 0 or not self._samples:
            return None
        current_time = time.perf_counter() if now is None else now
        points = []
        for bucket in sorted(self._samples):
            samples = self._samples[bucket]
            retained = [
                sample
                for sample in samples
                if 0 <= current_time - sample[2] < _MAX_SAMPLE_AGE_SECONDS
            ]
            if not retained:
                del self._samples[bucket]
                continue
            if len(retained) != len(samples):
                self._samples[bucket] = deque(retained, maxlen=_SAMPLES_PER_BUCKET)
            points.append(
                (
                    min(sample[0] for sample in retained),
                    max(sample[1] for sample in retained),
                )
            )
        if not points:
            return None
        left_tokens, left_seconds = 0, 0.0
        for right_tokens, observed_seconds in points:
            right_seconds = max(left_seconds, observed_seconds)
            if token_count <= right_tokens:
                fraction = (token_count - left_tokens) / (right_tokens - left_tokens)
                return left_seconds + fraction * (right_seconds - left_seconds)
            left_tokens, left_seconds = right_tokens, right_seconds
        return left_seconds * token_count / left_tokens
