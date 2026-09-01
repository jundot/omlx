# SPDX-License-Identifier: Apache-2.0
"""Service Level Objectives for the inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SLOMetricKind(Enum):
    """Type of metric an SLO measures."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"
    CACHE_HIT_RATE = "cache_hit_rate"


class SLOStatus(Enum):
    """Current compliance status of an SLO."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BREACHED = "breached"


@dataclass(frozen=True, slots=True)
class SLO:
    """A single Service Level Objective.

    Attributes:
        name: Human-readable identifier (e.g. ``"latency_p95_small"``).
        metric_kind: Which class of metric this SLO tracks.
        target: The target value.  For latency/throughput this is a
            threshold the measured value must meet or exceed; for
            availability/cache_hit_rate it is a percentage (0–100).
        window_seconds: Rolling window over which compliance is evaluated.
        metric_key: Key looked up in the metrics store to get raw samples.
        degraded_target: If set, compliance between ``degraded_target`` and
            ``target`` yields ``DEGRADED`` rather than ``HEALTHY``.
        description: Short human-readable explanation for dashboard cards.
    """

    name: str
    metric_kind: SLOMetricKind
    target: float
    window_seconds: int
    metric_key: str
    degraded_target: float | None = None
    description: str = ""


# ---------------------------------------------------------------------------
# Default SLOs for the inference pipeline
# ---------------------------------------------------------------------------

DEFAULT_SLOS: list[SLO] = [
    # Latency targets (seconds, per request end-to-end)
    SLO(
        name="latency_p50_small",
        metric_kind=SLOMetricKind.LATENCY,
        target=0.5,
        window_seconds=3600,
        metric_key="latency_p50",
        degraded_target=0.75,
        description="P50 end-to-end latency for small models (<7B params)",
    ),
    SLO(
        name="latency_p95_small",
        metric_kind=SLOMetricKind.LATENCY,
        target=2.0,
        window_seconds=3600,
        metric_key="latency_p95",
        degraded_target=3.0,
        description="P95 end-to-end latency for small models (<7B params)",
    ),
    SLO(
        name="latency_p99_small",
        metric_kind=SLOMetricKind.LATENCY,
        target=5.0,
        window_seconds=3600,
        metric_key="latency_p99",
        degraded_target=7.5,
        description="P99 end-to-end latency for small models (<7B params)",
    ),
    SLO(
        name="latency_p50_large",
        metric_kind=SLOMetricKind.LATENCY,
        target=2.0,
        window_seconds=3600,
        metric_key="latency_p50",
        degraded_target=3.0,
        description="P50 end-to-end latency for large models (≥7B params)",
    ),
    SLO(
        name="latency_p95_large",
        metric_kind=SLOMetricKind.LATENCY,
        target=8.0,
        window_seconds=3600,
        metric_key="latency_p95",
        degraded_target=12.0,
        description="P95 end-to-end latency for large models (≥7B params)",
    ),
    SLO(
        name="latency_p99_large",
        metric_kind=SLOMetricKind.LATENCY,
        target=15.0,
        window_seconds=3600,
        metric_key="latency_p99",
        degraded_target=20.0,
        description="P99 end-to-end latency for large models (≥7B params)",
    ),
    # Throughput targets (tokens/second per node)
    SLO(
        name="throughput_min_small",
        metric_kind=SLOMetricKind.THROUGHPUT,
        target=30.0,
        window_seconds=3600,
        metric_key="tokens_per_second",
        degraded_target=20.0,
        description="Minimum throughput (tokens/s) for small models",
    ),
    SLO(
        name="throughput_min_large",
        metric_kind=SLOMetricKind.THROUGHPUT,
        target=8.0,
        window_seconds=3600,
        metric_key="tokens_per_second",
        degraded_target=5.0,
        description="Minimum throughput (tokens/s) for large models",
    ),
    # Availability target
    SLO(
        name="availability",
        metric_kind=SLOMetricKind.AVAILABILITY,
        target=99.9,
        window_seconds=604800,
        metric_key="uptime_percentage",
        degraded_target=99.5,
        description="99.9% uptime over a 7-day rolling window",
    ),
    # Cache hit rate target
    SLO(
        name="cache_hit_rate",
        metric_kind=SLOMetricKind.CACHE_HIT_RATE,
        target=70.0,
        window_seconds=3600,
        metric_key="cache_hit_rate",
        degraded_target=55.0,
        description="70% cache hit rate for repeated prefixes (1h window)",
    ),
]


def slo_by_name(name: str) -> SLO | None:
    """Look up a predefined SLO by name.  Returns *None* if not found."""
    for slo in DEFAULT_SLOS:
        if slo.name == name:
            return slo
    return None
