# SPDX-License-Identifier: Apache-2.0
"""Capacity planner — collects per-node utilization and produces scaling recommendations."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from .error_budget import ErrorBudgetTracker
from .slo_tracker import SLOTracker

# Trend resolution: 15-minute buckets over 24 hours.
_TREND_BUCKET_SECONDS = 900
_TREND_BUCKETS = 96  # 24h / 15min

# Utilization thresholds for scaling decisions.
_SCALE_UP_CPU_THRESHOLD = 85.0
_SCALE_UP_MEM_THRESHOLD = 90.0
_STABLE_CPU_CEILING = 75.0
_STABLE_MEM_CEILING = 80.0


@dataclass(slots=True)
class NodeCapacity:
    """Per-node utilization snapshot."""

    node_id: str
    hostname: str
    cpu_usage_pct: float
    memory_usage_pct: float
    memory_used_bytes: int
    memory_total_bytes: int
    gpu_utilization_pct: float | None
    active_connections: int
    accelerator: str | None = None


@dataclass(slots=True)
class TrendPoint:
    """One bucket in the utilization trend."""

    timestamp: float
    avg_cpu_pct: float
    avg_memory_pct: float


@dataclass(slots=True)
class ScalingRecommendation:
    """Scaling decision with reasoning."""

    action: Literal["scale_up", "scale_down", "stable"]
    reasons: list[str]
    confidence: float


@dataclass(slots=True)
class CapacityReport:
    """Full capacity report returned by the planner."""

    nodes: list[NodeCapacity]
    total_nodes: int
    avg_cpu_pct: float
    avg_memory_pct: float
    total_headroom_pct: float
    saturation_pct: float
    trend: list[TrendPoint]
    recommendation: ScalingRecommendation
    collected_at: float


class CapacityPlanner:
    """Collects utilization metrics and produces scaling recommendations.

    Integrates error budget status (from #2771) and SLO compliance (from
    #2769) into the scaling decision.
    """

    def __init__(
        self,
        *,
        error_budget: ErrorBudgetTracker | None = None,
        slo_tracker: SLOTracker | None = None,
    ) -> None:
        self._error_budget = error_budget or ErrorBudgetTracker(
            slo_tracker=slo_tracker or SLOTracker()
        )
        self._lock = threading.Lock()
        # Rolling trend storage — circular buffer of TrendPoints.
        self._trend: deque[TrendPoint] = deque(maxlen=_TREND_BUCKETS)

    @property
    def error_budget(self) -> ErrorBudgetTracker:
        return self._error_budget

    def _snapshot_nodes(self) -> list[NodeCapacity]:
        """Gather current per-node utilization.

        Subclasses or tests can override this to inject synthetic data.
        In production this reads system metrics via psutil or the
        existing cluster probe infrastructure.
        """
        import os

        try:
            import psutil  # noqa: PLC0415
        except ImportError:
            psutil = None  # type: ignore[assignment]

        node_id = os.environ.get("OMLX_NODE_ID", "local")
        hostname = os.environ.get("OMLX_HOSTNAME", os.uname().nodename)

        if psutil is not None:
            mem = psutil.virtual_memory()
            cpu_pct = psutil.cpu_percent(interval=0.1)
            active_conns = len(psutil.net_connections(kind="inet"))
        else:
            # Fallback: read /proc on Linux when psutil is unavailable.
            cpu_pct = _read_proc_cpu()
            mem = _read_proc_mem()
            active_conns = 0

        accelerator = os.environ.get("OMLX_ACCELERATOR")
        gpu_util: float | None = None
        if accelerator == "cuda":
            gpu_util = _read_nvidia_gpu_util()

        return [
            NodeCapacity(
                node_id=node_id,
                hostname=hostname,
                cpu_usage_pct=round(cpu_pct, 1),
                memory_usage_pct=round(
                    (mem.used / mem.total * 100) if mem.total else 0.0, 1
                ),
                memory_used_bytes=mem.used,
                memory_total_bytes=mem.total,
                gpu_utilization_pct=gpu_util,
                active_connections=active_conns,
                accelerator=accelerator,
            )
        ]

    def collect_metrics(self, *, now: float | None = None) -> CapacityReport:
        """Collect current utilization and update the trend buffer."""

        ts = now if now is not None else time.time()
        nodes = self._snapshot_nodes()

        avg_cpu = sum(n.cpu_usage_pct for n in nodes) / max(len(nodes), 1)
        avg_mem = sum(n.memory_usage_pct for n in nodes) / max(len(nodes), 1)

        total_memory = sum(n.memory_total_bytes for n in nodes)
        used_memory = sum(n.memory_used_bytes for n in nodes)
        headroom = (
            ((total_memory - used_memory) / total_memory * 100)
            if total_memory
            else 100.0
        )
        saturation = used_memory / total_memory * 100 if total_memory else 0.0

        point = TrendPoint(
            timestamp=ts,
            avg_cpu_pct=round(avg_cpu, 1),
            avg_memory_pct=round(avg_mem, 1),
        )
        with self._lock:
            self._trend.append(point)

        recommendation = self._recommend(
            avg_cpu,
            avg_mem,
            nodes,
            now=ts,
        )

        return CapacityReport(
            nodes=nodes,
            total_nodes=len(nodes),
            avg_cpu_pct=round(avg_cpu, 1),
            avg_memory_pct=round(avg_mem, 1),
            total_headroom_pct=round(headroom, 1),
            saturation_pct=round(saturation, 1),
            trend=list(self._trend),
            recommendation=recommendation,
            collected_at=ts,
        )

    def _recommend(
        self,
        avg_cpu: float,
        avg_mem: float,
        nodes: list[NodeCapacity],
        *,
        now: float | None = None,
    ) -> ScalingRecommendation:
        """Derive a scaling recommendation from utilization, budgets, and SLOs."""

        reasons: list[str] = []
        action: Literal["scale_up", "scale_down", "stable"] = "stable"
        confidence = 0.8

        # --- Error budget signals ---
        can_deploy, blocking_slos = self._error_budget.can_deploy(now=now)
        if not can_deploy:
            reasons.append(
                f"Error budget depleted for: {', '.join(blocking_slos)}"
            )
            action = "scale_up"
            confidence = min(confidence + 0.1, 1.0)

        # --- SLO compliance signals ---
        budgets = self._error_budget.all_budgets(now=now)
        low_budget = [b for b in budgets if b.budget_remaining_pct < 25.0]
        if low_budget:
            names = ", ".join(b.slo_name for b in low_budget)
            reasons.append(f"Low error budget (<25%): {names}")
            if action == "stable":
                action = "scale_up"
            confidence = min(confidence + 0.05, 1.0)

        # --- Utilization signals ---
        if avg_cpu > _SCALE_UP_CPU_THRESHOLD or avg_mem > _SCALE_UP_MEM_THRESHOLD:
            reasons.append(
                f"High utilization — CPU {avg_cpu:.1f}%, memory {avg_mem:.1f}%"
            )
            action = "scale_up"
        elif avg_cpu < 20.0 and avg_mem < 30.0:
            reasons.append(
                f"Low utilization — CPU {avg_cpu:.1f}%, memory {avg_mem:.1f}%"
            )
            if action == "stable":
                action = "scale_down"
                confidence = min(confidence + 0.05, 1.0)
        else:
            reasons.append(
                f"Moderate utilization — CPU {avg_cpu:.1f}%, memory {avg_mem:.1f}%"
            )

        # --- Per-node hotspots ---
        hotspots = [
            n
            for n in nodes
            if n.cpu_usage_pct > 90.0 or n.memory_usage_pct > 95.0
        ]
        if hotspots:
            hotspot_ids = ", ".join(n.node_id for n in hotspots)
            reasons.append(f"Node hotspot: {hotspot_ids}")
            action = "scale_up"
            confidence = min(confidence + 0.1, 1.0)

        # --- Trend: detect sustained growth ---
        with self._lock:
            recent = list(self._trend)
        if len(recent) >= 4:
            last_four = recent[-4:]
            mem_increasing = all(
                last_four[i].avg_memory_pct <= last_four[i + 1].avg_memory_pct
                for i in range(len(last_four) - 1)
            )
            cpu_increasing = all(
                last_four[i].avg_cpu_pct <= last_four[i + 1].avg_cpu_pct
                for i in range(len(last_four) - 1)
            )
            if mem_increasing and last_four[-1].avg_memory_pct > 70.0:
                reasons.append("Memory utilization trending upward")
                if action == "stable":
                    action = "scale_up"
            if cpu_increasing and last_four[-1].avg_cpu_pct > 70.0:
                reasons.append("CPU utilization trending upward")
                if action == "stable":
                    action = "scale_up"

        if action == "stable" and not reasons:
            reasons.append("All metrics within normal range")

        return ScalingRecommendation(
            action=action,
            reasons=reasons,
            confidence=round(confidence, 2),
        )

    def get_trend(self) -> list[TrendPoint]:
        """Return the current trend buffer."""
        with self._lock:
            return list(self._trend)

    def clear_trend(self) -> None:
        """Reset the trend buffer."""
        with self._lock:
            self._trend.clear()


# ------------------------------------------------------------------
# Linux /proc fallback helpers
# ------------------------------------------------------------------


def _read_proc_cpu() -> float:
    """Best-effort CPU usage from /proc/stat (one instant sample)."""
    try:
        with open("/proc/stat") as fh:
            parts = fh.readline().split()
        user, nice, system, idle = (int(parts[i]) for i in range(1, 5))
        total = user + nice + system + idle
        busy = user + nice + system
        return round(busy / total * 100, 1) if total else 0.0
    except (OSError, ValueError, IndexError):
        return 0.0


def _read_proc_mem() -> Any:
    """Best-effort memory info from /proc/meminfo."""

    class _MemInfo:
        total = 0
        used = 0

    info = _MemInfo()
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    info.total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    info.used = info.total - available
                    break
    except (OSError, ValueError, IndexError):
        pass
    return info


def _read_nvidia_gpu_util() -> float | None:
    """Best-effort GPU utilization via nvidia-smi."""
    import subprocess  # noqa: PLC0415

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().splitlines()[0])
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None
