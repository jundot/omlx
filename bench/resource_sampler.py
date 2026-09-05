# SPDX-License-Identifier: Apache-2.0
"""Lightweight resource sampler for benches (GPU/CPU/IO/RAM).

Runs a daemon thread that snapshots once per interval:

- process CPU % and system CPU %        (psutil)
- process RSS                            (psutil)
- global disk read/write throughput      (psutil.disk_io_counters deltas)
- GPU utilization %                      (ioreg IOAccelerator, no sudo needed)
- MLX active/cache memory                (optional callbacks)

Usage:
    s = ResourceSampler(interval=1.0)
    s.start()
    ...
    s.mark("prefill")
    ...
    s.mark("decode")
    s.stop()
    summary = s.summary()   # per-phase means + overall
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Any, Callable


_GPU_RE = re.compile(r'"Device Utilization %"=(\d+)')


def _gpu_util_percent() -> int | None:
    """Max GPU utilization across accelerators. None when unavailable."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout
        vals = [int(v) for v in _GPU_RE.findall(out)]
        if vals:
            return max(vals)
    except Exception:
        pass
    return None


class ResourceSampler:
    def __init__(
        self,
        interval: float = 1.0,
        mlx_callbacks: dict[str, Callable[[], float]] | None = None,
    ):
        import psutil

        self._psutil = psutil
        self._proc = psutil.Process()
        self.interval = interval
        self._mlx = mlx_callbacks or {}
        self._samples: list[dict[str, Any]] = []
        self._marks: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_disk: tuple[float, float] | None = None  # (read_bytes, write_bytes)
        self._last_disk_t: float | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._proc.cpu_percent()  # prime the counter
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + 2)

    def mark(self, label: str) -> None:
        self._marks.append({"label": label, "t": time.perf_counter()})

    # -- sampling ----------------------------------------------------------

    def _disk_kbps(self) -> tuple[float | None, float | None]:
        try:
            cnt = self._psutil.disk_io_counters()
            if cnt is None:
                return None, None
            now = time.perf_counter()
            cur = (cnt.read_bytes, cnt.write_bytes)
            if self._last_disk is None or self._last_disk_t is None:
                self._last_disk, self._last_disk_t = cur, now
                return None, None
            dt = now - self._last_disk_t
            if dt <= 0:
                return None, None
            r = (cur[0] - self._last_disk[0]) / dt / 1024  # KiB/s
            w = (cur[1] - self._last_disk[1]) / dt / 1024
            self._last_disk, self._last_disk_t = cur, now
            return r, w
        except Exception:
            return None, None

    def _snapshot(self) -> dict[str, Any]:
        s: dict[str, Any] = {"t": time.perf_counter()}
        try:
            s["proc_cpu_pct"] = self._proc.cpu_percent()
            s["sys_cpu_pct"] = self._psutil.cpu_percent()
            s["rss_gib"] = self._proc.memory_info().rss / 1024**3
        except Exception:
            pass
        r, w = self._disk_kbps()
        if r is not None:
            s["disk_read_kib_s"] = r
            s["disk_write_kib_s"] = w
        g = _gpu_util_percent()
        if g is not None:
            s["gpu_util_pct"] = g
        for name, cb in self._mlx.items():
            try:
                s[name] = cb() / 1024**3
            except Exception:
                pass
        return s

    def _run(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self._snapshot())
            self._stop.wait(self.interval)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Per-phase means (samples between consecutive marks; trailing = 'after_last')."""
        phases: dict[str, list[dict[str, Any]]] = {}
        marks = sorted(self._marks, key=lambda m: m["t"])
        bounds: list[tuple[float, str]] = [(0.0, "before_first")]
        for m in marks:
            bounds.append((m["t"], m["label"]))
        for s in self._samples:
            label = "before_first"
            for t, lab in bounds:
                if s["t"] >= t:
                    label = lab
                else:
                    break
            phases.setdefault(label, []).append(s)

        def _phase_stats(samples: list[dict[str, Any]]) -> dict[str, float]:
            keys = [k for k in samples[0] if k != "t"] if samples else []
            out: dict[str, float] = {}
            for k in keys:
                vals = [s[k] for s in samples if k in s]
                if vals:
                    out[f"{k}_avg"] = round(sum(vals) / len(vals), 2)
                    out[f"{k}_max"] = round(max(vals), 2)
            out["samples"] = len(samples)
            return out

        result = {
            "phases": {lab: _phase_stats(samps) for lab, samps in phases.items()},
            "marks": [
                {"label": m["label"], "t_s": round(m["t"], 2)} for m in marks
            ],
        }
        return result

    def samples(self) -> list[dict[str, Any]]:
        return self._samples
