# SPDX-License-Identifier: Apache-2.0
"""Where an offloaded MoE layer's time goes, in order.

The per-request offload line reports totals (hits, misses, bytes), and a
stack sampler cannot resolve a decode layer that lasts a few milliseconds.
Neither shows whether the GPU sat idle while a read was outstanding, or
whether a read could have started earlier: both depend on the order of
events inside one layer.

``OMLX_MOE_OFFLOAD_TIMELINE=<path>`` records one span per step of the common
offload adapter and appends JSON lines to ``<path>`` when a request finishes:
``span``, ``layer``, ``count``, ``start_ns`` and ``end_ns``, on the
``time.perf_counter_ns`` timebase. ``layer`` is the position of the wrapped
layer in load order. ``count`` is the step's size: routes for ``layer`` and
``routing``, missing experts for ``ensure``, bytes for ``read``, tensors for
``install``, pending reads for ``keepalive`` and routes for ``eval``.

MLX arrays are lazy, so ``layer`` covers building the layer's graph and
every wait inside it, not the GPU work that graph triggers later.

Off unless the variable is set; each site then costs one module-level bool.
The counters belong to the process, so requests that run at the same time
share one file.
"""

from __future__ import annotations

import json
import os
import threading
import time

SPANS = ("layer", "routing", "ensure", "read", "install", "keepalive", "eval")

PATH = os.environ.get("OMLX_MOE_OFFLOAD_TIMELINE") or None
ON = PATH is not None

# A 256-token decode at 48 layers with about three misses per layer stays
# under 150 thousand spans; the cap bounds a prefill that reads every expert.
_CAPACITY = 1 << 21
_lock = threading.Lock()
_events: list[tuple[str, int, int, int, int]] = []
_dropped = 0

now = time.perf_counter_ns


def record(span: str, layer: int, count: int, start_ns: int) -> None:
    """Close a span that began at ``start_ns``. IO workers call this concurrently."""
    global _dropped
    end_ns = now()
    with _lock:
        if len(_events) < _CAPACITY:
            _events.append((span, layer, count, start_ns, end_ns))
        else:
            _dropped += 1


def json_line(span: str, layer: int, count: int, start_ns: int, end_ns: int) -> str:
    return (
        json.dumps(
            {
                "span": span,
                "layer": layer,
                "count": count,
                "start_ns": start_ns,
                "end_ns": end_ns,
            },
            separators=(",", ":"),
        )
        + "\n"
    )


def flush() -> str | None:
    """Append the recorded spans to the file; return a one-line report."""
    global _events, _dropped
    if not ON:
        return None
    with _lock:
        events, _events = _events, []
        dropped, _dropped = _dropped, 0
    if not events:
        return None
    try:
        with open(PATH, "a", encoding="utf-8") as f:
            f.writelines(json_line(*event) for event in events)
    except OSError as error:
        return f"could not write {PATH}: {error}"
    return f"{len(events)} spans, {dropped} dropped -> {PATH}"
