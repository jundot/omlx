# SPDX-License-Identifier: Apache-2.0
"""DFlash routing decision metric writer with size guard.

Path A produces one routing decision per request (dflash vs bg, plus reason
and a sampled KV pressure). D3 lands a minimal append-only jsonl writer so
the choice can be analyzed offline (acceptance ratio under load, KV pressure
trip rates, etc.) without taking a dependency on a metrics backend.

Guards:
- Size guard: once the file exceeds ``DFLASH_METRIC_MAX_SIZE`` bytes
  (default 500 MiB) we stop writing and log once. Rotation is a deferred
  D3.x story — for now, ``rm`` or archive the file to reset.
- Test/disable guard: ``DFLASH_METRIC_DISABLE=1`` short-circuits the write.
  Used by pytest to avoid filesystem writes during unit tests that
  exercise ``_route()`` directly.
- I/O errors are swallowed at DEBUG level — metric writes must never
  break the inference path.

Spec reference: ``docs/dflash-pathA-spec.md`` §2 (A.1 record schema) and
§11.1 (size guard rationale).
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 500 * 1024 * 1024  # 500 MiB
_lock = threading.Lock()
_size_warned = False


def get_metric_path() -> Path:
    """Resolve the dflash routing metric file path.

    Honors ``DFLASH_METRIC_DIR`` for tests / staging environments that
    want isolation. Directory is created on demand so a fresh box can
    start emitting metrics without a manual ``mkdir``.
    """
    base = Path(
        os.environ.get(
            "DFLASH_METRIC_DIR",
            str(Path.home() / ".omlx" / "metrics"),
        )
    )
    base.mkdir(parents=True, exist_ok=True)
    return base / "dflash_routing.jsonl"


def write_routing_decision(record: dict[str, Any]) -> None:
    """Append one routing decision to the jsonl log.

    Args:
        record: A dict with at least ``ts``, ``model_name``, ``routed_to``,
            ``reason``. ``ctx_len``, ``active_count``, ``kv_usage_ratio``
            are expected by downstream analysis but not required here —
            the writer just dumps what the caller passes.

    Returns:
        None. Errors are logged at DEBUG; the inference path never sees them.
    """
    global _size_warned
    if os.environ.get("DFLASH_METRIC_DISABLE"):
        return

    max_size = int(os.environ.get("DFLASH_METRIC_MAX_SIZE", _DEFAULT_MAX_SIZE))
    try:
        path = get_metric_path()
        with _lock:
            if path.exists() and path.stat().st_size > max_size:
                if not _size_warned:
                    logger.warning(
                        "dflash_routing.jsonl exceeded %d bytes — STOPPED "
                        "writing routing metrics. Rotate the file (rm or "
                        "archive) or set DFLASH_METRIC_MAX_SIZE higher.",
                        max_size,
                    )
                    _size_warned = True
                return
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.debug("dflash metric write failed: %s", e)
