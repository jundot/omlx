# SPDX-License-Identifier: Apache-2.0
"""Admin view of remote prefill: each loaded model that uses it, with its state and latest handoff."""

from __future__ import annotations

from typing import Any


def _scheduler(engine: Any) -> Any:
    """The engine's scheduler, through its engine core when it has one."""
    core = getattr(getattr(engine, "_engine", None), "engine", None)
    return getattr(core, "scheduler", None) or getattr(engine, "scheduler", None)


def remote_prefill_status(pool: Any) -> dict[str, Any]:
    """Status of every loaded model whose scheduler prefills remotely."""
    models = []
    for model_id in pool.get_loaded_model_ids():
        entry = pool.get_entry(model_id)
        status = getattr(
            _scheduler(getattr(entry, "engine", None)), "remote_prefill_status", None
        )
        state = status() if callable(status) else None
        if state is not None:
            models.append({"model_id": model_id, **state})
    return {"models": models}
