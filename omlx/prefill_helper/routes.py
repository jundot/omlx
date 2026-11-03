# SPDX-License-Identifier: Apache-2.0
"""Admin API for the Prefill Accelerator, on both Macs.

This Mac: candidate helpers for a model, the model's helper status and the
copy / retry / stop actions. The helper Mac: what it is lending and a Stop.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException

from .. import __version__
from ..cluster.registry import get_device_registry
from . import launcher
from .adapters import adapter_for
from .manager import fit_layers
from .worker import active_marker_path

router = APIRouter(prefix="/admin/api/prefill-helper", tags=["prefill-helper"])

_get_engine_pool: Callable[[], Any] = lambda: None  # noqa: E731


def set_engine_pool_getter(getter: Callable[[], Any]) -> None:
    global _get_engine_pool
    _get_engine_pool = getter


def _entry(model_id: str) -> Any:
    pool = _get_engine_pool()
    entry = pool.get_entry(model_id) if pool is not None else None
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return entry


def _manager(model_id: str) -> Any:
    engine = _entry(model_id).engine
    manager = getattr(engine, "prefill_helper", None)
    if manager is None:
        raise HTTPException(
            status_code=409, detail="Load the model to use the Prefill Accelerator."
        )
    return manager


def _candidate(device: dict[str, Any], model_path: str, adapter: Any) -> dict:
    caps = device.get("caps") or {}
    row = {
        "node_id": device["node_id"],
        "name": device["friendly_name"],
        "chip": caps.get("chip"),
        "ram_gb": caps.get("ram_gb"),
    }
    try:
        target = launcher.resolve_target(device["node_id"])
        info = launcher.remote_info(target)
    except launcher.LaunchError as error:
        return {**row, "online": False, "reason": str(error)}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {**row, "online": False, "reason": str(error)}
    shared, per_layer = adapter.memory_plan(model_path)
    held = fit_layers(shared, per_layer, int(info["budget"]), len(per_layer))
    return {
        **row,
        "online": True,
        "link": target.link,
        "device": info.get("device"),
        "version": info.get("version"),
        "version_ok": info.get("version") == __version__,
        "layers_fit": held,
        "budget": info.get("budget"),
    }


@router.get("/candidates")
async def candidates(model_id: str) -> dict:
    """Paired Macs and whether each can help this model."""
    entry = _entry(model_id)
    adapter = adapter_for(
        (getattr(entry, "config_model_type", "") or "").replace("-", "_").lower()
    )
    if adapter is None:
        raise HTTPException(
            status_code=400, detail="This model has no Prefill Accelerator."
        )
    devices = get_device_registry().paired()
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max(1, min(len(devices), 4))) as pool:
        rows = await asyncio.gather(
            *[
                loop.run_in_executor(pool, _candidate, d, entry.model_path, adapter)
                for d in devices
            ]
        )
    return {"version": __version__, "devices": rows}


@router.get("/models/{model_id}")
async def model_status(model_id: str) -> dict:
    engine = _entry(model_id).engine
    manager = getattr(engine, "prefill_helper", None)
    return manager.status() if manager is not None else {"state": "unloaded"}


@router.post("/models/{model_id}/{action}")
async def model_action(model_id: str, action: str) -> dict:
    manager = _manager(model_id)
    actions = {"copy": manager.copy_files, "retry": manager.retry, "stop": manager.stop}
    if action not in actions:
        raise HTTPException(status_code=404, detail=f"Unknown action: {action}")
    actions[action]()
    return manager.status()


def _lending() -> dict | None:
    """The helper running on this Mac, if its process is alive."""
    try:
        marker = json.loads(active_marker_path().read_text())
        os.kill(int(marker["pid"]), 0)
    except (OSError, ValueError, KeyError):
        return None
    return marker


@router.get("/lending")
async def lending() -> dict:
    marker = _lending()
    return {"active": marker is not None, **(marker or {})}


@router.post("/lending/stop")
async def stop_lending() -> dict:
    marker = _lending()
    if marker is None:
        return {"active": False}
    # The worker tells its host it was stopped here, so the host stays off.
    os.kill(int(marker["pid"]), signal.SIGTERM)
    return {"active": False, "stopped": True}
