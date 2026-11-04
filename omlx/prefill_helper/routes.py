# SPDX-License-Identifier: Apache-2.0
"""Admin API for the Prefill Accelerator, on both Macs.

This Mac: candidate helpers for a model, the model's helper status and the
copy / retry / stop actions, also while files are copied before the model
loads. The helper Mac: what it is lending and a Stop.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from .. import __version__
from ..cluster.registry import get_device_registry
from . import launcher
from .adapters import adapter_for
from .lending import active_lending, recorded_host
from .manager import (
    computer_name,
    file_prep,
    fit_layers,
    helper_budget,
    prepare,
    this_host,
)

router = APIRouter(prefix="/admin/api/prefill-helper", tags=["prefill-helper"])

_get_engine_pool: Callable[[], Any] = lambda: None  # noqa: E731
# The helper sends up to ~180 MB/s on the tested models: 5 GbE carries that
# with room to spare, 2.5 GbE with little.
_FULL_SPEED_MBPS = 5000


def set_engine_pool_getter(getter: Callable[[], Any]) -> None:
    global _get_engine_pool
    _get_engine_pool = getter


def _entry(model_id: str) -> Any:
    pool = _get_engine_pool()
    entry = pool.get_entry(model_id) if pool is not None else None
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    return entry


def _adapter(entry: Any) -> Any:
    model_type = getattr(entry, "config_model_type", "") or ""
    return adapter_for(model_type.replace("-", "_").lower())


def prepare_unloaded(entry: Any, settings: Any) -> None:
    """Copy the helper's files for a saved choice before the model loads."""
    adapter = _adapter(entry)
    if adapter is None or getattr(entry.engine, "prefill_helper", None) is not None:
        return
    prepare(
        entry.model_path,
        adapter,
        getattr(settings, "prefill_helper_node_id", None),
        getattr(settings, "prefill_helper_layers", None),
    )


def helper_status(entry: Any) -> dict | None:
    """The loaded model's helper, else the copy made before it loads."""
    manager = getattr(entry.engine, "prefill_helper", None)
    if manager is not None:
        return manager.status()
    prep = file_prep(entry.model_path)
    return prep.status() if prep is not None else None


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
    limit = adapter.split_limit(len(per_layer))
    held = fit_layers(shared, per_layer, helper_budget(info), limit)
    lending = info.get("lending")
    if lending:
        lending = {
            "host": lending.get("host"),
            "model": Path(str(lending.get("model") or "")).name,
        }
        if lending == {"host": computer_name(), "model": Path(model_path).name}:
            # The helper already runs for this model.
            lending = None
    return {
        **row,
        "online": True,
        "link": target.link,
        "device": info.get("device"),
        "version": info.get("version"),
        "version_ok": info.get("version") == __version__,
        "layers_fit": held,
        "budget": info.get("budget"),
        "on_battery": bool(info.get("on_battery")),
        "lending": lending,
    }


@router.get("/candidates")
async def candidates(model_id: str) -> dict:
    """Paired Macs and whether each can help this model."""
    entry = _entry(model_id)
    adapter = _adapter(entry)
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


def _check(node_id: str) -> dict:
    """The accelerator checks for one paired Mac, as rows of key, status, detail."""
    device = get_device_registry().get(node_id)
    if device is None:
        raise HTTPException(status_code=404, detail="This Mac is not paired.")
    rows = []
    try:
        target = launcher.resolve_target(node_id)
    except (launcher.LaunchError, OSError, subprocess.SubprocessError) as error:
        rows.append({"key": "ssh", "status": "fail", "detail": str(error)})
        return {"node_id": node_id, "name": device["friendly_name"], "rows": rows}
    rows.append({"key": "ssh", "status": "pass", "detail": target.ssh_target})
    fast = (
        target.link in ("thunderbolt", "rdma")
        or (target.speed_mbps or 0) >= _FULL_SPEED_MBPS
    )
    rows.append(
        {
            "key": "link",
            "status": "pass" if fast else "warn",
            "detail": target.link,
            "speed_mbps": target.speed_mbps,
            "fix": " ".join(target.warnings),
        }
    )
    try:
        info = launcher.remote_info(target, host=this_host())
    except (launcher.LaunchError, OSError, subprocess.SubprocessError) as error:
        rows.append({"key": "version", "status": "fail", "detail": str(error)})
        return {"node_id": node_id, "name": target.name, "rows": rows}
    version = info.get("version")
    rows.append(
        {
            "key": "version",
            "status": "pass" if version == __version__ else "fail",
            "detail": version,
            "expected": __version__,
        }
    )
    rows.append({"key": "memory", "status": "pass", "bytes": helper_budget(info)})
    rows.append({"key": "disk", "status": "pass", "bytes": info.get("disk_free")})
    rows.append(
        {"key": "power", "status": "warn" if info.get("on_battery") else "pass"}
    )
    return {"node_id": node_id, "name": target.name, "rows": rows}


@router.get("/check")
async def check(node_id: str) -> dict:
    """Whether a paired Mac can serve as a prefill helper."""
    return await asyncio.get_running_loop().run_in_executor(None, _check, node_id)


def _copies(node_id: str, delete: str | None = None) -> dict:
    if get_device_registry().get(node_id) is None:
        raise HTTPException(status_code=404, detail="This Mac is not paired.")
    try:
        target = launcher.resolve_target(node_id)
        if delete:
            launcher.delete_copies(target, [delete])
        return launcher.list_copies(target)
    except (launcher.LaunchError, OSError, subprocess.SubprocessError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.get("/copies")
async def copies(node_id: str) -> dict:
    """Model copies this Mac made on a paired Mac, and its free disk."""
    return await asyncio.get_running_loop().run_in_executor(None, _copies, node_id)


@router.post("/copies/delete")
async def delete_copy(node_id: str, name: str) -> dict:
    """Remove one model copy from a paired Mac (never its own model folders)."""
    return await asyncio.get_running_loop().run_in_executor(
        None, _copies, node_id, name
    )


@router.get("/models/{model_id}")
async def model_status(model_id: str) -> dict:
    return helper_status(_entry(model_id)) or {"state": "unloaded"}


@router.post("/models/{model_id}/{action}")
async def model_action(model_id: str, action: str) -> dict:
    if action not in ("copy", "retry", "stop"):
        raise HTTPException(status_code=404, detail=f"Unknown action: {action}")
    entry = _entry(model_id)
    manager = getattr(entry.engine, "prefill_helper", None)
    if manager is not None:
        {"copy": manager.copy_files, "retry": manager.retry, "stop": manager.stop}[
            action
        ]()
        return manager.status()
    prep = file_prep(entry.model_path)
    if prep is None:
        raise HTTPException(
            status_code=409, detail="Load the model to use the Prefill Accelerator."
        )
    if action == "copy":
        prep.copy_files()
    elif action == "stop":
        prep.cancel()
    else:
        prepare(entry.model_path, prep.adapter, prep.node_id, prep.layers, restart=True)
    return helper_status(entry) or {"state": "unloaded"}


@router.get("/lending")
async def lending() -> dict:
    """On the helper Mac: the running helper and the Mac that set this one up."""
    marker = active_lending()
    host = recorded_host()
    device = get_device_registry().get(host["node_id"]) if host else None
    # A host that is no longer paired here no longer uses this Mac.
    helps = (
        {
            "node_id": host["node_id"],
            "name": device.get("friendly_name") or host["name"],
        }
        if device
        else None
    )
    return {"active": marker is not None, **(marker or {}), "helps": helps}


@router.post("/lending/stop")
async def stop_lending() -> dict:
    marker = active_lending()
    if marker is None:
        return {"active": False}
    # The worker tells its host it was stopped here, so the host stays off.
    os.kill(int(marker["pid"]), signal.SIGTERM)
    return {"active": False, "stopped": True}
