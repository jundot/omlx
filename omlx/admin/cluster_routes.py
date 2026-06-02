# SPDX-License-Identifier: Apache-2.0
"""Admin API for managing the cluster router (see omlx/cluster/router.py).

The cluster router is a SEPARATE process -- on macOS a launchd agent labeled
``com.flyto.mlx.cluster-router``. This admin API never imports the router; it
only:

- reads/writes the router's JSON config (``<base_path>/cluster.json``),
- proxies the router's ``/health`` for a live status view,
- controls the launchd agent via ``launchctl``.

Secrets (``router_api_key`` and each backend's ``api_key``) are never sent to the
browser in clear: GET replaces a stored secret with the sentinel ``__SAVED__``;
POST treats an incoming ``__SAVED__`` as "keep the stored value", anything else
as a new secret. The on-disk config keeps the real values.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .auth import require_admin

logger = logging.getLogger(__name__)

cluster_router = APIRouter(prefix="/admin/api/cluster", tags=["admin"])

LAUNCHD_LABEL = "com.flyto.mlx.cluster-router"
SECRET_SENTINEL = "__SAVED__"  # client round-trips this to keep a stored secret
_SECRET_KEYS = ("router_api_key",)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _base_path() -> Path:
    """Resolve the omlx base path (where settings.json / cluster.json live)."""
    try:
        from .routes import _get_global_settings

        gs = _get_global_settings() if _get_global_settings else None
        bp = getattr(gs, "base_path", None) if gs is not None else None
        if bp:
            return Path(bp)
    except Exception:  # pragma: no cover - defensive, getter not wired in tests
        pass
    from ..settings import resolve_default_base_path

    return resolve_default_base_path()


def _config_path() -> Path:
    """The cluster.json path the router reads. Honor OMLX_CLUSTER_CONFIG if the
    admin process has it set, else <base_path>/cluster.json (the router's
    default), so admin writes where the router reads."""
    env = os.environ.get("OMLX_CLUSTER_CONFIG")
    return Path(env) if env else _base_path() / "cluster.json"


def _plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


# --------------------------------------------------------------------------- #
# Config read / write
# --------------------------------------------------------------------------- #
def _read_config_raw() -> dict[str, Any]:
    p = _config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"cluster.json unreadable: {exc}")


def _mask(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with secrets replaced by SECRET_SENTINEL when present."""
    out = json.loads(json.dumps(cfg))  # deep copy via round-trip
    for k in _SECRET_KEYS:
        if out.get(k):
            out[k] = SECRET_SENTINEL
    for b in out.get("backends", []) or []:
        if isinstance(b, dict) and b.get("api_key"):
            b["api_key"] = SECRET_SENTINEL
    return out


def _unmask(incoming: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    """Replace any SECRET_SENTINEL in ``incoming`` with the stored secret from
    ``existing``, matching per-backend keys by ``name`` ONLY.

    A sentinel on a backend whose name has no stored match resolves to "" (the
    user must re-enter the key). We deliberately do NOT fall back to positional
    matching: under a delete+reorder that would graft one backend's key onto a
    different backend's URL -- a cross-backend secret leak. An empty key is a
    visible, fixable failure; a leaked key is not."""
    merged = json.loads(json.dumps(incoming))
    for k in _SECRET_KEYS:
        if merged.get(k) == SECRET_SENTINEL:
            merged[k] = existing.get(k, "")
    old_by_name = {
        b.get("name"): b.get("api_key", "")
        for b in (existing.get("backends") or [])
        if isinstance(b, dict)
    }
    for b in merged.get("backends", []) or []:
        if isinstance(b, dict) and b.get("api_key") == SECRET_SENTINEL:
            b["api_key"] = old_by_name.get(b.get("name"), "")
    return merged


def _validate(cfg: dict[str, Any]) -> None:
    """Mirror omlx/cluster/router.py load_config() constraints. Raise 400 on
    violation."""
    listen = cfg.get("listen", "0.0.0.0:9000")
    host, _, port = str(listen).partition(":")
    if not port or not port.isdigit():
        raise HTTPException(status_code=400, detail=f"invalid listen {listen!r}; expected host:port")
    backends = cfg.get("backends")
    if not isinstance(backends, list) or not backends:
        raise HTTPException(status_code=400, detail="at least one backend is required")
    seen: set[str] = set()
    for b in backends:
        if not isinstance(b, dict):
            raise HTTPException(status_code=400, detail="each backend must be an object")
        name = b.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="each backend needs a name")
        if name in seen:
            raise HTTPException(status_code=400, detail=f"duplicate backend name {name!r}")
        seen.add(name)
        if not b.get("base_url"):
            raise HTTPException(status_code=400, detail=f"backend {name!r} needs a base_url")
        try:
            w = float(b.get("weight", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"backend {name!r} weight must be a number")
        if w <= 0:
            raise HTTPException(status_code=400, detail=f"backend {name!r} weight must be > 0")
    for fld, lo in (("poll_interval", 0.0), ("affinity_hysteresis", 0.0),
                    ("mem_soft_floor", 0.0), ("mem_penalty", 0.0)):
        if fld in cfg and cfg[fld] is not None:
            try:
                if float(cfg[fld]) < lo:
                    raise HTTPException(status_code=400, detail=f"{fld} must be >= {lo}")
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"{fld} must be a number")


def _write_config_atomic(cfg: dict[str, Any]) -> None:
    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    data = json.dumps(cfg, indent=2)
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(p))
    try:
        os.chmod(str(p), 0o600)  # config holds secrets
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# launchd control
# --------------------------------------------------------------------------- #
def _launchctl(args: list[str]) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            ["launchctl", *args], capture_output=True, text=True, timeout=15,
        )
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        raise HTTPException(status_code=501, detail="launchctl not available (non-macOS host)")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="launchctl timed out")


def _agent_state() -> dict[str, Any]:
    """installed (plist present), loaded (in launchd), running (has a PID)."""
    installed = _plist_path().exists()
    rc, out, _ = _launchctl(["list", LAUNCHD_LABEL])
    loaded = rc == 0
    pid = None
    if loaded:
        for line in out.splitlines():
            s = line.strip()
            if s.startswith('"PID"'):
                tail = s.split("=", 1)[1].strip().rstrip(";").strip()
                if tail.isdigit():
                    pid = int(tail)
                break
    return {"installed": installed, "loaded": loaded, "running": pid is not None, "pid": pid}


def _uid_domain() -> str:
    return f"gui/{os.getuid()}/{LAUNCHD_LABEL}"


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
class ClusterConfigRequest(BaseModel):
    config: dict[str, Any]


class ClusterControlRequest(BaseModel):
    action: str  # start | stop | restart


@cluster_router.get("/config")
async def get_cluster_config(_: bool = Depends(require_admin)):
    cfg = _read_config_raw()
    return {"exists": bool(cfg), "path": str(_config_path()), "config": _mask(cfg)}


@cluster_router.post("/config")
async def save_cluster_config(req: ClusterConfigRequest, _: bool = Depends(require_admin)):
    existing = _read_config_raw()
    merged = _unmask(req.config, existing)
    _validate(merged)
    _write_config_atomic(merged)
    logger.info("cluster.json updated via admin (%d backends)", len(merged.get("backends", [])))
    return {"ok": True, "path": str(_config_path()),
            "note": "restart the cluster router for changes to take effect"}


@cluster_router.get("/status")
async def get_cluster_status(_: bool = Depends(require_admin)):
    agent = _agent_state()
    cfg = _read_config_raw()
    listen = str(cfg.get("listen", "0.0.0.0:9000"))
    _, _, port = listen.partition(":")
    port = port or "9000"
    health: Any = None
    error = ""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            r = await client.get(f"http://127.0.0.1:{port}/health")
            r.raise_for_status()
            health = r.json()
    except Exception as exc:  # router down / refused / bad response
        error = f"{type(exc).__name__}: {exc}"
    return {"agent": agent, "listen": listen, "health": health, "error": error}


@cluster_router.post("/control")
async def control_cluster(req: ClusterControlRequest, _: bool = Depends(require_admin)):
    action = (req.action or "").lower()
    plist = str(_plist_path())
    if action in ("start", "restart") and not _plist_path().exists():
        raise HTTPException(
            status_code=404,
            detail=f"LaunchAgent not installed at {plist}. Create the plist first.",
        )

    if action == "start":
        rc, out, err = _launchctl(["load", "-w", plist])
        if rc != 0 and "already loaded" not in (err + out).lower():
            # Already loaded -> just (re)kick it.
            _launchctl(["kickstart", _uid_domain()])
    elif action == "stop":
        rc, out, err = _launchctl(["unload", plist])
        if rc != 0:
            _launchctl(["bootout", _uid_domain()])
    elif action == "restart":
        rc, out, err = _launchctl(["kickstart", "-k", _uid_domain()])
        if rc != 0:
            # Not loaded yet -> load it.
            _launchctl(["load", "-w", plist])
    else:
        raise HTTPException(status_code=400, detail=f"unknown action {action!r}; use start|stop|restart")

    return {"ok": True, "action": action, "agent": _agent_state()}
