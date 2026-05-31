# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the admin cluster-router management API (cluster_routes.py).

Covers secret masking round-trip, config validation, atomic read/write, and the
four endpoints via FastAPI TestClient with require_admin bypassed and
launchctl/httpx mocked (no real launchd, no real router).
"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import cluster_routes as cr
from omlx.admin.auth import require_admin


def _cfg(**over):
    base = {
        "listen": "0.0.0.0:9000",
        "router_api_key": "secret-router",
        "affinity_hysteresis": 1.0,
        "mem_soft_floor": 0.05,
        "mem_penalty": 10.0,
        "backends": [
            {"name": "m2max", "base_url": "http://127.0.0.1:8000", "api_key": "k2", "weight": 1.0},
            {"name": "m5max", "base_url": "http://10.0.0.1:8000", "api_key": "k5", "weight": 1.3},
        ],
    }
    base.update(over)
    return base


@pytest.fixture
def client(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cluster.json"
    monkeypatch.setenv("OMLX_CLUSTER_CONFIG", str(cfg_path))
    app = FastAPI()
    app.include_router(cr.cluster_router)
    app.dependency_overrides[require_admin] = lambda: True
    return TestClient(app), cfg_path


# ---- masking ----
def test_mask_hides_all_secrets():
    masked = cr._mask(_cfg())
    assert masked["router_api_key"] == cr.SECRET_SENTINEL
    assert all(b["api_key"] == cr.SECRET_SENTINEL for b in masked["backends"])
    # non-secret fields untouched
    assert masked["backends"][1]["weight"] == 1.3
    assert masked["listen"] == "0.0.0.0:9000"


def test_mask_leaves_empty_secret_empty():
    cfg = _cfg(router_api_key="")
    cfg["backends"][0]["api_key"] = ""
    masked = cr._mask(cfg)
    assert masked["router_api_key"] == ""
    assert masked["backends"][0]["api_key"] == ""


def test_unmask_restores_stored_secrets_by_name():
    existing = _cfg()
    incoming = cr._mask(existing)              # all secrets are sentinels
    incoming["backends"][0]["weight"] = 2.0    # a real edit
    merged = cr._unmask(incoming, existing)
    assert merged["router_api_key"] == "secret-router"
    assert {b["name"]: b["api_key"] for b in merged["backends"]} == {"m2max": "k2", "m5max": "k5"}
    assert merged["backends"][0]["weight"] == 2.0


def test_unmask_takes_new_secret_when_changed():
    existing = _cfg()
    incoming = cr._mask(existing)
    incoming["backends"][0]["api_key"] = "rotated"   # user typed a new key
    merged = cr._unmask(incoming, existing)
    assert merged["backends"][0]["api_key"] == "rotated"
    assert merged["backends"][1]["api_key"] == "k5"  # untouched stays stored


def test_unmask_renamed_backend_with_sentinel_blanks_not_leaks():
    # Security: a sentinel on a backend whose NAME has no stored match must NOT
    # inherit some other backend's key by position -- it must blank, forcing
    # re-entry. Prevents cross-backend secret leak under delete+rename.
    existing = _cfg()
    incoming = cr._mask(existing)
    incoming["backends"][0]["name"] = "renamed-host"   # name no longer matches
    merged = cr._unmask(incoming, existing)
    assert merged["backends"][0]["api_key"] == ""       # blanked, not "k2"/"k5"
    assert merged["backends"][1]["api_key"] == "k5"     # untouched name still resolves


# ---- validation ----
def test_validate_accepts_good_config():
    cr._validate(_cfg())  # no raise


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: c.update(listen="0.0.0.0"), "listen"),
    (lambda c: c.update(backends=[]), "backend"),
    (lambda c: c["backends"][0].pop("base_url"), "base_url"),
    (lambda c: c["backends"][0].update(weight=0), "weight"),
    (lambda c: c["backends"][0].update(weight=-1), "weight"),
    (lambda c: c["backends"][0].pop("name"), "name"),
])
def test_validate_rejects_bad_config(mutate, msg):
    from fastapi import HTTPException
    cfg = _cfg()
    mutate(cfg)
    with pytest.raises(HTTPException) as ei:
        cr._validate(cfg)
    assert ei.value.status_code == 400
    assert msg in ei.value.detail.lower()


def test_validate_rejects_duplicate_backend_names():
    from fastapi import HTTPException
    cfg = _cfg()
    cfg["backends"][1]["name"] = "m2max"
    with pytest.raises(HTTPException) as ei:
        cr._validate(cfg)
    assert "duplicate" in ei.value.detail.lower()


# ---- endpoints ----
def test_get_config_empty_when_no_file(client):
    c, _ = client
    r = c.get("/admin/api/cluster/config")
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is False
    assert body["config"] == {}


def test_save_then_get_roundtrips_and_masks(client):
    c, cfg_path = client
    r = c.post("/admin/api/cluster/config", json={"config": _cfg()})
    assert r.status_code == 200 and r.json()["ok"] is True
    # on disk the real secrets are present
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["router_api_key"] == "secret-router"
    assert on_disk["backends"][0]["api_key"] == "k2"
    # but GET masks them
    got = c.get("/admin/api/cluster/config").json()
    assert got["exists"] is True
    assert got["config"]["router_api_key"] == cr.SECRET_SENTINEL
    assert got["config"]["backends"][0]["api_key"] == cr.SECRET_SENTINEL


def test_save_with_sentinel_keeps_existing_secret(client):
    c, cfg_path = client
    c.post("/admin/api/cluster/config", json={"config": _cfg()})
    # client re-saves the masked config (secrets are sentinels) with a weight tweak
    masked = c.get("/admin/api/cluster/config").json()["config"]
    masked["backends"][0]["weight"] = 3.0
    r = c.post("/admin/api/cluster/config", json={"config": masked})
    assert r.status_code == 200
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["router_api_key"] == "secret-router"      # preserved
    assert on_disk["backends"][0]["api_key"] == "k2"          # preserved
    assert on_disk["backends"][0]["weight"] == 3.0            # applied


def test_save_rejects_invalid_config(client):
    c, _ = client
    bad = _cfg()
    bad["backends"][0]["weight"] = 0
    r = c.post("/admin/api/cluster/config", json={"config": bad})
    assert r.status_code == 400


def test_written_config_is_owner_only(client):
    import stat
    c, cfg_path = client
    c.post("/admin/api/cluster/config", json={"config": _cfg()})
    mode = stat.S_IMODE(cfg_path.stat().st_mode)
    assert mode == 0o600


def test_status_reports_router_down(client, monkeypatch):
    c, _ = client
    # Point at a port guaranteed closed so the health probe is refused, even if a
    # real router happens to be running on :9000 on this dev host.
    c.post("/admin/api/cluster/config", json={"config": _cfg(listen="127.0.0.1:59599")})
    monkeypatch.setattr(cr, "_agent_state", lambda: {"installed": True, "loaded": False, "running": False, "pid": None})
    r = c.get("/admin/api/cluster/status")
    assert r.status_code == 200
    body = r.json()
    assert body["agent"]["installed"] is True
    assert body["health"] is None
    assert body["error"]  # connection refused surfaced
    assert body["listen"] == "127.0.0.1:59599"


def test_control_actions_invoke_launchctl(client, tmp_path, monkeypatch):
    c, _ = client
    plist = tmp_path / "agent.plist"
    plist.write_text("<plist/>")                       # a real, existing plist
    calls = []
    monkeypatch.setattr(cr, "_launchctl", lambda args: (calls.append(args) or (0, "", "")))
    monkeypatch.setattr(cr, "_plist_path", lambda: plist)
    monkeypatch.setattr(cr, "_agent_state", lambda: {"installed": True, "loaded": True, "running": True, "pid": 1})
    for action in ("start", "stop", "restart"):
        r = c.post("/admin/api/cluster/control", json={"action": action})
        assert r.status_code == 200, (action, r.text)
        assert r.json()["action"] == action
    assert calls, "launchctl was invoked"


def test_control_rejects_unknown_action(client, tmp_path, monkeypatch):
    c, _ = client
    plist = tmp_path / "agent.plist"
    plist.write_text("<plist/>")
    monkeypatch.setattr(cr, "_plist_path", lambda: plist)
    r = c.post("/admin/api/cluster/control", json={"action": "frobnicate"})
    assert r.status_code == 400


def test_control_start_requires_installed_plist(client, tmp_path, monkeypatch):
    c, _ = client
    monkeypatch.setattr(cr, "_plist_path", lambda: tmp_path / "missing.plist")  # does not exist
    r = c.post("/admin/api/cluster/control", json={"action": "start"})
    assert r.status_code == 404
