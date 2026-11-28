#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only check that every cluster Mac runs the same assistant route surface.

Why: the coordinator must answer `/admin/api/cluster/assistant/operations/*` for
the cluster-wide operation lock, and the assistant routes for tuning. An older
build on one Mac answers 405 (route absent) where a current build answers 401
(auth required) or 422 (bad body). Detecting the skew early beats a mysterious
"the coordinator did not accept the request: Method Not Allowed".

This never sends a key and never writes anything. A route is "present" when the
status is not 404/405; 401/403/422 all mean the route exists (it answered before
auth/validation).

Usage:
    python scripts/check_peer_versions.py [--port 8099]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

#: A route exists when it answers before validation/auth. 404/405 mean absent.
PRESENT = {200, 401, 403, 422}
ROUTES = (
    ("/admin/api/cluster/stage", "POST"),
    ("/admin/api/cluster/plan", "POST"),
    ("/admin/api/cluster/assistant/operations/active", "POST"),
    ("/admin/api/cluster/assistant/tuning-preset", "POST"),
)


def _base_path() -> Path:
    override = os.environ.get("OMLX_BASE_PATH")
    return Path(override) if override else Path.home() / ".omlx"


def _local_port(default: int) -> int:
    try:
        settings = json.loads((_base_path() / "settings.json").read_text())
        return int(settings.get("server", {}).get("port") or default)
    except (OSError, ValueError, TypeError):
        return default


def _nodes(port: int) -> list[tuple[str, str, str]]:
    """(label, base_url, source) for this Mac and every paired peer."""

    base = _base_path() / "cluster"
    nodes: list[tuple[str, str, str]] = []
    try:
        identity = json.loads((base / "identity.json").read_text())
        local_id = str(identity.get("node_id") or "local")
    except (OSError, ValueError):
        local_id = "local"
    nodes.append((f"{local_id[:8]} (this Mac)", f"http://127.0.0.1:{port}", "local"))
    try:
        raw = json.loads((base / "devices.json").read_text())
    except (OSError, ValueError):
        return nodes
    if isinstance(raw, dict):
        peers = raw.get("devices") or raw.get("paired") or []
    else:
        peers = raw
    for peer in peers or ():
        if not isinstance(peer, dict):
            continue
        node_id = str(peer.get("node_id") or "")[:8]
        addrs = [str(a) for a in (peer.get("last_addrs") or [])]
        tb = [
            str(item.get("ip"))
            for item in (peer.get("addrs") or [])
            if isinstance(item, dict) and item.get("if_type") == "thunderbolt"
        ]
        ip = (tb or [a for a in addrs if a.startswith("172.16.")] or addrs or [""])[0]
        if not ip:
            continue
        http_port = int(peer.get("http_port") or port)
        name = peer.get("friendly_name") or node_id
        nodes.append((f"{node_id} {name}", f"http://{ip}:{http_port}", "peer"))
    return nodes


def _status(base: str, path: str, method: str) -> int | None:
    body = b"{}" if method == "POST" else None
    request = urllib.request.Request(base + path, data=body, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=4) as reply:
            return int(getattr(reply, "status", 200))
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, OSError):
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args(argv)
    port = _local_port(args.port)

    skew = False
    for label, base, source in _nodes(port):
        health = _status(base, "/health", "GET")
        if health != 200:
            if source == "local":
                print(f"{label:42s} unreachable (this Mac's server is not running)")
            else:
                print(f"{label:42s} unreachable (health={health})")
            continue
        present = {
            path: (_status(base, path, method) in PRESENT) for path, method in ROUTES
        }
        missing = [path for path, ok in present.items() if not ok]
        flag = "OK" if not missing else "OLDER BUILD"
        print(f"{label:42s} {flag}")
        for path in missing:
            skew = True
            print(f"    missing (405/404): {path}")
    if skew:
        print(
            "\nSkew detected. A coordinator on an older build cannot hold the "
            "cluster-wide operation lock, and the tuning routes are absent there. "
            "Update every Mac to the same build (see "
            "docs/cluster-assistant/PEER-VERSIONS.md). Read-only check; nothing "
            "was changed."
        )
        return 1
    print("\nEvery reachable Mac exposes the assistant route surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
