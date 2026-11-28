# SPDX-License-Identifier: Apache-2.0
"""Seams between the cluster dashboard scripts and the cluster admin routes."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DASHBOARD_SCRIPTS = (
    ROOT / "omlx_web" / "static" / "js" / "dashboard.js",
    ROOT / "omlx_web" / "static" / "js" / "cluster.js",
)

_PREFIX = "/admin/api/cluster"
_CLUSTER_URL = re.compile(
    re.escape(_PREFIX)
    + r"(?P<path>(?:\$\{[^{}]*(?:\([^)]*\))?[^{}]*\}|[A-Za-z0-9/_\-.])*)"
)


def _registered_routes() -> set[str]:
    from omlx.cluster import routes
    from omlx.cluster.assistant import routes as assistant_routes

    # The tab now drives both the existing cluster admin router and the
    # Cluster Assistant router (Screen-3 minimal slice). A URL is real when
    # either registers it, so keep this seam honest across both mounts.
    registered = {
        re.sub(r"\{[^{}]+\}", "{parameter}", route.path)
        for router in (routes.router, assistant_routes.router)
        for route in router.routes
        if getattr(route, "path", None)
    }
    return registered


def _js_called_paths() -> set[str]:
    """Cluster URLs the dashboard builds, normalised to their route shape."""

    called = set()
    for script in _DASHBOARD_SCRIPTS:
        for match in _CLUSTER_URL.finditer(script.read_text()):
            path = match.group("path").split("?")[0]
            # Any interpolated segment stands for a path parameter.
            path = re.sub(r"\$\{[^{}]*(?:\([^)]*\))?[^{}]*\}", "{parameter}", path)
            path = path.rstrip("/") if path not in ("", "/") else path
            called.add(_PREFIX + path)
    return called


def test_every_cluster_url_the_dashboard_calls_is_a_real_route():
    """A typo or a renamed endpoint here is a 404 no unit test would notice."""

    missing = _js_called_paths() - _registered_routes()
    assert not missing, (
        f"dashboard.js calls cluster endpoints that are not registered: "
        f"{sorted(missing)}"
    )


def test_no_cluster_route_is_unreachable_from_the_dashboard():
    """Every route should have a caller.

    A route with no caller is either dead or a feature that was never wired up —
    both worth knowing about.
    """

    # These are compatibility/manual operator APIs retained after the v1
    # dashboard console was removed. Cluster v2 uses discovery/pairing,
    # autoconfigure and the deployment lifecycle; scripts and older clients
    # may still use these explicit low-level probes. CUDA worker enrollment,
    # the ConnectX/RDMA link checks and the diagnostics report have had no
    # dashboard panel since the advanced tools were removed.
    allowed_without_caller: set[str] = {
        "/admin/api/cluster/cuda-fabric/verify",
        "/admin/api/cluster/diagnostics",
        "/admin/api/cluster/join-keys",
        "/admin/api/cluster/join-keys/{parameter}",
        "/admin/api/cluster/join-status",
        "/admin/api/cluster/rdma-links",
        "/admin/api/cluster/rdma-links/verify",
        "/admin/api/cluster/backend-selection",
        "/admin/api/cluster/collective-smoke",
        "/admin/api/cluster/discover",
        "/admin/api/cluster/fabric",
        "/admin/api/cluster/guidance",
        "/admin/api/cluster/incidents",
        "/admin/api/cluster/incidents/{parameter}/dismiss",
        "/admin/api/cluster/link-setup",
        "/admin/api/cluster/link-status",
        "/admin/api/cluster/pairing-token",
        "/admin/api/cluster/peer-health",
        "/admin/api/cluster/pipeline-smoke",
        "/admin/api/cluster/plan",
        "/admin/api/cluster/ssh-key",
        "/admin/api/cluster/ssh-key/exchange",
        "/admin/api/cluster/ssh-key/exchange-token",
        "/admin/api/cluster/ssh-key/generate",
        "/admin/api/cluster/ssh-key/store-keychain",
        "/admin/api/cluster/status",
        "/admin/api/cluster/transports",
        "/admin/api/cluster/verify-pairing-token",
        "/admin/api/cluster/worker-smoke",
        # Not for the browser: the elected coordinator answers another origin's
        # "are you already running an apply/start?" over the admin seam (the
        # cluster-wide operation lock). The card still calls the eight it owns,
        # including the free-space relay (the live-run notes).
        "/admin/api/cluster/assistant/operations/active",
        # Not for the browser either: the origin mirrors its running operation
        # onto the elected coordinator over the same admin seam, so a second
        # origin sees the cluster-wide lock (the apply/start design notes). Server-to-server.
        "/admin/api/cluster/assistant/operations/mirror",
        # The automatic constellation search loop is API-first; the dashboard UI
        # for it comes later. It only calls the assistant's own apply/start/
        # observe/stop seams, never a raw cluster route from the browser.
        "/admin/api/cluster/assistant/search",
        # Fine-tuning the winner is API-first too; it only runs completions on
        # the winning coordinator through the assistant's own seam.
        "/admin/api/cluster/assistant/tune",
        # Read-only: the saved tuning preset for an operation's model + node set.
        # The card shows the preset key the tune/tune-redeploy responses already
        # return; this route serves API clients and a future on-load display, so
        # the browser does not call it yet.
        "/admin/api/cluster/assistant/tuning-preset",
        # Not for the browser: the assistant origin asks the elected
        # coordinator's own route for the synthetic compute probe after apply
        # (Phase B1). The browser only sees the assistant's session/operation.
        "/admin/api/cluster/performance-probe",
    }
    unreachable = _registered_routes() - _js_called_paths() - allowed_without_caller
    assert not unreachable, (
        f"cluster routes nothing calls: {sorted(unreachable)} — wire them up or "
        f"delete them"
    )


def test_fetch_calls_never_use_a_params_option():
    """`fetch(url, {params})` is silently ignored; query strings must be built.

    This exact mistake made every pairing and key-exchange call return 422 while
    the suite stayed green.
    """

    source = "\n".join(script.read_text() for script in _DASHBOARD_SCRIPTS)
    offenders = []
    for index, line in enumerate(source.splitlines(), start=1):
        if re.search(r"^\s*params:\s*\{", line):
            window = "\n".join(source.splitlines()[max(0, index - 6) : index])
            if "fetch(" in window:
                offenders.append(index)
    assert not offenders, (
        f"dashboard.js:{offenders} pass `params` to fetch(); fetch ignores it — "
        f"use URLSearchParams and put it in the URL"
    )
