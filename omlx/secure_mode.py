# SPDX-License-Identifier: Apache-2.0
"""Secure Server mode: the oMLX server runs as a network-isolated sandbox.

While the mode is on, the server is spawned by the privileged helper
(``oMLXSecureServerHelper``) as a dedicated macOS account (``_omlx``), and pf rules
scoped with ``user _omlx`` drop every packet that account sends or receives except
loopback. Nothing else on the Mac is matched: the rules name a single uid, so a
browser, a shell or a software update keeps its network while the server cannot
reach it. The server stays reachable from this Mac, because loopback is allowed and
it is pinned to a loopback bind address.

Enforcement is in the kernel, not here. This module only *reads* the root-owned
state file so the server can:

* answer "is the sandbox on?" for the dashboard (read-only status), and
* refuse egress-triggering endpoints with a 403 that explains why, instead of
  letting a download hang against a dropped packet.

That 403 is UX, and a second line of defence for the case where pf is not in fact
enforcing -- someone ran ``pfctl -d``, or the flag says on while the app is still
starting the server as its own user. The list below is deliberately short:
inference, health and task polling have to keep working. Anything new that reaches
the internet belongs here as well as being covered by the rules.

Missing, unreadable or malformed state means "off"; a user process cannot flip the
flag either way because the file is ``root:wheel 0644``.
"""

from __future__ import annotations

import json
import os
import threading

# Root-owned config directory. Created and written only by the privileged
# helper; world-readable so both the server and the macOS app can read it.
CONFIG_DIR = "/Library/Application Support/oMLX/secure-server"
STATE_PATH = CONFIG_DIR + "/state.json"
PF_ANCHOR_PATH = CONFIG_DIR + "/pf-anchor.conf"

# pf sub-anchor carrying the uid-scoped ruleset. Loading under ``com.apple/*`` means
# /etc/pf.conf is never edited.
PF_ANCHOR = "com.apple/900.omlx.secure"

# Testability hook: point the reader at a temp state file.
STATE_PATH_ENV = "OMLX_SECURE_MODE_STATE_PATH"

# The dashboard shows the same wording (``js.error.secure_server_blocked``). Local
# files are the way to get a model in while the mode is on: the Hub *upload*
# endpoints are blocked too, because pushing a model to Hugging Face is egress like
# any other, so there is no import button to point at.
SECURE_MODE_DETAIL = (
    "Secure Server mode is on, so the oMLX server has no network access of its own. "
    "Turn it off in the oMLX macOS app (requires administrator) to perform this "
    "action, or work with model files that are already on this Mac."
)

# Endpoints that need off-box egress. Inference, health and task-polling endpoints
# are deliberately absent: they must keep working while the server is sandboxed, and
# refusing them would break localhost inference for no gain.
BLOCKED_EXACT = frozenset(
    {
        ("POST", "/admin/api/hf/download"),
        ("GET", "/admin/api/hf/recommended"),
        ("GET", "/admin/api/hf/search"),
        ("GET", "/admin/api/hf/model-info"),
        ("POST", "/admin/api/ms/download"),
        ("GET", "/admin/api/ms/recommended"),
        ("GET", "/admin/api/ms/search"),
        ("GET", "/admin/api/ms/model-info"),
        ("POST", "/admin/api/upload/start"),
        ("POST", "/admin/api/upload/validate-token"),
        ("POST", "/admin/api/presets/refresh"),
        ("POST", "/admin/api/web-search/test"),
        ("GET", "/admin/api/update-check"),
        ("POST", "/admin/api/oq/start"),
        ("POST", "/v1/web/search"),
        ("POST", "/v1/web/fetch"),
    }
)

# Prefix rules, written with their trailing slash. Retry endpoints carry a
# task id in the path, so they only match as prefixes.
BLOCKED_PREFIXES = (
    ("POST", "/admin/api/hf/retry/"),
    ("POST", "/admin/api/ms/retry/"),
)

_lock = threading.Lock()
_cached_key: tuple[str, int, int] | None = None
_cached_enabled = False


def state_path() -> str:
    """Resolve the state file path, honouring the test override."""
    return os.environ.get(STATE_PATH_ENV) or STATE_PATH


def is_blocked(method: str, path: str) -> bool:
    """True when this request would try to leave the box."""
    method = method.upper()
    if (method, path) in BLOCKED_EXACT:
        return True
    return any(method == m and path.startswith(p) for m, p in BLOCKED_PREFIXES)


def is_enabled() -> bool:
    """Read the lockdown flag, re-reading only when the file changes.

    Fail-open for the *display* flag: a missing, unreadable or malformed
    state file reads as off. Enforcement never depends on this answer.
    """
    global _cached_key, _cached_enabled

    path = state_path()
    try:
        stat = os.stat(path)
    except OSError:
        key: tuple[str, int, int] = (path, -1, -1)
        with _lock:
            _cached_key, _cached_enabled = key, False
        return False

    key = (path, stat.st_mtime_ns, stat.st_size)
    with _lock:
        if key == _cached_key:
            return _cached_enabled

    enabled = False
    try:
        with open(path, "rb") as handle:
            data = json.loads(handle.read() or b"{}")
        if isinstance(data, dict):
            enabled = bool(data.get("enabled", False))
    except (OSError, ValueError):
        enabled = False

    with _lock:
        _cached_key, _cached_enabled = key, enabled
    return enabled


def reset_cache() -> None:
    """Drop the cached flag (tests and state-file writers)."""
    global _cached_key, _cached_enabled
    with _lock:
        _cached_key, _cached_enabled = None, False
