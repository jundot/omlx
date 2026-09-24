# SPDX-License-Identifier: Apache-2.0
"""oMLX menubar sidecar — live prefill/decode status in the macOS menu bar.

A lightweight companion process spawned by ``omlx serve`` (see cli.py). It
polls the running server's ``/admin/api/activity`` and ``/api/status``
endpoints and renders the live state as colored menu bar text:

  PP 45%   blue   — prompt processing (prefill) in progress
  GEN 42.1 t/s  green — tokens being generated (decode)
  WAIT 3   orange — requests queued
  idle / offline

Clicking the icon opens a menu with per-model progress, session averages,
cache hit rate, and memory. Requires PyObjC (``pip install "omlx[menubar]"``);
without it ``omlx serve`` runs unchanged. The native oMLX.app already owns
the menu bar, so the sidecar skips spawning when ``OMLX_SUPERVISED`` is set.

Run standalone (normally only spawned by the CLI):
  python -m omlx.menubar_sidecar [--host H] [--port N] [--parent PID]
"""

from __future__ import annotations

import argparse
import contextlib
import http.cookiejar
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SETTINGS_PATH = os.path.expanduser("~/.omlx/settings.json")
PIDFILE = os.path.expanduser("~/.omlx/menubar_sidecar.pid")
ACTIVITY_INTERVAL = 1.0  # live prefill/decode cadence (seconds)
STATUS_INTERVAL = 5.0  # session-average stats cadence
HTTP_TIMEOUT = 2.5


def _debug() -> bool:
    return bool(os.environ.get("OMLX_MENUBAR_DEBUG"))


# --------------------------------------------------------------------------
# Autostart gate (pure, unit-tested)
# --------------------------------------------------------------------------


def pyobjc_available() -> bool:
    """True when AppKit can be imported (does not actually import it)."""
    try:
        return importlib.util.find_spec("AppKit") is not None
    except (ImportError, ValueError):
        return False


def should_autostart(
    *,
    cli_flag: bool | None = None,
    settings_enabled: bool = True,
    env: dict[str, str] | None = None,
    platform: str | None = None,
    pyobjc: bool | None = None,
) -> tuple[bool, str | None]:
    """Decide whether ``omlx serve`` should spawn the menubar sidecar.

    Returns ``(enabled, hint)``; ``hint`` is a user-facing line to print
    when the feature was implicitly wanted but cannot run.

    Precedence: macOS-only → oMLX.app supervision (``OMLX_SUPERVISED``,
    set by the native app which has its own menubar) → ``--no-menubar`` →
    ``--menubar`` → ``server.menubar`` setting → PyObjC availability.
    """
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if platform != "darwin":
        return False, None
    if env.get("OMLX_SUPERVISED"):
        # The oMLX.app menubar app supervises this server and shows its
        # own live icon; a second one would be noise.
        return False, None
    if cli_flag is False:
        return False, None
    if cli_flag is None and not settings_enabled:
        return False, None
    if pyobjc is None:
        pyobjc = pyobjc_available()
    if not pyobjc:
        hint = (
            "Menubar sidecar skipped: PyObjC not installed. "
            'Enable it with: pip install "omlx[menubar]"'
        )
        return False, hint
    return True, None


def spawn(
    host: str, port: int, parent_pid: int | None = None
) -> subprocess.Popen | None:
    """Launch the sidecar as a detached companion process. Best-effort."""
    cmd = [
        sys.executable,
        "-m",
        "omlx.menubar_sidecar",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if parent_pid:
        cmd += ["--parent", str(parent_pid)]
    try:
        return subprocess.Popen(
            cmd,
            stdout=None if _debug() else subprocess.DEVNULL,
            stderr=None if _debug() else subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return None


def terminate(proc: subprocess.Popen | None) -> None:
    """Stop a spawned sidecar (SIGTERM; the sidecar exits immediately)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        with contextlib.suppress(OSError):
            proc.kill()


# --------------------------------------------------------------------------
# Server client (stdlib only)
# --------------------------------------------------------------------------


class ServerClient:
    def __init__(self, host: str, port: int, api_key: str):
        self.host = host
        self.port = port
        self.api_key = api_key
        self._cookies = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self._cookies)
        )

    def _url(self, path: str) -> str:
        return f"http://{self.host}:{self.port}{path}"

    def _request(
        self,
        path: str,
        method: str = "GET",
        body: dict | None = None,
        bearer: bool = False,
    ):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self._url(path), data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if bearer and self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with self._opener.open(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode())

    def _login(self):
        if not self.api_key:
            return
        with contextlib.suppress(Exception):
            self._request(
                "/admin/api/login", method="POST", body={"api_key": self.api_key}
            )

    def activity(self) -> dict:
        """Live per-model prefill/decode progress. May raise on failure."""
        try:
            return self._request("/admin/api/activity")
        except urllib.error.HTTPError as e:
            if e.code == 401:
                self._login()
                return self._request("/admin/api/activity")
            raise

    def status(self) -> dict:
        """Server-wide stats (averages, memory). May raise on failure."""
        try:
            return self._request("/api/status", bearer=True)
        except urllib.error.HTTPError as e:
            if e.code == 401 and self.api_key:
                self._login()
            return self._request("/api/status", bearer=True)


# --------------------------------------------------------------------------
# Formatting helpers (pure, unit-tested)
# --------------------------------------------------------------------------


def fmt_tokens(n) -> str:
    if not isinstance(n, (int, float)):
        return "?"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def fmt_duration(s) -> str:
    if not isinstance(s, (int, float)):
        return ""
    s = max(0, int(round(s)))
    if s >= 60:
        m, r = divmod(s, 60)
        return f"{m}m{r}s" if r else f"{m}m"
    return f"{s}s"


# --------------------------------------------------------------------------
# Live state computation (pure, unit-tested)
# --------------------------------------------------------------------------


def compute_live_state(activity: dict | None, online: bool) -> tuple[str, str, list]:
    """Derive the menu bar state from an activity snapshot.

    Returns ``(title, kind, detail_lines)`` where kind is one of
    ``offline|prefill|decode|waiting|idle``. Prefill takes priority over
    decode (matches the native oMLX.app behavior).
    """
    if not online or activity is None:
        return "offline", "offline", []

    models = (activity.get("active_models") or {}).get("models") or []
    total_waiting = (activity.get("active_models") or {}).get(
        "total_waiting_requests"
    ) or 0

    for m in models:
        prefs = m.get("prefilling") or []
        if not prefs:
            continue
        p = prefs[0]
        processed = p.get("processed") or 0
        total = p.get("total") or 0
        pct = int(round(processed / total * 100)) if total else 0
        speed = p.get("speed") or 0
        n_extra = sum(len(x.get("prefilling") or []) for x in models) - 1
        extra = f" +{n_extra}" if n_extra > 0 else ""
        detail = [m.get("id", "?")]
        parts = [f"PP {pct}%  {fmt_tokens(processed)}/{fmt_tokens(total)}"]
        if speed > 0:
            parts.append(f"{int(round(speed))} t/s")
            eta = p.get("eta")
            if isinstance(eta, (int, float)) and eta >= 0:
                parts.append(f"ETA {fmt_duration(eta)}")
        detail.append(" · ".join(parts))
        return f"PP {pct}%{extra}", "prefill", detail

    for m in models:
        gens = m.get("generating") or []
        if not gens:
            continue
        g = gens[0]
        tps = g.get("tokens_per_second") or 0
        toks = g.get("generated_tokens") or 0
        n_extra = sum(len(x.get("generating") or []) for x in models) - len(gens)
        extra = f" +{n_extra}" if n_extra > 0 else ""
        detail = [
            m.get("id", "?"),
            f"GEN {tps:.1f} t/s · {fmt_tokens(toks)} tok · "
            f"{fmt_duration(g.get('elapsed_seconds'))}",
        ]
        return f"GEN {tps:.1f} t/s{extra}", "decode", detail

    if total_waiting > 0:
        return (
            f"WAIT {total_waiting}",
            "waiting",
            [f"{total_waiting} request(s) queued"],
        )

    loaded = [m["id"] for m in models if not m.get("is_loading")]
    detail = [f"loaded: {', '.join(loaded)}"] if loaded else ["no models"]
    return "idle", "idle", detail


# --------------------------------------------------------------------------
# Poller thread
# --------------------------------------------------------------------------


class Poller(threading.Thread):
    """Fetches activity + status off the main thread; publishes snapshots."""

    def __init__(self, client: ServerClient):
        super().__init__(daemon=True)
        self.client = client
        self.online = False
        self.activity: dict | None = None
        self.status: dict | None = None
        self._last_status_fetch = 0.0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            ok = False
            try:
                self.activity = self.client.activity()
                ok = True
            except Exception:
                if _debug():
                    import traceback

                    traceback.print_exc(file=sys.stderr)
                self.activity = None
            now = time.monotonic()
            if ok and now - self._last_status_fetch >= STATUS_INTERVAL:
                self._last_status_fetch = now
                with contextlib.suppress(Exception):
                    self.status = self.client.status()
            self.online = ok
            self._stop.wait(ACTIVITY_INTERVAL)

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------
# Menu bar UI
# --------------------------------------------------------------------------

_STATE_COLORS: dict[str, str] = {
    "offline": "tertiaryLabelColor",
    "prefill": "systemBlueColor",
    "decode": "systemGreenColor",
    "waiting": "systemOrangeColor",
    "idle": "secondaryLabelColor",
}


class MenubarApp:
    def __init__(self, client: ServerClient):
        # Imported lazily so the module stays importable headless (tests,
        # CI, non-GUI contexts).
        import AppKit

        self._kit = AppKit
        self.client = client
        self.poller = Poller(client)
        self.app = AppKit.NSApplication.sharedApplication()
        self.app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
        # NSVariableLengthStatusItem is a C macro, not exported by PyObjC.
        self.status_item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(
            -1.0
        )
        self._last_title = None

    def _color(self, kind: str):
        return getattr(self._kit.NSColor, _STATE_COLORS[kind])()

    def _set_title(self, text: str, color) -> None:
        if text == self._last_title:
            return
        self._last_title = text
        attrs = {
            self._kit.NSFontAttributeName: self._kit.NSFont.menuBarFontOfSize_(0),
            self._kit.NSForegroundColorAttributeName: color,
        }
        self.status_item.button().setAttributedTitle_(
            self._kit.NSAttributedString.alloc().initWithString_attributes_(
                " " + text, attrs
            )
        )

    def _rebuild_menu(self) -> None:
        act = self.poller.activity
        st = self.poller.status
        online = self.poller.online and act is not None
        menu = self._kit.NSMenu.alloc().init()

        def add(title, action=None, bold=False):
            item = self._kit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                title, None, ""
            )
            item.setEnabled_(action is not None)
            if action is not None:
                item.setTarget_(self)
                item.setAction_(action)
            if bold:
                attrs = {
                    self._kit.NSFontAttributeName: self._kit.NSFont.boldSystemFontOfSize_(
                        self._kit.NSFont.menuBarFontOfSize_(0).pointSize()
                    )
                }
                item.setAttributedTitle_(
                    self._kit.NSAttributedString.alloc().initWithString_attributes_(
                        title, attrs
                    )
                )
            menu.addItem_(item)

        add("oMLX · " + f"{self.client.host}:{self.client.port}", bold=True)
        add("---")

        if not online:
            add("server offline")
            add("---")
        else:
            models = (act.get("active_models") or {}).get("models") or []
            any_live = False
            for m in models:
                prefs = m.get("prefilling") or []
                gens = m.get("generating") or []
                acts = m.get("activities") or []
                waiting = m.get("waiting_requests") or 0
                if m.get("is_loading"):
                    add(f"{m['id']}  (loading…)")
                    any_live = True
                    continue
                if not (prefs or gens or acts or waiting):
                    continue
                any_live = True
                add(m["id"], bold=True)
                for p in prefs:
                    processed = p.get("processed") or 0
                    total = p.get("total") or 0
                    pct = int(round(processed / total * 100)) if total else 0
                    line = (
                        f"  PP {pct}%  {fmt_tokens(processed)}" f"/{fmt_tokens(total)}"
                    )
                    speed = p.get("speed") or 0
                    if speed > 0:
                        line += f"  ·  {int(round(speed))} t/s"
                        eta = p.get("eta")
                        if isinstance(eta, (int, float)) and eta >= 0:
                            line += f"  ·  ETA {fmt_duration(eta)}"
                    add(line)
                for g in gens[:6]:
                    add(
                        f"  GEN {(g.get('tokens_per_second') or 0):.1f} t/s"
                        f"  ·  {fmt_tokens(g.get('generated_tokens'))} tok"
                        f"  ·  {fmt_duration(g.get('elapsed_seconds'))}"
                    )
                if len(gens) > 6:
                    add(f"  … {len(gens) - 6} more decoding")
                for a in acts[:3]:
                    add(
                        f"  RUN {(a.get('kind') or 'active')}"
                        f"  ·  {fmt_duration(a.get('elapsed_seconds'))}"
                    )
                if waiting:
                    add(f"  waiting: {waiting}")
            if not any_live:
                add("idle — no active requests")
            add("---")

        if st:
            add(
                f"session avg  prefill {st.get('avg_prefill_tps', 0):.0f} t/s"
                f"  ·  decode {st.get('avg_generation_tps', 0):.1f} t/s"
            )
            add(
                f"requests {st.get('total_requests', 0)}  ·  cache hit "
                f"{st.get('cache_efficiency', 0)}%"
            )
            add(
                f"mem {st.get('model_memory_used_formatted', '?')}"
                f" / {st.get('model_memory_max_formatted', '?')}"
            )
            add("---")

        add("Open Dashboard", action="open_dashboard")
        add("Quit Sidecar", action="quit")
        self.status_item.setMenu_(menu)

    # ---- actions ----

    def open_dashboard(self, _sender=None):
        subprocess.Popen(
            ["open", f"http://{self.client.host}:{self.client.port}/admin/"]
        )

    def quit(self, _sender=None):
        self.app.terminate_(None)

    # ---- tick ----

    def _tick(self, _timer=None):
        # An escaping exception would invalidate the repeating timer and
        # freeze the icon at its last title; catch and log instead.
        try:
            act = self.poller.activity
            title, kind, _detail = compute_live_state(act, self.poller.online)
            self._set_title(title, self._color(kind))
            self._rebuild_menu()
        except Exception:
            import traceback

            traceback.print_exc(file=sys.stderr)

    def run(self):
        self.poller.start()
        self._kit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            ACTIVITY_INTERVAL, True, lambda _t: self._tick()
        )
        self._tick()
        self.app.run()


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------


def _read_settings() -> dict:
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}  # missing/corrupt settings → defaults


def _already_running() -> bool:
    try:
        with open(PIDFILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return pid != os.getpid()
    except (OSError, ValueError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m omlx.menubar_sidecar",
        description="oMLX menubar sidecar — live prefill/decode status",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--parent", type=int, default=None, help="exit when this PID goes away"
    )
    args = parser.parse_args(argv)

    if _already_running():
        sys.exit(0)
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))

    settings = _read_settings()
    host = args.host or (settings.get("server") or {}).get("host") or "127.0.0.1"
    if "," in host:
        host = host.split(",")[0].strip()
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = args.port or (settings.get("server") or {}).get("port") or 8000
    api_key = ((settings.get("auth") or {}).get("api_key")) or ""

    if args.parent:

        def watch_parent():
            while True:
                time.sleep(2)
                try:
                    os.kill(args.parent, 0)
                except OSError:
                    os._exit(0)

        threading.Thread(target=watch_parent, daemon=True).start()

    # sys.exit raises inside the AppKit run loop and gets swallowed;
    # os._exit is the only reliable quit path from a Python signal handler.
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))

    MenubarApp(ServerClient(host, port, api_key)).run()


if __name__ == "__main__":
    main()
