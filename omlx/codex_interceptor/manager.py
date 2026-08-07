"""Managed, process-scoped Codex interceptor lifecycle.

Codex keeps its normal account, feature flags, projects, tools, MCP servers and
cloud routes.  Only the fresh Codex process launched here inherits the loopback
proxy environment.  This module deliberately never writes Codex configuration.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import plistlib
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_LOCAL_SLOT = "gpt-5.3-codex-spark"
_CODEX_CONFIG = Path.home() / ".codex" / "config.toml"
_RUNTIME_ROOT = Path.home() / ".omlx" / "codex-interceptor"
_SAFE_EVENT_FIELDS = {
    "time",
    "event",
    "session_id",
    "status",
    "method",
    "original_host",
    "original_path",
    "local_model",
    "local_server",
    "requested_model",
    "effective_model",
    "local_slot",
    "transport",
    "inference_kind",
    "duration_ms",
    "first_byte_ms",
    "first_visible_ms",
    "connect_ms",
    "dispatch_ms",
    "transform_ms",
    "route_to_first_byte_ms",
    "bridge_overhead_ms",
    "connection_reused",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cache_hit_percent",
    "tokens_per_second",
    "output_tokens_per_second",
    "request_bytes",
    "tool_count",
    "tool_names_remapped",
    "patched_model_count",
    "advertised_context_window",
    "prefix_prefill_status",
    "residency_status",
    "performance_warning",
    "previous_local_model",
    "route_revision",
    "warning",
    "error",
}


def _find_codex_app_bundle() -> Path | None:
    for root in (Path("/Applications"), Path.home() / "Applications"):
        for name in ("Codex.app", "ChatGPT.app"):
            bundle = root / name
            plist_path = bundle / "Contents" / "Info.plist"
            if not plist_path.is_file():
                continue
            try:
                with plist_path.open("rb") as handle:
                    info = plistlib.load(handle)
            except (OSError, plistlib.InvalidFileException):
                continue
            if info.get("CFBundleIdentifier") == "com.openai.codex":
                return bundle
    return None


def _resolve_codex_binary() -> str | None:
    executable = shutil.which("codex")
    if executable:
        return executable
    bundle = _find_codex_app_bundle()
    bundled = bundle / "Contents" / "Resources" / "codex" if bundle else None
    if bundled and bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)
    return None


@dataclass(frozen=True)
class CodexInterceptorConfig:
    model: str
    upstream_url: str
    api_key: str = ""
    auth_header: bool = True
    project: Path = Path.home()
    local_slot: str = DEFAULT_LOCAL_SLOT
    local_label: str = "Local · oMLX"
    context_window: int | None = None
    launch_app: bool = True
    replace_existing: bool = False
    listen_port: int | None = None


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.15)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _suggest_port(preferred: int = 9146) -> int:
    for port in range(preferred, preferred + 100):
        if not _port_is_open(port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _terminate(process: subprocess.Popen[Any] | None, timeout: float = 8.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()
        process.wait(timeout=2)


def _running_app_pids() -> list[int]:
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,command="],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    matches: list[int] = []
    suffixes = (
        "/Codex.app/Contents/MacOS/Codex",
        "/ChatGPT.app/Contents/MacOS/ChatGPT",
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        if command.strip().endswith(suffixes):
            with contextlib.suppress(ValueError):
                matches.append(int(pid_text))
    return matches


def _wait_for_new_app_pid(
    previous: set[int], opener: subprocess.Popen[Any]
) -> int | None:
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        new = set(_running_app_pids()) - previous
        if new:
            return max(new)
        if opener.poll() is not None:
            break
        time.sleep(0.15)
    return None


def _merged_no_proxy(env: dict[str, str]) -> str:
    values: list[str] = []
    for source in (env.get("NO_PROXY", ""), env.get("no_proxy", "")):
        values.extend(part.strip() for part in source.split(",") if part.strip())
    values.extend(
        [
            "localhost",
            "127.0.0.1",
            "::1",
            "192.168.0.0/16",
            "10.0.0.0/8",
            "172.16.0.0/12",
        ]
    )
    return ",".join(dict.fromkeys(values))


class CodexInterceptorManager:
    """Own one proxy and one freshly launched Codex process."""

    def __init__(self, runtime_root: Path = _RUNTIME_ROOT) -> None:
        self.runtime_root = runtime_root
        self._lock = threading.RLock()
        self._proxy: subprocess.Popen[Any] | None = None
        self._opener: subprocess.Popen[Any] | None = None
        self._app_pid: int | None = None
        self._phase = "stopped"
        self._error: str | None = None
        self._config: CodexInterceptorConfig | None = None
        self._session_id: str | None = None
        self._session_dir: Path | None = None
        self._status_path: Path | None = None
        self._route_path: Path | None = None
        self._route_revision = 0
        self._listen_port: int | None = None
        self._status_offset = 0
        self._status_remainder = ""
        self._started_at: float | None = None
        self._config_hash_before: str | None = None
        self._config_hash_recorded = False
        self._counts = {"local": 0, "cloud": 0, "completed": 0, "failed": 0}
        self._active_local = 0
        self._last_route: str | None = None
        self._last_requested_model: str | None = None
        self._last_effective_model: str | None = None
        self._active_model: str | None = None
        self._active_context_window: int | None = None
        self._pending_model: str | None = None
        self._pending_context_window: int | None = None
        self._switch_generation = 0
        self._switch_candidate: CodexInterceptorConfig | None = None
        self._model_switch_error: str | None = None
        self._effective_slot: str | None = None
        self._latest_metrics: dict[str, Any] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=40)
        self._warmup_status = "idle"
        self._warmup_model: str | None = None

    def doctor(self) -> dict[str, Any]:
        app = _find_codex_app_bundle()
        proxy_command = self._resolve_proxy_command()
        return {
            "ready": bool(app and proxy_command),
            "codex_app_installed": app is not None,
            "codex_app_path": str(app) if app else None,
            "codex_running": bool(_running_app_pids()),
            "mitmproxy_available": proxy_command is not None,
            "mitmproxy_source": (
                "bundled_python"
                if proxy_command and proxy_command[0] == sys.executable
                else "executable"
            )
            if proxy_command
            else None,
            "config_path": str(_CODEX_CONFIG),
            "config_will_be_modified": False,
        }

    def start(self, config: CodexInterceptorConfig) -> dict[str, Any]:
        with self._lock:
            if self._is_running_locked():
                if self._config == config:
                    return self._status_locked()
                raise RuntimeError(
                    "the Codex interceptor is already running; stop it before changing settings"
                )
            self._config = config
            self._reset_status_locked()
            self._phase = "starting"
            self._error = None
            self._effective_slot = config.local_slot
            self._started_at = time.time()
            self._config_hash_before = _sha256_file(_CODEX_CONFIG)
            self._config_hash_recorded = True

        try:
            project = config.project.expanduser().resolve()
            if not project.is_dir():
                raise FileNotFoundError(f"project directory does not exist: {project}")
            self._check_upstream(config)
            proxy_command = self._resolve_proxy_command()
            if proxy_command is None:
                raise FileNotFoundError(
                    "mitmproxy is unavailable; install the oMLX Codex interceptor extra or `brew install mitmproxy`"
                )
            if config.launch_app and _find_codex_app_bundle() is None:
                raise FileNotFoundError(
                    "Codex.app or ChatGPT.app (bundle com.openai.codex) is not installed"
                )
            existing = _running_app_pids()
            if config.launch_app and existing:
                if not config.replace_existing:
                    raise RuntimeError(
                        "Codex is already open. Quit it first, or use ‘Quit Codex & Start’ so the new process inherits the interceptor."
                    )
                self._quit_pids(existing)

            session_id = f"{int(time.time())}-{secrets.token_hex(4)}"
            self.runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.runtime_root, 0o700)
            session_dir = self.runtime_root / session_id
            session_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            os.chmod(session_dir, 0o700)
            status_path = session_dir / "status.jsonl"
            status_path.touch(mode=0o600)
            os.chmod(status_path, 0o600)
            route_path = session_dir / "route.json"
            self._write_route_config(
                route_path,
                revision=1,
                model=config.model,
                local_label=config.local_label,
                context_window=config.context_window,
            )
            listen_port = config.listen_port or _suggest_port()
            proxy = self._start_proxy(
                config=config,
                command=proxy_command,
                listen_port=listen_port,
                session_id=session_id,
                session_dir=session_dir,
                status_path=status_path,
                route_path=route_path,
            )
            with self._lock:
                self._proxy = proxy
                self._session_id = session_id
                self._session_dir = session_dir
                self._status_path = status_path
                self._route_path = route_path
                self._route_revision = 1
                self._listen_port = listen_port
                self._phase = "running"
            threading.Thread(
                target=self._watch_proxy_exit,
                args=(proxy,),
                daemon=True,
                name="omlx-codex-interceptor-proxy-watch",
            ).start()

            if config.launch_app:
                self._launch_codex(project)
            return self.status()
        except Exception as exc:
            with self._lock:
                proxy = self._proxy
                opener = self._opener
                self._proxy = None
                self._opener = None
                self._app_pid = None
                self._phase = "error"
                self._error = str(exc)
            _terminate(opener)
            _terminate(proxy)
            raise

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._consume_events_locked()
            self._phase = "stopping"
            app_pid = self._app_pid
            opener = self._opener
            proxy = self._proxy
            self._app_pid = None
            self._opener = None
            self._proxy = None
            self._listen_port = None
            self._switch_generation += 1
            self._switch_candidate = None
        if app_pid is not None:
            self._quit_pids([app_pid])
        _terminate(opener, timeout=2)
        _terminate(proxy)
        with self._lock:
            self._phase = "stopped"
            if (
                self._config_hash_recorded
                and _sha256_file(_CODEX_CONFIG) != self._config_hash_before
            ):
                self._error = (
                    "Codex config changed outside the interceptor during this session"
                )
            return self._status_locked()

    def run_cli(self, config: CodexInterceptorConfig, args: list[str]) -> int:
        """Run Codex CLI synchronously with only this child process proxied."""
        cli_config = CodexInterceptorConfig(**{**config.__dict__, "launch_app": False})
        self.start(cli_config)
        try:
            executable = _resolve_codex_binary()
            if executable is None:
                raise FileNotFoundError("the Codex CLI is not installed")
            env = self._codex_environment()
            return subprocess.run([executable, *args], env=env).returncode
        finally:
            self.stop()

    def run_desktop(self, config: CodexInterceptorConfig) -> int:
        """Run a managed desktop session until the user closes Codex."""
        app_config = CodexInterceptorConfig(**{**config.__dict__, "launch_app": True})
        self.start(app_config)
        try:
            with self._lock:
                opener = self._opener
            return opener.wait() if opener is not None else 0
        finally:
            self.stop()

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._consume_events_locked()
            if self._proxy is not None and self._proxy.poll() is not None:
                self._phase = "error"
                self._error = (
                    f"interceptor proxy exited with status {self._proxy.returncode}"
                )
                self._proxy = None
            return self._status_locked()

    def begin_model_switch(
        self,
        model: str,
        *,
        context_window: int | None,
        local_label: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reserve a safe model switch before asynchronously warming it.

        This performs context and concurrency checks before a potentially long
        model load. ``complete_model_switch`` publishes the route only after
        the selected engine is ready.
        """
        model = model.strip()
        if not model:
            raise ValueError("a local model is required")
        with self._lock:
            self._consume_events_locked()
            if not self._is_running_locked() or self._config is None:
                raise RuntimeError("the Codex interceptor is not running")
            if self._warmup_status == "loading":
                raise RuntimeError("wait for the current local model to finish loading")
            if self._switch_candidate is not None or self._pending_model is not None:
                raise RuntimeError("a local model switch is already loading or queued")
            active_model = self._active_model or self._config.model
            current_context = self._active_context_window
            target_context = (
                context_window
                if context_window is not None
                else current_context
                if model == active_model
                else None
            )
            if (
                target_context is not None
                and current_context is not None
                and target_context < current_context
            ) or (
                model != active_model
                and (current_context is None or target_context is None)
            ):
                raise RuntimeError(
                    "this model has a smaller or unknown context window; stop the "
                    "interceptor and start a fresh Codex session with that model"
                )
            candidate = replace(
                self._config,
                model=model,
                local_label=local_label or f"Local · oMLX · {model.rsplit('/', 1)[-1]}",
                context_window=target_context,
            )
            self._switch_generation += 1
            switch_generation = self._switch_generation
            self._switch_candidate = candidate
            self._warmup_status = "loading"
            self._warmup_model = model
            self._model_switch_error = None
            return switch_generation, self._status_locked()

    def complete_model_switch(self, switch_generation: int) -> dict[str, Any]:
        """Publish a previously reserved switch after its engine is warm."""
        with self._lock:
            if (
                switch_generation != self._switch_generation
                or self._switch_candidate is None
            ):
                raise RuntimeError("the local model switch is no longer current")
            candidate = self._switch_candidate
            route_path = self._route_path
        if route_path is None:
            raise RuntimeError("the interceptor route control is unavailable")
        self._check_upstream(candidate)
        with self._lock:
            if (
                switch_generation != self._switch_generation
                or self._switch_candidate != candidate
                or not self._is_running_locked()
                or self._route_path != route_path
            ):
                raise RuntimeError("the Codex interceptor stopped during model switch")
            self._route_revision += 1
            self._write_route_config(
                route_path,
                revision=self._route_revision,
                model=candidate.model,
                local_label=candidate.local_label,
                context_window=candidate.context_window,
            )
            self._config = candidate
            if (
                candidate.model == self._active_model
                and candidate.context_window == self._active_context_window
            ):
                self._pending_model = None
                self._pending_context_window = None
            else:
                self._pending_model = candidate.model
                self._pending_context_window = candidate.context_window
            self._switch_candidate = None
            self._warmup_status = "ready"
            self._warmup_model = candidate.model
            self._model_switch_error = None
            return self._status_locked()

    def fail_model_switch(
        self,
        switch_generation: int,
        error: str,
    ) -> dict[str, Any]:
        """Record a load failure only if it belongs to the current switch."""
        with self._lock:
            if switch_generation != self._switch_generation:
                return self._status_locked()
            self._switch_candidate = None
            self._warmup_status = "failed"
            self._model_switch_error = error[:1024]
            return self._status_locked()

    def switch_model(
        self,
        model: str,
        *,
        context_window: int | None,
        local_label: str | None = None,
    ) -> dict[str, Any]:
        """Synchronously publish a model that the caller has already warmed."""
        switch_generation, _ = self.begin_model_switch(
            model,
            context_window=context_window,
            local_label=local_label,
        )
        try:
            return self.complete_model_switch(switch_generation)
        except Exception as exc:
            self.fail_model_switch(switch_generation, str(exc))
            raise

    def _reset_status_locked(self) -> None:
        self._route_path = None
        self._route_revision = 0
        self._counts = {"local": 0, "cloud": 0, "completed": 0, "failed": 0}
        self._active_local = 0
        self._last_route = None
        self._last_requested_model = None
        self._last_effective_model = None
        self._active_model = self._config.model if self._config else None
        self._active_context_window = (
            self._config.context_window if self._config else None
        )
        self._pending_model = None
        self._pending_context_window = None
        self._switch_generation += 1
        self._switch_candidate = None
        self._model_switch_error = None
        self._effective_slot = None
        self._latest_metrics = {}
        self._events.clear()
        self._warmup_status = "loading"
        self._warmup_model = self._config.model if self._config else None
        self._status_offset = 0
        self._status_remainder = ""

    def set_warmup_status(
        self, session_id: str | None, status: str, model: str | None = None
    ) -> None:
        """Update readiness only when it still belongs to the active session."""
        with self._lock:
            if session_id == self._session_id:
                if model is not None and model != self._warmup_model:
                    return
                self._warmup_status = status
                if model is not None:
                    self._warmup_model = model

    def _is_running_locked(self) -> bool:
        return self._proxy is not None and self._proxy.poll() is None

    def _status_locked(self) -> dict[str, Any]:
        config = self._config
        config_hash_now = _sha256_file(_CODEX_CONFIG)
        return {
            "phase": self._phase,
            "running": self._is_running_locked(),
            "error": self._error,
            "session_id": self._session_id,
            "started_at": self._started_at,
            "model": self._active_model or (config.model if config else None),
            "active_model": self._active_model,
            "active_context_window": self._active_context_window,
            "pending_model": self._pending_model,
            "pending_context_window": self._pending_context_window,
            "model_switching": (
                self._switch_candidate is not None or self._pending_model is not None
            ),
            "model_switch_loading": self._switch_candidate is not None,
            "model_switch_error": self._model_switch_error,
            "local_slot": self._effective_slot,
            "project": str(config.project) if config else None,
            "proxy_pid": self._proxy.pid if self._is_running_locked() else None,
            "codex_pid": self._app_pid,
            "codex_running": bool(
                self._app_pid and self._app_pid in _running_app_pids()
            ),
            "active_local_requests": self._active_local,
            "local_requests": self._counts["local"],
            "cloud_requests": self._counts["cloud"],
            "completed_requests": self._counts["completed"],
            "failed_requests": self._counts["failed"],
            "last_route": self._last_route,
            "last_requested_model": self._last_requested_model,
            "last_effective_model": self._last_effective_model,
            "latest_metrics": dict(self._latest_metrics),
            "warmup_status": self._warmup_status,
            "warmup_model": self._warmup_model,
            "recent_events": list(self._events),
            "diagnostics_path": str(self._status_path) if self._status_path else None,
            "config_path": str(_CODEX_CONFIG),
            "config_modified": bool(
                self._config_hash_recorded
                and config_hash_now != self._config_hash_before
            ),
        }

    @staticmethod
    def _resolve_proxy_command() -> list[str] | None:
        try:
            if importlib.util.find_spec("mitmproxy") is not None:
                return [sys.executable, "-m", "omlx.codex_interceptor.mitmdump_runner"]
        except (ImportError, ValueError):
            pass
        executable = shutil.which("mitmdump")
        return [executable] if executable else None

    @staticmethod
    def _check_upstream(config: CodexInterceptorConfig) -> None:
        base = config.upstream_url.rsplit("/responses", 1)[0]
        request = Request(base + "/models", headers={"Accept": "application/json"})
        if config.api_key and config.auth_header:
            request.add_header("Authorization", f"Bearer {config.api_key}")
        try:
            with urlopen(request, timeout=4) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError(
                        f"oMLX returned HTTP {response.status} from /v1/models"
                    )
                payload = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(
                f"oMLX returned HTTP {exc.code} from /v1/models"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"cannot reach the local oMLX server at {base}") from exc
        model_ids = (
            {
                str(item.get("id"))
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            }
            if isinstance(payload, dict)
            else set()
        )
        if config.model not in model_ids:
            raise ValueError(f"local model is not available from oMLX: {config.model}")

    def _start_proxy(
        self,
        *,
        config: CodexInterceptorConfig,
        command: list[str],
        listen_port: int,
        session_id: str,
        session_dir: Path,
        status_path: Path,
        route_path: Path,
    ) -> subprocess.Popen[Any]:
        confdir = session_dir / "mitmproxy"
        confdir.mkdir(mode=0o700)
        env = os.environ.copy()
        for key in tuple(env):
            if (
                "CLOUD_AUDIT" in key
                or key.startswith("HARNESS_INTERCEPTOR_")
                or key.startswith("OMLX_CODEX_INTERCEPTOR_")
            ):
                env.pop(key, None)
        env.update(
            {
                "OMLX_CODEX_INTERCEPTOR_UPSTREAM_URL": config.upstream_url,
                "OMLX_CODEX_INTERCEPTOR_MODEL": config.model,
                "OMLX_CODEX_INTERCEPTOR_AUTH_HEADER": "1"
                if config.auth_header
                else "0",
                "OMLX_CODEX_INTERCEPTOR_STATUS_PATH": str(status_path),
                "OMLX_CODEX_INTERCEPTOR_ROUTE_PATH": str(route_path),
                "OMLX_CODEX_INTERCEPTOR_SESSION_ID": session_id,
                "OMLX_CODEX_INTERCEPTOR_LOCAL_LABEL": config.local_label,
                "OMLX_CODEX_INTERCEPTOR_LOCAL_SERVER": "oMLX",
                "OMLX_CODEX_INTERCEPTOR_LOCAL_SLOT": config.local_slot,
                "OMLX_CODEX_INTERCEPTOR_LOCAL_SLOT_AUTO": "1",
                "OMLX_CODEX_INTERCEPTOR_CAPABILITIES_PATH": str(
                    session_dir / "capabilities.json"
                ),
                "OMLX_CODEX_INTERCEPTOR_PREFIX_CACHE_PATH": str(
                    self.runtime_root / "prefix-cache.json"
                ),
                "OMLX_CODEX_INTERCEPTOR_PREFIX_PREFILL": "1",
                "OMLX_CODEX_INTERCEPTOR_RESIDENCY_KEEPALIVE": "1",
            }
        )
        # Homebrew's mitmdump runs under its own interpreter during source
        # development. Give only the proxy child access to this checkout;
        # bundled builds use the in-environment module runner above.
        package_root = str(Path(__file__).resolve().parents[2])
        inherited_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            package_root
            if not inherited_pythonpath
            else package_root + os.pathsep + inherited_pythonpath
        )
        if config.api_key:
            env["OMLX_CODEX_INTERCEPTOR_UPSTREAM_API_KEY"] = config.api_key
        if config.context_window:
            env["OMLX_CODEX_INTERCEPTOR_LOCAL_CONTEXT_WINDOW"] = str(
                config.context_window
            )
        argv = command + [
            "--quiet",
            "--listen-host",
            "127.0.0.1",
            "--listen-port",
            str(listen_port),
            "--set",
            f"confdir={confdir}",
            "--set",
            "block_global=true",
            "--set",
            "flow_detail=0",
            "--set",
            "console_eventlog_verbosity=error",
            "-s",
            str(Path(__file__).with_name("addon.py")),
        ]
        log_path = session_dir / "proxy.log"
        log_path.touch(mode=0o600)
        log = log_path.open("a", encoding="utf-8")
        try:
            proxy = subprocess.Popen(
                argv,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                text=True,
                start_new_session=True,
            )
        finally:
            log.close()
        cert_path = confdir / "mitmproxy-ca-cert.pem"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if proxy.poll() is not None:
                detail = log_path.read_text(encoding="utf-8", errors="replace")[
                    -2000:
                ].strip()
                raise RuntimeError(
                    detail or f"mitmproxy exited with status {proxy.returncode}"
                )
            if cert_path.is_file() and _port_is_open(listen_port):
                return proxy
            time.sleep(0.1)
        _terminate(proxy)
        raise TimeoutError("interceptor proxy did not become ready within 20 seconds")

    def _launch_codex(self, project: Path) -> None:
        app = _find_codex_app_bundle()
        if app is None:
            raise FileNotFoundError("Codex desktop app is not installed")
        child = self._codex_environment()
        command = ["/usr/bin/open", "-n", "-W", "-a", str(app)]
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "CODEX_CA_CERTIFICATE",
            "SSL_CERT_FILE",
            "CURL_CA_BUNDLE",
            "GIT_SSL_CAINFO",
            "REQUESTS_CA_BUNDLE",
            "NODE_EXTRA_CA_CERTS",
            "NO_PROXY",
            "no_proxy",
        ):
            value = child.get(name)
            if value is not None:
                command.extend(["--env", f"{name}={value}"])
        command.append(f"codex://threads/new?path={quote(str(project), safe='')}")
        previous = set(_running_app_pids())
        opener = subprocess.Popen(command, start_new_session=True)
        app_pid = _wait_for_new_app_pid(previous, opener)
        if opener.poll() is not None and app_pid is None:
            raise RuntimeError(
                f"Codex failed to launch (open exited {opener.returncode})"
            )
        with self._lock:
            self._opener = opener
            self._app_pid = app_pid
        threading.Thread(
            target=self._watch_codex_exit,
            args=(opener,),
            daemon=True,
            name="omlx-codex-interceptor-app-watch",
        ).start()

    def _watch_codex_exit(self, opener: subprocess.Popen[Any]) -> None:
        """Reap the proxy when the managed Codex desktop session closes."""
        returncode = opener.wait()
        with self._lock:
            if self._opener is not opener:
                return
            proxy = self._proxy
            self._opener = None
            self._app_pid = None
            self._proxy = None
            self._listen_port = None
            self._phase = "stopped" if returncode == 0 else "error"
            if returncode != 0:
                self._error = f"Codex exited with status {returncode}"
        _terminate(proxy)

    def _watch_proxy_exit(self, proxy: subprocess.Popen[Any]) -> None:
        """Close managed Codex if its process-scoped proxy dies unexpectedly."""
        returncode = proxy.wait()
        with self._lock:
            if self._proxy is not proxy:
                return
            app_pid = self._app_pid
            opener = self._opener
            self._proxy = None
            self._opener = None
            self._app_pid = None
            self._listen_port = None
            self._phase = "error"
            self._error = f"interceptor proxy exited with status {returncode}"
        if app_pid is not None:
            with contextlib.suppress(RuntimeError):
                self._quit_pids([app_pid])
        _terminate(opener, timeout=2)

    def _codex_environment(self) -> dict[str, str]:
        with self._lock:
            session_dir = self._session_dir
            listen_port = self._listen_port
        if session_dir is None or listen_port is None:
            raise RuntimeError(
                "the interceptor must be running before Codex is launched"
            )
        cert_path = session_dir / "mitmproxy" / "mitmproxy-ca-cert.pem"
        combined_ca = self._combined_ca_bundle(cert_path, session_dir)
        proxy_url = f"http://127.0.0.1:{listen_port}"
        env = os.environ.copy()
        for name in ("CODEX_HOME", "CODEX_CLI_PATH", "ALL_PROXY", "all_proxy"):
            env.pop(name, None)
        no_proxy = _merged_no_proxy(env)
        env.update(
            {
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "CODEX_CA_CERTIFICATE": str(cert_path),
                "SSL_CERT_FILE": str(combined_ca),
                "CURL_CA_BUNDLE": str(combined_ca),
                "GIT_SSL_CAINFO": str(combined_ca),
                "REQUESTS_CA_BUNDLE": str(combined_ca),
                "NODE_EXTRA_CA_CERTS": str(cert_path),
                "NO_PROXY": no_proxy,
                "no_proxy": no_proxy,
            }
        )
        return env

    @staticmethod
    def _combined_ca_bundle(cert_path: Path, session_dir: Path) -> Path:
        target = session_dir / "combined-ca.pem"
        chunks: list[bytes] = []
        for candidate in (
            Path("/etc/ssl/cert.pem"),
            Path("/opt/homebrew/etc/ca-certificates/cert.pem"),
        ):
            if candidate.is_file():
                chunks.append(candidate.read_bytes().rstrip() + b"\n")
                break
        chunks.append(cert_path.read_bytes().rstrip() + b"\n")
        target.write_bytes(b"".join(chunks))
        os.chmod(target, 0o600)
        return target

    @staticmethod
    def _write_route_config(
        path: Path,
        *,
        revision: int,
        model: str,
        local_label: str,
        context_window: int | None,
    ) -> None:
        payload = {
            "schema_version": 1,
            "revision": revision,
            "model": model,
            "local_label": local_label,
            "context_window": context_window,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(path)
        os.chmod(path, 0o600)

    @staticmethod
    def _quit_pids(pids: list[int]) -> None:
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and any(
            pid in _running_app_pids() for pid in pids
        ):
            time.sleep(0.15)
        remaining = [pid for pid in pids if pid in _running_app_pids()]
        if remaining:
            raise RuntimeError(
                "Codex did not quit normally; quit it from the app and try again"
            )

    def _consume_events_locked(self) -> None:
        path = self._status_path
        if path is None:
            return
        try:
            size = path.stat().st_size
            if size < self._status_offset:
                self._status_offset = 0
                self._status_remainder = ""
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._status_offset)
                data = handle.read()
                self._status_offset = handle.tell()
        except OSError:
            return
        lines = (self._status_remainder + data).split("\n")
        self._status_remainder = lines.pop() if lines else ""
        for line in lines:
            try:
                raw = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(raw, dict):
                continue
            event = str(raw.get("event", ""))
            if event == "local_slot_recovered" and isinstance(
                raw.get("local_slot"), str
            ):
                self._effective_slot = raw["local_slot"]
            if event == "local_model_changed" and isinstance(
                raw.get("local_model"), str
            ):
                self._active_model = raw["local_model"]
                context_window = raw.get("advertised_context_window")
                self._active_context_window = (
                    context_window if isinstance(context_window, int) else None
                )
                if self._pending_model == self._active_model:
                    self._pending_model = None
                    self._pending_context_window = None
            if event == "inference_routed":
                if isinstance(raw.get("local_model"), str):
                    self._active_model = raw["local_model"]
                    if self._pending_model == self._active_model:
                        self._active_context_window = self._pending_context_window
                        self._pending_model = None
                        self._pending_context_window = None
                self._last_requested_model = (
                    raw["requested_model"]
                    if isinstance(raw.get("requested_model"), str)
                    else None
                )
                self._last_effective_model = self._active_model
                self._counts["local"] += 1
                self._active_local += 1
                self._last_route = "local"
            elif event == "remote_inference_routed":
                self._last_requested_model = (
                    raw["requested_model"]
                    if isinstance(raw.get("requested_model"), str)
                    else None
                )
                self._last_effective_model = (
                    raw["effective_model"]
                    if isinstance(raw.get("effective_model"), str)
                    else self._last_requested_model
                )
                self._counts["cloud"] += 1
                self._last_route = "cloud"
            elif event == "inference_completed":
                self._counts["completed"] += 1
                self._active_local = max(0, self._active_local - 1)
            elif event in {
                "request_rejected",
                "local_request_invalid",
                "local_request_refused",
                "inference_error",
                "inference_stream_closed",
            }:
                self._counts["failed"] += 1
                self._active_local = max(0, self._active_local - 1)
            for name in (
                "duration_ms",
                "first_byte_ms",
                "first_visible_ms",
                "connect_ms",
                "connection_reused",
                "input_tokens",
                "output_tokens",
                "cached_tokens",
                "cache_hit_percent",
                "tokens_per_second",
                "output_tokens_per_second",
                "residency_status",
                "prefix_prefill_status",
                "performance_warning",
            ):
                if raw.get(name) is not None:
                    metric_name = (
                        "tokens_per_second"
                        if name == "output_tokens_per_second"
                        else name
                    )
                    self._latest_metrics[metric_name] = raw[name]
            safe = {key: raw[key] for key in _SAFE_EVENT_FIELDS if key in raw}
            if safe:
                self._events.append(safe)


_manager = CodexInterceptorManager()


def get_codex_interceptor_manager() -> CodexInterceptorManager:
    return _manager
