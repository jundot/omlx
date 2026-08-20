# SPDX-License-Identifier: Apache-2.0
"""Managed DS4 subprocess scaffolding.

This module owns launch-command construction, localhost port allocation,
readiness probing, and stdout/stderr capture for the future DS4 engine/proxy.
It intentionally does not implement request forwarding yet.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .ds4_support import DS4SupportStatus, ensure_ds4_support
from .settings import DEFAULT_BASE_PATH, DS4Settings

DS4_HOST = "127.0.0.1"
_DS4_FS_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")
logger = logging.getLogger(__name__)

# Per-process-instance counter for short log identifiers (DS4-1, DS4-2, ...).
_ds4_instance_counter = 0


def _next_ds4_instance_id() -> str:
    global _ds4_instance_counter
    _ds4_instance_counter += 1
    return f"DS4-{_ds4_instance_counter}"


class DS4ProcessError(RuntimeError):
    """Raised when a managed DS4 process cannot start or become ready."""


@dataclass(frozen=True)
class DS4KVPruneResult:
    """Summary of DS4 disk KV cache budget enforcement."""

    root: Path
    max_bytes: int
    files_before: int
    bytes_before: int
    files_after: int
    bytes_after: int
    deleted_files: tuple[Path, ...]
    deleted_bytes: int


@dataclass(frozen=True)
class DS4LogLine:
    """Captured DS4 stdout/stderr line."""

    stream: Literal["stdout", "stderr"]
    text: str
    monotonic_time: float


@dataclass(frozen=True)
class DS4LaunchConfig:
    """Inputs needed to launch one DS4 subprocess."""

    model_id: str
    gguf_path: Path
    settings: DS4Settings = field(default_factory=DS4Settings)
    base_path: Path = field(default_factory=lambda: DEFAULT_BASE_PATH)
    context_tokens: int | None = None
    port: int | None = None
    host: str = DS4_HOST
    auto_enable_ssd_streaming: bool = False
    mtp_path: Path | None = None
    mtp_kind: Literal["legacy_mtp", "dspark"] | None = None
    mtp_draft: int | None = None
    mtp_margin: float | None = None
    trace_timestamp: str | None = None
    platform_system: str | None = None
    platform_machine: str | None = None

    def __post_init__(self) -> None:
        """Enforce DS4's private localhost-only binding for managed launches."""
        if self.host != DS4_HOST:
            raise ValueError("Managed DS4 processes must bind to 127.0.0.1")

    def support_status(self) -> DS4SupportStatus:
        """Validate configured support files for this launch."""
        return ensure_ds4_support(
            self.settings,
            base_path=self.base_path,
            system=self.platform_system,
            machine=self.platform_machine,
        )

    @property
    def support_dir(self) -> Path:
        """Effective support directory used as DS4 --chdir target."""
        return self.settings.get_support_dir(self.base_path)

    @property
    def kv_dir(self) -> Path:
        """Per-model DS4 KV directory under the shared global root."""
        return self.settings.get_kv_root(self.base_path) / safe_ds4_fs_name(
            self.model_id
        )

    @property
    def debug_dir(self) -> Path:
        """Per-model DS4 debug artifact directory."""
        return self.settings.get_debug_dir(self.base_path) / safe_ds4_fs_name(
            self.model_id
        )

    @property
    def log_path(self) -> Path:
        """Persistent stdout/stderr capture path for this DS4 model."""
        return self.debug_dir / "ds4.log"

    @property
    def trace_path(self) -> Path:
        """Trace path for this launch when tracing is enabled."""
        timestamp = self.trace_timestamp or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        filename = f"{safe_ds4_fs_name(self.model_id)}-{timestamp}.trace"
        return self.settings.get_trace_dir(self.base_path) / filename

    def resolve_port(self) -> int:
        """Return configured port or reserve a currently-free localhost port."""
        if self.port is not None:
            return self.port
        return find_free_localhost_port()

    def build_command(self, port: int) -> list[str]:
        """Build the ds4-server argv for this launch."""
        binary = self.settings.get_binary_path(self.base_path)
        args = [
            str(binary),
            "--chdir",
            str(self.support_dir),
            "--model",
            str(self.gguf_path),
            "--host",
            self.host,
            "--port",
            str(port),
            "--power",
            str(self.settings.power),
        ]
        platform_system = self.platform_system or platform.system()
        if platform_system == "Darwin":
            # Require the accelerated backend explicitly. A CPU-only DS4 build
            # otherwise defaults to the reference backend and can consume all
            # host cores for minutes without surfacing a configuration error.
            args.append("--metal")
        context_tokens = self.context_tokens or self.settings.get_auto_context_tokens()
        if context_tokens:
            args.extend(["--ctx", str(context_tokens)])
        if self.mtp_path is not None:
            args.extend(["--mtp", str(self.mtp_path)])
            if self.mtp_kind == "dspark":
                args.append("--dspark")
            else:
                if self.mtp_draft is not None:
                    args.extend(["--mtp-draft", str(self.mtp_draft)])
                if self.mtp_margin is not None:
                    args.extend(["--mtp-margin", str(self.mtp_margin)])
        if self.settings.kv_cache_enabled:
            args.extend(
                [
                    "--kv-disk-dir",
                    str(self.kv_dir),
                    "--kv-disk-space-mb",
                    str(self.settings.kv_disk_space_mb),
                    "--kv-cache-continued-interval-tokens",
                    str(self.settings.kv_cache_continued_interval_tokens),
                ]
            )
            if self.settings.kv_cache_reject_different_quant:
                args.append("--kv-cache-reject-different-quant")
        if self.should_enable_ssd_streaming():
            args.append("--ssd-streaming")
        if self.settings.trace_enabled:
            args.extend(["--trace", str(self.trace_path)])
        return args

    def should_enable_ssd_streaming(self) -> bool:
        """Resolve SSD streaming mode for this launch."""
        if self.mtp_path is not None and self.mtp_kind != "dspark":
            return False
        if self.settings.ssd_streaming == "on":
            return True
        if self.settings.ssd_streaming == "off":
            return False
        return self.auto_enable_ssd_streaming


def safe_ds4_fs_name(model_id: str) -> str:
    """Return a filesystem-safe per-model DS4 artifact directory name."""
    value = _DS4_FS_SAFE_RE.sub("-", model_id.strip()).strip("-.").lower()
    return value or "ds4-model"


def find_free_localhost_port() -> int:
    """Return an available private localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DS4_HOST, 0))
        return int(sock.getsockname()[1])


def _scan_ds4_kv_files(root: Path) -> list[tuple[float, Path, int]]:
    """Return `(mtime, path, size)` for all recursive DS4 `*.kv` files."""
    if not root.exists():
        return []
    files: list[tuple[float, Path, int]] = []
    for path in root.rglob("*.kv"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        files.append((stat.st_mtime, path, int(stat.st_size)))
    return files


def _cleanup_kv_tmp_files(kv_dir: Path) -> None:
    """Delete orphaned .kv.tmp.<pid> files in *kv_dir*.

    ds4-server writes KV checkpoints to ``<sha>.kv.tmp.<pid>`` and renames
    on success.  If the server is SIGKILLed mid-persist the temp file
    remains behind forever — the server's own scan only considers exact
    ``<40-hex>.kv`` names and the ``*.kv`` pruner glob misses the ``.tmp``
    suffix.  These orphans are safe to delete unconditionally: the rename
    never happened so there is no consistent data to preserve, and a
    running server only holds a temp file for its own pid.
    """
    if not kv_dir.exists():
        return
    for path in kv_dir.rglob("*.kv.tmp.*"):
        if not path.is_file():
            continue
        try:
            path.unlink()
            logger.debug("Cleaned up orphaned KV temp file: %s", path)
        except OSError as exc:
            logger.warning("Failed to clean up orphaned KV temp %s: %s", path, exc)


def prune_ds4_kv_cache(root: Path, max_bytes: int) -> DS4KVPruneResult:
    """Prune oldest recursive DS4 `*.kv` files until the root is under budget."""
    root = Path(root).expanduser().resolve()
    before = _scan_ds4_kv_files(root)
    bytes_before = sum(size for _, _, size in before)
    deleted_files: list[Path] = []
    deleted_bytes = 0
    current_bytes = bytes_before

    for _mtime, path, size in sorted(before, key=lambda item: (item[0], str(item[1]))):
        if current_bytes <= max_bytes:
            break
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Failed to prune DS4 KV cache file %s: %s", path, exc)
            continue
        deleted_files.append(path)
        deleted_bytes += size
        current_bytes = max(0, current_bytes - size)

    after = _scan_ds4_kv_files(root)
    return DS4KVPruneResult(
        root=root,
        max_bytes=max_bytes,
        files_before=len(before),
        bytes_before=bytes_before,
        files_after=len(after),
        bytes_after=sum(size for _, _, size in after),
        deleted_files=tuple(deleted_files),
        deleted_bytes=deleted_bytes,
    )


def _readiness_probe(host: str, port: int) -> bool:
    url = f"http://{host}:{port}/v1/models"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=0.2) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


class DS4ManagedProcess:
    """Lifecycle wrapper for one managed ds4-server subprocess."""

    def __init__(self, config: DS4LaunchConfig, *, max_log_lines: int = 500):
        self.config = config
        self.max_log_lines = max_log_lines
        self.process: asyncio.subprocess.Process | None = None
        self.port: int | None = None
        self.command: list[str] | None = None
        self.logs: list[DS4LogLine] = []
        self.log_total: int = 0
        self.log_path: Path | None = None
        self.last_kv_prune_result: DS4KVPruneResult | None = None
        self._became_ready: bool = False
        self._log_tasks: list[asyncio.Task[None]] = []
        self._log_file_handle: "TextIOWrapper | None" = None
        self._instance_id: str = _next_ds4_instance_id()

    @property
    def is_running(self) -> bool:
        """Return True while the subprocess exists and has not exited."""
        return self.process is not None and self.process.returncode is None

    async def start(self) -> None:
        """Launch ds4-server and wait for /v1/models readiness."""
        if self.is_running:
            return
        self._became_ready = False
        logger.info("[%s] Provisioning DS4 engine support if needed", self._instance_id)
        await asyncio.to_thread(self.config.support_status)
        port = self.config.resolve_port()
        self.port = port
        self.command = self.config.build_command(port)
        self._prepare_directories()

        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info("[%s] Loading model: %s", self._instance_id, self.config.model_id)
        self._start_log_capture()

        try:
            await self.wait_ready()
        except DS4ProcessError as error:
            await self.stop()
            message = str(error)
            if "logs=" in message:
                prefix = message.split("logs=", 1)[0]
                raise DS4ProcessError(
                    f"{prefix}logs={self.recent_log_text()}"
                ) from error
            raise
        except Exception:
            await self.stop()
            raise

    async def wait_ready(self) -> None:
        """Wait until DS4 responds to /v1/models or timeout expires."""
        if self.process is None or self.port is None:
            raise DS4ProcessError("DS4 process has not been started")

        deadline = time.monotonic() + (self.config.settings.ready_timeout_ms / 1000.0)
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                await asyncio.sleep(0.05)
                raise DS4ProcessError(
                    "DS4 process exited before readiness: "
                    f"code={self.process.returncode}; logs={self.recent_log_text()}"
                )
            ready = await asyncio.to_thread(
                _readiness_probe, self.config.host, self.port
            )
            if ready:
                self._became_ready = True
                return
            await asyncio.sleep(0.1)

        await asyncio.sleep(0.05)
        raise DS4ProcessError(
            "DS4 readiness timed out after "
            f"{self.config.settings.ready_timeout_ms}ms; logs={self.recent_log_text()}"
        )

    async def stop(self, *, timeout: float = 5.0) -> None:
        """Terminate the subprocess and stop log capture tasks.

        When KV cache is enabled and the server reached readiness, the timeout
        is raised to 60s so ds4-server can persist the live session before
        SIGKILL.  The extended timeout is skipped for pre-readiness failures
        (crash on load, readiness timeout) where no KV state exists to save.
        """
        if self.config.settings.kv_cache_enabled and self._became_ready:
            timeout = max(timeout, 60.0)
        process = self.process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        await self._stop_log_capture()

    def recent_log_text(self) -> str:
        """Return captured stdout/stderr text for diagnostics."""
        return "\n".join(f"{line.stream}: {line.text}" for line in self.logs)

    def _prepare_directories(self) -> None:
        if self.config.settings.kv_cache_enabled:
            kv_root = self.config.settings.get_kv_root(self.config.base_path)
            kv_root.mkdir(parents=True, exist_ok=True)
            self.config.kv_dir.mkdir(parents=True, exist_ok=True)
            # DS4 launch-time KV pruning is intentionally omitted here.
            # ds4-server's ds4_kvstore_open() already evicts the launching
            # model's own directory to budget using its score-based policy
            # (hit decay + token density).  A global mtime-based prune
            # across all models would fight that policy, delete valuable
            # checkpoints from other models, and duplicate work that the
            # server does better at startup.
            self.last_kv_prune_result = None
            # Clean up orphaned .kv.tmp.<pid> files left behind when a
            # previous ds4-server was SIGKILLed mid persist.  The kvstore
            # never scans for stale temp files — it only unlinks its own
            # temp on a failed write — so these orphans leak forever.
            _cleanup_kv_tmp_files(self.config.kv_dir)
        if self.config.settings.trace_enabled:
            self.config.trace_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.settings.logs_to_disk:
            self.config.debug_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.config.log_path
            self._append_log_file_header()
        else:
            self.log_path = None

    def _start_log_capture(self) -> None:
        if self.process is None:
            return
        # Open a persistent log file handle so per-line writes do not
        # open/close the file on every event-loop tick.
        if self.log_path is not None:
            try:
                self._log_file_handle = self.log_path.open("a", encoding="utf-8")
            except OSError as exc:
                logger.warning("Cannot open DS4 log %s: %s", self.log_path, exc)
                self._log_file_handle = None
        if self.process.stdout is not None:
            self._log_tasks.append(
                asyncio.create_task(self._capture_stream("stdout", self.process.stdout))
            )
        if self.process.stderr is not None:
            self._log_tasks.append(
                asyncio.create_task(self._capture_stream("stderr", self.process.stderr))
            )

    async def _stop_log_capture(self) -> None:
        if not self._log_tasks:
            return
        await asyncio.gather(*self._log_tasks, return_exceptions=True)
        self._log_tasks.clear()
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except OSError:
                pass
            finally:
                self._log_file_handle = None

    def _append_log_file_header(self) -> None:
        if self.log_path is None:
            return
        timestamp = datetime.now(UTC).isoformat()
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n# DS4 launch {timestamp}\n")
                handle.write(f"model_id: {self.config.model_id}\n")
                if self.command:
                    handle.write(f"command: {' '.join(self.command)}\n")
        except OSError as exc:
            logger.warning(
                "Failed to write DS4 log header to %s: %s", self.log_path, exc
            )

    def _append_log_file_line(self, line: DS4LogLine) -> None:
        if self.log_path is None or not self.config.settings.logs_to_disk:
            return
        timestamp = datetime.now(UTC).isoformat()
        try:
            handle = self._log_file_handle
            if handle is not None:
                handle.write(f"{timestamp} {line.stream}: {line.text}\n")
                handle.flush()
            else:
                # Fallback: open/close if the persistent handle was unavailable.
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"{timestamp} {line.stream}: {line.text}\n")
        except OSError as exc:
            logger.warning("Failed to write DS4 log line to %s: %s", self.log_path, exc)

    async def _capture_stream(
        self,
        stream: Literal["stdout", "stderr"],
        reader: asyncio.StreamReader,
    ) -> None:
        while True:
            line = await reader.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            log_line = DS4LogLine(stream, text, time.monotonic())
            self.logs.append(log_line)
            self.log_total += 1
            self._append_log_file_line(log_line)
            logger.info("[%s] %s: %s", self._instance_id, stream, text)
            if len(self.logs) > self.max_log_lines:
                del self.logs[: len(self.logs) - self.max_log_lines]
