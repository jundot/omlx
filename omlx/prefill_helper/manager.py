# SPDX-License-Identifier: Apache-2.0
"""Keeps one loaded model's prefill helper running, attached and reported.

One worker thread walks the steps (find the Mac, check its oMLX and memory,
find or copy the files, start the helper, attach it) and then watches the
connection. A lost helper is detached at once and retried with backoff;
problems a retry cannot fix (version, memory, disk, pairing) wait for the
user. Copying model files only starts on an explicit request. The measured
speeds of each model and helper Mac are kept on disk, so a restarted helper
splits its first prompt by them instead of guessing.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import __version__
from ..cluster.registry import default_devices_path
from . import launcher as default_launcher
from .adapter import StageAdapter
from .client import HelperClient
from .session import PrefillHelperSession

logger = logging.getLogger(__name__)

# Activations of one chunk and allocator headroom on the helper.
_RESERVE_BYTES = 8 << 30
# macOS and the helper Mac's other apps; the GPU budget alone let the helper
# swap (105 GiB of layers on a 128 GB Mac).
_OS_BYTES = 16 << 30
_RETRY_DELAYS_S = (15.0, 30.0, 60.0, 120.0, 300.0)
_WATCH_S = 2.0
# A helper with open jobs that sends nothing for this long is gone.
_STALL_S = 60.0
_SPEEDS_LOCK = threading.Lock()


def default_speeds_path() -> Path:
    return default_devices_path().with_name("prefill_helper_speeds.json")


class _PermanentError(Exception):
    """A problem only the user can fix; no automatic retry."""


def helper_budget(info: dict) -> int:
    """Bytes the helper may hold, from its ``remote_info``."""
    budget = int(info["budget"])
    if info.get("memory"):
        budget = min(budget, int(info["memory"]) - _OS_BYTES)
    if info.get("available"):
        # Models the helper Mac's own server holds stay where they are.
        budget = min(budget, int(info["available"]))
    return budget


def fit_layers(shared: int, per_layer: list[int], budget: int, limit: int) -> int:
    """How many leading layers fit in a helper's memory budget."""
    held, used = 0, shared + _RESERVE_BYTES
    for size in per_layer[:limit]:
        if used + size > budget:
            break
        used, held = used + size, held + 1
    return held


def _computer_name() -> str:
    try:
        name = subprocess.run(
            ["scutil", "--get", "ComputerName"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    return name or socket.gethostname()


class PrefillHelperManager:
    def __init__(
        self,
        *,
        model_path: str,
        model_type: str,
        adapter: StageAdapter,
        language_model: Any,
        attach: Callable[[PrefillHelperSession | None], None],
        chunk_tokens: int,
        launcher: Any = default_launcher,
        connect: Callable[..., HelperClient] = HelperClient,
        speeds_path: Path | None = None,
    ):
        self.model_path = str(Path(model_path).expanduser())
        self.speeds_path = speeds_path or default_speeds_path()
        self._saved: tuple | None = None
        self.model_type, self.adapter, self.lm = model_type, adapter, language_model
        self.attach, self.chunk_tokens = attach, chunk_tokens
        self.launcher, self.connect = launcher, connect
        self.node_id: str | None = None
        self.fixed_layers: int | None = None
        self._status: dict[str, Any] = {"state": "off"}
        self._commands: queue.Queue = queue.Queue()
        self._process = None
        self._session: PrefillHelperSession | None = None
        self._copy_ok = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="omlx-prefill-helper", daemon=True
        )
        self._thread.start()

    # -- commands (any thread) ------------------------------------------------

    def configure(self, node_id: str | None, layers: int | None) -> None:
        self._commands.put(("configure", node_id, layers))

    def copy_files(self) -> None:
        self._copy_ok.set()

    def retry(self) -> None:
        self._commands.put(("retry",))

    def stop(self) -> None:
        """Stop the helper but keep the choice; Retry starts it again."""
        self._commands.put(("stop",))

    def close(self) -> None:
        self._commands.put(("close",))
        self._thread.join(timeout=30)

    def status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self._status)
        session = self._session
        if session is not None and status.get("state") == "ready":
            status["layers"] = session.pick_layers()
            status["last_prompt"] = session.last_prompt
            if session.client.job_count():
                status["state"] = "working"
        return status

    # -- worker thread ---------------------------------------------------------

    def _set(self, state: str, **fields: Any) -> None:
        with self._lock:
            keep = {k: self._status[k] for k in ("name", "link") if k in self._status}
            self._status = {"state": state, **keep, **fields, "since": time.time()}

    def _run(self) -> None:
        attempt, wanted = 0, False
        while True:
            timeout = None
            if wanted and self._session is not None:
                timeout = _WATCH_S
            elif wanted and attempt:
                timeout = _RETRY_DELAYS_S[min(attempt - 1, len(_RETRY_DELAYS_S) - 1)]
            try:
                command = self._commands.get(timeout=timeout)
            except queue.Empty:
                command = ("tick",)
            kind = command[0]
            if kind == "close":
                self._teardown()
                return
            if kind == "configure":
                _, node_id, layers = command
                if (node_id, layers) == (self.node_id, self.fixed_layers):
                    # Saving other settings must not restart a stopped helper.
                    continue
                # Speeds are saved under the helper that measured them.
                self._teardown()
                self.node_id, self.fixed_layers = node_id, layers
                wanted, attempt = node_id is not None, 0
                if not wanted:
                    self._set("off")
                    continue
            elif kind == "stop":
                self._teardown()
                wanted = False
                self._set("stopped", node_id=self.node_id)
                continue
            elif kind == "retry":
                wanted, attempt = self.node_id is not None, 0
            if not wanted:
                continue
            if self._session is not None:
                client = self._session.client
                if client.stalled(_STALL_S):
                    self._session.note_failure("the helper stopped answering")
                    client.close()
                if self._session.alive:
                    self._save_speeds()
                    continue
                name = self._session.name
                stopped = self._session.client.stopped_by_helper
                reason = self._session.last_error or "the connection was lost"
                self._teardown()
                if stopped:
                    # The user stopped it on that Mac; do not start it again.
                    wanted = False
                    self._set(
                        "stopped", node_id=self.node_id, reason=f"Stopped on {name}"
                    )
                else:
                    attempt = 1
                    self._set("offline", reason=reason)
                continue
            try:
                self._start()
                attempt = 0
            except _PermanentError as error:
                wanted = False
                self._set("error", reason=str(error))
            except Exception as error:  # network, SSH, helper crash
                logger.warning("Prefill helper did not start: %s", error)
                attempt += 1
                self._set("offline", reason=str(error))

    def _start(self) -> None:
        launcher = self.launcher
        self._set("connecting", node_id=self.node_id)
        try:
            target = launcher.resolve_target(self.node_id)
        except launcher.LaunchError as error:
            raise _PermanentError(str(error)) from error
        self._set(
            "connecting", node_id=self.node_id, name=target.name, link=target.link
        )
        info = launcher.remote_info(target)
        if info.get("version") != __version__:
            raise _PermanentError(
                f"{target.name} runs oMLX {info.get('version')}; update it to "
                f"{__version__}."
            )
        held = self._held_layers(target.name, helper_budget(info))
        files = launcher.required_files(self.model_path, held, self.adapter)
        where = launcher.locate_files(target, Path(self.model_path).name, files)
        model_dir = where.get("found")
        if model_dir is None:
            missing = where["missing"]
            size = sum(files[name] for name in missing)
            if size > where["free"]:
                raise _PermanentError(
                    f"{target.name} needs {size / 2**30:.0f} GiB free for the "
                    f"model files; {where['free'] / 2**30:.0f} GiB is free."
                )
            model_dir = where["copy"]
            if missing:
                self._copy(target, model_dir, missing, files, size, free=where["free"])
        self._set("starting", node_id=self.node_id, layers=held)
        process = launcher.launch(
            target,
            model_dir,
            self.model_type,
            held,
            host_name=_computer_name(),
            warm_tokens=self.chunk_tokens,
        )
        try:
            client = self.connect(target.address, process.port, process.token)
        except Exception:
            process.stop()
            raise
        self._process = process
        saved = self._load_speeds().get(self._speeds_key(), {})
        self._session = PrefillHelperSession(
            client,
            self.adapter,
            self.lm,
            held,
            target.name,
            self.adapter.layer_work(self.lm),
            fixed_layers=self.fixed_layers,
            unit_helper=saved.get("helper"),
            unit_host=saved.get("host"),
        )
        self._saved = (saved.get("helper"), saved.get("host"))
        self.attach(self._session)
        self._set(
            "ready",
            node_id=self.node_id,
            held=held,
            device=info.get("device"),
            on_battery=bool(info.get("on_battery")),
        )

    def _held_layers(self, name: str, budget: int) -> int:
        shared, per_layer = self.adapter.memory_plan(self.model_path)
        limit = self.adapter.max_split(self.lm)
        held = fit_layers(shared, per_layer, budget, limit)
        if self.fixed_layers is not None:
            if self.fixed_layers > limit:
                raise _PermanentError(
                    f"At most {limit} layers of this model can run on a helper."
                )
            if self.fixed_layers > held:
                raise _PermanentError(
                    f"{self.fixed_layers} layers do not fit in {name}'s GPU memory; "
                    f"at most {held} do."
                )
            held = self.fixed_layers
        if held < 1:
            raise _PermanentError(f"Not even one layer fits in {name}'s GPU memory.")
        return held

    def _copy(self, target, model_dir, missing, files, size, *, free) -> None:
        self._copy_ok.clear()
        self._set(
            "needs_files",
            node_id=self.node_id,
            path=model_dir,
            bytes=size,
            free=free,
        )
        while not self._copy_ok.wait(timeout=1.0):
            if not self._commands.empty():
                raise RuntimeError("cancelled while waiting to copy files")
        done = 0

        def progress(name: str) -> None:
            nonlocal done
            done += files[name]
            self._set(
                "copying", node_id=self.node_id, path=model_dir, done=done, bytes=size
            )

        self._set("copying", node_id=self.node_id, path=model_dir, done=0, bytes=size)
        self.launcher.copy_files(target, self.model_path, model_dir, missing, progress)

    def _speeds_key(self) -> str:
        return f"{self.node_id}:{self.model_path}"

    def _load_speeds(self) -> dict[str, Any]:
        try:
            return json.loads(self.speeds_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save_speeds(self) -> None:
        session = self._session
        speeds = (session.unit_helper, session.unit_host)
        if None in speeds or speeds == self._saved:
            return
        with _SPEEDS_LOCK:
            data = self._load_speeds()
            data[self._speeds_key()] = {
                "helper": speeds[0],
                "host": speeds[1],
                "at": time.time(),
            }
            try:
                self.speeds_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.speeds_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(data, indent=1))
                os.replace(temporary, self.speeds_path)
            except OSError as error:
                logger.warning("Could not save prefill helper speeds: %s", error)
                return
        self._saved = speeds

    def _teardown(self) -> None:
        if self._session is not None:
            self._save_speeds()
            self.attach(None)
            self._session.close()
            self._session = None
        if self._process is not None:
            self._process.stop()
            self._process = None
