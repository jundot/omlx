# SPDX-License-Identifier: Apache-2.0
"""Keeps one loaded model's prefill helper running, attached and reported.

One worker thread walks the steps (find the Mac, check its oMLX and memory,
find or copy the files, start the helper, attach it) and then watches the
connection. A lost helper is detached at once and retried with backoff;
problems a retry cannot fix (version, memory, disk, pairing) wait for the
user. Model files are copied at once when 50 GB stay free, else after the
user agrees; a copy started before the model loads is taken over, never run
twice. The measured speeds of each model and helper Mac are kept on disk, so
a restarted helper splits its first prompt by them instead of guessing.
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
from ..cluster.identity import get_node_identity
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
# Disk a model copy leaves free on the helper Mac without asking.
_DISK_SPARE_BYTES = 50 << 30
_RETRY_DELAYS_S = (15.0, 30.0, 60.0, 120.0, 300.0)
_WATCH_S = 2.0
# A helper with open jobs that sends nothing for this long is gone.
_STALL_S = 60.0
_SPEEDS_LOCK = threading.Lock()


def default_speeds_path() -> Path:
    return default_devices_path().with_name("prefill_helper_speeds.json")


class _PermanentError(Exception):
    """A problem only the user can fix; no automatic retry."""


class _CancelledError(Exception):
    """The helper choice changed while its files were being prepared."""


def helper_budget(info: dict) -> int:
    """Bytes the helper may hold, from its ``remote_info``."""
    budget = int(info["budget"])
    if info.get("memory"):
        budget = min(budget, int(info["memory"]) - _OS_BYTES)
    if info.get("available"):
        # Models the helper Mac's own server holds stay where they are; a
        # helper running there for another model makes way for this one.
        lending = info.get("lending") or {}
        freed = int(float(lending.get("active_gib") or 0) * 2**30)
        budget = min(budget, int(info["available"]) + freed)
    return budget


def copy_plan(size: int, free: int, copies: list[dict]) -> dict:
    """How to fit *size* bytes of model files on the helper Mac.

    ``copy`` leaves the spare free and runs at once. ``make_room`` first removes
    the least recently used other copies, ``low_space`` fills the disk past the
    spare; both wait for the user. ``no_space`` cannot fit without removing
    more than the other copies.
    """
    if free - size >= _DISK_SPARE_BYTES:
        return {"kind": "copy", "delete": [], "left": free - size}
    freed, delete = 0, []
    for copy in sorted(copies, key=lambda item: item["used"]):
        delete.append(copy)
        freed += copy["bytes"]
        if free + freed - size >= _DISK_SPARE_BYTES:
            return {"kind": "make_room", "delete": delete, "left": free + freed - size}
    kind = "low_space" if free >= size else "no_space"
    return {"kind": kind, "delete": [], "left": free - size}


def fit_layers(shared: int, per_layer: list[int], budget: int, limit: int) -> int:
    """How many leading layers fit in a helper's memory budget."""
    held, used = 0, shared + _RESERVE_BYTES
    for size in per_layer[:limit]:
        if used + size > budget:
            break
        used, held = used + size, held + 1
    return held


def held_layers(
    adapter: StageAdapter,
    model_path: str,
    budget: int,
    name: str,
    fixed: int | None = None,
) -> int:
    """Layers the helper holds: as many as fit, or the fixed number."""
    shared, per_layer = adapter.memory_plan(model_path)
    limit = adapter.split_limit(len(per_layer))
    held = fit_layers(shared, per_layer, budget, limit)
    if fixed is not None:
        if fixed > limit:
            raise _PermanentError(
                f"At most {limit} layers of this model can run on a helper."
            )
        if fixed > held:
            raise _PermanentError(
                f"{fixed} layers do not fit in {name}'s GPU memory; at most {held} do."
            )
        held = fixed
    if held < 1:
        raise _PermanentError(f"Not even one layer fits in {name}'s GPU memory.")
    return held


def this_host() -> dict | None:
    """This Mac as the helper Mac records it, once the cluster identity exists."""
    try:
        identity = get_node_identity()
    except RuntimeError:
        return None
    return {"node_id": identity.node_id, "name": identity.friendly_name}


def _reach(launcher: Any, node_id: str, report: Callable[..., None]) -> tuple:
    """The helper Mac and its ``remote_info``, if it runs this oMLX."""
    try:
        target = launcher.resolve_target(node_id)
    except launcher.LaunchError as error:
        raise _PermanentError(str(error)) from error
    report("connecting", name=target.name, link=target.link)
    info = launcher.remote_info(target, host=this_host())
    if info.get("version") != __version__:
        raise _PermanentError(
            f"{target.name} runs oMLX {info.get('version')}; update it to "
            f"{__version__}."
        )
    return target, info


def _prepare_files(
    launcher: Any,
    target: Any,
    model_path: str,
    held: int,
    adapter: StageAdapter,
    report: Callable[..., None],
    ask: Callable[..., None],
    cancelled: Callable[[], bool] | None = None,
) -> str:
    """The folder on the helper Mac with the files for *held* layers.

    Missing files are copied there. ``ask`` blocks until the user agrees to
    remove other copies or fill the disk.
    """
    files = launcher.required_files(model_path, held, adapter)
    where = launcher.locate_files(target, Path(model_path).name, files)
    if where.get("found"):
        return where["found"]
    model_dir, missing = where["copy"], where["missing"]
    if not missing:
        return model_dir
    size = sum(files[name] for name in missing)
    plan = copy_plan(size, where["free"], where.get("copies", []))
    if plan["kind"] == "no_space":
        raise _PermanentError(
            f"{target.name} needs {size / 2**30:.0f} GiB free for the "
            f"model files; {where['free'] / 2**30:.0f} GiB is free."
        )
    if plan["kind"] != "copy":
        ask(
            path=model_dir,
            bytes=size,
            free=where["free"],
            prompt=plan["kind"],
            left=plan["left"],
            delete=[{"name": c["name"], "bytes": c["bytes"]} for c in plan["delete"]],
        )
    if plan["delete"]:
        launcher.delete_copies(target, [item["name"] for item in plan["delete"]])
    done = 0

    def progress(name: str) -> None:
        nonlocal done
        done += files[name]
        report("copying", path=model_dir, done=done, bytes=size)

    report("copying", path=model_dir, done=0, bytes=size)
    launcher.copy_files(
        target, model_path, model_dir, missing, progress, cancelled=cancelled
    )
    if cancelled is not None and cancelled():
        raise _CancelledError
    return model_dir


class FilePrep:
    """Copies a model's helper files before the model is loaded."""

    def __init__(
        self,
        model_path: str,
        adapter: StageAdapter,
        node_id: str,
        layers: int | None,
        launcher: Any = default_launcher,
        after: FilePrep | None = None,
    ):
        self.model_path, self.adapter, self.launcher = model_path, adapter, launcher
        self.node_id, self.layers, self._after = node_id, layers, after
        self._status: dict[str, Any] = {"state": "connecting", "node_id": node_id}
        self._ok, self._cancel = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="omlx-prefill-helper-files", daemon=True
        )
        self._thread.start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def copy_files(self) -> None:
        self._ok.set()

    def cancel(self) -> None:
        self._cancel.set()

    def running(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _set(self, state: str, **fields: Any) -> None:
        with self._lock:
            keep = {k: self._status[k] for k in ("name", "link") if k in self._status}
            self._status = {"state": state, "node_id": self.node_id, **keep, **fields}

    def _ask(self, **fields: Any) -> None:
        self._set("needs_files", **fields)
        while not self._ok.wait(timeout=1.0):
            if self._cancel.is_set():
                raise _CancelledError

    def _run(self) -> None:
        if self._after is not None:
            # Its last files may still be in flight to the same folder.
            self._after.join()
        try:
            if self._cancel.is_set():
                raise _CancelledError
            target, info = _reach(self.launcher, self.node_id, self._set)
            held = held_layers(
                self.adapter,
                self.model_path,
                helper_budget(info),
                target.name,
                self.layers,
            )
            _prepare_files(
                self.launcher,
                target,
                self.model_path,
                held,
                self.adapter,
                self._set,
                self._ask,
                self._cancel.is_set,
            )
            self._set("files_ready", held=held)
        except _CancelledError:
            self._set("stopped")
        except Exception as error:  # SSH, version, memory, disk
            logger.warning("Prefill helper files not prepared: %s", error)
            self._set("error", reason=str(error))


_PREPS: dict[str, FilePrep] = {}
_PREPS_LOCK = threading.Lock()


def prepare(
    model_path: str,
    adapter: StageAdapter,
    node_id: str | None,
    layers: int | None,
    *,
    restart: bool = False,
    launcher: Any = default_launcher,
) -> None:
    """Start, keep or cancel the file copy for a model that is not loaded."""
    key = str(Path(model_path).expanduser())
    with _PREPS_LOCK:
        prep = _PREPS.pop(key, None)
        if (
            prep is not None
            and not restart
            and (prep.node_id, prep.layers) == (node_id, layers)
            and prep.status()["state"] not in ("error", "stopped")
        ):
            # Saving the same choice again keeps the copy going.
            _PREPS[key] = prep
            return
        if prep is not None:
            prep.cancel()
        if node_id is not None:
            _PREPS[key] = FilePrep(key, adapter, node_id, layers, launcher, prep)


def file_prep(model_path: str) -> FilePrep | None:
    with _PREPS_LOCK:
        return _PREPS.get(str(Path(model_path).expanduser()))


def _take_prep(model_path: str) -> FilePrep | None:
    with _PREPS_LOCK:
        return _PREPS.pop(model_path, None)


def computer_name() -> str:
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
        # The model's A8 prefill setting, which the helper's layers follow.
        self.a8_min_tokens: int | None = None
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

    def configure(
        self,
        node_id: str | None,
        layers: int | None,
        a8_min_tokens: int | None = None,
    ) -> None:
        self._commands.put(("configure", node_id, layers, a8_min_tokens))

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
                _, node_id, layers, a8 = command
                if (node_id, layers, a8) == (
                    self.node_id,
                    self.fixed_layers,
                    self.a8_min_tokens,
                ):
                    # Saving other settings must not restart a stopped helper.
                    continue
                # Speeds are saved under the helper that measured them.
                self._teardown()
                self.node_id, self.fixed_layers, self.a8_min_tokens = (
                    node_id,
                    layers,
                    a8,
                )
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
                replaced = self._session.client.replaced_on_helper
                reason = self._session.last_error or "the connection was lost"
                self._teardown()
                if stopped:
                    # The user stopped it on that Mac; do not start it again.
                    wanted = False
                    self._set(
                        "stopped",
                        node_id=self.node_id,
                        reason=(
                            f"{name} is now helping another model"
                            if replaced
                            else f"Stopped on {name}"
                        ),
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

    def _report(self, state: str, **fields: Any) -> None:
        self._set(state, node_id=self.node_id, **fields)

    def _start(self) -> None:
        launcher = self.launcher
        self._set("connecting", node_id=self.node_id)
        prep = _take_prep(self.model_path)
        if prep is not None:
            self._follow(prep)
        target, info = _reach(launcher, self.node_id, self._report)
        held = held_layers(
            self.adapter,
            self.model_path,
            helper_budget(info),
            target.name,
            self.fixed_layers,
        )
        model_dir = _prepare_files(
            launcher,
            target,
            self.model_path,
            held,
            self.adapter,
            self._report,
            self._ask,
        )
        self._set("starting", node_id=self.node_id, layers=held)
        process = launcher.launch(
            target,
            model_dir,
            self.model_type,
            held,
            host_name=computer_name(),
            warm_tokens=self.chunk_tokens,
            a8_min_tokens=self.a8_min_tokens,
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

    def _ask(self, **fields: Any) -> None:
        # Removing other copies or filling the disk waits for the user.
        self._copy_ok.clear()
        self._report("needs_files", **fields)
        while not self._copy_ok.wait(timeout=1.0):
            if not self._commands.empty():
                raise RuntimeError("cancelled while waiting to copy files")

    def _follow(self, prep: FilePrep) -> None:
        """Wait for a copy started before the load; never copy alongside it."""
        if prep.node_id != self.node_id:
            prep.cancel()
            prep.join()
            return
        self._copy_ok.clear()
        while prep.running():
            status = prep.status()
            if status["state"] in ("needs_files", "copying"):
                self._set(**status)
            if self._copy_ok.is_set():
                prep.copy_files()
            if not self._commands.empty():
                prep.cancel()
                prep.join()
                raise RuntimeError("cancelled while copying files")
            prep.join(timeout=1.0)

    def _speeds_key(self) -> str:
        # A8 changes the helper's speed, so each setting keeps its own.
        a8 = ":a8" if self.a8_min_tokens is not None else ""
        return f"{self.node_id}:{self.model_path}{a8}"

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
