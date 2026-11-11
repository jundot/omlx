# SPDX-License-Identifier: Apache-2.0
"""Keeps one loaded model's prefill helper running, attached and reported.

One worker thread walks the steps (find the Mac, check its oMLX and memory,
find or copy the files, start the helper, attach it) and then watches the
connection. A lost helper is detached at once and retried with backoff;
problems a retry cannot fix (version, memory, disk, pairing) wait for the
user. Model files are copied at once when 50 GB stay free, else after the
user agrees. While this Mac loads the model, the helper is copied to and
started on the other Mac, and the loaded model takes it over; nothing is
copied or started twice. The measured speeds of each model and helper Mac are kept on disk, so
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

from ..cluster.identity import get_node_identity
from ..cluster.pairing import get_pairing_manager
from ..cluster.registry import default_devices_path
from . import agent
from . import launcher as default_launcher
from .adapter import StageAdapter
from .client import HelperClient
from .launcher import UnreachableError
from .lending import forget_host
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
# A prompt that could use a helper in backoff retries it at most this often.
_RETRY_ON_USE_S = 15.0
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


def transport_for(node_id: str) -> Any:
    """What drives a paired device: SSH for a Mac, the control channel for a
    helper daemon. Both modules offer the same functions."""
    try:
        key = get_pairing_manager().control_key(node_id)
    except RuntimeError:
        # No pairing manager, so no helper daemon was ever paired here.
        key = None
    return agent if key is not None else default_launcher


def helper_budget(info: dict) -> list[int]:
    """Bytes the helper may hold on each of its GPUs, from its ``remote_info``."""
    if "budgets" in info:
        # A helper daemon reports the free memory of each GPU itself.
        return [int(b) for b in info["budgets"]]
    budget = int(info["budget"])
    if info.get("memory"):
        budget = min(budget, int(info["memory"]) - _OS_BYTES)
    if info.get("available"):
        # Models the helper Mac's own server holds stay where they are; a
        # helper running there for another model makes way for this one.
        lending = info.get("lending") or {}
        freed = int(float(lending.get("active_gib") or 0) * 2**30)
        budget = min(budget, int(info["available"]) + freed)
    return [budget]


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


def fit_layers(
    shared: int, per_layer: list[int], budgets: list[int], limit: int
) -> int:
    """How many leading layers fit in a helper's memory, one GPU after another.

    The GPU with the most room comes first and also holds the tensors outside
    the layers; the helper places its stages the same way, on at most two GPUs.
    """
    held, layers = 0, per_layer[:limit]
    for index, budget in enumerate(sorted(budgets, reverse=True)[:2]):
        start, used = held, _RESERVE_BYTES + (shared if index == 0 else 0)
        while held < len(layers) and used + layers[held] <= budget:
            used, held = used + layers[held], held + 1
        if held == start:
            # The layers stay in order, so a GPU that holds none ends the chain.
            break
    return held


def _with_time_left(status: dict[str, Any]) -> dict[str, Any]:
    finish_at = status.pop("finish_at", None)
    if finish_at is not None and status.get("state") == "copying":
        status["seconds_left"] = max(0, round(finish_at - time.time()))
    return status


def held_layers(
    adapter: StageAdapter,
    model_path: str,
    budgets: list[int],
    name: str,
    fixed: int | None = None,
) -> int:
    """Layers the helper holds: as many as fit, or the fixed number."""
    shared, per_layer = adapter.memory_plan(model_path)
    limit = adapter.split_limit(len(per_layer))
    held = fit_layers(shared, per_layer, budgets, limit)
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


def _reach(
    launcher: Any, node_id: str, report: Callable[..., None], adapter: StageAdapter
) -> tuple:
    """The helper and its ``remote_info``, if it can serve this family."""
    try:
        target = launcher.resolve_target(node_id)
    except UnreachableError:
        # Asleep, unplugged, or off the network: retried with backoff.
        raise
    except launcher.LaunchError as error:
        raise _PermanentError(str(error)) from error
    forget_host(node_id)
    report("connecting", name=target.name, link=target.link)
    info = launcher.remote_info(target, host=this_host())
    problem = launcher.incompatible(info, adapter, target.name)
    if problem is not None:
        raise _PermanentError(problem)
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
    done, started = 0, time.time()

    def progress(name: str) -> None:
        nonlocal done
        done += files[name]
        # The average speed so far sets when the copy should end.
        finish_at = started + (time.time() - started) * size / done
        report("copying", path=model_dir, done=done, bytes=size, finish_at=finish_at)

    report("copying", path=model_dir, done=0, bytes=size)
    launcher.copy_files(
        target, model_path, model_dir, missing, progress, cancelled=cancelled
    )
    if cancelled is not None and cancelled():
        raise _CancelledError
    return model_dir


class FilePrep:
    """Copies a model's helper files before the model is loaded.

    With ``launch`` ({model_type, warm_tokens, a8_min_tokens}) it then starts
    the helper, for the model being loaded to take over. ``check_only`` only
    reports whether the files are there.
    """

    def __init__(
        self,
        model_path: str,
        adapter: StageAdapter,
        node_id: str,
        layers: int | None,
        launcher: Any = None,
        after: FilePrep | None = None,
        launch: dict[str, Any] | None = None,
        check_only: bool = False,
    ):
        self.model_path, self.adapter = model_path, adapter
        self.launcher = launcher or transport_for(node_id)
        self.check_only = check_only
        self.node_id, self.layers, self._after = node_id, layers, after
        self._launch, self._sealed = launch, False
        # (target, remote_info, held layers, HelperProcess) once started.
        self.launched: tuple | None = None
        self._status: dict[str, Any] = {"state": "connecting", "node_id": node_id}
        self._ok, self._cancel = threading.Event(), threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="omlx-prefill-helper-files", daemon=True
        )
        self._thread.start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return _with_time_left(dict(self._status))

    def copy_files(self) -> None:
        self._ok.set()

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            launched, self.launched = self.launched, None
        if launched is not None:
            launched[3].stop()

    def launch_with(self, launch: dict[str, Any]) -> bool:
        """Start the helper after the files; False once that step has passed."""
        with self._lock:
            if self._sealed:
                return False
            self._launch = launch
            return True

    def adopt(self, node_id: str, layers: int | None, a8: int | None) -> tuple | None:
        """The started helper if it matches the loaded model's choice."""
        with self._lock:
            launched, self.launched = self.launched, None
            wanted = (
                self.node_id,
                self.layers,
                (self._launch or {}).get("a8_min_tokens"),
            )
        if launched is not None and wanted != (node_id, layers, a8):
            launched[3].stop()
            return None
        return launched

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

    def _check(self, target: Any, held: int) -> None:
        files = self.launcher.required_files(self.model_path, held, self.adapter)
        where = self.launcher.locate_files(target, Path(self.model_path).name, files)
        missing = [] if where.get("found") else where["missing"]
        if missing:
            size = sum(files[name] for name in missing)
            self._set("files_missing", held=held, bytes=size, free=where["free"])
        else:
            self._set("files_ready", held=held)

    def _run(self) -> None:
        if self._after is not None:
            # Its last files may still be in flight to the same folder.
            self._after.join()
        try:
            if self._cancel.is_set():
                raise _CancelledError
            target, info = _reach(self.launcher, self.node_id, self._set, self.adapter)
            held = held_layers(
                self.adapter,
                self.model_path,
                helper_budget(info),
                target.name,
                self.layers,
            )
            if self.check_only:
                self._check(target, held)
                return
            model_dir = _prepare_files(
                self.launcher,
                target,
                self.model_path,
                held,
                self.adapter,
                self._set,
                self._ask,
                self._cancel.is_set,
            )
            with self._lock:
                launch, self._sealed = self._launch, True
            if launch is None:
                self._set("files_ready", held=held)
                return
            self._set("starting", layers=held)
            process = self.launcher.launch(
                target,
                model_dir,
                launch["model_type"],
                held,
                host_name=computer_name(),
                warm_tokens=launch["warm_tokens"],
                a8_min_tokens=launch["a8_min_tokens"],
            )
            with self._lock:
                if not self._cancel.is_set():
                    self.launched, process = (target, info, held, process), None
            if process is not None:
                process.stop()
                raise _CancelledError
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
    launch: dict[str, Any] | None = None,
    check_only: bool = False,
    launcher: Any = None,
) -> None:
    """Start, keep or cancel the file copy (and with ``launch`` the helper)."""
    key = str(Path(model_path).expanduser())
    with _PREPS_LOCK:
        prep = _PREPS.pop(key, None)
        if (
            prep is not None
            and not restart
            and (prep.node_id, prep.layers) == (node_id, layers)
            and prep.status()["state"] not in ("error", "stopped")
            # A check never replaces a copy; a copy replaces a check.
            and (check_only or not prep.check_only)
            and (launch is None or prep.launch_with(launch))
        ):
            # Saving the same choice again keeps the copy going.
            _PREPS[key] = prep
            return
        if prep is not None:
            prep.cancel()
        if node_id is not None:
            _PREPS[key] = FilePrep(
                key, adapter, node_id, layers, launcher, prep, launch, check_only
            )


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
        launcher: Any = None,
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

    def retry_if_offline(self) -> None:
        """Only when offline: any queued command cancels a copy waiting for the user."""
        with self._lock:
            status = self._status
            if (
                status["state"] == "offline"
                and time.time() - status["since"] >= _RETRY_ON_USE_S
                and self._commands.empty()
            ):
                # Unlike Retry, a tick keeps the backoff growing if it fails.
                self._commands.put(("tick",))

    def stop(self) -> None:
        """Stop the helper but keep the choice; Retry starts it again."""
        self._commands.put(("stop",))

    def close(self) -> None:
        self._commands.put(("close",))
        self._thread.join(timeout=30)

    def status(self) -> dict[str, Any]:
        with self._lock:
            status = _with_time_left(dict(self._status))
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
        launcher = self.launcher or transport_for(self.node_id)
        self._set("connecting", node_id=self.node_id)
        prep = _take_prep(self.model_path)
        adopted = None
        if prep is not None:
            self._follow(prep)
            adopted = prep.adopt(self.node_id, self.fixed_layers, self.a8_min_tokens)
        if adopted is not None:
            target, info, held, process = adopted
        else:
            target, info = _reach(launcher, self.node_id, self._report, self.adapter)
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
        # A helper daemon counts what it really loads and may hold fewer
        # layers than this Mac's estimate from the file sizes.
        held = min(held, int(client.info.get("layers") or held))
        saved = self._saved_speeds()
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
            unit_extra=saved.get("extra"),
            alone_ratio=saved.get("alone"),
        )
        self._saved = tuple(saved.get(k) for k in ("helper", "host", "extra", "alone"))
        self.attach(self._session)
        self._set(
            "ready",
            node_id=self.node_id,
            held=held,
            device=info.get("device"),
            memory=info.get("memory"),
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
            if status["state"] in ("needs_files", "copying", "starting"):
                self._set(**status)
            if self._copy_ok.is_set():
                prep.copy_files()
            if not self._commands.empty():
                prep.cancel()
                prep.join()
                raise RuntimeError("cancelled while copying files")
            prep.join(timeout=1.0)

    def _speeds_key(self) -> str:
        # The folder name survives moving or linking the model elsewhere. A8
        # changes the helper's speed, so each setting keeps its own.
        a8 = ":a8" if self.a8_min_tokens is not None else ""
        return f"{self.node_id}:{Path(self.model_path).name}{a8}"

    def _saved_speeds(self) -> dict[str, Any]:
        data = self._load_speeds()
        if self._speeds_key() in data:
            return data[self._speeds_key()]
        # Older entries were keyed by the full model path.
        a8 = ":a8" if self.a8_min_tokens is not None else ""
        older = [
            value
            for key, value in data.items()
            if key.startswith(f"{self.node_id}:")
            and key.endswith(f"/{Path(self.model_path).name}{a8}")
        ]
        return max(older, key=lambda value: value.get("at", 0), default={})

    def _load_speeds(self) -> dict[str, Any]:
        try:
            return json.loads(self.speeds_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save_speeds(self) -> None:
        session = self._session
        speeds = (
            session.unit_helper,
            session.unit_host,
            session.unit_extra,
            session.alone_ratio,
        )
        if None in speeds[:2] or speeds == self._saved:
            return
        with _SPEEDS_LOCK:
            data = self._load_speeds()
            data[self._speeds_key()] = {
                "helper": speeds[0],
                "host": speeds[1],
                "extra": speeds[2],
                "alone": speeds[3],
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
