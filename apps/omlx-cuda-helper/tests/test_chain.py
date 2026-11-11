# SPDX-License-Identifier: Apache-2.0
"""Two GPU stages in a chain must answer like one stage with all the layers."""

from __future__ import annotations

import queue
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_family.stage import FakeStage  # noqa: E402
from omlx_cuda_helper import families, gpus, wire, worker  # noqa: E402
from omlx_cuda_helper.control import ControlServer  # noqa: E402
from omlx_cuda_helper.files import ModelStore  # noqa: E402
from omlx_cuda_helper.state import State  # noqa: E402
from omlx_cuda_helper.supervisor import Supervisor, plan_stages  # noqa: E402
from omlx_cuda_helper.worker import (  # noqa: E402
    WINDOW,
    StageError,
    StageServer,
    Upstream,
)

TOKEN = b"t" * 64
KEY = bytes(range(32))
CHUNKS = [5, 3, 4, 2, 6]
TOKENS = np.arange(3, 3 + sum(CHUNKS), dtype=np.int32)
INFO = {"protocol": 1, "warm_seconds": 0.0, "active_gib": 0.0}


@pytest.fixture
def tmp_path():
    """A short folder: Unix socket paths are limited to about 100 bytes."""
    path = Path(tempfile.mkdtemp(prefix="och-", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _serve(stage, tmp_path, name, upstream=None):
    """A stage server on a Unix socket; returns a host connection to it."""
    path = str(tmp_path / name)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(path)
    listener.listen(1)
    server = StageServer(stage, TOKEN, INFO, upstream)
    threading.Thread(target=server.serve, args=(listener,), daemon=True).start()
    sock = socket.socket(socket.AF_UNIX)
    sock.connect(path)
    return Upstream(sock, TOKEN)


def _whole(layers, tmp_path):
    return _serve(FakeStage(0, layers), tmp_path, "whole")


def _chain(cut, layers, tmp_path):
    first = _serve(FakeStage(0, cut), tmp_path, "s0")
    return _serve(FakeStage(cut, layers), tmp_path, "s1", upstream=first)


def _run(
    host, job_id="job", *, layers=None, prefix=None, feed=False, capture=None, late=0.0
):
    meta = {"op": "job", "job_id": job_id, "chunks": CHUNKS}
    if layers is not None:
        meta["layers"] = layers
    if feed:
        meta["feed"] = True
    if capture:
        meta["capture"] = capture
    arrays = {"tokens": (TOKENS.reshape(1, -1), "int32")}
    arrays.update({f"p:{n}": v for n, v in (prefix or {}).items()})
    host.submit(meta, arrays)
    out = []
    for index, size in enumerate(CHUNKS):
        if feed:
            # A host may send a chunk's inputs after the stages reached it.
            time.sleep(late)
            rows = np.full((1, size, 4), index + 1, np.float32)
            host.send(
                {"op": "inputs", "job_id": job_id, "index": index},
                {"rows": (rows, "float32")},
            )
        meta, arrays = host.take(job_id, index)
        out.append({name: data for name, (data, _) in arrays.items()})
    return out


def _until(ready, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not (found := ready()):
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)
    return found


def _same(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert sorted(left) == sorted(right)
        for name in left:
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)


@pytest.mark.parametrize(
    "options",
    [
        {"capture": {"layers": [1, 4], "from": 7}},
        # The second stage's first layer reads the prefix rows of the layer
        # before it, which only the first stage holds.
        {
            "prefix": {
                **{f"{i}.acc": (np.array([i + 0.5]), "float64") for i in range(6)},
                **{
                    f"{i}.rows": (np.full((1, 2, 1), i + 0.25, np.float32), "float32")
                    for i in range(6)
                },
            }
        },
        # A job may use fewer layers than the helper holds: the cut moves into
        # the first stage, or the second stage runs only part of its layers.
        {"layers": 2},
        {"layers": 5, "feed": True},
        {"feed": True, "late": 0.1},
    ],
    ids=[
        "capture",
        "prefix",
        "first-stage-only",
        "partial-second",
        "late-feed",
    ],
)
def test_two_stages_answer_like_one(tmp_path, options):
    _same(_run(_chain(3, 6, tmp_path), **options), _run(_whole(6, tmp_path), **options))


def test_stages_stay_a_window_ahead_of_the_host(tmp_path):
    host = _chain(3, 6, tmp_path)
    host.submit(
        {"op": "job", "job_id": "job", "chunks": CHUNKS},
        {"tokens": (TOKENS.reshape(1, -1), "int32")},
    )
    frames = host._frames["job"]
    # Nothing is acknowledged, so only the first WINDOW chunks may arrive.
    for _ in range(WINDOW):
        frames.get(timeout=10)
    with pytest.raises(queue.Empty):
        frames.get(timeout=0.5)
    host.send({"op": "ack", "job_id": "job", "index": 0})
    assert frames.get(timeout=10)[0]["index"] == WINDOW


class _FailsFirst(FakeStage):
    """Fails the first chunk it runs, like a GPU out of memory."""

    failed = False

    def run(self, job, **chunk):
        if not self.failed:
            self.failed = True
            raise RuntimeError("out of memory")
        return super().run(job, **chunk)


def test_a_job_that_ends_early_frees_both_stages(tmp_path, monkeypatch):
    # A stage that stays in an ended job shows here as a timeout.
    monkeypatch.setattr(worker, "_UPSTREAM_TIMEOUT_S", 10.0)
    first = _serve(FakeStage(0, 3), tmp_path, "s0")
    host = _serve(_FailsFirst(3, 6), tmp_path, "s1", upstream=first)
    # Cancelled while the first stage waits for the host's inputs and the
    # second stage for the first one's chunk.
    host.submit(
        {"op": "job", "job_id": "cancelled", "chunks": CHUNKS, "feed": True},
        {"tokens": (TOKENS.reshape(1, -1), "int32")},
    )
    host.send({"op": "cancel", "job_id": "cancelled"})
    # The second stage fails while the first one is chunks ahead of it.
    with pytest.raises(StageError, match="out of memory"):
        _run(host, "failed")
    _same(_run(host, "next"), _run(_whole(6, tmp_path)))


def test_a_silent_peer_does_not_hold_the_data_port(monkeypatch):
    monkeypatch.setattr(worker, "_AUTH_TIMEOUT_S", 0.2)
    listener, port = worker._listen("tcp:127.0.0.1:0")
    server = StageServer(FakeStage(0, 3), TOKEN, INFO)
    threading.Thread(target=server.serve, args=(listener,), daemon=True).start()
    silent = socket.create_connection(("127.0.0.1", port))
    # The host came second and waits only 10 s for its greeting.
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    assert Upstream(sock, TOKEN).info["protocol"] == 1
    silent.close()


def test_plan_splits_layers_evenly_within_each_gpu_memory():
    gib = 1 << 30
    even = plan_stages(gib, [2 * gib] * 20, 16, [48 * gib, 48 * gib], 2)
    assert [(g, a, b) for g, a, b, _ in even] == [(0, 0, 8), (1, 8, 16)]
    # A full first GPU moves the cut; a full second GPU leaves one stage.
    tight = plan_stages(gib, [2 * gib] * 20, 16, [17 * gib, 48 * gib], 2)
    assert [(g, a, b) for g, a, b, _ in tight] == [(0, 0, 5), (1, 5, 16)]
    single = plan_stages(gib, [2 * gib] * 20, 8, [48 * gib, 7 * gib], 2)
    assert [(g, a, b) for g, a, b, _ in single] == [(0, 0, 8)]
    with pytest.raises(RuntimeError, match="do not fit"):
        plan_stages(gib, [2 * gib] * 20, 16, [10 * gib, 10 * gib], 2)
    # A busy first GPU: the stages swap places instead of giving up a GPU.
    swapped = plan_stages(gib, [2 * gib] * 20, 8, [12 * gib, 48 * gib], 2)
    assert [(g, a, b) for g, a, b, _ in swapped] == [(1, 0, 5), (0, 5, 8)]
    # Of three GPUs, the two with the most free memory hold the stages.
    three = plan_stages(gib, [2 * gib] * 20, 16, [9 * gib, 48 * gib, 48 * gib], 2)
    assert [(g, a, b) for g, a, b, _ in three] == [(1, 0, 8), (2, 8, 16)]


def test_supervisor_runs_stage_processes_and_frees_them(tmp_path, monkeypatch):
    fake_gpus = [
        {"index": 0, "name": "GPU A", "total": 48 << 30, "free": 48 << 30},
        {"index": 1, "name": "GPU B", "total": 48 << 30, "free": 48 << 30},
    ]
    monkeypatch.setattr(gpus, "gpus", lambda only=None: fake_gpus)
    monkeypatch.setitem(
        families.FAMILIES, "fake", families.Family(contract=7, module="fake_family")
    )
    worker = [sys.executable, str(Path(__file__).parent / "fake_worker.py")]
    supervisor = Supervisor(None, tmp_path / "run", worker=worker)
    owner = object()
    ready = supervisor.start(
        owner,
        model_dir=str(tmp_path / "Model"),
        model_type="fake",
        layers=6,
        warm_tokens=8,
        host={"node_id": "studio", "name": "Studio"},
        token=TOKEN,
        bind="127.0.0.1",
    )
    status = supervisor.status()
    assert (status["stages"], status["state"]) == ([3, 6], "ready")
    assert ready["hello"]["layers"] == 6 and ready["hello"]["contract"] == 7
    # The endpoint adds up what both stages hold.
    assert ready["hello"]["active_gib"] == 6.0

    sock = socket.create_connection(("127.0.0.1", ready["port"]), timeout=10)
    host = Upstream(sock, TOKEN)
    _same(_run(host), _run(_whole(6, tmp_path)))

    supervisor.stop(object())  # another connection cannot end the session
    assert supervisor.status() is not None
    procs = list(supervisor._session.procs)
    supervisor.stop(owner)
    assert supervisor.status() is None
    assert all(proc.poll() is not None for proc in procs)


def test_a_loading_session_ends_with_a_stop_or_with_its_host(tmp_path, monkeypatch):
    gpu = {"name": "GPU", "total": 48 << 30, "free": 48 << 30}
    monkeypatch.setattr(
        gpus, "gpus", lambda only=None: [{"index": 0, **gpu}, {"index": 1, **gpu}]
    )
    monkeypatch.setitem(
        families.FAMILIES, "fake", families.Family(contract=7, module="fake_family")
    )
    # Stages that never get ready, like a model that loads for minutes.
    asleep = [sys.executable, "-c", "import time; time.sleep(60)"]
    supervisor = Supervisor(None, tmp_path / "run", worker=asleep)

    def loading():
        session = supervisor._session
        return list(session.procs) if session and len(session.procs) == 2 else None

    def exited(procs):
        return all(proc.poll() is not None for proc in procs)

    # The daemon stops (SIGTERM) while a session loads.
    errors = []

    def start():
        try:
            supervisor.start(
                object(),
                model_dir=str(tmp_path / "Model"),
                model_type="fake",
                layers=6,
                warm_tokens=8,
                host={},
                token=TOKEN,
                bind="127.0.0.1",
            )
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=start)
    thread.start()
    procs = _until(loading)
    supervisor.stop()
    thread.join(timeout=10)
    assert errors == ["the session was stopped while it loaded"] and exited(procs)

    # The host that asked for a session goes away while it loads.
    state = State(tmp_path / "state")
    state.save_host("studio", {"cluster_key": KEY.hex()})
    control = ControlServer(
        state,
        ModelStore(tmp_path / "models"),
        supervisor,
        {"fake": 7},
        bind="127.0.0.1",
    )
    control.start()
    sock = socket.create_connection(("127.0.0.1", control.port), timeout=10)
    wire.send(sock, {"op": "control", "v": 1, "node_id": "studio"})
    wire.authenticate(sock, wire.control_key(KEY), server=False)
    wire.recv(sock)
    request = {"model": "Model", "model_type": "fake", "layers": 6, "contract": 7}
    wire.send(sock, {"op": "start", "session": "00", **request})
    procs = _until(loading)
    sock.close()
    _until(lambda: exited(procs))
    assert supervisor.status() is None
    control.stop()
