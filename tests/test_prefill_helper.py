"""Prefill Accelerator: a split prefill must match the whole-model prefill."""

import itertools
import json
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_deepseek_v41 import _ENGRAM, load_reference_weights, tiny, tiny_ced

from omlx.patches.deepseek_v41.language import LanguageModel
from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding
from omlx.prefill_helper import launcher, wire
from omlx.prefill_helper import manager as manager_module
from omlx.prefill_helper import routes as helper_routes
from omlx.prefill_helper.adapters import adapter_for
from omlx.prefill_helper.client import HelperClient, HelperError
from omlx.prefill_helper.launcher import HelperTarget
from omlx.prefill_helper.manager import PrefillHelperManager
from omlx.prefill_helper.session import PrefillHelperSession
from omlx.prefill_helper.worker import WINDOW, HelperServer
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig

DSV41 = adapter_for("deepseek_v41")
_TABLES = tempfile.TemporaryDirectory(prefix="omlx-prefill-helper-")
_TABLE_IDS = itertools.count()


def _model(config):
    model = LanguageModel(config)
    load_reference_weights(model)
    if config.engram_layer_ids:
        model.set_token_map(np.arange(config.vocab_size))
        # Tables on disk, as a loaded checkpoint has them: this Mac reads the
        # helper's rows from them.
        for layer in model.layers:
            if "engram" in layer:
                path = Path(_TABLES.name) / f"{next(_TABLE_IDS)}.safetensors"
                mx.save_safetensors(str(path), {"weight": layer.engram.embed.weight})
                layer.engram.embed = DiskEngramEmbedding(path, "weight", None)
    return model


def _chunks(ids, sizes):
    out, start = [], 0
    for size in sizes:
        out.append(ids[:, start : start + size])
        start += size
    return out


def _assert_same_cache(expected, actual):
    assert len(expected) == len(actual)
    for i, (a, b) in enumerate(zip(expected, actual)):
        for slot in range(7):
            x, y = a[slot], b[slot]
            assert (x is None) == (y is None), (i, slot)
            if x is not None:
                np.testing.assert_array_equal(
                    np.asarray(x), np.asarray(y), err_msg=f"layer {i} slot {slot}"
                )


def _finish(model, cache, last):
    """Last prompt token, then three greedy steps."""
    logits = model(last, cache=cache)[:, -1]
    out = [np.asarray(logits)]
    token = mx.argmax(logits, -1)[:, None]
    for _ in range(3):
        logits = model(token, cache=cache)[:, -1]
        out.append(np.asarray(logits))
        token = mx.argmax(logits, -1)[:, None]
    return out


def _run(config, k, sizes, prefix=0):
    """Whole-model reference vs split prefill over the same chunks.

    ``prefix`` chunks run on this Mac alone first, like a cached prefix.
    """
    host, helper = _model(config), _model(config)
    DSV41.keep_first(helper, k)
    ids = mx.array(np.random.default_rng(7).integers(3, 60, (1, sum(sizes) + 1)))
    chunks = _chunks(ids, sizes)
    nexts = [c.shape[1] for c in chunks[1:]] + [0]

    reference = host.make_cache()
    for chunk, nxt in zip(chunks, nexts):
        host._omlx_prefill(chunk, cache=reference, next_prefill_tokens=nxt)

    cache, helper_cache = host.make_cache(), helper.make_cache()
    for chunk, nxt in zip(chunks[:prefix], nexts[:prefix]):
        host._omlx_prefill(chunk, cache=cache, next_prefill_tokens=nxt)
    if prefix:
        DSV41.load_prefix(helper_cache, DSV41.prefix_state(cache, k))
    rest = chunks[prefix:]
    feed = DSV41.job_feed(
        host,
        cache if prefix else None,
        np.concatenate([np.asarray(c[0]) for c in rest]),
        [c.shape[1] for c in rest],
        k,
    )
    for index, (chunk, nxt) in enumerate(zip(rest, nexts[prefix:])):
        inputs = feed(index) if feed else {}
        boundary, state = DSV41.run_first(helper, helper_cache, chunk, nxt, inputs)
        DSV41.run_rest(
            host._omlx_prefill,
            host,
            cache,
            chunk,
            k,
            boundary,
            state,
            next_prefill_tokens=nxt,
        )
        DSV41.apply_state(cache, k, state)
        mx.eval([c.state for c in cache])

    _assert_same_cache(reference, cache)
    last = ids[:, -1:]
    for a, b in zip(_finish(host, reference, last), _finish(host, cache, last)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_split_matches_whole_model(k):
    _run(tiny(), k, [7, 5, 3])


@pytest.mark.parametrize("k", [1, 2, 3])
def test_split_matches_whole_model_with_ced(k):
    # Chunks wider than the window skip the decoder tail, then replay it.
    _run(tiny_ced(), k, [9, 6, 7])


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_split_matches_whole_model_with_engram(k):
    _run(tiny(**_ENGRAM), k, [7, 5, 3])


def test_split_continues_a_cached_prefix():
    _run(tiny(**_ENGRAM), 2, [6, 5, 4], prefix=1)
    _run(tiny_ced(), 3, [9, 6, 7], prefix=1)


def test_split_limit_keeps_the_ced_midpoint_here():
    lm = LanguageModel(tiny_ced())
    assert DSV41.max_split(lm) == lm._config.n_layers // 2


# ---------------------------------------------------------------------------
# Helper process protocol over a loopback socket.
# ---------------------------------------------------------------------------

TOKEN = b"session-token"


def _serve(lm, k, *, once=True):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    server = HelperServer(lm, DSV41, k, TOKEN, {"protocol": wire.PROTOCOL, "layers": k})
    thread = threading.Thread(
        target=server.serve, args=(listener,), kwargs={"once": once}, daemon=True
    )
    thread.start()
    return listener, thread


def _helper_model(config, k):
    model = _model(config)
    DSV41.keep_first(model, k)
    mx.eval(model.parameters())
    return model


def test_helper_over_a_socket_matches_whole_model():
    config, k, sizes = tiny(**_ENGRAM), 2, [6, 5, 4]
    host = _model(config)
    ids = mx.array(np.random.default_rng(7).integers(3, 60, (1, sum(sizes) + 1)))
    chunks = _chunks(ids, sizes)
    nexts = [c.shape[1] for c in chunks[1:]] + [0]
    reference = host.make_cache()
    for chunk, nxt in zip(chunks, nexts):
        host._omlx_prefill(chunk, cache=reference, next_prefill_tokens=nxt)

    listener, thread = _serve(_helper_model(config, k), k)
    client = HelperClient("127.0.0.1", listener.getsockname()[1], TOKEN)
    tokens = np.asarray(ids[0, :-1])
    feed = DSV41.job_feed(host, None, tokens, sizes, k)
    job = client.submit(tokens, sizes, feed=feed)
    cache = host.make_cache()
    for chunk, nxt in zip(chunks, nexts):
        part = job.take(timeout=60)
        DSV41.run_rest(
            host._omlx_prefill,
            host,
            cache,
            chunk,
            k,
            part.boundary,
            part.state,
            next_prefill_tokens=nxt,
        )
        DSV41.apply_state(cache, k, part.state)
    assert job.finished
    client.close()
    thread.join(10)
    listener.close()
    _assert_same_cache(reference, cache)


def test_helper_rejects_a_wrong_token():
    listener, thread = _serve(_helper_model(tiny(), 2), 2, once=False)
    with pytest.raises((wire.ProtocolError, ConnectionError)):
        HelperClient("127.0.0.1", listener.getsockname()[1], b"wrong")
    listener.close()
    thread.join(10)


def test_helper_stays_a_window_ahead_and_runs_jobs_in_order():
    k = 2
    listener, thread = _serve(_helper_model(tiny(), k), k)
    client = HelperClient("127.0.0.1", listener.getsockname()[1], TOKEN)
    tokens = np.arange(3, 21, dtype=np.int32)
    first = client.submit(tokens, [3] * 6)
    second = client.submit(tokens[:6], [3, 3])
    # Nothing is acknowledged yet: the helper sends WINDOW chunks and waits.
    deadline = time.monotonic() + 30
    while first._frames.qsize() < WINDOW and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.3)
    assert first._frames.qsize() == WINDOW
    # The second job queues behind the first.
    with pytest.raises(HelperError):
        second.take(timeout=0.3)
    for _ in range(6):
        first.take(timeout=30)
    assert [second.take(timeout=30).index for _ in range(2)] == [0, 1]
    # A cancelled job frees the helper for the next one.
    third = client.submit(tokens, [3] * 6)
    third.take(timeout=30)
    third.cancel()
    assert client.submit(tokens[:3], [3]).take(timeout=30).size == 3
    client.close()
    thread.join(10)
    listener.close()


# ---------------------------------------------------------------------------
# Scheduler flow: helper-assisted chunks vs the same scheduler alone.
# ---------------------------------------------------------------------------


def _scheduler(model, step):
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            prefill_step_size=step, chunked_prefill=True, paged_cache_block_size=0
        ),
    )
    sched._qwen35_prefill_floor = 0
    sched.batch_generator = MagicMock()
    return sched


def _request(tokens, rid="req-1"):
    request = Request(
        request_id=rid,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = list(tokens)
    request.num_prompt_tokens = len(tokens)
    return request


def _session(host, lm, k, **kwargs):
    listener, thread = _serve(lm, k)
    client = HelperClient("127.0.0.1", listener.getsockname()[1], TOKEN)
    session = PrefillHelperSession(
        client,
        DSV41,
        host,
        k,
        "helper-mac",
        DSV41.layer_work(lm),
        fixed_layers=k,
        min_tokens=1,
        **kwargs,
    )
    return session, listener, thread


def _drive(sched, request, tokens):
    """Prefill one request through the scheduler; returns its final cache."""
    finished = {}

    def complete(self, request, state, scheduled, still):
        self._prefill_states.pop(request.request_id, None)
        if state.helper is not None:
            state.helper.session.finish(request.request_id, state.tokens_processed, 0.0)
        finished["state"] = state

    with patch.object(Scheduler, "_complete_chunked_prefill", complete):
        if sched._prefill_helper is None:
            state = sched._begin_prefill(request, tokens, None)
            sched._prefill_states[request.request_id] = state
            sched.prefilling.append(request)
        else:
            assert Scheduler._start_helper_prefill(
                sched, request, tokens, None, MagicMock(), None
            )
        deadline = time.monotonic() + 60
        while "state" not in finished and time.monotonic() < deadline:
            sched._advance_chunked_prefills([], [])
            if sched._prefill_helper is not None:
                sched._prefill_helper.sync(sched._prefill_states)
    return finished["state"]


@pytest.mark.parametrize(
    "config", [tiny(), tiny_ced(), tiny(**_ENGRAM)], ids=["plain", "ced", "engram"]
)
def test_scheduler_helper_prefill_matches_local(config):
    k, tokens = 2, [int(t) for t in np.random.default_rng(3).integers(3, 60, 23)]
    host = _model(config)
    local = _drive(_scheduler(host, 6), _request(tokens), tokens)

    sched = _scheduler(host, 6)
    session, listener, thread = _session(host, _helper_model(config, k), k)
    sched.attach_prefill_helper(session)
    state = _drive(sched, _request(tokens), tokens)
    assert state.helper is not None and state.helper.index == len(state.helper.grid)
    assert session.last_prompt["tokens"] == len(tokens) - 1
    _assert_same_cache(local.cache, state.cache)
    session.close()
    thread.join(10)
    listener.close()


@pytest.mark.parametrize("how", ["link", "off"])
def test_scheduler_finishes_locally_when_the_helper_leaves(how):
    config, k = tiny_ced(), 2
    tokens = [int(t) for t in np.random.default_rng(4).integers(3, 60, 23)]
    host = _model(config)
    local = _drive(_scheduler(host, 6), _request(tokens), tokens)

    sched = _scheduler(host, 6)
    session, listener, thread = _session(host, _helper_model(config, k), k)
    sched.attach_prefill_helper(session)
    original = Scheduler._forward_helper_chunk

    def forward_then_leave(self, state, chunk, prefill_model, chunk_kwargs):
        original(self, state, chunk, prefill_model, chunk_kwargs)
        if how == "link":
            session.client.close()
        else:
            sched.attach_prefill_helper(None)

    with patch.object(Scheduler, "_forward_helper_chunk", forward_then_leave):
        state = _drive(sched, _request(tokens), tokens)
    assert state.helper is None and session.last_error
    _assert_same_cache(local.cache, state.cache)
    session.close()
    thread.join(10)
    listener.close()


def test_scheduler_cancels_the_job_of_a_dropped_request():
    k, tokens = 2, list(range(3, 40))
    host = _model(tiny())
    sched = _scheduler(host, 4)
    session, listener, thread = _session(host, _helper_model(tiny(), k), k)
    sched.attach_prefill_helper(session)
    request = _request(tokens)
    assert Scheduler._start_helper_prefill(
        sched, request, tokens, None, MagicMock(), None
    )
    job = sched._prefill_states[request.request_id].helper.job
    sched._prefill_states.pop(request.request_id)
    sched._prefill_helper.sync(sched._prefill_states)
    assert job.finished and not session._jobs
    # The helper moves on to the next request.
    other = _request(tokens[:9], "req-2")
    assert Scheduler._start_helper_prefill(
        sched, other, tokens[:9], None, MagicMock(), None
    )
    assert sched._prefill_states["req-2"].helper.job.take(timeout=30).index == 0
    session.close()
    thread.join(10)
    listener.close()


def test_helper_grid_stops_at_blocks_and_tails():
    sched = _scheduler(MagicMock(), 8)
    state = SimpleNamespace(
        tokens_remaining=mx.zeros((1, 30), mx.int32),
        base_size=5,
        boundary_enabled=True,
        block_size=16,
        system_at=None,
        tail_at=29,
    )
    # Cached prefix of 5: the first block ends at 16, the tail at 29.
    assert Scheduler._helper_chunk_grid(sched, state) == [8, 3, 8, 5, 3, 3]


def test_auto_split_balances_measured_speed():
    work = [1.0] * 20 + [0.1] + [0.0] * 19  # DeepSeek V4.1 with CED
    client = SimpleNamespace(info={"warm_seconds": 18.0, "warm_tokens": 1}, alive=True)
    session = PrefillHelperSession(client, DSV41, None, 18, "helper-mac", work)
    # Before this Mac is timed, both Macs count as equally fast.
    assert session.pick_layers() == 10
    # This Mac takes twice as long per unit of work: the helper gets more.
    session.observe(10, 2048, helper_s=10 * 2048, host_s=10.1 * 2 * 2048)
    assert session.pick_layers() == 14
    session.fixed_layers = 5
    assert session.pick_layers() == 5


def test_job_uses_fewer_layers_than_the_helper_holds():
    config, held, k, sizes = tiny(**_ENGRAM), 3, 2, [6, 5, 4]
    host = _model(config)
    ids = mx.array(np.random.default_rng(9).integers(3, 60, (1, sum(sizes) + 1)))
    chunks = _chunks(ids, sizes)
    reference = host.make_cache()
    for chunk in chunks:
        host._omlx_prefill(chunk, cache=reference)

    listener, thread = _serve(_helper_model(config, held), held)
    client = HelperClient("127.0.0.1", listener.getsockname()[1], TOKEN)
    tokens = np.asarray(ids[0, :-1])
    feed = DSV41.job_feed(host, None, tokens, sizes, k)
    job = client.submit(tokens, sizes, layers=k, feed=feed)
    cache = host.make_cache()
    for chunk in chunks:
        part = job.take(timeout=60)
        assert len({name.split(".")[0] for name in part.state}) == k
        DSV41.run_rest(
            host._omlx_prefill, host, cache, chunk, k, part.boundary, part.state
        )
        DSV41.apply_state(cache, k, part.state)
    client.close()
    thread.join(10)
    listener.close()
    _assert_same_cache(reference, cache)


# ---------------------------------------------------------------------------
# Launcher: the file probe runs on the helper with stdlib Python only.
# ---------------------------------------------------------------------------


def _probe(home, files):
    target = HelperTarget("node", "M5max", "me@10.0.0.2", "10.0.0.2", "thunderbolt")

    def runner(argv, **kwargs):
        # Run the remote command here, with HOME pointing at the fake Mac.
        command = argv[-1]
        return subprocess.run(
            ["/bin/sh", "-c", command],
            env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
            **kwargs,
        )

    return launcher.locate_files(target, "Model-oQ3e", files, runner=runner)


def test_probe_finds_an_existing_copy_or_lists_missing_files(tmp_path):
    files = {"config.json": 3, "model-00001.safetensors": 5}
    models = tmp_path / "Models" / "org" / "Model-oQ3e"
    models.mkdir(parents=True)
    (models / "config.json").write_bytes(b"abc")
    (models / "model-00001.safetensors").write_bytes(b"12345")
    (tmp_path / ".omlx").mkdir()
    (tmp_path / ".omlx" / "settings.json").write_text(
        json.dumps({"model": {"model_dirs": [str(tmp_path / "Models")]}})
    )
    found = _probe(tmp_path, files)
    assert found["found"] == str(models)

    # A size mismatch is not a copy; the copy folder then lacks everything.
    (models / "model-00001.safetensors").write_bytes(b"1234")
    result = _probe(tmp_path, files)
    assert result["found"] is None
    assert result["copy"] == str(tmp_path / ".omlx/prefill-helper/models/Model-oQ3e")
    assert sorted(result["missing"]) == sorted(files)
    assert result["free"] > 0


def test_helper_target_rejects_a_shell_unsafe_interpreter():
    with pytest.raises(ValueError):
        HelperTarget("n", "x", "u@1.2.3.4", "1.2.3.4", "thunderbolt", "py; rm -rf ~")


def test_helper_files_leave_out_engram_tables_and_later_layers(tmp_path):
    shards = {
        "a.safetensors": ["language_model.embed.weight", "language_model.layers.0.w"],
        "b.safetensors": [
            "language_model.layers.1.engram.embed.weight",
            "language_model.layers.1.engram.embed.scales",
        ],
        "c.safetensors": ["language_model.layers.1.ffn.w"],
        "d.safetensors": ["language_model.layers.3.w", "language_model.mtp.0.w"],
    }
    weight_map = {}
    for name, tensors in shards.items():
        mx.save_safetensors(str(tmp_path / name), {t: mx.zeros((2,)) for t in tensors})
        weight_map.update(dict.fromkeys(tensors, name))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (tmp_path / "config.json").write_text("{}")
    files = launcher.required_files(tmp_path, 2, DSV41)
    assert sorted(files) == [
        "a.safetensors",
        "c.safetensors",
        "config.json",
        "model.safetensors.index.json",
    ]


# ---------------------------------------------------------------------------
# Manager: the steps from a chosen Mac to an attached session.
# ---------------------------------------------------------------------------


class _PlanAdapter(type(DSV41)):
    """DeepSeek V4.1 adapter with a fixed memory plan (no checkpoint on disk)."""

    def memory_plan(self, model_path):
        return 1 << 30, [10 << 30] * 5


class _FakeLauncher:
    LaunchError = launcher.LaunchError

    def __init__(self, lm, k=4, *, version=None, budget=60 << 30, missing=()):
        self.lm, self.k = lm, k
        self.version = version or launcher_version()
        self.budget, self.missing = budget, list(missing)
        self.copied, self.servers, self.paired = [], [], True

    def resolve_target(self, node_id):
        if not self.paired:
            raise launcher.LaunchError("This Mac is not paired.")
        return HelperTarget(
            node_id, "M5max", "me@127.0.0.1", "127.0.0.1", "thunderbolt"
        )

    def remote_info(self, target):
        return {
            "version": self.version,
            "budget": self.budget,
            "device": "Apple M5 Max",
        }

    def required_files(self, model_path, layers, adapter):
        return {"config.json": 10, "model-1.safetensors": 1 << 20}

    def locate_files(self, target, name, files):
        return {
            "found": None,
            "copy": "/helper/copy",
            "missing": self.missing,
            "free": 1 << 40,
        }

    def copy_files(self, target, model_path, dest, names, progress):
        for name in names:
            self.copied.append(name)
            progress(name)
        self.missing = []

    def launch(self, target, model_dir, model_type, layers, *, host_name, warm_tokens):
        listener, thread = _serve(_helper_model(tiny(), layers), layers)
        server = SimpleNamespace(listener=listener, thread=thread, layers=layers)
        self.servers.append(server)
        return SimpleNamespace(
            port=listener.getsockname()[1], token=TOKEN, hello={}, stop=listener.close
        )


def launcher_version():
    from omlx import __version__

    return __version__


def _manager(fake, attached, speeds_path):
    return PrefillHelperManager(
        model_path="/models/x",
        model_type="deepseek_v41",
        adapter=_PlanAdapter(),
        language_model=_model(tiny()),
        attach=attached.append,
        chunk_tokens=8,
        launcher=fake,
        speeds_path=speeds_path,
    )


def _wait_state(manager, *states, reason="", timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status()
        if status["state"] in states and reason in status.get("reason", ""):
            return status
        time.sleep(0.02)
    raise AssertionError(f"state stayed {manager.status()}")


def test_manager_waits_for_consent_before_copying_then_attaches(tmp_path):
    attached = []
    fake = _FakeLauncher(None, missing=["model-1.safetensors"])
    manager = _manager(fake, attached, tmp_path / "speeds.json")
    manager.configure("node-1", None)
    status = _wait_state(manager, "needs_files")
    assert status["bytes"] == 1 << 20 and fake.copied == []
    manager.copy_files()
    status = _wait_state(manager, "ready")
    # 60 GiB budget, 1 GiB shared and 8 GiB reserve hold five 10 GiB layers,
    # but DeepSeek V4.1 keeps the CED midpoint (layer 2 of 5) here.
    assert fake.copied == ["model-1.safetensors"] and status["held"] == 2
    assert isinstance(attached[-1], PrefillHelperSession)
    manager.configure(None, None)
    _wait_state(manager, "off")
    assert attached[-1] is None
    manager.close()


def test_manager_blocks_what_a_retry_cannot_fix(tmp_path):
    attached = []
    fake = _FakeLauncher(None, version="0.0.1")
    manager = _manager(fake, attached, tmp_path / "speeds.json")
    manager.configure("node-1", None)
    _wait_state(manager, "error", reason="update it to")
    fake.version, fake.budget = launcher_version(), 12 << 30
    manager.retry()
    _wait_state(manager, "error", reason="Not even one layer")
    fake.budget = 60 << 30
    manager.configure("node-1", 3)
    _wait_state(manager, "error", reason="at most 2")
    fake.paired = False
    manager.configure("node-2", None)
    _wait_state(manager, "error", reason="not paired")
    assert attached == []
    manager.close()


def test_manager_retries_a_lost_helper_but_not_a_stopped_one(tmp_path, monkeypatch):
    monkeypatch.setattr(manager_module, "_RETRY_DELAYS_S", (0.05,))
    monkeypatch.setattr(manager_module, "_WATCH_S", 0.05)
    attached = []
    fake = _FakeLauncher(None)
    manager = _manager(fake, attached, tmp_path / "speeds.json")
    manager.configure("node-1", None)
    _wait_state(manager, "ready")
    first = attached[-1]
    first.client.close()
    _wait_state(manager, "offline", "connecting", "starting")
    _wait_state(manager, "ready")
    assert attached[-1] is not first and len(fake.servers) == 2
    # Stopped on the helper Mac: stays off until the user retries.
    attached[-1].client.stopped_by_helper = True
    attached[-1].client.close()
    assert _wait_state(manager, "stopped")["reason"] == "Stopped on M5max"
    time.sleep(0.3)
    assert manager.status()["state"] == "stopped" and len(fake.servers) == 2
    manager.close()


def test_manager_starts_a_new_helper_from_the_saved_speeds(tmp_path):
    attached, speeds = [], tmp_path / "speeds.json"
    manager = _manager(_FakeLauncher(None), attached, speeds)
    manager.configure("node-1", None)
    _wait_state(manager, "ready")
    attached[-1].observe(2, 2048, helper_s=1.0, host_s=3.0)
    measured = (attached[-1].unit_helper, attached[-1].unit_host)
    manager.close()

    again = _manager(_FakeLauncher(None), attached, speeds)
    again.configure("node-1", None)
    _wait_state(again, "ready")
    assert (attached[-1].unit_helper, attached[-1].unit_host) == measured
    again.close()


# ---------------------------------------------------------------------------
# Admin routes.
# ---------------------------------------------------------------------------


def test_routes_report_status_forward_actions_and_stop_lending(tmp_path, monkeypatch):
    calls = []
    manager = SimpleNamespace(
        status=lambda: {"state": "needs_files"},
        copy_files=lambda: calls.append("copy"),
        retry=lambda: calls.append("retry"),
        stop=lambda: calls.append("stop"),
    )
    entries = {
        "loaded": SimpleNamespace(engine=SimpleNamespace(prefill_helper=manager)),
        "idle": SimpleNamespace(engine=None),
    }
    helper_routes.set_engine_pool_getter(lambda: SimpleNamespace(get_entry=entries.get))
    app = FastAPI()
    app.include_router(helper_routes.router)
    client = TestClient(app)
    base = "/admin/api/prefill-helper"
    assert client.get(f"{base}/models/idle").json() == {"state": "unloaded"}
    assert client.get(f"{base}/models/loaded").json()["state"] == "needs_files"
    assert client.post(f"{base}/models/loaded/copy").status_code == 200
    assert client.post(f"{base}/models/idle/copy").status_code == 409
    assert client.post(f"{base}/models/loaded/reboot").status_code == 404
    assert calls == ["copy"]

    marker = tmp_path / "active.json"
    monkeypatch.setattr(helper_routes, "active_marker_path", lambda: marker)
    assert client.get(f"{base}/lending").json() == {"active": False}
    marker.write_text(json.dumps({"pid": os.getpid(), "host": "Mac Studio"}))
    assert client.get(f"{base}/lending").json()["host"] == "Mac Studio"
    killed = []
    monkeypatch.setattr(helper_routes.os, "kill", lambda pid, sig: killed.append(sig))
    assert client.post(f"{base}/lending/stop").json()["stopped"]
    assert killed[-1] == signal.SIGTERM


def test_accelerator_check_reports_each_row(monkeypatch):
    monkeypatch.setattr(
        helper_routes,
        "get_device_registry",
        lambda: SimpleNamespace(get=lambda node: {"friendly_name": "M5max"}),
    )
    target = HelperTarget(
        "n", "M5max", "u@10.0.0.2", "10.0.0.2", "ethernet", warnings=("Use TB.",)
    )
    monkeypatch.setattr(helper_routes.launcher, "resolve_target", lambda node: target)
    monkeypatch.setattr(
        helper_routes.launcher,
        "remote_info",
        lambda t: {
            "version": "0.0.1",
            "budget": 100 << 30,
            "memory": 128 << 30,
            "on_battery": True,
            "disk_free": 1 << 40,
        },
    )
    rows = {row["key"]: row for row in helper_routes._check("n")["rows"]}
    assert [rows[k]["status"] for k in ("ssh", "link", "version", "power")] == [
        "pass",
        "warn",
        "fail",
        "warn",
    ]
    assert rows["link"]["fix"] == "Use TB." and rows["memory"]["bytes"] == 100 << 30
