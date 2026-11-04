"""Prefill Accelerator: a split prefill must match the whole-model prefill."""

import itertools
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from test_deepseek_v41 import _ENGRAM, load_reference_weights, tiny, tiny_ced
from test_mlx_vlm_minimax_m3_compat import _tiny_text_config
from test_scheduler_chunked_prefill import (
    _tiny_glm5_next,
    _tiny_qwen4_exp,
    _tiny_vlm_adapter,
)

from omlx.cluster.transport import HostInterfaces, InterfaceAddress
from omlx.patches.deepseek_v41.language import LanguageModel
from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding
from omlx.patches.mlx_vlm_minimax_m3_compat import (
    apply_mlx_vlm_minimax_m3_compat_patch,
)
from omlx.prefill_helper import launcher, lending, wire
from omlx.prefill_helper import manager as manager_module
from omlx.prefill_helper import routes as helper_routes
from omlx.prefill_helper import worker as worker_module
from omlx.prefill_helper.adapters import adapter_for
from omlx.prefill_helper.client import HelperClient, HelperError
from omlx.prefill_helper.launcher import HelperTarget
from omlx.prefill_helper.manager import PrefillHelperManager
from omlx.prefill_helper.session import PrefillHelperSession
from omlx.prefill_helper.worker import WINDOW, HelperServer
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig

DSV41 = adapter_for("deepseek_v41")
GLM = adapter_for("glm5_next")
QWEN4 = adapter_for("qwen4_exp")
MINIMAX = adapter_for("minimax_m3_vl")
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


@pytest.mark.parametrize("k", [1, 4])
def test_split_matches_whole_model(k):
    _run(tiny(), k, [7, 5, 3])


@pytest.mark.parametrize("k", [1, 3])
def test_split_matches_whole_model_with_ced(k):
    # Chunks wider than the window skip the decoder tail, then replay it.
    _run(tiny_ced(), k, [9, 6, 7])


@pytest.mark.parametrize("k", [1, 3])
def test_split_matches_whole_model_with_engram(k):
    _run(tiny(**_ENGRAM), k, [7, 5, 3])


def test_split_continues_a_cached_prefix():
    _run(tiny(**_ENGRAM), 2, [6, 5, 4], prefix=1)
    _run(tiny_ced(), 3, [9, 6, 7], prefix=1)


# ---------------------------------------------------------------------------
# Helper process protocol over a loopback socket.
# ---------------------------------------------------------------------------

TOKEN = b"session-token"


def _serve(lm, k, *, once=True, adapter=DSV41):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    server = HelperServer(
        lm, adapter, k, TOKEN, {"protocol": wire.PROTOCOL, "layers": k}
    )
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


def test_client_notices_a_helper_that_stops_sending():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def silent():
        sock, _ = listener.accept()
        wire.authenticate(sock, TOKEN, server=True)
        wire.send(sock, {"op": "hello", "protocol": wire.PROTOCOL})
        time.sleep(2)
        sock.close()

    thread = threading.Thread(target=silent, daemon=True)
    thread.start()
    client = HelperClient("127.0.0.1", listener.getsockname()[1], TOKEN)
    assert not client.stalled(0.05)
    client.submit(np.arange(3, 9, dtype=np.int32), [6])
    time.sleep(0.2)
    assert client.stalled(0.1)
    client.close()
    thread.join(5)
    listener.close()


def test_a_retried_request_cancels_its_old_job():
    host, k = _model(tiny()), 2
    session, listener, thread = _session(host, _helper_model(tiny(), k), k)
    tokens = np.arange(3, 21, dtype=np.int32)
    first = session.start("req-1", tokens, [6, 6, 6], None)
    second = session.start("req-1", tokens, [6, 6, 6], None)
    assert first.job.finished and not second.job.finished
    session.close()
    thread.join(10)
    listener.close()


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


def _session(host, lm, k, *, adapter=DSV41, **kwargs):
    listener, thread = _serve(lm, k, adapter=adapter)
    client = HelperClient("127.0.0.1", listener.getsockname()[1], TOKEN)
    session = PrefillHelperSession(
        client,
        adapter,
        host,
        k,
        "helper-mac",
        adapter.layer_work(lm),
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


_REMOTE = HelperTarget("node", "M5max", "me@10.0.0.2", "10.0.0.2", "thunderbolt")


def _remote(home):
    def runner(argv, **kwargs):
        # Run the remote command here, with HOME pointing at the fake Mac.
        command = argv[-1]
        return subprocess.run(
            ["/bin/sh", "-c", command],
            env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
            **kwargs,
        )

    return runner


def _probe(home, files):
    return launcher.locate_files(_REMOTE, "Model-oQ3e", files, runner=_remote(home))


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


def test_other_copies_are_listed_and_deleted_but_never_the_running_one(tmp_path):
    copies = tmp_path / ".omlx" / "prefill-helper" / "models"
    for name in ("Model-oQ3e", "Old", "Running"):
        (copies / name).mkdir(parents=True)
        (copies / name / "config.json").write_bytes(b"1234")
    own = tmp_path / "Models" / "Own"
    own.mkdir(parents=True)
    (copies / "Link").symlink_to(own)
    (tmp_path / ".omlx" / "prefill-helper" / "active.json").write_text(
        json.dumps({"model": str(copies / "Running")})
    )
    result = _probe(tmp_path, {"config.json": 3})
    assert [(c["name"], c["bytes"]) for c in result["copies"]] == [("Old", 4)]
    assert (copies / "Model-oQ3e" / ".last_used").exists()

    names = ["Running", "Link", "../Models", "Old"]
    assert launcher.delete_copies(_REMOTE, names, runner=_remote(tmp_path)) > 0
    assert sorted(p.name for p in copies.iterdir()) == ["Link", "Model-oQ3e", "Running"]
    assert own.is_dir()


def test_copy_plan_keeps_50_gib_free_or_asks():
    gib = 1 << 30
    old = {"name": "Old", "bytes": 20 * gib, "used": 1.0}
    older = {"name": "Older", "bytes": 20 * gib, "used": 0.0}
    plan = manager_module.copy_plan
    assert plan(100 * gib, 200 * gib, [old])["kind"] == "copy"
    # The least recently used copies go first, only as many as needed.
    room = plan(100 * gib, 120 * gib, [old, older, {**old, "name": "New", "used": 9}])
    assert room["kind"] == "make_room" and room["left"] == 60 * gib
    assert [c["name"] for c in room["delete"]] == ["Older", "Old"]
    assert plan(100 * gib, 120 * gib, [])["kind"] == "low_space"
    assert plan(100 * gib, 90 * gib, [old])["kind"] == "no_space"


def test_arrays_decoded_on_one_thread_run_on_another():
    # The helper decodes a job's prefix on its reader thread and runs the job
    # on another; a pending bfloat16 view there raised "no Stream(gpu, N)".
    data, dtype = wire.to_host(mx.ones((2, 3), dtype=mx.bfloat16))
    decoded = {}
    reader = threading.Thread(
        target=lambda: decoded.update(a=wire.from_host(data, dtype))
    )
    reader.start()
    reader.join()
    assert decoded["a"].dtype == mx.bfloat16
    assert mx.sum(decoded["a"]).item() == 6


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
    # Hugging Face layout names count as language-model tensors too.
    assert QWEN4.resident("model.language_model.layers.0.mlp.gate.weight")
    assert not QWEN4.resident(
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight"
    )
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

    def __init__(
        self,
        lm,
        k=4,
        *,
        version=None,
        budget=60 << 30,
        missing=(),
        free=1 << 40,
        copies=(),
        hold=None,
    ):
        self.lm, self.k = lm, k
        self.version = version or launcher_version()
        self.budget, self.missing = budget, list(missing)
        self.free, self.copies, self.hold = free, list(copies), hold
        self.copied, self.deleted, self.servers, self.paired = [], [], [], True

    def resolve_target(self, node_id):
        if not self.paired:
            raise launcher.LaunchError("This Mac is not paired.")
        return HelperTarget(
            node_id, "M5max", "me@127.0.0.1", "127.0.0.1", "thunderbolt"
        )

    def remote_info(self, target, host=None):
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
            "copies": self.copies,
            "free": self.free,
        }

    def delete_copies(self, target, names):
        self.deleted.append(names)

    def copy_files(self, target, model_path, dest, names, progress, cancelled=None):
        for name in names:
            if self.hold is not None:
                self.hold.wait(10)
            self.copied.append(name)
            progress(name)
        self.missing = []

    def launch(
        self,
        target,
        model_dir,
        model_type,
        layers,
        *,
        host_name,
        warm_tokens,
        a8_min_tokens=None,
    ):
        listener, thread = _serve(_helper_model(tiny(), layers), layers)
        server = SimpleNamespace(
            listener=listener, thread=thread, layers=layers, a8=a8_min_tokens
        )
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


def test_manager_asks_before_deleting_other_copies_then_attaches(tmp_path):
    attached = []
    old = {"name": "Old", "bytes": 20 << 30, "used": 0.0}
    fake = _FakeLauncher(
        None, missing=["model-1.safetensors"], free=40 << 30, copies=[old]
    )
    manager = _manager(fake, attached, tmp_path / "speeds.json")
    manager.configure("node-1", None)
    status = _wait_state(manager, "needs_files")
    assert status["prompt"] == "make_room"
    assert status["delete"] == [{"name": "Old", "bytes": 20 << 30}]
    assert fake.copied == [] and fake.deleted == []
    manager.copy_files()
    status = _wait_state(manager, "ready")
    # 60 GiB budget, 1 GiB shared and 8 GiB reserve hold five 10 GiB layers,
    # but DeepSeek V4.1 keeps the CED midpoint (layer 2 of 5) here.
    assert fake.deleted == [["Old"]] and fake.copied == ["model-1.safetensors"]
    assert status["held"] == 2
    assert isinstance(attached[-1], PrefillHelperSession)
    manager.configure(None, None)
    _wait_state(manager, "off")
    assert attached[-1] is None
    manager.close()


def test_files_copied_before_the_load_are_taken_over_not_copied_twice(tmp_path):
    release = threading.Event()
    fake = _FakeLauncher(None, missing=["model-1.safetensors"], hold=release)
    manager_module.prepare("/models/x", _PlanAdapter(), "node-1", None, launcher=fake)
    prep = manager_module.file_prep("/models/x")
    _wait_state(prep, "copying")
    # Saving the same choice again keeps that copy going.
    manager_module.prepare("/models/x", _PlanAdapter(), "node-1", None, launcher=fake)
    assert manager_module.file_prep("/models/x") is prep

    attached = []
    manager = _manager(fake, attached, tmp_path / "speeds.json")
    manager.configure("node-1", None)
    _wait_state(manager, "copying")
    release.set()
    _wait_state(manager, "ready")
    assert fake.copied == ["model-1.safetensors"]
    assert manager_module.file_prep("/models/x") is None
    manager.close()


def test_copy_time_left_follows_the_average_speed(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(manager_module.time, "time", lambda: now[0])
    fake = _FakeLauncher(None, missing=["a", "b"])
    fake.required_files = lambda path, layers, adapter: {"a": 5, "b": 5}

    def copy_files(target, model_path, dest, names, progress, cancelled=None):
        for name in names:
            now[0] += 10
            progress(name)

    fake.copy_files = copy_files
    seen = []

    def report(state, **fields):
        seen.append(manager_module._with_time_left({"state": state, **fields}))

    target = SimpleNamespace(name="M5max")
    manager_module._prepare_files(fake, target, "/models/x", 2, None, report, None)
    # Half the bytes in 10 s leaves 10 s.
    assert [status.get("seconds_left") for status in seen] == [None, 10, 0]


def test_a_helper_started_during_the_load_is_taken_over_not_started_twice(
    tmp_path,
):
    fake = _FakeLauncher(None)
    launch = {"model_type": "deepseek_v41", "warm_tokens": 8, "a8_min_tokens": None}
    adapter = _PlanAdapter()
    manager_module.prepare(
        "/models/z", adapter, "node-1", None, launch=launch, launcher=fake
    )
    prep = manager_module.file_prep("/models/z")
    prep.join(30)
    assert prep.launched is not None and len(fake.servers) == 1

    attached = []
    manager = PrefillHelperManager(
        model_path="/models/z",
        model_type="deepseek_v41",
        adapter=adapter,
        language_model=_model(tiny()),
        attach=attached.append,
        chunk_tokens=8,
        launcher=fake,
        speeds_path=tmp_path / "speeds.json",
    )
    manager.configure("node-1", None)
    _wait_state(manager, "ready")
    assert len(fake.servers) == 1
    manager.close()

    # A load that fails stops the helper it started.
    manager_module.prepare(
        "/models/z", adapter, "node-1", None, launch=launch, launcher=fake
    )
    prep = manager_module.file_prep("/models/z")
    prep.join(30)
    stopped = []
    prep.launched[3].stop = lambda: stopped.append(True)
    manager_module.prepare("/models/z", None, None, None)
    assert stopped == [True] and manager_module.file_prep("/models/z") is None


def test_a_check_reports_the_files_and_copies_nothing():
    fake = _FakeLauncher(None, missing=["model-1.safetensors"])
    adapter = _PlanAdapter()
    manager_module.prepare(
        "/models/c", adapter, "node-1", None, check_only=True, launcher=fake
    )
    prep = manager_module.file_prep("/models/c")
    status = _wait_state(prep, "files_missing")
    assert status["bytes"] == 1 << 20 and fake.copied == []
    # Saving the choice again copies instead of checking.
    manager_module.prepare("/models/c", adapter, "node-1", None, launcher=fake)
    _wait_state(manager_module.file_prep("/models/c"), "files_ready")
    assert fake.copied == ["model-1.safetensors"]
    # A later check keeps that result rather than replacing it.
    copy = manager_module.file_prep("/models/c")
    manager_module.prepare(
        "/models/c", adapter, "node-1", None, check_only=True, launcher=fake
    )
    assert manager_module.file_prep("/models/c") is copy
    manager_module.prepare("/models/c", None, None, None)


def test_cancel_before_the_load_stops_a_waiting_copy_without_deleting():
    old = {"name": "Old", "bytes": 20 << 30, "used": 0.0}
    fake = _FakeLauncher(
        None, missing=["model-1.safetensors"], free=40 << 30, copies=[old]
    )
    manager_module.prepare("/models/y", _PlanAdapter(), "node-1", None, launcher=fake)
    prep = manager_module.file_prep("/models/y")
    assert _wait_state(prep, "needs_files")["prompt"] == "make_room"
    manager_module.prepare("/models/y", _PlanAdapter(), None, None, launcher=fake)
    _wait_state(prep, "stopped")
    assert fake.deleted == [] and fake.copied == []
    assert manager_module.file_prep("/models/y") is None


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
    _wait_state(manager, "error", reason="At most 2 layers")
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
    # Saving other settings does not start it either.
    manager.configure("node-1", None)
    time.sleep(0.3)
    assert manager.status()["state"] == "stopped" and len(fake.servers) == 2
    # Replaced by another model's helper: stays off the same way.
    manager.retry()
    _wait_state(manager, "ready")
    attached[-1].client.stopped_by_helper = True
    attached[-1].client.replaced_on_helper = True
    attached[-1].client.close()
    status = _wait_state(manager, "stopped", reason="now helping another model")
    assert status["reason"] == "M5max is now helping another model"
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
    # Switching Macs keeps the measured speeds under the Mac that measured them.
    again.configure("node-2", None)
    _wait_state(again, "ready")
    saved = json.loads(speeds.read_text())
    assert [key.split(":")[0] for key in saved] == ["node-1"]
    again.close()

    # The helper follows the model's A8 setting, and A8 keeps its own speeds.
    fake = _FakeLauncher(None)
    a8 = _manager(fake, attached, speeds)
    a8.configure("node-1", None, 256)
    _wait_state(a8, "ready")
    assert fake.servers[-1].a8 == 256
    assert (attached[-1].unit_helper, attached[-1].unit_host) != measured
    a8.close()


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
        "idle": SimpleNamespace(engine=None, model_path="/models/idle"),
    }
    saved = SimpleNamespace(prefill_helper_node_id=None)
    helper_routes.set_engine_pool_getter(
        lambda: SimpleNamespace(
            get_entry=entries.get,
            _settings_manager=SimpleNamespace(get_settings=lambda model_id: saved),
        )
    )
    app = FastAPI()
    app.include_router(helper_routes.router)
    client = TestClient(app)
    base = "/admin/api/prefill-helper"
    assert client.get(f"{base}/models/idle").json() == {"state": "unloaded"}
    assert client.get(f"{base}/models/loaded").json()["state"] == "needs_files"
    assert client.post(f"{base}/models/loaded/copy").status_code == 200
    assert client.post(f"{base}/models/idle/copy").status_code == 409
    # Without a saved helper there is nothing to check.
    assert client.post(f"{base}/models/idle/check").json() == {"state": "unloaded"}
    assert client.post(f"{base}/models/loaded/reboot").status_code == 404
    assert calls == ["copy"]

    marker = tmp_path / "active.json"
    monkeypatch.setattr(lending, "active_marker_path", lambda: marker)
    monkeypatch.setattr(lending, "_is_helper", lambda pid: pid == os.getpid())
    monkeypatch.setattr(lending, "host_marker_path", lambda: tmp_path / "host.json")
    paired = {}
    registry = SimpleNamespace(get=paired.get, is_paired=paired.__contains__)
    monkeypatch.setattr(helper_routes, "get_device_registry", lambda: registry)
    assert client.get(f"{base}/lending").json() == {"active": False, "helps": None}
    lending.record_host("studio-node", "Studio")
    # Only a host still paired here uses this Mac.
    assert client.get(f"{base}/lending").json()["helps"] is None
    paired["studio-node"] = {"friendly_name": "Mac Studio"}
    helps = client.get(f"{base}/lending").json()["helps"]
    assert helps == {"node_id": "studio-node", "name": "Mac Studio"}
    assert lending.lending_note() == ""
    marker.write_text(
        json.dumps({"pid": os.getpid(), "host": "Mac Studio", "active_gib": 82.4})
    )
    assert client.get(f"{base}/lending").json()["host"] == "Mac Studio"
    assert "lending 82 GB to Mac Studio" in lending.lending_note()
    killed = []
    monkeypatch.setattr(helper_routes.os, "kill", lambda pid, sig: killed.append(sig))
    assert client.post(f"{base}/lending/stop").json()["stopped"]
    assert killed[-1] == signal.SIGTERM


def test_a_mac_helping_another_model_counts_that_memory_as_free(monkeypatch):
    info = {
        "budget": 100 << 30,
        "available": 20 << 30,
        "lending": {"host": "Studio", "model": "/m/Other", "active_gib": 60.0},
    }
    assert manager_module.helper_budget(info) == 80 << 30
    target = HelperTarget("n", "M5max", "u@10.0.0.2", "10.0.0.2", "thunderbolt")
    monkeypatch.setattr(helper_routes.launcher, "resolve_target", lambda node: target)
    monkeypatch.setattr(helper_routes.launcher, "remote_info", lambda t: info)
    monkeypatch.setattr(helper_routes, "computer_name", lambda: "Studio")
    device = {"node_id": "n", "friendly_name": "M5max"}
    row = helper_routes._candidate(device, "/models/Mine", _PlanAdapter())
    assert row["lending"] == {"host": "Studio", "model": "Other"}
    # Helping this same model is not a takeover.
    row = helper_routes._candidate(device, "/models/Other", _PlanAdapter())
    assert row["lending"] is None


def test_a_new_helper_stops_the_one_running_on_its_mac(tmp_path, monkeypatch):
    running = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"pid": running.pid, "host": "Studio"}))
    monkeypatch.setattr(lending, "active_marker_path", lambda: marker)
    monkeypatch.setattr(lending, "_is_helper", lambda pid: pid == running.pid)
    # Reaped like an SSH session reaps the real helper.
    reaper = threading.Thread(target=running.wait)
    reaper.start()
    began = time.monotonic()
    worker_module._replace_running_helper(timeout=10)
    reaper.join(5)
    assert running.returncode == -signal.SIGUSR1 and time.monotonic() - began < 5


def test_resolve_target_uses_the_thunderbolt_link_local_without_ipv4(monkeypatch):
    lan = InterfaceAddress
    local = HostInterfaces(
        "localhost",
        (lan("en0", "192.168.1.30", 24),),
        thunderbolt_interfaces=frozenset({"bridge0", "en2"}),
        link_local_addresses={"bridge0": "fe80::1", "en2": "fe80::3"},
        media_speeds={"en0": 10000},
    )
    peer = HostInterfaces(
        "me@192.168.1.31",
        (lan("en0", "192.168.1.31", 24),),
        thunderbolt_interfaces=frozenset({"bridge0"}),
        link_local_addresses={"bridge0": "fe80::2"},
        media_speeds={"en0": 2500},
    )
    device = {
        "friendly_name": "M5max",
        "ssh_user": "me",
        "last_addrs": ["192.168.1.31"],
    }
    registry = SimpleNamespace(get=lambda n: device, is_paired=lambda n: True)
    monkeypatch.setattr(launcher, "get_device_registry", lambda: registry)
    monkeypatch.setattr(
        launcher,
        "probe_host_interfaces",
        lambda host: local if host == "localhost" else peer,
    )
    answered = SimpleNamespace(returncode=0)

    def connect(address, timeout):
        if address != ("fe80::2%en2", 22):
            raise OSError("no route")
        return MagicMock()

    real = launcher.thunderbolt_link_local
    monkeypatch.setattr(
        launcher,
        "thunderbolt_link_local",
        lambda local, peer: real(local, peer, connect=connect),
    )
    target = launcher.resolve_target("n", runner=lambda *a, **k: answered)
    # Each Mac scopes the address to its own end of the cable.
    assert (target.address, target.bind_address, target.link) == (
        "fe80::2%en2",
        "fe80::2%bridge0",
        "thunderbolt",
    )

    # Without the cable the LAN is used, judged by its slower end.
    peer = replace(peer, link_local_addresses={})
    target = launcher.resolve_target("n", runner=lambda *a, **k: answered)
    assert (target.address, target.bind_address, target.link) == (
        "192.168.1.31",
        "",
        "ethernet",
    )
    assert target.speed_mbps == 2500


def test_accelerator_check_reports_each_row(tmp_path, monkeypatch):
    paired = {"n": {"friendly_name": "M5max"}}
    registry = SimpleNamespace(get=paired.get, is_paired=paired.__contains__)
    monkeypatch.setattr(helper_routes, "get_device_registry", lambda: registry)
    monkeypatch.setattr(lending, "host_marker_path", lambda: tmp_path / "host.json")
    # Only discovered, not paired: no checks.
    with pytest.raises(HTTPException):
        helper_routes._check("other")
    lending.record_host("n", "M5max")
    target = HelperTarget(
        "n", "M5max", "u@10.0.0.2", "10.0.0.2", "ethernet", warnings=("Use TB.",)
    )
    monkeypatch.setattr(helper_routes.launcher, "resolve_target", lambda node: target)
    hosts = []

    def remote_info(target, host=None):
        hosts.append(host)
        return {
            "version": "0.0.1",
            "budget": 100 << 30,
            "memory": 128 << 30,
            "on_battery": True,
            "disk_free": 1 << 40,
        }

    me = {"node_id": "studio", "name": "Studio"}
    monkeypatch.setattr(helper_routes.launcher, "remote_info", remote_info)
    monkeypatch.setattr(helper_routes, "this_host", lambda: me)
    rows = {row["key"]: row for row in helper_routes._check("n")["rows"]}
    # The checks note on the helper Mac which Mac uses it, and this Mac stops
    # counting itself as that Mac's helper.
    assert hosts == [me] and lending.recorded_host() is None
    assert [rows[k]["status"] for k in ("ssh", "link", "version", "power")] == [
        "pass",
        "warn",
        "fail",
        "warn",
    ]
    assert rows["link"]["fix"] == "Use TB." and rows["memory"]["bytes"] == 100 << 30
    target = replace(target, speed_mbps=5000)
    rows = {row["key"]: row for row in helper_routes._check("n")["rows"]}
    assert rows["link"]["status"] == "pass"


# ---------------------------------------------------------------------------
# GLM-5.3: KDA and sparse-attention layers on either side of the cut.
# ---------------------------------------------------------------------------


def _glm():
    mx.random.seed(7)
    lm, _ = _tiny_glm5_next()
    mx.eval(lm.parameters())
    return lm


def _glm_state(cache):
    out = []
    for item in cache:
        if hasattr(item, "caches"):
            keys, _ = item[0].state
            out += [keys, *[a for a in item[1].state if a is not None]]
        else:
            out += [item[0], item[1]]
    return [np.asarray(a) for a in out]


def _logits(lm, cache, last):
    logits = lm(last, cache=cache).logits[:, -1]
    out = [np.asarray(logits)]
    for _ in range(3):
        token = mx.argmax(logits, -1)[:, None]
        logits = lm(token, cache=cache).logits[:, -1]
        out.append(np.asarray(logits))
    return out


def _split_matches(adapter, make, ids, sizes, k, prefix, state_of):
    """Prefill ``ids`` split at ``k`` after ``prefix`` local chunks; the caches
    and the next greedy logits must match one model's prefill."""
    host, helper = make(), make()
    adapter.keep_first(helper, k)
    chunks = _chunks(ids, sizes)
    reference = host.make_cache()
    for chunk in chunks:
        host(chunk, cache=reference)

    cache, helper_cache = host.make_cache(), helper.make_cache()
    for chunk in chunks[:prefix]:
        host(chunk, cache=cache)
    if prefix:
        adapter.load_prefix(helper_cache, adapter.prefix_state(cache, k))
    rest = chunks[prefix:]
    tokens = np.concatenate([np.asarray(c[0]) for c in rest])
    feed = adapter.job_feed(
        host, cache if prefix else None, tokens, [c.shape[1] for c in rest], k
    )
    for index, chunk in enumerate(rest):
        inputs = feed(index) if feed else {}
        boundary, state = adapter.run_first(helper, helper_cache, chunk, 0, inputs)
        adapter.run_rest(None, host, cache, chunk, k, boundary, state)
        adapter.apply_state(cache, k, state)
        mx.eval([c.state for c in cache])

    for a, b in zip(state_of(reference), state_of(cache)):
        np.testing.assert_array_equal(a, b)
    last = ids[:, -1:]
    for a, b in zip(_logits(host, reference, last), _logits(host, cache, last)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("k,prefix", [(1, 0), (2, 0), (3, 0), (2, 1)])
def test_glm_split_matches_whole_model(k, prefix):
    ids = mx.array(np.random.default_rng(5).integers(3, 60, (1, 23)))
    _split_matches(GLM, _glm, ids, [9, 6, 7], k, prefix, _glm_state)


# ---------------------------------------------------------------------------
# Qwen3.8 Flash-Next: GDN, PLE and QSA layers on either side of the cut.
# ---------------------------------------------------------------------------


def _qwen4(layers=2):
    """The tiny Qwen4 stack; four layers put PLE on layer 1 and QSA on 2."""
    mx.random.seed(7)
    lm, config = _tiny_qwen4_exp()
    if layers == 4:
        text = replace(
            config.text_config,
            num_hidden_layers=4,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
            ],
            ple_layer_ids=[2],
        )
        mx.random.seed(7)
        lm = type(lm)(text, replace(config, text_config=text))
    mx.eval(lm.parameters())
    return lm


def _qwen4_state(cache):
    out = []
    for item in cache:
        if hasattr(item, "update_indexer"):
            out += [a for a in item.state if a is not None]
        else:
            out += [a for a in item.cache if a is not None]
    return [np.asarray(a.astype(mx.float32)) for a in out]


@pytest.mark.parametrize(
    "layers,k,prefix", [(2, 1, 0), (4, 1, 0), (4, 2, 0), (4, 3, 0), (4, 2, 1)]
)
def test_qwen4_split_matches_whole_model(layers, k, prefix):
    ids = mx.array(np.random.default_rng(5).integers(3, 58, (1, 23)))
    _split_matches(
        QWEN4, lambda: _qwen4(layers), ids, [9, 6, 7], k, prefix, _qwen4_state
    )


@pytest.mark.parametrize(
    "factory,make,adapter,k,state_of",
    [
        (_tiny_glm5_next, _glm, GLM, 2, _glm_state),
        (_tiny_qwen4_exp, _qwen4, QWEN4, 1, _qwen4_state),
    ],
    ids=["glm", "qwen4"],
)
def test_scheduler_family_helper_prefill_matches_local(
    factory, make, adapter, k, state_of
):
    tokens = [int(t) for t in np.random.default_rng(3).integers(3, 58, 23)]
    host = _tiny_vlm_adapter(factory)
    local = _drive(_scheduler(host, 6), _request(tokens), tokens)

    helper = make()
    adapter.keep_first(helper, k)
    sched = _scheduler(host, 6)
    session, listener, thread = _session(
        host._language_model, helper, k, adapter=adapter
    )
    sched.attach_prefill_helper(session)
    state = _drive(sched, _request(tokens), tokens)
    assert state.helper is not None and state.helper.index == len(state.helper.grid)
    for a, b in zip(state_of(local.cache), state_of(state.cache)):
        np.testing.assert_array_equal(a, b)
    session.close()
    thread.join(10)
    listener.close()


# ---------------------------------------------------------------------------
# MiniMax M3: dense and MoE layers with sparse attention on either side.
# ---------------------------------------------------------------------------


def _minimax():
    apply_mlx_vlm_minimax_m3_compat_patch()
    from mlx_vlm.models.minimax_m3_vl.language import LanguageModel

    args = _tiny_text_config(
        pack_shared_expert=False,
        num_hidden_layers=4,
        moe_layer_freq=[0, 0, 1, 1],
        vocab_size=64,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1, 1, 1, 1],
            "sparse_index_dim": 16,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 8,
            "sparse_topk_blocks": 2,
        },
    )
    mx.random.seed(7)
    lm = LanguageModel(args)
    mx.eval(lm.parameters())
    return lm


def _minimax_state(cache):
    return [np.asarray(a) for item in cache for a in (*item.state[0], item.state[1])]


@pytest.mark.parametrize("k,prefix", [(1, 0), (2, 0), (3, 0), (2, 1)])
def test_minimax_split_matches_whole_model(k, prefix):
    # Past 16 blocks of 8 the layers take the sparse path; the 9-token tail
    # keeps explicit positions.
    ids = mx.array(np.random.default_rng(5).integers(3, 60, (1, 200)))
    _split_matches(MINIMAX, _minimax, ids, [96, 94, 9], k, prefix, _minimax_state)


class _Drafter:
    """Records what prefill seeds a block drafter with."""

    def __init__(self, layers, window):
        self.target_layer_ids, self.window = layers, window
        self.seeds = []

    def seed_request(self, request_id, layers, position):
        self.seeds.append((position, [np.asarray(x) for x in layers]))

    def prepare_request(self, *args):
        pass

    def bind_uid(self, *args):
        pass


def test_scheduler_seeds_a_block_drafter_from_both_macs():
    k, tokens = 2, [int(t) for t in np.random.default_rng(3).integers(3, 60, 100)]

    def drive(helper_k):
        host = _tiny_vlm_adapter(
            lambda: (_minimax(), SimpleNamespace(model_type="minimax_m3_vl"))
        )
        drafter = _Drafter([0, 3], window=40)
        host._language_model._omlx_drafter = drafter
        sched = _scheduler(host, 32)
        session = None
        if helper_k:
            helper = _minimax()
            MINIMAX.keep_first(helper, helper_k)
            session, listener, thread = _session(
                host._language_model, helper, helper_k, adapter=MINIMAX
            )
            sched.attach_prefill_helper(session)
        state = _drive(sched, _request(tokens), tokens)
        if session is not None:
            assert state.helper is not None
            session.close()
            thread.join(10)
            listener.close()
        return drafter.seeds

    local, split = drive(0), drive(k)
    # Layer 0 runs on the helper and layer 3 here; the window starts in the
    # second chunk.
    assert [p for p, _ in split] == [p for p, _ in local] == [60, 64, 96]
    for (_, a), (_, b) in zip(local, split):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
