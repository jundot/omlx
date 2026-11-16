"""Prefill Accelerator: a split prefill must match the whole-model prefill."""

import itertools
import json
import os
import shutil
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
from mlx.utils import tree_flatten, tree_unflatten
from test_deepseek_v41 import _ENGRAM, load_reference_weights, tiny, tiny_ced
from test_inkling_vlm_mtp import _inkling2_tiny
from test_mlx_vlm_minimax_m3_compat import _tiny_text_config
from test_scheduler_chunked_prefill import (
    _tiny_glm5_next,
    _tiny_qwen3_5,
    _tiny_qwen4_exp,
    _tiny_vlm_adapter,
)

from omlx.cluster.transport import HostInterfaces, InterfaceAddress
from omlx.patches.deepseek_v41.language import LanguageModel
from omlx.patches.deepseek_v41.storage import DiskEngramEmbedding
from omlx.patches.mlx_lm_mtp import prompt_priming
from omlx.patches.mlx_vlm_minimax_m3_compat import (
    apply_mlx_vlm_minimax_m3_compat_patch,
)
from omlx.prefill_helper import agent, launcher, lending, wire
from omlx.prefill_helper import manager as manager_module
from omlx.prefill_helper import routes as helper_routes
from omlx.prefill_helper import worker as worker_module
from omlx.prefill_helper.adapter import Capture
from omlx.prefill_helper.adapters import adapter_for, adapter_for_model
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
INKLING2 = adapter_for("inkling2")
QWEN35 = adapter_for("qwen3_5")
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

    ``prefix`` chunks run on this Mac alone first, like a cached prefix. A list
    ``k`` splits each chunk at its own layer, as a helper that drops its last
    layers during a long prompt.
    """
    splits = k if isinstance(k, list) else [k] * (len(sizes) - prefix)
    k = splits[0]
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
        k = splits[index]
        with DSV41.use_first(helper, k):
            boundary, state = DSV41.run_first(
                helper, helper_cache[:k], chunk, nxt, inputs
            )
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


@pytest.mark.parametrize("k", [1, 3, [3, 2, 1]])
def test_split_matches_whole_model_with_ced(k):
    # Chunks wider than the window skip the decoder tail, then replay it.
    _run(tiny_ced(), k, [9, 6, 7])


@pytest.mark.parametrize("k", [1, 3, [3, 1, 1]])
def test_split_matches_whole_model_with_engram(k):
    _run(tiny(**_ENGRAM), k, [7, 5, 3])


def test_split_continues_a_cached_prefix():
    _run(tiny(**_ENGRAM), 2, [6, 5, 4], prefix=1)
    _run(tiny_ced(), 3, [9, 6, 7], prefix=1)


def _dsv41_contract():
    """Below the CED midpoint: a window-only layer, KV sources of ratio 2 and
    1 with their readers, and Engram on the first two layers. A second helper
    stage can start at the sources 3 and 5."""
    engram = {**_ENGRAM, "engram_layer_ids": (0, 1)}
    return _model(
        tiny_ced(
            n_layers=12,
            compress_ratios=(0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1),
            kv_source_layers=(1, 3, 5, 6),
            index_source_layers=(1, 3, 5, 6, 8, 10),
            candidate_source_layer=6,
            **engram,
        )
    )


# ---------------------------------------------------------------------------
# Helper process protocol over a loopback socket.
# ---------------------------------------------------------------------------

TOKEN = b"session-token"


def _serve(lm, k, *, once=True, adapter=DSV41, memory_limit=0):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    server = HelperServer(
        lm,
        adapter,
        k,
        TOKEN,
        {"protocol": wire.PROTOCOL, "layers": k},
        memory_limit=memory_limit,
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


def test_a_prompt_past_the_helpers_memory_is_handed_back_as_out_of_memory():
    config, k, sizes = tiny(**_ENGRAM), 2, [6, 5]
    host = _model(config)
    listener, thread = _serve(_helper_model(config, k), k, memory_limit=1)
    client = HelperClient("127.0.0.1", listener.getsockname()[1], TOKEN)
    tokens = np.arange(3, 3 + sum(sizes))
    job = client.submit(
        tokens, sizes, feed=DSV41.job_feed(host, None, tokens, sizes, k)
    )
    with pytest.raises(HelperError, match="no GPU memory left"):
        job.take(timeout=60)
    assert client.out_of_memory
    client.close()
    thread.join(10)
    listener.close()


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
    session.alone_ratio = 1.0  # a prompt has run on this Mac alone
    # Until this Mac is timed, the helper takes every layer it holds.
    assert session.pick_layers() == 18
    # This Mac takes twice as long per unit of work: the helper gets more.
    session.observe(10, 2048, helper_s=10 * 2048, host_s=10.1 * 2 * 2048)
    assert session.pick_layers() == 14
    # A slow link is a stage too: each layer's state costs transfer time, so
    # fewer layers cross (4 s per token for the boundary, 1.5 s per layer).
    client.link_bps = 1.0
    client.token_bytes = {-1: 4.0, **{i: 1.5 for i in range(18)}}
    assert session.pick_layers() == 10
    # A job's chunks carry only the layers it used, here ten that cost less.
    # What the layers above them cost stays known: the split does not take
    # them for free and swing back.
    sent = {"b:h": 4096, **{f"s:{i}.k": 512 for i in range(10)}}
    chunk = {name: (np.zeros(size, np.uint8), "uint8") for name, size in sent.items()}
    HelperClient._measure(client, {"size": 1024}, chunk, 1.0)
    assert session.pick_layers() == 13
    # A layer no job has used yet is taken to cost what the measured ones do.
    client.token_bytes = {-1: 4.0, **{i: 1.5 for i in range(10)}}
    assert session.pick_layers() == 10
    # Slower than this Mac alone at any split: the prompt stays here.
    client.link_bps = 0.05
    assert session.pick_layers() == 0
    assert session.start("r", np.arange(4096), [4096], None) is None
    # A helper that timed slow is not a slow link: it keeps a job, whose
    # chunks correct the estimate.
    client.link_bps = 1.0
    session.observe(10, 2048, helper_s=10 * 2048 * 1000, host_s=10.1 * 2 * 2048)
    assert session.pick_layers() == 1
    session.fixed_layers = 5
    assert session.pick_layers() == 5
    # A split that is faster once it runs can still lose a short prompt to
    # the wait for its first chunk: 4.5 s per token on the link here, against
    # 10 s per token alone and 6 s per token split.
    link = SimpleNamespace(
        info={}, alive=True, link_bps=1.0, token_bytes={-1: 3.5, 0: 0.25}
    )
    quick = PrefillHelperSession(link, DSV41, None, 4, "helper", [1.0] * 10)
    quick.unit_helper, quick.unit_host = 0.1, 1.0
    assert quick.pick_layers() == 4
    assert quick.pick_layers([100] * 10) == 4
    assert quick.pick_layers([100]) == 0


def test_auto_split_keeps_the_cost_of_a_split_out_of_this_mac_alone():
    link = SimpleNamespace(info={}, alive=True, link_bps=1.0, token_bytes={-1: 12.0})
    session = PrefillHelperSession(link, DSV41, None, 10, "helper", [1.0] * 10)
    # Split at 8: this Mac's two layers took 1 s per token each, taking in the
    # helper's state and finishing the chunk 3 s per token more.
    session.observe(8, 2048, helper_s=0.08 * 2048, host_s=2 * 2048, extra_s=3 * 2048)
    # Alone this Mac needs 10 s per token, less than the link's 12.
    session.alone_ratio = 1.0
    assert session.pick_layers() == 0
    # On a faster link the split also pays the 3 s on this Mac.
    link.token_bytes = {-1: 9.0}
    assert session.pick_layers() == 4


def test_auto_split_times_this_mac_alone_before_a_close_call():
    def session(alone_s):
        link = SimpleNamespace(info={}, alive=True, link_bps=1.0, token_bytes={-1: 9.5})
        split = PrefillHelperSession(link, DSV41, None, 10, "helper", [1.0] * 10)
        split.unit_helper, split.unit_host = 0.01, 1.0
        # The link's 9.5 s per token is close to the 10 s estimated alone:
        # this prompt stays here and is timed.
        assert split.pick_layers() == 0
        split.observe_alone(2048, alone_s * 2048)
        return split.pick_layers()

    assert session(12.0) == 1
    assert session(9.0) == 0


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
    incompatible = staticmethod(launcher.incompatible)

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
        fits=None,
    ):
        self.lm, self.k, self.fits, self.launched = lm, k, fits, []
        self.version = version or launcher_version()
        self.budget, self.missing = budget, list(missing)
        self.free, self.copies, self.hold = free, list(copies), hold
        self.copied, self.deleted, self.servers, self.paired = [], [], [], True
        self.unreachable = False

    def resolve_target(self, node_id):
        if self.unreachable:
            raise launcher.UnreachableError("M5max does not answer over SSH.")
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
        self.launched.append(layers)
        if self.fits is not None and layers > self.fits:
            raise launcher.LaunchError(
                "The helper on M5max did not start: RuntimeError: [METAL] Command "
                "buffer execution failed: Insufficient Memory "
                "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
            )
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


def _wait_until(predicate, message, timeout=30):
    """Wait for something append-only to happen, not for a state to be visible."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(message)


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


def test_manager_retries_a_helper_that_does_not_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(manager_module, "_RETRY_DELAYS_S", (0.05,))
    attached = []
    fake = _FakeLauncher(None)
    fake.unreachable = True
    manager = _manager(fake, attached, tmp_path / "speeds.json")
    manager.configure("node-1", None)
    _wait_state(manager, "offline", reason="does not answer over SSH")
    fake.unreachable = False
    _wait_state(manager, "ready")
    assert attached
    manager.close()


def test_manager_retries_an_offline_helper_on_use(tmp_path, monkeypatch):
    monkeypatch.setattr(manager_module, "_RETRY_DELAYS_S", (600.0,))
    monkeypatch.setattr(manager_module, "_RETRY_ON_USE_S", 0.0)
    fake = _FakeLauncher(None)
    fake.unreachable = True
    manager = _manager(fake, [], tmp_path / "speeds.json")
    manager.configure("node-1", None)
    _wait_state(manager, "offline", reason="does not answer over SSH")
    fake.unreachable = False
    manager.retry_if_offline()
    _wait_state(manager, "ready")
    manager.close()
    # A copy that waits for the user is not cancelled.
    old = {"name": "Old", "bytes": 20 << 30, "used": 0.0}
    fake = _FakeLauncher(
        None, missing=["model-1.safetensors"], free=40 << 30, copies=[old]
    )
    manager = _manager(fake, [], tmp_path / "speeds.json")
    manager.configure("node-1", None)
    _wait_state(manager, "needs_files")
    manager.retry_if_offline()
    assert manager._commands.empty()
    assert manager.status()["state"] == "needs_files"
    manager.close()


def test_manager_retries_a_lost_helper_but_not_a_stopped_one(tmp_path, monkeypatch):
    monkeypatch.setattr(manager_module, "_RETRY_DELAYS_S", (0.05,))
    monkeypatch.setattr(manager_module, "_WATCH_S", 0.05)
    attached = []
    fake = _FakeLauncher(None)
    manager = _manager(fake, attached, tmp_path / "speeds.json")
    seen = []
    reported = manager._set

    def record(state, **fields):
        seen.append(state)
        reported(state, **fields)

    monkeypatch.setattr(manager, "_set", record)
    manager.configure("node-1", None)
    _wait_state(manager, "ready")
    first = attached[-1]
    first.client.close()
    # The retry can run to completion between two polls, so wait for the second
    # helper to be launched instead of catching a transient state; the recorded
    # transitions still show the manager left "ready" on the way there.
    _wait_until(lambda: len(fake.servers) >= 2, "the lost helper was not retried")
    _wait_state(manager, "ready")
    assert "offline" in seen
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


def test_auto_starts_with_a_layer_fewer_after_running_out_of_memory(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(manager_module, "_RETRY_DELAYS_S", (0.05,))
    attached = []
    fake = _FakeLauncher(None, fits=1)
    manager = _manager(fake, attached, tmp_path / "speeds.json")
    manager.configure("node-1", None)
    assert _wait_state(manager, "ready")["held"] == 1 and fake.launched == [2, 1]
    # A prompt went past the helper's memory and ran here: the status says so.
    attached[-1].client.out_of_memory = True
    assert manager.status()["out_of_memory"]
    # A fixed count is not changed.
    manager.configure("node-1", 2)
    _wait_state(manager, "offline", reason="Insufficient Memory")
    assert set(fake.launched[2:]) == {2}
    # Not even one layer: waits for the user instead of retrying.
    fake.fits = 0
    manager.configure("node-1", None)
    _wait_state(manager, "error", reason="Not even one layer")
    manager.close()


def test_manager_starts_a_new_helper_from_the_saved_speeds(tmp_path):
    attached, speeds = [], tmp_path / "speeds.json"
    # Saved before the model moved: keyed by its old full path.
    old = {"helper": 1e-5, "host": 3e-5, "at": 1.0}
    speeds.write_text(json.dumps({"node-1:/moved/from/x": old}))
    manager = _manager(_FakeLauncher(None), attached, speeds)
    manager.configure("node-1", None)
    _wait_state(manager, "ready")
    assert (attached[-1].unit_helper, attached[-1].unit_host) == (1e-5, 3e-5)
    attached[-1].observe(2, 2048, helper_s=1.0, host_s=3.0, extra_s=0.5)
    measured = (attached[-1].unit_helper, attached[-1].unit_host)
    manager.close()

    again = _manager(_FakeLauncher(None), attached, speeds)
    again.configure("node-1", None)
    _wait_state(again, "ready")
    assert (attached[-1].unit_helper, attached[-1].unit_host) == measured
    assert attached[-1].unit_extra == 0.5 / 2048
    # Switching Macs keeps the measured speeds under the Mac that measured them.
    again.configure("node-2", None)
    _wait_state(again, "ready")
    saved = json.loads(speeds.read_text())
    assert {key.split(":")[0] for key in saved} == {"node-1"} and "node-1:x" in saved
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
    assert manager_module.helper_budget(info) == [80 << 30]
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
# A helper daemon (Linux, NVIDIA) is driven over its control channel.
# ---------------------------------------------------------------------------

_KEY = bytes(range(32))
_GPU = {"name": "NVIDIA RTX A6000", "total": 48 << 30}


def _fake_daemon(root, events):
    """A helper daemon's control channel, as far as the host drives it."""
    listener = socket.create_server(("127.0.0.1", 0))

    def reply_to(meta, arrays):
        op = meta["op"]
        if op == "info":
            return {
                "op": "ok",
                "version": "0.1.0",
                "data_protocol": wire.PROTOCOL,
                "contracts": {"deepseek_v41": DSV41.contract},
                "gpus": [{**_GPU, "free": 40 << 30}, {**_GPU, "free": 44 << 30}],
                "disk_free": 1 << 40,
                "link_mbps": 1000,
                "session": {"host": "Studio", "model": "Other", "gpu_bytes": [4 << 30]},
            }
        if op == "probe":
            folder = root / meta["name"]
            partial = {
                name: (folder / f"{name}.part").stat().st_size
                for name in meta["files"]
                if (folder / f"{name}.part").exists()
            }
            missing = [n for n in meta["files"] if not (folder / n).exists()]
            return {
                "op": "ok",
                "found": None,
                "copy": str(folder),
                "missing": missing,
                "partial": partial,
                "copies": [],
                "free": 1 << 40,
            }
        if op == "put":
            folder = root / meta["name"]
            folder.mkdir(parents=True, exist_ok=True)
            part = folder / f"{meta['file']}.part"
            if (part.stat().st_size if part.exists() else 0) != meta["offset"]:
                return {"op": "error", "error": "wrong offset"}
            with open(part, "ab") as f:
                f.write(arrays["data"][0].tobytes())
            if meta["last"]:
                part.rename(folder / meta["file"])
            return {"op": "ok"}
        events.append((op, meta))
        if op == "start":
            return {"op": "ready", "port": 4242, "hello": {"layers": meta["layers"]}}
        return {"op": "ok", "free": 7}

    def one(sock):
        try:
            wire.recv(sock)
            wire.authenticate(sock, agent._control_key(_KEY), server=True)
            wire.send(sock, {"op": "hello", "v": 1, "node_id": "helper"})
            while True:
                wire.send(sock, reply_to(*wire.recv(sock)))
        except (OSError, wire.ProtocolError):
            events.append(("closed", {}))
        finally:
            sock.close()

    def accept():
        while True:
            try:
                sock, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=one, args=(sock,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    return listener


def test_agent_transport_drives_a_helper_daemon(tmp_path, monkeypatch):
    events = []
    listener = _fake_daemon(tmp_path / "helper", events)
    port = listener.getsockname()[1]
    me = SimpleNamespace(node_id="studio")
    monkeypatch.setattr(agent, "get_node_identity", lambda: me)
    target = agent.AgentTarget("helper", "nvidia1", "127.0.0.1", "ethernet", port, _KEY)

    # The memory of a session already running there counts as free.
    info = agent.remote_info(target)
    assert manager_module.helper_budget(info) == [44 << 30, 44 << 30]
    # An integrated GPU leaves the OS its share of the RAM.
    spark = {"name": "NVIDIA GB10", "total": 121 << 30, "free": 119 << 30}
    with monkeypatch.context() as patch:
        patch.setattr(agent, "_call", lambda sock, meta: {"gpus": [spark]})
        assert manager_module.helper_budget(agent.remote_info(target)) == [119 << 30]
        spark["integrated"] = True
        assert manager_module.helper_budget(agent.remote_info(target)) == [105 << 30]
    assert info["device"] == "2x NVIDIA RTX A6000"
    assert info["lending"] == {"host": "Studio", "model": "Other"}
    assert agent.incompatible(info, DSV41, "nvidia1") is None
    assert "does not support" in agent.incompatible(info, GLM, "nvidia1")
    older = {**info, "contracts": {"deepseek_v41": DSV41.contract - 1}}
    assert "update the helper" in agent.incompatible(older, DSV41, "nvidia1")

    # A copy stopped halfway resumes from the bytes the helper kept.
    model = tmp_path / "Model"
    model.mkdir()
    (model / "config.json").write_bytes(b"{}")
    (model / "w.safetensors").write_bytes(os.urandom(2500))
    monkeypatch.setattr(agent, "_CHUNK_BYTES", 1000)
    names, sent = ["config.json", "w.safetensors"], []
    agent.copy_files(
        target, model, "/x/Model", names, sent.append, cancelled=lambda: bool(sent)
    )
    assert sent == ["config.json"]
    where = agent.locate_files(target, "Model", {"w.safetensors": 2500})
    assert where["missing"] == ["w.safetensors"] and not where["partial"]
    stops = iter([False, True])
    agent.copy_files(
        target, model, "/x/Model", ["w.safetensors"], cancelled=lambda: next(stops)
    )
    where = agent.locate_files(target, "Model", {"w.safetensors": 2500})
    assert where["partial"] == {"w.safetensors": 1000}
    agent.copy_files(target, model, "/x/Model", ["w.safetensors"])
    copied = tmp_path / "helper" / "Model" / "w.safetensors"
    assert copied.read_bytes() == (model / "w.safetensors").read_bytes()
    assert agent.delete_copies(target, ["Old"]) == 7

    # The data token never crosses: both ends derive it from the session id.
    process = agent.launch(
        target, "/x/Model", "deepseek_v41", 5, host_name="Studio", warm_tokens=8
    )
    start = next(meta for op, meta in events if op == "start")
    assert (start["model"], start["contract"]) == ("Model", DSV41.contract)
    assert process.token == agent._data_token(_KEY, bytes.fromhex(start["session"]))
    assert (process.port, process.hello) == (4242, {"layers": 5})
    process.stop()

    stranger = replace(target, key=bytes(32))
    with pytest.raises(launcher.LaunchError, match="Pair it again"):
        agent.remote_info(stranger)
    listener.close()


def test_a_helper_daemon_moves_to_a_faster_wired_link(monkeypatch):
    lan = agent.AgentTarget(
        "gpu", "gpu-box", "192.168.1.230", "ethernet", 8851, _KEY, speed_mbps=1000
    )
    links = [
        {"ipv4": ["192.168.1.230/24"], "link_local": "fe80::1", "mbps": 1000},
        {"ipv4": [], "link_local": "fe80::2", "mbps": 2500},
    ]
    ports = SimpleNamespace(media_speeds={"en0": 1000, "en8": 10000, "en7": 2500})
    answers = {"fe80::2%en7", "10.10.0.2"}
    tried = []

    def connect(target, timeout):
        tried.append(target.address)
        if target.address not in answers:
            raise launcher.LaunchError("no route")
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(agent, "remote_info", lambda target: {"links": links})
    monkeypatch.setattr(agent, "probe_host_interfaces", lambda host: ports)
    monkeypatch.setattr(agent, "_local_link", lambda address: ("ethernet", 2500))
    monkeypatch.setattr(agent, "_connect", connect)
    # A cable without DHCP: the helper's link-local address, scoped to the
    # port of this Mac that reaches it. A port at the LAN's speed is not tried.
    moved = agent._fastest_link(lan)
    assert (moved.address, moved.speed_mbps) == ("fe80::2%en7", 2500)
    assert moved.endpoint == "[fe80::2%en7]:8851"
    assert tried == ["fe80::2%en8", "fe80::2%en7"]
    # A self-assigned address does not route reliably from macOS, also when
    # discovery found the helper there.
    links[1].update(ipv4=["169.254.10.2/16"])
    assert agent._fastest_link(lan).address == "fe80::2%en7"
    cable = replace(lan, address="169.254.10.2", speed_mbps=None)
    assert agent._fastest_link(cable).address == "fe80::2%en7"
    # Static addresses on the cable work the same way.
    links[1].update(ipv4=["10.10.0.2/30"])
    assert agent._fastest_link(lan).address == "10.10.0.2"
    # A link no faster than the LAN is not worth the move.
    links[1]["mbps"] = 1000
    assert agent._fastest_link(lan) is lan


def test_helper_daemon_checks_and_layers_over_two_gpus(monkeypatch):
    # The first GPU also holds the shared tensors; each keeps 8 GiB spare.
    gib = 1 << 30
    assert manager_module.fit_layers(2 * gib, [4 * gib] * 20, [20 * gib] * 2, 19) == 5
    # A busy GPU does not end the chain: the roomier one takes the first stage.
    assert (
        manager_module.fit_layers(2 * gib, [4 * gib] * 20, [9 * gib, 20 * gib], 19) == 2
    )
    # More GPUs hold no more layers: the helper runs its stages on two.
    assert manager_module.fit_layers(2 * gib, [4 * gib] * 20, [20 * gib] * 4, 19) == 5

    paired = {"n": {"friendly_name": "nvidia1"}}
    registry = SimpleNamespace(get=paired.get, is_paired=paired.__contains__)
    monkeypatch.setattr(helper_routes, "get_device_registry", lambda: registry)
    monkeypatch.setattr(helper_routes, "forget_host", lambda node: None)
    target = agent.AgentTarget(
        "n", "nvidia1", "10.0.0.9", "ethernet", 7001, _KEY, speed_mbps=10000
    )
    info = {
        "version": "0.1.0",
        "data_protocol": wire.PROTOCOL,
        "contracts": {"qwen4_exp": 1},
        "budgets": [40 * gib, 44 * gib],
        "disk_free": 1 << 40,
        "link_mbps": 1000,
    }
    daemon = SimpleNamespace(
        VIA=agent.VIA,
        resolve_target=lambda node: target,
        remote_info=lambda target, host=None: info,
        incompatible=agent.incompatible,
    )
    monkeypatch.setattr(helper_routes, "transport_for", lambda node: daemon)
    rows = helper_routes._check("n")["rows"]
    # No SSH and no battery on a helper daemon; its own version is fine.
    assert [(r["key"], r["status"]) for r in rows] == [
        ("control", "pass"),
        ("link", "warn"),
        ("version", "pass"),
        ("memory", "pass"),
        ("disk", "pass"),
    ]
    assert rows[0]["detail"] == "10.0.0.9:7001"
    # This Mac's port is 10 Gb, the helper's 1 Gb: the slower end is the link.
    assert rows[1]["speed_mbps"] == 1000
    assert rows[2]["families"] == ["qwen4_exp"] and "expected" not in rows[2]
    assert (rows[3]["bytes"], rows[3]["gpus"]) == (84 * gib, 2)


# ---------------------------------------------------------------------------
# GLM-5.3: KDA and sparse-attention layers on either side of the cut.
# ---------------------------------------------------------------------------


def _glm():
    mx.random.seed(7)
    lm, _ = _tiny_glm5_next()
    mx.eval(lm.parameters())
    return lm


def _glm_contract():
    """_glm with its identity gains and its zero gates, biases and mixes moved,
    so the fixture pins the hyper-connection, the router and the index pool."""
    lm = _glm()
    lm.apply(
        lambda p: (
            p + mx.random.uniform(-0.3, 0.3, p.shape)
            if p.ndim == 1 or not mx.any(p).item()
            else p
        )
    )
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
    def step(tokens):
        out = lm(tokens, cache=cache)
        return getattr(out, "logits", out)[:, -1]

    logits = step(last)
    out = [np.asarray(logits)]
    for _ in range(3):
        logits = step(mx.argmax(logits, -1)[:, None])
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


def test_glm_helper_drops_the_mtp_head():
    lm = _glm()
    # What an MTP checkpoint attaches, also when the helper lacks its files.
    lm.mtp = [lm.model.layers[-1]]
    GLM.keep_first(lm, 1)
    assert "mtp" not in lm


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


# ---------------------------------------------------------------------------
# Qwen3.5: Gated DeltaNet and full-attention layers on either side of the cut.
# ---------------------------------------------------------------------------


def _tiny_qwen35():
    """The tiny Qwen3.5 stack with six layers (L L L F L L) and a partial
    rotary width, so a cut can sit inside a linear run or right after
    attention. Keys of 32 dims in inference mode take the recurrence kernel
    a loaded model runs."""
    lm, _ = _tiny_qwen3_5()
    rope = {**lm.args.rope_parameters, "partial_rotary_factor": 0.5}
    text = replace(
        lm.args, num_hidden_layers=6, linear_key_head_dim=32, rope_parameters=rope
    )
    # A call without position ids reads these to number a text prompt.
    config = SimpleNamespace(
        model_type="qwen3_5",
        text_config=text,
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=61,
        video_token_id=62,
        vision_start_token_id=63,
    )
    return type(lm)(text, config).eval(), config


def _qwen35():
    mx.random.seed(7)
    lm, _ = _tiny_qwen35()
    mx.eval(lm.parameters())
    return lm


def _qwen35_contract():
    """_qwen35 with its identity norm gains and gate biases moved."""
    lm = _qwen35()
    lm.apply(lambda p: p + mx.random.uniform(-0.3, 0.3, p.shape) if p.ndim == 1 else p)
    mx.eval(lm.parameters())
    return lm


def _qwen35_state(cache):
    out = []
    for item in cache:
        out += item.cache if hasattr(item, "cache") else item.state
    return [np.asarray(a) for a in out]


@pytest.mark.parametrize(
    "k,sizes,prefix",
    [
        (2, [9, 6, 7], 0),
        (4, [9, 6, 7], 0),
        (5, [12, 10], 0),
        (2, [9, 6, 7], 1),
        (4, [9, 6, 7], 1),
    ],
)
def test_qwen35_split_matches_whole_model(k, sizes, prefix):
    ids = mx.array(np.random.default_rng(5).integers(3, 58, (1, 23)))
    _split_matches(QWEN35, _qwen35, ids, sizes, k, prefix, _qwen35_state)


def test_qwen35_split_feeds_a_block_drafter_and_the_mtp_head(monkeypatch):
    k, start = 4, 3
    host, helper = _qwen35(), _qwen35()
    QWEN35.keep_first(helper, k)
    # A helper Mac whose GPU rounds its rotary frequencies another way: its
    # attention layer must turn by the frequencies the host sends.
    rotary = helper.model.layers[3].self_attn.rotary_emb
    rotary._inv_freq = rotary.inv_freq * 1.01
    ids = mx.array(np.random.default_rng(5).integers(3, 58, (1, 12)))
    whole = host(ids, cache=host.make_cache(), capture_layer_ids=[1, 5])
    # A model with an active Lightning MTP head, as its loader marks it.
    host._omlx_mtp_decode_enabled = host._omlx_mtp_chain = True
    host.mtp = SimpleNamespace()
    primed = []
    monkeypatch.setattr(prompt_priming, "maybe_capture", lambda *a: primed.append(a))

    cache = host.make_cache()
    here, there = Capture([1, 5], start), Capture([1], start)
    inputs = QWEN35.job_feed(host, None, np.asarray(ids[0]), [ids.shape[1]], k)(0)
    boundary, state = QWEN35.run_first(
        helper, helper.make_cache(), ids, 0, inputs, there
    )
    out = QWEN35.run_rest(None, host, cache, ids, k, boundary, state, capture=here)
    QWEN35.apply_state(cache, k, state)
    QWEN35.finish_chunk(host, ids, out, cache)

    lower, upper = (np.asarray(a[:, start:]) for a in whole.hidden_states)
    np.testing.assert_array_equal(np.asarray(there.out[1]), lower)
    np.testing.assert_array_equal(np.asarray(here.out[5]), upper)
    assert list(here.out) == [5]
    # The head sees the final norm of the last layer, after the helper's
    # attention rows moved the cache offset.
    ((lm, tokens, normed, primed_cache),) = primed
    assert lm is host and tokens is ids and primed_cache is cache
    assert cache[3].offset == ids.shape[1]
    np.testing.assert_array_equal(
        np.asarray(normed), np.asarray(host.model.norm(whole.hidden_states[1]))
    )
    # A priming failure leaves the head unprimed, as in a local prefill.
    failing = MagicMock(side_effect=RuntimeError("priming"))
    monkeypatch.setattr(prompt_priming, "maybe_capture", failing)
    QWEN35.finish_chunk(host, ids, out, cache)
    failing.assert_called_once()


def test_only_a_model_that_gets_a_helper_has_an_adapter(tmp_path):
    path = str(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert adapter_for_model("qwen3_5", "vlm", path) is QWEN35
    # Text-only conversions run on the LLM or decision engine: no helper.
    assert adapter_for_model("qwen3_5", "batched", path) is None
    entry = SimpleNamespace(
        config_model_type="qwen3_5", engine_type="decision", model_path=path
    )
    assert helper_routes._adapter(entry) is None
    # The server loads a ParoQuant checkpoint with a loader of its own.
    config = {"model_type": "qwen3_5", "quantization_config": {}}
    config["quantization_config"]["quant_method"] = "paroquant"
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert adapter_for_model("qwen3_5", "vlm", path) is None
    # The other families keep their adapter on every engine.
    assert adapter_for_model("qwen4-exp", "batched", path) is QWEN4


def test_qwen35_keeps_a_yarn_prompt_off_the_helpers_attention_layers():
    host = _qwen35()
    host.args.rope_parameters["type"] = "yarn"
    tokens = np.arange(8)
    # Three linear layers need no positions; the fourth is attention.
    assert QWEN35.job_feed(host, None, tokens, [8], 3) is None
    with pytest.raises(RuntimeError, match="YaRN"):
        QWEN35.job_feed(host, None, tokens, [8], 4)


# ---------------------------------------------------------------------------
# Inkling 2: attention-residual blocks and conv windows on either side.
# ---------------------------------------------------------------------------


def _inkling2():
    """The tiny Inkling 2 stack: NVFP4 experts in layers 0-5 and MXFP8 in
    6-11, block boundaries at 0 and 6, full attention at 5 and 11."""
    lm = _inkling2_tiny(formats=("nvfp4",) * 6 + ("mxfp8",) * 6)
    mx.eval(lm.parameters())
    return lm


def _inkling2_contract():
    """Eight narrower layers (the helper takes NVFP4 0-5 and MXFP8 6) with
    norm gains near one and NVFP4 block scales and tensor amax of the size the
    checkpoint ships. 32 experts take the router kernel; MLX CUDA's NVFP4
    matmul is exact from 64 inputs per row on."""
    lm = _inkling2_tiny(
        formats=("nvfp4",) * 6 + ("mxfp8",) * 2,
        hidden_size=64,
        head_dim=16,
        swa_head_dim=16,
        intermediate_size=64,
        n_routed_experts=32,
        num_experts_per_tok=4,
    )
    rng = np.random.default_rng(11)
    for layer in lm.model.layers:
        for proj in (layer.mlp.experts.w13, layer.mlp.experts.w2):
            if proj.mode == "nvfp4":
                scales = rng.integers(0x30, 0x41, proj.scales.shape)
                proj.scales = mx.array(scales.astype(np.uint8))
                amax = rng.uniform(20, 40, proj.global_scale.shape)
                proj.global_scale = mx.array(amax.astype(np.float32))
    gains = [
        (name, value + 1.0)
        for name, value in tree_flatten(lm.parameters())
        if name.endswith("norm.weight")
    ]
    lm.update(tree_unflatten(gains))
    mx.eval(lm.parameters())
    return lm


def _inkling2_state(cache):
    out = []
    for item in cache:
        kv = item[0]
        rows = [a[..., : kv.offset, :] for a in (kv.keys, kv.values)]
        out += [*rows, *[a for a in item[1].cache if a is not None]]
    return [np.asarray(a) for a in out]


@pytest.mark.parametrize("k,prefix", [(1, 0), (5, 0), (6, 0), (7, 0), (11, 0), (7, 1)])
def test_inkling2_split_matches_whole_model(k, prefix):
    # Chunks past the 8-key window; cuts before, at and after a block
    # boundary (6) and a full-attention layer (5).
    ids = mx.array(np.random.default_rng(5).integers(3, 58, (1, 30)))
    _split_matches(INKLING2, _inkling2, ids, [12, 10, 8], k, prefix, _inkling2_state)


@pytest.mark.parametrize(
    "factory,make,adapter,k,state_of",
    [
        (_tiny_glm5_next, _glm, GLM, 2, _glm_state),
        (_tiny_qwen4_exp, _qwen4, QWEN4, 1, _qwen4_state),
        (_tiny_qwen35, _qwen35, QWEN35, 4, _qwen35_state),
        (
            lambda: (_inkling2(), SimpleNamespace(model_type="inkling2")),
            _inkling2,
            INKLING2,
            7,
            _inkling2_state,
        ),
    ],
    ids=["glm", "qwen4", "qwen35", "inkling2"],
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


def _minimax(**overrides):
    apply_mlx_vlm_minimax_m3_compat_patch()
    from mlx_vlm.models.minimax_m3_vl.language import LanguageModel

    params = dict(
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
    args = _tiny_text_config(**{**params, **overrides})
    mx.random.seed(7)
    lm = LanguageModel(args)
    mx.eval(lm.parameters())
    return lm


def _minimax_contract():
    """The M3.1 layout: scaled routed experts with a routing bias, an index
    head per KV head, and one layer without a sparse index."""
    lm = _minimax(
        routed_expert_scales=True,
        use_routing_bias=True,
        moe_layer_freq=[0, 1, 1, 1],
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1, 0, 1, 1],
            "sparse_index_dim": 16,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 2,
            "sparse_topk_blocks": 3,
        },
    )
    # Norm gains, routing biases and expert scales (every 1-D parameter) start
    # at identity values; move them so the fixture pins their use.
    lm.apply(lambda p: p + mx.random.uniform(-0.3, 0.3, p.shape) if p.ndim == 1 else p)
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


@pytest.mark.parametrize(
    "make,model_type,adapter,k,layers",
    [
        (_minimax, "minimax_m3_vl", MINIMAX, 2, [0, 3]),
        # Inkling 2 taps the partial sum entering a layer, and the final norm
        # as layer 12.
        (_inkling2, "inkling2", INKLING2, 5, [3, 9, 12]),
    ],
    ids=["minimax", "inkling2"],
)
def test_scheduler_seeds_a_block_drafter_from_both_macs(
    make, model_type, adapter, k, layers
):
    tokens = [int(t) for t in np.random.default_rng(3).integers(3, 60, 100)]

    def drive(helper_k):
        host = _tiny_vlm_adapter(
            lambda: (make(), SimpleNamespace(model_type=model_type))
        )
        drafter = _Drafter(layers, window=40)
        host._language_model._omlx_drafter = drafter
        sched = _scheduler(host, 32)
        session = None
        if helper_k:
            helper = make()
            adapter.keep_first(helper, helper_k)
            session, listener, thread = _session(
                host._language_model, helper, helper_k, adapter=adapter
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
    # The first tap runs on the helper and the others here; the window starts
    # in the second chunk.
    assert [p for p, _ in split] == [p for p, _ in local] == [60, 64, 96]
    for (_, a), (_, b) in zip(local, split):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


# ---------------------------------------------------------------------------
# Contract fixtures: the arrays each family sends over the link, for helper
# builds that share no code with oMLX (apps/omlx-cuda-helper).
# ---------------------------------------------------------------------------

_CONTRACT_DIR = Path(__file__).parent / "fixtures" / "prefill_contract"
_CONTRACT_SIZES, _CONTRACT_PREFIX = [9, 6, 7], 1
# family -> (adapter, tiny model, layers on the helper, checkpoint name prefix)
_CONTRACT_CASES = {
    "qwen4_exp": (QWEN4, lambda: _qwen4(4), 3, "language_model."),
    "minimax_m3_vl": (MINIMAX, _minimax_contract, 3, "language_model."),
    "qwen3_5": (QWEN35, _qwen35_contract, 5, "language_model."),
    "glm5_next": (GLM, _glm_contract, 3, "language_model."),
    "deepseek_v41": (DSV41, _dsv41_contract, 6, "language_model."),
    "inkling2": (INKLING2, _inkling2_contract, 7, ""),
}


def _contract_case(family, weights=None):
    """One prompt through a split: a prefix on this Mac, then helper chunks.

    Returns the host model, the token ids and every array as it crosses the
    link: ``p:`` prefix state, ``in<chunk>:`` host-fed rows, ``b<chunk>:``
    boundary and ``s<chunk>:`` state.
    """
    adapter, make, k, _ = _CONTRACT_CASES[family]
    host, helper = make(), make()
    if weights is not None:
        for model in (host, helper):
            model.update(tree_unflatten(list(weights.items())))
    adapter.keep_first(helper, k)
    rng = np.random.default_rng(5)
    ids = mx.array(rng.integers(3, 58, (1, sum(_CONTRACT_SIZES))))
    chunks = _chunks(ids, _CONTRACT_SIZES)
    cache, helper_cache = host.make_cache(), helper.make_cache()
    for chunk in chunks[:_CONTRACT_PREFIX]:
        host(chunk, cache=cache)
    state = adapter.prefix_state(cache, k)
    adapter.load_prefix(helper_cache, state)
    arrays = {f"p:{n}": wire.to_host(v) for n, v in state.items()}
    rest = chunks[_CONTRACT_PREFIX:]
    tokens = np.concatenate([np.asarray(c[0]) for c in rest])
    feed = adapter.job_feed(host, cache, tokens, [c.shape[1] for c in rest], k)
    for index, chunk in enumerate(rest):
        inputs = feed(index) if feed else {}
        arrays.update({f"in{index}:{n}": v for n, v in inputs.items()})
        boundary, state = adapter.run_first(helper, helper_cache, chunk, 0, inputs)
        arrays.update({f"b{index}:{n}": wire.to_host(v) for n, v in boundary.items()})
        arrays.update({f"s{index}:{n}": wire.to_host(v) for n, v in state.items()})
        # DeepSeek runs its layers through the prefill entry point.
        prefill = getattr(host, "_omlx_prefill", None)
        adapter.run_rest(prefill, host, cache, chunk, k, boundary, state)
        adapter.apply_state(cache, k, state)
        mx.eval([c.state for c in cache])
    return host, np.asarray(ids), arrays


def _contract_manifest(arrays):
    return [[n, d, a.dtype.str, list(a.shape)] for n, (a, d) in arrays.items()]


@pytest.mark.parametrize("family", sorted(_CONTRACT_CASES))
def test_payload_matches_the_contract_fixture(family):
    folder = _CONTRACT_DIR / family
    meta = json.loads((folder / "meta.json").read_text())
    adapter, _, _, prefix = _CONTRACT_CASES[family]
    hint = (
        f"the {family} payload changed: raise {type(adapter).__name__}.contract "
        "and run tests/fixtures/record_prefill_contract.py"
    )
    assert adapter.contract == meta["contract"], hint
    stored = mx.load(str(folder / "model" / "model.safetensors"))
    weights = {name.removeprefix(prefix): value for name, value in stored.items()}
    _, _, arrays = _contract_case(family, weights)
    assert _contract_manifest(arrays) == meta["manifest"], hint
    expected = np.load(folder / "case.npz")
    for name, (data, _) in arrays.items():
        if data.dtype.kind == "f":
            np.testing.assert_allclose(
                data, expected[name], rtol=meta["rtol"], atol=meta["atol"], err_msg=hint
            )
        else:
            np.testing.assert_array_equal(data, expected[name], err_msg=hint)


def test_inkling2_helper_loads_only_its_layers(tmp_path):
    source = _CONTRACT_DIR / "inkling2" / "model"
    weights = mx.load(str(source / "model.safetensors"))
    shutil.copy(source / "config.json", tmp_path / "config.json")
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    lm = INKLING2.load_first(str(tmp_path), 3)
    assert len(lm.model.layers) == 3 and "lm_head" not in lm
    host = _inkling2_contract()
    host.update(tree_unflatten(list(weights.items())))
    ids = mx.array([[5, 9, 11, 3, 7]])
    boundary, _ = INKLING2.run_first(lm, lm.make_cache(), ids, 0)
    INKLING2.keep_first(host, 3)
    expected, _ = INKLING2.run_first(host, host.make_cache(), ids, 0)
    np.testing.assert_array_equal(np.asarray(boundary["h"]), np.asarray(expected["h"]))
    # Without a tensor of layer 1 a non-strict load would leave it at zero.
    del weights["model.layers.1.mlp_norm.weight"]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    assert len(INKLING2.load_first(str(tmp_path), 1).model.layers) == 1
    with pytest.raises(ValueError, match="not in the checkpoint"):
        INKLING2.load_first(str(tmp_path), 2)


def test_qwen35_helper_refuses_a_checkpoint_it_cannot_load(tmp_path, monkeypatch):
    # The server's prefill patches are process-wide; xdist runs other files next.
    for name in (
        "apply_sdpa256_attention_patch",
        "apply_qwen35_fa256_attention_patch",
        "apply_qwen35_gdn_prework_patch",
        "apply_qwen35_gdn_prefill_patch",
        "apply_qwen35_q4_mlp_patch",
        "apply_qwen35_q4_prefill_linear_patch",
    ):
        monkeypatch.setattr(
            f"omlx.prefill_helper.adapters.qwen3_5.{name}", lambda *_: None
        )
    source = _CONTRACT_DIR / "qwen3_5" / "model"
    weights = mx.load(str(source / "model.safetensors"))
    config = json.loads((source / "config.json").read_text())
    # Without a tensor of layer 1 a non-strict load would leave it random.
    del weights["language_model.model.layers.1.mlp.up_proj.weight"]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert len(QWEN35.load_first(str(tmp_path), 1).model.layers) == 1
    with pytest.raises(ValueError, match="not in the checkpoint"):
        QWEN35.load_first(str(tmp_path), 2)
    # The server loads a ParoQuant checkpoint with a loader of its own.
    config["quantization_config"] = {"quant_method": "paroquant"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="own loader"):
        QWEN35.load_first(str(tmp_path), 1)
