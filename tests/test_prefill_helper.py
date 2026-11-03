"""Prefill Accelerator: a split prefill must match the whole-model prefill."""

import json
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import _ENGRAM, load_reference_weights, tiny, tiny_ced

from omlx.patches.deepseek_v41.language import LanguageModel
from omlx.prefill_helper import launcher, wire
from omlx.prefill_helper.adapters import adapter_for
from omlx.prefill_helper.client import HelperClient, HelperError
from omlx.prefill_helper.launcher import HelperTarget
from omlx.prefill_helper.session import PrefillHelperSession
from omlx.prefill_helper.worker import WINDOW, HelperServer
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig

DSV41 = adapter_for("deepseek_v41")


def _model(config):
    model = LanguageModel(config)
    load_reference_weights(model)
    if config.engram_layer_ids:
        model.set_token_map(np.arange(config.vocab_size))
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
    for chunk, nxt in zip(chunks[prefix:], nexts[prefix:]):
        boundary, state = DSV41.run_first(helper, helper_cache, chunk, nxt)
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
    job = client.submit(np.asarray(ids[0, :-1]), sizes)
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


def _session(lm, k, **kwargs):
    listener, thread = _serve(lm, k)
    client = HelperClient("127.0.0.1", listener.getsockname()[1], TOKEN)
    session = PrefillHelperSession(
        client,
        DSV41,
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


@pytest.mark.parametrize("config", [tiny(), tiny_ced()], ids=["plain", "ced"])
def test_scheduler_helper_prefill_matches_local(config):
    k, tokens = 2, [int(t) for t in np.random.default_rng(3).integers(3, 60, 23)]
    host = _model(config)
    local = _drive(_scheduler(host, 6), _request(tokens), tokens)

    sched = _scheduler(host, 6)
    session, listener, thread = _session(_helper_model(config, k), k)
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
    session, listener, thread = _session(_helper_model(config, k), k)
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
    sched = _scheduler(_model(tiny()), 4)
    session, listener, thread = _session(_helper_model(tiny(), k), k)
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
    session = PrefillHelperSession(client, DSV41, 18, "helper-mac", work)
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
    job = client.submit(np.asarray(ids[0, :-1]), sizes, layers=k)
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
