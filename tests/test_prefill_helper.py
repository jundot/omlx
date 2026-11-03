"""Prefill Accelerator: a split prefill must match the whole-model prefill."""

import socket
import threading
import time

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import _ENGRAM, load_reference_weights, tiny, tiny_ced

from omlx.patches.deepseek_v41.language import LanguageModel
from omlx.prefill_helper import adapter_for, wire
from omlx.prefill_helper.client import HelperClient, HelperError
from omlx.prefill_helper.worker import WINDOW, HelperServer

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
