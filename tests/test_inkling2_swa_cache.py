# SPDX-License-Identifier: Apache-2.0
"""Inkling 2's opt-in sliding-window ring (OMLX_INKLING_SWA_CACHE=1).

A sliding-window visit on a RotatingKVCache must attend exactly as on a full
KVCache: through chunked prefill, the ring wrapping, single-token decode and
speculative-verify rollback before, across and after the wrap.
"""

import logging

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import (
    ArraysCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)
from test_inkling_vlm_mtp import _inkling2_tiny

from omlx.cache.type_registry import CacheTypeRegistry
from omlx.patches.inkling2 import kernels
from omlx.patches.inkling2 import model as inkling2_model
from omlx.patches.inkling2.model import _temporal_kv, _trim_kv

# The real model's head size; a short window so the ring wraps quickly.
WINDOW, HQ, HKV, D = 128, 4, 2, 128


def _fwd(n):
    return ("fwd", n)


def _trim(n):
    return ("trim", n)


# Prefill blocks of 32 or more queries take the flash kernel, verify chunks of
# a few the masked one, and single tokens the decode one.
_PLANS = {
    "prefill-wrap-prefill": [_fwd(256), _fwd(256)]
    + [_fwd(1)] * 190
    + [_fwd(90)]
    + [_fwd(1)] * 40,
    "verify-before-wrap": [_fwd(40), _fwd(4), _trim(2), _fwd(1), _fwd(4), _trim(3)]
    + [_fwd(1)] * 100
    + [_fwd(4), _trim(1), _fwd(1)],
    "verify-after-wrap": [_fwd(140)]
    + [_fwd(1)] * 300
    + [_fwd(4), _trim(1), _fwd(1), _fwd(4), _trim(3), _fwd(1), _fwd(5), _trim(0)] * 5
    + [_fwd(1)] * 20,
    # A verify chunk that crosses max_size from a linear buffer, then has
    # most of itself rejected: the rollback must return to the growth phase.
    "verify-crosses-window": [_fwd(WINDOW - 2), _fwd(5), _trim(4)]
    + [_fwd(1)] * 40
    + [_fwd(4), _trim(3), _fwd(1)],
    "verify-crosses-2x-window": [_fwd(2 * WINDOW - 2), _fwd(5), _trim(4)]
    + [_fwd(1)] * 40
    + [_fwd(4), _trim(3), _fwd(1)],
}
_RINGS = {
    "ring": lambda: RotatingKVCache(max_size=WINDOW, keep=0),
    "ring-2x": lambda: RotatingKVCache(max_size=2 * WINDOW, keep=0),
    "batch-ring": lambda: BatchRotatingKVCache(max_size=WINDOW, left_padding=[0]),
}


def _inputs(length):
    q = mx.random.normal((1, HQ, length, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, HKV, length, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, HKV, length, D)).astype(mx.bfloat16)
    rel = (0.1 * mx.random.normal((1, length, HQ, WINDOW))).astype(mx.bfloat16)
    sink = mx.random.normal((1, length, HQ)).astype(mx.bfloat16)
    return q, k, v, rel, sink


def _run(cache, q, k, v, rel, sink):
    k, v = cache.update_and_fetch(k, v)
    k, v = _temporal_kv(cache, k, v, WINDOW, q.shape[2])
    return kernels.banded_sink_attention(
        q,
        k,
        v,
        rel,
        sink,
        scale=D**-0.5,
        window=WINDOW,
        left_padding=getattr(cache, "left_padding", None),
    )


@pytest.mark.parametrize("ring", sorted(_RINGS))
@pytest.mark.parametrize("plan", sorted(_PLANS))
def test_ring_attends_as_the_full_cache(plan, ring):
    mx.random.seed(0)
    full, rotating = KVCache(), _RINGS[ring]()
    for step, (op, n) in enumerate(_PLANS[plan]):
        if op == "trim":
            assert _trim_kv(rotating, n) == full.trim(n), step
            continue
        inputs = _inputs(n)
        expected, actual = _run(full, *inputs), _run(rotating, *inputs)
        err = mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32)).max()
        # bf16 rounding only: a wrong or missing key moves outputs by O(1).
        assert err.item() <= 1e-2, (step, op, n, full.offset)
    # The ring stayed bounded: at most what the concat path keeps.
    chunk = max(n for op, n in _PLANS[plan] if op == "fwd")
    assert rotating.keys.shape[2] <= rotating.max_size - 1 + chunk


def test_trim_refuses_to_drop_history_the_window_needs():
    cache = RotatingKVCache(max_size=WINDOW, keep=0)
    for _ in range(WINDOW + 10):
        cache.update_and_fetch(*_inputs(1)[1:3])
    # A wrapped ring of exactly the window holds no spare keys.
    assert _trim_kv(cache, 1, dry_run=True) == 0
    assert _trim_kv(cache, 1) == 0
    assert cache.offset == WINDOW + 10


def test_rollback_refusal_leaves_every_layer_untouched():
    lm = _inkling2_tiny()
    # Layer 0 could roll back one rejected draft; layer 1 cannot.
    ok = RotatingKVCache(max_size=WINDOW, keep=0)
    ok.update_and_fetch(*_inputs(WINDOW + 20)[1:3])
    ok.update_and_fetch(*_inputs(4)[1:3])
    stuck = RotatingKVCache(max_size=WINDOW, keep=0)
    for _ in range(WINDOW + 10):
        stuck.update_and_fetch(*_inputs(1)[1:3])
    cache = []
    for kv in (ok, stuck):
        conv = ArraysCache(4)
        conv._omlx_verify_xp = {}
        cache.append((kv, conv))
    before = [(kv.offset, kv._idx, kv.keys.shape[2]) for kv, _ in cache]
    assert lm.mtp_partial_rollback(cache, 2, 3) is False
    assert [(kv.offset, kv._idx, kv.keys.shape[2]) for kv, _ in cache] == before
    assert all(conv._omlx_verify_xp == {} for _, conv in cache)


@pytest.mark.parametrize(
    "on,size,expected,warnings",
    [
        (None, None, (False, 2048), 0),
        ("1", "0", (True, 0), 0),
        ("1", " 1024 ", (True, 1024), 0),
        ("true", None, (False, 2048), 1),
        ("1", "2k", (True, 2048), 1),
    ],
)
def test_env_settings(monkeypatch, caplog, on, size, expected, warnings):
    for name, value in (
        ("OMLX_INKLING_SWA_CACHE", on),
        ("OMLX_INKLING_SWA_CACHE_SIZE", size),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with caplog.at_level(logging.WARNING, logger=inkling2_model.__name__):
        assert inkling2_model._env_swa() == expected
    ours = [r for r in caplog.records if r.name == inkling2_model.__name__]
    assert len(ours) == warnings


@pytest.mark.parametrize("bounded,size", [(False, 0), (True, 0), (True, 2048)])
def test_make_cache_layout(monkeypatch, bounded, size):
    monkeypatch.setattr(inkling2_model, "_SWA_BOUNDED", bounded)
    monkeypatch.setattr(inkling2_model, "_SWA_SIZE", size)
    lm = _inkling2_tiny()
    cache = [entry[0] for entry in lm.make_cache()]
    kinds = [type(kv) for kv in cache]
    if not bounded:
        assert set(kinds) == {KVCache}
        return
    # Full attention at layers 5 and 11 of the tiny stack; the rest slide on
    # a ring of the window (8 keys) or the larger size asked for.
    assert kinds[5] is kinds[11] is KVCache
    assert kinds.count(RotatingKVCache) == len(kinds) - 2
    rings = {kv.max_size for kv in cache if isinstance(kv, RotatingKVCache)}
    assert rings == {max(8, size)}


def test_ring_model_matches_the_full_cache_model(monkeypatch):
    lm = _inkling2_tiny()
    mx.eval(lm.parameters())
    ids = mx.array(np.random.default_rng(7).integers(3, 58, (1, 40)))

    def logits(bounded):
        monkeypatch.setattr(inkling2_model, "_SWA_BOUNDED", bounded)
        monkeypatch.setattr(inkling2_model, "_SWA_SIZE", 0)
        cache = lm.make_cache()
        out = []
        # Chunks past the 8-key window, then decode on the wrapped ring.
        bounds = [0, 12, 30, *range(31, 41)]
        for start, end in zip(bounds, bounds[1:]):
            step = lm(ids[:, start:end], cache=cache)
            out.append(np.asarray(getattr(step, "logits", step)[:, -1]))
        return out

    for expected, actual in zip(logits(False), logits(True)):
        np.testing.assert_array_equal(expected, actual)


def _ssd_round_trip(cache):
    """Rebuild every entry through oMLX's cache handlers, as an SSD hit does."""
    out = []
    for entry in cache:
        handler = CacheTypeRegistry.get_handler_for_object(entry)
        out.append(
            handler.deserialize_state(
                handler.serialize_state(entry), handler.serialize_meta_state(entry)
            )
        )
    return out


@pytest.mark.parametrize("bounded", [False, True])
def test_cache_survives_an_ssd_round_trip(monkeypatch, bounded):
    monkeypatch.setattr(inkling2_model, "_SWA_BOUNDED", bounded)
    monkeypatch.setattr(inkling2_model, "_SWA_SIZE", 0)
    lm = _inkling2_tiny()
    mx.eval(lm.parameters())
    ids = mx.array(np.random.default_rng(11).integers(3, 58, (1, 40)))
    caches = {}
    for name in ("cold", "restored"):
        cache = lm.make_cache()
        for start, end in ((0, 12), (12, 24)):  # the 8-key ring has wrapped
            lm(ids[:, start:end], cache=cache)
        caches[name] = cache
    caches["restored"] = _ssd_round_trip(caches["restored"])
    ring = caches["restored"][0][0]
    assert isinstance(ring, RotatingKVCache) is bounded
    assert ring.offset == 24
    out = {name: [] for name in caches}
    bounds = [24, 32, *range(33, 41)]  # a prefill chunk, then decode
    for start, end in zip(bounds, bounds[1:]):
        for name, cache in caches.items():
            step = lm(ids[:, start:end], cache=cache)
            out[name].append(np.asarray(getattr(step, "logits", step)[:, -1]))
    for expected, actual in zip(out["cold"], out["restored"]):
        np.testing.assert_array_equal(expected, actual)
