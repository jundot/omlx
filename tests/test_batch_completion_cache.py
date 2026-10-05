# SPDX-License-Identifier: Apache-2.0
"""Compare terminal cohort handoff with the native response/filter path."""

import gc
import importlib
import importlib.util
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.generate import GenerationBatch
from mlx_lm.models import cache as lm_cache
from mlx_vlm.models import cache as vlm_cache

import omlx.scheduler  # noqa: F401
from omlx.patches.batch_completion_cache import defer_terminal_cache_extraction


@pytest.fixture(scope="module")
def native_batch():
    module = importlib.import_module("mlx_lm.generate")
    spec = importlib.util.spec_from_file_location(
        "mlx_lm._native_generate", module.__file__
    )
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    return native.GenerationBatch


def make_layer(module, lengths, layer):
    rows = []
    for idx, length in enumerate(lengths):
        row = module.KVCache()
        x = (
            mx.arange(2 * length * 8).reshape(1, 2, length, 8) % 29 + idx + layer
        ).astype(mx.bfloat16)
        row.update_and_fetch(x, x / 4)
        rows.append(row)
    cache = module.BatchKVCache.merge(rows)
    mx.eval(cache.keys, cache.values)
    return cache


def make_batch(cls, module, finished, *, stop=False, lengths=None, layers=2):
    batch = object.__new__(cls)
    batch.model = SimpleNamespace()
    batch.uids = list(range(6))
    batch.prompt_cache = [
        make_layer(module, lengths or [513, 509, 507, 512, 510, 505], i)
        for i in range(layers)
    ]
    batch.tokens = [[idx] for idx in range(6)]
    batch.samplers = [None] * 6
    batch.logits_processors = [[] for _ in range(6)]
    batch.max_tokens = [1 if idx in finished and not stop else 3 for idx in range(6)]
    batch.stop_sequences = [None] * 6
    batch._num_tokens = [0] * 6
    batch._matchers = [
        SimpleNamespace(advance=lambda token, done=idx in finished and stop: done)
        for idx in range(6)
    ]
    batch._next_tokens = mx.arange(6, dtype=mx.uint32)
    batch._next_logprobs = [mx.array([float(idx)]) for idx in range(6)]
    batch._token_context = list(range(6))
    batch._step = lambda: (list(range(6)), batch._next_logprobs)
    return batch


def equal_state(a, b):
    for x, y in zip(a.state, b.state):
        if isinstance(x, mx.array):
            assert x.dtype == y.dtype and x.shape == y.shape
            assert mx.array_equal(x, y).item()
        else:
            assert x == y


@pytest.mark.parametrize("module", [lm_cache, vlm_cache], ids=["lm", "vlm"])
@pytest.mark.parametrize(
    "finished", [[], [0], [1, 4], [0, 1, 2], [0, 1, 2, 3, 4], list(range(6))]
)
@pytest.mark.parametrize("stop", [False, True], ids=["length", "stop"])
def test_terminal_cohorts_preserve_responses_and_survivors(
    native_batch, module, finished, stop
):
    before = make_batch(native_batch, module, finished, stop=stop)
    after = make_batch(GenerationBatch, module, finished, stop=stop)
    expected = before.next()
    actual = defer_terminal_cache_extraction(native_batch.next)(after)
    for a, b in zip(expected, actual):
        assert (a.uid, a.token, a.finish_reason, a.all_tokens) == (
            b.uid,
            b.token,
            b.finish_reason,
            b.all_tokens,
        )
        assert mx.array_equal(a.logprobs, b.logprobs).item()
        if a.prompt_cache is None:
            assert b.prompt_cache is None
        else:
            assert len(a.prompt_cache) == len(b.prompt_cache)
            for x, y in zip(a.prompt_cache, b.prompt_cache):
                equal_state(x, y)
    for field in ("uids", "tokens", "max_tokens", "_num_tokens", "_token_context"):
        assert getattr(before, field) == getattr(after, field)
    assert len(before.prompt_cache) == len(after.prompt_cache)
    for x, y in zip(before.prompt_cache, after.prompt_cache):
        equal_state(x, y)
    assert not hasattr(after, "_omlx_terminal_caches")
    # Real activation/rollback extraction outside response emission stays eager.
    if after.uids:
        assert len(after.extract_cache(0)) == 2


@pytest.mark.parametrize("module", [lm_cache, vlm_cache], ids=["lm", "vlm"])
@pytest.mark.parametrize("count", [3, 6])
def test_simultaneous_completion_does_not_duplicate_all_cache_layers(
    native_batch, module, count
):
    gc.collect()
    batch = make_batch(
        GenerationBatch, module, list(range(count)), lengths=[65536] * 6, layers=8
    )
    layer_bytes = (
        batch.prompt_cache[0].keys.nbytes + batch.prompt_cache[0].values.nbytes
    )
    gc.collect()
    mx.reset_peak_memory()
    before = mx.get_active_memory()
    responses = defer_terminal_cache_extraction(native_batch.next)(batch)
    assert sum(r.finish_reason is not None for r in responses) == count
    # Extra live storage must be bounded by a layer, not the terminal cohort
    # across every layer. This fails on the row-at-a-time native handoff.
    assert mx.get_peak_memory() - before < 2 * layer_bytes


def test_failed_emission_does_not_leave_extraction_deferred():
    batch = make_batch(GenerationBatch, lm_cache, [0])

    @defer_terminal_cache_extraction
    def fail(batch):
        batch.extract_cache(0)
        raise ValueError("emission failed")

    with pytest.raises(ValueError, match="emission failed"):
        fail(batch)
    assert not hasattr(batch, "_omlx_terminal_caches")
    assert len(batch.extract_cache(0)) == 2
