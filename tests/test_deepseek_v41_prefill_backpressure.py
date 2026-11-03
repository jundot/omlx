# SPDX-License-Identifier: MIT
"""Regression guards for the DeepSeek V4.1 prefill memory diagnosis
(2026-09-21). Observed: each 2048-token prefill chunk pinned ~14.5GB of
GPU-side residency (footprint sawtooth 89->102GB, flat across chunks,
100% IOAccelerator per `footprint -d` at peak vs trough) because the
40-layer vendor loop enqueues the whole chunk lazily while the GPU runs
~15s behind, and the Metal allocator caches every freed size class in the
pool. The root causes the GLM-5.x fix (0fe20e20) addressed and confirmed
here for v41: (C) no per-layer eval backpressure in the vendor loop and
(D) per-size-class pool hoarding, plus (guard) `make_prefill_memory_profile`
explicitly excluded deepseek_v41 so the throttle priced chunks from the
footprint-delta EWMA alone (it predicted 2.81GB for a chunk that grew the
footprint past the throttle target).

Root cause A (patch registration defeated by upstream merge + import-once)
was verified NOT present for v41, but apply_patch relied on a silent
``sys.modules.setdefault``; these tests lock in vendor-resolution
self-verification so a future mlx-vlm release shipping a competing
``deepseek_v41`` package cannot silently disable the vendor tree again.
"""

import logging
import sys
import types

import mlx.core as mx
from test_deepseek_v41 import load_reference_weights, tiny

import omlx.patches.deepseek_v41.language as v41_lang
from omlx.memory_monitor import MemoryMonitor, make_prefill_memory_profile
from omlx.patches.deepseek_v41 import apply_patch
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.scheduler import Scheduler

_TEST_CACHE_LIMIT = 3 << 20


def _current_cache_limit():
    previous = mx.set_cache_limit(0)
    mx.set_cache_limit(previous)
    return previous


def test_v41_prefill_evals_stream_per_layer(monkeypatch):
    """A lazy 40-layer prefill loop pins every intermediate until the logits
    eval (the measured +14.5GB/chunk sawtooth). Prefill must eval the stream
    per layer under one bounded pool, without a per-layer clear; narrower
    calls stay lazy and leave the limit alone."""
    model = v41_lang.LanguageModel(tiny())
    n_layers = model._config.n_layers
    before = _current_cache_limit()
    events = {"evals": 0, "clears": 0, "limits": []}
    real_eval, real_clear, real_limit = mx.eval, mx.clear_cache, mx.set_cache_limit

    def eval_spy(*args, **kw):
        events["evals"] += 1
        return real_eval(*args, **kw)

    def clear_spy(*args, **kw):
        events["clears"] += 1
        return real_clear(*args, **kw)

    def limit_spy(limit):
        events["limits"].append(limit)
        return real_limit(limit)

    monkeypatch.setattr(v41_lang.mx, "eval", eval_spy)
    monkeypatch.setattr(v41_lang.mx, "clear_cache", clear_spy)
    monkeypatch.setattr(v41_lang.mx, "set_cache_limit", limit_spy)
    monkeypatch.setattr(v41_lang, "_prefill_cache_limit", lambda: _TEST_CACHE_LIMIT)

    real_eval(model(mx.zeros((1, 300), dtype=mx.int32)))  # >= 256 prefill gate
    assert events["evals"] >= n_layers, (
        f"prefill width must eval the stream per layer, got {events['evals']}"
        f" eval calls for {n_layers} layers"
    )
    assert events["clears"] == 0, "prefill must not clear the pool per layer"
    assert events["limits"] == [_TEST_CACHE_LIMIT, before]
    assert _current_cache_limit() == before

    events.update(evals=0, limits=[])
    narrow = mx.zeros((1, 255), dtype=mx.int32)
    real_eval(model(narrow, cache=model.make_cache()))
    assert events["evals"] < n_layers, "narrow width must stay lazy"
    assert events["clears"] == 0
    assert events["limits"] == []


def test_prefill_cache_scope_restore_survives_non_lifo_interleaving(monkeypatch):
    """``mx.set_cache_limit`` is process-global with no per-thread scoping,
    and each engine's forward runs on its own executor thread, so two prefill
    scopes can exit in either order. Per-scope save/restore pinned the limit
    at the 512 MiB-class prefill bound whenever the first-entered scope exited
    first, instead of the server's startup total (the enforcer and engine_pool
    rely on the limit staying pinned there). Only the outermost scope may
    restore, and it must restore the pre-entry limit. Two engines' scopes are
    replayed on this thread as a deterministic non-LIFO enter/exit sequence.
    """
    before = _current_cache_limit()
    calls = []
    current = [before]

    def fake_set_cache_limit(limit):
        calls.append(limit)
        previous = current[0]
        current[0] = limit
        return previous

    monkeypatch.setattr(v41_lang.mx, "set_cache_limit", fake_set_cache_limit)
    monkeypatch.setattr(v41_lang, "_prefill_cache_limit", lambda: _TEST_CACHE_LIMIT)

    scope_a = v41_lang._prefill_cache_scope()
    scope_b = v41_lang._prefill_cache_scope()
    opened = []
    try:
        scope_a.__enter__()
        opened.append(scope_a)
        scope_b.__enter__()
        opened.append(scope_b)
        assert current[0] == _TEST_CACHE_LIMIT
        # Non-LIFO: the first-entered scope exits while B is still running.
        scope_a.__exit__(None, None, None)
        opened.remove(scope_a)
        assert current[0] == _TEST_CACHE_LIMIT, (
            "the still-open scope must keep the bound"
        )
        scope_b.__exit__(None, None, None)
        opened.remove(scope_b)
        assert current[0] == before, "the last exit must restore the pre-entry limit"
        assert calls == [_TEST_CACHE_LIMIT, _TEST_CACHE_LIMIT, before]
    finally:
        for scope in opened:
            scope.__exit__(None, None, None)

    # Nested same-thread scopes still apply the bound and restore exactly once.
    calls.clear()
    with v41_lang._prefill_cache_scope():
        assert current[0] == _TEST_CACHE_LIMIT
        with v41_lang._prefill_cache_scope():
            assert current[0] == _TEST_CACHE_LIMIT
        assert current[0] == _TEST_CACHE_LIMIT, "inner exit must not restore early"
    assert current[0] == before
    assert calls == [_TEST_CACHE_LIMIT, _TEST_CACHE_LIMIT, before]


def test_apply_patch_replaces_foreign_vendor_alias_and_warns(caplog):
    """apply_patch used ``sys.modules.setdefault``, which silently keeps a
    pre-existing ``mlx_vlm.models.deepseek_v41`` (the exact failure mode that
    made three rounds of GLM fixes dead code after upstream merged the
    fork). An alias that is not the vendor tree must be replaced and the
    event logged at least at WARNING level."""
    pkg = "mlx_vlm.models.deepseek_v41"
    import omlx.patches.deepseek_v41.model as vendor_model

    saved = sys.modules.get(pkg)
    foreign = types.ModuleType(pkg)
    foreign.__file__ = "/fake/site-packages/mlx_vlm/models/deepseek_v41/model.py"
    sys.modules[pkg] = foreign
    try:
        with caplog.at_level(logging.WARNING):
            apply_patch()
        assert sys.modules[pkg] is vendor_model, (
            "apply_patch must install the vendor model module over a foreign alias"
        )
        warned = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "deepseek_v41" in r.getMessage()
        ]
        assert warned, f"foreign alias replacement not logged: {caplog.records}"
    finally:
        if saved is None:
            sys.modules.pop(pkg, None)
        else:
            sys.modules[pkg] = saved


def _v41_release_text_dict():
    """HF text_config keys of the released DeepSeek-V4.1-Flash shape (the
    subset the memory profile consumes)."""
    ratios = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
    return {
        "vocab_size": 129280,
        "hidden_size": 5120,
        "moe_intermediate_size": 2304,
        "num_hidden_layers": 40,
        "num_attention_heads": 64,
        "head_dim": 512,
        "q_lora_rank": 1280,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "sliding_window": 128,
        "compress_ratios": ratios,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "n_routed_experts": 384,
        "num_experts_per_tok": 6,
    }


def test_set_model_info_wires_v41_profile_through_scheduler():
    """The scheduler probes configs with HF names; the v41 runtime config is
    the vendor ModelConfig (n_layers / dim / n_heads). Before the alias
    widening, every probe returned None for v41, the whole model-info block
    was skipped, and the guard priced chunks from the footprint-delta EWMA
    alone. The vendor naming must resolve and land a static prefill profile
    on the monitor with flat-overhead accounting enabled."""
    from omlx.patches.deepseek_v41.model import Model as V41Model

    config = ModelConfig.from_dict(
        {"model_type": "deepseek_v41", "text_config": _v41_release_text_dict()}
    )
    model = V41Model(config)

    monitor = MemoryMonitor(max_kv_cache_memory=8 * 1024**3, eviction_enabled=False)
    ns = Scheduler.__new__(Scheduler)
    ns.model = model
    ns.memory_monitor = monitor
    # The TQ/MLA probes are methods called inside the model-info block.
    ns._model_uses_attention_sinks = lambda: False
    ns._model_uses_mla = lambda: True
    ns._turboquant_eligible = lambda cache_list: False
    ns._turboquant_kv_bits = 0
    ns._turboquant_skip_last = False
    ns._fixed_state_measure_armed = False

    Scheduler._set_model_info_for_monitor(ns)

    profile = monitor._prefill_memory_profile
    assert profile is not None, (
        "v41 vendor-named config must resolve through the scheduler probes and "
        "register a static prefill profile"
    )
    assert monitor.uses_flat_overhead_accounting() is True
    # 40 backbone layers reach the monitor (vendor n_layers alias).
    assert ns.memory_monitor is monitor
    estimated = profile.estimate_resident_kv_bytes(8192)
    assert estimated > 0


def test_v41_resident_estimate_matches_stored_cache_bytes():
    config = tiny(index_head_dim=64)
    model = v41_lang.LanguageModel(config)
    load_reference_weights(model)
    cache = model.make_cache()
    tokens = mx.arange(16)[None, :]
    output = model(tokens, cache=cache)
    mx.eval(output, [item.state for item in cache])

    profile = make_prefill_memory_profile(config, compute_dtype_size=4)
    stored_bytes = sum(item[2].nbytes + item[3].nbytes for item in cache)
    assert stored_bytes > 0
    assert profile.estimate_resident_kv_bytes(tokens.shape[1]) == stored_bytes
