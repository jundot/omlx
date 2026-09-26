# SPDX-License-Identifier: Apache-2.0
"""Verify bounded SDPA256 routing and matching prefill memory estimates."""

import logging

import mlx.core as mx
import pytest

from omlx import memory_monitor as mm
from omlx.memory_monitor import MemoryMonitor, estimate_unfused_sdpa_call_bytes
from omlx.patches import sdpa256_attention as sdpa256
from omlx.scheduler import Scheduler

GIB = 1024**3
HEAD_DIM = sdpa256.HEAD_DIM
# A GQA long-context geometry. These are inputs to the estimator, not
# assumptions baked into the code under test.
N_Q, N_KV, N_LAYERS = 24, 4, 16


@pytest.fixture(autouse=True)
def _clean_route_state():
    saved = dict(mm._SDPA_TILED_PREFILL_HEAD_DIMS)
    force, logged = sdpa256._FORCE_TILED, set(sdpa256._TILED_ROUTE_LOGGED)
    mm._SDPA_TILED_PREFILL_HEAD_DIMS.clear()
    sdpa256._FORCE_TILED = None
    sdpa256._TILED_ROUTE_LOGGED.clear()
    yield
    mm._SDPA_TILED_PREFILL_HEAD_DIMS.clear()
    mm._SDPA_TILED_PREFILL_HEAD_DIMS.update(saved)
    sdpa256._FORCE_TILED = force
    sdpa256._TILED_ROUTE_LOGGED.clear()
    sdpa256._TILED_ROUTE_LOGGED.update(logged)


def _qk(q_len, kv_len, n_q=N_Q, n_kv=N_KV, head_dim=HEAD_DIM):
    q = mx.zeros((1, n_q, q_len, head_dim), dtype=mx.float16)
    k = mx.zeros((1, n_kv, kv_len, head_dim), dtype=mx.float16)
    return q, k


def _monitor(head_dim=HEAD_DIM):
    m = MemoryMonitor(max_kv_cache_memory=GIB)
    m.set_model_info(
        num_layers=N_LAYERS,
        num_kv_heads=N_KV,
        head_dim=head_dim,
        dtype_size=2,
        num_attention_heads=N_Q,
    )
    return m


# --- 1. qualifying prefill is always bounded -------------------------------


@pytest.mark.parametrize("kv_len", [8192, 28672, 65536, 262144])
@pytest.mark.parametrize("q_len", [16, 2048, 4096])
def test_qualifying_prefill_is_always_bounded(q_len, kv_len):
    q, k = _qk(q_len, kv_len)
    assert sdpa256._should_route(q, k, None, "causal", None) is True


def test_route_is_identical_across_repeated_calls():
    """No process history: the same shape answers the same way every time,
    including after other shapes have gone through the gate."""
    answers = []
    for _ in range(3):
        answers.append(
            sdpa256._should_route(*_qk(4096, 28672), None, "causal", None)
        )
        sdpa256._should_route(*_qk(2048, 262144), None, "causal", None)
    assert answers == [True, True, True]


# --- 2/3. no live-memory input, no process-history input -------------------


def test_scheduler_no_longer_exposes_a_route_headroom_provider():
    for attr in ("_sdpa256_unfused_headroom", "_sdpa256_bounded_route_changed"):
        assert not hasattr(Scheduler, attr), f"{attr} should be gone"
    assert not hasattr(sdpa256, "set_unfused_headroom_provider")
    assert not hasattr(sdpa256, "_get_unfused_headroom_provider")


# --- 4. the #2025 allocation is never selected -----------------------------


def test_2025_dense_allocation_is_never_selected_for_qualifying_shapes():
    """At the context where #2025 hit its wall the unfused matrix is tens of
    GiB. The gate must not be able to choose it, whatever the machine."""
    q_len, kv_len = 2048, 240 * 1024
    unfused = estimate_unfused_sdpa_call_bytes(
        N_Q, q_len, kv_len, HEAD_DIM, mm.SDPA256_UNFUSED_SCORE_DTYPE_SIZE
    )
    assert unfused > 20 * GIB  # the allocation #2025 was about
    assert sdpa256._should_route(*_qk(q_len, kv_len), None, "causal", None) is True


# --- 5. estimator agrees with the route ------------------------------------


def test_estimator_prices_the_route_that_will_run():
    assert sdpa256._register_bounded_route(sdpa256._SDPA256_MIN_KV_LEN)
    monitor = _monitor()
    q_len, kv_len = 4096, 28672
    assert sdpa256._should_route(*_qk(q_len, kv_len), None, "causal", None) is True
    charged = monitor.estimate_chunk_transient_bytes(q_len, kv_len)
    unfused = estimate_unfused_sdpa_call_bytes(
        N_Q, q_len, kv_len, HEAD_DIM, mm.SDPA256_UNFUSED_SCORE_DTYPE_SIZE
    )
    assert charged < unfused / 10


def test_estimator_stays_o_l_across_the_whole_window():
    """#2025's symptom was the guard rejecting long prompts because the
    estimate grew O(L^2). It must not grow with context at all."""
    assert sdpa256._register_bounded_route(sdpa256._SDPA256_MIN_KV_LEN)
    monitor = _monitor()
    at_32k = monitor.estimate_chunk_transient_bytes(4096, 32768)
    at_256k = monitor.estimate_chunk_transient_bytes(4096, 262144)
    assert at_256k == at_32k


# --- 6. VLM / Qwen4 profile agrees too -------------------------------------


def test_vlm_prefill_profile_prices_the_bounded_route():
    """The Qwen4 QSA profile keeps its own copy of the bounded-route lookup;
    it must reach the same conclusion as the generic estimator."""
    from types import SimpleNamespace

    from omlx.memory_monitor import make_prefill_memory_profile

    assert sdpa256._register_bounded_route(sdpa256._SDPA256_MIN_KV_LEN)
    config = SimpleNamespace(
        model_type="qwen4_exp",
        num_hidden_layers=48,
        num_attention_heads=N_Q,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        indexer_n_heads=4,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        full_attention_interval=4,
    )
    profile = make_prefill_memory_profile(config, compute_dtype_size=2)
    assert profile is not None
    q_len, kv_len = 4096, 28672
    charged = profile.estimate_prefill_transient_bytes(q_len, kv_len)
    unfused = estimate_unfused_sdpa_call_bytes(
        N_Q, q_len, kv_len, HEAD_DIM, mm.SDPA256_UNFUSED_SCORE_DTYPE_SIZE
    )
    assert charged < unfused


# --- 7. fail-safe when the registry is missing or unusable -----------------


def test_route_is_bounded_with_an_empty_registry():
    """Registration is the estimator's business. A route that depended on it
    could be talked out of the safe path by a failed registration."""
    assert not mm._SDPA_TILED_PREFILL_HEAD_DIMS
    assert sdpa256._tiled_route_required() is True


def test_route_survives_a_broken_memory_monitor(monkeypatch):
    """A registration failure must not change the route."""

    def boom(*args, **kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(mm, "register_tiled_prefill_head_dim", boom)
    assert sdpa256._register_bounded_route(8192) is False
    assert not mm._SDPA_TILED_PREFILL_HEAD_DIMS
    assert sdpa256._tiled_route_required() is True


def test_unknown_model_geometry_does_not_unlock_the_unfused_path():
    """The route takes no model information, so a monitor that knows nothing
    cannot move it."""
    monitor = MemoryMonitor(max_kv_cache_memory=GIB)  # no set_model_info
    assert monitor.has_model_info() is False
    assert sdpa256._tiled_route_required() is True


def test_non_qualifying_shapes_keep_the_stock_path():
    """Decode-shaped calls (MTP verify) and short contexts must still fall
    through: their score matrix is small and the stock kernel is faster."""
    assert sdpa256._should_route(*_qk(8, 65536), None, "causal", None) is False
    assert sdpa256._should_route(*_qk(4096, 1024), None, "causal", None) is False
    assert (
        sdpa256._should_route(*_qk(4096, 65536, head_dim=128), None, "causal", None)
        is False
    )


# --- 8. explicit override behaviour is unchanged ---------------------------


def test_explicit_override_still_forces_and_disables_the_route():
    q, k = _qk(2048, 16384)
    sdpa256._FORCE_TILED = True
    assert sdpa256._should_route(q, k, None, "causal", None) is True
    sdpa256._FORCE_TILED = False
    assert sdpa256._should_route(q, k, None, "causal", None) is False
    sdpa256._FORCE_TILED = None
    assert sdpa256._should_route(q, k, None, "causal", None) is True


def test_opting_out_withdraws_the_o_l_admission_promise():
    """OMLX_SDPA256_TILED=0 restores the unfused path, so the estimator must
    not go on charging the bounded route's transient."""
    sdpa256._FORCE_TILED = False
    assert sdpa256._register_bounded_route(sdpa256._SDPA256_MIN_KV_LEN) is False
    assert HEAD_DIM not in mm._SDPA_TILED_PREFILL_HEAD_DIMS


def test_bounded_route_logs_once_not_per_call(caplog):
    with caplog.at_level(logging.INFO, logger=sdpa256.__name__):
        for _ in range(5):
            sdpa256._tiled_route_required()
    notices = [r for r in caplog.records if "memory-bounded path" in r.getMessage()]
    assert len(notices) == 1
