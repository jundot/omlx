# SPDX-License-Identifier: Apache-2.0
"""The sdpa256 route must be a pure function of the request, not of process
history.

The bounded (native fused, online softmax) and stock (unfused fp32 score
matrix) routes are different floating-point reductions. Deciding between
them from live process usage made the prefill numerics of a head-dim-256
model depend on what the process had done before: two processes serving the
same prompt with the same chunk partition took the bounded route at
different kv_len and produced different temperature-0 output. These tests
drive the production route gate and the production scheduler provider.
"""

import mlx.core as mx
import pytest

from omlx.patches import sdpa256_attention as sdpa256
from omlx.scheduler import Scheduler

GIB = 1024**3


def _qkv(q_len, k_len, n_q=24, n_kv=4, head_dim=256, dtype=mx.float16):
    mx.random.seed(0)
    q = mx.random.normal((1, n_q, q_len, head_dim)).astype(dtype)
    k = mx.random.normal((1, n_kv, k_len, head_dim)).astype(dtype)
    mx.eval(q, k)
    return q, k


class _Monitor:
    def __init__(self, kv_bytes_per_token, fixed_state_bytes=0, has_info=True):
        self._per_token = kv_bytes_per_token
        self.fixed_state_bytes = fixed_state_bytes
        self._has_info = has_info

    def has_model_info(self):
        return self._has_info

    def estimate_prompt_kv_bytes(self, num_tokens):
        return num_tokens * self._per_token


class _Config:
    hot_cache_max_size = 0


class _Model:
    """Parameter tree of a known size (no evaluation needed for nbytes)."""

    def __init__(self, nbytes):
        self._nbytes = nbytes

    def parameters(self):
        # Lazy zeros: nbytes needs no evaluation. Two axes keep each dim
        # inside int32 for multi-GiB fakes.
        elements = self._nbytes // 4
        return {"w": mx.zeros((elements // 1024, 1024), dtype=mx.float32)}


class _Sched:
    """Minimal scheduler state for the production provider methods."""

    _memory_abort_limit_bytes = 0
    _memory_limits_propagated = True
    _prefill_memory_guard = True
    _sdpa256_unguarded_logged = False
    _prefill_headroom_safety = 0.90
    _PREFILL_HEADROOM_SAFETY = 0.90
    _prefill_abort_margin = 0.95
    _prefill_abort_cap = Scheduler._prefill_abort_cap
    _sdpa256_unfused_headroom = Scheduler._sdpa256_unfused_headroom
    # Bound with getattr so the module still collects on a scheduler that
    # predates the static budget; the regression test then fails instead
    # of erroring.
    _sdpa256_parameter_bytes = getattr(Scheduler, "_sdpa256_parameter_bytes", None)
    _sdpa256_static_resident_bytes = getattr(
        Scheduler, "_sdpa256_static_resident_bytes", None
    )
    route_changes: list

    def __init__(self, *, hard_cap, params_bytes, kv_per_token, live_usage):
        self._memory_hard_limit_bytes = hard_cap
        self.model = _Model(params_bytes)
        self.memory_monitor = _Monitor(kv_per_token)
        self.config = _Config()
        self._live_usage = live_usage
        self.route_changes = []

    def _current_usage_bytes(self, **_):
        return self._live_usage

    def _sdpa256_bounded_route_changed(self, active):
        self.route_changes.append(active)


@pytest.fixture
def _provider_reset(monkeypatch):
    import threading

    monkeypatch.setattr(sdpa256, "_HEADROOM_PROVIDER_LOCAL", threading.local())
    monkeypatch.setattr(sdpa256, "_FORCE_TILED", None)
    monkeypatch.setattr(sdpa256, "_TILED_ROUTE_LOGGED", set())
    return sdpa256


def _routes(sched, chunk=4096, kv_lens=(8192, 16384, 32768, 49152, 65536)):
    sdpa256.set_unfused_headroom_provider(sched._sdpa256_unfused_headroom)
    out = []
    for kv in kv_lens:
        q, k = _qkv(chunk, kv)
        out.append(sdpa256._should_route(q, k, None, "causal", None))
    return out


def test_route_is_independent_of_live_usage(_provider_reset):
    """Regression: a fresh process and a process that already served a
    request differ only in live usage. The route sequence over a request's
    chunks must be identical for both."""
    # 48 GiB cap, 16 GiB of weights, 128 KiB of KV per token: the 24-head
    # fp32 score matrix of a 4096-token chunk fits at 8K context and no
    # longer fits at 64K, so a request crosses the route boundary.
    common = dict(hard_cap=48 * GIB, params_bytes=16 * GIB, kv_per_token=131072)
    fresh = _Sched(live_usage=17 * GIB, **common)
    warm = _Sched(live_usage=27 * GIB, **common)

    fresh_routes = _routes(fresh)
    warm_routes = _routes(warm)

    assert fresh_routes == warm_routes
    # The route still moves with geometry: unfused while the score matrix
    # fits the static budget, bounded once the context is long enough.
    assert fresh_routes[0] is False
    assert fresh_routes[-1] is True
    assert fresh.route_changes == warm.route_changes


def test_headroom_never_samples_live_usage():
    sched = _Sched(
        hard_cap=64 * GIB, params_bytes=16 * GIB, kv_per_token=131072, live_usage=0
    )

    def _boom(**_):
        raise AssertionError("live usage sampled by the route provider")

    sched._current_usage_bytes = _boom
    assert sched._sdpa256_unfused_headroom(16384) > 0


def test_static_headroom_math():
    sched = _Sched(
        hard_cap=100 * GIB, params_bytes=10 * GIB, kv_per_token=1024, live_usage=0
    )
    sched.memory_monitor.fixed_state_bytes = 3 * GIB
    sched.config.hot_cache_max_size = 2 * GIB
    kv_len = 8192
    expected_resident = 10 * GIB + kv_len * 1024 + 3 * GIB + 2 * GIB
    target = int(100 * GIB * 0.90)  # abort cap 100 * 0.95 is higher
    assert sched._sdpa256_static_resident_bytes(kv_len) == expected_resident
    assert sched._sdpa256_unfused_headroom(kv_len) == target - expected_resident
    # Longer context, less headroom, monotonically.
    assert sched._sdpa256_unfused_headroom(
        2 * kv_len
    ) < sched._sdpa256_unfused_headroom(kv_len)


def test_unknown_model_dims_keep_bounded_default(_provider_reset):
    sched = _Sched(
        hard_cap=64 * GIB, params_bytes=16 * GIB, kv_per_token=131072, live_usage=0
    )
    sched.memory_monitor = _Monitor(131072, has_info=False)
    assert sched._sdpa256_unfused_headroom(16384) == -1
    assert _routes(sched, kv_lens=(8192,)) == [True]


def test_model_without_parameter_tree_keeps_bounded_default():
    sched = _Sched(
        hard_cap=64 * GIB, params_bytes=16 * GIB, kv_per_token=131072, live_usage=0
    )
    sched.model = object()
    assert sched._sdpa256_parameter_bytes() == -1
    assert sched._sdpa256_unfused_headroom(16384) == -1


def test_parameter_bytes_include_private_submodules():
    """VLMModelAdapter keeps the language model behind ``_language_model``,
    which ``nn.Module.parameters()`` skips; the budget must still see it."""
    import mlx.nn as nn

    class _Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self._language_model = nn.Linear(256, 256, bias=False)  # 256 KiB
            self.vision = nn.Linear(16, 16, bias=False)  # 1 KiB

    sched = _Sched(
        hard_cap=64 * GIB, params_bytes=1 * GIB, kv_per_token=131072, live_usage=0
    )
    sched.model = _Adapter()
    assert sched._sdpa256_parameter_bytes() == 256 * 256 * 4 + 16 * 16 * 4


def test_parameter_bytes_computed_once():
    sched = _Sched(
        hard_cap=64 * GIB, params_bytes=1 * GIB, kv_per_token=131072, live_usage=0
    )
    first = sched._sdpa256_parameter_bytes()
    assert first == 1 * GIB
    sched.model = object()  # would fail if recomputed
    assert sched._sdpa256_parameter_bytes() == first


def test_route_gate_passes_kv_len_to_provider(_provider_reset):
    seen = []

    class _Owner:
        def headroom(self, kv_len):
            seen.append(kv_len)
            return 1 << 40

    owner = _Owner()
    sdpa256.set_unfused_headroom_provider(owner.headroom)
    q, k = _qkv(2048, 16384)
    assert sdpa256._should_route(q, k, None, "causal", None) is False
    assert seen == [16384]
