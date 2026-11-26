"""Price transient Gemma projections before the first prefill measurement."""

from types import SimpleNamespace

import pytest
from mlx_vlm.models.cache import KVCache, RotatingKVCache

from omlx import memory_monitor as mm
from omlx.exceptions import PrefillMemoryExceededError
from omlx.scheduler import Scheduler

_RESERVE = 2 * 3840 * 15360 * 2


def _model(*, optimized=True, shape="direct"):
    config = SimpleNamespace(
        model_type="gemma3_text",
        hidden_size=3840,
        intermediate_size=15360,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
    )
    model = SimpleNamespace(
        config=config,
        layers=[],
        make_cache=lambda: [KVCache() for _ in range(8)]
        + [RotatingKVCache(max_size=1024, keep=0) for _ in range(40)],
    )
    if optimized:
        model._omlx_prefill_scratch_bytes = _RESERVE
        model._omlx_prefill_scratch_min_tokens = 512
    if shape == "direct":
        return model
    wrapper = SimpleNamespace(
        config=SimpleNamespace(text_config=config),
        layers=model.layers,
        make_cache=model.make_cache,
    )
    setattr(
        wrapper, "_language_model" if shape == "adapter" else "language_model", model
    )
    return wrapper


def _monitor(model):
    monitor = mm.MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
    mm.set_model_info_from_model(monitor, model)
    return monitor


def _scheduler(model):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    scheduler.memory_monitor = mm.MemoryMonitor(
        max_kv_cache_memory=None, eviction_enabled=False
    )
    scheduler._turboquant_kv_bits = None
    scheduler._prefill_transient_tracker = None
    scheduler._prefill_min_chunk_tokens = 32
    scheduler._set_model_info_for_monitor()
    return scheduler


@pytest.fixture(autouse=True)
def _stable_attention_estimator(monkeypatch):
    monkeypatch.setattr(mm, "_SDPA_TILED_PREFILL_HEAD_DIMS", {})


def test_lookup_defaults_to_zero_without_route_metadata():
    assert _RESERVE == 225 * 1024**2
    assert mm._prefill_scratch_requirements(_model(optimized=False)) == (0, 1)
    assert mm._prefill_scratch_requirements(SimpleNamespace()) == (0, 1)
    assert mm._prefill_scratch_requirements(None) == (0, 1)
    invalid = SimpleNamespace(
        _omlx_prefill_scratch_bytes=True, _omlx_prefill_scratch_min_tokens=512
    )
    assert mm._prefill_scratch_requirements(invalid) == (0, 1)


@pytest.mark.parametrize("entry", ["shared", "scheduler"])
@pytest.mark.parametrize("shape", ["direct", "raw", "adapter"])
def test_model_info_entry_points_price_and_reset_reserve(entry, shape):
    model = _model(shape=shape)
    if entry == "scheduler":
        scheduler = _scheduler(model)
        monitor = scheduler.memory_monitor
    else:
        monitor = _monitor(model)
    assert monitor._prefill_scratch_bytes == _RESERVE
    assert monitor._prefill_scratch_min_tokens == 512
    assert monitor._num_kv_cache_layers == 8
    assert monitor._rotating_layer_specs == ((40, 1024),)

    other_model = _model(optimized=False, shape=shape)
    if entry == "scheduler":
        scheduler.model = other_model
        scheduler._set_model_info_for_monitor()
    else:
        mm.set_model_info_from_model(monitor, other_model)
    assert monitor._prefill_scratch_bytes == 0


@pytest.mark.parametrize("rows", [0, 1, 511, 512, 1024, 2048, 2049, 4096])
@pytest.mark.parametrize("cached", [0, 32768])
def test_reserve_reaches_upfront_and_per_chunk_estimates(rows, cached):
    base = _monitor(_model(optimized=False))
    optimized = _monitor(_model())
    expected = _RESERVE if rows >= 512 else 0
    assert (
        optimized.estimate_chunk_transient_bytes(rows, cached + rows)
        - base.estimate_chunk_transient_bytes(rows, cached + rows)
    ) == expected
    assert (
        optimized.estimate_prefill_peak_bytes(rows, 4096, cached_tokens=cached)
        - base.estimate_prefill_peak_bytes(rows, 4096, cached_tokens=cached)
    ) == expected


def test_direct_model_info_reset_removes_scratch():
    monitor = _monitor(_model())
    monitor.set_model_info(num_layers=48, num_kv_heads=8, head_dim=256)
    assert monitor._prefill_scratch_bytes == 0


@pytest.mark.parametrize("cached", [0, 32768])
def test_frontdoor_guard_rejects_first_eligible_chunk_without_scratch_headroom(cached):
    base = _monitor(_model(optimized=False))
    optimized = _monitor(_model())
    current = 8 * 1024**3
    old_peak = base.estimate_prefill_peak_bytes(512, 2048, cached_tokens=cached)
    arguments = dict(
        prefill_memory_guard=True,
        hard_limit_bytes=int(current + old_peak + _RESERVE // 2),
        current_usage_bytes=current,
        prefill_step_size=2048,
        num_prompt_tokens=cached + 512,
        cached_tokens=cached,
    )
    mm.raise_if_prefill_exceeds(base, **arguments)
    with pytest.raises(PrefillMemoryExceededError):
        mm.raise_if_prefill_exceeds(optimized, **arguments)


@pytest.mark.parametrize("cached", [0, 32768])
def test_first_chunk_prediction_and_bisection_include_scratch(cached):
    scheduler = _scheduler(_model())
    base = _scheduler(_model(optimized=False))
    assert scheduler._prefill_transient_tracker is None
    delta = scheduler._predicted_chunk_transient(
        512, cached
    ) - base._predicted_chunk_transient(512, cached)
    assert delta == pytest.approx(_RESERVE * Scheduler._PREFILL_TRANSIENT_SAFETY)

    rows = [511, 512, 1024, 2047, 2048, 2049, 4096]
    predictions = [scheduler._predicted_chunk_transient(n, cached) for n in rows]
    assert predictions == sorted(predictions)
    headroom = scheduler._predicted_chunk_transient(511, cached)
    fitting = scheduler._largest_fitting_prefill_chunk(
        4096, 32, headroom, cached, gathered_core=False
    )
    assert 32 <= fitting < 512
    assert scheduler._predicted_chunk_transient(fitting, cached) <= headroom
    assert scheduler._predicted_chunk_transient(fitting + 32, cached) > headroom
