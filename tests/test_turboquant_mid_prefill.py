# SPDX-License-Identifier: Apache-2.0
"""Focused contracts for bounded TurboQuant conversion during prefill."""

from __future__ import annotations

import math
import multiprocessing
import os
import threading
import weakref
from multiprocessing.connection import Connection
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_vlm.turboquant import TurboQuantKVCache

from omlx.exceptions import PrefillMemoryExceededError
from omlx.request import Request, SamplingParams
from omlx.scheduler import (
    Scheduler,
    SchedulerConfig,
    _PrefillAbortedError,
    _PrefillEvictionNeeded,
    _PrefillKVPhase,
)
from omlx.turboquant_kv import (
    TurboQuantConversionStats,
    convert_kv_cache_sliced,
    estimate_turboquant_conversion_peak_bytes,
    estimate_turboquant_prefill_attention_workspace_bytes,
    turboquant_mse_bytes_per_element,
)
from omlx.utils.metal_sync import (
    _conversion_coordinator,
    _ConversionCoordinator,
    _mx_buffer_access_lock,
)


class _AppendModel:
    """Small cache-mutating model stand-in for both scheduler prefill paths."""

    def __init__(self) -> None:
        self.layers: list[Any] = []
        self.dtype = mx.float16
        self.config = SimpleNamespace(
            model_type="unit",
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=64,
            head_dim=32,
        )
        self.calls = 0

    def make_cache(self) -> list[Any]:
        return [KVCache(), KVCache()]

    def __call__(
        self,
        tokens: mx.array,
        cache: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.calls += 1
        if cache is None:
            return
        n_tokens = int(tokens.shape[1])
        value = float(self.calls)
        for cache_obj in cache:
            if not isinstance(cache_obj, KVCache | TurboQuantKVCache):
                continue
            keys = mx.full((1, 2, n_tokens, 32), value, dtype=mx.float16)
            values = mx.full((1, 2, n_tokens, 32), value + 0.25, dtype=mx.float16)
            cache_obj.update_and_fetch(keys, values)


class _PartialArrayOwner:
    """Weak-referenceable owner for a converter-local partial MLX array."""

    def __init__(self, array: mx.array) -> None:
        self.array = array


def _assert_buffer_access_lock_available() -> None:
    acquired = threading.Event()

    def _acquire() -> None:
        with _mx_buffer_access_lock:
            acquired.set()

    thread = threading.Thread(target=_acquire, daemon=True)
    thread.start()
    assert acquired.wait(timeout=2)
    thread.join(timeout=2)
    assert not thread.is_alive()


class _TestProcessOwner:
    pass


_active_test_process_owner: _TestProcessOwner | None = None


@pytest.fixture(autouse=True)
def _claim_test_process_owner() -> Any:
    """Give direct Scheduler fixtures the same fail-closed capability as EnginePool."""
    global _active_test_process_owner
    owner = _TestProcessOwner()
    _conversion_coordinator.register_engine(owner)
    _conversion_coordinator.claim_process_exclusive(owner)
    _active_test_process_owner = owner
    try:
        yield
    finally:
        _active_test_process_owner = None
        _conversion_coordinator.unregister_engine(owner)


def _make_scheduler(*, step_size: int = 4) -> Scheduler:
    model = _AppendModel()
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            prefill_step_size=step_size,
            chunked_prefill=True,
            paged_cache_block_size=0,
        ),
    )
    scheduler._turboquant_kv_bits = 4.0
    scheduler._turboquant_skip_last = True
    scheduler._turboquant_mid_prefill = True
    scheduler._set_model_info_for_monitor()
    assert _active_test_process_owner is not None
    scheduler._metal_process_owner = _active_test_process_owner
    return scheduler


def _make_request(
    request_id: str,
    tokens: list[int],
    cache: list[Any] | None = None,
) -> Request:
    request = Request(
        request_id=request_id,
        prompt=tokens,
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = list(tokens)
    request.remaining_tokens = list(tokens)
    request.num_prompt_tokens = len(tokens)
    request.prompt_cache = cache
    return request


def _append_dense(cache_obj: KVCache, *, tokens: int, value: float = 1.0) -> None:
    cache_obj.update_and_fetch(
        mx.full((1, 2, tokens, 32), value, dtype=mx.float16),
        mx.full((1, 2, tokens, 32), value + 0.5, dtype=mx.float16),
    )


def _dense_cache(*, tokens: int = 0, layers: int = 2) -> list[Any]:
    cache: list[Any] = [KVCache() for _ in range(layers)]
    if tokens > 0:
        for index, cache_obj in enumerate(cache):
            _append_dense(cache_obj, tokens=tokens, value=float(index + 1))
        mx.eval([cache_obj.state for cache_obj in cache])
    return cache


def _configure_pressure(
    scheduler: Scheduler,
    *,
    pressure_after_tokens: int = 4,
) -> int:
    cap = 64 * 1024**2
    usage = 1024**2
    scheduler._memory_limit_bytes = 0
    scheduler._memory_hard_limit_bytes = cap
    scheduler._memory_abort_limit_bytes = cap
    scheduler._prefill_abort_margin = 1.0
    scheduler._prefill_min_chunk_tokens = 1

    def _current(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del self, refresh_mlx_active
        return usage

    def _reclaim(self: Scheduler) -> int:
        del self
        return usage

    def _bound(
        self: Scheduler,
        n_tokens: int,
        kv_len: int,
        *,
        phase: _PrefillKVPhase = _PrefillKVPhase.DENSE,
    ) -> float:
        del self, n_tokens
        if phase is _PrefillKVPhase.DENSE and kv_len >= pressure_after_tokens:
            return float(cap * 2)
        return 1024.0

    scheduler._current_usage_bytes = MethodType(_current, scheduler)
    scheduler._reclaim_prefill_headroom = MethodType(_reclaim, scheduler)
    scheduler._admission_transient_bound = MethodType(_bound, scheduler)
    return cap


def _configure_sizing_band(scheduler: Scheduler) -> tuple[int, int]:
    mib = 1024**2
    cap = 10 * mib
    usage = 6 * mib
    dense_per_token = 768 * 1024
    scheduler._memory_limit_bytes = cap
    scheduler._memory_hard_limit_bytes = cap
    scheduler._memory_abort_limit_bytes = cap
    scheduler._prefill_abort_margin = 1.0
    scheduler._prefill_headroom_safety = 0.8
    scheduler._prefill_min_chunk_tokens = 1
    scheduler._prefill_speed_priority = False
    scheduler._prefill_eviction_callback_configured = False

    def _current(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del self, refresh_mlx_active
        return usage

    def _reclaim(self: Scheduler) -> int:
        del self
        return usage

    def _predicted(
        self: Scheduler,
        n_tokens: int,
        kv_len: int,
        *,
        phase: _PrefillKVPhase = _PrefillKVPhase.DENSE,
    ) -> float:
        del self, kv_len
        if phase is _PrefillKVPhase.TURBOQUANT:
            return 1024.0
        return float(n_tokens * dense_per_token)

    scheduler._current_usage_bytes = MethodType(_current, scheduler)
    scheduler._reclaim_prefill_headroom = MethodType(_reclaim, scheduler)
    scheduler._predicted_chunk_transient = MethodType(_predicted, scheduler)
    scheduler._admission_transient_bound = MethodType(_predicted, scheduler)
    return usage, cap


def _state_equal(left: Any, right: Any) -> bool:
    return bool(
        mx.all(left.norms == right.norms).item()
        and mx.all(left.indices == right.indices).item()
    )


def _measure_turboquant_conversion_mlx_peak(connection: Connection) -> None:
    tokens = 33
    layers = 3
    slice_tokens = 8
    cache = _dense_cache(tokens=tokens, layers=layers)
    retained_source_states = [cache_obj.state for cache_obj in cache]
    mx.eval(retained_source_states)
    mx.synchronize()
    estimate = estimate_turboquant_conversion_peak_bytes(
        cache,
        bits=4.0,
        skip_last=False,
        slice_tokens=slice_tokens,
    )
    mx.clear_cache()
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    stats = convert_kv_cache_sliced(
        cache,
        bits=4.0,
        skip_last=False,
        slice_tokens=slice_tokens,
    )
    mx.eval([cache_obj.state for cache_obj in cache])
    mx.synchronize()
    observed_incremental = max(0, int(mx.get_peak_memory()) - active_before)
    allocator_tolerance = 2 * int(os.sysconf("SC_PAGE_SIZE")) * 64
    assert retained_source_states
    connection.send(
        (
            observed_incremental,
            estimate,
            allocator_tolerance,
            stats.converted_layers,
            stats.slices,
        )
    )
    connection.close()


def test_sliced_converter_matches_whole_conversion_and_appends() -> None:
    source = KVCache()
    _append_dense(source, tokens=17)
    mx.eval(source.state)
    whole = TurboQuantKVCache.from_cache(source, bits=4.0)

    sliced_source = KVCache()
    sliced_source.update_and_fetch(*source.state)
    cache: list[Any] = [sliced_source]
    stats = convert_kv_cache_sliced(
        cache,
        bits=4.0,
        skip_last=False,
        slice_tokens=5,
    )
    sliced = cache[0]

    assert isinstance(sliced, TurboQuantKVCache)
    assert stats.converted_layers == 1
    assert stats.slices == math.ceil(17 / 5)
    assert _state_equal(whole.keys, sliced.keys)
    assert _state_equal(whole.values, sliced.values)

    append_keys = mx.full((1, 2, 2, 32), 3.0, dtype=mx.float16)
    append_values = mx.full((1, 2, 2, 32), 4.0, dtype=mx.float16)
    whole.update_and_fetch(append_keys, append_values)
    sliced.update_and_fetch(append_keys, append_values)
    mx.eval(whole.keys, whole.values, sliced.keys, sliced.values)
    assert whole.offset == sliced.offset == 19
    assert _state_equal(whole.keys, sliced.keys)
    assert _state_equal(whole.values, sliced.values)

    queries = mx.ones((1, 2, 1, 32), dtype=mx.float16)
    whole_attention = whole.decode_attention(queries, scale=32**-0.5)
    sliced_attention = sliced.decode_attention(queries, scale=32**-0.5)
    mx.eval(whole_attention, sliced_attention)
    assert mx.allclose(whole_attention, sliced_attention).item()

    second = convert_kv_cache_sliced(
        cache,
        bits=4.0,
        skip_last=False,
        slice_tokens=5,
    )
    assert cache[0] is sliced
    assert second.converted_layers == 0
    assert second.already_quantized_layers == 1


@pytest.mark.parametrize(
    ("head_dim", "bits"),
    [
        (32, 4.0),
        (80, 3.0),
        (80, 3.5),
    ],
)
def test_resident_width_matches_actual_packed_mse_state(
    head_dim: int,
    bits: float,
) -> None:
    batch_size = 1
    num_heads = 2
    num_tokens = 5
    source = KVCache()
    source.update_and_fetch(
        mx.ones(
            (batch_size, num_heads, num_tokens, head_dim),
            dtype=mx.float16,
        ),
        mx.ones(
            (batch_size, num_heads, num_tokens, head_dim),
            dtype=mx.float16,
        ),
    )
    mx.eval(source.state)
    converted = TurboQuantKVCache.from_cache(source, bits=bits)
    mx.eval(converted.keys, converted.values)
    actual_state_bytes = sum(
        int(array.nbytes)
        for state in converted.state
        for array in (state.norms, state.indices)
    )
    logical_elements = 2 * batch_size * num_heads * num_tokens * head_dim
    actual_width = actual_state_bytes / logical_elements

    assert turboquant_mse_bytes_per_element(head_dim, bits) == actual_width

    scheduler = _make_scheduler()
    scheduler.model.config.head_dim = head_dim
    scheduler.model.config.hidden_size = num_heads * head_dim
    scheduler._turboquant_kv_bits = bits
    scheduler._turboquant_skip_last = False
    scheduler._set_model_info_for_monitor()
    assert scheduler._prefill_tq_kv_dtype_size == actual_width


def test_sliced_converter_is_layer_atomic() -> None:
    cache = _dense_cache(tokens=4, layers=2)
    callbacks = 0

    def _cancel_after_first_layer() -> None:
        nonlocal callbacks
        callbacks += 1
        if callbacks == 3:
            raise _PrefillAbortedError([], 4)

    with pytest.raises(_PrefillAbortedError):
        convert_kv_cache_sliced(
            cache,
            bits=4.0,
            skip_last=False,
            slice_tokens=4,
            check_cancelled=_cancel_after_first_layer,
        )

    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[1], KVCache)


def test_converter_drops_cached_prefix_before_third_and_later_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _dense_cache(tokens=16, layers=1)
    observed_prefixes: list[tuple[Any, int]] = []
    original_update = TurboQuantKVCache.update_and_fetch

    def _tracking_update(
        cache_obj: TurboQuantKVCache,
        keys: mx.array,
        values: mx.array,
    ) -> Any:
        observed_prefixes.append(
            (cache_obj._cached_state, cache_obj._cached_state_offset)
        )
        return original_update(cache_obj, keys, values)

    monkeypatch.setattr(TurboQuantKVCache, "update_and_fetch", _tracking_update)
    stats = convert_kv_cache_sliced(
        cache,
        bits=4.0,
        skip_last=False,
        slice_tokens=4,
    )

    assert stats.slices == 4
    assert observed_prefixes == [(None, -1)] * 4


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_sliced_converter_incremental_mlx_peak_is_within_estimate() -> None:
    spawn_context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = spawn_context.Pipe(duplex=False)
    process = spawn_context.Process(
        target=_measure_turboquant_conversion_mlx_peak,
        args=(child_connection,),
    )

    try:
        process.start()
        child_connection.close()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            pytest.fail("TurboQuant MLX peak subprocess timed out")
        assert process.exitcode == 0
        assert parent_connection.poll(1.0)
        (
            observed_incremental,
            estimate,
            allocator_tolerance,
            converted_layers,
            slices,
        ) = parent_connection.recv()
    finally:
        parent_connection.close()
        child_connection.close()

    assert converted_layers == 3
    assert slices == 15
    assert observed_incremental > 1
    assert observed_incremental <= estimate + allocator_tolerance


def test_mid_prefill_process_exclusivity_fails_closed() -> None:
    class _EngineOwner:
        pass

    coordinator = _ConversionCoordinator()
    first = _EngineOwner()
    second = _EngineOwner()
    later = _EngineOwner()
    assert coordinator.process_exclusive(None) is False
    assert coordinator.process_exclusive(first) is False
    assert coordinator.snapshot() == (0, 0, False, 0)
    with (
        pytest.raises(RuntimeError, match="lacks process-exclusive"),
        coordinator.conversion(process_owner=None),
    ):
        pass

    coordinator.register_engine(first)
    coordinator.register_engine(second)
    try:
        with pytest.raises(RuntimeError, match="unload all other engines"):
            coordinator.claim_process_exclusive(first)

        coordinator.unregister_engine(second)
        coordinator.claim_process_exclusive(first)
        assert coordinator.process_exclusive(first) is True

        with pytest.raises(RuntimeError, match="unload the mid-prefill model"):
            coordinator.register_engine(later)
        with (
            pytest.raises(RuntimeError, match="Independent Metal work"),
            coordinator.background_metal_operation(),
        ):
            pass

        with coordinator.conversion(process_owner=first):
            assert coordinator.snapshot()[2] is True
    finally:
        coordinator.unregister_engine(later)
        coordinator.unregister_engine(second)
        coordinator.unregister_engine(first)

    assert coordinator.snapshot() == (0, 0, False, 0)


def test_guarded_converter_holds_buffer_lock_until_final_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _make_scheduler()
    cache = _dense_cache(tokens=4)
    request = _make_request("buffer-lock", list(range(9)), cache)
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )
    converted_first = TurboQuantKVCache.from_cache(cache[0], bits=4.0)
    mx.eval(converted_first.keys, converted_first.values)
    converter_entered = threading.Event()
    release_converter = threading.Event()
    store_attempting = threading.Event()
    store_entered = threading.Event()
    errors: list[BaseException] = []

    def _current_usage(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del self, refresh_mlx_active
        return 4096

    def _convert(
        prompt_cache: list[Any],
        *,
        check_cancelled: Any = None,
        log_result: bool = True,
    ) -> TurboQuantConversionStats:
        del check_cancelled, log_result
        prompt_cache[0] = converted_first
        converter_entered.set()
        release_converter.wait(timeout=5)
        return TurboQuantConversionStats(
            converted_layers=1,
            already_quantized_layers=0,
            skipped_dense_layers=1,
            slices=1,
            source_bytes=0,
            converted_bytes=0,
        )

    def _no_sync(stream: Any | None = None) -> None:
        del stream

    def _run_conversion() -> None:
        try:
            scheduler._attempt_mid_prefill_conversion(
                request=request,
                prompt_cache=cache,
                context=context,
                processed_tokens=4,
                safety_cap=0,
            )
        except BaseException as exc:
            errors.append(exc)

    def _store_read() -> None:
        try:
            store_attempting.set()
            with _mx_buffer_access_lock:
                store_entered.set()
        except BaseException as exc:
            errors.append(exc)

    scheduler._current_usage_bytes = MethodType(_current_usage, scheduler)
    scheduler._apply_turboquant_kv_convert_sliced = _convert
    monkeypatch.setattr("omlx.scheduler._sync_and_clear_cache", _no_sync)
    conversion_thread = threading.Thread(target=_run_conversion, daemon=True)
    store_thread = threading.Thread(target=_store_read, daemon=True)
    try:
        conversion_thread.start()
        assert converter_entered.wait(timeout=5)
        lock_was_available = _mx_buffer_access_lock.acquire(blocking=False)
        if lock_was_available:
            _mx_buffer_access_lock.release()
        assert lock_was_available is False
        store_thread.start()
        assert store_attempting.wait(timeout=5)
        assert not store_entered.is_set()
        release_converter.set()
        assert store_entered.wait(timeout=5)
    finally:
        release_converter.set()
        conversion_thread.join(timeout=5)
        store_thread.join(timeout=5)

    assert not errors
    assert not conversion_thread.is_alive()
    assert not store_thread.is_alive()
    assert context.phase is _PrefillKVPhase.TURBOQUANT
    assert _conversion_coordinator.snapshot() == (0, 0, False, 0)


def test_outstanding_conversion_peak_affects_nonholder_not_holder_post_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _make_scheduler()
    base_bytes = 4096
    peak_bytes = 2048

    def _physical_usage() -> int:
        return base_bytes

    def _zero_hot_cache() -> int:
        return 0

    monkeypatch.setattr("omlx.scheduler.get_phys_footprint", _physical_usage)
    scheduler._hot_cache_cpu_bytes = _zero_hot_cache
    scheduler._last_mlx_active_memory_bytes = base_bytes

    with _conversion_coordinator.conversion(
        process_owner=_active_test_process_owner
    ) as owner:
        accepted, estimated = _conversion_coordinator.try_reserve(
            owner,
            current_bytes=base_bytes,
            peak_bytes=peak_bytes,
            limit_bytes=base_bytes + peak_bytes,
        )
        assert accepted is True
        assert estimated == base_bytes + peak_bytes
        assert (
            scheduler._current_usage_bytes(refresh_mlx_active=False)
            == base_bytes + peak_bytes
        )

        _conversion_coordinator.release_reservation(owner)
        holder_post_sample = scheduler._current_usage_bytes(refresh_mlx_active=False)

    assert holder_post_sample == base_bytes
    assert _conversion_coordinator.snapshot() == (0, 0, False, 0)


def test_process_schedulers_cannot_accept_same_conversion_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_scheduler = _make_scheduler()
    second_scheduler = _make_scheduler()
    base_bytes = 4096
    peak_bytes = 2048
    cap_bytes = base_bytes + peak_bytes
    realized_usage = [base_bytes]
    first_reserved = threading.Event()
    release_first = threading.Event()
    outcomes: dict[str, bool] = {}
    errors: list[BaseException] = []

    def _physical_usage() -> int:
        return realized_usage[0]

    def _zero_hot_cache() -> int:
        return 0

    monkeypatch.setattr("omlx.scheduler.get_phys_footprint", _physical_usage)
    first_scheduler._hot_cache_cpu_bytes = _zero_hot_cache
    second_scheduler._hot_cache_cpu_bytes = _zero_hot_cache

    def _first_conversion() -> None:
        try:
            with _conversion_coordinator.conversion(
                process_owner=_active_test_process_owner
            ) as owner:
                current = first_scheduler._current_usage_bytes(refresh_mlx_active=False)
                outcomes["first"], _ = _conversion_coordinator.try_reserve(
                    owner,
                    current_bytes=current,
                    peak_bytes=peak_bytes,
                    limit_bytes=cap_bytes,
                )
                first_reserved.set()
                release_first.wait(timeout=5)
                realized_usage[0] = cap_bytes
        except BaseException as exc:
            errors.append(exc)

    def _second_conversion() -> None:
        try:
            first_reserved.wait(timeout=5)
            with _conversion_coordinator.conversion(
                process_owner=_active_test_process_owner
            ) as owner:
                current = second_scheduler._current_usage_bytes(
                    refresh_mlx_active=False
                )
                outcomes["second"], _ = _conversion_coordinator.try_reserve(
                    owner,
                    current_bytes=current,
                    peak_bytes=peak_bytes,
                    limit_bytes=cap_bytes,
                )
        except BaseException as exc:
            errors.append(exc)

    def _second_conversion_waiting() -> bool:
        return _conversion_coordinator._waiting_conversions == 1

    first_thread = threading.Thread(target=_first_conversion, daemon=True)
    second_thread = threading.Thread(target=_second_conversion, daemon=True)
    try:
        first_thread.start()
        assert first_reserved.wait(timeout=5)
        second_thread.start()
        with _conversion_coordinator._condition:
            assert _conversion_coordinator._condition.wait_for(
                _second_conversion_waiting,
                timeout=5,
            )
        assert outcomes == {"first": True}
        assert (
            second_scheduler._current_usage_bytes(refresh_mlx_active=False) == cap_bytes
        )
        release_first.set()
    finally:
        release_first.set()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)

    assert not errors
    assert outcomes == {"first": True, "second": False}
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert _conversion_coordinator.snapshot() == (0, 0, False, 0)


def test_phase_classifier_requires_one_complete_representation() -> None:
    scheduler = _make_scheduler()
    dense = _dense_cache(tokens=4, layers=3)
    phase, eligible = scheduler._classify_prefill_cache(dense)
    assert phase is _PrefillKVPhase.DENSE
    assert eligible is True

    converted = _dense_cache(tokens=4, layers=3)
    scheduler._apply_turboquant_kv_convert_sliced(converted, log_result=False)
    phase, eligible = scheduler._classify_prefill_cache(converted)
    assert phase is _PrefillKVPhase.TURBOQUANT
    assert eligible is False

    partial = _dense_cache(tokens=4, layers=3)
    partial[0] = TurboQuantKVCache.from_cache(partial[0], bits=4.0)
    assert (
        scheduler._classify_prefill_cache(partial)[0] is _PrefillKVPhase.INVALID_PARTIAL
    )

    wrong_bits = _dense_cache(tokens=4, layers=2)
    wrong_bits[0] = TurboQuantKVCache.from_cache(wrong_bits[0], bits=6.0)
    assert (
        scheduler._classify_prefill_cache(wrong_bits)[0]
        is _PrefillKVPhase.INVALID_PARTIAL
    )

    mismatched = _dense_cache(tokens=4, layers=2)
    mismatched[1] = KVCache()
    _append_dense(mismatched[1], tokens=3)
    assert (
        scheduler._classify_prefill_cache(mismatched)[0]
        is _PrefillKVPhase.INVALID_PARTIAL
    )


def test_conversion_safety_cap_rejects_one_byte_before_converter() -> None:
    scheduler = _make_scheduler()
    cache = _dense_cache(tokens=4)
    request = _make_request("cap-rejected", list(range(9)), cache)
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )
    usage = 4096

    def _current_usage(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del self, refresh_mlx_active
        return usage

    scheduler._current_usage_bytes = MethodType(_current_usage, scheduler)
    conversion_peak = estimate_turboquant_conversion_peak_bytes(
        cache,
        bits=4.0,
        skip_last=True,
    )
    estimated_peak = usage + conversion_peak
    converter_entered = False

    def _unexpected_convert(
        prompt_cache: list[Any],
        *,
        check_cancelled: Any = None,
        log_result: bool = True,
    ) -> TurboQuantConversionStats:
        nonlocal converter_entered
        del prompt_cache, check_cancelled, log_result
        converter_entered = True
        raise AssertionError("converter entered before safety-cap rejection")

    scheduler._apply_turboquant_kv_convert_sliced = _unexpected_convert
    with pytest.raises(PrefillMemoryExceededError) as exc:
        scheduler._attempt_mid_prefill_conversion(
            request=request,
            prompt_cache=cache,
            context=context,
            processed_tokens=4,
            safety_cap=estimated_peak - 1,
        )

    assert converter_entered is False
    assert context.conversion_attempted is True
    assert request.turboquant_mid_prefill_attempted is True
    assert exc.value.estimated_bytes == estimated_peak
    assert exc.value.limit_bytes == estimated_peak - 1
    assert cache == []
    assert request.prompt_cache is None
    assert exc.value.__context__ is None
    assert _conversion_coordinator.snapshot() == (0, 0, False, 0)
    _assert_buffer_access_lock_available()


@pytest.mark.parametrize("headroom_bytes", [0, 1])
def test_conversion_safety_cap_at_or_above_boundary_enters_converter(
    headroom_bytes: int,
) -> None:
    scheduler = _make_scheduler()
    cache = _dense_cache(tokens=4)
    request = _make_request(f"cap-sufficient-{headroom_bytes}", list(range(9)), cache)
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )
    usage = 4096

    def _current_usage(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del self, refresh_mlx_active
        return usage

    scheduler._current_usage_bytes = MethodType(_current_usage, scheduler)
    conversion_peak = estimate_turboquant_conversion_peak_bytes(
        cache,
        bits=4.0,
        skip_last=True,
    )
    estimated_peak = usage + conversion_peak
    original_converter = scheduler._apply_turboquant_kv_convert_sliced
    converter_entered = False

    def _tracking_convert(
        prompt_cache: list[Any],
        *,
        check_cancelled: Any = None,
        log_result: bool = True,
    ) -> TurboQuantConversionStats:
        nonlocal converter_entered
        converter_entered = True
        return original_converter(
            prompt_cache,
            check_cancelled=check_cancelled,
            log_result=log_result,
        )

    scheduler._apply_turboquant_kv_convert_sliced = _tracking_convert
    scheduler._attempt_mid_prefill_conversion(
        request=request,
        prompt_cache=cache,
        context=context,
        processed_tokens=4,
        safety_cap=estimated_peak + headroom_bytes,
    )

    assert converter_entered is True
    assert context.phase is _PrefillKVPhase.TURBOQUANT


def test_guard_evicts_once_before_converting() -> None:
    scheduler = _make_scheduler()
    cap = _configure_pressure(scheduler)
    cache = _dense_cache(tokens=4)
    request = _make_request("guard", list(range(9)), cache)
    scheduler.requests[request.request_id] = request
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )

    with pytest.raises(_PrefillEvictionNeeded) as exc:
        scheduler._guard_prefill_chunk(
            4,
            kv_len=4,
            progress=4,
            loop_label="external",
            request_id=request.request_id,
            request=request,
            prompt_cache=cache,
            prefill_context=context,
        )
    assert exc.value.request.reason == "turboquant_mid_prefill"
    assert exc.value.request.processed_tokens == 4
    assert request.prefill_eviction_retries == 1
    assert context.conversion_attempted is False
    assert request.turboquant_mid_prefill_attempted is False
    assert all(isinstance(cache_obj, KVCache) for cache_obj in cache)

    result = scheduler._guard_prefill_chunk(
        4,
        kv_len=4,
        progress=4,
        loop_label="external",
        request_id=request.request_id,
        request=request,
        prompt_cache=cache,
        prefill_context=context,
    )
    assert result == 4
    assert context.phase is _PrefillKVPhase.TURBOQUANT
    assert request.turboquant_mid_prefill_attempted is True
    assert context.memory_after_bytes < cap
    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[1], KVCache)


def test_guard_skips_eviction_pause_without_callback() -> None:
    scheduler = _make_scheduler()
    cap = _configure_pressure(scheduler)
    scheduler._prefill_eviction_callback_configured = False
    cache = _dense_cache(tokens=4)
    request = _make_request("no-eviction-callback", list(range(9)), cache)
    scheduler.requests[request.request_id] = request
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )

    result = scheduler._guard_prefill_chunk(
        4,
        kv_len=4,
        progress=4,
        loop_label="external",
        request_id=request.request_id,
        request=request,
        prompt_cache=cache,
        prefill_context=context,
    )

    assert result == 4
    assert request.prefill_eviction_retries == 0
    assert context.phase is _PrefillKVPhase.TURBOQUANT
    assert context.memory_after_bytes < cap


def test_sizing_target_pressure_triggers_conversion_before_abort_cap() -> None:
    scheduler = _make_scheduler(step_size=4)
    usage, cap = _configure_sizing_band(scheduler)
    cache = _dense_cache(tokens=4)
    request = _make_request("target-band", list(range(9)), cache)
    scheduler.requests[request.request_id] = request
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )

    candidate = scheduler._adaptive_chunk_size(
        4,
        request_id=request.request_id,
        loop_label="external",
        kv_len=4,
        prefill_context=context,
    )
    assert candidate == 4
    assert scheduler._prefill_sizing_target() == 8 * 1024**2
    assert usage + scheduler._admission_transient_bound(4, 4) < cap

    result = scheduler._guard_prefill_chunk(
        candidate,
        kv_len=4,
        progress=4,
        loop_label="external",
        request_id=request.request_id,
        request=request,
        prompt_cache=cache,
        prefill_context=context,
    )

    assert result == 4
    assert context.phase is _PrefillKVPhase.TURBOQUANT
    assert context.trigger_tokens == 4
    assert request.turboquant_mid_prefill_attempted is True


def test_organic_pressure_triggers_eviction_pause_then_retry_converts() -> None:
    scheduler = _make_scheduler(step_size=4)
    cap = _configure_pressure(scheduler)
    cache = _dense_cache(tokens=4)
    request = _make_request("organic-flow", list(range(9)), cache)
    scheduler.requests[request.request_id] = request
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )

    candidate = scheduler._adaptive_chunk_size(
        4,
        request_id=request.request_id,
        loop_label="external",
        kv_len=4,
        prefill_context=context,
    )
    assert candidate == 4

    with pytest.raises(_PrefillEvictionNeeded) as exc:
        scheduler._guard_prefill_chunk(
            candidate,
            kv_len=4,
            progress=4,
            loop_label="external",
            request_id=request.request_id,
            request=request,
            prompt_cache=cache,
            prefill_context=context,
        )

    assert exc.value.request.reason == "turboquant_mid_prefill"
    assert request.prefill_eviction_retries == 1
    assert context.conversion_attempted is False
    assert request.turboquant_mid_prefill_attempted is False
    assert all(isinstance(cache_obj, KVCache) for cache_obj in cache)

    retry_candidate = scheduler._adaptive_chunk_size(
        4,
        request_id=request.request_id,
        loop_label="external",
        kv_len=4,
        prefill_context=context,
    )
    assert retry_candidate == 4

    result = scheduler._guard_prefill_chunk(
        retry_candidate,
        kv_len=4,
        progress=4,
        loop_label="external",
        request_id=request.request_id,
        request=request,
        prompt_cache=cache,
        prefill_context=context,
    )

    assert result == 4
    assert context.phase is _PrefillKVPhase.TURBOQUANT
    assert context.trigger_tokens == 4
    assert request.turboquant_mid_prefill_attempted is True
    assert context.memory_after_bytes < cap
    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[1], KVCache)


def test_empty_fresh_cache_resizes_without_conversion() -> None:
    scheduler = _make_scheduler(step_size=4)
    _configure_sizing_band(scheduler)
    cache = _dense_cache()
    request = _make_request("empty-target-band", list(range(9)), cache)
    scheduler.requests[request.request_id] = request
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )

    candidate = scheduler._adaptive_chunk_size(
        4,
        request_id=request.request_id,
        loop_label="external",
        kv_len=0,
        prefill_context=context,
    )
    assert candidate == 4
    result = scheduler._guard_prefill_chunk(
        candidate,
        kv_len=0,
        progress=0,
        loop_label="external",
        request_id=request.request_id,
        request=request,
        prompt_cache=cache,
        prefill_context=context,
    )

    assert result == 2
    assert context.phase is _PrefillKVPhase.DENSE
    assert context.conversion_attempted is False
    assert request.turboquant_mid_prefill_attempted is False
    assert all(cache_obj.empty() for cache_obj in cache)


def test_flag_off_guard_never_converts() -> None:
    scheduler = _make_scheduler()
    _configure_pressure(scheduler, pressure_after_tokens=0)
    scheduler._turboquant_mid_prefill = False
    cache = _dense_cache(tokens=4)
    request = _make_request("off", list(range(9)), cache)
    request.prefill_eviction_retries = 1
    scheduler.requests[request.request_id] = request
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )

    with pytest.raises(PrefillMemoryExceededError):
        scheduler._guard_prefill_chunk(
            4,
            kv_len=4,
            progress=4,
            loop_label="external",
            request_id=request.request_id,
            request=request,
            prompt_cache=cache,
            prefill_context=context,
        )
    assert context.conversion_attempted is False
    assert request.turboquant_mid_prefill_attempted is False
    assert all(isinstance(cache_obj, KVCache) for cache_obj in cache)


def test_flag_off_finalization_uses_ordinary_direct_converter() -> None:
    scheduler = _make_scheduler()
    scheduler._turboquant_mid_prefill = False
    cache = _dense_cache(tokens=4)
    request = _make_request("off-final", list(range(5)), cache)
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )

    def _unexpected(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("mid-prefill-only path ran while disabled")

    scheduler._classify_prefill_cache = _unexpected
    scheduler._apply_turboquant_kv_convert_sliced = _unexpected
    scheduler._run_guarded_turboquant_conversion = _unexpected
    result = scheduler._finalize_turboquant_prefill_cache(
        request,
        cache,
        processed_tokens=4,
        context=context,
    )

    assert result is None
    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[1], KVCache)
    assert context.phase is _PrefillKVPhase.DENSE


def test_flag_off_restored_prefix_uses_turboquant_suffix_workspace() -> None:
    scheduler = _make_scheduler(step_size=2048)
    scheduler._turboquant_mid_prefill = False
    scheduler._turboquant_kv_bits = 8.0
    scheduler._set_model_info_for_monitor()
    assert scheduler.memory_monitor is not None
    scheduler.memory_monitor.set_model_info(
        num_layers=64,
        num_kv_heads=4,
        head_dim=256,
        dtype_size=turboquant_mse_bytes_per_element(256, 8.0),
        num_attention_heads=24,
        num_kv_cache_layers=8,
        compute_dtype_size=2,
    )
    cache = _dense_cache(tokens=4)
    scheduler._apply_turboquant_kv_convert(cache)
    first_layer = cache[0]
    request = _make_request("off-restored", [4, 5, 6], cache)
    request.cached_tokens = 4
    scheduler.requests[request.request_id] = request
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )
    phases: list[_PrefillKVPhase] = []
    original_bound = scheduler._admission_transient_bound

    def _track_bound(
        self: Scheduler,
        n_tokens: int,
        kv_len: int,
        *,
        phase: _PrefillKVPhase = _PrefillKVPhase.DENSE,
    ) -> float:
        del self
        phases.append(phase)
        return original_bound(n_tokens, kv_len, phase=phase)

    def _current_usage(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del self, refresh_mlx_active
        return 0

    scheduler._admission_transient_bound = MethodType(_track_bound, scheduler)
    scheduler._current_usage_bytes = MethodType(_current_usage, scheduler)
    scheduler._memory_hard_limit_bytes = 16 * 1024**3
    scheduler._memory_abort_limit_bytes = 16 * 1024**3
    scheduler._prefill_abort_margin = 1.0

    admitted = scheduler._guard_prefill_chunk(
        2048,
        kv_len=131071 - 2048,
        progress=0,
        loop_label="external",
        request_id=request.request_id,
        request=request,
        prompt_cache=cache,
        prefill_context=context,
    )
    result, last_token = scheduler._do_external_prefill(
        request,
        [4, 5, 6],
        cache,
    )

    assert admitted == 2048
    assert phases and all(phase is _PrefillKVPhase.TURBOQUANT for phase in phases)
    assert context.phase is _PrefillKVPhase.TURBOQUANT
    assert context.conversion_eligible is False
    assert context.started_at is None
    assert request.turboquant_mid_prefill_attempted is False
    assert result is cache
    assert last_token == [6]
    assert cache[0] is first_layer
    assert cache[0].offset == cache[1].offset == 6


def test_mid_prefill_defers_dense_preflight_to_guard() -> None:
    scheduler = _make_scheduler(step_size=4)
    cap = _configure_pressure(scheduler, pressure_after_tokens=0)
    scheduler._memory_hard_watermark_bytes = cap
    scheduler._prefill_memory_guard = True
    scheduler._prefill_eviction_callback_configured = False
    request = _make_request("deferred-preflight", list(range(9)))
    dense_estimate = scheduler._admission_estimate(
        num_prompt_tokens=request.num_prompt_tokens,
        cached_tokens=0,
        current=scheduler._current_usage_bytes(),
        phase=_PrefillKVPhase.DENSE,
    )
    assert dense_estimate is not None
    assert dense_estimate.estimated > cap

    assert scheduler._preflight_memory_check(request) is None
    scheduler.preflight_or_raise(
        num_prompt_tokens=request.num_prompt_tokens,
        request_id=request.request_id,
    )

    request.turboquant_mid_prefill_attempted = True
    assert scheduler._preflight_memory_check(request) is not None
    request.turboquant_mid_prefill_attempted = False

    scheduler._turboquant_mid_prefill = False
    assert scheduler._preflight_memory_check(request) is not None
    with pytest.raises(PrefillMemoryExceededError):
        scheduler.preflight_or_raise(
            num_prompt_tokens=request.num_prompt_tokens,
            request_id=request.request_id,
        )


def test_conversion_failure_discards_every_cache_reference() -> None:
    scheduler = _make_scheduler()
    cap = _configure_pressure(scheduler)
    cache = _dense_cache(tokens=4, layers=3)
    request = _make_request("failure", list(range(9)), cache)
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )

    def _partially_fail(
        prompt_cache: list[Any],
        *,
        check_cancelled: Any = None,
        log_result: bool = True,
    ) -> TurboQuantConversionStats:
        del check_cancelled, log_result
        prompt_cache[0] = TurboQuantKVCache.from_cache(prompt_cache[0], bits=4.0)
        raise RuntimeError("injected conversion failure")

    scheduler._apply_turboquant_kv_convert_sliced = _partially_fail
    with pytest.raises(
        PrefillMemoryExceededError,
        match="cache was discarded",
    ) as raised:
        scheduler._attempt_mid_prefill_conversion(
            request=request,
            prompt_cache=cache,
            context=context,
            processed_tokens=4,
            safety_cap=cap,
        )
    assert cache == []
    assert request.prompt_cache is None
    assert request.turboquant_mid_prefill_attempted is True
    assert raised.value.__context__ is None
    assert _conversion_coordinator.snapshot() == (0, 0, False, 0)
    _assert_buffer_access_lock_available()


def test_post_prefill_conversion_failure_discards_partial_cache() -> None:
    scheduler = _make_scheduler()
    _configure_pressure(scheduler)
    cache = _dense_cache(tokens=4, layers=3)
    request = _make_request("post-prefill-failure", list(range(5)), cache)
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )

    def _partially_fail(
        prompt_cache: list[Any],
        *,
        check_cancelled: Any = None,
        log_result: bool = True,
    ) -> TurboQuantConversionStats:
        del check_cancelled, log_result
        prompt_cache[0] = TurboQuantKVCache.from_cache(prompt_cache[0], bits=4.0)
        raise RuntimeError("injected post-prefill conversion failure")

    scheduler._apply_turboquant_kv_convert_sliced = _partially_fail
    with pytest.raises(
        PrefillMemoryExceededError,
        match="post-prefill conversion failed.*cache was discarded",
    ) as raised:
        scheduler._finalize_turboquant_prefill_cache(
            request,
            cache,
            processed_tokens=4,
            context=context,
        )

    assert cache == []
    assert request.prompt_cache is None
    assert request.turboquant_mid_prefill_attempted is False
    assert raised.value.__context__ is None
    assert _conversion_coordinator.snapshot() == (0, 0, False, 0)
    _assert_buffer_access_lock_available()


def test_failure_reclaims_partial_arrays_before_metal_clear(
    monkeypatch: pytest.MonkeyPatch,
    caplog: Any,
) -> None:
    scheduler = _make_scheduler()
    cap = _configure_pressure(scheduler)
    cache = _dense_cache(tokens=4, layers=3)
    request = _make_request("reclaim-partial", list(range(9)), cache)
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )
    partial_refs: list[weakref.ReferenceType[_PartialArrayOwner]] = []
    collected_after_release: list[bool] = []
    cleared_after_collect: list[bool] = []

    def _fail_with_partial(
        prompt_cache: list[Any],
        *,
        check_cancelled: Any = None,
        log_result: bool = True,
    ) -> TurboQuantConversionStats:
        del check_cancelled, log_result
        partial = _PartialArrayOwner(mx.ones((8,), dtype=mx.float16))
        partial_refs.append(weakref.ref(partial))
        prompt_cache.append(partial)
        raise RuntimeError("injected partial-array failure")

    def _collect() -> int:
        collected_after_release.append(bool(partial_refs and partial_refs[0]() is None))
        return 0

    def _clear(stream: Any | None = None) -> None:
        del stream
        cleared_after_collect.append(collected_after_release == [True])

    scheduler._apply_turboquant_kv_convert_sliced = _fail_with_partial
    monkeypatch.setattr("omlx.scheduler.gc.collect", _collect)
    monkeypatch.setattr("omlx.scheduler._sync_and_clear_cache", _clear)
    with (
        caplog.at_level("ERROR", logger="omlx.scheduler"),
        pytest.raises(PrefillMemoryExceededError) as raised,
    ):
        scheduler._attempt_mid_prefill_conversion(
            request=request,
            prompt_cache=cache,
            context=context,
            processed_tokens=4,
            safety_cap=cap,
        )

    assert partial_refs[0]() is None
    assert collected_after_release == [True]
    assert cleared_after_collect == [True]
    assert cache == []
    assert request.prompt_cache is None
    assert raised.value.__context__ is None
    traceback_names: set[str] = set()
    current_traceback = raised.value.__traceback__
    while current_traceback is not None:
        traceback_names.add(current_traceback.tb_frame.f_code.co_name)
        current_traceback = current_traceback.tb_next
    assert "_fail_with_partial" not in traceback_names
    assert caplog.records
    assert all(record.exc_info is None for record in caplog.records)
    assert not any(
        isinstance(argument, BaseException)
        for record in caplog.records
        for argument in (
            record.args if isinstance(record.args, tuple) else (record.args,)
        )
    )
    assert _conversion_coordinator.snapshot() == (0, 0, False, 0)
    _assert_buffer_access_lock_available()


@pytest.mark.parametrize("path", ["external", "chunked"])
def test_prefill_paths_surface_conversion_failure_and_clear_cache(path: str) -> None:
    scheduler = _make_scheduler(step_size=4)
    _configure_pressure(scheduler)
    cache = _dense_cache()
    tokens = list(range(9))
    request = _make_request(f"{path}-failure", tokens, cache)
    request.prefill_eviction_retries = 1
    scheduler.requests[request.request_id] = request

    def _partially_fail(
        prompt_cache: list[Any],
        *,
        check_cancelled: Any = None,
        log_result: bool = True,
    ) -> TurboQuantConversionStats:
        del check_cancelled, log_result
        prompt_cache[0] = TurboQuantKVCache.from_cache(prompt_cache[0], bits=4.0)
        raise RuntimeError("injected path failure")

    scheduler._apply_turboquant_kv_convert_sliced = _partially_fail
    if path == "external":
        with pytest.raises(PrefillMemoryExceededError, match="cache was discarded"):
            scheduler._do_external_prefill(request, tokens, cache)
    else:
        state = scheduler._begin_prefill(request, tokens, cache)
        assert scheduler._step_prefill_chunk(state) is False
        with pytest.raises(PrefillMemoryExceededError, match="cache was discarded"):
            scheduler._step_prefill_chunk(state)

    assert cache == []
    assert request.prompt_cache is None


@pytest.mark.parametrize("path", ["external", "chunked"])
def test_post_conversion_oom_requeue_does_not_repeat_mid_prefill_conversion(
    path: str,
) -> None:
    scheduler = _make_scheduler(step_size=4)
    cap = _configure_pressure(scheduler)
    scheduler._prefill_eviction_callback_configured = False
    scheduler._memory_limit_bytes = cap // 2
    cache = _dense_cache()
    tokens = list(range(9))
    request = _make_request(f"{path}-post-conversion-oom", tokens, cache)
    scheduler.requests[request.request_id] = request
    original_converter = scheduler._apply_turboquant_kv_convert_sliced
    conversion_calls = 0

    def _tracking_convert(
        prompt_cache: list[Any],
        *,
        check_cancelled: Any = None,
        log_result: bool = True,
    ) -> TurboQuantConversionStats:
        nonlocal conversion_calls
        assert request.turboquant_mid_prefill_attempted is True
        conversion_calls += 1
        return original_converter(
            prompt_cache,
            check_cancelled=check_cancelled,
            log_result=log_result,
        )

    def _current_usage(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del refresh_mlx_active
        return cap * 2 if self.model.calls >= 2 else 1024**2

    def _reclaim(self: Scheduler) -> int:
        return self._current_usage_bytes()

    scheduler._apply_turboquant_kv_convert_sliced = _tracking_convert
    scheduler._current_usage_bytes = MethodType(_current_usage, scheduler)
    scheduler._reclaim_prefill_headroom = MethodType(_reclaim, scheduler)

    if path == "external":
        with pytest.raises(
            RuntimeError,
            match="Memory limit exceeded during prefill",
        ) as raised:
            scheduler._do_external_prefill(request, tokens, cache)
        scheduler.requests.pop(request.request_id)
        assert scheduler._requeue_or_fail_prefill(request, raised.value) is True
    else:
        state = scheduler._begin_prefill(request, tokens, cache)
        assert scheduler._step_prefill_chunk(state) is False
        scheduler.prefilling.append(request)
        scheduler._prefill_states[request.request_id] = state
        scheduled: list[Request] = []
        rejected: list[Any] = []
        scheduler._advance_chunked_prefills(scheduled, rejected)
        assert scheduled == []
        assert rejected == []
        assert request.request_id not in scheduler._prefill_states

    assert conversion_calls == 1
    assert request.turboquant_mid_prefill_attempted is True
    assert request.prefill_oom_retries == 1
    assert request.prompt_cache is None
    assert scheduler.waiting[0] is request

    scheduler.waiting.clear()
    scheduler.model.calls = 0
    retry_cache = _dense_cache()
    request.prompt_cache = retry_cache
    retry_context = scheduler._new_prefill_context(
        request,
        retry_cache,
        loop_label=f"{path}_retry",
    )
    assert retry_context.phase is _PrefillKVPhase.DENSE
    assert retry_context.conversion_attempted is True
    assert (
        scheduler._mid_prefill_conversion_available(
            retry_context,
            retry_cache,
            request,
        )
        is False
    )

    if path == "external":
        with pytest.raises(PrefillMemoryExceededError):
            scheduler._do_external_prefill(request, tokens, retry_cache)
    else:
        retry_state = scheduler._begin_prefill(request, tokens, retry_cache)
        assert scheduler._step_prefill_chunk(retry_state) is False
        with pytest.raises(PrefillMemoryExceededError):
            scheduler._step_prefill_chunk(retry_state)

    assert conversion_calls == 1
    assert all(isinstance(cache_obj, KVCache) for cache_obj in retry_cache)


def test_cancelled_conversion_clears_request_cache() -> None:
    scheduler = _make_scheduler()
    cache = _dense_cache(tokens=4)
    request = _make_request("cancelled-conversion", list(range(9)), cache)
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )
    scheduler._pending_abort_ids.add(request.request_id)

    with pytest.raises(_PrefillAbortedError) as raised:
        scheduler._attempt_mid_prefill_conversion(
            request=request,
            prompt_cache=cache,
            context=context,
            processed_tokens=4,
            safety_cap=0,
        )

    assert cache == []
    assert request.prompt_cache is None
    assert request.turboquant_mid_prefill_attempted is True
    assert raised.value.__context__ is None
    assert _conversion_coordinator.snapshot() == (0, 0, False, 0)
    _assert_buffer_access_lock_available()


def test_mid_prefill_attempt_state_is_request_scoped() -> None:
    scheduler = _make_scheduler()
    first_cache = _dense_cache(tokens=4)
    second_cache = _dense_cache(tokens=4)
    first = _make_request("first-context", list(range(9)), first_cache)
    second = _make_request("second-context", list(range(9)), second_cache)
    first_context = scheduler._new_prefill_context(
        first,
        first_cache,
        loop_label="external",
    )
    second_context = scheduler._new_prefill_context(
        second,
        second_cache,
        loop_label="chunked_step",
    )

    scheduler._attempt_mid_prefill_conversion(
        request=first,
        prompt_cache=first_cache,
        context=first_context,
        processed_tokens=4,
        safety_cap=0,
    )
    assert first_context.conversion_attempted is True
    assert first_context.mid_triggered is True
    assert first.turboquant_mid_prefill_attempted is True
    retry_cache = _dense_cache(tokens=4)
    retry_context = scheduler._new_prefill_context(
        first,
        retry_cache,
        loop_label="external_retry",
    )
    assert retry_context.started_at == first_context.started_at
    assert retry_context.conversion_attempted is True
    assert first.prefill_started_at == first_context.started_at
    assert (
        scheduler._mid_prefill_conversion_available(
            retry_context,
            retry_cache,
            first,
        )
        is False
    )
    assert second_context.conversion_attempted is False
    assert second_context.mid_triggered is False
    assert second_context.phase is _PrefillKVPhase.DENSE
    assert second.turboquant_mid_prefill_attempted is False

    scheduler._attempt_mid_prefill_conversion(
        request=second,
        prompt_cache=second_cache,
        context=second_context,
        processed_tokens=4,
        safety_cap=0,
    )
    assert second_context.conversion_attempted is True
    assert second_context.mid_triggered is True
    assert second.turboquant_mid_prefill_attempted is True


def test_summary_uses_post_conversion_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
    caplog: Any,
) -> None:
    scheduler = _make_scheduler()
    cache = _dense_cache(tokens=4)
    request = _make_request("timed-summary", list(range(9)), cache)
    converted_first = TurboQuantKVCache.from_cache(cache[0], bits=4.0)
    mx.eval(converted_first.keys, converted_first.values)
    clock_values = iter((10.0, 12.0, 14.0, 20.0))
    observed_clock_values: list[float] = []
    events: list[str] = []

    def _clock() -> float:
        value = next(clock_values)
        observed_clock_values.append(value)
        events.append(f"clock:{value}")
        return value

    def _current_usage(
        self: Scheduler,
        refresh_mlx_active: bool = True,
    ) -> int:
        del self, refresh_mlx_active
        return 1024**3

    def _sync_and_clear(stream: Any) -> None:
        del stream
        events.append("sync")

    def _convert(
        prompt_cache: list[Any],
        *,
        check_cancelled: Any = None,
        log_result: bool = True,
    ) -> TurboQuantConversionStats:
        del check_cancelled, log_result
        events.append("convert")
        prompt_cache[0] = converted_first
        return TurboQuantConversionStats(
            converted_layers=1,
            already_quantized_layers=0,
            skipped_dense_layers=1,
            slices=1,
            source_bytes=0,
            converted_bytes=0,
        )

    monkeypatch.setattr("omlx.scheduler.time.perf_counter", _clock)
    monkeypatch.setattr("omlx.scheduler._sync_and_clear_cache", _sync_and_clear)
    scheduler._current_usage_bytes = MethodType(_current_usage, scheduler)
    scheduler._apply_turboquant_kv_convert_sliced = _convert
    first_context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external",
    )
    context = scheduler._new_prefill_context(
        request,
        cache,
        loop_label="external_retry",
    )
    assert context.started_at == first_context.started_at == 10.0

    with caplog.at_level("INFO", logger="omlx.scheduler"):
        scheduler._attempt_mid_prefill_conversion(
            request=request,
            prompt_cache=cache,
            context=context,
            processed_tokens=4,
            safety_cap=0,
        )
        context.post_trigger_tokens = 4
        scheduler._log_mid_prefill_summary(context)

    trigger_log = next(
        message for message in caplog.messages if "mid-prefill trigger" in message
    )
    summary_log = next(
        message for message in caplog.messages if "mid-prefill complete" in message
    )
    assert observed_clock_values == [10.0, 12.0, 14.0, 20.0]
    assert events == [
        "clock:10.0",
        "clock:12.0",
        "convert",
        "sync",
        "clock:14.0",
        "clock:20.0",
    ]
    assert context.conversion_completed_at == 14.0
    assert "conversion_pause=2.000s" in summary_log
    assert "post_trigger_prefill_tps=0.67 tok/s" in summary_log
    assert "total_wall=10.000s" in summary_log
    assert "GiB" in trigger_log
    assert "GiB" in summary_log


def test_contexts_and_pretrigger_chunks_avoid_telemetry_clocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled = _make_scheduler(step_size=4)
    enabled_cache = _dense_cache()
    enabled_request = _make_request(
        "enabled-pretrigger",
        list(range(9)),
        enabled_cache,
    )
    enabled_state = enabled._begin_prefill(
        enabled_request,
        list(range(9)),
        enabled_cache,
    )
    assert enabled_state.prefill_context is not None
    assert enabled_state.prefill_context.started_at is not None

    disabled = _make_scheduler(step_size=4)
    disabled._turboquant_mid_prefill = False
    disabled_cache = _dense_cache()
    disabled_request = _make_request(
        "disabled-clock",
        list(range(9)),
        disabled_cache,
    )

    restored = _make_scheduler(step_size=4)
    restored_cache = _dense_cache(tokens=4)
    restored._apply_turboquant_kv_convert(restored_cache)
    restored_request = _make_request(
        "noneligible-clock",
        [4, 5, 6],
        restored_cache,
    )

    clock_calls = 0

    def _unexpected_clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        raise AssertionError("prefill chunk read the telemetry clock")

    monkeypatch.setattr(
        "omlx.scheduler.time.perf_counter",
        _unexpected_clock,
    )
    assert enabled._step_prefill_chunk(enabled_state) is False

    disabled_state = disabled._begin_prefill(
        disabled_request,
        list(range(9)),
        disabled_cache,
    )
    assert disabled_state.prefill_context is not None
    assert disabled_state.prefill_context.started_at is None
    assert disabled._step_prefill_chunk(disabled_state) is False

    restored_state = restored._begin_prefill(
        restored_request,
        [4, 5, 6],
        restored_cache,
    )
    assert restored_state.prefill_context is not None
    assert restored_state.prefill_context.started_at is None
    assert restored._step_prefill_chunk(restored_state) is True
    assert clock_calls == 0


def test_external_prefill_converts_once_and_reports_metrics(caplog: Any) -> None:
    scheduler = _make_scheduler(step_size=4)
    _configure_pressure(scheduler)
    cache = _dense_cache()
    tokens = list(range(9))
    request = _make_request("external", tokens, cache)
    request.prefill_eviction_retries = 1
    scheduler.requests[request.request_id] = request

    with caplog.at_level("INFO", logger="omlx.scheduler"):
        result, last_token = scheduler._do_external_prefill(
            request,
            tokens,
            cache,
        )

    assert result is cache
    assert last_token == [8]
    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[1], KVCache)
    assert cache[0].offset == cache[1].offset == 8
    trigger_logs = [m for m in caplog.messages if "mid-prefill trigger" in m]
    summary_logs = [m for m in caplog.messages if "mid-prefill complete" in m]
    assert len(trigger_logs) == 1
    assert len(summary_logs) == 1
    assert "post_trigger_tokens=4" in summary_logs[0]
    assert "conversion_pause=" in summary_logs[0]
    assert "post_trigger_prefill_tps=" in summary_logs[0]
    assert "GiB" in trigger_logs[0]
    assert "GiB" in summary_logs[0]
    assert "total_wall=" in summary_logs[0]


def test_below_trigger_feature_is_cache_output_equivalent() -> None:
    enabled = _make_scheduler(step_size=4)
    disabled = _make_scheduler(step_size=4)
    disabled._turboquant_mid_prefill = False
    sliced_converter = MagicMock(wraps=enabled._apply_turboquant_kv_convert_sliced)
    enabled._apply_turboquant_kv_convert_sliced = sliced_converter

    def _unexpected_direct_converter(prompt_cache: list[Any]) -> None:
        del prompt_cache
        raise AssertionError("feature-on finalization used the unbounded converter")

    enabled._apply_turboquant_kv_convert = _unexpected_direct_converter
    tokens = list(range(9))
    enabled_cache = _dense_cache()
    disabled_cache = _dense_cache()
    enabled_request = _make_request("below-enabled", tokens, enabled_cache)
    disabled_request = _make_request("below-disabled", tokens, disabled_cache)
    enabled.requests[enabled_request.request_id] = enabled_request
    disabled.requests[disabled_request.request_id] = disabled_request

    enabled_result, enabled_last = enabled._do_external_prefill(
        enabled_request,
        tokens,
        enabled_cache,
    )
    disabled_result, disabled_last = disabled._do_external_prefill(
        disabled_request,
        tokens,
        disabled_cache,
    )

    assert enabled_result is enabled_cache
    assert disabled_result is disabled_cache
    assert enabled_last == disabled_last == [8]
    assert isinstance(enabled_cache[0], TurboQuantKVCache)
    assert isinstance(disabled_cache[0], TurboQuantKVCache)
    enabled_keys, enabled_values = enabled_cache[0].state
    disabled_keys, disabled_values = disabled_cache[0].state
    assert _state_equal(enabled_keys, disabled_keys)
    assert _state_equal(enabled_values, disabled_values)
    assert enabled_request.turboquant_mid_prefill_attempted is False
    assert disabled_request.turboquant_mid_prefill_attempted is False
    assert sliced_converter.call_count == 1
    assert mx.array_equal(enabled_cache[1].state[0], disabled_cache[1].state[0]).item()
    assert mx.array_equal(enabled_cache[1].state[1], disabled_cache[1].state[1]).item()


def test_mid_prefill_attempt_flag_does_not_block_post_prefill_conversion() -> None:
    scheduler = _make_scheduler(step_size=4)
    cache = _dense_cache()
    tokens = list(range(9))
    request = _make_request("post-prefill-after-attempt", tokens, cache)
    request.turboquant_mid_prefill_attempted = True
    scheduler.requests[request.request_id] = request

    result, last_token = scheduler._do_external_prefill(
        request,
        tokens,
        cache,
    )

    assert result is cache
    assert last_token == [8]
    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[1], KVCache)
    assert request.turboquant_mid_prefill_attempted is True


def test_pressure_conversion_matches_post_prefill_cache_and_attention() -> None:
    pressured = _make_scheduler(step_size=4)
    normal = _make_scheduler(step_size=4)
    _configure_pressure(pressured)
    tokens = list(range(9))
    pressured_cache = _dense_cache()
    normal_cache = _dense_cache()
    pressured_request = _make_request("quality-pressured", tokens, pressured_cache)
    normal_request = _make_request("quality-normal", tokens, normal_cache)
    pressured_request.prefill_eviction_retries = 1
    pressured.requests[pressured_request.request_id] = pressured_request
    normal.requests[normal_request.request_id] = normal_request

    pressured._do_external_prefill(pressured_request, tokens, pressured_cache)
    normal._do_external_prefill(normal_request, tokens, normal_cache)

    assert isinstance(pressured_cache[0], TurboQuantKVCache)
    assert isinstance(normal_cache[0], TurboQuantKVCache)
    pressured_keys, pressured_values = pressured_cache[0].state
    normal_keys, normal_values = normal_cache[0].state
    assert _state_equal(pressured_keys, normal_keys)
    assert _state_equal(pressured_values, normal_values)
    assert mx.array_equal(pressured_cache[1].state[0], normal_cache[1].state[0]).item()
    assert mx.array_equal(pressured_cache[1].state[1], normal_cache[1].state[1]).item()

    queries = mx.ones((1, 2, 1, 32), dtype=mx.float16)
    pressured_attention = pressured_cache[0].decode_attention(
        queries,
        scale=32**-0.5,
    )
    normal_attention = normal_cache[0].decode_attention(
        queries,
        scale=32**-0.5,
    )
    mx.eval(pressured_attention, normal_attention)
    assert mx.allclose(pressured_attention, normal_attention).item()


def test_chunked_prefill_converts_once_and_inserts() -> None:
    scheduler = _make_scheduler(step_size=4)
    _configure_pressure(scheduler)
    cache = _dense_cache()
    tokens = list(range(9))
    request = _make_request("chunked", tokens, cache)
    request.prefill_eviction_retries = 1
    scheduler.requests[request.request_id] = request
    state = scheduler._begin_prefill(request, tokens, cache)

    assert scheduler._step_prefill_chunk(state) is False
    assert scheduler._step_prefill_chunk(state) is True
    assert state.prefill_context is not None
    assert state.prefill_context.phase is _PrefillKVPhase.TURBOQUANT
    assert state.prefill_context.post_trigger_tokens == 4

    batch_generator = MagicMock()
    batch_generator.insert.return_value = [77]
    scheduler.batch_generator = batch_generator
    scheduled: list[Request] = []
    scheduler._insert_prefilled_request(request, state, scheduled)
    assert scheduled == [request]
    assert scheduler.running[request.request_id] is request
    assert state.prefill_context.summary_logged is True


def test_complete_turboquant_prefix_extends_without_reconversion() -> None:
    scheduler = _make_scheduler(step_size=4)
    scheduler._memory_hard_limit_bytes = 0
    scheduler._memory_abort_limit_bytes = 0
    cache = _dense_cache(tokens=4)
    scheduler._apply_turboquant_kv_convert(cache)
    first_layer = cache[0]
    request = _make_request("restore", [4, 5, 6], cache)
    request.cached_tokens = 4
    scheduler.requests[request.request_id] = request

    result, last_token = scheduler._do_external_prefill(
        request,
        [4, 5, 6],
        cache,
    )
    assert result is cache
    assert last_token == [6]
    assert cache[0] is first_layer
    assert scheduler._classify_prefill_cache(cache)[0] is _PrefillKVPhase.TURBOQUANT
    assert cache[0].offset == cache[1].offset == 6


def test_qwen_style_hybrid_turboquant_prefix_restores_and_extends() -> None:
    scheduler = _make_scheduler(step_size=4)
    scheduler._memory_hard_limit_bytes = 0
    scheduler._memory_abort_limit_bytes = 0
    first = KVCache()
    recurrent = ArraysCache(1)
    last = KVCache()
    _append_dense(first, tokens=4, value=1.0)
    _append_dense(last, tokens=4, value=2.0)
    recurrent[0] = mx.ones((1, 2, 32), dtype=mx.float16)
    cache: list[Any] = [first, recurrent, last]
    scheduler._apply_turboquant_kv_convert(cache)
    converted_first = cache[0]
    recurrent_state = recurrent[0]
    request = _make_request("qwen-hybrid-restore", [4, 5, 6], cache)
    request.cached_tokens = 4
    scheduler.requests[request.request_id] = request

    result, last_token = scheduler._do_external_prefill(
        request,
        [4, 5, 6],
        cache,
    )

    assert result is cache
    assert last_token == [6]
    assert cache[0] is converted_first
    assert cache[1] is recurrent
    assert recurrent[0] is recurrent_state
    assert isinstance(cache[0], TurboQuantKVCache)
    assert isinstance(cache[2], KVCache)
    assert cache[0].offset == cache[2].offset == 6
    assert scheduler._classify_prefill_cache(cache)[0] is _PrefillKVPhase.TURBOQUANT


def test_phase_widths_and_transient_histories_remain_separate() -> None:
    scheduler = _make_scheduler()
    assert scheduler._prefill_dense_kv_dtype_size == 2.0
    assert scheduler._prefill_tq_kv_dtype_size is not None
    assert scheduler._prefill_tq_kv_dtype_size < scheduler._prefill_dense_kv_dtype_size
    expected_tq_width = (turboquant_mse_bytes_per_element(32, 4.0) + 2.0) / 2
    assert scheduler._prefill_tq_kv_dtype_size == expected_tq_width

    dense = scheduler._admission_estimate(
        num_prompt_tokens=1024,
        cached_tokens=0,
        current=0,
        phase=_PrefillKVPhase.DENSE,
    )
    turboquant = scheduler._admission_estimate(
        num_prompt_tokens=1024,
        cached_tokens=0,
        current=0,
        phase=_PrefillKVPhase.TURBOQUANT,
    )
    assert dense is not None and turboquant is not None
    assert turboquant.kv_exact < dense.kv_exact

    scheduler._prefill_min_chunk_tokens = 1
    scheduler._record_chunk_transient(
        1,
        0,
        100,
        request_id="dense",
        loop_label="unit",
        phase=_PrefillKVPhase.DENSE,
    )
    scheduler._record_chunk_transient(
        1,
        0,
        200,
        request_id="tq",
        loop_label="unit",
        phase=_PrefillKVPhase.TURBOQUANT,
    )
    assert scheduler._prefill_transient_tracker.samples == 1
    assert scheduler._prefill_tq_transient_tracker.samples == 1
    assert scheduler._prefill_transient_tracker.last_delta_bytes == 100
    assert scheduler._prefill_tq_transient_tracker.last_delta_bytes == 200


def test_first_qwen_q8_suffix_uses_structural_workspace_bound() -> None:
    workspace = estimate_turboquant_prefill_attention_workspace_bytes(
        query_tokens=2048,
        kv_len=131071,
        num_query_heads=24,
        num_kv_heads=4,
        head_dim=256,
        bits=8.0,
        compute_dtype_size=2,
        causal=True,
    )
    assert workspace == 2_687_666_176

    scheduler = _make_scheduler(step_size=2048)
    scheduler._turboquant_kv_bits = 8.0
    assert scheduler.memory_monitor is not None
    scheduler.memory_monitor.set_model_info(
        num_layers=64,
        num_kv_heads=4,
        head_dim=256,
        dtype_size=turboquant_mse_bytes_per_element(256, 8.0),
        num_attention_heads=24,
        num_kv_cache_layers=8,
        compute_dtype_size=2,
    )
    predicted = scheduler._predicted_chunk_transient(
        2048,
        131071 - 2048,
        phase=_PrefillKVPhase.TURBOQUANT,
    )
    admission = scheduler._admission_transient_bound(
        2048,
        131071 - 2048,
        phase=_PrefillKVPhase.TURBOQUANT,
    )

    assert predicted >= workspace * scheduler._PREFILL_TRANSIENT_SAFETY
    assert admission >= predicted
