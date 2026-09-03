# SPDX-License-Identifier: Apache-2.0
"""Safety, lifecycle, and isolation gates for latent Metal keepwarm."""

from __future__ import annotations

import concurrent.futures
import threading
from contextlib import contextmanager, suppress
from types import SimpleNamespace
from unittest.mock import MagicMock

import omlx.engine_core as engine_core_module
from omlx.engine_core import EngineCore
from omlx.keepwarm import (
    CompiledMetalKeepwarmTouch,
    KeepwarmAction,
    KeepwarmConfig,
    KeepwarmController,
    MetalKeepwarmTouchResult,
)


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def config(**overrides) -> KeepwarmConfig:
    values = {
        "enabled": True,
        "interval_seconds": 10.0,
        "idle_after_seconds": 2.0,
        "matrix_size": 1,
        "repeats": 1,
        "request_start_enabled": True,
        "request_start_idle_seconds": 2.0,
        "request_start_matrix_size": 128,
        "post_response_enabled": True,
        "post_response_delay_seconds": 5.0,
        "post_response_matrix_size": 128,
        "large_cache_tokens": 8192,
        "large_cache_interval_seconds": 60.0,
        "slow_threshold_seconds": 1.0,
        "slow_backoff_seconds": 60.0,
    }
    values.update(overrides)
    return KeepwarmConfig(**values)


def test_keepwarm_is_default_off(monkeypatch):
    monkeypatch.delenv("OMLX_KEEPWARM", raising=False)
    assert KeepwarmConfig.from_env().enabled is False


def test_local_async_defaults_are_physically_qualified(monkeypatch):
    for name in (
        "OMLX_KEEPWARM_INTERVAL_SECONDS",
        "OMLX_KEEPWARM_POST_RESPONSE_DELAY_SECONDS",
        "OMLX_KEEPWARM_LARGE_CACHE_INTERVAL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    conservative = KeepwarmConfig.from_env()
    local = KeepwarmConfig.for_local_engine()

    assert (
        conservative.interval_seconds,
        conservative.post_response_delay_seconds,
        conservative.large_cache_interval_seconds,
    ) == (10.0, 5.0, 60.0)
    assert (
        local.interval_seconds,
        local.post_response_delay_seconds,
        local.large_cache_interval_seconds,
    ) == (2.0, 1.0, 2.0)


def test_local_async_cadence_keeps_explicit_environment_overrides(monkeypatch):
    monkeypatch.setenv("OMLX_KEEPWARM_INTERVAL_SECONDS", "3")
    monkeypatch.setenv("OMLX_KEEPWARM_POST_RESPONSE_DELAY_SECONDS", "1.5")
    monkeypatch.setenv("OMLX_KEEPWARM_LARGE_CACHE_INTERVAL_SECONDS", "4")

    local = KeepwarmConfig.for_local_engine()

    assert local.interval_seconds == 3.0
    assert local.post_response_delay_seconds == 1.5
    assert local.large_cache_interval_seconds == 4.0


def test_idle_touch_does_not_arm_before_a_real_request_or_cache():
    clock = Clock()
    controller = KeepwarmController(config(), clock=clock)
    clock.value = 300.0
    assert controller.idle_action(cache_tokens=0) is None
    assert controller.snapshot()["cache_armed"] is False


def test_completed_request_arms_request_start_and_post_response_actions():
    clock = Clock()
    controller = KeepwarmController(config(), clock=clock)
    controller.observe_request_state(True)
    controller.observe_request_state(False, cache_tokens=4096)

    clock.value = 3.0
    request_start = controller.request_start_action()
    assert request_start is not None
    assert request_start.kind == "request_start"
    assert request_start.matrix_size == 128

    clock.value = 4.0
    controller.observe_request_state(False, cache_tokens=4096)
    clock.value = 9.0
    post_response = controller.idle_action(cache_tokens=4096)
    assert post_response is not None
    assert post_response.kind == "post_response"


def test_large_cache_stretches_periodic_interval():
    clock = Clock()
    controller = KeepwarmController(
        config(post_response_enabled=False),
        clock=clock,
    )
    controller.observe_request_state(True)
    controller.observe_request_state(False, cache_tokens=10_000)
    clock.value = 2.0
    first = controller.idle_action(cache_tokens=10_000)
    assert first is not None and first.kind == "idle"
    controller.record(first, elapsed_seconds=0.001, ok=True)

    clock.value = 20.0
    assert controller.idle_action(cache_tokens=10_000) is None
    clock.value = 62.0
    assert controller.idle_action(cache_tokens=10_000) is not None


def test_slow_touch_enters_backoff_and_bounds_failure():
    clock = Clock()
    controller = KeepwarmController(
        config(post_response_enabled=False),
        clock=clock,
    )
    controller.observe_request_state(True)
    controller.observe_request_state(False)
    clock.value = 2.0
    action = controller.idle_action()
    assert action is not None
    controller.record(action, elapsed_seconds=1.5, ok=False, error="x" * 1000)
    clock.value = 30.0
    assert controller.idle_action() is None
    snapshot = controller.snapshot()
    assert snapshot["failures"] == 1
    assert snapshot["slow_count"] == 1
    assert len(snapshot["last_event"]["error"]) == 500


def test_quick_failed_touch_also_enters_backoff():
    clock = Clock()
    controller = KeepwarmController(
        config(post_response_enabled=False),
        clock=clock,
    )
    controller.observe_request_state(True)
    controller.observe_request_state(False)
    clock.value = 2.0
    action = controller.idle_action()
    assert action is not None
    controller.record(action, elapsed_seconds=0.001, ok=False, error="failed")
    clock.value = 30.0
    assert controller.idle_action() is None
    clock.value = 62.0
    assert controller.idle_action() is not None


def test_live_toggle_preserves_cache_arming_and_disable_stops_actions():
    clock = Clock()
    controller = KeepwarmController(config(enabled=False), clock=clock)
    controller.observe_request_state(True)
    controller.observe_request_state(False, cache_tokens=2048)
    clock.value = 3.0
    assert controller.idle_action() is None

    controller.configure(True)
    assert controller.config.prompt_tail_prewarm_enabled is True
    queued = controller.idle_action()
    assert queued is not None
    assert controller.should_execute(queued) is True
    controller.configure(False)
    assert controller.should_execute(queued) is False
    clock.value = 100.0
    assert controller.idle_action() is None


def test_clear_and_shutdown_disarm_without_retaining_cache_state():
    clock = Clock()
    controller = KeepwarmController(config(), clock=clock)
    controller.observe_request_state(True)
    controller.observe_request_state(False, cache_tokens=32_000)
    controller.disarm_cache()
    clock.value = 100.0
    # Stale cache accounting must not undo an explicit clear.
    assert controller.idle_action(cache_tokens=32_000) is None
    assert controller.snapshot()["clear_inhibited"] is True

    controller.observe_request_state(True)
    controller.observe_request_state(False, cache_tokens=32_000)
    clock.value = 102.0
    assert controller.idle_action(cache_tokens=32_000) is not None
    assert controller.snapshot()["clear_inhibited"] is False

    controller.shutdown()
    controller.configure(True)
    assert controller.request_start_action() is None
    assert controller.snapshot()["closed"] is True


class Scheduler:
    def __init__(
        self,
        *,
        busy: bool = False,
        housekeeping: bool = False,
        cache_tokens: int = 0,
        exact_resident_tokens: int = 0,
    ) -> None:
        self.busy = busy
        self.housekeeping = housekeeping
        self.added = []
        self._exact_resident_tokens = exact_resident_tokens
        self.block_aware_cache = SimpleNamespace(
            paged_cache=SimpleNamespace(
                stats=SimpleNamespace(total_tokens_cached=cache_tokens)
            )
        )

    def _exact_resident_stats(self):
        return {"max_token_count": self._exact_resident_tokens}

    def has_requests(self) -> bool:
        return self.busy or self.housekeeping

    def has_active_user_requests(self) -> bool:
        return self.busy

    def add_request(self, request) -> None:
        self.added.append(request)


def core_with_scheduler(scheduler: Scheduler) -> EngineCore:
    core = EngineCore.__new__(EngineCore)
    core.scheduler = scheduler
    core._keepwarm = KeepwarmController(config())
    core.config = SimpleNamespace(keepwarm_config=core._keepwarm.config)
    core._pending_admissions_lock = threading.Lock()
    core._pending_admissions = 0
    core._closed = False
    core._running = True
    core._prompt_tail_lock = threading.Lock()
    core._prompt_tail_plan = None
    core._prompt_tail_epoch = 0
    core._prompt_tail_inflight_epoch = None
    core._prompt_tail_stats = {
        "scheduled": 0,
        "published": 0,
        "skipped": 0,
        "cancelled": 0,
        "failures": 0,
        "last_result": None,
    }
    core._wake_engine_loop = lambda: None
    return core


def test_prompt_tail_plan_is_latest_wins_due_and_cancelled_by_disable():
    core = core_with_scheduler(Scheduler())
    core._keepwarm.configure(True)
    core.config.keepwarm_config = core._keepwarm.config

    assert core.schedule_prompt_tail_prewarm("first") is True
    assert core.schedule_prompt_tail_prewarm("second") is True
    prompt, due, epoch = core._prompt_tail_plan
    assert prompt == "second"
    assert core._take_prompt_tail_prewarm(due - 0.001) is None
    assert core._take_prompt_tail_prewarm(due) == ("second", epoch)
    core._record_prompt_tail_result(epoch, {"status": "skipped"})

    assert core.schedule_prompt_tail_prewarm("third") is True
    core.configure_keepwarm(False)
    assert core._prompt_tail_plan is None
    assert core._prompt_tail_stats["cancelled"] == 1


def test_prompt_tail_waits_for_source_cache_housekeeping():
    scheduler = Scheduler(housekeeping=True)
    core = core_with_scheduler(scheduler)
    core._keepwarm.configure(True)
    core.config.keepwarm_config = core._keepwarm.config
    assert core.schedule_prompt_tail_prewarm("prompt") is True
    _prompt, due, _epoch = core._prompt_tail_plan

    assert core._take_prompt_tail_prewarm(due) is None
    assert core._prompt_tail_plan is not None

    scheduler.housekeeping = False
    assert core._take_prompt_tail_prewarm(due) is not None


def test_prompt_tail_run_forwards_bounded_policy_and_records_result():
    scheduler = Scheduler()
    scheduler.prewarm_prompt_tail = lambda prompt, **kwargs: {
        "status": "published",
        "prompt": prompt,
        **kwargs,
    }
    core = core_with_scheduler(scheduler)
    core._keepwarm.configure(True)
    core.config.keepwarm_config = core._keepwarm.config

    core.schedule_prompt_tail_prewarm([1, 2, 3])
    _prompt, due, epoch = core._prompt_tail_plan
    assert core._take_prompt_tail_prewarm(due) == ([1, 2, 3], epoch)
    result = core._run_prompt_tail_prewarm([1, 2, 3], epoch)
    assert result["status"] == "published"
    assert result["prompt"] == [1, 2, 3]
    assert result["max_suffix_tokens"] == 4096
    assert result["chunk_size"] == 128

    core._record_prompt_tail_result(epoch, result)
    assert core._prompt_tail_stats["published"] == 1
    assert core._prompt_tail_stats["last_result"]["status"] == "published"


def test_prompt_tail_inflight_epoch_is_cancelled_by_clear_or_stop():
    scheduler = Scheduler()
    scheduler.prewarm_prompt_tail = MagicMock(return_value={"status": "published"})
    core = core_with_scheduler(scheduler)
    core._keepwarm.configure(True)
    core.config.keepwarm_config = core._keepwarm.config
    core.schedule_prompt_tail_prewarm("prompt")
    prompt, due, epoch = core._prompt_tail_plan
    assert core._take_prompt_tail_prewarm(due) == (prompt, epoch)

    core.disarm_keepwarm_cache()

    assert core._prompt_tail_abort_requested(epoch) is True
    result = core._run_prompt_tail_prewarm(prompt, epoch)
    assert result["status"] == "skipped"
    scheduler.prewarm_prompt_tail.assert_not_called()


def test_prompt_tail_publish_cannot_race_after_hot_clear():
    scheduler = Scheduler()
    scheduler._exact_resident_cache = SimpleNamespace(put=MagicMock(return_value=True))
    core = core_with_scheduler(scheduler)
    core._keepwarm.configure(True)
    core.config.keepwarm_config = core._keepwarm.config
    core.schedule_prompt_tail_prewarm("prompt")
    prompt, due, epoch = core._prompt_tail_plan
    assert core._take_prompt_tail_prewarm(due) == (prompt, epoch)

    core.disarm_keepwarm_cache()
    published = core._publish_prompt_tail_if_current(
        epoch,
        [1, 2],
        [object()],
        128,
        0,
    )

    assert published is False
    scheduler._exact_resident_cache.put.assert_not_called()


def test_prompt_tail_never_arms_under_b2_b4_or_b6_admission_pressure():
    for concurrency in (2, 4, 6):
        scheduler = Scheduler()
        core = core_with_scheduler(scheduler)
        core._keepwarm.configure(True)
        core.config.keepwarm_config = core._keepwarm.config
        core._pending_admissions = concurrency

        assert core.schedule_prompt_tail_prewarm("prompt") is False
        assert core._prompt_tail_plan is None
        assert core._prompt_tail_stats["last_result"] == {
            "status": "skipped",
            "reason": "request-already-active",
        }


def test_prompt_tail_inflight_aborts_when_batched_streams_become_active():
    scheduler = Scheduler()
    scheduler._exact_resident_cache = SimpleNamespace(put=MagicMock(return_value=True))
    core = core_with_scheduler(scheduler)
    core._keepwarm.configure(True)
    core.config.keepwarm_config = core._keepwarm.config
    assert core.schedule_prompt_tail_prewarm("prompt") is True
    prompt, due, epoch = core._prompt_tail_plan
    assert core._take_prompt_tail_prewarm(due) == (prompt, epoch)

    scheduler.busy = True

    assert core._prompt_tail_abort_requested(epoch) is True
    assert (
        core._publish_prompt_tail_if_current(epoch, [1, 2], [object()], 128, 0)
        is False
    )
    scheduler._exact_resident_cache.put.assert_not_called()


def test_prompt_tail_process_global_gate_refuses_another_engine_prefill():
    owner = core_with_scheduler(Scheduler())
    other = core_with_scheduler(Scheduler(busy=True))
    owner._keepwarm.configure(True)
    owner.config.keepwarm_config = owner._keepwarm.config
    engine_core_module._register_prompt_tail_engine(owner)
    engine_core_module._register_prompt_tail_engine(other)
    try:
        assert owner.schedule_prompt_tail_prewarm("prompt") is True
        prompt, due, epoch = owner._prompt_tail_plan
        assert owner._take_prompt_tail_prewarm(due) == (prompt, epoch)

        result = owner._run_prompt_tail_prewarm(prompt, epoch)

        assert result == {
            "status": "skipped",
            "reason": "engine-busy-or-disabled",
        }
    finally:
        engine_core_module._unregister_prompt_tail_engine(owner)
        engine_core_module._unregister_prompt_tail_engine(other)


def test_prompt_tail_global_epoch_invalidates_on_any_engine_admission():
    owner = core_with_scheduler(Scheduler())
    engine_core_module._register_prompt_tail_engine(owner)
    try:
        epoch = engine_core_module._global_prompt_tail_epoch()
        assert engine_core_module._global_prompt_tail_invalid(owner, epoch) is False

        engine_core_module._invalidate_global_prompt_tail()

        assert engine_core_module._global_prompt_tail_invalid(owner, epoch) is True
    finally:
        engine_core_module._unregister_prompt_tail_engine(owner)


def test_resident_cache_tokens_include_exact_resident_l0():
    core = core_with_scheduler(
        Scheduler(cache_tokens=4096, exact_resident_tokens=220_000)
    )

    assert core._resident_cache_tokens() == 220_000


def test_concurrent_admission_skips_keepwarm_and_still_adds_request():
    scheduler = Scheduler()
    core = core_with_scheduler(scheduler)
    core._pending_admissions = 2
    core._run_keepwarm_action = lambda _action: (_ for _ in ()).throw(
        AssertionError("keepwarm must skip concurrent admission")
    )

    request = object()
    core._admit_request(request)
    assert scheduler.added == [request]
    snapshot = core._keepwarm.snapshot()
    assert snapshot["skips"] == 1
    assert snapshot["request_active"] is True


def test_failed_exclusive_admission_does_not_arm_or_leave_request_active():
    scheduler = Scheduler()
    core = core_with_scheduler(scheduler)
    core._pending_admissions = 1

    def reject(_request):
        raise ValueError("rejected")

    scheduler.add_request = reject
    try:
        core._admit_request(object())
    except ValueError:
        pass
    else:
        raise AssertionError("admission must propagate scheduler failure")

    snapshot = core._keepwarm.snapshot()
    assert snapshot["request_active"] is False
    assert snapshot["cache_armed"] is False


def test_all_rejected_concurrent_admissions_never_arm_keepwarm():
    scheduler = Scheduler()
    core = core_with_scheduler(scheduler)
    core._pending_admissions = 2

    def reject(_request):
        raise ValueError("rejected")

    scheduler.add_request = reject
    for _ in range(2):
        with suppress(ValueError):
            core._admit_request(object())

    snapshot = core._keepwarm.snapshot()
    assert snapshot["request_active"] is False
    assert snapshot["cache_armed"] is False


def test_second_admission_arriving_before_request_start_touch_wins():
    scheduler = Scheduler()
    core = core_with_scheduler(scheduler)
    core._keepwarm = KeepwarmController(
        config(request_start_idle_seconds=0.0),
    )
    core.config.keepwarm_config = core._keepwarm.config
    core._keepwarm.observe_request_state(True)
    core._keepwarm.observe_request_state(False, cache_tokens=4096)
    core._pending_admissions = 1
    original = core._keepwarm.request_start_action

    def race_second_admission():
        action = original()
        core._pending_admissions = 2
        return action

    core._keepwarm.request_start_action = race_second_admission
    core._compiled_metal_keepwarm = SimpleNamespace(
        touch=lambda _action: (_ for _ in ()).throw(
            AssertionError("touch must skip after a second admission arrives")
        )
    )

    request = object()
    core._admit_request(request)
    assert scheduler.added == [request]
    assert core._keepwarm.snapshot()["skips"] == 1


def test_idle_lane_rechecks_pending_admission_before_touching():
    scheduler = Scheduler(cache_tokens=16_384)
    core = core_with_scheduler(scheduler)
    core._keepwarm.observe_request_state(True)
    core._keepwarm.observe_request_state(False, cache_tokens=16_384)
    core._pending_admissions = 1
    core._run_keepwarm_action = lambda _action: (_ for _ in ()).throw(
        AssertionError("keepwarm must skip queued admission")
    )

    core._idle_keepwarm_if_due()
    assert core._keepwarm.snapshot()["request_active"] is True


def test_loaded_engine_live_reconfigure_updates_config_and_controller(monkeypatch):
    monkeypatch.delenv("OMLX_EXACT_RESIDENT_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("OMLX_EXACT_RESIDENT_CACHE_SLOTS", raising=False)
    scheduler = Scheduler()
    scheduler._exact_resident_cache = SimpleNamespace(
        max_entries=0,
        max_bytes=8 * 1024**3,
    )
    core = core_with_scheduler(scheduler)
    core.configure_keepwarm(False)
    assert core.config.keepwarm_config.enabled is False
    assert core._keepwarm.snapshot()["enabled"] is False
    core.configure_keepwarm(True)
    assert core.config.keepwarm_config.enabled is True
    assert core._keepwarm.snapshot()["enabled"] is True
    assert scheduler._exact_resident_cache.max_entries == 2
    core.configure_keepwarm(False)
    assert scheduler._exact_resident_cache.max_entries == 0


def test_explicit_zero_resident_slots_remains_a_hard_disable(monkeypatch):
    monkeypatch.setenv("OMLX_EXACT_RESIDENT_MAX_ENTRIES", "0")
    scheduler = Scheduler()
    scheduler._exact_resident_cache = SimpleNamespace(
        max_entries=0,
        max_bytes=8 * 1024**3,
    )
    core = core_with_scheduler(scheduler)

    core.configure_keepwarm(True)

    assert scheduler._exact_resident_cache.max_entries == 0



def test_action_shape_is_bounded_by_configuration_parser(monkeypatch):
    monkeypatch.setenv("OMLX_KEEPWARM", "1")
    monkeypatch.setenv("OMLX_KEEPWARM_MATRIX_SIZE", "99999")
    monkeypatch.setenv("OMLX_KEEPWARM_REPEATS", "999")
    parsed = KeepwarmConfig.from_env()
    assert parsed.enabled is True
    assert parsed.matrix_size == 1024
    assert parsed.repeats == 16


def test_action_is_an_immutable_transport_value():
    action = KeepwarmAction("idle", 1, 1, 2.0)
    assert action.kind == "idle"


class SyntheticArray:
    def __init__(self, value: float) -> None:
        self.value = value
        self.shape = (1,)
        self.dtype = "float32"
        self.nbytes = 4
        self.item_calls = 0

    def item(self):
        self.item_calls += 1
        return self.value


class SyntheticFast:
    def __init__(self, owner: SyntheticMX) -> None:
        self.owner = owner

    def metal_kernel(self, **kwargs):
        self.owner.execution_threads.append(threading.current_thread().name)
        self.owner.kernel_creations += 1
        self.owner.kernel_options.append(kwargs)
        if self.owner.creation_fails:
            raise RuntimeError("JIT creation failed")

        def kernel(**call_kwargs):
            self.owner.execution_threads.append(threading.current_thread().name)
            self.owner.kernel_runs += 1
            self.owner.kernel_calls.append(call_kwargs)
            if self.owner.invocation_fails:
                raise RuntimeError("JIT invocation failed")
            output = SyntheticArray(self.owner.output_value)
            output.shape = self.owner.output_shape
            output.dtype = self.owner.output_dtype
            return [output]

        return kernel


class SyntheticMX:
    float32 = "float32"

    def __init__(
        self,
        *,
        creation_fails: bool = False,
        invocation_fails: bool = False,
        async_fails: bool = False,
        synchronize_fails: bool = False,
        output_value: float = 1.0 + 1e-7,
        output_shape: tuple[int, ...] = (1,),
        output_dtype: str = "float32",
    ) -> None:
        self.creation_fails = creation_fails
        self.invocation_fails = invocation_fails
        self.async_fails = async_fails
        self.synchronize_fails = synchronize_fails
        self.output_value = output_value
        self.output_shape = output_shape
        self.output_dtype = output_dtype
        self.fast = SyntheticFast(self)
        self.array_allocations = 0
        self.kernel_creations = 0
        self.kernel_runs = 0
        self.kernel_options: list[dict] = []
        self.kernel_calls: list[dict] = []
        self.evaluated: list[SyntheticArray] = []
        self.async_eval_calls = 0
        self.latest_async_output: SyntheticArray | None = None
        self.synchronize_calls: list[object] = []
        self.stream_entries: list[object] = []
        self.stream_creations: list[tuple[object, str]] = []
        self.execution_threads: list[str] = []

    def array(self, values, *, dtype):
        assert values == [1.0]
        assert dtype == self.float32
        self.array_allocations += 1
        return SyntheticArray(values[0])

    def eval(self, value):
        self.execution_threads.append(threading.current_thread().name)
        self.evaluated.append(value)

    def async_eval(self, value):
        self.execution_threads.append(threading.current_thread().name)
        self.async_eval_calls += 1
        if self.async_fails:
            raise RuntimeError("async submission failed")
        self.latest_async_output = value

    def synchronize(self, stream):
        self.execution_threads.append(threading.current_thread().name)
        self.synchronize_calls.append(stream)
        if self.synchronize_fails:
            raise RuntimeError("stream synchronization failed")

    @staticmethod
    def default_device():
        return "gpu"

    def new_stream(self, device):
        stream = f"dedicated-stream-{len(self.stream_creations) + 1}"
        self.stream_creations.append((device, threading.current_thread().name))
        return stream

    @contextmanager
    def stream(self, stream):
        self.stream_entries.append(stream)
        yield


def test_metal_pulse_jits_once_during_idle_then_request_start_reuses_it():
    fake_mx = SyntheticMX()
    touch = CompiledMetalKeepwarmTouch(fake_mx, stream="engine-stream")

    prepared = touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
    request = touch.touch(KeepwarmAction("request_start", 128, 1, 2.0))
    post = touch.touch(KeepwarmAction("post_response", 128, 2, 2.0))

    assert prepared.execution_mode == "async_prepared"
    assert request.execution_mode == "async_submitted"
    assert post.execution_mode == "async_submitted"
    assert fake_mx.array_allocations == 1
    assert fake_mx.kernel_creations == 1
    assert fake_mx.kernel_runs == 4
    assert fake_mx.stream_entries == ["engine-stream"] * 3
    options = fake_mx.kernel_options[0]
    assert options["name"] == "omlx_keepwarm_pulse"
    assert options["compile_options"] == {"math_mode": "safe"}
    assert options["atomic_outputs"] is False
    assert options["ensure_row_contiguous"] is True
    for call in fake_mx.kernel_calls:
        assert call["grid"] == (1, 1, 1)
        assert call["threadgroup"] == (1, 1, 1)
    # No production pulse performs a synchronous eval, scalar read, or drain.
    assert fake_mx.evaluated == []
    assert fake_mx.async_eval_calls == 4
    assert fake_mx.latest_async_output.item_calls == 0
    assert fake_mx.synchronize_calls == []


def test_request_start_miss_skips_without_allocation_or_jit():
    fake_mx = SyntheticMX()
    touch = CompiledMetalKeepwarmTouch(fake_mx)

    result = touch.touch(KeepwarmAction("request_start", 128, 1, 2.0))

    assert result is None
    assert fake_mx.array_allocations == 0
    assert fake_mx.kernel_creations == 0
    assert fake_mx.kernel_runs == 0
    assert fake_mx.evaluated == []
    assert fake_mx.stream_creations == []


def test_metal_pulse_lazily_creates_and_reuses_one_dedicated_stream():
    fake_mx = SyntheticMX()
    touch = CompiledMetalKeepwarmTouch(fake_mx)

    assert touch._stream is None
    touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
    touch.touch(KeepwarmAction("request_start", 128, 1, 2.0))

    assert fake_mx.stream_creations == [("gpu", threading.current_thread().name)]
    assert fake_mx.stream_entries == ["dedicated-stream-1"] * 2
    assert touch._stream == "dedicated-stream-1"


def test_hot_pulses_keep_only_latest_four_byte_output_without_sync_or_list_growth():
    fake_mx = SyntheticMX()
    touch = CompiledMetalKeepwarmTouch(fake_mx)
    touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
    sync_eval_count = len(fake_mx.evaluated)

    for _ in range(100):
        result = touch.touch(KeepwarmAction("request_start", 128, 1, 2.0))
        assert result.execution_mode == "async_submitted"

    assert fake_mx.async_eval_calls == 101
    assert len(fake_mx.evaluated) == sync_eval_count
    assert fake_mx.synchronize_calls == []
    assert touch._latest_output is fake_mx.latest_async_output
    assert touch._latest_output.nbytes == 4
    assert not isinstance(touch._latest_output, list)


def test_metal_pulse_create_use_and_close_share_one_executor_thread():
    fake_mx = SyntheticMX()
    touch = CompiledMetalKeepwarmTouch(fake_mx)
    close_threads = []
    original_close = touch.close

    def observed_close():
        close_threads.append(threading.current_thread().name)
        original_close()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="keepwarm-owner",
    ) as executor:
        executor.submit(
            touch.touch,
            KeepwarmAction("idle", 1, 1, 2.0),
        ).result()
        executor.submit(
            touch.touch,
            KeepwarmAction("request_start", 128, 1, 2.0),
        ).result()
        executor.submit(observed_close).result()

    owner_thread = fake_mx.stream_creations[0][1]
    assert owner_thread.startswith("keepwarm-owner")
    assert set(fake_mx.execution_threads) == {owner_thread}
    assert close_threads == [owner_thread]
    assert touch._stream is None


def test_missing_metal_kernel_and_jit_failure_never_fall_back_to_matmul():
    unavailable_mx = SyntheticMX()
    unavailable_mx.fast = SimpleNamespace()
    unavailable = CompiledMetalKeepwarmTouch(unavailable_mx)
    failing_mx = SyntheticMX(invocation_fails=True)
    failing = CompiledMetalKeepwarmTouch(failing_mx)

    for touch in (unavailable, failing):
        try:
            touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Metal pulse failure must propagate to backoff")

    assert unavailable_mx.array_allocations == 0
    assert failing_mx.array_allocations == 1
    assert failing._input is None
    assert failing._kernel is None
    assert failing._prepared is False


def test_missing_or_failed_async_eval_is_nonfatal_and_drops_prepared_refs():
    unavailable_mx = SyntheticMX()
    unavailable = CompiledMetalKeepwarmTouch(unavailable_mx)
    unavailable.touch(KeepwarmAction("idle", 1, 1, 2.0))
    unavailable_mx.async_eval = None
    failing_mx = SyntheticMX()
    failing = CompiledMetalKeepwarmTouch(failing_mx)
    failing.touch(KeepwarmAction("idle", 1, 1, 2.0))
    failing_mx.async_fails = True

    for touch in (unavailable, failing):
        try:
            touch.touch(KeepwarmAction("request_start", 128, 1, 2.0))
        except RuntimeError:
            pass
        else:
            raise AssertionError("async pulse failure must reach outer backoff")
        assert touch._prepared is False
        assert touch._input is None
        assert touch._kernel is None
        assert touch._latest_output is None


def test_metal_pulse_rejects_wrong_output_metadata_without_a_host_read():
    fake_mx = SyntheticMX(output_shape=(2,))
    touch = CompiledMetalKeepwarmTouch(fake_mx)

    try:
        touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
    except RuntimeError as exc:
        assert "unexpected output shape" in str(exc)
    else:
        raise AssertionError("wrong Metal pulse metadata must fail closed")

    assert "out[elem] = inp[elem] + (T)1e-7;" in touch.KERNEL_SOURCE
    assert fake_mx.evaluated == []
    assert touch._input is None
    assert touch._kernel is None
    assert touch._prepared is False


def test_metal_pulse_enforces_existing_action_bounds_before_jit():
    fake_mx = SyntheticMX()
    touch = CompiledMetalKeepwarmTouch(fake_mx)

    for action in (
        KeepwarmAction("idle", 0, 1, 2.0),
        KeepwarmAction("idle", 1025, 1, 2.0),
        KeepwarmAction("idle", 1, 0, 2.0),
        KeepwarmAction("idle", 1, 17, 2.0),
        KeepwarmAction("unknown", 1, 1, 2.0),
    ):
        try:
            touch.touch(action)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-bounds Metal pulse must fail closed")

    assert fake_mx.array_allocations == 0
    assert fake_mx.kernel_creations == 0


def test_metal_pulse_retains_exactly_one_four_byte_input_and_close_drops_it():
    fake_mx = SyntheticMX()
    touch = CompiledMetalKeepwarmTouch(fake_mx, stream="engine-stream")
    touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
    touch.touch(KeepwarmAction("request_start", 128, 1, 2.0))

    assert touch._input is not None
    assert touch._input.nbytes == 4
    assert touch._kernel is not None
    assert touch._latest_output is not None
    assert touch._latest_output.nbytes == 4

    touch.close()

    assert fake_mx.synchronize_calls == ["engine-stream"]
    assert touch._input is None
    assert touch._kernel is None
    assert touch._latest_output is None
    assert touch._mx is None
    assert touch._stream is None
    try:
        touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed Metal pulse must not allocate again")


def test_close_clears_every_reference_even_when_exact_stream_drain_fails():
    fake_mx = SyntheticMX(synchronize_fails=True)
    touch = CompiledMetalKeepwarmTouch(fake_mx)
    touch.touch(KeepwarmAction("idle", 1, 1, 2.0))
    touch.touch(KeepwarmAction("request_start", 128, 1, 2.0))
    pulse_stream = touch._stream

    try:
        touch.close()
    except RuntimeError as exc:
        assert "synchronization failed" in str(exc)
    else:
        raise AssertionError("exact-stream drain failure must be observable")

    assert fake_mx.synchronize_calls == [pulse_stream]
    assert touch._stream is None
    assert touch._input is None
    assert touch._kernel is None
    assert touch._latest_output is None


def test_engine_compiled_touch_failure_is_nonfatal_to_inference():
    class FailingTouch:
        @staticmethod
        def touch(_action):
            raise RuntimeError("synthetic failure")

    core = core_with_scheduler(Scheduler())
    core._compiled_metal_keepwarm = FailingTouch()
    core._keepwarm.observe_request_state(True)
    core._keepwarm.observe_request_state(False, cache_tokens=4096)
    action = KeepwarmAction("idle", 1, 1, 2.0, cache_tokens=4096)

    assert core._run_keepwarm_action(action) is False
    assert core._keepwarm.snapshot()["failures"] == 1


def test_engine_request_start_pulse_miss_is_a_skip_not_a_failure():
    core = core_with_scheduler(Scheduler())
    core._compiled_metal_keepwarm = SimpleNamespace(touch=lambda _action: None)
    core._keepwarm.observe_request_state(True)
    action = KeepwarmAction("request_start", 128, 1, 2.0, cache_tokens=4096)

    assert core._run_keepwarm_action(action) is False
    snapshot = core._keepwarm.snapshot()
    assert snapshot["skips"] == 1
    assert snapshot["failures"] == 0


def test_engine_telemetry_marks_hot_pulse_as_async_submission():
    core = core_with_scheduler(Scheduler())
    core._compiled_metal_keepwarm = SimpleNamespace(
        touch=lambda _action: MetalKeepwarmTouchResult(
            elapsed_seconds=0.0002,
            execution_mode="async_submitted",
        )
    )
    core._keepwarm.observe_request_state(True)
    core._keepwarm.observe_request_state(False, cache_tokens=4096)
    action = KeepwarmAction("idle", 1, 1, 2.0, cache_tokens=4096)

    assert core._run_keepwarm_action(action) is True
    event = core._keepwarm.snapshot()["last_event"]
    assert event["execution_mode"] == "async_submitted"
    assert event["elapsed_ms"] == 0.2
