# SPDX-License-Identifier: Apache-2.0
"""Live-apply of max_concurrent_requests without unloading models.

The single settings value feeds two engine limits (GlobalSettings.
to_scheduler_config): max_num_seqs bounds admission, completion_batch_size
bounds the decode batch. These tests pin the reproduction the maintainer
reported: a 1-to-4 update reported success but left completion_batch_size
at 1 in the pool template, the scheduler config and the live BatchGenerator,
so decoding stayed serial; and the DFlash fallback scheduler plus its copied
config kept the old limit because the handler only followed
``_engine.engine.scheduler``.

They also pin safety properties of the live-apply itself:

- the live BatchGenerator cap is written ON the engine's MLX executor, not
  from the admin thread (the MTP wrapper zeroes and restores that attribute
  around generation steps; an admin-thread write can be lost to the
  restore); real ThreadPoolExecutors are used so Future semantics are real,
- nothing is applied to the pool until validation and save succeed, so a
  rejected save cannot leave runtime limits diverging from the file, and
- distributed (cluster) engines are skipped: their rank processes own the
  schedulers and pick up the new limit on reload only.

Coverage is assignment-level: none of these tests decode real rows.
"""

import asyncio
import concurrent.futures
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.admin.routes as admin_routes
import omlx.server  # noqa: F401 — ensure server module is imported first (triggers set_admin_getters)
from omlx.admin.routes import GlobalSettingsRequest


@contextmanager
def _patched_global_settings(gs):
    original = admin_routes._get_global_settings
    admin_routes._get_global_settings = lambda: gs
    try:
        yield
    finally:
        admin_routes._get_global_settings = original


def _make_global_settings(max_concurrent_requests=1):
    gs = MagicMock()
    gs.server.host = "127.0.0.1"
    gs.auth.api_key = None
    gs.auth.skip_api_key_verification = False
    gs.scheduler.max_concurrent_requests = max_concurrent_requests
    gs.scheduler.embedding_batch_size = 32
    gs.scheduler.chunked_prefill = False
    gs.validate.return_value = []
    gs.save.return_value = None
    return gs


# Real single-worker executors, like EngineCore._mlx_executor: posted work
# runs on the worker thread and real Future semantics (add_done_callback
# signature, exceptions) are exercised instead of a hand-written stand-in.
_EXECUTORS: list[concurrent.futures.ThreadPoolExecutor] = []


def _mlx_executor():
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    _EXECUTORS.append(executor)
    return executor


def _drain_executors():
    """Wait until every posted decode-cap write has run."""
    while _EXECUTORS:
        _EXECUTORS.pop().shutdown(wait=True)


def _fake_scheduler(
    max_num_seqs=1, completion_batch_size=1, generator=True, executor=None
):
    """Stand-in for Scheduler: its own config copy + optional live generator."""
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs, completion_batch_size=completion_batch_size
    )
    batch_generator = (
        SimpleNamespace(
            completion_batch_size=completion_batch_size, prefill_batch_size=1
        )
        if generator
        else None
    )
    scheduler = SimpleNamespace(config=config, batch_generator=batch_generator)
    core = SimpleNamespace(
        scheduler=scheduler,
        _mlx_executor=executor if executor is not None else _mlx_executor(),
    )
    # entry.engine._engine (AsyncEngineCore) -> .engine (EngineCore) -> .scheduler
    engine = SimpleNamespace(_engine=SimpleNamespace(engine=core))
    return engine


def _save(request_value, pool, current=1, gs=None, drain=True):
    gs = gs if gs is not None else _make_global_settings(
        max_concurrent_requests=current
    )
    server_state = SimpleNamespace(engine_pool=pool)
    request = GlobalSettingsRequest(max_concurrent_requests=request_value)
    with _patched_global_settings(gs), patch.object(
        omlx.server, "_server_state", server_state
    ):
        result = asyncio.run(
            admin_routes.update_global_settings(request=request, is_admin=True)
        )
    if drain:
        _drain_executors()
    return result, gs


class TestPoolTemplateAndLoadedSchedulers:
    def test_raise_updates_admission_and_decode_limits_everywhere(self):
        batched = _fake_scheduler()
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={"m": SimpleNamespace(engine=batched)},
        )

        result, gs = _save(4, pool)

        assert "max_concurrent_requests" in result["runtime_applied"]
        assert gs.scheduler.max_concurrent_requests == 4
        # Pool template: both limits (engines loaded later must not decode serially)
        assert pool._scheduler_config.max_num_seqs == 4
        assert pool._scheduler_config.completion_batch_size == 4
        # Loaded scheduler config: both limits
        sched = batched._engine.engine.scheduler
        assert sched.config.max_num_seqs == 4
        assert sched.config.completion_batch_size == 4
        # Live BatchGenerator: the cap it re-reads on every decode step
        assert sched.batch_generator.completion_batch_size == 4

    def test_lowering_updates_all_limits(self):
        # Assignment-level check only; that in-flight rows keep decoding
        # relies on BatchGenerator._next() never evicting batched rows when
        # the cap drops (it only stops admitting). Not exercised here.
        batched = _fake_scheduler(max_num_seqs=8, completion_batch_size=8)
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=8, completion_batch_size=8),
            _entries={"m": SimpleNamespace(engine=batched)},
        )

        result, _ = _save(2, pool, current=8)

        assert "max_concurrent_requests" in result["runtime_applied"]
        sched = batched._engine.engine.scheduler
        assert sched.config.max_num_seqs == 2
        assert sched.config.completion_batch_size == 2
        # The generator cap drops.
        assert sched.batch_generator.completion_batch_size == 2

    def test_generator_never_dropped_below_prefill_rows(self):
        batched = _fake_scheduler(max_num_seqs=8, completion_batch_size=8)
        sched = batched._engine.engine.scheduler
        sched.batch_generator.prefill_batch_size = 4
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=8, completion_batch_size=8),
            _entries={"m": SimpleNamespace(engine=batched)},
        )

        _save(1, pool, current=8)

        # mirrors BatchGenerator.__init__: max(completion, prefill)
        assert sched.batch_generator.completion_batch_size == 4

    def test_missing_generator_is_not_created(self):
        batched = _fake_scheduler(generator=False)
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={"m": SimpleNamespace(engine=batched)},
        )

        result, _ = _save(4, pool)

        assert "max_concurrent_requests" in result["runtime_applied"]
        assert batched._engine.engine.scheduler.batch_generator is None

    def test_unchanged_value_is_a_no_op(self):
        batched = _fake_scheduler()
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={"m": SimpleNamespace(engine=batched)},
        )

        result, _ = _save(1, pool)

        assert "max_concurrent_requests" not in result["runtime_applied"]
        assert pool._scheduler_config.max_num_seqs == 1
        assert pool._scheduler_config.completion_batch_size == 1

    def test_non_positive_rejected_with_400(self):
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={},
        )
        with pytest.raises(HTTPException) as exc:
            _save(0, pool)
        assert exc.value.status_code == 400


class TestExecutorSerializedWrite:
    """The live generator cap must be written on the engine's MLX executor.

    Writing it from the admin thread races the MTP wrapper, which zeroes
    completion_batch_size before a generation step and restores its saved
    value afterwards — the update could be silently lost.
    """

    def test_generator_write_is_posted_to_engine_executor_not_admin_thread(self):
        executor = _mlx_executor()
        step_running = threading.Event()
        finish_step = threading.Event()

        def busy_step():  # stands in for a generation step in progress
            step_running.set()
            finish_step.wait(timeout=5)

        executor.submit(busy_step)
        assert step_running.wait(timeout=5)
        batched = _fake_scheduler(executor=executor)
        sched = batched._engine.engine.scheduler
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={"m": SimpleNamespace(engine=batched)},
        )

        _save(4, pool, drain=False)

        # While the step thread is busy, the cap is unchanged: the admin
        # thread did not write the generator directly.
        assert sched.batch_generator.completion_batch_size == 1
        # Once the step finishes, the posted write runs between steps.
        finish_step.set()
        _drain_executors()
        assert sched.batch_generator.completion_batch_size == 4

    def test_generator_write_failure_is_logged_not_raised(self):
        batched = _fake_scheduler()
        sched = batched._engine.engine.scheduler
        # prefill_batch_size missing -> the posted write raises on the worker.
        sched.batch_generator = SimpleNamespace(completion_batch_size=1)
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={"m": SimpleNamespace(engine=batched)},
        )

        with patch.object(admin_routes.logger, "warning") as warning:
            result, _ = _save(4, pool)

        assert "max_concurrent_requests" in result["runtime_applied"]
        assert warning.call_count == 1
        assert "Live decode-cap update failed" in warning.call_args[0][0]

    def test_shut_down_executor_does_not_fail_the_save(self):
        # Engine stopping: submit() raises RuntimeError after the save; the
        # request must still succeed.
        executor = _mlx_executor()
        executor.shutdown(wait=True)
        batched = _fake_scheduler(executor=executor)
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={"m": SimpleNamespace(engine=batched)},
        )

        result, _ = _save(4, pool)

        assert "max_concurrent_requests" in result["runtime_applied"]
        assert batched._engine.engine.scheduler.config.completion_batch_size == 4

    def test_executor_post_survives_missing_executor(self):
        # An engine without an executor (mock/stub engines in tests and
        # partial startup) must not crash; config limits still apply.
        batched = _fake_scheduler()
        batched._engine.engine._mlx_executor = None
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={"m": SimpleNamespace(engine=batched)},
        )

        result, _ = _save(4, pool)

        assert "max_concurrent_requests" in result["runtime_applied"]
        assert batched._engine.engine.scheduler.config.completion_batch_size == 4


class TestRollbackOnFailedSave:
    def _pool(self):
        batched = _fake_scheduler()
        return SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={"m": SimpleNamespace(engine=batched)},
        )

    def test_validation_error_restores_settings_and_skips_live_apply(self):
        pool = self._pool()
        gs = _make_global_settings()
        gs.validate.return_value = ["some error"]

        with pytest.raises(HTTPException) as exc:
            _save(4, pool, gs=gs)
        assert exc.value.status_code == 400
        assert gs.scheduler.max_concurrent_requests == 1
        assert pool._scheduler_config.max_num_seqs == 1
        assert pool._scheduler_config.completion_batch_size == 1

    def test_save_failure_restores_settings_and_skips_live_apply(self):
        pool = self._pool()
        gs = _make_global_settings()
        gs.save.side_effect = OSError("disk full")

        with pytest.raises(HTTPException) as exc:
            _save(4, pool, gs=gs)
        assert exc.value.status_code == 500
        assert gs.scheduler.max_concurrent_requests == 1
        assert pool._scheduler_config.max_num_seqs == 1


class TestDFlashFallbackCoverage:
    """The handler must reach the DFlash copied config and fallback scheduler."""

    def _dflash_engine(self, fallback_started=True):
        # DFlash snapshots the pool template (copy.copy) at build time.
        dflash_copy = SimpleNamespace(max_num_seqs=1, completion_batch_size=1)
        dflash = SimpleNamespace(
            _scheduler_config=dflash_copy,
            _fallback_engine=(_fake_scheduler() if fallback_started else None),
        )
        return dflash, dflash_copy

    def test_dflash_copied_config_updated_before_fallback_starts(self):
        dflash, dflash_copy = self._dflash_engine(fallback_started=False)
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={"d": SimpleNamespace(engine=dflash)},
        )

        _save(4, pool)

        # The lazily-started fallback engine is built from this copy later.
        assert dflash_copy.max_num_seqs == 4
        assert dflash_copy.completion_batch_size == 4

    def test_started_fallback_scheduler_and_generator_updated(self):
        dflash, dflash_copy = self._dflash_engine(fallback_started=True)
        fallback_sched = dflash._fallback_engine._engine.engine.scheduler
        # The fallback holds its own config copy — the old handler missed it.
        assert fallback_sched.config is not dflash_copy
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={"d": SimpleNamespace(engine=dflash)},
        )

        _save(4, pool)

        assert fallback_sched.config.max_num_seqs == 4
        assert fallback_sched.config.completion_batch_size == 4
        assert fallback_sched.batch_generator.completion_batch_size == 4


class TestSharedTemplateIdentity:
    def test_pool_template_not_double_written_via_engine_reference(self):
        # BatchedEngine stores the pool template BY REFERENCE; the identity
        # check must skip it (writes stay idempotent, no stale copy shadowing).
        template = SimpleNamespace(max_num_seqs=1, completion_batch_size=1)
        batched = _fake_scheduler()
        batched._scheduler_config = template
        pool = SimpleNamespace(_scheduler_config=template, _entries={"m": SimpleNamespace(engine=batched)})

        _save(4, pool)

        assert template.max_num_seqs == 4
        assert template.completion_batch_size == 4
        assert batched._scheduler_config is template


class TestDistributedEnginesNotLiveApplied:
    def test_cluster_engine_is_skipped(self):
        # DistributedBatchedEngine proxies to rank processes that own their
        # own schedulers; the handler cannot reach them and must not touch
        # the coordinator (same marker ProcessMemoryEnforcer skips on).
        cluster_copy = SimpleNamespace(max_num_seqs=1, completion_batch_size=1)
        cluster = SimpleNamespace(
            _prefill_memory_guard_managed_externally=True,
            _scheduler_config=cluster_copy,
        )
        batched = _fake_scheduler()
        pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(max_num_seqs=1, completion_batch_size=1),
            _entries={
                "cluster": SimpleNamespace(engine=cluster),
                "local": SimpleNamespace(engine=batched),
            },
        )

        _save(4, pool)

        assert cluster_copy.max_num_seqs == 1
        assert cluster_copy.completion_batch_size == 1
        # Local engines in the same pool still get the live update.
        assert batched._engine.engine.scheduler.batch_generator.completion_batch_size == 4
