# SPDX-License-Identifier: Apache-2.0
"""Regression tests for idempotent uid-map cleanup in the scheduler.

``uid_to_request_id`` and ``request_id_to_uid`` are plain dicts shared between
the scheduler's inference thread and ``fail_all_requests()``, which the engine
loop dispatches onto the MLX executor thread (``run_in_executor(...,
self.scheduler.fail_all_requests)``). The cleanup sites used a bare ``del``,
so a pop landing between the guard and the delete raised ``KeyError`` and
killed the scheduler step — taking the server down with it (#1031).

These tests reproduce that interleaving deterministically by popping the maps
from inside a patched hook, which is where the other thread's write would
land.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


def _make_scheduler(model, tokenizer) -> Scheduler:
    return Scheduler(model=model, tokenizer=tokenizer)


def _queue(scheduler: Scheduler, request: Request) -> None:
    request.prompt_token_ids = [11, 12, 13, 14]
    request.num_prompt_tokens = 4
    request.remaining_tokens = [11, 12, 13, 14]
    request.cached_tokens = 0
    scheduler.waiting.append(request)
    scheduler.requests[request.request_id] = request


class TestTempUidCleanupRace:
    """The success-path temp UID cleanup must tolerate a concurrent pop."""

    def test_schedule_waiting_survives_concurrent_pop_during_prefill(
        self, mock_model, mock_tokenizer
    ):
        """A pop that lands while external prefill runs must not raise."""
        scheduler = _make_scheduler(mock_model, mock_tokenizer)
        request = Request(
            request_id="req-temp-uid-race",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=4),
        )
        _queue(scheduler, request)

        def prefill_then_lose_the_mapping(*args, **kwargs):
            # Stand in for fail_all_requests() on the MLX executor thread:
            # it pops the maps while this thread is inside external prefill,
            # so the success-path cleanup finds nothing to delete.
            scheduler.uid_to_request_id.clear()
            scheduler.request_id_to_uid.clear()
            return [MagicMock()], [14]

        scheduler._do_external_prefill = MagicMock(
            side_effect=prefill_then_lose_the_mapping
        )

        # Before the fix this raised KeyError inside _schedule_waiting.
        scheduled, rejected = scheduler._schedule_waiting()

        assert rejected == []
        scheduler._do_external_prefill.assert_called_once()

    def test_temp_uid_cleanup_leaves_re_registered_reverse_mapping(
        self, mock_model, mock_tokenizer
    ):
        """A request already holding a real BatchGenerator UID must survive.

        The cleanup only drops the reverse mapping when it still points at the
        temp UID this call registered. The assertion samples the maps inside
        ``insert()``, i.e. the window between the temp-UID cleanup and the real
        re-registration, which is where a clobber would strand a pending abort.
        """
        scheduler = _make_scheduler(mock_model, mock_tokenizer)
        request = Request(
            request_id="req-temp-uid-reregistered",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=4),
        )
        _queue(scheduler, request)

        real_batch_uid = 987654321

        def prefill_then_re_register(*args, **kwargs):
            # The request picked up a real BatchGenerator UID before the
            # success-path cleanup ran.
            temp_uid = scheduler.request_id_to_uid[request.request_id]
            scheduler.uid_to_request_id.pop(temp_uid, None)
            scheduler.request_id_to_uid[request.request_id] = real_batch_uid
            scheduler.uid_to_request_id[real_batch_uid] = request.request_id
            return [MagicMock()], [14]

        scheduler._do_external_prefill = MagicMock(side_effect=prefill_then_re_register)

        seen: dict = {}

        def capture_then_insert(*args, **kwargs):
            seen["reverse"] = scheduler.request_id_to_uid.get(request.request_id)
            seen["forward"] = scheduler.uid_to_request_id.get(real_batch_uid)
            return [real_batch_uid]

        scheduler._ensure_batch_generator(request.sampling_params)
        scheduler.batch_generator.insert = MagicMock(side_effect=capture_then_insert)

        scheduler._schedule_waiting()

        assert seen["reverse"] == real_batch_uid
        assert seen["forward"] == request.request_id


class TestAbortUidCleanupRace:
    """_do_abort_request must tolerate a concurrent pop of the same maps."""

    def test_do_abort_request_survives_concurrent_pop(
        self, mock_model, mock_tokenizer
    ):
        scheduler = _make_scheduler(mock_model, mock_tokenizer)
        request = Request(
            request_id="req-abort-race",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=4),
        )
        scheduler.requests[request.request_id] = request
        uid = 4242
        scheduler.request_id_to_uid[request.request_id] = uid
        scheduler.uid_to_request_id[uid] = request.request_id

        def lose_the_mapping(*args, **kwargs):
            scheduler.uid_to_request_id.pop(uid, None)
            scheduler.request_id_to_uid.pop(request.request_id, None)

        # The pop has to land after the membership guard but before the
        # delete, which is exactly where _remove_uid_from_active_batch runs.
        scheduler._remove_uid_from_active_batch = MagicMock(side_effect=lose_the_mapping)

        scheduler._do_abort_request(request.request_id)

        scheduler._remove_uid_from_active_batch.assert_called_once_with(uid)
        assert request.request_id not in scheduler.request_id_to_uid
        assert uid not in scheduler.uid_to_request_id

    def test_abort_cleanup_drops_stale_uid_mapping_already(
        self, mock_model, mock_tokenizer
    ):
        """An abort arriving after the mappings were already popped is a no-op."""
        scheduler = _make_scheduler(mock_model, mock_tokenizer)
        request = Request(
            request_id="req-abort-already-clean",
            prompt=[11, 12, 13, 14],
            sampling_params=SamplingParams(max_tokens=4),
        )
        scheduler.requests[request.request_id] = request

        assert scheduler._do_abort_request(request.request_id) is not False


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("_do_abort_request", {}),
    ],
)
def test_no_bare_del_remains_in_scheduler_source(method, kwargs):
    """Guard against a new unguarded delete reappearing on these maps."""
    import inspect

    from omlx import scheduler as scheduler_module

    source = inspect.getsource(scheduler_module)
    assert "del self.uid_to_request_id[" not in source
    assert "del self.request_id_to_uid[" not in source
