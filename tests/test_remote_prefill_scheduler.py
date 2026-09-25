# SPDX-License-Identifier: Apache-2.0
"""The scheduler consults remote prefill before admission, hands it each request, and forgets aborted ones."""

from __future__ import annotations

from unittest.mock import MagicMock

from omlx.remote_prefill.service import RemotePrefill
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig


class _Remote:
    def __init__(self, hold):
        self.hold = hold
        self.deferred, self.injected, self.forgotten = [], [], []

    def defer(self, request):
        self.deferred.append(request.request_id)
        return self.hold

    def inject(self, request):
        self.injected.append((request.request_id, list(request.remaining_tokens)))

    def forget(self, request_id):
        self.forgotten.append(request_id)

    def clear(self):
        pass


def _request(request_id="req-long"):
    return Request(
        request_id=request_id,
        prompt=list(range(9000)),
        sampling_params=SamplingParams(max_tokens=8),
    )


def test_a_request_prefilling_remotely_holds_admission(mock_model, mock_tokenizer):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    scheduler._remote_prefill = _Remote(hold=True)
    scheduler._ensure_batch_generator = MagicMock()
    request = _request()
    scheduler.add_request(request)

    scheduled, rejected = scheduler._schedule_waiting()

    assert (scheduled, rejected) == ([], [])
    assert list(scheduler.waiting) == [request]
    assert scheduler._remote_prefill.deferred == ["req-long"]
    scheduler._ensure_batch_generator.assert_not_called()


def test_the_handoff_follows_the_local_prefix_lookup(mock_model, mock_tokenizer):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    scheduler._remote_prefill = _Remote(hold=False)
    request = _request()
    scheduler.add_request(request)

    scheduler._prepare_prefix_cache_for_request(request)

    # Without a local cache every token is still to process when the handoff is offered.
    assert scheduler._remote_prefill.injected == [("req-long", list(range(9000)))]


def test_a_request_that_leaves_the_queue_is_forgotten(mock_model, mock_tokenizer):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    scheduler._remote_prefill = _Remote(hold=True)
    scheduler._clear_request_admission_bookkeeping("req-gone")
    assert scheduler._remote_prefill.forgotten == ["req-gone"]


def test_remote_prefill_serves_only_the_model_it_names(
    mock_model, mock_tokenizer, monkeypatch
):
    monkeypatch.setenv("OMLX_REMOTE_PREFILL_URL", "http://prefill.invalid:8000")
    monkeypatch.setenv("OMLX_REMOTE_PREFILL_MODEL", "org/model")
    monkeypatch.setenv("OMLX_REMOTE_PREFILL_LINKS", "sparka,sparkb")
    monkeypatch.setenv("OMLX_REMOTE_PREFILL_FOR", "local-model")
    other = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(model_name="another-model"),
    )
    named = Scheduler(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(model_name="local-model"),
    )
    assert other._remote_prefill is None
    assert isinstance(named._remote_prefill, RemotePrefill)
    assert named._remote_prefill.settings.links == ("sparka", "sparkb")


def test_unconfigured_servers_have_no_remote_prefill(mock_model, mock_tokenizer):
    assert Scheduler(model=mock_model, tokenizer=mock_tokenizer)._remote_prefill is None
