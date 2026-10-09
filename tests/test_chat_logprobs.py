# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAI-compatible logprobs on non-streaming chat completions.

The scheduler keeps one record per output token, read from the row the
sampler drew it from, and never mixes rows between batched requests. The
collector keeps the records through merges, and /v1/chat/completions formats
them in the OpenAI shape. Requests without logprobs keep their output.
"""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

import omlx.server as srv
from omlx.engine.base import GenerationOutput
from omlx.output_collector import RequestOutputCollector
from omlx.request import (
    Request,
    RequestOutput,
    RequestStatus,
    SamplingParams,
    TokenLogprob,
)
from omlx.scheduler import Scheduler


def _row(*logits):
    """A logprob row the way mlx-lm's GenerationBatch emits it."""
    values = mx.array(logits, dtype=mx.float32)
    return values - mx.logsumexp(values)


def _log_softmax(logits):
    norm = math.log(sum(math.exp(v) for v in logits))
    return [v - norm for v in logits]


def _response(uid, token, row, finish_reason=None):
    return SimpleNamespace(
        uid=uid,
        token=token,
        finish_reason=finish_reason,
        match_sequence=None,
        logprobs=row,
        prompt_cache=None,
    )


class TestSchedulerLogprobRecords:
    @pytest.fixture
    def scheduler(self, mock_model, mock_tokenizer):
        return Scheduler(model=mock_model, tokenizer=mock_tokenizer)

    @staticmethod
    def _running(scheduler, request_id, uid, **params):
        request = Request(
            request_id=request_id,
            prompt="prompt",
            sampling_params=SamplingParams(max_tokens=8, temperature=0.0, **params),
            prompt_token_ids=[1],
            num_prompt_tokens=1,
            status=RequestStatus.RUNNING,
            batch_uid=uid,
        )
        scheduler.running[request_id] = request
        scheduler.requests[request_id] = request
        scheduler.uid_to_request_id[uid] = request_id
        scheduler.request_id_to_uid[request_id] = uid
        return request

    @staticmethod
    def _finished(outputs, request_id):
        return next(o for o in outputs if o.request_id == request_id and o.finished)

    def test_records_follow_output_tokens_of_each_batched_row(self, scheduler):
        self._running(scheduler, "asks", 1, logprobs=True, top_logprobs=2)
        self._running(scheduler, "plain", 2)
        eos = scheduler.tokenizer.eos_token_id
        a1, a2 = (0.0, 3.0, 1.0, 2.0), (4.0, 0.5, 0.0, 1.0)
        plain_rows = [_row(9.0, 0.0, 0.0, 0.0), _row(0.0, 0.0, 0.0, 9.0)]

        outputs, _ = scheduler._process_batch_responses(
            [_response(1, 1, _row(*a1)), _response(2, 0, plain_rows[0])]
        )
        more, _ = scheduler._process_batch_responses(
            [_response(1, 0, _row(*a2)), _response(2, 3, plain_rows[1], "length")]
        )
        last, finished = scheduler._process_batch_responses(
            [_response(1, eos, _row(0.0, 0.0, 9.0, 0.0), "stop")]
        )
        outputs += more + last

        assert finished == {"asks"}
        records = self._finished(outputs, "asks").logprobs
        # The EOS step adds no output token, so it adds no record.
        assert [r.token for r in records] == [1, 0]
        expected = [_log_softmax(a1), _log_softmax(a2)]
        for record, logprobs in zip(records, expected):
            assert record.logprob == pytest.approx(logprobs[record.token], abs=1e-5)
            assert record.top_logprobs[0][0] == record.token
            ids = sorted(range(4), key=lambda i: -logprobs[i])[:2]
            assert [t for t, _ in record.top_logprobs] == ids
            assert [lp for _, lp in record.top_logprobs] == pytest.approx(
                [logprobs[i] for i in ids], abs=1e-5
            )
        assert self._finished(outputs, "plain").logprobs is None

    def test_rows_are_released_after_reading(self, scheduler):
        self._running(scheduler, "asks", 1, logprobs=True)
        self._running(scheduler, "plain", 2)
        responses = [
            _response(1, 1, _row(0.0, 1.0)),
            _response(2, 1, _row(0.0, 1.0)),
        ]

        scheduler._process_batch_responses(responses)

        assert [r.logprobs for r in responses] == [None, None]
        record = scheduler.running["asks"].output_logprobs[0]
        assert record.top_logprobs == []
        assert scheduler.running["plain"].output_logprobs is None

    def test_missing_row_reports_no_logprobs(self, scheduler):
        self._running(scheduler, "asks", 1, logprobs=True)

        scheduler._process_batch_responses([_response(1, 1, _row(0.0, 1.0))])
        outputs, _ = scheduler._process_batch_responses(
            [_response(1, 0, None, "length")]
        )

        assert outputs[-1].finished
        assert outputs[-1].logprobs is None

    def test_mtp_decode_reports_no_logprobs(self, scheduler):
        # Accepted MTP drafts carry the draft head's distribution.
        scheduler.model._omlx_mtp_decode_enabled = True
        self._running(scheduler, "asks", 1, logprobs=True)

        outputs, _ = scheduler._process_batch_responses(
            [_response(1, 1, _row(0.0, 1.0), "length")]
        )

        assert outputs[-1].finished
        assert outputs[-1].logprobs is None

    def test_reprefill_drops_records_of_discarded_tokens(self, scheduler):
        request = self._running(scheduler, "asks", 1, logprobs=True)
        scheduler._process_batch_responses([_response(1, 1, _row(0.0, 1.0))])
        scheduler._reset_request_for_reprefill(request)
        request.status = RequestStatus.RUNNING

        outputs, _ = scheduler._process_batch_responses(
            [_response(1, 0, _row(2.0, 0.0), "length")]
        )

        assert [r.token for r in outputs[-1].logprobs] == [0]


def test_collector_merge_keeps_logprobs():
    records = [TokenLogprob(7, -0.25, [(7, -0.25)])]
    collector = RequestOutputCollector(aggregate=True)
    collector.put(RequestOutput(request_id="r", new_token_ids=[7]))
    collector.put(
        RequestOutput(request_id="r", finished=True, logprobs=records),
    )
    assert collector.get_nowait().logprobs == records


class _Tokenizer:
    pieces = {5: "hi", 6: "�", 7: "!"}

    def decode(self, token_ids):
        return "".join(self.pieces[t] for t in token_ids)


@pytest.fixture
def chat_route(monkeypatch):
    output = GenerationOutput(
        text="hi!",
        prompt_tokens=4,
        completion_tokens=2,
        finish_reason="stop",
        logprobs=[
            TokenLogprob(5, -0.125, [(5, -0.125), (6, -2.5)]),
            TokenLogprob(7, -0.5, [(7, -0.5), (6, -math.inf)]),
        ],
    )
    stream_kwargs = {}

    async def stream_chat(**kwargs):
        stream_kwargs.update(kwargs)
        yield GenerationOutput(text="hi!", new_text="hi!", completion_tokens=2)

    engine = SimpleNamespace(
        model_type="llama",
        is_diffusion_model=False,
        tokenizer=_Tokenizer(),
        start=AsyncMock(),
        count_chat_tokens=lambda *args, **kwargs: 4,
        preflight_chat=AsyncMock(),
        chat=AsyncMock(return_value=output),
        stream_chat=stream_chat,
        stream_kwargs=stream_kwargs,
    )
    pool = SimpleNamespace(
        get_entry=lambda _: None,
        preload_pinned_models=AsyncMock(),
        check_ttl_expirations=AsyncMock(),
        shutdown=AsyncMock(),
    )
    monkeypatch.setattr(srv._server_state, "engine_pool", pool)
    monkeypatch.setattr(srv, "get_engine_for_model", AsyncMock(return_value=engine))
    monkeypatch.setattr(srv, "get_server_metrics", Mock(return_value=Mock()))
    monkeypatch.setattr(srv, "resolve_model_id", lambda name: name)
    monkeypatch.setattr(srv, "validate_context_window", lambda *a, **k: None)
    monkeypatch.setattr(srv, "get_model_settings_for_request", lambda name: None)
    monkeypatch.setitem(
        srv.app.dependency_overrides, srv.verify_inference_api_key, lambda: True
    )
    return engine


def _post(body):
    with TestClient(srv.app, raise_server_exceptions=False) as client:
        return client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                **body,
            },
        )


def test_chat_completion_returns_openai_logprobs(chat_route):
    response = _post({"logprobs": True, "top_logprobs": 2})

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["logprobs"] == {
        "content": [
            {
                "token": "hi",
                "logprob": -0.125,
                "bytes": [104, 105],
                "top_logprobs": [
                    {"token": "hi", "logprob": -0.125, "bytes": [104, 105]},
                    {"token": "�", "logprob": -2.5, "bytes": None},
                ],
            },
            {
                "token": "!",
                "logprob": -0.5,
                "bytes": [33],
                "top_logprobs": [
                    {"token": "!", "logprob": -0.5, "bytes": [33]},
                    {"token": "�", "logprob": -9999.0, "bytes": None},
                ],
            },
        ]
    }
    kwargs = chat_route.chat.call_args.kwargs
    assert (kwargs["logprobs"], kwargs["top_logprobs"]) == (True, 2)


def test_chat_completion_without_logprobs_is_unchanged(chat_route):
    response = _post({})

    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert "logprobs" not in choice
    assert choice["message"]["content"] == "hi!"
    assert "logprobs" not in chat_route.chat.call_args.kwargs


def test_streaming_ignores_logprobs(chat_route):
    response = _post({"stream": True, "logprobs": True, "top_logprobs": 2})

    assert response.status_code == 200, response.text
    assert '"logprobs":{' not in response.text
    assert "logprobs" not in chat_route.stream_kwargs
    chat_route.chat.assert_not_called()


@pytest.mark.parametrize("top_logprobs", [-1, 21])
def test_top_logprobs_must_be_within_openai_range(chat_route, top_logprobs):
    response = _post({"logprobs": True, "top_logprobs": top_logprobs})

    assert response.status_code == 422
    chat_route.chat.assert_not_called()
