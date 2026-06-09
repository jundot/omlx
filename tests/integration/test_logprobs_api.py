# SPDX-License-Identifier: Apache-2.0
"""End-to-end API tests for OpenAI-style chat logprobs (#1549, Phase 4).

Exercises the real /v1/chat/completions handler (non-streaming + streaming) via
TestClient with a mock engine, asserting the OpenAI logprobs shape, streaming
content-token alignment, validation, and the server-side cap clamp. No model is
loaded.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from omlx.engine.base import BaseEngine, GenerationOutput
from omlx.request import TokenLogprob

from .test_e2e_streaming import MockBaseEngine, MockEnginePool, parse_sse_events


class LogprobEngine(MockBaseEngine):
    """Mock engine returning real GenerationOutput objects with logprobs."""

    def __init__(self):
        super().__init__()
        self.chat_output = GenerationOutput(
            text="Hi", prompt_tokens=5, completion_tokens=1, finish_reason="stop"
        )
        self.stream_output = []
        self.last_kwargs = None

    def count_chat_tokens(self, *args, **kwargs):
        return 5

    async def chat(self, messages, **kwargs):
        self.last_kwargs = kwargs
        return self.chat_output

    async def stream_chat(self, messages, **kwargs):
        self.last_kwargs = kwargs
        for out in self.stream_output:
            yield out

    async def preflight_chat(self, messages, **kwargs):
        # No-op preflight (mirrors BaseEngine.preflight_chat default); the chat
        # handler now calls this before the logprobs path under the prefill guard.
        return None


# get_engine() gates on isinstance(engine, BaseEngine); register the mock so the
# /v1/chat/completions LLM-type check accepts it without implementing the ABC.
BaseEngine.register(LogprobEngine)


class LogprobPool(MockEnginePool):
    """Adds get_entry() (used by the chat handler for preserve_thinking) and
    absorbs the leased-engine plumbing — upstream's LLM lease feature threads
    ``_lease``/``runtime_settings`` through ``get_engine()`` and releases via
    ``release_engine()`` once the response stream completes."""

    def get_entry(self, model_id):
        return None

    async def get_engine(self, model_id: str, **kwargs):
        return await super().get_engine(model_id)

    async def release_engine(self, model_id: str):
        return None


@pytest.fixture
def engine():
    return LogprobEngine()


@pytest.fixture
def client(engine):
    from omlx.server import _server_state, app

    orig_pool = _server_state.engine_pool
    orig_default = _server_state.default_model
    _server_state.engine_pool = LogprobPool(engine)
    _server_state.default_model = "test-model"
    yield TestClient(app, raise_server_exceptions=False)
    _server_state.engine_pool = orig_pool
    _server_state.default_model = orig_default


def _body(**extra):
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    }
    body.update(extra)
    return body


class TestNonStreamingLogprobs:
    def test_logprobs_present_and_shaped(self, client, engine):
        engine.chat_output = GenerationOutput(
            text="Hi",
            prompt_tokens=5,
            completion_tokens=1,
            finish_reason="stop",
            logprobs=[
                TokenLogprob(
                    token_id=1, logprob=-0.12, top_ids=[1, 4],
                    top_logprobs=[-0.12, -1.8], text="Hi",
                )
            ],
        )
        r = client.post("/v1/chat/completions", json=_body(logprobs=True, top_logprobs=2))
        assert r.status_code == 200
        lp = r.json()["choices"][0]["logprobs"]
        assert lp is not None
        assert lp["content"][0]["logprob"] == -0.12
        assert lp["content"][0]["bytes"] is not None
        assert len(lp["content"][0]["top_logprobs"]) == 2

    def test_logprobs_absent_when_not_requested(self, client, engine):
        engine.chat_output = GenerationOutput(
            text="Hi", completion_tokens=1, finish_reason="stop",
            logprobs=[TokenLogprob(token_id=1, logprob=-0.1)],
        )
        r = client.post("/v1/chat/completions", json=_body())
        assert r.status_code == 200
        assert r.json()["choices"][0].get("logprobs") is None

    def test_logprobs_omitted_when_reasoning_cleanup_changes_visible_content(
        self, client, engine
    ):
        raw = "<think>private</think>visible"
        engine.chat_output = GenerationOutput(
            text=raw,
            completion_tokens=3,
            finish_reason="stop",
            logprobs=[TokenLogprob(token_id=1, logprob=-0.1, text=raw)],
        )

        response = client.post("/v1/chat/completions", json=_body(logprobs=True))

        assert response.status_code == 200
        choice = response.json()["choices"][0]
        assert choice["message"]["content"] == "visible"
        assert "logprobs" not in choice

    def test_disabled_path_forwards_no_logprob_kwargs(self, client, engine):
        # Perf invariant: when logprobs aren't requested, nothing logprobs-related
        # is forwarded to the engine, so the generation path is unchanged.
        engine.chat_output = GenerationOutput(
            text="Hi", completion_tokens=1, finish_reason="stop"
        )
        r = client.post("/v1/chat/completions", json=_body())
        assert r.status_code == 200
        assert "logprobs" not in engine.last_kwargs
        assert "top_logprobs" not in engine.last_kwargs


class TestStreamingLogprobs:
    def test_streaming_logprobs_aligned_with_angle_bracket(self, client, engine):
        # "a", "<", "b" — the '<' triggers tag-lookahead buffering; all three
        # content tokens must still be represented exactly once.
        engine.stream_output = [
            GenerationOutput(
                text="a", new_text="a", completion_tokens=1,
                logprobs=[TokenLogprob(token_id=1, logprob=-0.1, text="a")],
            ),
            GenerationOutput(
                text="a<", new_text="<", completion_tokens=2,
                logprobs=[TokenLogprob(token_id=2, logprob=-0.2, text="<")],
            ),
            GenerationOutput(
                text="a<b", new_text="b", completion_tokens=3, finished=True,
                finish_reason="stop",
                logprobs=[TokenLogprob(token_id=3, logprob=-0.3, text="b")],
            ),
        ]
        r = client.post("/v1/chat/completions", json=_body(stream=True, logprobs=True))
        assert r.status_code == 200
        events = parse_sse_events(r.text)
        entries = []
        content = ""
        for ev in events:
            if not ev.get("choices"):
                continue
            ch = ev["choices"][0]
            content += (ch.get("delta") or {}).get("content") or ""
            lp = ch.get("logprobs")
            if lp and lp.get("content"):
                entries.extend(lp["content"])
        assert content == "a<b"
        assert len(entries) == 3  # one entry per content token — aligned

    def test_streaming_no_logprobs_when_not_requested(self, client, engine):
        engine.stream_output = [
            GenerationOutput(
                text="hi", new_text="hi", completion_tokens=1, finished=True,
                finish_reason="stop",
                logprobs=[TokenLogprob(token_id=1, logprob=-0.1, text="hi")],
            ),
        ]
        r = client.post("/v1/chat/completions", json=_body(stream=True))
        assert r.status_code == 200
        for ev in parse_sse_events(r.text):
            if ev.get("choices"):
                assert ev["choices"][0].get("logprobs") is None

    def test_streaming_mixed_aggregated_logprobs_never_drop_text(self, client, engine):
        # A collector can merge a supported token with an unsupported one. The
        # partial logprob list must be omitted, while the original text remains.
        engine.stream_output = [
            GenerationOutput(
                text="ab",
                new_text="ab",
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                logprobs=[TokenLogprob(token_id=1, logprob=-0.1, text="a")],
            )
        ]

        response = client.post(
            "/v1/chat/completions", json=_body(stream=True, logprobs=True)
        )

        assert response.status_code == 200
        entries = [
            event["choices"][0]
            for event in parse_sse_events(response.text)
            if event.get("choices")
        ]
        assert "".join(
            (entry.get("delta") or {}).get("content") or "" for entry in entries
        ) == "ab"
        assert all(entry.get("logprobs") is None for entry in entries)

    def test_streaming_lookahead_spanning_token_omits_misaligned_logprobs(
        self, client, engine
    ):
        # The first token contributes both emitted content and a buffered '<'.
        # Re-emitting that token's logprob after the lookahead resolves would
        # duplicate it, so both affected deltas omit the field.
        engine.stream_output = [
            GenerationOutput(
                text="a<",
                new_text="a<",
                completion_tokens=1,
                logprobs=[TokenLogprob(token_id=1, logprob=-0.1, text="a<")],
            ),
            GenerationOutput(
                text="a<b",
                new_text="b",
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                logprobs=[TokenLogprob(token_id=2, logprob=-0.2, text="b")],
            ),
        ]

        response = client.post(
            "/v1/chat/completions", json=_body(stream=True, logprobs=True)
        )

        assert response.status_code == 200
        entries = [
            event["choices"][0]
            for event in parse_sse_events(response.text)
            if event.get("choices")
        ]
        assert "".join(
            (entry.get("delta") or {}).get("content") or "" for entry in entries
        ) == "a<b"
        assert all(entry.get("logprobs") is None for entry in entries)

    def test_streaming_untracked_lookahead_then_logprobs_keeps_text(self, client, engine):
        # The '<' arrives without a logprob, then the following supported
        # token resolves the lookahead. This must not index mismatched parser
        # bookkeeping or fabricate an aligned result.
        engine.stream_output = [
            GenerationOutput(text="<", new_text="<", completion_tokens=1),
            GenerationOutput(
                text="<b",
                new_text="b",
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                logprobs=[TokenLogprob(token_id=2, logprob=-0.2, text="b")],
            ),
        ]

        response = client.post(
            "/v1/chat/completions", json=_body(stream=True, logprobs=True)
        )

        assert response.status_code == 200
        entries = [
            event["choices"][0]
            for event in parse_sse_events(response.text)
            if event.get("choices")
        ]
        assert "".join(
            (entry.get("delta") or {}).get("content") or "" for entry in entries
        ) == "<b"
        assert all(entry.get("logprobs") is None for entry in entries)


class TestValidation:
    def test_top_logprobs_above_20_rejected(self, client):
        r = client.post("/v1/chat/completions", json=_body(logprobs=True, top_logprobs=21))
        assert r.status_code == 422

    def test_top_logprobs_without_logprobs_rejected(self, client):
        r = client.post("/v1/chat/completions", json=_body(top_logprobs=3))
        assert r.status_code == 422


class TestServerCapClamp:
    def test_requested_top_logprobs_clamped_to_server_cap(self, client, engine):
        from omlx.server import _server_state

        engine.chat_output = GenerationOutput(
            text="Hi", completion_tokens=1, finish_reason="stop",
            logprobs=[TokenLogprob(token_id=1, logprob=-0.1)],
        )
        orig = _server_state.global_settings
        _server_state.global_settings = SimpleNamespace(
            sampling=SimpleNamespace(top_logprobs_k=1),
            server=SimpleNamespace(preserve_mid_system_cache=True),
        )
        try:
            r = client.post(
                "/v1/chat/completions", json=_body(logprobs=True, top_logprobs=5)
            )
            assert r.status_code == 200
            # Handler clamped 5 -> server cap 1 before forwarding to the engine.
            assert engine.last_kwargs["top_logprobs"] == 1
            assert engine.last_kwargs["logprobs"] is True
        finally:
            _server_state.global_settings = orig
