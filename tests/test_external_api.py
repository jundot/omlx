# SPDX-License-Identifier: Apache-2.0
"""Tests for the external OpenAI-compatible endpoint client."""

import json

import httpx
import pytest

from omlx.admin.external_api import (
    ExternalAPIClient,
    ExternalChatAdapter,
    ExternalEndpointConfig,
    ExternalEndpointError,
)

API_KEY = "sk-secret-test-key"


def _config(**overrides):
    params = {
        "base_url": "https://api.example.com/v1",
        "api_key": API_KEY,
        "model": "test-model",
    }
    params.update(overrides)
    return ExternalEndpointConfig(**params)


def _client(handler, **config_overrides):
    return ExternalAPIClient(
        _config(**config_overrides),
        transport=httpx.MockTransport(handler),
    )


def _sse_body(chunks, done=True):
    """Build an SSE response body from chunk dicts / raw lines."""
    lines = []
    for chunk in chunks:
        if isinstance(chunk, str):
            lines.append(chunk)
        else:
            lines.append("data: " + json.dumps(chunk))
    if done:
        lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def _content_chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


def _usage_chunk(prompt=10, completion=5, cached=None):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"choices": [], "usage": usage}


def _completion_response(
    text="A", prompt=10, completion=5, extra_message=None, finish_reason="stop"
):
    message = {"role": "assistant", "content": text}
    if extra_message:
        message.update(extra_message)
    return httpx.Response(
        200,
        json={
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        },
    )


# =============================================================================
# Config validation
# =============================================================================


class TestExternalEndpointConfig:
    def test_trailing_slash_stripped(self):
        cfg = _config(base_url="https://api.example.com/v1/")
        assert cfg.base_url == "https://api.example.com/v1"

    def test_whitespace_stripped(self):
        cfg = _config(base_url="  http://localhost:8001/v1  ", model=" m ")
        assert cfg.base_url == "http://localhost:8001/v1"
        assert cfg.model == "m"

    def test_invalid_scheme_rejected(self):
        with pytest.raises(ValueError, match="http:// or https://"):
            _config(base_url="ftp://api.example.com/v1")

    def test_empty_model_rejected(self):
        with pytest.raises(ValueError, match="model must not be empty"):
            _config(model="   ")

    def test_api_key_hidden_from_repr(self):
        cfg = _config()
        assert API_KEY not in repr(cfg)
        assert API_KEY not in str(cfg)
        assert cfg.api_key.get_secret_value() == API_KEY

    def test_empty_api_key_allowed(self):
        cfg = _config(api_key="")
        assert cfg.api_key.get_secret_value() == ""

    def test_extra_body_defaults_to_empty_object(self):
        cfg = _config()
        assert cfg.extra_body == {}

    def test_extra_body_accepts_provider_specific_fields(self):
        extra = {"thinking": {"type": "disabled"}, "seed": 42}
        cfg = _config(extra_body=extra)
        assert cfg.extra_body == extra

    @pytest.mark.parametrize(
        "field",
        [
            "model",
            "messages",
            "stream",
            "stream_options",
            "max_tokens",
            "temperature",
            "api_key",
            "authorization",
            "Authorization",
        ],
    )
    def test_extra_body_rejects_protected_fields(self, field):
        with pytest.raises(ValueError, match="protected field"):
            _config(extra_body={field: "override"})

    @pytest.mark.parametrize("value", [[], "value", 1, True])
    def test_extra_body_requires_object(self, value):
        with pytest.raises(ValueError, match="JSON object"):
            _config(extra_body=value)


# =============================================================================
# Streaming chat completion
# =============================================================================


class TestStreamChatCompletion:
    async def test_counts_come_from_usage_not_chunks(self):
        """Providers batch multiple tokens per chunk — only usage counts."""
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(
                200,
                content=_sse_body([
                    {"choices": [{"delta": {"role": "assistant"}}]},
                    _content_chunk("Hello "),
                    _content_chunk("world"),
                    _usage_chunk(prompt=1000, completion=128),
                ]),
            )

        client = _client(handler)
        try:
            stats = await client.stream_chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=128,
                temperature=0.0,
            )
        finally:
            await client.aclose()

        # 2 content chunks but usage says 128 completion tokens
        assert stats.completion_tokens == 128
        assert stats.prompt_tokens == 1000
        assert stats.cached_tokens == 0
        assert stats.text == "Hello world"
        assert stats.start_time <= stats.first_content_time
        assert stats.first_content_time <= stats.last_content_time <= stats.end_time
        assert captured["auth"] == f"Bearer {API_KEY}"
        assert captured["body"]["stream"] is True
        assert captured["body"]["stream_options"] == {"include_usage": True}

    async def test_missing_usage_is_hard_error(self):
        def handler(request):
            return httpx.Response(
                200,
                content=_sse_body([_content_chunk("Hello")]),
            )

        client = _client(handler)
        try:
            with pytest.raises(ExternalEndpointError, match="stream usage"):
                await client.stream_chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    temperature=0.0,
                )
        finally:
            await client.aclose()

    async def test_cached_tokens_parsed(self):
        def handler(request):
            return httpx.Response(
                200,
                content=_sse_body([
                    _content_chunk("x"),
                    _usage_chunk(prompt=100, completion=10, cached=64),
                ]),
            )

        client = _client(handler)
        try:
            stats = await client.stream_chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=0.0,
            )
        finally:
            await client.aclose()
        assert stats.cached_tokens == 64

    async def test_keepalive_and_malformed_lines_skipped(self):
        def handler(request):
            return httpx.Response(
                200,
                content=_sse_body([
                    ": keepalive",
                    "data: not-json",
                    _content_chunk("ok"),
                    _usage_chunk(),
                ]),
            )

        client = _client(handler)
        try:
            stats = await client.stream_chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=0.0,
            )
        finally:
            await client.aclose()
        assert stats.text == "ok"

    async def test_single_chunk_dump_has_zero_gen_duration(self):
        def handler(request):
            return httpx.Response(
                200,
                content=_sse_body([
                    _content_chunk("everything at once"),
                    _usage_chunk(prompt=10, completion=20),
                ]),
            )

        client = _client(handler)
        try:
            stats = await client.stream_chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=32,
                temperature=0.0,
            )
        finally:
            await client.aclose()
        assert stats.first_content_time == stats.last_content_time

    async def test_reasoning_content_sets_timing_but_not_text(self):
        def handler(request):
            return httpx.Response(
                200,
                content=_sse_body([
                    {"choices": [{"delta": {"reasoning_content": "thinking..."}}]},
                    _content_chunk("answer"),
                    _usage_chunk(),
                ]),
            )

        client = _client(handler)
        try:
            stats = await client.stream_chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=0.0,
            )
        finally:
            await client.aclose()
        assert stats.text == "answer"
        assert stats.first_content_time < stats.last_content_time

    async def test_auth_error_message_without_key(self):
        def handler(request):
            return httpx.Response(401, json={"error": {"message": "bad key"}})

        client = _client(handler)
        try:
            with pytest.raises(ExternalEndpointError) as exc_info:
                await client.stream_chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    temperature=0.0,
                )
        finally:
            await client.aclose()
        assert "rejected the API key" in str(exc_info.value)
        assert API_KEY not in str(exc_info.value)

    async def test_http_error_includes_json_detail(self):
        def handler(request):
            return httpx.Response(
                500, json={"error": {"message": "model exploded"}}
            )

        client = _client(handler)
        try:
            with pytest.raises(ExternalEndpointError, match="model exploded"):
                await client.stream_chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    temperature=0.0,
                )
        finally:
            await client.aclose()

    async def test_connect_error_mapped(self):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        client = _client(handler)
        try:
            with pytest.raises(ExternalEndpointError, match="Cannot connect"):
                await client.stream_chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    temperature=0.0,
                )
        finally:
            await client.aclose()


# =============================================================================
# Non-streaming chat completion
# =============================================================================


class TestChatCompletion:
    async def test_returns_text_and_usage(self):
        def handler(request):
            return _completion_response(text="B", prompt=42, completion=7)

        client = _client(handler)
        try:
            result = await client.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=0.0,
            )
        finally:
            await client.aclose()
        assert result.text == "B"
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 7
        assert result.finish_reason == "stop"
        assert result.status == "ok"

    async def test_empty_content_preserves_reasoning_diagnostics(self):
        def handler(request):
            return _completion_response(
                text=None,
                extra_message={"reasoning_content": "private reasoning"},
            )

        client = _client(handler)
        try:
            result = await client.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=None,
            )
        finally:
            await client.aclose()

        assert result.text == ""
        assert result.status == "empty_content"
        assert result.reasoning_fields_present == ("reasoning_content",)
        assert result.reasoning_fields_nonempty == ("reasoning_content",)

    async def test_length_finish_reason_is_truncated(self):
        def handler(request):
            return _completion_response(text="partial", finish_reason="length")

        client = _client(handler)
        try:
            result = await client.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=None,
            )
        finally:
            await client.aclose()

        assert result.status == "truncated"
        assert result.finish_reason == "length"

    async def test_non_text_content_is_invalid_response(self):
        def handler(request):
            return _completion_response(text=[{"type": "text", "text": "A"}])

        client = _client(handler)
        try:
            with pytest.raises(ExternalEndpointError) as exc_info:
                await client.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    temperature=None,
                )
        finally:
            await client.aclose()

        assert exc_info.value.status == "invalid_response"

    async def test_non_json_response_is_invalid_response(self):
        def handler(request):
            return httpx.Response(200, text="not json")

        client = _client(handler)
        try:
            with pytest.raises(ExternalEndpointError) as exc_info:
                await client.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    temperature=None,
                )
        finally:
            await client.aclose()

        assert exc_info.value.status == "invalid_response"

    async def test_unexpected_shape_raises(self):
        def handler(request):
            return httpx.Response(200, json={"unexpected": True})

        client = _client(handler)
        try:
            with pytest.raises(
                ExternalEndpointError, match="unexpected response"
            ) as exc_info:
                await client.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    temperature=0.0,
                )
        finally:
            await client.aclose()
        assert exc_info.value.status == "invalid_response"

    async def test_http_error_status_and_secret_redaction(self):
        def handler(request):
            return httpx.Response(
                500,
                json={"error": {"message": f"provider echoed {API_KEY}"}},
            )

        client = _client(handler)
        try:
            with pytest.raises(ExternalEndpointError) as exc_info:
                await client.chat_completion(
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=8,
                    temperature=None,
                )
        finally:
            await client.aclose()

        assert exc_info.value.status == "http_error"
        assert API_KEY not in str(exc_info.value)

    async def test_no_auth_header_without_key(self):
        captured = {}

        def handler(request):
            captured["auth"] = request.headers.get("authorization")
            return _completion_response()

        client = _client(handler, api_key="")
        try:
            await client.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=None,
            )
        finally:
            await client.aclose()
        assert captured["auth"] is None

    async def test_merges_provider_specific_extra_body(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return _completion_response()

        client = _client(
            handler,
            extra_body={"thinking": {"type": "disabled"}, "seed": 42},
        )
        try:
            await client.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                temperature=0.0,
            )
        finally:
            await client.aclose()

        assert captured["body"]["thinking"] == {"type": "disabled"}
        assert captured["body"]["seed"] == 42


# =============================================================================
# Accuracy adapter
# =============================================================================


class TestExternalChatAdapter:
    def _adapter(self, handler, profile="deterministic"):
        client = _client(handler)
        return ExternalChatAdapter(client, profile), client

    async def test_deterministic_sends_temperature_zero_whitelist_only(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return _completion_response()

        adapter, client = self._adapter(handler)
        try:
            # Simulate _eval_single's setdefault-injected kwargs — the
            # adapter must drop everything except the whitelist.
            output = await adapter.chat(
                messages=[{"role": "user", "content": "q"}],
                max_tokens=128,
                temperature=0.7,
                presence_penalty=0.0,
                repetition_penalty=1.0,
                chat_template_kwargs={"enable_thinking": False},
            )
        finally:
            await client.aclose()
        assert output.text == "A"
        body = captured["body"]
        assert set(body.keys()) == {"model", "messages", "max_tokens", "temperature"}
        assert body["temperature"] == 0.0
        assert body["max_tokens"] == 128

    async def test_model_settings_omits_sampling_params(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return _completion_response()

        adapter, client = self._adapter(handler, profile="model_settings")
        try:
            await adapter.chat(
                messages=[{"role": "user", "content": "q"}],
                max_tokens=64,
                temperature=0.0,
            )
        finally:
            await client.aclose()
        body = captured["body"]
        assert set(body.keys()) == {"model", "messages", "max_tokens"}

    async def test_text_uses_content_not_reasoning(self):
        def handler(request):
            return _completion_response(
                text="final", extra_message={"reasoning_content": "hidden"}
            )

        adapter, client = self._adapter(handler)
        try:
            output = await adapter.chat(messages=[{"role": "user", "content": "q"}])
        finally:
            await client.aclose()
        assert output.text == "final"

    def test_model_type_is_none(self):
        def handler(request):
            return _completion_response()

        adapter, _ = self._adapter(handler)
        assert adapter.model_type is None
        assert adapter.is_external_api is True

    async def test_preflight_raises_on_auth_error(self):
        def handler(request):
            return httpx.Response(401, json={"error": {"message": "nope"}})

        adapter, client = self._adapter(handler)
        try:
            with pytest.raises(ExternalEndpointError, match="rejected the API key"):
                await adapter.preflight()
        finally:
            await client.aclose()

    async def test_preflight_passes_on_success(self):
        def handler(request):
            return _completion_response(text="OK")

        adapter, client = self._adapter(handler)
        try:
            await adapter.preflight()
        finally:
            await client.aclose()

    @pytest.mark.parametrize(
        ("text", "extra_message", "finish_reason", "expected_status"),
        [
            (None, None, "stop", "empty_content"),
            (None, {"reasoning_content": "thinking"}, "stop", "empty_content"),
            ("partial", None, "length", "truncated"),
            ("not the sentinel", None, "stop", "parse_error"),
            ("NOT OK", None, "stop", "parse_error"),
        ],
    )
    async def test_preflight_rejects_incompatible_final_answer(
        self, text, extra_message, finish_reason, expected_status
    ):
        def handler(request):
            return _completion_response(
                text=text,
                extra_message=extra_message,
                finish_reason=finish_reason,
            )

        adapter, client = self._adapter(handler)
        try:
            with pytest.raises(ExternalEndpointError) as exc_info:
                await adapter.preflight()
        finally:
            await client.aclose()

        assert exc_info.value.status == expected_status

    async def test_question_error_is_returned_as_diagnostic_output(self):
        def handler(request):
            return httpx.Response(429, json={"error": {"message": "rate limited"}})

        adapter, client = self._adapter(handler)
        try:
            output = await adapter.chat(
                messages=[{"role": "user", "content": "q"}],
                max_tokens=8,
            )
        finally:
            await client.aclose()

        assert output.text == ""
        assert output.external_status == "http_error"
        assert "429" in output.error_message
