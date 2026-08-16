# SPDX-License-Identifier: Apache-2.0
"""Verify PrefillMemoryExceededError maps to HTTP 400 in server.py.

Regression-arming test for the actual prefill-guard chain validated
end-to-end on 2026-05-15: the message string format matches what the
guard surfaces in production, so a refactor that changes either the
error body shape or the HTTP code will be caught here.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.exceptions import (
    ModelUnloadedError,
    PrefillMemoryAbortedError,
    PrefillMemoryExceededError,
)


def _build_test_app():
    """Build a minimal FastAPI app that re-uses the production handler."""
    import omlx.server as srv

    app = FastAPI()
    app.add_exception_handler(
        PrefillMemoryExceededError, srv.prefill_memory_exceeded_handler
    )
    app.add_exception_handler(ModelUnloadedError, srv.model_unloaded_handler)

    @app.get("/v1/raise")
    def raise_prefill_too_large():
        raise PrefillMemoryExceededError(
            message=(
                "Prefill would require ~43.56 GB peak "
                "(current 28.00 GB + KV+SDPA 15.56 GB) "
                "but limit is 40.00 GB. "
                "Reduce context length or increase --max-process-memory."
            ),
            request_id="req-abc",
            estimated_bytes=46_775_000_000,
            limit_bytes=42_949_672_960,
        )

    @app.get("/v1/raise-abort")
    def raise_prefill_aborted():
        raise PrefillMemoryAbortedError(
            message=(
                "Request aborted: process memory limit exceeded "
                "(usage 4.4 GB, abort threshold (hard watermark) 4.1 GB, "
                "dynamic ceiling 4.3 GB). "
                "Raise custom_ceiling_bytes in admin Memory settings."
            ),
            request_id="req-abort",
            limit_bytes=4_100_000_000,
        )

    @app.get("/health/raise")
    def raise_prefill_too_large_health():
        raise PrefillMemoryExceededError(
            message="Prefill would require ~50 GB peak but limit is 40 GB.",
            request_id="req-xyz",
        )

    @app.get("/v1/raise-unload")
    def raise_model_unloaded():
        raise ModelUnloadedError(
            message="Request aborted: model 'demo' was unloaded",
            request_id="req-unload",
        )

    return app


class TestPrefillMemoryHandler:
    def test_returns_400(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        assert resp.status_code == 400

    def test_api_route_uses_openai_error_body(self):
        """/v1/* routes get the OpenAI-style {"error": {"message": ...}} wrapper."""
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        body = resp.json()
        assert body["type"] == "error"
        assert "error" in body
        msg = body["error"]["message"]
        # The guard's diagnostic format is part of the public contract — the
        # CLI hint at the end tells the user exactly how to recover.
        assert "Prefill would require" in msg
        assert "KV+SDPA" in msg
        assert "--max-process-memory" in msg
        assert "Memory Guard to aggressive" in msg
        assert "custom memory guard ceiling" in msg
        assert body["error"]["code"] == "prefill_memory_exceeded"
        assert body["error"]["omlx_code"] == "prefill_memory_exceeded"

    def test_api_route_body_carries_estimated_and_limit_bytes(self):
        """Clients branch on the numeric ``estimated_bytes`` /
        ``limit_bytes`` fields rather than regex-matching the human
        message (which is localized / format-prone). Regression for
        the body-shape gap: prior to the fix on 2026-05-15 the handler
        embedded these numbers only inside ``message`` and dropped the
        structured fields, defeating the point of the typed exception
        carrying them.
        """
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        body = resp.json()
        assert body["error"]["estimated_bytes"] == 46_775_000_000
        assert body["error"]["limit_bytes"] == 42_949_672_960

    def test_mid_prefill_abort_reuses_the_400_mapping(self):
        """The enforcer's mid-prefill abort is the same memory condition as
        the pre-flight rejection and must reach the client the same way.
        Before the subclass existed it escaped as a bare RuntimeError, so
        the client got a truncated body and a 500 traceback instead."""
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise-abort")
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "prefill_memory_aborted"
        assert body["error"]["omlx_code"] == "prefill_memory_aborted"
        assert body["error"]["limit_bytes"] == 4_100_000_000

    def test_abort_wording_does_not_claim_the_prompt_was_rejected(self):
        """This request was admitted and then killed, so the pre-flight
        wording would misdescribe it — and its message already carries the
        binding ceiling plus advice, so the generic ladder is not appended."""
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise-abort")
        msg = resp.json()["error"]["message"]
        assert "aborted this request mid-prefill" in msg
        assert "rejected this prompt" not in msg
        assert "Memory Guard to aggressive" not in msg

    def test_non_api_route_uses_plain_detail(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/health/raise")
        body = resp.json()
        assert "detail" in body
        assert "Prefill would require" in body["detail"]
        assert body["omlx_code"] == "prefill_memory_exceeded"


class TestModelUnloadedHandler:
    """A ModelUnloadedError that escapes before the response starts must
    reach the client as a clean, typed 503 -- not fall through to the
    generic 500 handler and lose the message. No current production path
    raises it that early (generation errors surface inside the keepalive
    wrappers), so this pins the handler's contract as the backstop for
    any future pre-response raise site."""

    def test_returns_503(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise-unload")
        assert resp.status_code == 503

    def test_api_route_uses_openai_error_body(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise-unload")
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["message"] == "Request aborted: model 'demo' was unloaded"
        assert body["error"]["code"] == "model_unloaded"
        assert body["error"]["omlx_code"] == "model_unloaded"
        # 503 (server_error), not 400 (invalid_request_error) -- retryable,
        # operator-initiated, not a malformed/oversized client request.
        assert body["error"]["type"] == "server_error"
        assert "memory" not in body["error"]["message"].lower()


class TestPostCommitPrefillMemorySurface:
    @pytest.mark.asyncio
    async def test_json_keepalive_emits_openai_error_body(self):
        import json

        import omlx.server as srv

        class _Request:
            async def is_disconnected(self):
                return False

        async def _raise_late():
            raise PrefillMemoryExceededError(
                message="Prefill context too large for available memory",
                request_id="req-json",
                estimated_bytes=123,
                limit_bytes=100,
            )

        chunks = [
            chunk
            async for chunk in srv._with_json_keepalive(
                _Request(), _raise_late(), disconnect_poll=0.001
            )
        ]
        body = json.loads("".join(chunks))
        assert body["error"]["code"] == "prefill_memory_exceeded"
        assert body["error"]["omlx_code"] == "prefill_memory_exceeded"
        assert body["error"]["estimated_bytes"] == 123
        assert body["error"]["limit_bytes"] == 100

    @pytest.mark.asyncio
    async def test_sse_keepalive_emits_openai_error_chunk(self):
        import json

        import omlx.server as srv

        async def _gen():
            raise PrefillMemoryExceededError(
                message="Prefill context too large for available memory",
                request_id="req-sse",
                estimated_bytes=123,
                limit_bytes=100,
            )
            yield ""

        chunks = [
            chunk
            async for chunk in srv._with_sse_keepalive(_gen(), keepalive_chunk=None)
        ]
        assert chunks[-1] == "data: [DONE]\n\n"
        data = chunks[0].removeprefix("data: ").strip()
        body = json.loads(data)
        assert body["error"]["code"] == "prefill_memory_exceeded"
        assert body["error"]["omlx_code"] == "prefill_memory_exceeded"


class TestPostCommitModelUnloadedSurface:
    """A manual model unload mid-request must surface the same way the
    memory-guard abort does above: a clean, parseable JSON error body, not
    a truncated read.

    Before this wrapper case existed, error_code="model_unloaded" fell
    through _raise_request_output_error() to a bare RuntimeError, which
    _with_json_keepalive() did not catch. Since the response had already
    started (headers + leading keepalive space already on the wire), the
    uncaught exception left the client with a dropped connection instead
    of an error body.
    """

    @pytest.mark.asyncio
    async def test_json_keepalive_emits_model_unloaded_body(self):
        import json

        import omlx.server as srv

        class _Request:
            async def is_disconnected(self):
                return False

        async def _raise_late():
            raise ModelUnloadedError(
                message="Request aborted: model 'demo' was unloaded",
                request_id="req-unload",
            )

        chunks = [
            chunk
            async for chunk in srv._with_json_keepalive(
                _Request(), _raise_late(), disconnect_poll=0.001
            )
        ]
        # A truncated/uncaught path would raise out of the loop above
        # instead of yielding a complete body to parse here.
        body = json.loads("".join(chunks))
        assert body["type"] == "error"
        assert body["error"]["code"] == "model_unloaded"
        assert body["error"]["omlx_code"] == "model_unloaded"
        assert body["error"]["message"] == "Request aborted: model 'demo' was unloaded"
        assert "memory" not in body["error"]["message"].lower()
        # 503 (transient, retryable), not 400 (invalid_request_error, which
        # OpenAI-SDK clients treat as do-not-retry) -- an operator unload is
        # a server condition, not a malformed/oversized request.
        assert body["error"]["type"] == "server_error"

    @pytest.mark.asyncio
    async def test_sse_keepalive_emits_model_unloaded_chunk(self):
        """Drives ``_with_sse_keepalive`` directly: an escaping
        ModelUnloadedError yields the same code/omlx_code fields as the
        JSON-keepalive path above. On today's streaming routes the route
        generator's own except block terminates the stream first, so
        this branch is the wrapper-level backstop, mirroring the prefill
        branch beside it."""
        import json

        import omlx.server as srv

        async def _gen():
            raise ModelUnloadedError(
                message="Request aborted: model 'demo' was unloaded",
                request_id="req-sse-unload",
            )
            yield ""

        chunks = [
            chunk
            async for chunk in srv._with_sse_keepalive(_gen(), keepalive_chunk=None)
        ]
        assert chunks[-1] == "data: [DONE]\n\n"
        data = chunks[0].removeprefix("data: ").strip()
        body = json.loads(data)
        assert body["error"]["code"] == "model_unloaded"
        assert body["error"]["omlx_code"] == "model_unloaded"
        assert body["error"]["type"] == "server_error"


class TestResponsesEndpointReaches400:
    """End-to-end regression for ``/v1/responses``. The handler-shape tests
    above use a synthetic ``/v1/raise`` route, which proves the handler
    body but NOT the wiring of every prompt-bearing endpoint to the
    preflight call. ``/v1/responses`` is the one route most-likely to
    silently regress because it shares the StreamingResponse pattern
    with ``/v1/chat/completions`` and reaches preflight via the same
    code path. This test forces the preflight to raise and asserts
    the route returns 400 instead of 200/500.
    """

    def _make_app_with_failing_preflight(self):
        """Mount the real ``/v1/responses`` route with a mocked
        engine_pool that returns an engine whose ``preflight_chat``
        raises ``PrefillMemoryExceededError``. Hits the *production*
        handler — not a synthesized stub — so a wiring regression is
        caught.
        """
        from unittest.mock import AsyncMock, MagicMock

        import omlx.server as srv

        # Build an engine mock whose preflight_chat raises. The
        # production handler awaits this BEFORE constructing
        # StreamingResponse, so the raise propagates to the
        # exception handler and the route can still emit 400.
        async def _raising_preflight(*args, **kwargs):
            raise PrefillMemoryExceededError(
                message=(
                    "Prefill would require ~50 GB peak "
                    "(current 30 GB + KV+SDPA 20 GB) but limit "
                    "is 40 GB. Reduce context length or "
                    "increase --max-process-memory."
                ),
                request_id="req-responses",
                estimated_bytes=53_687_091_200,
                limit_bytes=42_949_672_960,
            )

        engine = MagicMock()
        engine.preflight_chat = AsyncMock(side_effect=_raising_preflight)
        engine.start = AsyncMock()
        # The handler calls ``count_chat_tokens`` and feeds the result
        # into ``validate_context_window``; without a real int the
        # comparison ``num_prompt_tokens > max_context`` raises before
        # preflight ever runs.
        engine.count_chat_tokens = MagicMock(return_value=128)

        async def _get_engine_for_model(model_id, *, lease=None):
            return engine

        # Override the engine resolver and disable auth so the test
        # talks to the real route.
        srv.app.dependency_overrides[srv.verify_api_key] = lambda: True
        srv.get_engine_for_model = _get_engine_for_model  # type: ignore[assignment]

        return srv.app

    def test_v1_responses_returns_400_when_preflight_rejects(self):
        from unittest.mock import MagicMock, patch

        import omlx.server as srv

        original_get_engine = srv.get_engine_for_model
        original_overrides = dict(srv.app.dependency_overrides)
        original_engine_pool = srv._server_state.engine_pool
        try:
            app = self._make_app_with_failing_preflight()
            # Mock engine_pool so get_engine_pool() doesn't raise 503.
            # get_entry returns None so the handler's preserve_thinking
            # short-circuit doesn't fire.
            from unittest.mock import AsyncMock

            fake_pool = MagicMock()
            fake_pool.get_entry = MagicMock(return_value=None)
            fake_pool.preload_pinned_models = AsyncMock()
            fake_pool.check_ttl_expirations = AsyncMock()
            fake_pool.shutdown = AsyncMock()
            srv._server_state.engine_pool = fake_pool
            with TestClient(app, raise_server_exceptions=False) as client:
                with (
                    patch.object(srv, "resolve_model_id", lambda name: name),
                    patch.object(srv, "validate_context_window", lambda *a, **k: None),
                ):
                    resp = client.post(
                        "/v1/responses",
                        json={
                            "model": "test-model",
                            "input": "Hello, world.",
                            "stream": False,
                        },
                    )
            assert (
                resp.status_code == 400
            ), f"expected 400, got {resp.status_code}: {resp.text}"
            body = resp.json()
            assert "error" in body, body
            assert "Prefill would require" in body["error"]["message"]
            assert "--max-process-memory" in body["error"]["message"]
        finally:
            srv.get_engine_for_model = original_get_engine
            srv._server_state.engine_pool = original_engine_pool
            srv.app.dependency_overrides.clear()
            srv.app.dependency_overrides.update(original_overrides)
