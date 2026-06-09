# SPDX-License-Identifier: Apache-2.0
"""Verify PrefillMemoryExceededError maps to HTTP 413 in server.py.

Regression-arming test for the actual prefill-guard chain validated
end-to-end on 2026-05-15: the message string format matches what the
guard surfaces in production, so a refactor that changes either the
error body shape or the HTTP code will be caught here.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.exceptions import PrefillMemoryExceededError


def _build_test_app():
    """Build a minimal FastAPI app that re-uses the production handler."""
    import omlx.server as srv

    app = FastAPI()
    app.add_exception_handler(
        PrefillMemoryExceededError, srv.prefill_memory_exceeded_handler
    )

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

    @app.get("/health/raise")
    def raise_prefill_too_large_health():
        raise PrefillMemoryExceededError(
            message="Prefill would require ~50 GB peak but limit is 40 GB.",
            request_id="req-xyz",
        )

    return app


class TestPrefillMemoryHandler:
    def test_returns_413(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        assert resp.status_code == 413

    def test_api_route_uses_openai_error_body(self):
        """/v1/* routes get the OpenAI-style {"error": {"message": ...}} wrapper."""
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        body = resp.json()
        assert "error" in body
        msg = body["error"]["message"]
        # The guard's diagnostic format is part of the public contract — the
        # CLI hint at the end tells the user exactly how to recover.
        assert "Prefill would require" in msg
        assert "KV+SDPA" in msg
        assert "--max-process-memory" in msg

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

    def test_non_api_route_uses_plain_detail(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/health/raise")
        body = resp.json()
        assert "detail" in body
        assert "Prefill would require" in body["detail"]


class TestResponsesEndpointReaches413:
    """End-to-end regression for ``/v1/responses``. The handler-shape tests
    above use a synthetic ``/v1/raise`` route, which proves the handler
    body but NOT the wiring of every prompt-bearing endpoint to the
    preflight call. ``/v1/responses`` is the one route most-likely to
    silently regress because it shares the StreamingResponse pattern
    with ``/v1/chat/completions`` and reaches preflight via the same
    code path. This test forces the preflight to raise and asserts
    the route returns 413 instead of 200/500.
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
        # exception handler and the route can still emit 413.
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

        async def _get_engine_for_model(model_id):
            return engine

        # Override the engine resolver and disable auth so the test
        # talks to the real route.
        srv.app.dependency_overrides[srv.verify_api_key] = lambda: True
        srv.get_engine_for_model = _get_engine_for_model  # type: ignore[assignment]

        return srv.app

    def test_v1_responses_returns_413_when_preflight_rejects(self):
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
                with patch.object(
                    srv, "resolve_model_id", lambda name: name
                ), patch.object(
                    srv, "validate_context_window", lambda *a, **k: None
                ):
                    resp = client.post(
                        "/v1/responses",
                        json={
                            "model": "test-model",
                            "input": "Hello, world.",
                            "stream": False,
                        },
                    )
            assert resp.status_code == 413, (
                f"expected 413, got {resp.status_code}: {resp.text}"
            )
            body = resp.json()
            assert "error" in body, body
            assert "Prefill would require" in body["error"]["message"]
            assert "--max-process-memory" in body["error"]["message"]
        finally:
            srv.get_engine_for_model = original_get_engine
            srv._server_state.engine_pool = original_engine_pool
            srv.app.dependency_overrides.clear()
            srv.app.dependency_overrides.update(original_overrides)
