# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest

from omlx.api.openai_models import Usage
from omlx.server import _usage_timing_fields


def test_non_streaming_usage_carries_rates():
    fields = _usage_timing_fields(
        1200, 300, ttft=0.6, prefill_duration=0.6, generation_duration=4.0
    )
    dumped = Usage(
        prompt_tokens=1200,
        completion_tokens=300,
        total_tokens=1500,
        total_time=4.6,
        **fields,
    ).model_dump(exclude_none=True)
    assert dumped["time_to_first_token"] == 0.6
    assert dumped["prompt_eval_duration"] == 0.6
    assert dumped["generation_duration"] == 4.0
    assert dumped["prompt_tokens_per_second"] == 2000.0
    assert dumped["generation_tokens_per_second"] == 75.0


def test_non_streaming_usage_drops_fields_without_timings():
    fields = _usage_timing_fields(
        10, 0, ttft=None, prefill_duration=0.0, generation_duration=0.0
    )
    dumped = Usage(
        prompt_tokens=10, completion_tokens=0, total_tokens=10, total_time=0.1, **fields
    ).model_dump(exclude_none=True)
    assert "time_to_first_token" not in dumped
    assert "prompt_eval_duration" not in dumped
    assert "prompt_tokens_per_second" not in dumped
    assert "generation_tokens_per_second" not in dumped


def test_unknown_first_token_time_reports_no_ttft():
    fields = _usage_timing_fields(
        50, 20, ttft=None, prefill_duration=0.0, generation_duration=2.0
    )
    assert fields["time_to_first_token"] is None
    assert fields["prompt_eval_duration"] is None
    assert fields["prompt_tokens_per_second"] is None
    assert fields["generation_tokens_per_second"] == 10.0


def test_prompt_rate_counts_only_uncached_tokens():
    fields = _usage_timing_fields(
        165514,
        100,
        ttft=0.5,
        prefill_duration=0.5,
        generation_duration=2.0,
        cached_tokens=164623,
    )
    assert fields["prompt_tokens_per_second"] == 1782.0  # 891 computed tokens / 0.5 s
    full = _usage_timing_fields(
        1000,
        10,
        ttft=0.5,
        prefill_duration=0.5,
        generation_duration=1.0,
        cached_tokens=1000,
    )
    assert full["prompt_tokens_per_second"] == 0.0


class TestNonStreamingUsageEndpoint:
    """Handler-level coverage: rate fields must survive into the HTTP usage."""

    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        from omlx import server
        from omlx.settings import GlobalSettings

        settings = GlobalSettings(base_path=tmp_path)
        settings.auth.allow_unauthenticated_inference = True
        monkeypatch.setattr(server._server_state, "global_settings", settings)
        monkeypatch.setattr(server._server_state, "api_key", None)
        monkeypatch.setattr(
            server, "get_max_context_window", lambda model_id=None: None
        )

        pool = MagicMock()
        pool.resolve_model_id.side_effect = lambda mid, _sm: mid
        from omlx.engine.base import BaseEngine

        engine = MagicMock(spec=BaseEngine)
        engine.is_diffusion_model = False

        async def mock_get_engine(model_id, **kwargs):
            return engine

        pool.get_engine = AsyncMock(side_effect=mock_get_engine)
        pool.release_engine = AsyncMock()
        pool.get_abort_requested_reason = MagicMock(return_value=None)
        monkeypatch.setattr(server._server_state, "engine_pool", pool)
        self.engine = engine
        engine.count_chat_tokens = MagicMock(return_value=4)
        engine.tokenizer = MagicMock()
        engine.tokenizer.tool_call_start = None
        engine.tokenizer.tool_call_end = None
        return TestClient(server.app)

    def test_chat_usage_reports_rates(self, client):
        import asyncio
        import time

        from omlx.engine.base import GenerationOutput

        async def chat_with_first_token(messages, **kwargs):
            await asyncio.sleep(0.15)
            return GenerationOutput(
                text="hello world",
                prompt_tokens=100,
                completion_tokens=50,
                finish_reason="stop",
                cached_tokens=90,
                first_token_at=time.perf_counter(),
            )

        self.engine.chat = AsyncMock(side_effect=chat_with_first_token)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        usage = resp.json()["usage"]
        assert usage["time_to_first_token"] > 0
        assert usage["prompt_eval_duration"] > 0
        assert usage["prompt_tokens_per_second"] > 0
        assert usage["generation_tokens_per_second"] > 0

    def test_chat_usage_omits_rates_without_first_token(self, client):
        from omlx.engine.base import GenerationOutput

        self.engine.chat = AsyncMock(
            return_value=GenerationOutput(
                text="hi there",
                prompt_tokens=100,
                completion_tokens=20,
                finish_reason="stop",
                cached_tokens=0,
            )
        )
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        usage = resp.json()["usage"]
        assert "time_to_first_token" not in usage
        assert "prompt_eval_duration" not in usage
        assert "prompt_tokens_per_second" not in usage
        assert usage["generation_tokens_per_second"] > 0

    def test_multi_prompt_usage_omits_timing_fields(self, client):
        import time

        from omlx.engine.base import GenerationOutput

        self.engine.generate = AsyncMock(
            side_effect=[
                GenerationOutput(
                    text="a",
                    prompt_tokens=10,
                    completion_tokens=5,
                    first_token_at=time.perf_counter() + 0.1,
                ),
                GenerationOutput(text="b", prompt_tokens=10, completion_tokens=5),
            ]
        )
        resp = client.post(
            "/v1/completions", json={"model": "m", "prompt": ["p1", "p2"]}
        )
        assert resp.status_code == 200
        usage = resp.json()["usage"]
        assert "time_to_first_token" not in usage
        assert "prompt_eval_duration" not in usage
        assert "prompt_tokens_per_second" not in usage
        assert "generation_duration" not in usage
        assert "generation_tokens_per_second" not in usage
        assert usage["prompt_tokens"] == 20
