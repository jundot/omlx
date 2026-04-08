"""Regression tests for SpecPrefill parameter forwarding in VLM engine."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import pytest

from omlx.engine.vlm import VLMBatchedEngine


@pytest.mark.asyncio
async def test_vlm_chat_forwards_specprefill_threshold_and_keep_pct():
    """VLM chat must pass both SpecPrefill overrides through to add_request()."""
    engine = VLMBatchedEngine(model_name="test-vlm")
    engine._loaded = True
    engine._tokenizer = MagicMock()
    engine._tokenizer.apply_chat_template.return_value = "<prompt>"
    engine._tokenizer.encode.side_effect = lambda text, **kwargs: list(range(max(1, len(text.split()))))
    engine._engine = MagicMock()
    engine._engine._mlx_executor = ThreadPoolExecutor(max_workers=1)
    engine._engine.add_request = AsyncMock(return_value="req-1")
    engine._engine.abort_request = AsyncMock(return_value=True)

    async def _one_output_stream(_request_id):
        yield MagicMock(
            output_text="ok",
            new_text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            tool_calls=None,
            cached_tokens=0,
        )

    engine._engine.stream_outputs = _one_output_stream

    async for _ in engine.stream_chat(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1,
        specprefill=True,
        specprefill_keep_pct=0.2,
        specprefill_threshold=1024,
    ):
        pass

    try:
        _, kwargs = engine._engine.add_request.call_args
        assert kwargs["specprefill"] is True
        assert kwargs["specprefill_keep_pct"] == 0.2
        assert kwargs["specprefill_threshold"] == 1024
    finally:
        engine._engine._mlx_executor.shutdown(wait=False)
