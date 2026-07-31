import json

import pytest

from omlx.api.openai_models import ChatCompletionRequest, Message
from omlx.engine.base import GenerationOutput
from omlx.server import stream_chat_completion


class _Tokenizer:
    chat_template = "test"

    @staticmethod
    def decode(token_ids):
        return {1: "你", 2: "好"}.get(token_ids[0], "")

    @staticmethod
    def apply_chat_template(*_args, **_kwargs):
        return "prompt"

    @staticmethod
    def encode(_prompt):
        return [7]


class _Engine:
    tokenizer = _Tokenizer()

    async def stream_chat(self, **_kwargs):
        yield GenerationOutput(
            text="你",
            new_text="你",
            prompt_tokens=1,
            completion_tokens=1,
            finished=False,
        )
        yield GenerationOutput(
            text="你",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            logprobs=[
                {"token_id": 1, "logprob": -0.1, "top": [(2, -0.3)]}
            ],
        )


@pytest.mark.asyncio
async def test_stream_emits_content_then_verified_logprobs_then_finish():
    request = ChatCompletionRequest(
        model="test",
        messages=[Message(role="user", content="hi")],
        stream=True,
        logprobs=True,
        top_logprobs=1,
    )
    events = [
        event
        async for event in stream_chat_completion(
            _Engine(),
            [{"role": "user", "content": "hi"}],
            request,
            logprobs=True,
            top_logprobs=1,
        )
    ]
    payloads = [
        json.loads(event.removeprefix("data: ").strip())
        for event in events
        if event.startswith("data: {")
    ]
    assert any(
        choice["delta"].get("content") == "你"
        for payload in payloads
        for choice in payload.get("choices", [])
    )
    logprob_choices = [
        choice
        for payload in payloads
        for choice in payload.get("choices", [])
        if "logprobs" in choice
    ]
    assert len(logprob_choices) == 1
    item = logprob_choices[0]["logprobs"]["content"][0]
    assert item["token"] == "你"
    assert item["top_logprobs"][0]["token"] == "好"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
