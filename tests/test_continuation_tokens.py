import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from omlx.api.openai_models import ChatCompletionRequest, Message, StreamOptions
from omlx.engine.base import GenerationOutput
from omlx.engine.batched import BatchedEngine
from omlx.engine.dflash import DFlashEngine
from omlx.engine.vlm import VLMBatchedEngine
from omlx.server import (
    _build_protocol_pause_preview,
    _count_reasoning_token_ids,
    _resolve_engine_think_end_token_ids,
    stream_chat_completion,
)


class _ContinuationTokenizer:
    think_start_id = 8
    think_end_id = 9
    unk_token_id = -1

    def encode(self, text, add_special_tokens=False):
        if text == "</think>":
            return [self.think_end_id]
        if text == "<think>":
            return [self.think_start_id]
        if text == "base-prompt":
            return [10, 11]
        return [ord(char) for char in str(text)]

    def decode(self, token_ids, skip_special_tokens=False):
        if list(token_ids) == [1, 2]:
            return "old reasoning "
        pieces = {3: "more", 4: "answer", 9: "</think>"}
        return "".join(pieces.get(int(token_id), "") for token_id in token_ids)


def test_continuation_request_requires_a_real_token_prefix():
    with pytest.raises(ValidationError, match="must not be empty"):
        ChatCompletionRequest(
            model="model",
            messages=[Message(role="user", content="hello")],
            continuation_token_ids=[],
        )

    with pytest.raises(ValidationError, match="requires continuation_token_ids"):
        ChatCompletionRequest(
            model="model",
            messages=[Message(role="user", content="hello")],
            continuation_in_thinking=True,
        )


def test_reasoning_token_count_stops_at_raw_close_marker():
    tokenizer = _ContinuationTokenizer()
    assert _count_reasoning_token_ids(
        [1, 2, 3],
        tokenizer,
        starts_in_thinking=True,
        still_in_thinking=True,
    ) == 3
    assert _count_reasoning_token_ids(
        [1, 2, 3, 9, 4],
        tokenizer,
        starts_in_thinking=True,
        still_in_thinking=False,
    ) == 3


def test_reasoning_token_count_uses_nested_engine_native_close_marker():
    class _Scheduler:
        @staticmethod
        def _resolve_think_end_token_ids():
            return [70, 71]

    engine = SimpleNamespace(
        _engine=SimpleNamespace(
            engine=SimpleNamespace(scheduler=_Scheduler()),
        )
    )
    native_close = _resolve_engine_think_end_token_ids(engine)

    assert native_close == [70, 71]
    assert _count_reasoning_token_ids(
        [1, 2, 70, 71, 3],
        _ContinuationTokenizer(),
        starts_in_thinking=True,
        still_in_thinking=False,
        native_close_token_ids=native_close,
    ) == 2


def test_protocol_pause_preview_normalizes_native_reasoning_markers():
    class _Result:
        def __init__(self, stream_text=""):
            self.stream_text = stream_text

    class _Session:
        pieces = {1: "reason", 70: "</think>\n", 2: "answer"}

        def notify_prefilled_thought(self):
            pass

        def process_token(self, token_id):
            return _Result(self.pieces[token_id])

        def finalize(self):
            return _Result()

    factory = SimpleNamespace(
        create_session=lambda tokenizer: _Session(),
        thinking_start_output_text="<think>\n",
    )
    engine = SimpleNamespace(
        tokenizer=_ContinuationTokenizer(),
        _engine=SimpleNamespace(
            engine=SimpleNamespace(
                scheduler=SimpleNamespace(_output_parser_factory=factory)
            )
        ),
    )

    preview = _build_protocol_pause_preview(
        engine,
        {
            "output_token_ids": [1, 70, 2],
            "continuation_in_thinking": False,
            "reasoning_end_token_index": 1,
        },
    )

    assert preview == {"reasoning_content": "reason", "content": "answer"}


@pytest.mark.asyncio
async def test_stream_exposes_incremental_ids_and_cumulative_reasoning_count():
    tokenizer = _ContinuationTokenizer()

    class _Engine:
        def __init__(self):
            self.tokenizer = tokenizer

        def _apply_chat_template(self, *args, **kwargs):
            return "base-prompt"

        async def stream_chat(self, **kwargs):
            yield GenerationOutput(
                text="more</think>answer",
                new_text="more</think>answer",
                tokens=[3, 9, 4],
                prompt_tokens=4,
                completion_tokens=3,
                finished=True,
                finish_reason="stop",
            )

    request = ChatCompletionRequest(
        model="model",
        messages=[Message(role="user", content="hello")],
        stream=True,
        stream_options=StreamOptions(include_token_ids=True),
        continuation_token_ids=[1, 2],
        continuation_in_thinking=True,
    )
    events = []
    async for event in stream_chat_completion(
        _Engine(),
        [{"role": "user", "content": "hello"}],
        request,
        max_tokens=8,
        continuation_token_ids=[1, 2],
        continuation_in_thinking=True,
    ):
        if event.startswith("data: {"):
            events.append(json.loads(event[6:]))

    metadata = [
        event["choices"][0]["delta"]
        for event in events
        if event.get("choices")
        and event["choices"][0]["delta"].get("token_ids")
    ]
    assert metadata == [{"token_ids": [3, 9, 4], "reasoning_token_count": 3}]
    reasoning = "".join(
        event["choices"][0]["delta"].get("reasoning_content", "")
        for event in events
        if event.get("choices")
    )
    content = "".join(
        event["choices"][0]["delta"].get("content", "")
        for event in events
        if event.get("choices")
    )
    assert reasoning == "more"
    assert content == "answer"


@pytest.mark.asyncio
async def test_stream_uses_protocol_native_reasoning_boundary_for_token_metadata():
    tokenizer = _ContinuationTokenizer()

    class _Scheduler:
        @staticmethod
        def _resolve_think_end_token_ids():
            return [70, 71]

    class _Engine:
        def __init__(self):
            self.tokenizer = tokenizer
            self._engine = SimpleNamespace(
                engine=SimpleNamespace(scheduler=_Scheduler())
            )

        def _apply_chat_template(self, *args, **kwargs):
            return "base-prompt"

        async def stream_chat(self, **kwargs):
            yield GenerationOutput(
                text="more</think>answer",
                new_text="more</think>answer",
                tokens=[3, 70, 71, 4],
                prompt_tokens=6,
                completion_tokens=4,
                finished=True,
                finish_reason="stop",
            )

    request = ChatCompletionRequest(
        model="model",
        messages=[Message(role="user", content="hello")],
        stream=True,
        stream_options=StreamOptions(include_token_ids=True),
        continuation_token_ids=[1, 2],
        continuation_in_thinking=True,
    )
    events = []
    async for event in stream_chat_completion(
        _Engine(),
        [{"role": "user", "content": "hello"}],
        request,
        max_tokens=8,
        continuation_token_ids=[1, 2],
        continuation_in_thinking=True,
    ):
        if event.startswith("data: {"):
            events.append(json.loads(event[6:]))

    metadata = [
        event["choices"][0]["delta"]
        for event in events
        if event.get("choices")
        and event["choices"][0]["delta"].get("token_ids")
    ]
    assert metadata == [
        {
            "token_ids": [3, 70, 71, 4],
            "reasoning_token_count": 3,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("engine_type", "tokenizer_attr"),
    [(BatchedEngine, "_tokenizer"), (DFlashEngine, "_tokenizer_obj")],
)
async def test_text_chat_engines_append_raw_continuation_after_template(
    engine_type, tokenizer_attr
):
    engine = engine_type.__new__(engine_type)
    engine._loaded = True
    tokenizer = _ContinuationTokenizer()
    setattr(engine, tokenizer_attr, tokenizer)
    if engine_type is BatchedEngine:
        engine._preprocess_messages = lambda messages: messages
        engine._inject_specprefill_system_end = lambda *args, **kwargs: None
    else:
        engine._in_fallback_mode = False
        engine._fallback_engine = None
        engine._fallback_engine_type = "llm"
    captured = {}

    engine._apply_chat_template = lambda *args, **kwargs: "base-prompt"

    async def _stream_generate(**kwargs):
        captured.update(kwargs)
        yield GenerationOutput(text="", new_text="", finished=True)

    engine.stream_generate = _stream_generate
    outputs = [
        output
        async for output in engine.stream_chat(
            [{"role": "user", "content": "hello"}],
            continuation_token_ids=[90, 91],
        )
    ]
    assert outputs
    assert captured["prompt"] == [10, 11, 90, 91]


def test_vlm_prompt_processing_appends_continuation_after_vision_tokens(monkeypatch):
    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._apply_ocr_prompt = lambda messages: messages
    engine._prepare_vision_inputs = lambda *args, **kwargs: (
        [10, 11],
        None,
        None,
        None,
        0,
        [],
    )
    monkeypatch.setattr(
        "omlx.engine.vlm.extract_images_from_messages",
        lambda messages: (messages, [], []),
    )
    prompt, *_ = engine._process_chat_messages(
        [{"role": "user", "content": "hello"}],
        None,
        {"continuation_token_ids": [90, 91]},
    )
    assert prompt == [10, 11, 90, 91]
