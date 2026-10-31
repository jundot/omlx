# SPDX-License-Identifier: Apache-2.0
"""Client-level proof that /v1/responses names the message phase.

These exercise both Responses paths against the real route table, so the phase
is checked on the bytes a coding harness actually reads: the completed
response body and the response.output_item.done frames.
"""

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from omlx.engine.batched import BatchedEngine  # noqa: E402
from omlx.server import _server_state, app  # noqa: E402


@dataclass
class MockGenerationOutput:
    """Mirror of the generation output BatchedEngine yields."""

    text: str = ""
    tokens: List[int] = field(default_factory=list)
    prompt_tokens: int = 10
    completion_tokens: int = 0
    finish_reason: Optional[str] = None
    new_text: str = ""
    finished: bool = False
    tool_calls: Optional[List[Dict[str, Any]]] = None
    cached_tokens: int = 0


class MockTokenizer:
    """Tokenizer stand-in: plain chat rendering, opt-in tool-call markers."""

    def __init__(self):
        self.eos_token_id = 2
        self.tool_parser = None
        self.tool_call_start = None
        self.tool_call_end = None
        self.has_tool_calling = False

    def encode(self, text: str) -> List[int]:
        return [100 + i for i, _ in enumerate(text.split())]

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return f"<decoded:{len(tokens)} tokens>"

    def apply_chat_template(
        self, messages: List[Dict], tokenize: bool = False, **kwargs
    ) -> str:
        return "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages
        )

    def chat_template(self):
        return (
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n"
            "{% endfor %}"
        )


class MockEnginePool:
    def __init__(self, engine):
        self._engine = engine
        self._models = [
            {"id": "test-model", "loaded": True, "pinned": False, "size": 1000000}
        ]
        self._entries = {}

    @property
    def model_count(self):
        return 1

    @property
    def loaded_model_count(self):
        return 1

    @property
    def max_model_memory(self):
        return 32 * 1024 * 1024 * 1024

    @property
    def current_model_memory(self):
        return 1000000

    def get_entry(self, model_id):
        return self._entries.get(model_id)

    def resolve_model_id(self, model_id_or_alias, settings_manager=None):
        return model_id_or_alias

    def get_model_ids(self):
        return [model["id"] for model in self._models]

    def get_status(self):
        return {"models": self._models}

    async def get_engine(self, model_id, _lease=False):
        return self._engine

    async def release_engine(self, model_id):
        return None

TOOLS = [
    {
        "type": "function",
        "name": "lookup",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
]


def _client(monkeypatch, body: str, *, tool_calling: bool):
    """A /v1/responses client whose model always generates ``body``."""
    engine = BatchedEngine("test-model")
    engine._loaded = True
    engine._model = SimpleNamespace(args=SimpleNamespace(model_type="qwen3"))
    engine._tokenizer = MockTokenizer()
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=object()))
    if tool_calling:
        engine._tokenizer.has_tool_calling = True
        engine._tokenizer.tool_call_start = "<tool_call>"
        engine._tokenizer.tool_call_end = "</tool_call>"
    monkeypatch.setattr(
        engine, "_preflight_or_raise_with_eviction", AsyncMock(), raising=False
    )

    output = MockGenerationOutput(
        text=body,
        new_text=body,
        completion_tokens=6,
        finished=True,
        finish_reason="stop",
    )
    monkeypatch.setattr(engine, "generate", AsyncMock(return_value=output))

    async def generate_stream(*args, **kwargs):
        yield output

    monkeypatch.setattr(engine, "stream_generate", generate_stream)
    monkeypatch.setattr(_server_state, "engine_pool", MockEnginePool(engine))
    monkeypatch.setattr(_server_state, "default_model", "test-model")
    return TestClient(app)


def _items(response, stream):
    if not stream:
        return response.json()["output"]
    return [
        event["item"]
        for line in response.text.splitlines()
        if line.startswith("data: ")
        for event in [json.loads(line[6:])]
        if event.get("type") == "response.output_item.done"
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "body,expect_phase",
    [
        (
            "Thinking first.<tool_call>{\"name\": \"lookup\", \"arguments\": "
            "{\"q\": \"weather\"}}</tool_call>",
            "commentary",
        ),
        ("9 sheep are left.", "final_answer"),
    ],
    ids=["tool_turn", "answer_turn"],
)
def test_tool_enabled_turns_name_their_phase(monkeypatch, stream, body, expect_phase):
    client = _client(monkeypatch, body, tool_calling=True)
    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "What is the weather in Paris?",
            "stream": stream,
            "tools": TOOLS,
        },
    )
    assert response.status_code == 200, response.text

    items = _items(response, stream)
    messages = [item for item in items if item["type"] == "message"]
    assert messages, items
    assert [item["phase"] for item in messages] == [expect_phase], items
    assert all(
        "phase" not in item for item in items if item["type"] == "function_call"
    )


@pytest.mark.parametrize("stream", [False, True])
def test_tool_less_turn_stays_unnamed(monkeypatch, stream):
    """Without tools there is nothing to fold calls under, so no phase is claimed."""
    client = _client(monkeypatch, "Just an answer.", tool_calling=False)
    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "Hi", "stream": stream},
    )
    assert response.status_code == 200, response.text

    items = _items(response, stream)
    messages = [item for item in items if item["type"] == "message"]
    assert messages, items
    assert all("phase" not in item for item in messages), items


@pytest.mark.parametrize("stream", [False, True])
def test_message_item_closes_after_its_tool_calls(monkeypatch, stream):
    """Codex groups a turn under the message, so it must be the closing item."""
    client = _client(
        monkeypatch,
        "Thinking first.<tool_call>{\"name\": \"lookup\", \"arguments\": "
        "{\"q\": \"weather\"}}</tool_call>",
        tool_calling=True,
    )
    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "Hi", "stream": stream, "tools": TOOLS},
    )
    assert response.status_code == 200, response.text

    items = _items(response, stream)
    calls = [item for item in items if item["type"] == "function_call"]
    assert calls, items
    messages = [item for item in items if item["type"] == "message"]
    assert messages and messages[0]["phase"] == "commentary", items
    if stream:
        # In the stream the message item closes last, after its tool calls, so
        # its phase names the turn they belong to.
        assert items[-1]["type"] == "message", items
        assert items.index(calls[0]) < len(items) - 1, items

    if stream:
        completed = next(
            json.loads(line[6:])["response"]
            for line in response.text.splitlines()
            if line.startswith("data: ")
            and json.loads(line[6:]).get("type") == "response.completed"
        )
        # The completed payload keeps spec order: message first, then the call.
        assert [item["type"] for item in completed["output"]] == [
            "message",
            "function_call",
        ], completed["output"]
        assert completed["output"][0]["phase"] == "commentary"
