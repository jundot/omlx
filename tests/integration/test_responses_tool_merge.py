# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for two real bugs in /v1/responses' tool-merge logic in
omlx/server.py's create_response():

1. MCP tool-merge ordering: MCP-provided tools must reach the model even
   when the client sent no tools of its own. /v1/responses gated the merge
   on ``openai_tools`` (the client-supplied list) being truthy, silently
   dropping MCP tools whenever the client didn't also send tools of its
   own -- unlike the equivalent /v1/chat/completions path, which has always
   merged unconditionally.

2. ``tool_choice="none"`` template suppression: /v1/responses' ``tools_
   disabled`` computation never checked ``request.tool_choice`` at all, only
   whether the model was a diffusion model lacking tool-calling support --
   so a client that set ``tool_choice="none"`` still got its tools (and any
   MCP-merged tools) exposed to the chat template on this endpoint, unlike
   the equivalent /v1/chat/completions call.

Both are small, independently-mergeable fixes.

Uses a self-contained mock engine/pool (does not import from
test_server_endpoints.py to keep this file independent) with a
FastAPI TestClient, mirroring the mocking conventions used elsewhere in
tests/integration/.
"""

from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from omlx.engine.base import BaseEngine
from omlx.mcp.tools import merge_tools
from omlx.mcp.types import MCPTool


class MockTokenizer:
    """Mock tokenizer sufficient for chat-template / token counting."""

    def __init__(self):
        self.eos_token_id = 2

    def encode(self, text: str) -> List[int]:
        return [100 + i for i, _ in enumerate(text.split())]

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return f"<decoded:{len(tokens)} tokens>"

    def apply_chat_template(self, messages: List[Dict], tokenize: bool = False, **kwargs) -> str:
        parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
        return "\n".join(parts)


class MockGenerationOutput:
    def __init__(self, text="Chat response.", **kwargs):
        self.text = text
        self.tokens = kwargs.get("tokens", [1, 2, 3])
        self.prompt_tokens = kwargs.get("prompt_tokens", 10)
        self.completion_tokens = kwargs.get("completion_tokens", 5)
        self.finish_reason = kwargs.get("finish_reason", "stop")
        self.tool_calls = kwargs.get("tool_calls", None)
        self.cached_tokens = kwargs.get("cached_tokens", 0)


class RecordingEngine(BaseEngine):
    """Mock LLM engine that records the kwargs passed to chat().

    Subclasses BaseEngine (rather than a bare object) so it satisfies the
    ``isinstance(engine, BaseEngine)`` check server.py's get_engine() added
    for #507 (rejecting non-LLM engines with a clear 400 instead of an
    unhandled 500) — mirrors MockBaseEngine in test_server_endpoints.py.
    """

    def __init__(self, grammar_compiler=None):
        self._model_name = "test-model"
        self._tokenizer = MockTokenizer()
        self._model_type = "llama"
        self._grammar_compiler = grammar_compiler
        self.recorded_chat_kwargs: List[Dict[str, Any]] = []
        self.started = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_type(self) -> Optional[str]:
        return self._model_type

    @property
    def grammar_compiler(self):
        # BaseEngine.grammar_compiler is a plain (non-abstract) property
        # defaulting to None; override it here since __init__ can no longer
        # assign the instance attribute directly (no setter).
        return self._grammar_compiler

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        pass

    async def generate(self, prompt, **kwargs) -> MockGenerationOutput:
        return MockGenerationOutput(text="Generated response.")

    async def stream_generate(self, prompt, **kwargs):
        yield MockGenerationOutput(text="Hello world.")

    def count_chat_tokens(self, messages, tools=None, chat_template_kwargs=None, **kwargs) -> int:
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        return len(self._tokenizer.encode(prompt))

    async def chat(self, messages, **kwargs) -> MockGenerationOutput:
        self.recorded_chat_kwargs.append(kwargs)
        return MockGenerationOutput(text="Chat response.")

    async def stream_chat(self, messages, **kwargs):
        yield MockGenerationOutput(text="Chat response.")

    def get_stats(self) -> Dict[str, Any]:
        return {}

    def get_cache_stats(self):
        return None


class MockEnginePool:
    def __init__(self, engine):
        self._engine = engine
        self._models = [{"id": "test-model", "loaded": True, "pinned": False, "size": 1}]

    @property
    def model_count(self):
        return len(self._models)

    @property
    def loaded_model_count(self):
        return 1

    @property
    def max_model_memory(self):
        return 32 * 1024 * 1024 * 1024

    @property
    def current_model_memory(self):
        return 1

    def get_entry(self, model_id):
        return None

    def resolve_model_id(self, model_id_or_alias, settings_manager=None):
        return model_id_or_alias

    def get_model_ids(self):
        return [m["id"] for m in self._models]

    def get_status(self):
        return {
            "models": self._models,
            "loaded_count": self.loaded_model_count,
            "max_model_memory": self.max_model_memory,
        }

    async def get_engine(self, model_id, **kwargs):
        # kwargs absorb the lease protocol added by _LLMEngineLease
        # (``_lease``, optionally ``runtime_settings``) that the real
        # EnginePool.get_engine() now accepts — see get_engine() in
        # omlx/server.py, which always passes these through.
        return self._engine

    async def release_engine(self, model_id):
        # Counterpart to the lease taken above; _LLMEngineLease.release()
        # calls this unconditionally once a lease's model_id is set.
        pass


class FakeMCPManager:
    """Minimal stand-in for MCPClientManager exposing only get_merged_tools.

    Backed by the real merge_tools() implementation so the merge semantics
    under test are the genuine production logic, not a re-implementation.
    """

    def __init__(self, mcp_tools: List[MCPTool]):
        self._mcp_tools = mcp_tools

    def get_merged_tools(self, user_tools=None):
        return merge_tools(self._mcp_tools, user_tools)


MCP_WEATHER_TOOL = MCPTool(
    server_name="weather",
    name="get_weather",
    description="Get the weather",
    input_schema={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
)


@pytest.fixture()
def engine():
    return RecordingEngine()


@pytest.fixture()
def client_with_mcp(engine):
    """TestClient with an MCP manager configured but the client sending no tools."""
    from omlx.server import app, _server_state

    original_pool = _server_state.engine_pool
    original_default = _server_state.default_model
    original_mcp = _server_state.mcp_manager
    original_settings = _server_state.settings_manager

    _server_state.engine_pool = MockEnginePool(engine)
    _server_state.default_model = "test-model"
    _server_state.mcp_manager = FakeMCPManager([MCP_WEATHER_TOOL])
    _server_state.settings_manager = None

    yield TestClient(app)

    _server_state.engine_pool = original_pool
    _server_state.default_model = original_default
    _server_state.mcp_manager = original_mcp
    _server_state.settings_manager = original_settings


class TestMCPMergeOrderingChatCompletions:
    """/v1/chat/completions already merged unconditionally; guard against regression."""

    def test_mcp_tools_reach_engine_even_without_client_tools(self, client_with_mcp, engine):
        resp = client_with_mcp.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "What's the weather?"}],
            },
        )
        assert resp.status_code == 200
        assert engine.recorded_chat_kwargs, "engine.chat was never called"
        tools = engine.recorded_chat_kwargs[-1].get("tools")
        assert tools is not None
        names = {t["function"]["name"] for t in tools}
        assert "weather__get_weather" in names


class TestMCPMergeOrderingResponses:
    """Regression test for the real bug found in /v1/responses.

    Before the fix, the merge was gated on ``openai_tools`` (the
    client-supplied list) being truthy, so an MCP server's tools never
    reached the model when the client sent none of its own — even though
    /v1/chat/completions has always merged unconditionally.
    """

    def test_mcp_tools_reach_engine_even_without_client_tools(self, client_with_mcp, engine):
        resp = client_with_mcp.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "What's the weather?",
            },
        )
        assert resp.status_code == 200
        assert engine.recorded_chat_kwargs, "engine.chat was never called"
        tools = engine.recorded_chat_kwargs[-1].get("tools")
        assert tools is not None, (
            "MCP tools were dropped: the merge must run even when the "
            "client sent no tools of its own"
        )
        names = {t["function"]["name"] for t in tools}
        assert "weather__get_weather" in names

    def test_client_tools_still_override_mcp_on_name_conflict(self, client_with_mcp, engine):
        """User tools win on name conflicts — merge_tools' documented contract."""
        resp = client_with_mcp.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "hi",
                "tools": [{
                    "type": "function",
                    "name": "weather__get_weather",
                    "description": "client override",
                    "parameters": {"type": "object", "properties": {}},
                }],
            },
        )
        assert resp.status_code == 200
        tools = engine.recorded_chat_kwargs[-1].get("tools")
        matching = [t for t in tools if t["function"]["name"] == "weather__get_weather"]
        assert len(matching) == 1
        assert matching[0]["function"]["description"] == "client override"


class TestToolChoiceNoneSuppressesTemplateExposure:
    """Regression test for a real bug found in /v1/responses.

    /v1/chat/completions has always gated tool exposure to the chat template
    on ``request.tool_choice == "none"`` (``tools_disabled`` in
    create_chat_completion). /v1/responses' ``tools_disabled`` computation
    never checked ``request.tool_choice`` at all — only whether the model
    was a diffusion model lacking tool-calling support — so a client that
    set ``tool_choice="none"`` still got its tools (and any MCP-merged
    tools) exposed to the template on this endpoint, unlike the equivalent
    /v1/chat/completions call.
    """

    def test_tools_not_exposed_to_template_when_tool_choice_is_none(
        self, client_with_mcp, engine
    ):
        resp = client_with_mcp.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "What's the weather?",
                "tool_choice": "none",
                "tools": [{
                    "type": "function",
                    "name": "get_stock_price",
                    "description": "Get the current stock price",
                    "parameters": {
                        "type": "object",
                        "properties": {"ticker": {"type": "string"}},
                    },
                }],
            },
        )
        assert resp.status_code == 200
        assert engine.recorded_chat_kwargs, "engine.chat was never called"
        tools = engine.recorded_chat_kwargs[-1].get("tools")
        assert tools is None, (
            "tool_choice='none' must suppress tool exposure to the chat "
            "template, matching /v1/chat/completions — instead the client "
            f"tool and/or MCP-merged tools reached the engine: {tools!r}"
        )

    def test_mcp_tools_also_suppressed_when_tool_choice_is_none(
        self, client_with_mcp, engine
    ):
        """MCP-merged tools must be suppressed too, not just client tools —
        the merge itself must be skipped, same as create_chat_completion."""
        resp = client_with_mcp.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "What's the weather?",
                "tool_choice": "none",
            },
        )
        assert resp.status_code == 200
        assert engine.recorded_chat_kwargs, "engine.chat was never called"
        tools = engine.recorded_chat_kwargs[-1].get("tools")
        assert tools is None, (
            f"MCP tools leaked to the template despite tool_choice='none': {tools!r}"
        )
