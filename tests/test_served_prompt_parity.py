# SPDX-License-Identifier: Apache-2.0
"""Prompt parity between serving and the cache endpoints (#3615).

The prompt block cache is keyed on token ids, so the cache probe and the
cache-artifact export have to arrive at the *exact* token ids a real
``/v1/chat/completions`` turn prefilled. They used to render prompts
themselves, which dropped serving-only transformations (tool-result
truncation, MCP tool merge, thinking flags, JSON instructions, partial
mode). The result: an export answering "nothing for this prompt is durable
on disk" for a conversation that had just been served, while the same prompt
passed as ``token_ids`` exported fine.

These tests pin the shared replay: every transformation the serving path
applies must be visible in the prompt the cache endpoints hash.
"""

import asyncio
import json
import re
import zlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.admin.cache_artifacts as cache_artifacts
import omlx.admin.routes as admin_routes
import omlx.server as omlx_server
from omlx.engine.base import BaseEngine
from omlx.model_settings import ModelSettings

MODEL_ID = "parity-model"
MODEL_NAME = "/models/parity-model"
BLOCK_SIZE = 4


def _encode(text: str) -> list[int]:
    """Deterministic word -> token ids (stable across processes)."""
    return [zlib.crc32(word.encode()) % 100000 for word in re.findall(r"\S+", text)]


def _field(message, key, default=None):
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


class FakeTokenizer:
    """Tokenizer double: encodes text, and renders *naively*.

    ``apply_chat_template`` reproduces what the export/probe path used to do
    — join role and content, ignoring tools and template kwargs — so the
    tests can assert the served prompt is *not* that.
    """

    name = "fake"

    def __init__(self) -> None:
        self._words: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in re.findall(r"\S+", text):
            token_id = zlib.crc32(word.encode()) % 100000
            self._words.setdefault(token_id, word)
            ids.append(token_id)
        return ids

    def decode(self, token_ids) -> str:
        return " ".join(self._words.get(int(i), "?") for i in token_ids)

    def apply_chat_template(self, messages, **kwargs) -> str:
        return "".join(
            f"<|{_field(m, 'role', '')}|>{_field(m, 'content', '') or ''}"
            for m in messages
        )


class FakeEngine:
    """Engine double with the real render contract.

    The chat template renders everything that distinguishes one turn's prompt
    from another: message content, tool schemas, template kwargs, partial
    mode. If any of those is resolved differently by a caller, the rendered
    prompt (and therefore the block hash) differs.
    """

    tokenizer = FakeTokenizer()
    _tokenizer = tokenizer

    def _apply_chat_template(
        self, messages, tools=None, chat_template_kwargs=None, is_partial=None
    ) -> str:
        parts = []
        for message in messages:
            role = _field(message, "role", "")
            content = _field(message, "content", "") or ""
            chunk = f"<|{role}|>{content}"
            for call in _field(message, "tool_calls", None) or []:
                function = _field(call, "function", {}) or {}
                chunk += f"<|call|>{_field(function, 'name', '')}:{json.dumps(_field(function, 'arguments', ''), sort_keys=True, default=str)}"
            parts.append(chunk)
        if tools:
            parts.append(
                "<|tools|>"
                + ",".join(
                    sorted(
                        t.get("function", {}).get("name", "?")
                        for t in tools
                        if isinstance(t, dict)
                    )
                )
            )
        if chat_template_kwargs:
            parts.append(
                "<|ct|>"
                + json.dumps(dict(sorted(chat_template_kwargs.items())), default=str)
            )
        if is_partial:
            parts.append("<|partial|>")
        return "".join(parts)

    async def start(self) -> None:
        return None

    def count_chat_tokens(
        self, messages, tools=None, chat_template_kwargs=None, is_partial=None
    ) -> int:
        return len(
            self.tokenizer.encode(
                self.render_chat_prompt(
                    messages,
                    tools,
                    chat_template_kwargs=chat_template_kwargs,
                    is_partial=is_partial,
                )
            )
        )


def _make_engine() -> FakeEngine:
    engine = FakeEngine()
    # Use the production render helpers, not a second implementation.
    engine.prepare_chat_messages = BaseEngine.prepare_chat_messages.__get__(engine)
    engine.render_chat_prompt = BaseEngine.render_chat_prompt.__get__(engine)
    return engine


def _make_entry(engine) -> SimpleNamespace:
    scheduler = SimpleNamespace(
        block_aware_cache=SimpleNamespace(block_size=BLOCK_SIZE),
        paged_ssd_cache_manager=SimpleNamespace(
            _hot_cache={},
            _index=MagicMock(spec=[]),
            _cache_dir="/tmp/parity-cache",
        ),
        paged_cache_manager=SimpleNamespace(model_name=MODEL_NAME),
        config=SimpleNamespace(paged_cache_block_size=BLOCK_SIZE),
    )
    engine._tokenizer = engine.tokenizer
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
    return SimpleNamespace(
        engine=engine,
        preserve_thinking_default=None,
        config_model_type=None,
    )


def _pool(entry) -> MagicMock:
    pool = MagicMock(spec=[])
    pool._entries = {MODEL_ID: entry}
    return pool


def _settings(**overrides) -> ModelSettings:
    settings = ModelSettings()
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def _settings_manager(settings) -> MagicMock:
    manager = MagicMock(spec=[])
    manager.get_settings_for_request = MagicMock(return_value=settings)
    return manager


MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Read the config file."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"config.yaml"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "alpha beta gamma delta epsilon zeta eta theta iota kappa",
    },
    {"role": "user", "content": "Summarize it."},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]


class TestServedPromptTransformations:
    """Every serving-time transformation must be in the hashed prompt."""

    def test_tool_result_is_truncated_like_a_served_turn(self):
        """A long tool result is clipped by max_tool_result_tokens.

        The old export rendered the whole tool output, so every block after
        the tool message hashed differently.
        """
        engine = _make_engine()
        with patch.object(
            omlx_server._server_state,
            "settings_manager",
            _settings_manager(_settings(max_tool_result_tokens=4)),
        ):
            token_ids, prompt = asyncio.run(
                omlx_server.served_prompt_token_ids(
                    engine, MODEL_ID, messages=MESSAGES
                )
            )
        assert "epsilon" not in prompt, "tool result should be truncated"
        assert "alpha beta gamma delta" in prompt
        assert token_ids == engine.tokenizer.encode(prompt)

    def test_mcp_tools_join_the_rendered_tool_list(self):
        """MCP tools are merged before templating, so they are in the hash."""
        engine = _make_engine()
        mcp = MagicMock(spec=[])
        mcp.get_merged_tools = MagicMock(
            return_value=TOOLS
            + [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_search",
                        "description": "MCP tool",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        )
        with (
            patch.object(omlx_server._server_state, "mcp_manager", mcp),
            patch.object(
                omlx_server._server_state,
                "settings_manager",
                _settings_manager(None),
            ),
        ):
            _, prompt = asyncio.run(
                omlx_server.served_prompt_token_ids(
                    engine, MODEL_ID, messages=MESSAGES, tools=TOOLS
                )
            )
        assert "mcp_search" in prompt

    def test_reasoning_effort_reaches_the_template_kwargs(self):
        """reasoning_effort is part of the prompt hash, like in serving."""
        engine = _make_engine()
        with patch.object(
            omlx_server._server_state,
            "settings_manager",
            _settings_manager(None),
        ):
            _, prompt = asyncio.run(
                omlx_server.served_prompt_token_ids(
                    engine,
                    MODEL_ID,
                    messages=MESSAGES,
                    reasoning_effort="none",
                )
            )
        assert '"reasoning_effort": "none"' in prompt

    def test_model_settings_and_forced_keys_reach_the_prompt(self):
        """Model defaults are hashed; forced keys still override the caller.

        Precedence is the serving one (model defaults, then request kwargs,
        then forced keys) — a caller that re-implements the merge always gets
        a different order and therefore a different prompt.
        """
        engine = _make_engine()
        settings = _settings(
            chat_template_kwargs={
                "persisted_key": "persisted",
                "locked_key": "locked",
            },
            forced_ct_kwargs=["locked_key"],
        )
        with patch.object(
            omlx_server._server_state,
            "settings_manager",
            _settings_manager(settings),
        ):
            _, prompt = asyncio.run(
                omlx_server.served_prompt_token_ids(
                    engine,
                    MODEL_ID,
                    messages=MESSAGES,
                    chat_template_kwargs={
                        "persisted_key": "request",
                        "locked_key": "request",
                    },
                )
            )
        assert '"persisted_key": "request"' in prompt
        assert '"locked_key": "locked"' in prompt

    def test_model_thinking_toggle_reaches_the_template(self):
        """The original #3615 probe bug: a model-level toggle never rendered."""
        engine = _make_engine()
        with patch.object(
            omlx_server._server_state,
            "settings_manager",
            _settings_manager(_settings(enable_thinking=False)),
        ):
            _, prompt = asyncio.run(
                omlx_server.served_prompt_token_ids(engine, MODEL_ID, messages=MESSAGES)
            )
        assert '"enable_thinking": false' in prompt

    def test_json_mode_instruction_is_part_of_the_prompt(self):
        """json_object injects a system instruction; the hash must include it."""
        engine = _make_engine()
        with patch.object(
            omlx_server._server_state,
            "settings_manager",
            _settings_manager(None),
        ):
            _, prompt = asyncio.run(
                omlx_server.served_prompt_token_ids(
                    engine,
                    MODEL_ID,
                    messages=MESSAGES,
                    response_format={"type": "json_object"},
                )
            )
        assert "json" in prompt.lower()

    def test_partial_mode_reaches_the_template(self):
        """A trailing partial assistant turn changes the template render."""
        messages = MESSAGES + [
            {"role": "assistant", "content": "Partial answer", "partial": True}
        ]
        engine = _make_engine()
        with patch.object(
            omlx_server._server_state,
            "settings_manager",
            _settings_manager(None),
        ):
            _, prompt = asyncio.run(
                omlx_server.served_prompt_token_ids(
                    engine, MODEL_ID, messages=messages
                )
            )
        assert "<|partial|>" in prompt
        assert '"partial"' not in prompt, "partial key must not leak to template"

    def test_export_prompt_differs_from_the_old_naive_render(self):
        """The removed render ignored tools and kwargs — proof it was wrong."""
        engine = _make_engine()
        with patch.object(
            omlx_server._server_state,
            "settings_manager",
            _settings_manager(_settings(enable_thinking=False)),
        ):
            prompt, served_ids = asyncio.run(
                omlx_server.served_prompt_token_ids(
                    engine, MODEL_ID, messages=MESSAGES, tools=TOOLS
                )
            )
        naive = engine.tokenizer.apply_chat_template(MESSAGES)
        assert prompt != naive
        assert served_ids != engine.tokenizer.encode(naive)


class TestCacheEndpointParity:
    """The probe and export endpoints must hash the served token ids."""

    def _captured_export_ids(self, request_kwargs) -> list[int]:
        engine = _make_engine()
        entry = _make_entry(engine)
        captured = {}

        def fake_export(_manager, **kwargs):
            captured.update(kwargs)
            return {
                "artifact_path": "/tmp/parity.zip",
                "size_bytes": 1,
                "manifest": {"restorable_tokens": 1, "ram_only_blocks": 0},
            }

        with (
            patch.object(cache_artifacts, "_get_engine_pool", return_value=_pool(entry)),
            patch.object(cache_artifacts.artifact_store, "export_artifact", fake_export),
            patch.object(
                omlx_server._server_state,
                "settings_manager",
                _settings_manager(_settings(max_tool_result_tokens=8)),
            ),
        ):
            asyncio.run(
                cache_artifacts.export_cache_artifact(
                    cache_artifacts.CacheArtifactExportRequest(
                        model_id=MODEL_ID, messages=MESSAGES, **request_kwargs
                    ),
                    is_admin=True,
                )
            )
        return captured["token_ids"]

    def test_export_by_messages_matches_the_served_turn(self):
        """Messages export == the ids a served turn produced (#3615 repro)."""
        engine = _make_engine()
        request_kwargs = {"tools": TOOLS, "chat_template_kwargs": {"temperature": 0}}
        with patch.object(
            omlx_server._server_state,
            "settings_manager",
            _settings_manager(_settings(max_tool_result_tokens=8)),
        ):
            served_ids, _ = asyncio.run(
                omlx_server.served_prompt_token_ids(
                    engine,
                    MODEL_ID,
                    messages=MESSAGES,
                    **request_kwargs,
                )
            )
            export_ids = self._captured_export_ids(request_kwargs)
        assert export_ids == served_ids

    def test_probe_and_export_agree(self):
        """One conversation, two endpoints, one token id sequence."""
        engine = _make_engine()
        entry = _make_entry(engine)
        request_kwargs = {"tools": TOOLS, "thinking_budget": 64}
        with patch.object(
            omlx_server._server_state,
            "settings_manager",
            _settings_manager(_settings(max_tool_result_tokens=8)),
        ):
            with patch.object(
                admin_routes, "_get_engine_pool", return_value=_pool(entry)
            ):
                probe = asyncio.run(
                    admin_routes.probe_cache(
                        admin_routes.CacheProbeRequest(
                            model_id=MODEL_ID, messages=MESSAGES, **request_kwargs
                        ),
                        is_admin=True,
                    )
                )
            export_ids = self._captured_export_ids(request_kwargs)
        assert probe["total_tokens"] == len(export_ids)

    def test_multimodal_messages_are_rejected_with_a_hint(self):
        """Image parts hash through a different channel — say so, don't lie."""
        engine = _make_engine()
        entry = _make_entry(engine)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                ],
            }
        ]
        with (
            patch.object(cache_artifacts, "_get_engine_pool", return_value=_pool(entry)),
            pytest.raises(HTTPException) as excinfo,
        ):
            asyncio.run(
                cache_artifacts.export_cache_artifact(
                    cache_artifacts.CacheArtifactExportRequest(
                        model_id=MODEL_ID, messages=messages
                    ),
                    is_admin=True,
                )
            )
        assert "token_ids" in str(excinfo.value.detail)
