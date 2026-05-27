# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from dflash_mlx.engine.events import SummaryEvent, TokenEvent


@dataclass
class _GenerationOutput:
    text: str
    tokens: List[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = "stop"
    new_text: str = ""
    finished: bool = True
    tool_calls: Optional[List[Dict[str, Any]]] = None
    cached_tokens: int = 0


class _BaseEngine:
    pass


class _FakeTokenizer:
    clean_up_tokenization_spaces = False

    def encode(self, prompt):
        return [1, 2, 3, 4]

    def decode(self, tokens, skip_special_tokens=True):
        return "decoded"

    def apply_chat_template(self, messages, **kwargs):
        return "templated prompt"


class _FakePrefixFlow:
    hit_tokens = 3


@pytest.fixture
def dflash_module(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    executor = ThreadPoolExecutor(max_workers=1)

    omlx = types.ModuleType("omlx")
    omlx.__path__ = [str(repo_root / "omlx")]
    monkeypatch.setitem(sys.modules, "omlx", omlx)

    engine_pkg = types.ModuleType("omlx.engine")
    engine_pkg.__path__ = [str(repo_root / "omlx" / "engine")]
    monkeypatch.setitem(sys.modules, "omlx.engine", engine_pkg)

    base = types.ModuleType("omlx.engine.base")
    base.GenerationOutput = _GenerationOutput
    base.BaseEngine = _BaseEngine
    monkeypatch.setitem(sys.modules, "omlx.engine.base", base)

    adapter_pkg = types.ModuleType("omlx.adapter")
    adapter = types.ModuleType("omlx.adapter.output_parser")
    adapter.detect_output_parser = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "omlx.adapter", adapter_pkg)
    monkeypatch.setitem(sys.modules, "omlx.adapter.output_parser", adapter)

    api_pkg = types.ModuleType("omlx.api")
    api_tool = types.ModuleType("omlx.api.tool_calling")
    api_tool.convert_tools_for_template = lambda tools: tools
    api_utils = types.ModuleType("omlx.api.utils")
    api_utils.clean_special_tokens = lambda text: text
    api_utils.detect_and_strip_partial = lambda messages: False
    monkeypatch.setitem(sys.modules, "omlx.api", api_pkg)
    monkeypatch.setitem(sys.modules, "omlx.api.tool_calling", api_tool)
    monkeypatch.setitem(sys.modules, "omlx.api.utils", api_utils)

    utils_pkg = types.ModuleType("omlx.utils")
    utils_model = types.ModuleType("omlx.utils.model_loading")
    utils_model.maybe_apply_pre_load_patches = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "omlx.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "omlx.utils.model_loading", utils_model)

    engine_core = types.ModuleType("omlx.engine_core")
    engine_core.get_mlx_executor = lambda: executor
    monkeypatch.setitem(sys.modules, "omlx.engine_core", engine_core)

    spec = importlib.util.spec_from_file_location(
        "omlx.engine.dflash",
        repo_root / "omlx" / "engine" / "dflash.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "omlx.engine.dflash", module)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        executor.shutdown(wait=True)


def _summary_event():
    return SummaryEvent(
        elapsed_us=1000,
        prompt_token_count=4,
        generated_token_ids=(10,),
        generation_tokens=1,
        accepted_from_draft=1,
        acceptance_ratio=1.0,
        cycles_completed=1,
        phase_timings_us={},
    )


def _make_engine(dflash_module):
    engine = dflash_module.DFlashEngine.__new__(dflash_module.DFlashEngine)
    engine._loaded = True
    engine._tokenizer_obj = _FakeTokenizer()
    engine._executor_tokenizer = _FakeTokenizer()
    engine._in_fallback_mode = False
    engine._fallback_engine = None
    engine._fallback_engine_type = "batched"
    engine._output_parser_factory = None
    engine._active_request = False
    engine._should_fallback = lambda prompt_tokens: False
    engine._detect_needs_think_prefix = lambda prompt_tokens: False
    engine._think_prefix_text = lambda: "<think>\n"

    def fake_stream(prompt_tokens, max_tokens, **kwargs):
        events = iter(
            [
                TokenEvent(
                    token_id=10,
                    generated_tokens=1,
                    acceptance_ratio=1.0,
                    cycles_completed=1,
                ),
                _summary_event(),
            ]
        )
        return events, _FakePrefixFlow(), {0}

    engine._stream_dflash_events = fake_stream
    return engine


def test_dflash_generate_surfaces_prefix_flow_hit_tokens(dflash_module):
    engine = _make_engine(dflash_module)

    output = asyncio.run(engine.generate("prompt", max_tokens=1))

    assert output.cached_tokens == 3


def test_dflash_stream_generate_surfaces_prefix_flow_hit_tokens(dflash_module):
    engine = _make_engine(dflash_module)

    async def collect_outputs():
        return [
            output async for output in engine.stream_generate("prompt", max_tokens=1)
        ]

    outputs = asyncio.run(collect_outputs())

    assert outputs[-1].finished is True
    assert outputs[-1].cached_tokens == 3


def test_dflash_generate_forwards_prefix_cache_metadata_to_flow(dflash_module):
    engine = _make_engine(dflash_module)
    captured = {}
    request = types.SimpleNamespace(request_type="chat", messages=[])
    chat_template_kwargs = {"enable_thinking": False}

    def fake_stream(prompt_tokens, max_tokens, **kwargs):
        captured.update(kwargs)
        events = iter(
            [
                TokenEvent(
                    token_id=10,
                    generated_tokens=1,
                    acceptance_ratio=1.0,
                    cycles_completed=1,
                ),
                _summary_event(),
            ]
        )
        return events, _FakePrefixFlow(), {0}

    engine._stream_dflash_events = fake_stream

    asyncio.run(
        engine.generate(
            "prompt",
            max_tokens=1,
            prefix_cache_request=request,
            prefix_cache_chat_template_kwargs=chat_template_kwargs,
        )
    )

    assert captured["prefix_cache_request"] is request
    assert captured["prefix_cache_chat_template_kwargs"] == chat_template_kwargs


def test_dflash_chat_forwards_prefix_cache_policy_metadata(dflash_module):
    engine = _make_engine(dflash_module)
    captured = {}

    async def fake_generate(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return _GenerationOutput(text="ok")

    engine.generate = fake_generate
    messages = [{"role": "user", "content": "hello"}]
    chat_template_kwargs = {"enable_thinking": False}

    asyncio.run(
        engine.chat(
            messages=messages,
            max_tokens=1,
            chat_template_kwargs=chat_template_kwargs,
            is_partial=False,
        )
    )

    assert captured["prompt"] == "templated prompt"
    request = captured["kwargs"]["prefix_cache_request"]
    assert request.request_type == "chat"
    assert request.messages == messages
    assert captured["kwargs"]["prefix_cache_chat_template_kwargs"] == chat_template_kwargs
