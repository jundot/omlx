# SPDX-License-Identifier: Apache-2.0
"""Tests for LLM prompt extension on the video path (omlx.api.prompt_extend)."""

import asyncio
import types

from omlx.api.prompt_extend import (
    _clean_rewrite,
    _system_prompt_for,
    extend_video_prompt,
)


class _FakeEngine:
    def __init__(self, text="", raises=None, delay=0.0):
        self._text = text
        self._raises = raises
        self._delay = delay
        self.calls = []

    async def chat(self, *, messages, max_tokens, temperature, **kwargs):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises:
            raise self._raises
        return types.SimpleNamespace(text=self._text)


class _FakePool:
    def __init__(self, engine=None, raises=None):
        self._engine = engine
        self._raises = raises
        self.requested = []

    async def get_engine(self, model_id, **kwargs):
        self.requested.append(model_id)
        if self._raises:
            raise self._raises
        return self._engine


def test_clean_rewrite_strips_think_and_quotes():
    raw = '<think>plan the motion</think>\n"A person slowly stretches."'
    assert _clean_rewrite(raw) == "A person slowly stretches."


def test_clean_rewrite_empty():
    assert _clean_rewrite("") == ""
    assert _clean_rewrite("<think>only reasoning</think>") == ""


def test_system_prompt_language_selection():
    # Chinese input -> Chinese system prompt (mentions 动作)
    assert "动作" in _system_prompt_for("让人物伸懒腰")
    # English input -> English system prompt (mentions ACTION)
    assert "ACTION" in _system_prompt_for("a person stretching")


def test_extend_disabled_when_no_model():
    out, ext = asyncio.run(
        extend_video_prompt("伸懒腰", model_id="", engine_pool=_FakePool())
    )
    assert out == "伸懒腰" and ext is False


def test_extend_skips_empty_prompt():
    pool = _FakePool(engine=_FakeEngine(text="x"))
    out, ext = asyncio.run(
        extend_video_prompt("   ", model_id="m", engine_pool=pool)
    )
    assert out == "   " and ext is False
    assert pool.requested == []  # never touched the pool


def test_extend_success():
    eng = _FakeEngine(text="A person slowly raises both arms overhead and arches their back, stretching lazily.")
    pool = _FakePool(engine=eng)
    out, ext = asyncio.run(
        extend_video_prompt("伸懒腰", model_id="small-llm", engine_pool=pool)
    )
    assert ext is True
    assert "arches their back" in out
    assert pool.requested == ["small-llm"]
    # system + user messages were passed
    assert eng.calls[0]["messages"][0]["role"] == "system"
    assert eng.calls[0]["messages"][1]["content"] == "伸懒腰"


def test_extend_empty_output_falls_back():
    pool = _FakePool(engine=_FakeEngine(text="  <think>hmm</think> "))
    out, ext = asyncio.run(
        extend_video_prompt("stretch", model_id="m", engine_pool=pool)
    )
    assert out == "stretch" and ext is False


def test_extend_engine_error_falls_back():
    pool = _FakePool(engine=_FakeEngine(raises=RuntimeError("boom")))
    out, ext = asyncio.run(
        extend_video_prompt("stretch", model_id="m", engine_pool=pool)
    )
    assert out == "stretch" and ext is False


def test_extend_get_engine_error_falls_back():
    pool = _FakePool(raises=KeyError("no such model"))
    out, ext = asyncio.run(
        extend_video_prompt("stretch", model_id="missing", engine_pool=pool)
    )
    assert out == "stretch" and ext is False


def test_extend_timeout_falls_back():
    pool = _FakePool(engine=_FakeEngine(text="late", delay=0.2))
    out, ext = asyncio.run(
        extend_video_prompt(
            "stretch", model_id="m", engine_pool=pool, timeout_s=0.01
        )
    )
    assert out == "stretch" and ext is False
