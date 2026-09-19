# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the SpecPrefill static-prefix boundary.

The boundary is consumed as a prefix length (``prompt_token_ids[:system_end]``),
so what matters is a property, not a token count: the protected prefix must
cover every system/developer and tool-instruction token, and must not reach
into conversation content. These tests assert that property against a
synthetic chat template that reproduces the behaviour which broke the previous
subtraction-based derivation — injecting a default system block whenever no
system message is supplied.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.specprefill.boundary import (
    common_prefix_length,
    resolve_static_prefix_end,
)

STATIC_ROLES = ("system", "developer")

# Text the template invents when the caller supplies no system message. Its
# presence is what makes "full render minus non-system render" unsound.
DEFAULT_SYSTEM_TEXT = "You are a helpful assistant that follows instructions."

_TOKEN_RE = re.compile(r"\s+|\w+|[^\s\w]")


class FakeTokenizer:
    """Deterministic word-level tokenizer plus a small chat template.

    The template mirrors the structural properties that matter here: tool
    instructions are emitted *before* the caller's system text inside the same
    leading block, and a default system block appears when the caller supplies
    none.
    """

    def __init__(self, inject_default_system: bool = True):
        self.inject_default_system = inject_default_system
        self._vocab: dict[str, int] = {}

    def _token_id(self, piece: str) -> int:
        return self._vocab.setdefault(piece, len(self._vocab) + 1000)

    def encode(self, text, **kwargs):
        return [self._token_id(p) for p in _TOKEN_RE.findall(text)]

    def render_static_prefix(self, messages, tools) -> str:
        """The exact static text this template puts ahead of the conversation."""
        static = [m for m in messages if m.get("role") in STATIC_ROLES]
        out = ["<block>"]
        if tools:
            out.append("<tools>")
            for tool in tools:
                out.append(f"use {tool['function']['name']} carefully;")
            out.append("</tools>")
            out.append("If no tool applies, answer normally and say nothing.")
        if static:
            for m in static:
                out.append(m["content"])
        elif self.inject_default_system:
            out.append(DEFAULT_SYSTEM_TEXT)
        out.append("</block>")
        return " ".join(out)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, tools=None, **kw
    ):
        parts = [self.render_static_prefix(messages, tools)]
        for m in messages:
            if m.get("role") in STATIC_ROLES:
                continue
            parts.append(f"<{m['role']}> {m['content']} </{m['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        text = " ".join(parts)
        return self.encode(text) if tokenize else text


def _tools(n):
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"Tool number {i}.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                },
            },
        }
        for i in range(n)
    ]


def _render_tokens_for(tok, tools):
    def render(messages):
        return tok.encode(
            tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, tools=tools
            )
        )

    return render


def _boundary(tok, messages, tools):
    full = tok.encode(
        tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )
    )
    end = resolve_static_prefix_end(messages, full, _render_tokens_for(tok, tools))
    return end, full


def _assert_protects_static_material(tok, messages, tools):
    """The property under test, expressed without any fixed token count."""
    end, full = _boundary(tok, messages, tools)

    # 1. The protected prefix covers every token of the template's static
    #    prefix — system/developer content and tool instructions alike.
    static_len = len(tok.encode(tok.render_static_prefix(messages, tools)))
    assert end >= static_len, (
        f"under-protected: boundary {end} < static prefix {static_len}"
    )

    # 2. It never reaches into conversation content.
    conversation = [m for m in messages if m.get("role") not in STATIC_ROLES]
    first_content = conversation[0]["content"]
    content_start = full.index(tok.encode(first_content)[0])
    assert end <= content_start, (
        f"over-protected: boundary {end} > conversation start {content_start}"
    )
    return end


USER = {"role": "user", "content": "please refactor the widget loader"}
SYS = {"role": "system", "content": "Operator policy: never modify production."}
DEV = {"role": "developer", "content": "Follow the repository conventions."}


@pytest.mark.parametrize("n_tools", [0, 1, 2, 8])
@pytest.mark.parametrize(
    "static",
    [
        pytest.param([SYS], id="system-only"),
        pytest.param([DEV], id="developer-only"),
        pytest.param([SYS, DEV], id="system+developer"),
    ],
)
def test_boundary_protects_all_static_material(static, n_tools):
    tok = FakeTokenizer()
    _assert_protects_static_material(tok, [*static, USER], _tools(n_tools) or None)


@pytest.mark.parametrize("n_tools", [0, 1, 8])
def test_default_system_injection_does_not_shrink_the_boundary(n_tools):
    """The Qwen-style case that made the old subtraction under-report.

    A template that invents a system block for the non-system re-render makes
    ``full - non_system`` smaller than the true boundary. Measuring the prefix
    directly has to be immune to it.
    """
    tools = _tools(n_tools) or None
    messages = [SYS, USER]

    injecting = FakeTokenizer(inject_default_system=True)
    end_injecting = _assert_protects_static_material(injecting, messages, tools)

    quiet = FakeTokenizer(inject_default_system=False)
    end_quiet = _assert_protects_static_material(quiet, messages, tools)

    # The injected default only appears when no system message is supplied, so
    # the real prompt — and therefore the boundary — is identical either way.
    assert end_injecting == end_quiet

    # And the old algorithm really would have been wrong here, so the test is
    # guarding something.
    non_system = [m for m in messages if m.get("role") not in STATIC_ROLES]
    subtraction = len(
        injecting.encode(injecting.apply_chat_template(messages, tools=tools))
    ) - len(injecting.encode(injecting.apply_chat_template(non_system, tools=tools)))
    assert subtraction < end_injecting


def test_no_static_messages_yields_no_boundary():
    tok = FakeTokenizer()
    end, _ = _boundary(tok, [USER], None)
    assert end == 0


def test_static_messages_only_yields_no_boundary():
    """Nothing to sparsify means nothing to protect."""
    tok = FakeTokenizer()
    end, _ = _boundary(tok, [SYS], None)
    assert end == 0


def test_boundary_survives_a_render_that_raises():
    tok = FakeTokenizer()
    full = tok.encode(tok.apply_chat_template([SYS, USER], tools=None))

    def exploding(_messages):
        raise RuntimeError("template refused this role combination")

    with pytest.raises(RuntimeError):
        resolve_static_prefix_end([SYS, USER], full, exploding)


def test_common_prefix_length_basics():
    assert common_prefix_length([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_length([], [1]) == 0
    assert common_prefix_length([1, 2], [1, 2]) == 2
    assert common_prefix_length([1, 2], [1, 2, 3]) == 2


# --------------------------------------------------------------------------
# Engine-level coverage: both engines, both streaming and non-streaming.
# --------------------------------------------------------------------------


def _fake_output():
    """Loose stand-in: only the counters the engine arithmetic touches matter."""
    output = MagicMock()
    output.output_text = "ok"
    output.prompt_tokens = 1
    output.completion_tokens = 1
    output.cached_tokens = 0
    output.finish_reason = "stop"
    output.tool_calls = None
    return output


def _expected_engine_boundary(tok, messages, tools):
    full = tok.encode(
        tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools
        )
    )
    return resolve_static_prefix_end(messages, full, _render_tokens_for(tok, tools))


@pytest.mark.asyncio
async def test_vlm_engine_injects_measured_boundary():
    from omlx.engine.vlm import VLMBatchedEngine

    tok = FakeTokenizer()
    messages = [SYS, USER]
    prompt_ids = tok.encode(tok.apply_chat_template(messages, tools=None))

    engine = VLMBatchedEngine(model_name="test-vlm")
    engine._loaded = True
    engine._model_settings = SimpleNamespace(specprefill_enabled=True)
    engine._tokenizer = tok
    engine._engine = MagicMock()
    engine._engine._mlx_executor = ThreadPoolExecutor(max_workers=1)
    engine._engine.generate = AsyncMock(return_value=_fake_output())

    def _mock_process(messages_, tools_, kwargs_):
        return prompt_ids, None, None, None, 0, []

    try:
        with patch.object(engine, "_process_chat_messages", side_effect=_mock_process):
            await engine.chat(messages)
    finally:
        engine._engine._mlx_executor.shutdown(wait=False)

    got = engine._engine.generate.call_args.kwargs["specprefill_system_end"]
    assert got == _expected_engine_boundary(tok, messages, None)
    # The protected prefix really does cover the operator's system text.
    assert got >= len(tok.encode(tok.render_static_prefix(messages, None)))


@pytest.mark.asyncio
async def test_batched_engine_injects_measured_boundary():
    from omlx.engine.batched import BatchedEngine

    tok = FakeTokenizer()
    messages = [SYS, USER]

    engine = BatchedEngine(model_name="test-model")
    engine._loaded = True
    engine._model_settings = SimpleNamespace(specprefill_enabled=True)
    engine._tokenizer = tok
    engine._engine = MagicMock()
    engine._engine._mlx_executor = ThreadPoolExecutor(max_workers=1)
    engine._engine.generate = AsyncMock(return_value=_fake_output())

    kwargs: dict = {}
    engine._inject_specprefill_system_end(
        messages,
        tok.apply_chat_template(messages, tools=None),
        None,
        None,
        kwargs,
    )
    engine._engine._mlx_executor.shutdown(wait=False)

    got = kwargs["specprefill_system_end"]
    assert got == _expected_engine_boundary(tok, messages, None)
    assert got >= len(tok.encode(tok.render_static_prefix(messages, None)))


@pytest.mark.parametrize("n_tools", [0, 2])
def test_both_engines_agree_on_the_boundary(n_tools):
    """Streaming and non-streaming share one helper, so the engines must too."""
    from omlx.engine.batched import BatchedEngine
    from omlx.engine.vlm import VLMBatchedEngine

    tok = FakeTokenizer()
    tools = _tools(n_tools) or None
    messages = [SYS, DEV, USER]
    prompt_text = tok.apply_chat_template(messages, tools=tools)
    prompt_ids = tok.encode(prompt_text)

    batched = BatchedEngine(model_name="m")
    batched._model_settings = SimpleNamespace(specprefill_enabled=True)
    batched._tokenizer = tok
    batched._enable_thinking = None
    batched_kwargs: dict = {}
    batched._inject_specprefill_system_end(
        messages, prompt_text, tools, None, batched_kwargs
    )

    vlm = VLMBatchedEngine(model_name="m")
    vlm._model_settings = SimpleNamespace(specprefill_enabled=True)
    vlm._tokenizer = tok
    vlm_kwargs: dict = {}
    vlm._inject_specprefill_system_end(messages, prompt_ids, tools, None, vlm_kwargs)

    assert batched_kwargs.get("specprefill_system_end") == vlm_kwargs.get(
        "specprefill_system_end"
    )


def test_conversation_starting_with_a_non_user_turn_stays_safe():
    """The probes use a user turn; a prefill-style prompt may not.

    When the real prompt's first conversation turn has another role the probes
    stop matching at the turn header. That must still cover the whole static
    block, and must still not reach conversation content.
    """
    tok = FakeTokenizer()
    messages = [SYS, {"role": "assistant", "content": "partial draft answer"}]
    _assert_protects_static_material(tok, messages, _tools(2))


def test_mid_conversation_system_message_does_not_over_protect():
    """A system message after the first turn is not part of the static prefix.

    The probes render every system message contiguously at the front, which the
    real prompt does not, so the two diverge early. Under-protecting is the
    acceptable outcome here; over-protecting into conversation content is not.
    """
    tok = FakeTokenizer()
    messages = [
        SYS,
        USER,
        {"role": "system", "content": "Mid-conversation policy update."},
        {"role": "user", "content": "and now regenerate the loader"},
    ]
    end, full = _boundary(tok, messages, None)
    content_start = full.index(tok.encode(USER["content"])[0])
    assert end <= content_start
