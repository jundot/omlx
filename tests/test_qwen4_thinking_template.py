"""Opt-in contract checks using an actual local Qwen4/Allosaurus template.

Only the tokenizer is loaded. No model weights, generation, or downloads are
needed. Set OMLX_TEST_QWEN4_TEMPLATE_MODEL to the local checkpoint directory.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from omlx.api.openai_models import Message
from omlx.api.thinking import ThinkingParser, extract_thinking, prompt_opens_thinking
from omlx.api.utils import (
    extract_multimodal_content,
    extract_text_content,
    uses_native_reasoning_content,
)
from omlx.model_settings import ModelSettings, merge_chat_template_kwargs


@pytest.fixture(scope="module")
def template_engine():
    model_path = os.environ.get("OMLX_TEST_QWEN4_TEMPLATE_MODEL")
    if not model_path:
        pytest.skip("Set OMLX_TEST_QWEN4_TEMPLATE_MODEL for local template checks")
    assert Path(model_path).is_dir(), "Local template checkpoint does not exist"

    from transformers import AutoTokenizer

    from omlx.engine.vlm import VLMBatchedEngine

    engine = VLMBatchedEngine(model_name=model_path)
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen4_exp")
    )
    engine._tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    return engine


MESSAGES = [
    {"role": "system", "content": "Answer accurately."},
    {"role": "user", "content": "Look up alpha."},
]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up a key",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
}]


@pytest.mark.parametrize("mode", [None, False, True], ids=["auto", "off", "on"])
@pytest.mark.parametrize("with_tools", [False, True])
def test_actual_template_thinking_and_response_channel_parity(
    template_engine, mode, with_tools
):
    kwargs = merge_chat_template_kwargs(ModelSettings(enable_thinking=mode))
    prompt = template_engine._apply_chat_template(
        [dict(message) for message in MESSAGES],
        tools=TOOLS if with_tools else None,
        chat_template_kwargs=kwargs,
    )
    opened, _ = prompt_opens_thinking(template_engine.tokenizer, prompt)
    assert opened is (mode is not False)
    assert "Answer accurately." in prompt
    assert "Look up alpha." in prompt
    assert ("lookup" in prompt) is with_tools

    # A prompt-opened block has no opening tag in decoded completion text.
    body = "Working it out</think>Result" if opened else "Result"
    expected = extract_thinking(body, start_in_thinking=opened)
    parser = ThinkingParser(start_in_thinking=opened)
    thinking, content = [], []
    for index in range(0, len(body), 2):
        thought, answer = parser.feed(body[index:index + 2])
        thinking.append(thought)
        content.append(answer)
    thought, answer = parser.finish()
    thinking.append(thought)
    content.append(answer)
    assert ("".join(thinking).strip(), "".join(content).strip()) == expected
    assert expected == (("Working it out", "Result") if opened else ("", "Result"))


@pytest.mark.parametrize(
    "settings, request_kwargs, expected",
    [
        (ModelSettings(enable_thinking=False), {"enable_thinking": True}, True),
        (
            ModelSettings(
                enable_thinking=False, forced_ct_kwargs=["enable_thinking"]
            ),
            {"enable_thinking": True},
            False,
        ),
        (
            ModelSettings(
                enable_thinking=False,
                thinking_budget_enabled=True,
                thinking_budget_tokens=128,
            ),
            None,
            False,
        ),
    ],
    ids=["request-over-model", "forced-admin", "explicit-off-over-budget"],
)
def test_actual_template_respects_existing_precedence(
    template_engine, settings, request_kwargs, expected
):
    prompt = template_engine._apply_chat_template(
        [dict(message) for message in MESSAGES],
        chat_template_kwargs=merge_chat_template_kwargs(settings, request_kwargs),
    )
    assert prompt_opens_thinking(template_engine.tokenizer, prompt)[0] is expected


@pytest.mark.parametrize(
    "extractor", [extract_text_content, extract_multimodal_content]
)
def test_actual_template_renders_one_history_reasoning_block(
    template_engine, extractor
):
    messages = [
        Message(role="user", content="What is two plus two?"),
        Message(role="assistant", content="Four.", reasoning_content="Two pairs."),
        Message(role="user", content="Thanks."),
    ]
    history = extractor(
        messages,
        native_reasoning_content=uses_native_reasoning_content(
            config_model_type="qwen4_exp"
        ),
    )
    prompt = template_engine._apply_chat_template(history)
    # One history block plus the new assistant generation prefix.
    assert prompt.count("<think>") == 2
    assert prompt.count("Two pairs.") == 1
    assert "</think>\n\n<think>" not in prompt
