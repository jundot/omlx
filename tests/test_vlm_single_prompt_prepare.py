# SPDX-License-Identifier: Apache-2.0
"""A text-only VLM request is rendered and tokenized once, off the MLX executor."""

import copy
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from omlx.api.tool_calling import convert_tools_for_template
from omlx.engine import vlm as vlm_module
from omlx.engine.vlm import VLMBatchedEngine, _VLMPreparedPrompt

_TEMPLATE = (
    "{%- if tools %}<|tools|>{%- for t in tools %}{{ t.function.name }};"
    "{%- endfor %}{%- endif %}"
    "{%- for m in messages %}<|{{ m.role }}|>{{ m.content }}"
    "{%- if not (loop.last and continue_final_message is defined"
    " and continue_final_message) %}<|end|>{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}<|assistant|>"
    "{%- if enable_thinking is defined and enable_thinking %}<think>{%- endif %}"
    "{%- endif %}"
)
_WORDS = ["Hello", "world", "What", "is", "weather", "in", "Seoul", "Count", "1", "2", "3", "Earlier", "answer"]  # fmt: skip
_SPECIAL = [
    "<unk>", "<bos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    "<|tools|>", "<think>", "get_weather",
]  # fmt: skip


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {tok: i for i, tok in enumerate(_SPECIAL + _WORDS + [";"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    # Adds <bos> only when special tokens are requested, so a count made with the
    # default encode() differs from the ids generation uses.
    backend.post_processor = TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<unk>"
    )
    tokenizer.add_special_tokens({"additional_special_tokens": _SPECIAL[2:]})
    tokenizer.chat_template = _TEMPLATE
    return tokenizer


def _engine(model_type: str, tokenizer=None) -> VLMBatchedEngine:
    tokenizer = tokenizer or _tokenizer()
    engine = VLMBatchedEngine(model_name="test-vlm")
    engine._loaded = True
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type=model_type, image_token_id=1)
    )
    # The processor keeps the executor's tokenizer; the worker gets a copy.
    engine._processor = SimpleNamespace(tokenizer=tokenizer)
    engine._tokenizer = copy.deepcopy(tokenizer)
    engine._engine = MagicMock()
    return engine


_TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]
_TURNS = [
    {"role": "system", "content": "Hello"},
    {"role": "user", "content": "What is weather in Seoul"},
    {"role": "assistant", "content": "Earlier answer"},
    {"role": "user", "content": "Count 1 2 3"},
]
# (messages, tools, chat_template_kwargs, is_partial)
_CASES = {
    "plain": ([_TURNS[1]], None, None, None),
    "multi-turn": (_TURNS, None, None, None),
    "tools": (_TURNS, _TOOLS, None, None),
    "thinking": (_TURNS, None, {"enable_thinking": True}, None),
    "partial": (
        [_TURNS[1], {"role": "assistant", "content": "Count 1"}],
        None,
        None,
        True,
    ),
}


def _generation_ids(engine, messages, tools, kwargs, partial) -> list[int]:
    """What ``_process_chat_messages`` produced before: render + encode on the executor."""
    _, text_messages, *_ = engine._extract_request_media(copy.deepcopy(messages))
    return engine._prepare_vision_inputs(
        text_messages,
        [],
        chat_template_kwargs=kwargs,
        tools=convert_tools_for_template(tools) if tools else None,
        is_partial=partial,
    )[0]


@pytest.mark.parametrize("model_type", ["qwen2_5_vl", "gemma4"])
@pytest.mark.parametrize("case", list(_CASES))
def test_prepared_ids_are_the_generation_ids(model_type, case):
    messages, tools, kwargs, partial = copy.deepcopy(_CASES[case])
    engine = _engine(model_type)
    expected = _generation_ids(engine, messages, tools, kwargs, partial)

    prepared = engine.prepare_chat_prompt(messages, tools, kwargs, partial)

    assert list(prepared[1]) == expected
    # The old count included the post-processor's <bos>.
    assert len(engine._tokenizer.encode(prepared[0])) == len(expected) + 1


@pytest.mark.parametrize("case", list(_CASES))
def test_generation_reuses_ids_without_encoding_on_the_executor(case):
    messages, tools, kwargs, partial = copy.deepcopy(_CASES[case])
    engine = _engine("qwen2_5_vl")
    expected = _generation_ids(engine, messages, tools, kwargs, partial)
    prepared = engine.prepare_chat_prompt(messages, tools, kwargs, partial)
    engine._prepare_vision_inputs = MagicMock(side_effect=AssertionError("re-encoded"))

    chat_kwargs = {
        "chat_template_kwargs": kwargs,
        "is_partial": partial,
        "_prepared_prompt": prepared,
    }
    result = engine._process_chat_messages(messages, tools, chat_kwargs)

    assert result == (expected, None, None, None, 0, [])
    engine._prepare_vision_inputs.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda kw: kw.update(is_partial=True), id="is_partial"),
        pytest.param(lambda kw: kw.update(tools=_TOOLS), id="tools"),
        pytest.param(
            lambda kw: kw.update(messages=copy.deepcopy(_TURNS)), id="messages"
        ),
        pytest.param(
            lambda kw: kw.update(chat_template_kwargs={"enable_thinking": True}),
            id="chat_template_kwargs",
        ),
    ],
)
def test_prepared_prompt_of_another_request_is_not_reused(change):
    engine = _engine("qwen2_5_vl")
    messages = copy.deepcopy(_TURNS)
    prepared = engine.prepare_chat_prompt(messages)
    request = {
        "messages": messages,
        "tools": None,
        "chat_template_kwargs": None,
        "is_partial": None,
    }
    change(request)
    engine._prepare_vision_inputs = MagicMock(
        return_value=([9], None, None, None, 0, [])
    )

    engine._process_chat_messages(
        request.pop("messages"),
        request.pop("tools"),
        {**request, "_prepared_prompt": prepared},
    )

    engine._prepare_vision_inputs.assert_called_once()


def test_equal_template_kwargs_in_a_new_dict_are_reused():
    # /v1/decisions prepares with a module constant and chats with a copy of it.
    engine = _engine("qwen2_5_vl")
    messages = copy.deepcopy(_TURNS)
    kwargs = {"enable_thinking": False}
    prepared = engine.prepare_chat_prompt(messages, None, kwargs, True)
    engine._prepare_vision_inputs = MagicMock(side_effect=AssertionError("re-encoded"))

    engine._process_chat_messages(
        messages,
        None,
        {
            "chat_template_kwargs": dict(kwargs),
            "is_partial": True,
            "_prepared_prompt": prepared,
        },
    )

    engine._prepare_vision_inputs.assert_not_called()


def test_template_kwargs_changed_in_place_are_not_reused():
    engine = _engine("qwen2_5_vl")
    messages = copy.deepcopy(_TURNS)
    kwargs = {"enable_thinking": False}
    prepared = engine.prepare_chat_prompt(messages, None, kwargs)
    kwargs["enable_thinking"] = True
    engine._prepare_vision_inputs = MagicMock(
        return_value=([9], None, None, None, 0, [])
    )

    engine._process_chat_messages(
        messages, None, {"chat_template_kwargs": kwargs, "_prepared_prompt": prepared}
    )

    engine._prepare_vision_inputs.assert_called_once()


def test_executor_gets_its_own_copy_of_the_prepared_ids():
    engine = _engine("qwen2_5_vl")
    messages = copy.deepcopy(_TURNS)
    prepared = engine.prepare_chat_prompt(messages)
    before = list(prepared[1])

    ids = engine._process_chat_messages(messages, None, {"_prepared_prompt": prepared})[0]
    ids.append(0)

    assert list(prepared[1]) == before


def test_untagged_prepared_prompt_is_not_reused():
    engine = _engine("qwen2_5_vl")
    engine._prepare_vision_inputs = MagicMock(
        return_value=([9], None, None, None, 0, [])
    )

    ids = engine._process_chat_messages(
        _TURNS, None, {"_prepared_prompt": ("prompt", [1, 2, 3])}
    )[0]

    assert ids == [9]


def test_diffusion_model_keeps_the_legacy_preparation(monkeypatch):
    engine = _engine("qwen2_5_vl")
    monkeypatch.setattr(
        VLMBatchedEngine, "is_diffusion_model", property(lambda s: True)
    )

    prepared = engine.prepare_chat_prompt(_TURNS)

    assert not isinstance(prepared, _VLMPreparedPrompt)
    assert list(prepared[1]) == engine._tokenizer.encode(prepared[0])


_IMAGE = {
    "type": "image_url",
    "image_url": {
        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        "nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
    },
}


_AUDIO = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
_VIDEO = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}


@pytest.mark.parametrize(
    "media", [_IMAGE, _AUDIO, _VIDEO], ids=["image", "audio", "video"]
)
def test_media_request_keeps_the_legacy_preparation(media, monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    monkeypatch.setattr(vlm_module, "write_video_data_uri", lambda uri: (clip, "hash"))
    monkeypatch.setattr(vlm_module, "probe_video", lambda path: None)
    monkeypatch.setattr(vlm_module, "native_video_token_count", lambda *a: 1)
    engine = _engine("qwen2_5_vl")
    engine._native_video = True  # accept the video part instead of rejecting it
    messages = [{"role": "user", "content": [media, {"type": "text", "text": "Hello"}]}]

    prepared = engine.prepare_chat_prompt(messages)
    assert not isinstance(prepared, _VLMPreparedPrompt)
    assert list(prepared[1]) == engine._tokenizer.encode(prepared[0])

    engine._prepare_vision_inputs = MagicMock(
        return_value=([9], "embeds", {}, "hash", 0, [])
    )
    engine._process_chat_messages(
        messages,
        None,
        {
            "_prepared_prompt": _VLMPreparedPrompt(
                "p", [1], (messages, None, None, None)
            )
        },
    )
    engine._prepare_vision_inputs.assert_called_once()


def test_processor_with_its_own_template_is_the_render_target():
    tokenizer = _tokenizer()
    engine = _engine("qwen2_5_vl", tokenizer)
    engine._processor = SimpleNamespace(
        tokenizer=tokenizer, apply_chat_template=tokenizer.apply_chat_template
    )
    for messages, tools, kwargs, partial in (_CASES["tools"], _CASES["thinking"]):
        expected = _generation_ids(engine, messages, tools, kwargs, partial)
        prepared = engine.prepare_chat_prompt(messages, tools, kwargs, partial)
        assert list(prepared[1]) == expected


def test_trailing_partial_flag_is_detected_the_same_way_in_both_paths():
    engine = _engine("qwen2_5_vl")
    messages = [_TURNS[1], {"role": "assistant", "content": "Count 1", "partial": True}]
    expected = _generation_ids(engine, messages, None, None, None)
    before = copy.deepcopy(messages)
    prepared = engine.prepare_chat_prompt(messages)
    assert list(prepared[1]) == expected
    # The render strips the flag from its own copy, so the second pass still sees it.
    assert messages == before
    engine._prepare_vision_inputs = MagicMock(side_effect=AssertionError("re-encoded"))
    ids = engine._process_chat_messages(messages, None, {"_prepared_prompt": prepared})[
        0
    ]
    assert ids == expected
    assert messages == before
    assert "<|assistant|>Count 1<|end|>" not in prepared[0]  # continued, not closed
    assert not prepared[0].endswith("<|assistant|>")


def test_text_turns_stay_a_prefix_of_a_later_image_turn():
    engine = _engine("qwen2_5_vl")
    first = [_TURNS[0], _TURNS[1]]
    later = [
        *first,
        _TURNS[2],
        {"role": "user", "content": [_IMAGE, {"type": "text", "text": "Hello"}]},
    ]

    text_prompt = engine.prepare_chat_prompt(first)[0]
    media_messages, _, images, *_ = engine._extract_request_media(later)
    media_prompt = engine._render_vlm_prompt(
        media_messages,
        model_type="qwen2_5_vl",
        num_images=len(images),
        num_audios=0,
        num_videos=0,
        chat_template_kwargs=None,
        tools=None,
        is_partial=None,
    )[0]

    shared = text_prompt.removesuffix("<|assistant|>")
    assert shared and media_prompt.startswith(shared)


@pytest.mark.skipif(
    not os.environ.get("OMLX_TEST_GLM_TOKENIZER_DIR"),
    reason="OMLX_TEST_GLM_TOKENIZER_DIR is not set",
)
@pytest.mark.parametrize("case", list(_CASES))
def test_glm_ids_are_the_generation_ids(case):
    """Needs tokenizer.json, tokenizer_config.json and chat_template.jinja of GLM-5.3."""
    tokenizer = AutoTokenizer.from_pretrained(os.environ["OMLX_TEST_GLM_TOKENIZER_DIR"])
    engine = _engine("glm5_next", tokenizer)
    messages, tools, kwargs, partial = copy.deepcopy(_CASES[case])
    expected = _generation_ids(engine, messages, tools, kwargs, partial)

    prepared = engine.prepare_chat_prompt(messages, tools, kwargs, partial)

    assert list(prepared[1]) == expected
