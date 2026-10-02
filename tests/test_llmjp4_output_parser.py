# SPDX-License-Identifier: Apache-2.0
"""Tests for the LLM-jp-4 Harmony output parser."""

from __future__ import annotations

import re

import pytest

from omlx.adapter.harmony import load_harmony_gpt_oss_encoding
from omlx.adapter.llmjp4 import Llmjp4OutputParserSession, is_llmjp4_tokenizer
from omlx.adapter.output_parser import detect_output_parser

# Harmony marker ids of the llm-jp-4 vocabulary (tokenizer.json added_tokens).
_MARKERS = {
    "<|return|>": 2,
    "<|endoftext|>": 4,
    "<|constrain|>": 8,
    "<|channel|>": 9,
    "<|start|>": 10,
    "<|end|>": 11,
    "<|message|>": 12,
    "<|call|>": 13,
}
_EOS_IDS = {2, 13}  # generation_config eos_token_id (with <|endoftext|>)
_PIECE_RE = re.compile(
    "|".join(re.escape(marker) for marker in _MARKERS) + r"|\n|[^\s<]+| +|<"
)


class Llmjp4Tokenizer:
    """Fake tokenizer with the llm-jp-4 marker ids and decode behavior.

    Text pieces use SentencePiece's U+2581 for spaces. Like the real
    ``Llmjp4Tokenizer``, decode drops the leading space of the decoded string
    and of each run after a Harmony marker, so decoding token by token
    loses spaces that decoding the whole sequence keeps.
    """

    def __init__(self) -> None:
        self._vocab = dict(_MARKERS)
        self._pieces = {token_id: piece for piece, token_id in _MARKERS.items()}

    def _piece_id(self, piece: str) -> int:
        if piece not in self._vocab:
            token_id = 100 + len(self._vocab)
            self._vocab[piece] = token_id
            self._pieces[token_id] = piece
        return self._vocab[piece]

    def encode(self, text: str) -> list[int]:
        """Split text into marker, word, space-run, and newline tokens.

        A space run before a word becomes ``"\u2581" * (n - 1)`` plus a
        ``"\u2581word"`` piece, mirroring how the real tokenizer attaches one
        space to the following word.
        """
        pieces: list[str] = []
        pending_spaces = 0
        for piece in _PIECE_RE.findall(text):
            if piece.strip(" ") == "":
                pending_spaces += len(piece)
                continue
            if pending_spaces:
                if pending_spaces > 1:
                    pieces.append("\u2581" * (pending_spaces - 1))
                if piece in _MARKERS or piece == "\n":
                    pieces.append("\u2581")
                    pending_spaces = 0
                else:
                    piece = "\u2581" + piece
                    pending_spaces = 0
            pieces.append(piece)
        return [self._piece_id(piece) for piece in pieces]

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._vocab.get(token)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        parts = []
        run = ""
        for token_id in token_ids:
            piece = self._pieces[token_id]
            if piece in _MARKERS:
                parts.append(run.replace("\u2581", " ").removeprefix(" "))
                run = ""
                if not skip_special_tokens:
                    parts.append(piece)
            else:
                run += piece
        parts.append(run.replace("\u2581", " ").removeprefix(" "))
        return "".join(parts)


def _run(session, token_ids):
    stream, stopped = [], False
    for token_id in token_ids:
        result = session.process_token(token_id)
        stream.append(result.stream_text)
        assert result.visible_text == result.stream_text
        if result.is_stop:
            stopped = True
            break
    final = session.finalize()
    stream.append(final.stream_text)
    return "".join(stream), stopped, final


@pytest.fixture
def tokenizer():
    return Llmjp4Tokenizer()


@pytest.fixture
def factory(tokenizer):
    factory = detect_output_parser(
        "ELYZA-Thinking-1.0-llm-jp-4-32b-a3b-mlx-4bit",
        tokenizer,
        {"model_type": "qwen3_moe"},
    )
    assert factory is not None
    assert factory.kind == "llmjp4"
    return factory


@pytest.fixture
def generate(tokenizer, factory):
    """Feed model output as the scheduler does: EOS tokens never reach it."""

    def run(text: str, *, keep_eos: bool = False):
        tokens = tokenizer.encode(text)
        if not keep_eos:
            tokens = [token for token in tokens if token not in _EOS_IDS]
        return _run(factory.create_session(tokenizer), tokens)

    return run


class _HarmonyTokenizer:
    def __init__(self, encoding):
        self._encoding = encoding

    def convert_tokens_to_ids(self, token: str) -> int:
        ids = self._encoding.encode(token, allowed_special="all")
        return ids[0] if ids else -1

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return self._encoding.decode(token_ids)


def test_fake_tokenizer_round_trips(tokenizer):
    text = "<|channel|>final<|message|>def f():\n    return 1<|return|>"
    assert tokenizer.decode(tokenizer.encode(text), skip_special_tokens=False) == text


def test_detection_requires_llmjp4_marker_ids(tokenizer):
    assert is_llmjp4_tokenizer(tokenizer)

    gpt_oss = detect_output_parser(
        "gpt-oss-20b",
        _HarmonyTokenizer(load_harmony_gpt_oss_encoding()),
        {"model_type": "gpt_oss"},
    )
    assert gpt_oss is not None
    assert gpt_oss.kind == "harmony"

    class PlainTokenizer:
        def convert_tokens_to_ids(self, token):
            return None

    assert not is_llmjp4_tokenizer(PlainTokenizer())


def test_factory_exposes_harmony_stops_and_budget_markers(factory):
    assert factory.stop_token_ids == {2, 13}
    assert factory.thinking_end_text == "<|end|>"
    assert (
        factory.thinking_end_trailing_text
        == "<|start|>assistant<|channel|>final<|message|>"
    )


def test_reasoning_then_answer(generate):
    stream, stopped, final = generate(
        "<|channel|>analysis<|message|>The user asks.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>東京です。<|return|>"
    )

    assert stream == "<think>The user asks.</think>東京です。"
    assert not stopped  # <|return|> is EOS; the scheduler ends the request.
    assert final.tool_calls == []
    assert final.finish_reason is None


def test_tool_call_without_call_marker(generate):
    stream, _, final = generate(
        "<|channel|>analysis<|message|>Need weather.<|end|>"
        "<|start|>assistant to=functions.get_weather<|channel|>commentary json"
        '<|message|>{"city": "東京"}<|call|>'
    )

    assert stream == "<think>Need weather.</think>"
    assert final.tool_calls == [
        {"name": "get_weather", "arguments": '{"city": "東京"}'}
    ]
    assert final.finish_reason == "tool_calls"


def test_tool_call_marker_stops_the_turn(generate):
    stream, stopped, final = generate(
        "assistant to=functions.lookup<|channel|>commentary<|constrain|>json"
        '<|message|>{"q": 1}<|call|>'
        "<|start|>assistant<|channel|>final<|message|>late",
        keep_eos=True,
    )

    assert stream == ""
    assert stopped
    assert final.tool_calls == [{"name": "lookup", "arguments": '{"q": 1}'}]


def test_tool_message_closed_by_end_is_not_a_call(generate):
    stream, _, final = generate(
        'to=functions.lookup<|channel|>commentary<|message|>{"q": 1}<|end|>'
        "<|start|>assistant<|channel|>final<|message|>Done."
    )

    assert stream == "Done."
    assert final.tool_calls == []


_PARALLEL_CALLS = (
    "<|channel|>analysis<|message|>Need both.<|end|>"
    "<|start|>assistant to=functions.get_weather<|channel|>commentary "
    '<|constrain|> json<|message|>{"city": "東京"}<|end|>'
    "<|start|>assistant to=functions.get_time<|channel|>commentary "
    '<|constrain|> json<|message|>{"tz": "Asia/Tokyo"}<|call|>'
)


@pytest.mark.parametrize("keep_eos", [False, True])
def test_parallel_tool_calls(generate, keep_eos):
    """LLM-jp-4.1 closes every parallel call but the last with <|end|>."""
    stream, _, final = generate(_PARALLEL_CALLS, keep_eos=keep_eos)

    assert stream == "<think>Need both.</think>"
    assert final.tool_calls == [
        {"name": "get_weather", "arguments": '{"city": "東京"}'},
        {"name": "get_time", "arguments": '{"tz": "Asia/Tokyo"}'},
    ]
    assert final.finish_reason == "tool_calls"


def test_parallel_messages_before_final_answer_are_not_calls(generate):
    stream, _, final = generate(
        _PARALLEL_CALLS.replace("<|call|>", "<|end|>")
        + "<|start|>assistant<|channel|>final<|message|>Done."
    )

    assert stream == "<think>Need both.</think>Done."
    assert final.tool_calls == []


def test_parallel_calls_cut_off_by_max_tokens_are_not_calls(generate):
    stream, _, final = generate(_PARALLEL_CALLS.split('"Asia')[0])

    assert stream == "<think>Need both.</think>"
    assert final.tool_calls == []
    assert final.finish_reason is None


def test_reasoning_between_tool_messages_splits_the_run(generate):
    _, _, final = generate(
        'to=functions.lookup<|channel|>commentary<|message|>{"q": 1}<|end|>'
        "<|start|>assistant<|channel|>analysis<|message|>Retry.<|end|>"
        '<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{"q": 2}'
    )

    assert final.tool_calls == [{"name": "lookup", "arguments": '{"q": 2}'}]


def test_truncated_arguments_are_not_a_call(generate):
    stream, _, final = generate(
        'to=functions.lookup<|channel|>commentary<|message|>{"q": "par'
    )

    assert stream == ""
    assert final.tool_calls == []
    assert final.finish_reason is None


def test_non_object_arguments_are_dropped(generate):
    _, stopped, final = generate(
        "to=functions.lookup<|channel|>commentary<|message|>[1, 2]<|call|>",
        keep_eos=True,
    )

    assert stopped
    assert final.tool_calls == []


def test_tool_syntax_quoted_in_reasoning_is_not_a_call(generate):
    stream, _, final = generate(
        "<|channel|>analysis<|message|>I could write to=functions.fake {}<|end|>"
        "<|start|>assistant<|channel|>final<|message|>No."
    )

    assert stream == "<think>I could write to=functions.fake {}</think>No."
    assert final.tool_calls == []


def test_final_answer_closed_by_end_drops_imagined_turn(generate):
    stream, stopped, final = generate(
        "<|channel|>final<|message|>2です。<|end|>"
        "<|start|>assistant to=functions.get_weather<|channel|>commentary"
        '<|message|>{"city": "x"}'
    )

    assert stream == "2です。"
    assert stopped
    assert final.tool_calls == []


def test_other_role_header_stops_generation(generate):
    stream, stopped, final = generate(
        "<|channel|>analysis<|message|>hmm<|end|><|start|>user<|message|>next"
    )

    assert stream == "<think>hmm</think>"
    assert stopped
    assert final.tool_calls == []


@pytest.mark.parametrize(
    "tail",
    [
        "user",
        "system ",
        "us",
        "functions.get_weather",
        "functi",
        "assistant analysis",
        "assistant anal",
        "assistant to",
        "assistant to=functions.get_weather",
        "assistant<|channel|>comm",
    ],
)
def test_header_cut_off_by_eos_is_not_visible(generate, tail):
    stream, _, final = generate(
        f"<|channel|>analysis<|message|>hmm<|end|><|start|>{tail}<|return|>"
    )

    assert stream == "<think>hmm</think>"
    assert final.tool_calls == []


@pytest.mark.parametrize("tail", ["assistant anal", "analysis", "user"])
def test_header_without_start_cut_off_by_eos_is_not_visible(generate, tail):
    """The model sometimes omits <|start|> after an end marker."""
    stream, _, _ = generate(f"<|channel|>analysis<|message|>hmm<|end|>{tail}")

    assert stream == "<think>hmm</think>"


@pytest.mark.parametrize("tail", ["a", "an", "as", "to", "us", "a user"])
def test_short_text_after_end_is_visible(generate, tail):
    stream, _, _ = generate(f"<|channel|>analysis<|message|>hmm<|end|>{tail}")

    assert stream == f"<think>hmm</think>{tail}"


def test_missing_channel_marker_uses_bare_channel_word(generate):
    stream, _, _ = generate(
        "analysis<|message|>think<|end|><|start|>assistant final<|message|>ok"
    )

    assert stream == "<think>think</think>ok"


def test_streamed_indentation_matches_full_decode(generate):
    """Per-line decoding used to drop one space of indentation per line."""
    stream, _, _ = generate(
        "<|channel|>final<|message|>def f():\n    if x:\n        return 1\n"
    )

    assert stream == "def f():\n    if x:\n        return 1\n"


def test_angle_bracket_text_is_not_a_marker(generate):
    stream, _, _ = generate("<|channel|>final<|message|>a <| b <tag>")

    assert stream == "a <| b <tag>"


@pytest.mark.parametrize("text", ["Plain answer.", "final", "user"])
def test_off_protocol_text_is_not_dropped(generate, text):
    stream, _, final = generate(text)

    assert stream == text
    assert final.tool_calls == []


def test_long_off_protocol_text_flushes_before_finalize(tokenizer):
    session = Llmjp4OutputParserSession(tokenizer)
    streamed = "".join(
        session.process_token(token).stream_text
        for token in tokenizer.encode("word " * 80)
    )

    assert streamed.startswith("word word")
