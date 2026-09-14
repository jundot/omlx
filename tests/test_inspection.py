# SPDX-License-Identifier: Apache-2.0
"""Inspection is readable and explicit about losses, never a restore input."""

import json

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models

from omlx.cache.inspection import (
    BlockInspection,
    InspectionRenderer,
    image_context_descriptors,
    media_context_for_block,
)


@pytest.fixture
def tokenizer():
    backend = Tokenizer(
        models.WordLevel({"hello": 0, "world": 1, "[UNK]": 2}, unk_token="[UNK]")
    )
    backend.add_special_tokens(
        [
            AddedToken("<|image_pad|>", special=True),
            AddedToken("<|im_start|>", special=True),
        ]
    )
    return backend


def test_json_preserves_exact_ids_and_parent(tokenizer):
    renderer = InspectionRenderer(tokenizer, "org/model")
    block = BlockInspection((0, 1, 999999), 256, "ab" * 32)
    raw, text = renderer.render(bytes(32), block)
    data = json.loads(raw)
    assert data["token_ids"] == [0, 1, 999999]
    assert data["parent_hash"] == "ab" * 32
    assert data["token_start"] == 256
    assert data["token_count"] == 3
    assert data["format_version"] == 1
    assert data["renderer_version"] == 2
    assert data["tokenizer"]["fingerprint"].startswith("sha256:")
    header = text.decode().split("--- decoded content ---\n")[0]
    assert "renderer v2" in header
    assert "Model: org/model\n" in header
    assert f"Parent block: {'ab' * 32}\n" in header
    assert "Tokens in this block: 3\n" in header
    assert "Token range: [256, 259)\n" in header
    assert "zero-based; start is inclusive, end is exclusive" in header
    assert b"hello world" in text
    assert b"unknown token: id=999999" in text


def test_registered_media_runs_collapse_but_other_specials_survive(tokenizer):
    image = tokenizer.token_to_id("<|image_pad|>")
    role = tokenizer.token_to_id("<|im_start|>")
    raw, text = InspectionRenderer(tokenizer, "model").render(
        bytes(32), BlockInspection((role, 0, image, image, image), 0, None)
    )
    body = text.decode().split("--- decoded content ---\n")[1]
    assert body.startswith("<|im_start|>hello")
    assert "× 3" in body
    assert "[2, 5)" in body
    assert json.loads(raw)["token_ids"] == [role, 0, image, image, image]


def test_byte_fallback_is_decoded_as_a_run():
    backend = Tokenizer(
        models.BPE({"<0xE2>": 0, "<0x82>": 1, "<0xAC>": 2}, [], byte_fallback=True)
    )
    backend.decoder = decoders.ByteFallback()
    renderer = InspectionRenderer(backend, "bytes")
    _, whole = renderer.render(bytes(32), BlockInspection((0, 1, 2), 0, None))
    assert "€" in whole.decode()
    _, partial = renderer.render(bytes(32), BlockInspection((0, 1), 0, None))
    assert "replacement characters" in partial.decode()


def test_control_sequences_and_literal_annotations_are_escaped():
    backend = Tokenizer(
        models.WordLevel({"\x1b[31m⟦image⟧\u202e\n\t": 0, "<|image_pad|>": 1})
    )
    model_name = "org/model\nInjected: label\t\x1b[31m⟦fake⟧\u202e"
    raw, text = InspectionRenderer(backend, model_name).render(
        bytes(32), BlockInspection((0, 1, 1), 0, None)
    )
    assert json.loads(raw)["model"] == model_name
    header = text.decode().split("--- decoded content ---\n")[0]
    assert (
        "Model: org/model\\nInjected: label\\t\\u001b[31m\\u27e6fake\\u27e7\\u202e\n"
        in header
    )
    assert "\nInjected:" not in header
    assert "\x1b" not in header and "\u202e" not in header
    body = text.decode().split("--- decoded content ---\n")[1]
    assert "\x1b" not in body and "\u202e" not in body
    assert "\\u001b[31m\\u27e6image\\u27e7\\u202e\n\t" in body
    assert body.count("<|image_pad|>") == 2  # Ordinary text, not registered media.


def test_snapshot_does_not_follow_tokenizer_mutations(tokenizer):
    renderer = InspectionRenderer(tokenizer, "model")
    tokenizer.add_tokens(["new-token"])
    _, text = renderer.render(
        bytes(32), BlockInspection((tokenizer.token_to_id("new-token"),), 0, None)
    )
    assert b"unknown token" in text


def test_missing_decoder_keeps_ids_with_explicit_annotations():
    raw, text = InspectionRenderer(None, "model").render(
        bytes(32), BlockInspection((44, 55), 0, None)
    )
    assert json.loads(raw)["token_ids"] == [44, 55]
    assert b"Parent block: none (root)\n" in text
    assert b"undecoded token: id=44" in text


def test_media_context_never_exposes_future_images():
    media = image_context_descriptors(
        [(640, 480), (800, 600)], "bb", [(10, "aa"), (300, "bb")], [1, 1]
    )
    assert media_context_for_block(media, 0, 10) == ()
    early = media_context_for_block(media, 0, 256)
    assert len(early) == 1
    assert early[0]["input_dimensions"] == [[640, 480]]
    assert early[0]["token_span"] is None
    assert early[0]["fingerprint"] == "sha256:aa"
    crossing = media_context_for_block(media, 256, 512)
    assert len(crossing) == 2
    assert crossing[1]["input_dimensions"] == [[640, 480], [800, 600]]


def test_media_fallback_is_explicit_whole_context():
    media = image_context_descriptors([(20, 30)], "abc", [], [])
    assert media[0]["key_start"] == 0
    assert media[0]["scope"] == "cumulative_cache_key_context"
    assert image_context_descriptors([], None, [], []) == ()
