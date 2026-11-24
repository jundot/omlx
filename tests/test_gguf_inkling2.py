# SPDX-License-Identifier: Apache-2.0
"""Tests for Inkling 2's GGUF layouts: the derived config and tokenizer."""

import json

from omlx import gguf


def _metadata(arch="inkling2_loop", **values):
    """An Inkling 2 header: sizes stated once, head counts per application."""
    layers = values.pop("applications", 24)
    tokens = values.pop("tokens", 200058)
    metadata = {
        "general.architecture": arch,
        f"{arch}.block_count": layers,
        f"{arch}.context_length": 1048576,
        f"{arch}.embedding_length": 2304,
        f"{arch}.expert_count": 128,
        f"{arch}.expert_used_count": 8,
        f"{arch}.attention.key_length": 128,
        f"{arch}.attention.head_count": [24] * layers,
        f"{arch}.attention.head_count_kv": [8] * layers,
        f"{arch}.vocab_size": 201024,
        "tokenizer.ggml.tokens": gguf.ArrayInfo(length=tokens),
    }
    metadata.update(values)
    return metadata


def test_llama_cpp_layout_derives_the_family_sizes():
    config = gguf.derive_config(_metadata())

    text = config["text_config"]
    assert config["model_type"] == "inkling2_loop"
    assert text["model_type"] == "inkling2_loop_text"
    assert text["num_hidden_layers"] == 24
    assert text["num_attention_heads"] == 24
    assert text["num_key_value_heads"] == 8
    # The token table holds the unpadded tokens; the model's rows are padded.
    assert text["vocab_size"] == 201024
    assert text["max_position_embeddings"] == 1048576


def test_unsloth_layout_counts_weight_layers_not_applications():
    # Unsloth's inkling2 runs the loop's core layers once per pass: 30
    # applications are 18 weight layers.
    metadata = _metadata(
        arch="inkling2",
        applications=30,
        **{
            "inkling2.loop.core_start": 6,
            "inkling2.loop.core_end": 18,
            "inkling2.num_loops": 2,
        },
    )

    text = gguf.derive_config(metadata)["text_config"]

    assert text["num_hidden_layers"] == 18
    assert text["num_attention_heads"] == 24
    assert text["num_key_value_heads"] == 8


def test_per_layer_head_counts_that_differ_are_not_stated():
    metadata = _metadata()
    metadata["inkling2_loop.attention.head_count"] = [24] * 23 + [32]

    text = gguf.derive_config(metadata)["text_config"]

    assert "num_attention_heads" not in text
    assert text["num_key_value_heads"] == 8


def test_derived_config_selects_the_splash_family():
    from omlx.engine.splash import splash_family

    config = gguf.derive_config(_metadata())

    assert splash_family(config) == "Inkling-2-Mini"


def test_loop_config_and_draft_target_layer_count():
    from omlx.engine.splash import _draft_target_layers

    metadata = _metadata(
        **{
            "inkling2_loop.loop.start": 6,
            "inkling2_loop.loop.end": 18,
            "inkling2_loop.loop.count": 3,
        }
    )

    text = gguf.derive_config(metadata)["text_config"]

    assert text["num_hidden_layers"] == 24
    assert text["loop_config"] == {
        "core_start": 6,
        "core_end": 18,
        "num_passes": 3,
    }
    # The published DFlash2 draft counts the loop's applications: 24 weight
    # layers, the 12 core ones run three times.
    assert _draft_target_layers(text) == 48


def test_inkling2_tokenizer_profile_keeps_whole_vocabulary_tokens():
    tokens = {
        "abc": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "ab": 4,
        "<|a|>": 5,
    }
    metadata = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "inkling2",
        # The vocabulary holds "abc" whole, which the merges do not build.
        "tokenizer.ggml.tokens": list(tokens),
        "tokenizer.ggml.token_type": [1, 1, 1, 1, 1, 3],
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.eos_token_id": 5,
        "general.architecture": "inkling2_loop",
        "inkling2_loop.context_length": 1048576,
    }

    files = gguf.tokenizer_files(metadata)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(files["tokenizer.json"])
    # ignore_merges: "abc" is a vocabulary token, so it is kept whole.
    assert tokenizer.encode("abc", add_special_tokens=False).tokens == ["abc"]
    # "ab" comes from the vocabulary as a piece too, not from a b + c merge.
    assert tokenizer.encode("ab", add_special_tokens=False).tokens == ["ab"]
    assert tokenizer.encode("ac", add_special_tokens=False).tokens == ["a", "c"]
    config = json.loads(files["tokenizer_config.json"])
    assert config["model_max_length"] == 1048576
