# SPDX-License-Identifier: Apache-2.0
"""Tests for the gemma4 assistant MTP combine step in omlx.oq.

Uses tiny synthetic checkpoints on disk — no model loading, no GPU work
beyond a few small mx arrays.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from omlx.oq import (
    GEMMA4_ASSISTANT_MTP_PREFIX,
    GEMMA4_ASSISTANT_MTP_SHARD,
    combine_gemma4_assistant_mtp,
    validate_gemma4_assistant_pair,
)

BASE_CONFIG = {
    "model_type": "gemma4",
    "vision_config": {},
    "text_config": {"model_type": "gemma4_text", "hidden_size": 24},
    "quantization": {"group_size": 64, "bits": 4},
}

ASSISTANT_CONFIG = {
    "model_type": "gemma4_assistant",
    "backbone_hidden_size": 24,
    "tie_word_embeddings": True,
    "text_config": {"model_type": "gemma4_text", "hidden_size": 8, "num_hidden_layers": 2},
}


def _write_base_output(tmp_path):
    out = tmp_path / "base-oQ4"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(BASE_CONFIG))
    weights = {"language_model.model.embed_tokens.weight": mx.zeros((4, 24))}
    mx.save_safetensors(str(out / "model-00001-of-00001.safetensors"), weights)
    index = {
        "metadata": {"total_size": 100},
        "weight_map": {
            k: "model-00001-of-00001.safetensors" for k in weights
        },
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index))
    return out


def _write_assistant(tmp_path, config=None):
    asst = tmp_path / "assistant"
    asst.mkdir()
    (asst / "config.json").write_text(json.dumps(config or ASSISTANT_CONFIG))
    weights = {
        "model.embed_tokens.weight": mx.ones((4, 8)),
        "pre_projection.weight": mx.ones((8, 48)),
        "post_projection.weight": mx.ones((24, 8)),
    }
    mx.save_safetensors(str(asst / "model.safetensors"), weights)
    return asst


def test_combine_writes_shard_index_and_config(tmp_path):
    out = _write_base_output(tmp_path)
    asst = _write_assistant(tmp_path)

    combine_gemma4_assistant_mtp(out, asst)

    shard = out / GEMMA4_ASSISTANT_MTP_SHARD
    assert shard.exists()
    merged = mx.load(str(shard))
    assert set(merged) == {
        GEMMA4_ASSISTANT_MTP_PREFIX + "model.embed_tokens.weight",
        GEMMA4_ASSISTANT_MTP_PREFIX + "pre_projection.weight",
        GEMMA4_ASSISTANT_MTP_PREFIX + "post_projection.weight",
    }

    index = json.loads((out / "model.safetensors.index.json").read_text())
    for key in merged:
        assert index["weight_map"][key] == GEMMA4_ASSISTANT_MTP_SHARD
    # Base entries survive and total_size grows by the mtp shard bytes.
    assert (
        index["weight_map"]["language_model.model.embed_tokens.weight"]
        == "model-00001-of-00001.safetensors"
    )
    mtp_bytes = sum(v.nbytes for v in merged.values())
    assert index["metadata"]["total_size"] == 100 + mtp_bytes

    config = json.loads((out / "config.json").read_text())
    tc = config["text_config"]
    assert tc["mtp_num_hidden_layers"] == 2
    assert tc["mtp_assistant_config"] == ASSISTANT_CONFIG
    # Base fields untouched.
    assert config["quantization"] == BASE_CONFIG["quantization"]
    assert tc["hidden_size"] == 24


def test_combine_rejects_non_assistant_model(tmp_path):
    out = _write_base_output(tmp_path)
    wrong = dict(ASSISTANT_CONFIG)
    wrong["model_type"] = "gemma4"
    asst = _write_assistant(tmp_path, config=wrong)
    with pytest.raises(ValueError, match="gemma4_assistant"):
        combine_gemma4_assistant_mtp(out, asst)


def test_validate_rejects_hidden_size_mismatch():
    mismatched = dict(ASSISTANT_CONFIG)
    mismatched["backbone_hidden_size"] = 32
    with pytest.raises(ValueError, match="backbone_hidden_size"):
        validate_gemma4_assistant_pair(BASE_CONFIG, mismatched)


def test_validate_rejects_non_gemma4_base():
    base = dict(BASE_CONFIG)
    base["model_type"] = "qwen3_5"
    with pytest.raises(ValueError, match="gemma4 base"):
        validate_gemma4_assistant_pair(base, ASSISTANT_CONFIG)


def test_validate_rejects_headless_assistant():
    headless = dict(ASSISTANT_CONFIG)
    headless["text_config"] = {"model_type": "gemma4_text"}
    with pytest.raises(ValueError, match="num_hidden_layers"):
        validate_gemma4_assistant_pair(BASE_CONFIG, headless)
