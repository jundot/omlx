# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Nemotron 3.5 Parse mlx-vlm compatibility layer."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np


def _patched_modules():
    from omlx.patches.mlx_vlm_nemotron_parse_compat import (
        apply_mlx_vlm_nemotron_parse_compat_patch,
    )

    apply_mlx_vlm_nemotron_parse_compat_patch()
    from mlx_vlm.models.nemotron_parse_v3_5 import (
        nemotron_parse_v3_5 as np_model,
    )
    from mlx_vlm.models.nemotron_parse_v3_5 import (
        processing_nemotron_parse_v3_5 as np_processing,
    )

    return np_model, np_processing


def test_nemotron_parse_processor_reserves_a_placeholder_per_visual_token():
    _, np_processing = _patched_modules()

    def image_processor(images):
        return {
            "pixel_values": np.zeros(((32 // 16) * (64 // 16), 768), dtype=np.float32),
            "image_sizes": np.array([[32, 64]], dtype=np.int32),
            "content_sizes": np.array([[32, 64]], dtype=np.int32),
        }

    def tokenizer(text, add_special_tokens=False, return_tensors="np"):
        return {
            "input_ids": np.array([[5, 6]], dtype=np.int32),
            "attention_mask": np.array([[1, 1]], dtype=np.int32),
        }

    processor = np_processing.Processor(image_processor, tokenizer)
    data = processor(images=[object()], text="a page")

    # (32 // 16) * (64 // 64) + 1 = 3 visual-prefix embeddings, so three <pad>
    # placeholders in front of the two text ids.
    assert data["input_ids"].tolist() == [[1, 1, 1, 5, 6]]
    assert data["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]


def test_nemotron_parse_embeddings_drop_the_placeholders():
    np_model, _ = _patched_modules()
    vision = mx.zeros((1, 3, 4), dtype=mx.bfloat16)

    def vision_tower(pixel_values, image_sizes, content_sizes):
        return vision

    def embed(ids):
        return mx.zeros((1, ids.shape[1], 4), dtype=mx.bfloat16)

    stub = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=1),
        vision_tower=vision_tower,
        language_model=SimpleNamespace(
            embed=embed,
            embed_tokens=SimpleNamespace(weight=mx.zeros((1, 1), dtype=mx.bfloat16)),
        ),
    )

    out = np_model.Model.get_input_embeddings(
        stub,
        mx.array([[1, 1, 1, 7, 8]]),
        mx.zeros((2, 768), dtype=mx.bfloat16),
    )

    # Three vision rows plus the two text rows the placeholders stood for.
    assert out.inputs_embeds.shape == (1, 5, 4)

    # A prompt that does not start with the placeholders keeps all its tokens.
    plain = np_model.Model.get_input_embeddings(
        stub,
        mx.array([[7, 8, 9, 10, 11]]),
        mx.zeros((2, 768), dtype=mx.bfloat16),
    )

    assert plain.inputs_embeds.shape == (1, 8, 4)


def test_nemotron_parse_compat_layer_is_wired_into_vlm_loads(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "nemotron_parse_v3_5"})
    )
    from omlx.patches import mlx_vlm_nemotron_parse_compat as compat
    from omlx.utils import model_loading

    calls: list[bool] = []
    monkeypatch.setattr(
        compat,
        "apply_mlx_vlm_nemotron_parse_compat_patch",
        lambda: calls.append(True) or False,
    )

    model_loading.maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)

    assert calls
