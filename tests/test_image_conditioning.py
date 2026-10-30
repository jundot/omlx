# SPDX-License-Identifier: Apache-2.0
"""Small independent numerical checks; real checkpoints are an explicit gate."""

import threading

import mlx.core as mx
import numpy as np
import pytest

from omlx.image.conditioning import check_cancel, encoder_shapes, prompt_tokens
from omlx.image.loading import transformer_shapes
from omlx.image.transformer import rope, rotate


def test_pre_cancelled_work_never_runs():
    event = threading.Event()
    event.set()
    with pytest.raises(InterruptedError):
        check_cancel(event)


@pytest.mark.parametrize("prompt", [None, "", "  ", "a" * 32769])
def test_bad_prompt_rejected_before_tokenizer_io(tmp_path, prompt):
    with pytest.raises(ValueError):
        prompt_tokens(tmp_path, prompt)


def test_encoder_contract_dimensions():
    shapes = encoder_shapes()
    assert len(shapes) == 399
    assert shapes["layers.35.self_attn.k_proj.weight"] == ([1024, 4096], "BF16")
    assert shapes["norm.weight"] == ([4096], "F32")
    assert shapes["lm_head.weight"] == shapes["embed_tokens.weight"]
    assert len(transformer_shapes()) == 265
    assert (
        transformer_shapes()["img_in.weight"][1]
        == transformer_shapes()["proj_out.weight"][0]
        == 64
    )


@pytest.mark.parametrize("height,width", [(2, 2), (3, 5), (4, 4)])
def test_rotary_matches_independent_complex_arithmetic(height, width):
    n = 3 + height * width
    cos, sin = rope(3, height, width)
    rng = np.random.default_rng(12)
    value = rng.normal(size=(1, n, 2, 128)).astype(np.float32)
    pairs = value.reshape(1, n, 2, 64, 2)
    complex_value = pairs[..., 0] + 1j * pairs[..., 1]
    phase = np.array(cos) + 1j * np.array(sin)
    reference = complex_value * phase[None, :, None, :]
    expected = np.stack([reference.real, reference.imag], axis=-1).reshape(value.shape)
    np.testing.assert_allclose(
        np.array(rotate(mx.array(value), cos, sin)), expected, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(np.array(cos) ** 2 + np.array(sin) ** 2, 1, atol=2e-6)


def test_bad_bundle_contract_does_not_load_weights(tmp_path):
    import json

    from omlx.image.conditioning import native_bundle

    (tmp_path / "omlx-pipeline.json").write_text(json.dumps({"schema_version": True}))
    with pytest.raises(ValueError, match="contract"):
        native_bundle(tmp_path)


def test_prompt_prefix_and_token_limit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from omlx.image import conditioning

    path = tmp_path / "processor"
    path.mkdir()
    (path / "tokenizer.json").write_text("{}")

    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            return SimpleNamespace(
                ids=list(range(14 if text == conditioning.PREFIX else 513))
            )

    monkeypatch.setattr(
        conditioning,
        "Tokenizer",
        SimpleNamespace(from_file=lambda path: FakeTokenizer()),
    )
    with pytest.raises(ValueError, match="truncation"):
        prompt_tokens(tmp_path, "valid prompt")
