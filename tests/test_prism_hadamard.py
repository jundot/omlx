# SPDX-License-Identifier: Apache-2.0
"""Hadamard pack regressions through the production custom-loader dispatch."""

import json
from unittest.mock import sentinel

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from omlx.patches.prism_hadamard import MODEL_TYPE, Packed, _install_packed
from omlx.utils.model_loading import maybe_load_custom_quantization


def _matrix(size):
    # Independent Sylvester construction, not the production FWHT kernel.
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < size:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(size)


@pytest.mark.parametrize("embedding", [False, True])
def test_packed_basis_matches_dense_reference(embedding):
    mx.random.seed(4)
    block, width, rows = 512, 1024, 8
    arrays = mx.quantize(
        mx.random.normal((rows, width)).astype(mx.float16), group_size=128, bits=2
    )
    signs = mx.array(
        np.random.default_rng(1).choice([-1, 1], size=width).astype(np.float32)
    )
    packed = Packed(arrays, block, signs, embedding)
    dense = np.asarray(mx.dequantize(*arrays, group_size=128, bits=2)).astype(
        np.float32
    )
    h = _matrix(block)
    if embedding:
        indices = mx.array([[0, 3, 0], [7, 1, 2]])
        expected = (dense[np.asarray(indices)].reshape(-1, block) @ h).reshape(
            2, 3, width
        ) * np.asarray(signs)
        actual = packed(indices)
    else:
        x = mx.random.normal((2, 3, width)).astype(mx.float16)
        rotated = (
            (
                (np.asarray(x).astype(np.float32) * np.asarray(signs)).reshape(
                    -1, block
                )
                @ h
            )
            .reshape(2, 3, width)
            .astype(np.float16)
        )
        expected = rotated.astype(np.float32) @ dense.T
        actual = packed(x)
        # Skipping the transform would be a plausible but incorrect loader fix.
        assert not np.allclose(
            np.asarray(actual), np.asarray(x).astype(np.float32) @ dense.T, atol=0.1
        )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.01, atol=0.03)


@pytest.mark.parametrize("embedding", [False, True])
def test_unrotated_packed_modules(embedding):
    arrays = mx.quantize(mx.ones((4, 128), dtype=mx.float16), group_size=128, bits=2)
    module = Packed(arrays, embedding=embedding)
    x = mx.array([[0, 3]]) if embedding else mx.ones((1, 128), dtype=mx.float16)
    dense = mx.dequantize(*arrays, group_size=128, bits=2)
    np.testing.assert_allclose(
        np.asarray(module(x)), np.asarray(dense[x] if embedding else x @ dense.T)
    )


@pytest.fixture
def text_pack(tmp_path):
    text_config = dict(
        model_type="qwen3_5_text",
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=128,
        vocab_size=16,
        full_attention_interval=1,
        tie_word_embeddings=False,
        linear_num_value_heads=4,
        linear_num_key_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
    )
    model = TextModel(TextModelArgs.from_dict(text_config))
    weights = dict(tree_flatten(model.parameters()))
    records = []
    # Both inverse embeddings and forward projections must survive strict load.
    for path, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        rows, width = module.weight.shape
        block = 512 if width % 512 == 0 else 0
        arrays = mx.quantize(module.weight.astype(mx.float16), group_size=128, bits=2)
        for suffix, array in zip(("weight", "scales", "biases"), arrays):
            weights[path + "." + suffix] = array
        if block:
            weights[path + ".signs"] = mx.ones((width,), dtype=mx.float32)
        records.append(
            dict(
                path=path,
                block=block,
                embedding=isinstance(module, nn.Embedding),
                dtype="float16",
            )
        )
    config = dict(
        schema_version=1,
        model_type=MODEL_TYPE,
        quantization=dict(bits=2, group_size=128, mode="affine"),
        text_config=text_config,
        modules=records,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return tmp_path, config, weights


def test_text_pack_loads_through_dispatch_without_checkpoint_code(
    text_pack, monkeypatch
):
    path, config, weights = text_pack
    monkeypatch.setattr(
        "mlx_lm.utils.load_tokenizer", lambda *a, **kw: sentinel.tokenizer
    )
    runtime = path / "runtime"
    runtime.mkdir()
    (runtime / "__init__.py").write_text(
        'raise RuntimeError("must not execute checkpoint code")'
    )
    model, tokenizer = maybe_load_custom_quantization(str(path), is_vlm=False)
    assert tokenizer is sentinel.tokenizer
    assert isinstance(model.model.embed_tokens, Packed)
    loaded = dict(tree_flatten(model.parameters()))
    for key, expected in weights.items():
        np.testing.assert_array_equal(np.asarray(loaded[key]), np.asarray(expected))
    logits = model(mx.array([[1, 2, 3]]))
    assert logits.shape == (1, 3, 16)
    assert mx.all(mx.isfinite(logits)).item()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"schema_version": 99}, "schema"),
        ({"quantization": {"bits": 4}}, "2-bit"),
        ({"gdn_activation_layout": "ungrouped"}, "layout"),
        ({"base_model_type": "llama"}, "base model"),
        ({"modules": []}, "manifest"),
        ({"components": {"vision": True}}, "matching text/vision"),
    ],
)
def test_rejects_unsupported_pack_metadata(text_pack, change, match):
    path, config, _ = text_pack
    (path / "config.json").write_text(json.dumps({**config, **change}))
    with pytest.raises(ValueError, match=match):
        maybe_load_custom_quantization(str(path), is_vlm=False)


@pytest.mark.parametrize(
    "case,match",
    [
        ("duplicate", "Duplicate"),
        ("target", "target"),
        ("block", "block size"),
        ("dtype", "activation dtype"),
        ("kind", "kind mismatch"),
        ("signs", "sign values"),
        ("missing_signs", "transform dimensions"),
        ("shape", "tensor shapes"),
        ("nan", "Non-finite"),
    ],
)
def test_rejects_invalid_manifest_or_weights(text_pack, case, match):
    _, config, weights = text_pack
    model = TextModel(TextModelArgs.from_dict(config["text_config"]))
    records = [dict(config["modules"][0])]
    r = records[0]
    if case == "duplicate":
        records.append(dict(r))
    elif case == "target":
        r["path"] = "model.nonexistent"
    elif case == "block":
        r["block"] = 3
    elif case == "dtype":
        r["dtype"] = "float32"
    elif case == "kind":
        r["embedding"] = not r["embedding"]
    elif case == "signs":
        weights[r["path"] + ".signs"] = mx.zeros_like(weights[r["path"] + ".signs"])
    elif case == "missing_signs":
        del weights[r["path"] + ".signs"]
    elif case == "shape":
        weights[r["path"] + ".weight"] = mx.zeros((1, 1), dtype=mx.uint32)
    elif case == "nan":
        weights[r["path"] + ".scales"] = mx.full(
            weights[r["path"] + ".scales"].shape, float("nan"), dtype=mx.float16
        )
    with pytest.raises(ValueError, match=match):
        _install_packed(model, records, weights)


def test_unrelated_models_keep_the_normal_loader(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5"}')
    assert maybe_load_custom_quantization(str(tmp_path), is_vlm=True) is None


def test_vision_pack_preserves_tower_and_registers_image_messages(
    text_pack, monkeypatch
):
    from mlx_vlm.models.qwen3_5 import Model, ModelConfig
    from mlx_vlm.prompt_utils import MODEL_CONFIG, get_message_json

    path, config, text_weights = text_pack
    config.update(
        schema_version=2,
        components={"text": True, "vision": True, "mtp": False},
        base_model_type="qwen3_5",
        vision_config=dict(
            model_type="qwen3_5",
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=512,
            num_position_embeddings=16,
            deepstack_visual_indexes=[],
        ),
    )
    base = Model(ModelConfig.from_dict(config))
    weights = dict(tree_flatten(base.parameters()))
    weights.update({"language_model." + k: v for k, v in text_weights.items()})
    (path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    monkeypatch.setattr(
        "omlx.patches.prism_hadamard._build_processor", lambda _: sentinel.processor
    )
    monkeypatch.delitem(MODEL_CONFIG, MODEL_TYPE, raising=False)
    model, processor = maybe_load_custom_quantization(str(path), is_vlm=True)
    assert processor is sentinel.processor
    assert isinstance(model.language_model.model.embed_tokens, Packed)
    loaded = dict(tree_flatten(model.parameters()))
    for key in weights:
        if not key.startswith("language_model."):
            np.testing.assert_array_equal(
                np.asarray(loaded[key]), np.asarray(weights[key])
            )
    for role, images in [("system", 0), ("user", 0), ("user", 1), ("assistant", 0)]:
        assert get_message_json(
            MODEL_TYPE, "A test turn", role=role, num_images=images
        ) == get_message_json("qwen3_5", "A test turn", role=role, num_images=images)
    with pytest.raises(ValueError, match="matching text/vision"):
        maybe_load_custom_quantization(str(path), is_vlm=False)


def test_missing_weight_is_not_silently_initialized(text_pack, monkeypatch):
    path, _, weights = text_pack
    del weights["model.norm.weight"]
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    with pytest.raises(ValueError, match="Missing"):
        maybe_load_custom_quantization(str(path), is_vlm=False)
