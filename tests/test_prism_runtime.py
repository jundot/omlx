# SPDX-License-Identifier: Apache-2.0
"""Runtime patches on real models loaded by the pinned native mlx-vlm loader."""

import copy
import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen3_5 import Model, ModelConfig
from mlx_vlm.models.qwen3_5.language import Qwen3_5Model
from mlx_vlm.utils import load_model

from omlx.patches.prism_hadamard import (
    MODEL_TYPE,
    apply_runtime_patches,
)
from omlx.patches.prism_hadamard.decode import CapacityPreservingModel
from omlx.utils.model_loading import maybe_load_custom_quantization


@pytest.fixture
def native_pack(tmp_path):
    mx.random.seed(7)
    config = dict(
        model_type=MODEL_TYPE,
        schema_version=2,
        base_model_type="qwen3_5",
        tensor_namespace="mlx-vlm-qwen3_5",
        gdn_activation_layout="grouped",
        quantization=dict(bits=2, group_size=128, mode="affine"),
        text_config=dict(
            model_type="qwen3_5_text",
            hidden_size=512,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=128,
            vocab_size=32,
            full_attention_interval=2,
            tie_word_embeddings=False,
            linear_num_value_heads=4,
            linear_num_key_heads=1,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
            max_position_embeddings=4096,
        ),
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
    records = []
    for path, module in base.language_model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        _, width = module.weight.shape
        block = 512 if width % 512 == 0 else 0
        arrays = mx.quantize(module.weight.astype(mx.float16), group_size=128, bits=2)
        key = "language_model." + path
        for suffix, array in zip(("weight", "scales", "biases"), arrays):
            weights[key + "." + suffix] = array
        if block:
            weights[key + ".signs"] = mx.ones((width,), dtype=mx.float32)
        records.append(
            dict(
                path=path,
                block=block,
                embedding=isinstance(module, nn.Embedding),
                dtype="float16",
            )
        )
    config["modules"] = records
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return tmp_path


@pytest.mark.parametrize("legacy_flag", [None, "1"])
def test_native_load_and_runtime_patch_preserve_weights(
    native_pack, monkeypatch, legacy_flag
):
    # The retired experimental setting must not change native precision.
    if legacy_flag is None:
        monkeypatch.delenv("OMLX_PRISM_FP16_ACTIVATIONS", raising=False)
    else:
        monkeypatch.setenv("OMLX_PRISM_FP16_ACTIVATIONS", legacy_flag)
    # The former custom loader must no longer intercept native pack loading.
    assert maybe_load_custom_quantization(str(native_pack), is_vlm=True) is None
    model = load_model(native_pack, strict=True)
    before = dict(tree_flatten(model.parameters()))
    reference = copy.deepcopy(model)
    assert apply_runtime_patches(model)
    assert apply_runtime_patches(model)  # Idempotent on this loaded instance.
    assert type(model.language_model.model) is CapacityPreservingModel
    after = dict(tree_flatten(model.parameters()))
    assert after.keys() == before.keys()
    for key, weight in after.items():
        assert weight is before[key]

    lm = model.language_model
    cache = lm.make_cache()
    inputs = mx.array([[1, 2, 3]])
    hidden = lm.model(inputs, cache=cache)
    logits = lm.lm_head(hidden)
    mx.eval(logits, [c.state for c in cache])
    assert mx.all(mx.isfinite(logits)).item()
    assert not hasattr(model, "_omlx_prism_activation_signature")
    assert hidden.dtype == mx.float32
    assert cache[0][1].dtype == mx.float32
    assert cache[lm.model.fa_idx].keys.dtype == mx.float32
    expected = reference.language_model(
        inputs, cache=reference.language_model.make_cache()
    ).logits
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(expected))


def test_other_models_and_decoder_subclasses_are_unchanged(monkeypatch):
    monkeypatch.setenv("OMLX_PRISM_FP16_ACTIVATIONS", "1")
    unrelated = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))
    assert not apply_runtime_patches(unrelated)

    class OtherDecoder(Qwen3_5Model):
        pass

    decoder = OtherDecoder.__new__(OtherDecoder)
    model = SimpleNamespace(
        config=SimpleNamespace(model_type=MODEL_TYPE),
        language_model=SimpleNamespace(model=decoder),
    )
    assert not apply_runtime_patches(model)
    assert type(decoder) is OtherDecoder
    assert not hasattr(model, "_omlx_prism_activation_signature")
