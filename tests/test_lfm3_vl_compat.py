# SPDX-License-Identifier: Apache-2.0
"""LFM3-VL has to resolve through mlx-vlm, or the 2.2 GB checkpoint is unservable.

Pinned mlx-vlm predates the upstream merge (PR 2607), so
``mlx_vlm.utils.get_model_and_args`` raises ``Model type lfm3_vl not
supported`` before a weight is read. These tests pin the failure path and the
vendored registration, not just the happy path.
"""

from __future__ import annotations

import importlib

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from omlx.patches import mlx_vlm_lfm3_vl_compat  # noqa: E402
from omlx.patches.mlx_vlm_lfm3_vl_compat import (  # noqa: E402
    apply_mlx_vlm_lfm3_vl_compat_patch,
    is_lfm3_vl,
)

TEXT_CONFIG = {
    "model_type": "lfm3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 512,
    "norm_eps": 1e-5,
    "conv_bias": False,
    "conv_L_cache": 3,
    "sliding_window": 8,
    "max_position_embeddings": 64,
    "use_sandwich_norm": True,
    "tie_word_embeddings": True,
    "layer_types": ["conv", "conv", "full_attention", "sliding_attention"],
    "rope_parameters": {
        "full_attention": {
            "rope_theta": 1000000.0,
            "rope_type": "default",
            "partial_rotary_factor": 0.5,
        },
        "sliding_attention": {"rope_theta": 10000, "rope_type": "default"},
    },
}

VISION_CONFIG = {
    "model_type": "lfm3_vl_vision",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "conv_kernel_size": 3,
    "norm_eps": 1e-5,
    "layer_types": ["conv", "full_attention"],
    "rope_parameters": {"rope_type": "axial", "rope_theta": 1000000.0},
}

CONFIG = {
    "model_type": "lfm3_vl",
    "architectures": ["Lfm3VlForConditionalGeneration"],
    "text_config": TEXT_CONFIG,
    "vision_config": VISION_CONFIG,
    "downsample_factor": 2,
    "projector_hidden_size": 32,
    "projector_hidden_act": "gelu",
    "projector_bias": True,
    "projector_layer_norm_eps": 1e-6,
    "image_token_id": 125030,
    "video_token_id": 125029,
}

IMAGE_TOKEN = CONFIG["image_token_id"]


@pytest.fixture(autouse=True)
def _fresh_patch_state():
    """``_APPLIED`` is process-wide; leaking it would hide a broken import."""
    mlx_vlm_lfm3_vl_compat._APPLIED = False
    yield


def test_the_patch_reports_which_models_it_is_for():
    assert is_lfm3_vl({"model_type": "lfm3_vl"})
    assert not is_lfm3_vl({"model_type": "lfm3"})
    assert not is_lfm3_vl({})
    assert not is_lfm3_vl(None)


def test_the_architecture_resolves_through_mlx_vlm():
    """This is the exact call that failed with ``lfm3_vl not supported``."""
    import mlx_vlm.utils as vlm_utils

    with pytest.raises(ValueError, match="lfm3_vl not supported"):
        vlm_utils.get_model_and_args(CONFIG)

    assert apply_mlx_vlm_lfm3_vl_compat_patch()

    module, model_type = vlm_utils.get_model_and_args(CONFIG)
    assert model_type == "lfm3_vl"
    assert module is importlib.import_module("mlx_vlm.models.lfm3_vl")
    assert module.__name__.startswith("mlx_vlm.models.")
    # The text tower the VL module imports relatively must resolve too.
    assert importlib.import_module("mlx_vlm.models.lfm3").Model is not None


def test_applying_twice_is_a_no_op():
    assert apply_mlx_vlm_lfm3_vl_compat_patch()
    assert not apply_mlx_vlm_lfm3_vl_compat_patch()
    assert mlx_vlm_lfm3_vl_compat.is_applied()


def test_prompt_formatting_and_processor_are_registered():
    import mlx_vlm.prompt_utils as prompt_utils
    from transformers import AutoProcessor

    before = AutoProcessor.__dict__.get("from_pretrained")
    assert apply_mlx_vlm_lfm3_vl_compat_patch()
    assert "lfm3_vl" in prompt_utils.MODEL_CONFIG
    assert (
        prompt_utils.MODEL_CONFIG["lfm3_vl"]
        == prompt_utils.MessageFormat.LIST_WITH_IMAGE_FIRST
    )
    # The processor class is not resolvable by name yet, so transformers is
    # intercepted before it falls back to the pytorch references.
    assert AutoProcessor.__dict__.get("from_pretrained") is not before


def test_a_tiny_checkpoint_runs_text_and_image_forwards():
    assert apply_mlx_vlm_lfm3_vl_compat_patch()

    module = importlib.import_module("mlx_vlm.models.lfm3_vl")
    model = module.Model(module.ModelConfig.from_dict(CONFIG))

    text_only = model(mx.array([[1, 2, 3]]))
    assert text_only.logits.shape == (1, 3, TEXT_CONFIG["vocab_size"])

    # One 4x4-patch image at 16 px: 2x2 spatial merge, so four visual tokens.
    patches = mx.zeros((16, 3 * 2 * 16 * 16))
    grid = mx.array([[1, 4, 4]])
    tokens = mx.array([[1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 2]])
    out = model(tokens, pixel_values=patches, image_grid_thw=grid)
    assert out.logits.shape == (1, 6, TEXT_CONFIG["vocab_size"])


def test_sanitize_rewrites_the_checkpoint_layout():
    module = importlib.import_module("mlx_vlm.models.lfm3_vl")
    model = module.Model(module.ModelConfig.from_dict(CONFIG))

    conv = mx.zeros((32, 1, 3))  # checkpoint layout: (out, 1, kernel)
    out = model.sanitize(
        {
            "model.language_model.layers.0.conv.conv.weight": conv,
            "model.language_model.embed_tokens.weight": mx.zeros((512, 64)),
            "model.vision_tower.patchifier.weight": mx.zeros((32, 1536)),
            "lm_head.weight": mx.zeros((512, 64)),  # tied: dropped
        }
    )
    assert set(out) == {
        "language_model.model.layers.0.conv.conv.weight",
        "language_model.model.embed_tokens.weight",
        "vision_tower.patchifier.weight",
    }
    assert out["language_model.model.layers.0.conv.conv.weight"].shape == (32, 3, 1)


def test_the_dispatch_hook_fires_only_for_the_vlm_engine(tmp_path):
    """The text loader must not register a vision architecture."""
    import json

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    (tmp_path / "config.json").write_text(json.dumps(CONFIG))

    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=False)
    assert not mlx_vlm_lfm3_vl_compat.is_applied()

    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)
    assert mlx_vlm_lfm3_vl_compat.is_applied()
