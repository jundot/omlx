# SPDX-License-Identifier: Apache-2.0
"""LFM3-Omni has to resolve through mlx-vlm, or the 2.7 GB checkpoint is
unservable.

Pinned mlx-vlm predates the upstream merge (PR #2610), so
``mlx_vlm.utils.get_model_and_args`` raises ``Model type lfm3_omni not
supported`` before a weight is read. LFM3-Omni is a shell around ``lfm3``,
``lfm3_vl`` and ``lfm3_audio``, so these tests pin the failure path, the whole
dependency chain and the vendored registration, not just the happy path.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from omlx.patches import mlx_vlm_lfm3_omni_compat  # noqa: E402
from omlx.patches.mlx_vlm_lfm3_omni_compat import (  # noqa: E402
    apply_mlx_vlm_lfm3_omni_compat_patch,
    is_lfm3_omni,
)

REAL_CHECKPOINT = "/Users/panda/.omlx/models/LiquidAI/LFM3-Omni-2B-MLX-8bit"

# The shapes mirror the checkpoint's config.json (``audio_encoder_config.
# text_config`` is the LFM2 trunk, ``depth_config`` the MaskGIT decoder), with
# every dimension shrunk so a forward is instant.
TEXT_CONFIG = {
    "model_type": "lfm3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
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
    "layer_types": ["conv", "full_attention"],
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

AUDIO_ENCODER_CONFIG = {
    "conv_channels": 8,
    "hidden_size": 16,
    "num_mel_bins": 8,  # the three stride-2 convolutions land on one mel bin
    "sliding_window": 4,
    "text_config": {
        "model_type": "lfm2",
        "vocab_size": 2,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 64,
        "norm_eps": 1e-5,
        "conv_bias": False,
        "conv_L_cache": 3,
        "block_dim": 16,
        "block_ff_dim": 32,
        "block_multiple_of": 8,
        "block_ffn_dim_multiplier": 1.0,
        "block_auto_adjust_ff_dim": False,
        "layer_types": ["conv", "full_attention"],
    },
}

DEPTH_CONFIG = {
    "model_type": "lfm3",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 1,
    "norm_eps": 1e-5,
    "conv_bias": False,
    "conv_L_cache": 7,
    "sliding_window": 8,
    "use_sandwich_norm": False,
    "layer_types": ["full_attention"],
}

CONFIG = {
    "model_type": "lfm3_omni",
    "architectures": ["Lfm3OmniForConditionalGeneration"],
    "text_config": TEXT_CONFIG,
    "vision_config": VISION_CONFIG,
    "audio_encoder_config": AUDIO_ENCODER_CONFIG,
    "depth_config": DEPTH_CONFIG,
    "downsample_factor": 2,
    "projector_hidden_size": 32,
    "projector_hidden_act": "gelu",
    "projector_bias": True,
    "projector_layer_norm_eps": 1e-6,
    "num_codebooks": 4,
    "codebook_size": 32,
    "maskgit_schedule": [2, 2],
    "audio_eos_code": 31,
    "image_token_id": 125030,
    "video_token_id": 125029,
    "audio_token_id": 125031,
    "audio_out_token_id": 125032,
}

IMAGE_TOKEN = CONFIG["image_token_id"]


@pytest.fixture(autouse=True)
def _fresh_patch_state():
    """``_APPLIED`` is process-wide; leaking it would hide a broken import."""
    mlx_vlm_lfm3_omni_compat._APPLIED = False
    yield


@pytest.fixture(scope="module", autouse=True)
def _leave_the_pinned_wheel_as_it_was():
    """Hand ``mlx_vlm`` back exactly as the pinned wheel ships it.

    One process runs every test file, and each sibling patch test starts from
    ``Model type lfm3_* not supported``. Registering four vendored trees here
    would make those assertions fail for whichever file runs next, so the
    vendored package paths, modules and prompt config are removed afterwards.
    """
    import mlx_vlm.models
    import mlx_vlm.prompt_utils as prompt_utils

    prefix = "mlx_vlm.models.lfm3"
    packages = (mlx_vlm, mlx_vlm.models)
    vendored = str(Path(__file__).resolve().parents[1] / "omlx" / "patches")
    saved_attributes = {
        name: getattr(mlx_vlm.models, name, None)
        for name in ("lfm3", "lfm3_vl", "lfm3_audio", "lfm3_omni")
    }
    saved_modules = {
        name: module for name, module in sys.modules.items() if name.startswith(prefix)
    }
    saved_config = dict(prompt_utils.MODEL_CONFIG)
    yield

    mlx_vlm_lfm3_omni_compat._APPLIED = False
    for package in packages:
        package.__path__[:] = [
            path for path in package.__path__ if not path.startswith(vendored)
        ]
    for name in [name for name in sys.modules if name.startswith(prefix)]:
        del sys.modules[name]
    sys.modules.update(saved_modules)
    for name, module in saved_attributes.items():
        if module is not None:
            setattr(mlx_vlm.models, name, module)
        elif hasattr(mlx_vlm.models, name):
            delattr(mlx_vlm.models, name)
    prompt_utils.MODEL_CONFIG.clear()
    prompt_utils.MODEL_CONFIG.update(saved_config)


def test_the_patch_reports_which_models_it_is_for():
    assert is_lfm3_omni({"model_type": "lfm3_omni"})
    assert not is_lfm3_omni({"model_type": "lfm3_audio"})
    assert not is_lfm3_omni({})
    assert not is_lfm3_omni(None)


def test_the_architecture_resolves_through_mlx_vlm():
    """This is the exact call that failed with ``lfm3_omni not supported``."""
    import mlx_vlm.utils as vlm_utils

    with pytest.raises(ValueError, match="lfm3_omni not supported"):
        vlm_utils.get_model_and_args(CONFIG)

    assert apply_mlx_vlm_lfm3_omni_compat_patch()

    module, model_type = vlm_utils.get_model_and_args(CONFIG)
    assert model_type == "lfm3_omni"
    assert module is importlib.import_module("mlx_vlm.models.lfm3_omni")
    assert module.__name__.startswith("mlx_vlm.models.")


def test_the_whole_dependency_chain_resolves_from_the_vendor_trees():
    """``lfm3_omni`` imports three trees at module import time."""
    assert apply_mlx_vlm_lfm3_omni_compat_patch()

    for name in ("lfm3", "lfm3_vl", "lfm3_audio", "lfm3_omni"):
        module = importlib.import_module(f"mlx_vlm.models.{name}")
        assert module.Model is not None
        # every tree resolves to a vendored copy, never to the pinned wheel
        assert "vendor/mlx_vlm/models" in module.__file__


def test_applying_twice_is_a_no_op():
    assert apply_mlx_vlm_lfm3_omni_compat_patch()
    assert not apply_mlx_vlm_lfm3_omni_compat_patch()
    assert mlx_vlm_lfm3_omni_compat.is_applied()


def test_a_missing_sibling_vendor_tree_fails_quietly(tmp_path, monkeypatch, caplog):
    """A partial install must not abort the loader that called the patch."""
    missing = tmp_path / "lfm3_audio_compat" / "vendor" / "mlx_vlm"
    monkeypatch.setattr(mlx_vlm_lfm3_omni_compat, "_AUDIO_VENDOR_MLX_VLM", missing)

    assert not apply_mlx_vlm_lfm3_omni_compat_patch()
    assert not mlx_vlm_lfm3_omni_compat.is_applied()
    assert str(missing) in caplog.text


def test_prompt_formatting_and_processor_are_registered():
    import mlx_vlm.prompt_utils as prompt_utils
    from transformers import AutoProcessor

    before = AutoProcessor.__dict__.get("from_pretrained")
    assert apply_mlx_vlm_lfm3_omni_compat_patch()
    assert "lfm3_omni" in prompt_utils.MODEL_CONFIG
    assert (
        prompt_utils.MODEL_CONFIG["lfm3_omni"]
        == prompt_utils.MessageFormat.LIST_WITH_IMAGE_FIRST
    )
    # The processor class is not resolvable by name yet, so transformers is
    # intercepted before it falls back to the pytorch references.
    assert AutoProcessor.__dict__.get("from_pretrained") is not before


def test_a_tiny_checkpoint_runs_text_and_image_forwards():
    assert apply_mlx_vlm_lfm3_omni_compat_patch()

    module = importlib.import_module("mlx_vlm.models.lfm3_omni")
    model = module.Model(module.ModelConfig.from_dict(CONFIG))

    text_only = model(mx.array([[1, 2, 3]]))
    assert text_only.logits.shape == (1, 3, TEXT_CONFIG["vocab_size"])

    # One 4x4-patch image at 16 px: 2x2 spatial merge, so four visual tokens.
    patches = mx.zeros((16, 3 * 2 * 16 * 16))
    grid = mx.array([[1, 4, 4]])
    tokens = mx.array([[1, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, 2]])
    out = model(tokens, pixel_values=patches, image_grid_thw=grid)
    assert out.logits.shape == (1, 6, TEXT_CONFIG["vocab_size"])


def test_the_real_checkpoint_config_parses():
    """The real ``config.json`` must land on the vendored config class."""
    import os

    path = os.path.join(REAL_CHECKPOINT, "config.json")
    if not os.path.isfile(path):
        pytest.skip("LFM3-Omni checkpoint is not downloaded")

    module = importlib.import_module("mlx_vlm.models.lfm3_omni")
    with open(path, encoding="utf-8") as file:
        config = module.ModelConfig.from_dict(json.load(file))

    assert isinstance(config, module.ModelConfig)
    assert config.model_type == "lfm3_omni"
    assert config.text_config.hidden_size == 2048
    assert config.lm_config is config.text_config
    assert sum(config.maskgit_schedule) == config.num_codebooks


def test_the_dispatch_hook_fires_only_for_the_vlm_engine(tmp_path):
    """The text loader must not register an any-to-any architecture."""
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    (tmp_path / "config.json").write_text(json.dumps(CONFIG))

    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=False)
    assert not mlx_vlm_lfm3_omni_compat.is_applied()

    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)
    assert mlx_vlm_lfm3_omni_compat.is_applied()
