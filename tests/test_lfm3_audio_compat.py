# SPDX-License-Identifier: Apache-2.0
"""LFM3-Audio has to resolve through mlx-vlm, or the 2.33 GB checkpoint is unservable.

Pinned mlx-vlm predates the upstream merge (PR 2609), so
``mlx_vlm.utils.get_model_and_args`` raises ``Model type lfm3_audio not
supported`` before a weight is read. These tests pin the failure path, the
vendored registration and both forwards, not just the happy path.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from omlx.patches import mlx_vlm_lfm3_audio_compat  # noqa: E402
from omlx.patches.mlx_vlm_lfm3_audio_compat import (  # noqa: E402
    apply_mlx_vlm_lfm3_audio_compat_patch,
    is_lfm3_audio,
)

# Shrunk from LiquidAI/LFM3-Audio-2B-MLX-8bit/config.json: same nesting
# (``lfm`` / ``encoder`` / ``depthformer``), tiny widths.
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

ENCODER_CONFIG = {
    "dim_in": 128,
    "dim_conv": 8,
    "dim_out": 32,
    "sliding_window": 4,
    "lfm": {
        "model_type": "lfm2",
        "vocab_size": 2,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 64,
        "norm_eps": 1e-5,
        "conv_bias": False,
        "conv_L_cache": 3,
        "block_multiple_of": 256,
        "block_ffn_dim_multiplier": 1.0,
        "block_auto_adjust_ff_dim": True,
        "layer_types": ["conv"],
    },
}

DEPTH_CONFIG = {
    "model_type": "lfm3",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "vocab_size": 1,
    "norm_eps": 1e-5,
    "conv_L_cache": 3,
    "sliding_window": 16,
    "layer_types": ["full_attention"],
    "rope_parameters": {
        "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 1.0}
    },
    "tie_word_embeddings": True,
}

CONFIG = {
    "model_type": "lfm3_audio",
    "architectures": ["Lfm3AudioForConditionalGeneration"],
    "codebooks": 2,
    "audio_vocab_size": 8,
    "maskgit_steps": 2,
    "interleaved_n_text": 2,
    "interleaved_n_audio": 2,
    "audio_token_id": 500,
    "audio_out_token_id": 501,
    "audio_start_token_id": 502,
    "text_end_token_id": 503,
    "mixed_start_token_id": 504,
    "eos_token_id": [1],
    "lfm": TEXT_CONFIG,
    "encoder": ENCODER_CONFIG,
    "depthformer": DEPTH_CONFIG,
}

AUDIO_TOKEN = CONFIG["audio_token_id"]


@pytest.fixture(autouse=True)
def _fresh_patch_state():
    """``_APPLIED`` is process-wide; leaking it would hide a broken import."""
    mlx_vlm_lfm3_audio_compat._APPLIED = False
    yield


@pytest.fixture(scope="module", autouse=True)
def _leave_the_pinned_wheel_as_it_was():
    """Hand ``mlx_vlm`` back exactly as the pinned wheel ships it.

    One process runs every test file, and each sibling patch test starts from
    ``Model type lfm3_* not supported``. Leaving this patch's vendor path on
    ``mlx_vlm.models.__path__`` would make the VL file's assertion fail
    whenever this file runs first.
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

    mlx_vlm_lfm3_audio_compat._APPLIED = False
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
    assert is_lfm3_audio({"model_type": "lfm3_audio"})
    assert not is_lfm3_audio({"model_type": "lfm3_vl"})
    assert not is_lfm3_audio({})
    assert not is_lfm3_audio(None)


def test_a_missing_text_tower_vendor_dir_is_named(monkeypatch, caplog):
    """Without ``lfm3`` the audio model cannot import; say which patch is missing."""
    monkeypatch.setattr(
        mlx_vlm_lfm3_audio_compat,
        "_TEXT_TOWER_VENDOR",
        mlx_vlm_lfm3_audio_compat._VENDOR_MLX_VLM / "absent",
    )
    assert not apply_mlx_vlm_lfm3_audio_compat_patch()
    assert not mlx_vlm_lfm3_audio_compat.is_applied()
    assert "mlx_vlm_lfm3_vl_compat" in caplog.text


def test_the_architecture_resolves_through_mlx_vlm():
    """This is the exact call that failed with ``lfm3_audio not supported``."""
    import mlx_vlm.utils as vlm_utils

    with pytest.raises(ValueError, match="lfm3_audio not supported"):
        vlm_utils.get_model_and_args(CONFIG)

    assert apply_mlx_vlm_lfm3_audio_compat_patch()

    module, model_type = vlm_utils.get_model_and_args(CONFIG)
    assert model_type == "lfm3_audio"
    assert module is importlib.import_module("mlx_vlm.models.lfm3_audio")
    assert module.__name__.startswith("mlx_vlm.models.")
    # The text tower the audio model imports relatively must resolve too.
    assert importlib.import_module("mlx_vlm.models.lfm3").Model is not None


def test_applying_twice_is_a_no_op():
    assert apply_mlx_vlm_lfm3_audio_compat_patch()
    assert not apply_mlx_vlm_lfm3_audio_compat_patch()
    assert mlx_vlm_lfm3_audio_compat.is_applied()


def test_prompt_formatting_and_processor_are_registered():
    import mlx_vlm.prompt_utils as prompt_utils
    from transformers import AutoProcessor

    before = AutoProcessor.__dict__.get("from_pretrained")
    assert apply_mlx_vlm_lfm3_audio_compat_patch()
    assert "lfm3_audio" in prompt_utils.MODEL_CONFIG
    assert (
        prompt_utils.MODEL_CONFIG["lfm3_audio"]
        == prompt_utils.MessageFormat.LIST_WITH_IMAGE_FIRST
    )
    # The processor class is not resolvable by name yet, so transformers is
    # intercepted before it falls back to the pytorch references.
    assert AutoProcessor.__dict__.get("from_pretrained") is not before


def test_a_tiny_checkpoint_runs_text_and_audio_forwards():
    assert apply_mlx_vlm_lfm3_audio_compat_patch()

    module = importlib.import_module("mlx_vlm.models.lfm3_audio")
    model = module.Model(module.ModelConfig.from_dict(CONFIG))

    text_only = model(mx.array([[1, 2, 3]]))
    assert text_only.logits.shape == (1, 3, TEXT_CONFIG["vocab_size"])

    # One 4000-sample clip: 24 mel frames, three encoder tokens after the
    # strides, so three placeholders. A 400-sample minimum is padded to 480.
    processing = importlib.import_module(
        "mlx_vlm.models.lfm3_audio.processing_lfm3_audio"
    )
    clip = (0.2 * np.sin(np.arange(4000) * 0.05)).astype(np.float32)
    texts, features = processing.Lfm3AudioProcessor.expand_speech(["<|audio|>"], [clip])
    assert texts == ["<|audio|>" * 3]
    tokens = [[1] + [AUDIO_TOKEN] * 3 + [2]]
    out = model(
        mx.array(tokens),
        input_features=mx.array(features["input_features"]),
        feature_attention_mask=mx.array(features["feature_attention_mask"]),
    )
    assert out.logits.shape == (1, 5, TEXT_CONFIG["vocab_size"])


def test_the_dispatch_hook_fires_only_for_the_vlm_engine(tmp_path):
    """The text loader must not register an omni architecture."""
    import json

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    (tmp_path / "config.json").write_text(json.dumps(CONFIG))

    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=False)
    assert not mlx_vlm_lfm3_audio_compat.is_applied()

    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)
    assert mlx_vlm_lfm3_audio_compat.is_applied()
