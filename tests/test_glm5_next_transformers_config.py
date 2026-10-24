# SPDX-License-Identifier: Apache-2.0
"""Config-only GLM-5.3 checkpoint compatibility; no weights or inference."""

import copy
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from huggingface_hub.errors import StrictDataclassClassValidationError
from transformers import AutoConfig

from omlx.patches import mlx_vlm_glm5_next_compat as compat


@pytest.fixture
def config_module(monkeypatch):
    module = pytest.importorskip(
        "transformers.models.glm5_next.configuration_glm5_next"
    )
    # Restore the real class after each test; no Transformers behavior is mocked.
    cls = module.Glm5NextTextConfig
    monkeypatch.setattr(cls, "__post_init__", cls.__post_init__)
    return module


@pytest.fixture
def patched_config(config_module):
    compat._accept_mtp_tail_in_transformers_config()
    return config_module


def test_mtp_tail_load_preserves_checkpoint_config(tmp_path, patched_config):
    # Shape of Vontra/GLM-5.3-Flash-MLX-oQ4-MTP: 45 backbone + 1 MTP.
    payload = {
        "model_type": "glm5_next",
        "text_config": {
            "num_hidden_layers": 45,
            "num_nextn_predict_layers": 1,
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 43,
        },
    }
    original = copy.deepcopy(payload)
    checkpoint = tmp_path / "config.json"
    checkpoint.write_text(json.dumps(payload))
    original_bytes = checkpoint.read_bytes()

    direct = patched_config.Glm5NextConfig(**payload)
    loaded = AutoConfig.from_pretrained(tmp_path, local_files_only=True)

    expected = ["dense"] * 3 + ["sparse"] * 42
    assert direct.text_config.mlp_layer_types == expected
    assert loaded.text_config.mlp_layer_types == expected
    assert loaded.text_config.num_nextn_predict_layers == 1
    assert payload == original
    assert checkpoint.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "entries,nextn",
    [(47, 1), (46, 0), (46, None), (46, -1), (46, 1.5), (46, True), (46, "1")],
)
def test_invalid_mlp_length_is_still_rejected(tmp_path, patched_config, entries, nextn):
    text = {
        "num_hidden_layers": 45,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (entries - 3),
    }
    if nextn is not None:
        text["num_nextn_predict_layers"] = nextn
    checkpoint = tmp_path / "config.json"
    checkpoint.write_text(json.dumps({"model_type": "glm5_next", "text_config": text}))
    with pytest.raises(StrictDataclassClassValidationError, match="mlp_layer_types"):
        AutoConfig.from_pretrained(tmp_path, local_files_only=True)


@pytest.mark.parametrize("nextn", [0, 1])
def test_backbone_only_config_is_unchanged(patched_config, nextn):
    # Official zai-org configs already list only the 45 backbone MLP layers.
    layers = ["dense"] * 3 + ["sparse"] * 42
    config = patched_config.Glm5NextTextConfig(
        num_hidden_layers=45,
        num_nextn_predict_layers=nextn,
        mlp_layer_types=layers,
    )
    assert config.mlp_layer_types == layers
    assert len(layers) == 45
    assert config.num_nextn_predict_layers == nextn


def test_default_mlp_schedule_is_unchanged(patched_config):
    config = patched_config.Glm5NextTextConfig(
        num_hidden_layers=45, num_nextn_predict_layers=1
    )
    assert config.mlp_layer_types == ["dense"] * 3 + ["sparse"] * 42


def test_repeated_patch_does_not_stack_wrappers(patched_config):
    wrapped = patched_config.Glm5NextTextConfig.__post_init__
    assert compat._accept_mtp_tail_in_transformers_config() is False
    assert compat._accept_mtp_tail_in_transformers_config() is False
    assert patched_config.Glm5NextTextConfig.__post_init__ is wrapped


def test_missing_transformers_model_is_a_noop():
    if importlib.util.find_spec("transformers.models.glm5_next") is not None:
        pytest.skip("requires a Transformers release without glm5_next")
    assert compat._accept_mtp_tail_in_transformers_config() is False


def test_registration_preserves_vendored_model_config():
    # Actual mlx-vlm registration requires Apple Silicon. Config-only checks
    # above can run without MLX using pytest --noconftest.
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    # Registration reloads GLM modules. Reapplying it in the pytest worker would
    # discard classes patched by the independent, process-wide MTP runtime.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import copy

                from transformers.models.glm5_next import configuration_glm5_next
                from omlx.patches import mlx_vlm_glm5_next_compat as compat

                assert compat.apply_mlx_vlm_glm5_next_compat_patch() is True
                wrapped = configuration_glm5_next.Glm5NextTextConfig.__post_init__
                assert compat.apply_mlx_vlm_glm5_next_compat_patch() is False
                assert configuration_glm5_next.Glm5NextTextConfig.__post_init__ is wrapped

                from mlx_vlm.models.glm5_next import TextConfig

                text = configuration_glm5_next.Glm5NextTextConfig().to_dict()
                text.update(
                    num_nextn_predict_layers=1,
                    mlp_layer_types=["dense"] * 3 + ["sparse"] * 43,
                    first_k_dense_replace=3,
                    linear_attn_config={},
                )
                original = copy.deepcopy(text)
                vendor_config = TextConfig.from_dict(text)
                config = configuration_glm5_next.Glm5NextTextConfig(**text)
                assert len(config.mlp_layer_types) == 45
                assert vendor_config.mlp_layer_types == ["dense"] * 3 + ["sparse"] * 43
                assert text == original
                """),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
