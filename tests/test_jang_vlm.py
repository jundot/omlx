# SPDX-License-Identifier: Apache-2.0
"""Tests for JANG VLM (Vision-Language Model) detection and loading.

Tests cover:
- JANG VLM detection via jang_config.architecture.has_vision
- JANG VLM detection via preprocessor_config.json
- JANG VLM detection via config.json.vision_config
- VLM model type detection in detect_model_type()
- Engine type selection for JANG VLM models
"""

import json
import tempfile
from pathlib import Path

import pytest

from omlx.engine.jang import JANGLoader
from omlx.model_discovery import detect_model_type


class TestJANGLoaderVlmDetection:
    """Tests for JANGLoader._is_vlm_model() detection logic."""

    def test_is_vlm_via_has_vision(self, tmp_path):
        """Test VLM detection via jang_config.architecture.has_vision."""
        # Create jang_config.json with has_vision=true
        jang_config = {
            "version": "1.0",
            "architecture": {
                "has_vision": True,
                "has_moe": False,
                "has_ssm": False,
            },
        }
        (tmp_path / "jang_config.json").write_text(json.dumps(jang_config))

        loader = JANGLoader(str(tmp_path))
        assert loader._is_vlm_model() is True

    def test_is_vlm_via_has_vision_false(self, tmp_path):
        """Test non-VLM detection via jang_config.architecture.has_vision=false."""
        jang_config = {
            "version": "1.0",
            "architecture": {
                "has_vision": False,
                "has_moe": True,
                "has_ssm": True,
            },
        }
        (tmp_path / "jang_config.json").write_text(json.dumps(jang_config))

        loader = JANGLoader(str(tmp_path))
        assert loader._is_vlm_model() is False

    def test_is_vlm_via_preprocessor_config(self, tmp_path):
        """Test VLM detection via preprocessor_config.json existence."""
        # Create preprocessor_config.json (common in VLMs)
        preprocessor_config = {
            "processor_type": "AutoProcessor",
            "vision_feature_extractor_type": "CLIPImageProcessor",
        }
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(preprocessor_config))

        loader = JANGLoader(str(tmp_path))
        assert loader._is_vlm_model() is True

    def test_is_vlm_via_vision_config(self, tmp_path):
        """Test VLM detection via config.json.vision_config."""
        config = {
            "model_type": "some_vlm",
            "vision_config": {"hidden_size": 1024},
            "text_config": {"hidden_size": 2048},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        loader = JANGLoader(str(tmp_path))
        assert loader._is_vlm_model() is True

    def test_is_vlm_vlm_architecture_in_config(self, tmp_path):
        """Test VLM detection via VLM architecture in config.json."""
        config = {
            "model_type": "qwen2_vl",
            "architectures": ["Qwen2VLForConditionalGeneration"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        loader = JANGLoader(str(tmp_path))
        assert loader._is_vlm_model() is True

    def test_is_not_vlm_no_indicators(self, tmp_path):
        """Test non-VLM detection when no indicators present."""
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        loader = JANGLoader(str(tmp_path))
        assert loader._is_vlm_model() is False

    def test_is_vlm_multiple_indicators(self, tmp_path):
        """Test VLM detection with multiple indicators present."""
        jang_config = {
            "version": "1.0",
            "architecture": {"has_vision": True},
        }
        (tmp_path / "jang_config.json").write_text(json.dumps(jang_config))
        (tmp_path / "preprocessor_config.json").write_text("{}")
        config = {"vision_config": {}}
        (tmp_path / "config.json").write_text(json.dumps(config))

        loader = JANGLoader(str(tmp_path))
        assert loader._is_vlm_model() is True


class TestDetectModelTypeJangVlm:
    """Tests for detect_model_type() with JANG VLM indicators."""

    def test_detect_vlm_via_jang_has_vision(self, tmp_path):
        """Test VLM detection via jang_config.architecture.has_vision in model_discovery."""
        jang_config = {
            "version": "1.0",
            "architecture": {"has_vision": True},
        }
        (tmp_path / "jang_config.json").write_text(json.dumps(jang_config))
        # config.json without vision_config should still be detected as VLM
        config = {"model_type": "jang_vlm"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_vlm_via_preprocessor_config(self, tmp_path):
        """Test VLM detection via preprocessor_config.json in model_discovery."""
        preprocessor_config = {
            "processor_type": "AutoProcessor",
        }
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(preprocessor_config))

        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_vlm_via_vision_config(self, tmp_path):
        """Test VLM detection via config.json.vision_config in model_discovery."""
        config = {
            "model_type": "jang_vlm",
            "vision_config": {"hidden_size": 1024},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_text_only_jang_as_llm(self, tmp_path):
        """Test text-only JANG model (no VLM indicators) is LLM."""
        config = {
            "model_type": "jang_llm",
            "architectures": ["JANGForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        assert detect_model_type(tmp_path) == "llm"

    def test_jang_has_vision_takes_precedence(self, tmp_path):
        """Test jang_config.has_vision takes precedence over other config."""
        # has_vision=False but preprocessor_config exists
        jang_config = {
            "version": "1.0",
            "architecture": {"has_vision": False},
        }
        (tmp_path / "jang_config.json").write_text(json.dumps(jang_config))
        preprocessor_config = {"processor_type": "AutoProcessor"}
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(preprocessor_config))

        # The jang_config.json.architecture.has_vision is checked first in detect_model_type
        # But now JANG models always use "jang" engine type regardless of model_type
        # has_vision=False in jang_config means it's a text-only JANG model
        assert detect_model_type(tmp_path) == "llm"


class TestJANGEngineTypeSelection:
    """Tests for engine type selection based on JANG VLM detection."""

    def test_jang_vlm_engine_type(self, tmp_path):
        """Test that JANG VLM models use jang engine type."""
        jang_config = {
            "version": "1.0",
            "architecture": {"has_vision": True},
        }
        (tmp_path / "jang_config.json").write_text(json.dumps(jang_config))
        # Create a minimal weight file for size estimation
        weight_file = tmp_path / "model-00001-of-00001.safetensors"
        weight_file.write_bytes(b"dummy")

        from omlx.model_discovery import discover_models

        models = discover_models(tmp_path)
        assert len(models) == 1
        model_id = list(models.keys())[0]
        assert models[model_id].model_type == "vlm"
        assert models[model_id].engine_type == "jang"

    def test_jang_llm_engine_type(self, tmp_path):
        """Test that non-JANG LLM models use batched engine type."""
        config = {
            "model_type": "jang_llm",
            "architectures": ["JANGForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        weight_file = tmp_path / "model-00001-of-00001.safetensors"
        weight_file.write_bytes(b"dummy")

        from omlx.model_discovery import discover_models

        models = discover_models(tmp_path)
        assert len(models) == 1
        model_id = list(models.keys())[0]
        assert models[model_id].model_type == "llm"
        assert models[model_id].engine_type == "batched"

    def test_jang_text_only_engine_type(self, tmp_path):
        """Test that text-only JANG models (with jang_config.json) use jang engine type."""
        jang_config = {
            "version": "1.0",
            "architecture": {
                "has_vision": False,
                "has_moe": True,
                "has_ssm": False,
            },
            "quantization": {
                "profile": "JANG_2L",
                "actual_bits": 2.31,
            },
        }
        (tmp_path / "jang_config.json").write_text(json.dumps(jang_config))
        config = {
            "model_type": "qwen2_moe",
            "architectures": ["Qwen2MoeForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        weight_file = tmp_path / "model-00001-of-00001.safetensors"
        weight_file.write_bytes(b"dummy")

        from omlx.model_discovery import discover_models

        models = discover_models(tmp_path)
        assert len(models) == 1
        model_id = list(models.keys())[0]
        assert models[model_id].model_type == "llm"
        assert models[model_id].engine_type == "jang"