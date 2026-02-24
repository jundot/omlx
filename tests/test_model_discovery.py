# SPDX-License-Identifier: Apache-2.0
"""Tests for model discovery functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from omlx.model_discovery import (
    DiscoveredModel,
    detect_model_type,
    discover_models,
    discover_models_from_dirs,
    estimate_model_size,
    format_size,
)


class TestDetectModelType:
    """Tests for detect_model_type function."""

    def test_detect_llm_model(self, tmp_path):
        """Test detection of LLM model."""
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_qwen_model(self, tmp_path):
        """Test detection of Qwen model as LLM."""
        config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_embedding_model_by_type(self, tmp_path):
        """Test detection of embedding model by model_type."""
        config = {
            "model_type": "bert",
            "architectures": ["BertModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_embedding_model_by_architecture(self, tmp_path):
        """Test detection of embedding model by architecture."""
        config = {
            "model_type": "unknown",
            "architectures": ["XLMRobertaModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_modernbert_embedding(self, tmp_path):
        """Test detection of ModernBERT as embedding model."""
        config = {
            "model_type": "modernbert",
            "architectures": ["ModernBertModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_reranker_model(self, tmp_path):
        """Test detection of reranker model by architecture."""
        config = {
            "model_type": "modernbert",
            "architectures": ["ModernBertForSequenceClassification"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "reranker"

    def test_detect_xlm_roberta_reranker(self, tmp_path):
        """Test detection of XLM-RoBERTa reranker."""
        config = {
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaForSequenceClassification"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "reranker"

    def test_missing_config_defaults_to_llm(self, tmp_path):
        """Test that missing config.json defaults to LLM."""
        assert detect_model_type(tmp_path) == "llm"

    def test_invalid_json_defaults_to_llm(self, tmp_path):
        """Test that invalid JSON defaults to LLM."""
        (tmp_path / "config.json").write_text("not valid json")
        assert detect_model_type(tmp_path) == "llm"

    def test_empty_config_defaults_to_llm(self, tmp_path):
        """Test that empty config defaults to LLM."""
        (tmp_path / "config.json").write_text("{}")
        assert detect_model_type(tmp_path) == "llm"


class TestEstimateModelSize:
    """Tests for estimate_model_size function."""

    def test_estimate_from_safetensors(self, tmp_path):
        """Test size estimation from safetensors files."""
        # Create dummy safetensors files
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"0" * 1000)
        (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"0" * 2000)

        size = estimate_model_size(tmp_path)
        # 3000 bytes + 5% overhead
        assert size == int(3000 * 1.05)

    def test_estimate_from_single_safetensors(self, tmp_path):
        """Test size estimation from single safetensors file."""
        (tmp_path / "model.safetensors").write_bytes(b"0" * 5000)

        size = estimate_model_size(tmp_path)
        assert size == int(5000 * 1.05)

    def test_estimate_from_bin_files(self, tmp_path):
        """Test size estimation from .bin files when no safetensors."""
        # Create dummy bin files
        (tmp_path / "pytorch_model.bin").write_bytes(b"0" * 5000)

        size = estimate_model_size(tmp_path)
        # 5000 bytes + 5% overhead
        assert size == int(5000 * 1.05)

    def test_skip_optimizer_files(self, tmp_path):
        """Test that optimizer files are skipped."""
        (tmp_path / "model.bin").write_bytes(b"0" * 1000)
        (tmp_path / "optimizer.bin").write_bytes(b"0" * 2000)

        size = estimate_model_size(tmp_path)
        # Only model.bin (1000 bytes) + 5% overhead
        assert size == int(1000 * 1.05)

    def test_skip_training_files(self, tmp_path):
        """Test that training files are skipped."""
        (tmp_path / "model.bin").write_bytes(b"0" * 1000)
        (tmp_path / "training_args.bin").write_bytes(b"0" * 500)

        size = estimate_model_size(tmp_path)
        assert size == int(1000 * 1.05)

    def test_no_weights_raises_error(self, tmp_path):
        """Test that missing weights raises ValueError."""
        with pytest.raises(ValueError, match="No model weights found"):
            estimate_model_size(tmp_path)

    def test_nested_safetensors(self, tmp_path):
        """Test size estimation from nested safetensors."""
        # Create subdirectory with safetensors
        subdir = tmp_path / "weights"
        subdir.mkdir()
        (subdir / "model.safetensors").write_bytes(b"0" * 3000)

        size = estimate_model_size(tmp_path)
        assert size == int(3000 * 1.05)

    def test_prefers_safetensors_over_bin(self, tmp_path):
        """Test that safetensors are preferred over bin files."""
        (tmp_path / "model.safetensors").write_bytes(b"0" * 1000)
        (tmp_path / "model.bin").write_bytes(b"0" * 5000)

        size = estimate_model_size(tmp_path)
        # Should use safetensors size only
        assert size == int(1000 * 1.05)


class TestDiscoverModels:
    """Tests for discover_models function."""

    def test_discover_single_model(self, tmp_path):
        """Test discovery of a single model."""
        model_dir = tmp_path / "llama-3b"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "llama-3b" in models
        assert models["llama-3b"].model_type == "llm"
        assert models["llama-3b"].engine_type == "batched"

    def test_discover_multiple_models(self, tmp_path):
        """Test discovery of multiple models."""
        # Create first LLM model
        llm_dir = tmp_path / "llama-3b"
        llm_dir.mkdir()
        (llm_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (llm_dir / "model.safetensors").write_bytes(b"0" * 1000)

        # Create second LLM model
        llm2_dir = tmp_path / "qwen-7b"
        llm2_dir.mkdir()
        (llm2_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        (llm2_dir / "model.safetensors").write_bytes(b"0" * 2000)

        models = discover_models(tmp_path)
        assert len(models) == 2
        assert models["llama-3b"].engine_type == "batched"
        assert models["qwen-7b"].engine_type == "batched"

    def test_discover_embedding_model(self, tmp_path):
        """Test discovery of embedding model with correct engine type."""
        emb_dir = tmp_path / "bge-small"
        emb_dir.mkdir()
        (emb_dir / "config.json").write_text(
            json.dumps({"model_type": "bert", "architectures": ["BertModel"]})
        )
        (emb_dir / "model.safetensors").write_bytes(b"0" * 500)

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert models["bge-small"].model_type == "embedding"
        assert models["bge-small"].engine_type == "embedding"

    def test_discover_reranker_model(self, tmp_path):
        """Test discovery of reranker model with correct engine type."""
        reranker_dir = tmp_path / "bge-reranker"
        reranker_dir.mkdir()
        (reranker_dir / "config.json").write_text(
            json.dumps({
                "model_type": "modernbert",
                "architectures": ["ModernBertForSequenceClassification"]
            })
        )
        (reranker_dir / "model.safetensors").write_bytes(b"0" * 500)

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert models["bge-reranker"].model_type == "reranker"
        assert models["bge-reranker"].engine_type == "reranker"

    def test_skip_invalid_directories(self, tmp_path):
        """Test that directories without config.json are skipped."""
        # Valid model
        valid_dir = tmp_path / "valid-model"
        valid_dir.mkdir()
        (valid_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (valid_dir / "model.safetensors").write_bytes(b"0" * 1000)

        # Invalid model (no config.json)
        invalid_dir = tmp_path / "invalid-model"
        invalid_dir.mkdir()
        (invalid_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "valid-model" in models
        assert "invalid-model" not in models

    def test_skip_hidden_directories(self, tmp_path):
        """Test that hidden directories are skipped."""
        # Hidden directory
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (hidden_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert len(models) == 0

    def test_skip_files(self, tmp_path):
        """Test that files are skipped (only directories processed)."""
        # Create a file at top level
        (tmp_path / "README.md").write_text("readme")

        # Create valid model
        model_dir = tmp_path / "llama-3b"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert len(models) == 1

    def test_nonexistent_directory_raises_error(self, tmp_path):
        """Test that nonexistent directory raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            discover_models(tmp_path / "nonexistent")

    def test_file_instead_of_directory_raises_error(self, tmp_path):
        """Test that file path raises ValueError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        with pytest.raises(ValueError, match="not a directory"):
            discover_models(file_path)

    def test_model_with_weight_error_skipped(self, tmp_path):
        """Test that models with no weights are skipped."""
        # Model with config but no weights
        model_dir = tmp_path / "no-weights"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

        models = discover_models(tmp_path)
        assert len(models) == 0

    def test_discovered_model_fields(self, tmp_path):
        """Test that DiscoveredModel has all expected fields."""
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        model = models["test-model"]

        assert model.model_id == "test-model"
        assert model.model_path == str(model_dir)
        assert model.model_type == "llm"
        assert model.engine_type == "batched"
        assert model.estimated_size == int(1000 * 1.05)


class TestFormatSize:
    """Tests for format_size function."""

    def test_format_bytes(self):
        """Test formatting bytes."""
        assert format_size(500) == "500.00B"

    def test_format_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_size(1024) == "1.00KB"
        assert format_size(2048) == "2.00KB"

    def test_format_megabytes(self):
        """Test formatting megabytes."""
        assert format_size(1024 * 1024) == "1.00MB"
        assert format_size(5 * 1024 * 1024) == "5.00MB"

    def test_format_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_size(1024 * 1024 * 1024) == "1.00GB"
        assert format_size(32 * 1024 * 1024 * 1024) == "32.00GB"

    def test_format_terabytes(self):
        """Test formatting terabytes."""
        assert format_size(1024 * 1024 * 1024 * 1024) == "1.00TB"

    def test_format_petabytes(self):
        """Test formatting petabytes."""
        assert format_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00PB"


class TestDiscoveredModel:
    """Tests for DiscoveredModel dataclass."""

    def test_create_discovered_model(self):
        """Test creating a DiscoveredModel."""
        model = DiscoveredModel(
            model_id="test-model",
            model_path="/path/to/model",
            model_type="llm",
            engine_type="batched",
            estimated_size=1024 * 1024 * 1024,  # 1GB
        )

        assert model.model_id == "test-model"
        assert model.model_path == "/path/to/model"
        assert model.model_type == "llm"
        assert model.engine_type == "batched"
        assert model.estimated_size == 1024 * 1024 * 1024

    def test_discovered_model_embedding(self):
        """Test DiscoveredModel for embedding type."""
        model = DiscoveredModel(
            model_id="emb-model",
            model_path="/path/to/emb",
            model_type="embedding",
            engine_type="embedding",
            estimated_size=500 * 1024 * 1024,
        )

        assert model.model_type == "embedding"
        assert model.engine_type == "embedding"


class TestTwoLevelDiscovery:
    """Tests for two-level model discovery."""

    def _make_model(self, path: Path, model_type: str = "llama"):
        """Helper to create a valid model directory."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"model_type": model_type}))
        (path / "model.safetensors").write_bytes(b"0" * 1000)

    def test_two_level_org_folder(self, tmp_path):
        """Test discovery through organization folders."""
        self._make_model(tmp_path / "mlx-community" / "llama-3b")
        self._make_model(tmp_path / "mlx-community" / "qwen-7b")

        models = discover_models(tmp_path)
        assert len(models) == 2
        assert "llama-3b" in models
        assert "qwen-7b" in models

    def test_mixed_flat_and_org(self, tmp_path):
        """Test mix of flat models and organization folders."""
        # Flat model at level 1
        self._make_model(tmp_path / "mistral-7b")
        # Org folder with models at level 2
        self._make_model(tmp_path / "Qwen" / "Qwen3-8B")

        models = discover_models(tmp_path)
        assert len(models) == 2
        assert "mistral-7b" in models
        assert "Qwen3-8B" in models

    def test_multiple_org_folders(self, tmp_path):
        """Test multiple organization folders."""
        self._make_model(tmp_path / "mlx-community" / "llama-3b")
        self._make_model(tmp_path / "Qwen" / "Qwen3-8B")
        self._make_model(tmp_path / "GLM" / "glm-4")

        models = discover_models(tmp_path)
        assert len(models) == 3

    def test_empty_org_folder_skipped(self, tmp_path):
        """Test that empty org folders are silently skipped."""
        self._make_model(tmp_path / "valid-model")
        (tmp_path / "empty-org").mkdir()

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "valid-model" in models

    def test_org_folder_hidden_children_skipped(self, tmp_path):
        """Test that hidden subdirs inside org folders are skipped."""
        org = tmp_path / "mlx-community"
        self._make_model(org / "llama-3b")
        self._make_model(org / ".hidden-model")

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "llama-3b" in models

    def test_org_folder_invalid_children_skipped(self, tmp_path):
        """Test that children without config.json in org folders are skipped."""
        org = tmp_path / "mlx-community"
        self._make_model(org / "llama-3b")
        # Child without config.json
        no_config = org / "broken-model"
        no_config.mkdir(parents=True)
        (no_config / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert len(models) == 1

    def test_two_level_model_path_is_correct(self, tmp_path):
        """Test that model_path points to the actual model dir, not the org."""
        self._make_model(tmp_path / "mlx-community" / "llama-3b")

        models = discover_models(tmp_path)
        assert models["llama-3b"].model_path == str(
            tmp_path / "mlx-community" / "llama-3b"
        )


class TestDiscoverModelsFromDirs:
    """Tests for discover_models_from_dirs function."""

    def _make_model(self, path, model_type="llama"):
        """Helper to create a mock model directory."""
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "model_type": model_type,
            "architectures": ["LlamaForCausalLM"],
        }
        (path / "config.json").write_text(json.dumps(config))
        # Create a small safetensors file
        (path / "model.safetensors").write_bytes(b"\x00" * 100)

    def test_multiple_dirs(self, tmp_path):
        """Test discovering models from multiple directories."""
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        self._make_model(dir_a / "model-1")
        self._make_model(dir_b / "model-2")

        models = discover_models_from_dirs([dir_a, dir_b])
        assert "model-1" in models
        assert "model-2" in models
        assert len(models) == 2

    def test_first_directory_wins_on_conflict(self, tmp_path):
        """Test that first directory takes priority on model_id conflicts."""
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        self._make_model(dir_a / "same-model")
        self._make_model(dir_b / "same-model")

        models = discover_models_from_dirs([dir_a, dir_b])
        assert len(models) == 1
        assert models["same-model"].model_path == str(dir_a / "same-model")

    def test_empty_list(self, tmp_path):
        """Test with empty directory list."""
        models = discover_models_from_dirs([])
        assert models == {}

    def test_nonexistent_directory_skipped(self, tmp_path):
        """Test that non-existent directories are skipped with warning."""
        dir_a = tmp_path / "dir_a"
        self._make_model(dir_a / "model-1")
        nonexistent = tmp_path / "does_not_exist"

        models = discover_models_from_dirs([dir_a, nonexistent])
        assert "model-1" in models
        assert len(models) == 1

    def test_mixed_valid_invalid_dirs(self, tmp_path):
        """Test with a mix of valid, empty, and non-existent directories."""
        dir_valid = tmp_path / "valid"
        dir_empty = tmp_path / "empty"
        dir_empty.mkdir(parents=True)
        self._make_model(dir_valid / "model-1")

        models = discover_models_from_dirs(
            [dir_valid, dir_empty, tmp_path / "nonexistent"]
        )
        assert "model-1" in models
        assert len(models) == 1
