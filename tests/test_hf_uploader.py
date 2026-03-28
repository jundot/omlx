# SPDX-License-Identifier: Apache-2.0
"""Tests for the HuggingFace model uploader."""

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin.hf_uploader import (
    HFUploader,
    UploadStatus,
    UploadTask,
    _format_size,
    _generate_model_card,
    _has_meaningful_readme,
    _is_oq_model,
)


# =============================================================================
# Helper Tests
# =============================================================================


class TestIsOqModel:
    """Test oQ model name detection."""

    def test_standard_oq_names(self):
        assert _is_oq_model("Qwen3.5-122B-oQ4") is True
        assert _is_oq_model("Llama-3B-oQ4e") is True
        assert _is_oq_model("Model-oQ3") is True
        assert _is_oq_model("Model-oQ8") is True

    def test_non_oq_names(self):
        assert _is_oq_model("Qwen3.5-122B") is False
        assert _is_oq_model("Llama-3B-4bit") is False
        assert _is_oq_model("oQ-model-name") is False  # oQ not in last 5

    def test_edge_cases(self):
        assert _is_oq_model("X-oQ2") is True
        assert _is_oq_model("oQ4") is True  # short name, still has oQ in last 5
        assert _is_oq_model("12oQ4") is True  # exactly 5 chars, oQ at pos 2
        assert _is_oq_model("ABCDE") is False


class TestHasMeaningfulReadme:
    """Test README content detection."""

    def test_no_readme(self, tmp_path):
        assert _has_meaningful_readme(tmp_path) is False

    def test_empty_readme(self, tmp_path):
        (tmp_path / "README.md").write_text("")
        assert _has_meaningful_readme(tmp_path) is False

    def test_frontmatter_only(self, tmp_path):
        (tmp_path / "README.md").write_text(
            "---\nlanguage: en\nlibrary_name: mlx\ntags:\n- mlx\n---\n"
        )
        assert _has_meaningful_readme(tmp_path) is False

    def test_frontmatter_with_body(self, tmp_path):
        (tmp_path / "README.md").write_text(
            "---\nlibrary_name: mlx\n---\n\n# My Model\nSome description.\n"
        )
        assert _has_meaningful_readme(tmp_path) is True

    def test_no_frontmatter(self, tmp_path):
        (tmp_path / "README.md").write_text("# My Model\nA great model.\n")
        assert _has_meaningful_readme(tmp_path) is True


class TestFormatSize:
    """Test size formatting."""

    def test_kb(self):
        assert _format_size(512 * 1024) == "512.0 KB"

    def test_mb(self):
        assert _format_size(100 * 1024**2) == "100.0 MB"

    def test_gb(self):
        assert _format_size(5 * 1024**3) == "5.0 GB"


class TestGenerateModelCard:
    """Test model card generation."""

    def test_basic_card(self):
        config = {
            "model_type": "qwen2",
            "quantization": {"bits": 4, "group_size": 64},
        }
        card = _generate_model_card("Qwen-7B-oQ4", config)
        assert "# Qwen-7B-oQ4" in card
        assert "library_name: mlx" in card
        assert "- oq" in card
        assert "qwen2" in card
        assert "4" in card

    def test_missing_quantization(self):
        config = {"model_type": "llama"}
        card = _generate_model_card("Model-oQ4", config)
        assert "# Model-oQ4" in card
        assert "?" in card  # missing bits


# =============================================================================
# UploadTask Tests
# =============================================================================


class TestUploadTask:
    """Test UploadTask dataclass."""

    def test_default_values(self):
        task = UploadTask(
            task_id="test-id",
            model_name="Model-oQ4",
            model_path="/models/Model-oQ4",
            repo_id="user/Model-oQ4",
        )
        assert task.task_id == "test-id"
        assert task.model_name == "Model-oQ4"
        assert task.status == UploadStatus.PENDING
        assert task.progress == 0.0
        assert task.error == ""
        assert task.repo_url == ""

    def test_to_dict(self):
        task = UploadTask(
            task_id="abc-123",
            model_name="Model-oQ4",
            model_path="/models/Model-oQ4",
            repo_id="user/Model-oQ4",
            status=UploadStatus.UPLOADING,
            progress=45.67,
            total_size=5 * 1024**3,
            created_at=1700000000.0,
        )
        d = task.to_dict()
        assert d["task_id"] == "abc-123"
        assert d["status"] == "uploading"
        assert d["progress"] == 45.7  # rounded
        assert d["total_size"] == 5 * 1024**3
        assert d["total_size_formatted"] == "5.0 GB"
        assert d["repo_url"] == ""

    def test_to_dict_completed(self):
        task = UploadTask(
            task_id="t",
            model_name="M",
            model_path="/m",
            repo_id="u/m",
            status=UploadStatus.COMPLETED,
            repo_url="https://huggingface.co/u/m",
        )
        d = task.to_dict()
        assert d["repo_url"] == "https://huggingface.co/u/m"


# =============================================================================
# HFUploader Tests
# =============================================================================


@pytest.fixture
def model_dirs(tmp_path):
    """Create temp model directories with oQ and non-oQ models."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # oQ model
    oq_model = model_dir / "Llama-3B-oQ4"
    oq_model.mkdir()
    (oq_model / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "quantization": {"bits": 4, "group_size": 64},
    }))
    # Create a fake safetensors file
    (oq_model / "model.safetensors").write_bytes(b"\x00" * 1024)

    # Another oQ model
    oq_model2 = model_dir / "Qwen-7B-oQ3"
    oq_model2.mkdir()
    (oq_model2 / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "quantization": {"bits": 3, "group_size": 64},
    }))
    (oq_model2 / "model.safetensors").write_bytes(b"\x00" * 2048)

    # Non-oQ model (has README)
    non_oq = model_dir / "Llama-3B-Instruct"
    non_oq.mkdir()
    (non_oq / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (non_oq / "model.safetensors").write_bytes(b"\x00" * 4096)
    (non_oq / "README.md").write_text("# Llama 3B Instruct\nA great model.")

    return [str(model_dir)]


class TestHFUploaderListModels:
    """Test model listing functionality."""

    @pytest.mark.asyncio
    async def test_list_oq_models(self, model_dirs):
        uploader = HFUploader(model_dirs=model_dirs)
        models = await uploader.list_oq_models()
        names = [m["name"] for m in models]
        assert "Llama-3B-oQ4" in names
        assert "Qwen-7B-oQ3" in names
        assert "Llama-3B-Instruct" not in names

    @pytest.mark.asyncio
    async def test_list_oq_models_has_size(self, model_dirs):
        uploader = HFUploader(model_dirs=model_dirs)
        models = await uploader.list_oq_models()
        for m in models:
            assert m["size"] > 0
            assert m["size_formatted"]

    @pytest.mark.asyncio
    async def test_list_all_models(self, model_dirs):
        uploader = HFUploader(model_dirs=model_dirs)
        models = await uploader.list_all_models()
        names = [m["name"] for m in models]
        assert "Llama-3B-oQ4" in names
        assert "Llama-3B-Instruct" in names

    @pytest.mark.asyncio
    async def test_list_all_models_has_readme(self, model_dirs):
        uploader = HFUploader(model_dirs=model_dirs)
        models = await uploader.list_all_models()
        instruct = next(m for m in models if m["name"] == "Llama-3B-Instruct")
        oq = next(m for m in models if m["name"] == "Llama-3B-oQ4")
        assert instruct["has_readme"] is True
        assert oq["has_readme"] is False

    @pytest.mark.asyncio
    async def test_empty_model_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        uploader = HFUploader(model_dirs=[str(empty_dir)])
        models = await uploader.list_oq_models()
        assert models == []


class TestHFUploaderValidateToken:
    """Test token validation."""

    @pytest.mark.asyncio
    async def test_valid_token(self):
        mock_info = {
            "name": "testuser",
            "orgs": [{"name": "myorg"}],
            "auth": {"accessToken": {"role": "write"}},
        }
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.whoami.return_value = mock_info
            result = await HFUploader.validate_token("hf_valid_token")
        assert result["username"] == "testuser"
        assert len(result["orgs"]) == 1
        assert result["orgs"][0]["name"] == "myorg"

    @pytest.mark.asyncio
    async def test_read_only_token(self):
        mock_info = {
            "name": "testuser",
            "orgs": [],
            "auth": {"accessToken": {"role": "read"}},
        }
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.whoami.return_value = mock_info
            with pytest.raises(ValueError, match="read-only"):
                await HFUploader.validate_token("hf_read_token")

    @pytest.mark.asyncio
    async def test_invalid_token(self):
        with patch("huggingface_hub.HfApi") as MockApi:
            MockApi.return_value.whoami.side_effect = Exception("Unauthorized")
            with pytest.raises(ValueError, match="Invalid token"):
                await HFUploader.validate_token("bad_token")


class TestHFUploaderTaskLifecycle:
    """Test upload task creation and management."""

    @pytest.mark.asyncio
    async def test_start_upload_invalid_path(self, model_dirs):
        uploader = HFUploader(model_dirs=model_dirs)
        with pytest.raises(ValueError, match="not found"):
            await uploader.start_upload(
                model_path="/nonexistent/path",
                repo_id="user/model",
                token="hf_token",
            )

    @pytest.mark.asyncio
    async def test_start_upload_invalid_repo_id(self, model_dirs):
        oq_path = str(Path(model_dirs[0]) / "Llama-3B-oQ4")
        uploader = HFUploader(model_dirs=model_dirs)
        with pytest.raises(ValueError, match="Invalid repository ID"):
            await uploader.start_upload(
                model_path=oq_path,
                repo_id="invalid-no-slash",
                token="hf_token",
            )

    @pytest.mark.asyncio
    async def test_start_upload_creates_task(self, model_dirs):
        oq_path = str(Path(model_dirs[0]) / "Llama-3B-oQ4")
        uploader = HFUploader(model_dirs=model_dirs)

        with patch("huggingface_hub.HfApi") as MockApi:
            mock_api = MockApi.return_value
            mock_api.create_repo.return_value = None
            mock_api.upload_folder.return_value = None

            task = await uploader.start_upload(
                model_path=oq_path,
                repo_id="user/Llama-3B-oQ4",
                token="hf_token",
            )

        assert task.model_name == "Llama-3B-oQ4"
        assert task.repo_id == "user/Llama-3B-oQ4"
        assert task.total_size > 0

        # Wait for async task to complete
        await asyncio.sleep(0.5)
        tasks = uploader.get_tasks()
        assert len(tasks) == 1

    @pytest.mark.asyncio
    async def test_duplicate_upload_rejected(self, model_dirs):
        oq_path = str(Path(model_dirs[0]) / "Llama-3B-oQ4")
        uploader = HFUploader(model_dirs=model_dirs)

        # Patch to prevent actual upload
        with patch("huggingface_hub.HfApi") as MockApi:
            mock_api = MockApi.return_value
            # Make upload_folder block
            mock_api.create_repo.return_value = None
            future = asyncio.get_event_loop().create_future()
            mock_api.upload_folder.side_effect = lambda **kwargs: future

            await uploader.start_upload(
                model_path=oq_path,
                repo_id="user/Llama-3B-oQ4",
                token="hf_token",
            )
            with pytest.raises(ValueError, match="already in progress"):
                await uploader.start_upload(
                    model_path=oq_path,
                    repo_id="user/Llama-3B-oQ4",
                    token="hf_token",
                )
            # Cleanup
            await uploader.shutdown()

    @pytest.mark.asyncio
    async def test_cancel_upload(self, model_dirs):
        oq_path = str(Path(model_dirs[0]) / "Llama-3B-oQ4")
        uploader = HFUploader(model_dirs=model_dirs)

        with patch("huggingface_hub.HfApi") as MockApi:
            mock_api = MockApi.return_value
            mock_api.create_repo.return_value = None
            future = asyncio.get_event_loop().create_future()
            mock_api.upload_folder.side_effect = lambda **kwargs: future

            task = await uploader.start_upload(
                model_path=oq_path,
                repo_id="user/Llama-3B-oQ4",
                token="hf_token",
            )
            result = await uploader.cancel_upload(task.task_id)
            assert result is True
            assert task.status == UploadStatus.CANCELLED

            # Cleanup
            await uploader.shutdown()

    @pytest.mark.asyncio
    async def test_remove_completed_task(self, model_dirs):
        oq_path = str(Path(model_dirs[0]) / "Llama-3B-oQ4")
        uploader = HFUploader(model_dirs=model_dirs)

        with patch("huggingface_hub.HfApi") as MockApi:
            mock_api = MockApi.return_value
            mock_api.create_repo.return_value = None
            mock_api.upload_folder.return_value = None

            task = await uploader.start_upload(
                model_path=oq_path,
                repo_id="user/Llama-3B-oQ4",
                token="hf_token",
            )
            # Wait for completion
            await asyncio.sleep(0.5)
            assert uploader.remove_task(task.task_id) is True
            assert uploader.get_tasks() == []

    @pytest.mark.asyncio
    async def test_remove_active_task_fails(self, model_dirs):
        oq_path = str(Path(model_dirs[0]) / "Llama-3B-oQ4")
        uploader = HFUploader(model_dirs=model_dirs)

        with patch("huggingface_hub.HfApi") as MockApi:
            mock_api = MockApi.return_value
            mock_api.create_repo.return_value = None
            future = asyncio.get_event_loop().create_future()
            mock_api.upload_folder.side_effect = lambda **kwargs: future

            task = await uploader.start_upload(
                model_path=oq_path,
                repo_id="user/Llama-3B-oQ4",
                token="hf_token",
            )
            assert uploader.remove_task(task.task_id) is False
            await uploader.shutdown()

    @pytest.mark.asyncio
    async def test_get_tasks_ordered_by_creation(self, model_dirs):
        uploader = HFUploader(model_dirs=model_dirs)
        oq_path1 = str(Path(model_dirs[0]) / "Llama-3B-oQ4")
        oq_path2 = str(Path(model_dirs[0]) / "Qwen-7B-oQ3")

        with patch("huggingface_hub.HfApi") as MockApi:
            mock_api = MockApi.return_value
            mock_api.create_repo.return_value = None
            mock_api.upload_folder.return_value = None

            await uploader.start_upload(
                model_path=oq_path1,
                repo_id="user/Llama-3B-oQ4",
                token="hf_token",
            )
            await asyncio.sleep(0.01)
            await uploader.start_upload(
                model_path=oq_path2,
                repo_id="user/Qwen-7B-oQ3",
                token="hf_token",
            )
            await asyncio.sleep(0.5)
            tasks = uploader.get_tasks()
            assert len(tasks) == 2
            assert tasks[0]["model_name"] == "Llama-3B-oQ4"
            assert tasks[1]["model_name"] == "Qwen-7B-oQ3"


class TestHFUploaderReadme:
    """Test README handling during upload."""

    @pytest.mark.asyncio
    async def test_auto_readme_created(self, model_dirs):
        """Auto-generated README should be created and cleaned up."""
        oq_path = Path(model_dirs[0]) / "Llama-3B-oQ4"
        uploader = HFUploader(model_dirs=model_dirs)

        with patch("huggingface_hub.HfApi") as MockApi:
            mock_api = MockApi.return_value
            mock_api.create_repo.return_value = None

            # Capture what upload_folder receives
            uploaded_files = []
            def fake_upload(**kwargs):
                folder = Path(kwargs["folder_path"])
                uploaded_files.extend([f.name for f in folder.iterdir()])

            mock_api.upload_folder.side_effect = fake_upload

            await uploader.start_upload(
                model_path=str(oq_path),
                repo_id="user/Llama-3B-oQ4",
                token="hf_token",
                auto_readme=True,
            )
            await asyncio.sleep(0.5)

            # README should have been present during upload
            assert "README.md" in uploaded_files
            # But cleaned up after
            assert not (oq_path / "README.md").exists()

    @pytest.mark.asyncio
    async def test_readme_from_source(self, model_dirs):
        """README should be copied from source model."""
        oq_path = Path(model_dirs[0]) / "Llama-3B-oQ4"
        source_path = str(Path(model_dirs[0]) / "Llama-3B-Instruct")
        uploader = HFUploader(model_dirs=model_dirs)

        with patch("huggingface_hub.HfApi") as MockApi:
            mock_api = MockApi.return_value
            mock_api.create_repo.return_value = None

            readme_contents = []
            def fake_upload(**kwargs):
                folder = Path(kwargs["folder_path"])
                readme = folder / "README.md"
                if readme.exists():
                    readme_contents.append(readme.read_text())

            mock_api.upload_folder.side_effect = fake_upload

            await uploader.start_upload(
                model_path=str(oq_path),
                repo_id="user/Llama-3B-oQ4",
                token="hf_token",
                readme_source_path=source_path,
            )
            await asyncio.sleep(0.5)

            assert len(readme_contents) == 1
            assert "Llama 3B Instruct" in readme_contents[0]
            # Cleaned up
            assert not (oq_path / "README.md").exists()

    @pytest.mark.asyncio
    async def test_auto_readme_overwrites_frontmatter_only(self, model_dirs):
        """Frontmatter-only README should be treated as empty and auto-generated."""
        oq_path = Path(model_dirs[0]) / "Llama-3B-oQ4"
        # Write a frontmatter-only stub (like mlx-lm default)
        (oq_path / "README.md").write_text(
            "---\nlanguage: en\nlibrary_name: mlx\ntags:\n- mlx\n---\n"
        )
        uploader = HFUploader(model_dirs=model_dirs)

        with patch("huggingface_hub.HfApi") as MockApi:
            mock_api = MockApi.return_value
            mock_api.create_repo.return_value = None

            readme_contents = []
            def fake_upload(**kwargs):
                folder = Path(kwargs["folder_path"])
                readme = folder / "README.md"
                if readme.exists():
                    readme_contents.append(readme.read_text())

            mock_api.upload_folder.side_effect = fake_upload

            await uploader.start_upload(
                model_path=str(oq_path),
                repo_id="user/Llama-3B-oQ4",
                token="hf_token",
                auto_readme=True,
            )
            await asyncio.sleep(0.5)

            assert len(readme_contents) == 1
            # Should contain auto-generated content, not the stub
            assert "# Llama-3B-oQ4" in readme_contents[0]
            assert "oQ" in readme_contents[0]
