# SPDX-License-Identifier: Apache-2.0
"""Tests for the OQManager admin component."""

import json

import pytest

from omlx.admin.oq_manager import OQManager


@pytest.fixture
def fp_model_dir(tmp_path):
    """One directory with a full-precision (quantizable) source model."""
    d = tmp_path / "models1"
    d.mkdir()
    model = d / "Llama-3B"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "num_hidden_layers": 32,
            }
        )
    )
    (model / "model.safetensors").write_bytes(b"\x00" * 4096)
    return d


@pytest.fixture
def second_fp_model_dir(tmp_path):
    """A second directory holding a different full-precision model."""
    d = tmp_path / "models2"
    d.mkdir()
    model = d / "Qwen-7B"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 28,
            }
        )
    )
    (model / "model.safetensors").write_bytes(b"\x00" * 4096)
    return d


class TestOQManagerUpdateModelDirs:
    @pytest.mark.asyncio
    async def test_picks_up_added_dir(self, fp_model_dir, second_fp_model_dir):
        # Mirrors the real Settings UI flow: server starts with one model
        # directory, the user adds a second one at runtime via Settings, and
        # _apply_model_dirs_runtime calls update_model_dirs(). Without that
        # call, models in the newly added directory never show up in the oQ
        # Quantization "Source Model" dropdown.
        manager = OQManager(model_dirs=[str(fp_model_dir)])
        source_before, _ = await manager.list_quantizable_models()
        names_before = {m["name"] for m in source_before}
        assert "Llama-3B" in names_before
        assert "Qwen-7B" not in names_before

        manager.update_model_dirs([str(fp_model_dir), str(second_fp_model_dir)])

        source_after, _ = await manager.list_quantizable_models()
        names_after = {m["name"] for m in source_after}
        assert "Llama-3B" in names_after
        assert "Qwen-7B" in names_after

    def test_output_dir_tracks_primary_dir(self, fp_model_dir, second_fp_model_dir):
        # Output is always written to the primary (first) directory.
        manager = OQManager(model_dirs=[str(fp_model_dir)])
        assert manager._output_dir == fp_model_dir

        manager.update_model_dirs([str(second_fp_model_dir), str(fp_model_dir)])
        assert manager._output_dir == second_fp_model_dir


class TestOQManagerMtpDetection:
    def _write_model(self, root, name, *, index_weight_map=None):
        model = root / name
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "text_config": {
                        "model_type": "qwen3_5_text",
                        "num_hidden_layers": 32,
                        "mtp_num_hidden_layers": 1,
                    },
                }
            )
        )
        (model / "model.safetensors").write_bytes(b"\x00" * 4096)
        if index_weight_map is not None:
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": index_weight_map,
                    }
                )
            )
        return model

    @pytest.mark.asyncio
    async def test_config_only_mtp_is_not_reported_as_preservable(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        self._write_model(root, "QwenPawLike")

        manager = OQManager(model_dirs=[str(root)])
        source_models, _ = await manager.list_quantizable_models()

        [model] = source_models
        assert model["has_mtp_heads"] is False

    @pytest.mark.asyncio
    async def test_mtp_weight_index_is_reported_as_preservable(self, tmp_path):
        root = tmp_path / "models"
        root.mkdir()
        self._write_model(
            root,
            "QwenMtp",
            index_weight_map={
                "language_model.mtp.fc.weight": "model.safetensors",
            },
        )

        manager = OQManager(model_dirs=[str(root)])
        source_models, _ = await manager.list_quantizable_models()

        [model] = source_models
        assert model["has_mtp_heads"] is True

    @pytest.mark.asyncio
    async def test_start_quantization_disables_preserve_mtp_without_weights(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "models"
        root.mkdir()
        self._write_model(root, "QwenPawLike")

        manager = OQManager(model_dirs=[str(root)])

        async def _noop_run(task_id):
            return None

        monkeypatch.setattr(manager, "_run_quantization", _noop_run)

        task = await manager.start_quantization(
            str(root / "QwenPawLike"),
            4,
            preserve_mtp=True,
        )
        await manager._active_tasks[task.task_id]

        assert task.preserve_mtp is False
        assert task.output_name == "QwenPawLike-oQ4"
