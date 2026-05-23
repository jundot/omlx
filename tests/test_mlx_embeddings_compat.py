# SPDX-License-Identifier: Apache-2.0
"""Tests for mlx-embeddings compatibility patches."""

import importlib.util
import sys
import types
from pathlib import Path


def _install_fake_module(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_qwen3_vl_auto_image_processor_uses_mlx_vlm_torch_free_loader(monkeypatch):
    """Qwen3-VL Processor should use mlx-vlm's torch-free image processor."""
    processor_module = types.ModuleType("mlx_embeddings.models.qwen3_vl.processor")

    class TorchBoundAutoImageProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("torch/torchvision required")

    processor_module.AutoImageProcessor = TorchBoundAutoImageProcessor

    qwen3_vl_package = types.ModuleType("mlx_embeddings.models.qwen3_vl")
    qwen3_vl_package.processor = processor_module

    _install_fake_module(
        monkeypatch, "mlx_embeddings", types.ModuleType("mlx_embeddings")
    )
    _install_fake_module(
        monkeypatch, "mlx_embeddings.models", types.ModuleType("mlx_embeddings.models")
    )
    _install_fake_module(
        monkeypatch, "mlx_embeddings.models.qwen3_vl", qwen3_vl_package
    )
    _install_fake_module(
        monkeypatch, "mlx_embeddings.models.qwen3_vl.processor", processor_module
    )

    mlx_vlm_processing = types.ModuleType("mlx_vlm.models.qwen3_vl.processing_qwen3_vl")
    captured = {}

    class TorchFreeImageProcessor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    def fake_image_kwargs(model_path, default_patch_size=16):
        captured["model_path"] = model_path
        captured["default_patch_size"] = default_patch_size
        return {"patch_size": default_patch_size, "merge_size": 2}

    mlx_vlm_processing.Qwen3VLImageProcessor = TorchFreeImageProcessor
    mlx_vlm_processing._qwen_vl_image_kwargs = fake_image_kwargs

    _install_fake_module(monkeypatch, "mlx_vlm", types.ModuleType("mlx_vlm"))
    _install_fake_module(
        monkeypatch, "mlx_vlm.models", types.ModuleType("mlx_vlm.models")
    )
    _install_fake_module(
        monkeypatch,
        "mlx_vlm.models.qwen3_vl",
        types.ModuleType("mlx_vlm.models.qwen3_vl"),
    )
    _install_fake_module(
        monkeypatch,
        "mlx_vlm.models.qwen3_vl.processing_qwen3_vl",
        mlx_vlm_processing,
    )

    module_path = (
        Path(__file__).resolve().parents[1] / "omlx/models/mlx_embeddings_compat.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mlx_embeddings_compat_under_test", module_path
    )
    mlx_embeddings_compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mlx_embeddings_compat)

    monkeypatch.setattr(mlx_embeddings_compat, "_QWEN3_VL_PROCESSOR_PATCHED", False)

    mlx_embeddings_compat.patch_qwen3_vl_processor_for_torch_free_image_loading()

    image_processor = processor_module.AutoImageProcessor.from_pretrained(
        "/models/Qwen3-VL-Embedding-8B-8bit-mlx",
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )

    assert isinstance(image_processor, TorchFreeImageProcessor)
    assert captured["model_path"] == "/models/Qwen3-VL-Embedding-8B-8bit-mlx"
    assert captured["default_patch_size"] == 16
    assert captured["kwargs"] == {"patch_size": 16, "merge_size": 2}


def test_qwen3_vl_build_processor_gets_multimodal_token_id_fields(monkeypatch):
    """Qwen3-VL ProcessorMixin fields should exist when __init__ is bypassed."""
    processor_module = types.ModuleType("mlx_embeddings.models.qwen3_vl.processor")

    class TorchBoundAutoImageProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("torch/torchvision required")

    class ManuallyBuiltProcessor:
        image_token_id = 151655
        video_token_id = 151656

    class MlxEmbeddingsProcessor:
        @staticmethod
        def _build_processor(tokenizer, image_processor):
            del tokenizer, image_processor
            return ManuallyBuiltProcessor()

    processor_module.AutoImageProcessor = TorchBoundAutoImageProcessor
    processor_module.Processor = MlxEmbeddingsProcessor

    qwen3_vl_package = types.ModuleType("mlx_embeddings.models.qwen3_vl")
    qwen3_vl_package.processor = processor_module

    _install_fake_module(
        monkeypatch, "mlx_embeddings", types.ModuleType("mlx_embeddings")
    )
    _install_fake_module(
        monkeypatch, "mlx_embeddings.models", types.ModuleType("mlx_embeddings.models")
    )
    _install_fake_module(
        monkeypatch, "mlx_embeddings.models.qwen3_vl", qwen3_vl_package
    )
    _install_fake_module(
        monkeypatch, "mlx_embeddings.models.qwen3_vl.processor", processor_module
    )

    mlx_vlm_processing = types.ModuleType("mlx_vlm.models.qwen3_vl.processing_qwen3_vl")

    class TorchFreeImageProcessor:
        pass

    mlx_vlm_processing.Qwen3VLImageProcessor = TorchFreeImageProcessor
    mlx_vlm_processing._qwen_vl_image_kwargs = lambda *args, **kwargs: {}

    _install_fake_module(monkeypatch, "mlx_vlm", types.ModuleType("mlx_vlm"))
    _install_fake_module(
        monkeypatch, "mlx_vlm.models", types.ModuleType("mlx_vlm.models")
    )
    _install_fake_module(
        monkeypatch,
        "mlx_vlm.models.qwen3_vl",
        types.ModuleType("mlx_vlm.models.qwen3_vl"),
    )
    _install_fake_module(
        monkeypatch,
        "mlx_vlm.models.qwen3_vl.processing_qwen3_vl",
        mlx_vlm_processing,
    )

    module_path = (
        Path(__file__).resolve().parents[1] / "omlx/models/mlx_embeddings_compat.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mlx_embeddings_compat_under_test_mm_ids", module_path
    )
    mlx_embeddings_compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mlx_embeddings_compat)

    monkeypatch.setattr(mlx_embeddings_compat, "_QWEN3_VL_PROCESSOR_PATCHED", False)

    mlx_embeddings_compat.patch_qwen3_vl_processor_for_torch_free_image_loading()

    processor = MlxEmbeddingsProcessor._build_processor(object(), object())

    assert processor.image_ids == [151655]
    assert processor.video_ids == [151656]
    assert processor.audio_ids == [None]
