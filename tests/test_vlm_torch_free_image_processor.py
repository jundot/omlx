"""Tests for the torch-free image processor patch in VLM loading.

Background: transformers 5.5+ ships ``AutoImageProcessor`` as a torch-gated
``DummyObject`` that raises ``ImportError`` on attribute access when torch
or torchvision is missing. mlx-vlm's ``GlmOcrProcessor.from_pretrained`` /
``DotsOcrProcessor.from_pretrained`` call ``AutoImageProcessor.from_pretrained``
internally, so they fail silently in oMLX's torch-free env — see #1131, #1175.

``_patch_torch_free_image_processor`` routes those processors to transformers'
PIL-backend image processor classes (``Glm46VImageProcessorPil``,
``Qwen2VLImageProcessorPil``, etc.) via the ``IMAGE_PROCESSOR_MAPPING_NAMES``
table, so they keep working without torch.
"""

import importlib
import json
import sys
import types
from collections import OrderedDict
from unittest.mock import patch

import pytest

from omlx.engine import vlm as vlm_mod
from omlx.engine.vlm import (
    _build_processor_via_pil_image_processor,
    _patch_torch_free_image_processor,
    _resolve_pil_image_processor_class,
    _wrap_from_pretrained_with_pil_image_processor,
)


@pytest.fixture(autouse=True)
def reset_patched_flag():
    """Reset module-level guard so each test can re-run the patch."""
    vlm_mod._torch_free_ip_patched = False
    yield
    vlm_mod._torch_free_ip_patched = False


# ---------------------------------------------------------------------------
# _resolve_pil_image_processor_class
# ---------------------------------------------------------------------------


def test_resolve_pil_class_from_torchvision_name():
    """Mapping like {'pil': 'FooImageProcessorPil', 'torchvision': 'FooImageProcessor'}
    should match by either entry."""
    fake_cls = type("FakePilCls", (), {})

    fake_module = types.ModuleType(
        "transformers.models.foo_model.image_processing_pil_foo_model"
    )
    fake_module.FooImageProcessorPil = fake_cls
    sys.modules[fake_module.__name__] = fake_module

    try:
        mapping_names = OrderedDict(
            [
                (
                    "foo_model",
                    {"pil": "FooImageProcessorPil", "torchvision": "FooImageProcessor"},
                )
            ]
        )
        resolved = _resolve_pil_image_processor_class("FooImageProcessor", mapping_names)
        assert resolved is fake_cls

        # PIL-name path also works.
        resolved = _resolve_pil_image_processor_class(
            "FooImageProcessorPil", mapping_names
        )
        assert resolved is fake_cls
    finally:
        sys.modules.pop(fake_module.__name__, None)


def test_resolve_pil_class_skips_dummy():
    """Dummy classes must be skipped — they raise on attribute access."""
    dummy_cls = type("DummyCls", (), {"is_dummy": True})

    fake_module = types.ModuleType(
        "transformers.models.bar_model.image_processing_pil_bar_model"
    )
    fake_module.BarImageProcessorPil = dummy_cls
    sys.modules[fake_module.__name__] = fake_module

    try:
        mapping_names = OrderedDict(
            [
                (
                    "bar_model",
                    {"pil": "BarImageProcessorPil", "torchvision": "BarImageProcessor"},
                )
            ]
        )
        resolved = _resolve_pil_image_processor_class("BarImageProcessor", mapping_names)
        assert resolved is None
    finally:
        sys.modules.pop(fake_module.__name__, None)


def test_resolve_pil_class_returns_none_when_no_match():
    mapping_names = OrderedDict()
    assert _resolve_pil_image_processor_class("Unknown", mapping_names) is None


# ---------------------------------------------------------------------------
# _wrap_from_pretrained_with_pil_image_processor
# ---------------------------------------------------------------------------


def test_wrap_falls_back_on_torch_import_error(tmp_path):
    """When the wrapped from_pretrained raises ImportError mentioning
    Torchvision / PyTorch, the fallback builder runs."""
    sentinel = object()

    class FakeProc:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            raise ImportError(
                "FakeProc requires the Torchvision library but it was not found"
            )

    _wrap_from_pretrained_with_pil_image_processor(FakeProc)

    with patch.object(
        vlm_mod,
        "_build_processor_via_pil_image_processor",
        return_value=sentinel,
    ) as builder:
        out = FakeProc.from_pretrained(str(tmp_path))

    assert out is sentinel
    builder.assert_called_once()


def test_wrap_reraises_unrelated_import_error(tmp_path):
    """ImportError that is not about torch/torchvision must propagate."""

    class FakeProc:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            raise ImportError("Some other missing module")

    _wrap_from_pretrained_with_pil_image_processor(FakeProc)

    with pytest.raises(ImportError, match="Some other missing module"):
        FakeProc.from_pretrained(str(tmp_path))


def test_wrap_is_idempotent():
    """Wrapping the same class twice keeps a single layer."""

    class FakeProc:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return ("ok", path)

    _wrap_from_pretrained_with_pil_image_processor(FakeProc)
    first_func = FakeProc.from_pretrained.__func__
    _wrap_from_pretrained_with_pil_image_processor(FakeProc)
    assert FakeProc.from_pretrained.__func__ is first_func


# ---------------------------------------------------------------------------
# _build_processor_via_pil_image_processor (mocked PIL class + tokenizer)
# ---------------------------------------------------------------------------


def test_build_processor_uses_pil_image_processor(tmp_path):
    """Given processor_config.json with image_processor_type, the builder
    resolves the matching PIL class and constructs the processor."""

    fake_image_processor = object()
    fake_tokenizer = object()

    class FakePilCls:
        @classmethod
        def from_pretrained(cls, path, trust_remote_code=False):
            return fake_image_processor

    class FakeProcessorCls:
        def __init__(self, image_processor=None, tokenizer=None):
            self.image_processor = image_processor
            self.tokenizer = tokenizer

    # Write processor_config.json with image_processor_type
    proc_cfg = tmp_path / "processor_config.json"
    proc_cfg.write_text(
        json.dumps({"image_processor": {"image_processor_type": "FooImageProcessor"}})
    )

    mapping_names = OrderedDict(
        [
            (
                "foo_model",
                {"pil": "FooImageProcessorPil", "torchvision": "FooImageProcessor"},
            )
        ]
    )

    with patch.object(vlm_mod, "_resolve_pil_image_processor_class", return_value=FakePilCls), \
         patch(
             "transformers.AutoTokenizer.from_pretrained",
             return_value=fake_tokenizer,
         ):
        out = _build_processor_via_pil_image_processor(
            FakeProcessorCls, str(tmp_path), trust_remote_code=True
        )

    assert isinstance(out, FakeProcessorCls)
    assert out.image_processor is fake_image_processor
    assert out.tokenizer is fake_tokenizer


def test_build_processor_falls_back_to_preprocessor_config(tmp_path):
    """When only preprocessor_config.json carries image_processor_type, that
    path is used."""

    fake_image_processor = object()
    fake_tokenizer = object()

    class FakePilCls:
        @classmethod
        def from_pretrained(cls, path, trust_remote_code=False):
            return fake_image_processor

    class FakeProcessorCls:
        def __init__(self, image_processor=None, tokenizer=None):
            self.image_processor = image_processor
            self.tokenizer = tokenizer

    preproc_cfg = tmp_path / "preprocessor_config.json"
    preproc_cfg.write_text(
        json.dumps({"image_processor_type": "BarImageProcessor"})
    )

    with patch.object(vlm_mod, "_resolve_pil_image_processor_class", return_value=FakePilCls), \
         patch(
             "transformers.AutoTokenizer.from_pretrained",
             return_value=fake_tokenizer,
         ):
        out = _build_processor_via_pil_image_processor(
            FakeProcessorCls, str(tmp_path)
        )

    assert isinstance(out, FakeProcessorCls)
    assert out.image_processor is fake_image_processor


def test_build_processor_raises_when_no_image_processor_type(tmp_path):
    """No processor_config.json + no preprocessor_config.json → clear error."""

    class FakeProcessorCls:
        pass

    with pytest.raises(ImportError, match="image_processor_type"):
        _build_processor_via_pil_image_processor(FakeProcessorCls, str(tmp_path))


def test_build_processor_raises_when_pil_class_missing(tmp_path):
    """processor_config.json says FooImageProcessor but no PIL class registered."""

    class FakeProcessorCls:
        pass

    proc_cfg = tmp_path / "processor_config.json"
    proc_cfg.write_text(
        json.dumps({"image_processor": {"image_processor_type": "NoSuchProcessor"}})
    )

    with patch.object(vlm_mod, "_resolve_pil_image_processor_class", return_value=None):
        with pytest.raises(ImportError, match="No torch-free PIL image processor"):
            _build_processor_via_pil_image_processor(FakeProcessorCls, str(tmp_path))


# ---------------------------------------------------------------------------
# _patch_torch_free_image_processor (top-level orchestrator)
# ---------------------------------------------------------------------------


def test_patch_noop_when_autoimageprocessor_not_dummy():
    """If AutoImageProcessor isn't a dummy (torch installed), the patch is a no-op."""
    fake_aip = type("RealAutoImageProcessor", (), {})  # no is_dummy

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoImageProcessor = fake_aip

    with patch.dict(sys.modules, {"transformers": fake_transformers}):
        with patch("importlib.import_module") as ii:
            _patch_torch_free_image_processor()
            ii.assert_not_called()


def test_patch_skips_missing_mlx_vlm_modules():
    """If a mlx-vlm processor module isn't importable, patch logs and continues
    without raising."""
    fake_aip = type("DummyAIP", (), {"is_dummy": True})
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoImageProcessor = fake_aip

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name.startswith("mlx_vlm.models."):
            raise ImportError(f"no module {name}")
        return real_import(name, *args, **kwargs)

    with patch.dict(sys.modules, {"transformers": fake_transformers}):
        with patch("omlx.engine.vlm.importlib.import_module", side_effect=fake_import):
            # Must not raise
            _patch_torch_free_image_processor()


def test_patch_wraps_target_processors():
    """When AutoImageProcessor is dummy and target modules exist, each target
    class's from_pretrained is wrapped exactly once."""
    fake_aip = type("DummyAIP", (), {"is_dummy": True})
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoImageProcessor = fake_aip

    # Build two fake mlx-vlm processor modules.
    class FakeGlmOcrProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return "glm"

    class FakeDotsOcrProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return "dots"

    glm_mod = types.ModuleType("mlx_vlm.models.glm_ocr.processing")
    glm_mod.GlmOcrProcessor = FakeGlmOcrProcessor
    dots_mod = types.ModuleType("mlx_vlm.models.dots_ocr.processing")
    dots_mod.DotsOcrProcessor = FakeDotsOcrProcessor

    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "mlx_vlm.models.glm_ocr.processing":
            return glm_mod
        if name == "mlx_vlm.models.dots_ocr.processing":
            return dots_mod
        return real_import(name, *args, **kwargs)

    with patch.dict(sys.modules, {"transformers": fake_transformers}):
        with patch("omlx.engine.vlm.importlib.import_module", side_effect=fake_import):
            _patch_torch_free_image_processor()

    assert getattr(
        FakeGlmOcrProcessor.from_pretrained, "_omlx_torch_free_patched", False
    )
    assert getattr(
        FakeDotsOcrProcessor.from_pretrained, "_omlx_torch_free_patched", False
    )
