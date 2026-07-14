# SPDX-License-Identifier: Apache-2.0
"""Tests for the torch-free AutoProcessor fallback in omlx.engine.vlm.

transformers 5.x gates processor classes whose default image processors are
torchvision-backed (e.g. PixtralProcessor for mistral3). These tests cover
the pure-math pixtral stub, the graceful-degradation paths, and the
AutoProcessor patch's no-op/idempotence behavior. They run identically with
and without torch installed: the stub is a faithful copy of the reference
implementation, so the math assertions hold against either.
"""

import importlib
import sys

import numpy as np
import pytest

from omlx.engine.vlm import (
    _build_torch_free_processor,
    _ensure_torch_free_pixtral_module,
    _patch_torch_free_auto_processor,
)

PIXTRAL_IP_MODULE = "transformers.models.pixtral.image_processing_pixtral"


def _get_pixtral_ip_module():
    _ensure_torch_free_pixtral_module()
    mod = sys.modules.get(PIXTRAL_IP_MODULE)
    if mod is None:
        mod = importlib.import_module(PIXTRAL_IP_MODULE)
    return mod


class TestPixtralMathStub:
    def test_module_importable_after_ensure(self):
        mod = _get_pixtral_ip_module()
        assert hasattr(mod, "get_resize_output_image_size")
        assert hasattr(mod, "_num_image_tokens")

    def test_num_image_tokens(self):
        mod = _get_pixtral_ip_module()
        assert mod._num_image_tokens((100, 60), (14, 14)) == (8, 5)
        assert mod._num_image_tokens((14, 14), (14, 14)) == (1, 1)
        assert mod._num_image_tokens((15, 14), 14) == (2, 1)

    def test_resize_no_downscale_rounds_to_patch_multiples(self):
        mod = _get_pixtral_ip_module()
        img = np.zeros((3, 700, 1200), dtype=np.uint8)
        out = mod.get_resize_output_image_size(
            img,
            size=(1540, 1540),
            patch_size=(14, 14),
            input_data_format="channels_first",
        )
        assert tuple(out) == (700, 1204)

    def test_resize_downscales_to_longest_edge(self):
        mod = _get_pixtral_ip_module()
        img = np.zeros((3, 3080, 1540), dtype=np.uint8)
        out = mod.get_resize_output_image_size(
            img,
            size=(1540, 1540),
            patch_size=(14, 14),
            input_data_format="channels_first",
        )
        assert tuple(out) == (1540, 770)


class TestBuildTorchFreeProcessor:
    def test_returns_none_for_empty_model_dir(self, tmp_path):
        # No processor_config.json / config.json: nothing to build, no crash.
        assert _build_torch_free_processor(tmp_path) is None

    def test_returns_none_for_unknown_processor_class(self, tmp_path):
        (tmp_path / "processor_config.json").write_text(
            '{"processor_class": "DefinitelyNotARealProcessor"}'
        )
        assert _build_torch_free_processor(tmp_path) is None


class TestAutoProcessorPatch:
    def test_noop_or_transparent_wrap(self):
        transformers = pytest.importorskip("transformers")
        before = transformers.AutoProcessor.from_pretrained
        _patch_torch_free_auto_processor()
        after = transformers.AutoProcessor.from_pretrained

        if getattr(transformers.AutoImageProcessor, "is_dummy", False):
            # torch-free env: wrapper installed exactly once.
            assert getattr(after, "_omlx_torch_free_processor", False)
            _patch_torch_free_auto_processor()
            assert transformers.AutoProcessor.from_pretrained is after
        else:
            # torch available: patch must be a no-op.
            assert after is before
