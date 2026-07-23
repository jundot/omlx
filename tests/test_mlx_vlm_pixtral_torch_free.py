"""Tests for the torch-free Pixtral/Mistral3 processor patch (issue #2263).

Covers the vendored PixtralImageProcessor geometry (ported from upstream
mlx-vlm PR #1502 tests), the method transplant onto the pinned classes,
and the TokenizersBackend pin for tekken.json checkpoints.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from omlx.patches.mlx_vlm_pixtral_torch_free import apply_pixtral_torch_free_patch
from omlx.patches.mlx_vlm_pixtral_torch_free.vendor.image_processing_pixtral import (
    PixtralImageProcessor,
    split_image_sizes_by_sample,
)

DEVSTRAL_DIR = Path.home() / "Workspace/models/Devstral-Small-2-24B-Instruct-2512-4bit"


def _make_image(h: int = 20, w: int = 20) -> Image.Image:
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


class _FakeTokenizer:
    model_input_names = ["input_ids"]

    _ids = {"[IMG]": 10, "[IMG_BREAK]": 12, "[IMG_END]": 13}

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token, 1)

    def __call__(self, text, **kwargs):
        return {"input_ids": [[1, 2, 3] for _ in text]}


def _fake_processor_init(
    self,
    image_processor=None,
    tokenizer=None,
    patch_size=16,
    spatial_merge_size=1,
    image_token="[IMG]",
    image_break_token="[IMG_BREAK]",
    image_end_token="[IMG_END]",
    chat_template=None,
    **kwargs,
):
    self.image_processor = image_processor
    self.tokenizer = tokenizer
    self.patch_size = patch_size
    self.spatial_merge_size = spatial_merge_size
    self.image_token = image_token
    self.image_break_token = image_break_token
    self.image_end_token = image_end_token
    self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    self.image_break_token_id = tokenizer.convert_tokens_to_ids(image_break_token)
    self.image_end_token_id = tokenizer.convert_tokens_to_ids(image_end_token)
    self.chat_template = chat_template


class TestVendoredImageProcessor:
    """Geometry tests ported from upstream mlx-vlm PR #1502."""

    def test_preprocess_resizes_to_patch_multiple_and_pads(self):
        image_processor = PixtralImageProcessor(
            size={"longest_edge": 40},
            patch_size=14,
            image_mean=[0, 0, 0],
            image_std=[1, 1, 1],
        )
        wide = _make_image(31, 55)
        square = _make_image(20, 20)

        output = image_processor([[wide, square]])

        assert output["image_sizes"] == [(28, 42), (28, 28)]
        assert output["pixel_values"].shape == (2, 3, 28, 42)

    def test_split_image_sizes_by_sample_handles_flat_sizes(self):
        images = [[_make_image(), _make_image()], [_make_image()]]
        sizes = [(28, 42), (28, 28), (56, 56)]

        assert split_image_sizes_by_sample(sizes, images) == [
            [(28, 42), (28, 28)],
            [(56, 56)],
        ]


class TestPatchInstallation:
    def test_apply_is_idempotent_and_transplants_methods(self):
        from mlx_vlm.models.mistral3 import processing_mistral3 as pin_m3
        from mlx_vlm.models.pixtral import processing_pixtral as pin_px

        from omlx.patches.mlx_vlm_pixtral_torch_free.vendor import (
            processing_mistral3 as vendor_m3,
        )
        from omlx.patches.mlx_vlm_pixtral_torch_free.vendor import (
            processing_pixtral as vendor_px,
        )

        assert apply_pixtral_torch_free_patch() is True
        assert apply_pixtral_torch_free_patch() is True

        assert (
            pin_m3.Mistral3Processor.__call__
            is vendor_m3.Mistral3Processor.__dict__["__call__"]
        )
        assert (
            pin_m3.Mistral3Processor.__dict__["from_pretrained"]
            is vendor_m3.Mistral3Processor.__dict__["from_pretrained"]
        )
        assert (
            pin_px.PixtralProcessor.__call__
            is vendor_px.PixtralProcessor.__dict__["__call__"]
        )
        assert (
            pin_px.PixtralProcessor.__dict__["from_pretrained"]
            is vendor_px.PixtralProcessor.__dict__["from_pretrained"]
        )

    def test_from_pretrained_uses_vendored_image_processor(self, tmp_path):
        assert apply_pixtral_torch_free_patch() is True

        from mlx_vlm.models.mistral3.processing_mistral3 import Mistral3Processor

        (tmp_path / "processor_config.json").write_text(
            json.dumps(
                {
                    "patch_size": 16,
                    "spatial_merge_size": 1,
                    "image_token": "[IMG]",
                    "image_break_token": "[IMG_BREAK]",
                    "image_end_token": "[IMG_END]",
                    "image_processor": {
                        "image_processor_type": "PixtralImageProcessorFast",
                        "patch_size": 14,
                        "size": {"longest_edge": 64},
                    },
                }
            )
        )
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "mistral3",
                    "spatial_merge_size": 2,
                    "vision_config": {"patch_size": 14},
                }
            )
        )

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=_FakeTokenizer(),
            ) as tok_mock,
            patch.object(Mistral3Processor, "__init__", _fake_processor_init),
        ):
            processor = Mistral3Processor.from_pretrained(str(tmp_path))

        assert isinstance(processor.image_processor, PixtralImageProcessor)
        assert processor.patch_size == 14
        assert processor.spatial_merge_size == 2
        # Tokenizer backend pin applies to the tokenizer load only.
        assert tok_mock.call_args.kwargs.get("fix_mistral_regex") is True

        output = processor(text=["[IMG]Describe"], images=[[_make_image()]])
        assert output["pixel_values"].shape[0] == 1
        assert output["pixel_values"].shape[1] == 3
        assert int(output["image_sizes"][0, 0].item()) % 28 == 0
        assert int(output["image_sizes"][0, 1].item()) % 28 == 0

    def test_pixtral_from_pretrained_pins_tokenizer_backend(self, tmp_path):
        assert apply_pixtral_torch_free_patch() is True

        from mlx_vlm.models.pixtral.processing_pixtral import PixtralProcessor

        (tmp_path / "processor_config.json").write_text(
            json.dumps({"patch_size": 14, "spatial_merge_size": 1})
        )

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=_FakeTokenizer(),
            ) as tok_mock,
            patch.object(PixtralProcessor, "__init__", _fake_processor_init),
        ):
            processor = PixtralProcessor.from_pretrained(str(tmp_path))

        assert isinstance(processor.image_processor, PixtralImageProcessor)
        assert tok_mock.call_args.kwargs.get("fix_mistral_regex") is True


@pytest.mark.skipif(
    not DEVSTRAL_DIR.exists(), reason="local Devstral checkpoint not present"
)
class TestTekkenCheckpointIntegration:
    """End-to-end processor load against a real tekken.json checkpoint.

    Requires no model weights; exercises the exact path that failed in
    issue #2263 (AutoProcessor -> Mistral3Processor -> image processor +
    tokenizer backend + streaming detokenizer construction).
    """

    def test_load_processor_torch_free(self):
        assert apply_pixtral_torch_free_patch() is True

        from mlx_vlm.utils import load_processor

        processor = load_processor(DEVSTRAL_DIR, add_detokenizer=True)
        tokenizer = getattr(processor, "tokenizer", processor)

        assert type(processor).__name__ == "Mistral3Processor"
        assert isinstance(processor.image_processor, PixtralImageProcessor)
        assert type(tokenizer).__name__ == "TokenizersBackend"
        assert hasattr(tokenizer, "vocab")
        assert type(processor.detokenizer).__name__ == "BPEStreamingDetokenizer"
