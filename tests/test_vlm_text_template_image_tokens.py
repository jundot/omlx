# SPDX-License-Identifier: Apache-2.0
"""Clear error when a vision request renders no image tokens.

A checkpoint whose ``chat_template`` is text-only renders image parts as prose
instead of image tokens, so the processor reports the misleading
``More images were provided than image tokens.`` for a request that carried a
single image. These tests pin the actionable replacement message and the cases
where it must stay silent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("mlx_vlm.utils")

import mlx.core as mx  # noqa: E402

from omlx.engine.vlm import VLMBatchedEngine  # noqa: E402
from omlx.exceptions import InvalidRequestError  # noqa: E402

IMAGE_TOKEN = "<|image|>"
IMAGE_TOKEN_ID = 154854
REMINDER = (
    "<reminder>You are unable to process this image because you don't have "
    "multi-modal input ability. Try different methods.</reminder>"
)


class _ImageProcessor:
    merge_size = 2

    def __call__(self, images, **kwargs):
        n = 1 if not isinstance(images, (list, tuple)) else len(images)
        return {"image_grid_thw": [[4, 4, 3]] * n}


class _Tokenizer:
    pad_token = "<|pad|>"
    eos_token = "<|eos|>"
    model_input_names = ["input_ids", "attention_mask"]

    def convert_ids_to_tokens(self, token_id):
        return {IMAGE_TOKEN_ID: IMAGE_TOKEN}.get(token_id)

    def __call__(self, text, **kwargs):
        texts = text if isinstance(text, list) else [text]
        rows = [[1, 2, 3] + [IMAGE_TOKEN_ID] * str(t).count(IMAGE_TOKEN) for t in texts]
        return MagicMock(input_ids=rows, attention_mask=[[True] * len(r) for r in rows])


class _Processor:
    """Mirrors the vendored glm5_next processor's image-token bookkeeping."""

    def __init__(self, declare_image_token=True, image_token_id=IMAGE_TOKEN_ID):
        self.image_processor = _ImageProcessor()
        self.tokenizer = _Tokenizer()
        self.image_token = IMAGE_TOKEN if declare_image_token else None
        self.image_token_id = image_token_id

    def __call__(self, text=None, images=None, **kwargs):
        text = [
            "" if text is None else str(t)
            for t in (text if isinstance(text, list) else [text])
        ]
        grids = (self.image_processor(images) or {}).get("image_grid_thw", [])
        placeholder = "<|placeholder|>"
        index = 0
        needle = self.image_token if self.image_token else "\x00never"
        for i, prompt in enumerate(text):
            while needle in prompt:
                if index >= len(grids):
                    raise ValueError("More image tokens were provided than images.")
                count = int(np.prod(grids[index]) // self.image_processor.merge_size**2)
                prompt = prompt.replace(needle, placeholder * count, 1)
                index += 1
            text[i] = prompt.replace(placeholder, self.image_token or IMAGE_TOKEN)
        if index != len(grids):
            raise ValueError("More images were provided than image tokens.")
        return _batch(text, len(grids), self.image_token or IMAGE_TOKEN)


class _SilentProcessor:
    """Accepts a zero-image-token prompt without validating it."""

    def __init__(self, image_token_id=IMAGE_TOKEN_ID):
        self.image_processor = _ImageProcessor()
        self.tokenizer = _Tokenizer()
        self.image_token = IMAGE_TOKEN
        self.image_token_id = image_token_id

    def __call__(self, text=None, images=None, **kwargs):
        text = [
            "" if text is None else str(t)
            for t in (text if isinstance(text, list) else [text])
        ]
        grids = (self.image_processor(images) or {}).get("image_grid_thw", [])
        return _batch(text, len(grids), self.image_token)


def _batch(texts, num_grids, token):
    rows = [[1, 2, 3] + [IMAGE_TOKEN_ID] * t.count(token) for t in texts]
    return {
        "input_ids": mx.array(rows, dtype=mx.int32),
        "attention_mask": mx.ones((len(rows), len(rows[0])), dtype=mx.int32),
        "pixel_values": mx.zeros((1, 4, 8), dtype=mx.float32),
        "image_grid_thw": [[4, 4, 3]] * num_grids,
    }


class _Template:
    """Renders ``markers`` image markers, or the text-only reminder instead."""

    def __init__(self, markers, text_only=False):
        self.markers = markers
        self.text_only = text_only

    def apply_chat_template(self, messages, **kwargs):
        out = []
        emitted = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        out.append(part.get("text", ""))
                    elif emitted < self.markers:
                        emitted += 1
                        out.append(IMAGE_TOKEN)
            else:
                out.append(str(content))
        rendered = "".join(out)
        if self.text_only:
            return rendered.replace(IMAGE_TOKEN, REMINDER)
        return rendered


def _engine(processor, template, model_type="glm5_next", image_token_id=IMAGE_TOKEN_ID):
    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._vlm_model = MagicMock()
    engine._vlm_model.config.model_type = model_type
    engine._vlm_model.config.image_token_id = image_token_id
    engine._vlm_model.has_encode_image = False
    engine._vlm_model.encode_image = None
    engine._processor = processor
    engine._vision_cache = None
    engine._vision_cache_enabled = False
    engine._model_name = "test-model"
    engine._enable_thinking = None
    engine._chat_template_target = MagicMock(return_value=template)
    return engine


def _messages(num_images=1):
    content = [{"type": "text", "text": "what is this"}]
    content.extend({"type": "image"} for _ in range(num_images))
    return [{"role": "user", "content": content}]


def _images(count=1):
    from PIL import Image

    return [Image.new("RGB", (8, 8), "red") for _ in range(count)]


def test_text_only_template_raises_actionable_error():
    """The reported case: a text-only template renders no image tokens."""
    engine = _engine(_Processor(), _Template(1, text_only=True))

    with pytest.raises(InvalidRequestError) as excinfo:
        engine._prepare_vision_inputs(_messages(), _images())

    message = str(excinfo.value)
    assert "chat template" in message
    assert "image tokens" in message
    assert "1 provided image" in message


def test_text_only_template_names_the_model():
    engine = _engine(_Processor(), _Template(1, text_only=True))

    with pytest.raises(InvalidRequestError) as excinfo:
        engine._prepare_vision_inputs(_messages(3), _images(3))

    assert "test-model" in str(excinfo.value)
    assert "3 provided images" in str(excinfo.value)


def test_vision_template_still_succeeds():
    """A template that emits image tokens must not be disturbed."""
    engine = _engine(_Processor(), _Template(1))

    ids = engine._prepare_vision_inputs(_messages(), _images())[0]

    assert IMAGE_TOKEN_ID in list(ids)


def test_unknown_image_token_does_not_fire():
    """No declared token string and no config id means 'unknown', not 'zero'."""
    processor = _Processor(declare_image_token=False, image_token_id=None)
    engine = _engine(processor, _Template(1, text_only=True), image_token_id=None)

    with pytest.raises(ValueError, match="More images were provided than image tokens"):
        engine._prepare_vision_inputs(_messages(), _images())


def test_gemma_style_processor_does_not_fire():
    """Gemma 3/4 declare no image token and expand soft tokens later.

    The rendered prompt carries ``<start_of_image>`` only, so deriving the
    token from ``config.image_token_id`` would wrongly report a missing token.
    """
    processor = _Processor(declare_image_token=False)
    engine = _engine(processor, _Template(1, text_only=True), model_type="gemma4")

    assert engine._image_token_string() is None


def test_vision_start_triple_template_does_not_fire():
    """Qwen-style templates wrap the image token; the token itself remains."""
    processor = _Processor()
    processor.image_token = "<|image_pad|>"
    engine = _engine(processor, _Template(1), model_type="qwen3_vl")

    class _Triple(_Template):
        def apply_chat_template(self, messages, **kwargs):
            return (
                super()
                .apply_chat_template(messages, **kwargs)
                .replace("<|image|>", "<|vision_start|><|image_pad|><|vision_end|>")
            )

    engine._chat_template_target = MagicMock(return_value=_Triple(1))

    ids = engine._prepare_vision_inputs(_messages(), _images())[0]

    assert len(ids) > 0


def test_text_only_request_without_images_does_not_fire():
    engine = _engine(_Processor(), _Template(1, text_only=True))

    ids = engine._prepare_vision_inputs(_messages(0), [])[0]

    assert len(ids) > 0


def test_partial_template_render_does_not_fire():
    """A template that expands only some images keeps the original error."""
    engine = _engine(_Processor(), _Template(1))

    with pytest.raises(ValueError, match="More images were provided than image tokens"):
        engine._prepare_vision_inputs(_messages(2), _images(2))


def test_zero_image_tokens_after_tokenization_raises():
    """A processor that never validates still yields zero image tokens."""
    engine = _engine(_SilentProcessor(), _Template(1, text_only=True))

    with pytest.raises(InvalidRequestError) as excinfo:
        engine._prepare_vision_inputs(_messages(), _images())

    assert "image tokens" in str(excinfo.value)
