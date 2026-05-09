# SPDX-License-Identifier: Apache-2.0
"""Tests for utils/image.py — image loading, extraction, and hashing."""

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from omlx.utils.image import (
    compute_image_hash,
    compute_per_image_hashes,
    extract_images_from_messages,
    load_image,
)


# =============================================================================
# Helper: create small test images
# =============================================================================

def _make_test_image(width: int = 4, height: int = 4, color: str = "red") -> Image.Image:
    """Create a small solid-color test image."""
    return Image.new("RGB", (width, height), color)


def _image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================================
# Tests: load_image
# =============================================================================

class TestLoadImage:
    """Tests for load_image()."""

    def test_load_base64_png(self):
        """Load image from data URI with base64 PNG."""
        img = _make_test_image(8, 8, "blue")
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        loaded = load_image(uri)
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (8, 8)

    def test_rgba_converted_to_rgb(self):
        """RGBA images (e.g. transparent PNGs) are converted to RGB."""
        rgba_img = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
        b64 = _image_to_base64(rgba_img)
        uri = f"data:image/png;base64,{b64}"

        loaded = load_image(uri)
        assert loaded.mode == "RGB"

    def test_load_base64_jpeg(self):
        """Load image from data URI with base64 JPEG."""
        img = _make_test_image(8, 8, "green")
        b64 = _image_to_base64(img, "JPEG")
        uri = f"data:image/jpeg;base64,{b64}"

        loaded = load_image(uri)
        assert isinstance(loaded, Image.Image)
        assert loaded.size == (8, 8)

    def test_load_base64_without_media_type(self):
        """Load image from data URI without explicit media type."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)
        # Minimal data URI format
        uri = f"data:;base64,{b64}"

        loaded = load_image(uri)
        assert isinstance(loaded, Image.Image)

    @patch("urllib.request.urlopen")
    def test_load_from_url(self, mock_urlopen):
        """Load image from HTTP URL."""
        img = _make_test_image(4, 4)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=buf)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        loaded = load_image("https://example.com/image.png")
        assert isinstance(loaded, Image.Image)

    def test_load_invalid_format_raises(self):
        """Invalid input raises ValueError."""
        with pytest.raises((ValueError, IOError)):
            load_image("not-a-valid-image-source")

    def test_load_invalid_base64_raises(self):
        """Invalid base64 data raises error."""
        with pytest.raises((ValueError, IOError)):
            load_image("data:image/png;base64,not_valid_base64!!!")


# =============================================================================
# Tests: extract_images_from_messages
# =============================================================================

class TestExtractImagesFromMessages:
    """Tests for extract_images_from_messages()."""

    def test_text_only_messages(self):
        """Text-only messages return empty image list."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        text_msgs, images = extract_images_from_messages(messages)
        assert len(text_msgs) == 2
        assert len(images) == 0
        assert text_msgs[0]["content"] == "Hello"

    def test_message_with_image_url(self):
        """Messages with image_url content parts extract images."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": "What is this?"},
                ],
            },
        ]
        text_msgs, images = extract_images_from_messages(messages)
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        # Text-only message should contain only text part
        assert text_msgs[0]["role"] == "user"

    def test_message_with_input_image(self):
        """Messages with input_image content parts extract images."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": uri},
                    {"type": "input_text", "text": "What is this?"},
                ],
            },
        ]
        text_msgs, images = extract_images_from_messages(messages)
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert text_msgs[0]["role"] == "user"

    def test_multiple_images_in_one_message(self):
        """Multiple images in a single message are all extracted."""
        img1 = _make_test_image(4, 4, "red")
        img2 = _make_test_image(4, 4, "blue")
        b64_1 = _image_to_base64(img1)
        b64_2 = _image_to_base64(img2)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_1}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_2}"}},
                    {"type": "text", "text": "Compare these"},
                ],
            },
        ]
        text_msgs, images = extract_images_from_messages(messages)
        assert len(images) == 2

    def test_mixed_text_and_image_messages(self):
        """Mix of text-only and image messages."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Describe this"},
                ],
            },
            {"role": "assistant", "content": "I see an image."},
        ]
        text_msgs, images = extract_images_from_messages(messages)
        assert len(images) == 1
        assert len(text_msgs) == 3
        # System message preserved as-is
        assert text_msgs[0]["content"] == "You are helpful."

    def test_preserves_extra_fields(self):
        """Extra fields like tool_calls are preserved."""
        messages = [
            {
                "role": "assistant",
                "content": "Using tool",
                "tool_calls": [{"id": "tc1", "function": {"name": "test"}}],
            },
        ]
        text_msgs, images = extract_images_from_messages(messages)
        assert "tool_calls" in text_msgs[0]

    def test_invalid_image_url_skipped(self):
        """Invalid image URLs are skipped with warning."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "invalid://not-real"}},
                    {"type": "text", "text": "test"},
                ],
            },
        ]
        text_msgs, images = extract_images_from_messages(messages)
        assert len(images) == 0

    def test_pydantic_model_content_parts(self):
        """Content parts as Pydantic-like objects with type/text/image_url attrs."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)

        # Simulate Pydantic ContentPart with image_url
        image_part = MagicMock(spec=[])
        image_part.type = "image_url"
        image_url = MagicMock(spec=[])
        image_url.url = f"data:image/png;base64,{b64}"
        image_part.image_url = image_url

        # Simulate Pydantic ContentPart with text
        text_part = MagicMock(spec=[])
        text_part.type = "text"
        text_part.text = "What?"

        messages = [
            {"role": "user", "content": [image_part, text_part]},
        ]
        text_msgs, images = extract_images_from_messages(messages)
        assert len(images) == 1


# =============================================================================
# Tests: compute_image_hash
# =============================================================================

class TestComputeImageHash:
    """Tests for compute_image_hash()."""

    def test_empty_list_returns_none(self):
        """Empty image list returns None."""
        assert compute_image_hash([]) is None

    def test_single_image_returns_hex_string(self):
        """Single image returns a hex hash string."""
        img = _make_test_image(4, 4, "red")
        result = compute_image_hash([img])
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 hex

    def test_deterministic(self):
        """Same image produces same hash."""
        img1 = _make_test_image(4, 4, "red")
        img2 = _make_test_image(4, 4, "red")
        assert compute_image_hash([img1]) == compute_image_hash([img2])

    def test_different_images_different_hash(self):
        """Different images produce different hashes."""
        img_red = _make_test_image(4, 4, "red")
        img_blue = _make_test_image(4, 4, "blue")
        assert compute_image_hash([img_red]) != compute_image_hash([img_blue])

    def test_order_matters(self):
        """Image order affects the hash."""
        img1 = _make_test_image(4, 4, "red")
        img2 = _make_test_image(4, 4, "blue")
        hash_12 = compute_image_hash([img1, img2])
        hash_21 = compute_image_hash([img2, img1])
        assert hash_12 != hash_21

    def test_multiple_images(self):
        """Multiple images produce a single hash."""
        images = [_make_test_image(4, 4, c) for c in ("red", "green", "blue")]
        result = compute_image_hash(images)
        assert isinstance(result, str)
        assert len(result) == 64


class TestComputePerImageHashes:
    """Tests for compute_per_image_hashes()."""

    def test_returns_one_hash_per_image(self):
        """Returns a list with the same length as the input."""
        images = [_make_test_image(4, 4, c) for c in ("red", "green", "blue")]
        hashes = compute_per_image_hashes(images)
        assert len(hashes) == 3
        assert all(isinstance(h, str) and len(h) == 64 for h in hashes)

    def test_per_image_matches_single_compute(self):
        """Each per-image hash matches compute_image_hash([single_image])."""
        images = [_make_test_image(4, 4, c) for c in ("red", "green")]
        per_hashes = compute_per_image_hashes(images)
        for img, h in zip(images, per_hashes):
            assert h == compute_image_hash([img])

    def test_different_images_different_hashes(self):
        """Different images produce different per-image hashes."""
        images = [_make_test_image(4, 4, "red"), _make_test_image(4, 4, "blue")]
        hashes = compute_per_image_hashes(images)
        assert hashes[0] != hashes[1]

    def test_empty_returns_empty(self):
        """Empty list returns empty list."""
        assert compute_per_image_hashes([]) == []
