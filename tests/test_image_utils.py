# SPDX-License-Identifier: Apache-2.0
"""Tests for utils/image.py — image loading, extraction, and hashing."""

import base64
import io
import struct
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from omlx.exceptions import InvalidRequestError
from omlx.utils.image import (
    compute_image_hash,
    compute_per_image_hashes,
    extract_images_from_messages,
    load_image,
)

# =============================================================================
# Helper: create small test images
# =============================================================================


def _make_test_image(
    width: int = 4, height: int = 4, color: str = "red"
) -> Image.Image:
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

    def test_rejects_data_uri_without_image_media_type(self):
        """Image data URIs must include an image media type."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)
        uri = f"data:;base64,{b64}"

        with pytest.raises(InvalidRequestError):
            load_image(uri)

    @patch("urllib.request.urlopen")
    def test_rejects_remote_url_without_fetching(self, mock_urlopen):
        """Remote URL image refs are rejected without server-side fetches."""
        with pytest.raises(InvalidRequestError):
            load_image("https://example.com/image.png")
        mock_urlopen.assert_not_called()

    def test_rejects_local_file_path(self, tmp_path):
        """Local filesystem image refs are rejected before opening files."""
        img = _make_test_image(4, 4)
        path = tmp_path / "local.png"
        img.save(path)

        with pytest.raises(InvalidRequestError):
            load_image(str(path))

    def test_load_invalid_format_raises(self):
        """Invalid input raises a request error."""
        with pytest.raises(InvalidRequestError):
            load_image("not-a-valid-image-source")

    def test_load_invalid_base64_raises(self):
        """Invalid base64 data raises error."""
        with pytest.raises(InvalidRequestError):
            load_image("data:image/png;base64,not_valid_base64!!!")

    def test_rejects_non_image_data_uri(self):
        """Non-image data URIs are rejected for image inputs."""
        data = base64.b64encode(b"hello").decode()
        with pytest.raises(InvalidRequestError):
            load_image(f"data:text/plain;base64,{data}")


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
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(text_msgs) == 2
        assert len(images) == 0
        assert len(audio) == 0
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
        text_msgs, images, audio = extract_images_from_messages(messages)
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
        text_msgs, images, audio = extract_images_from_messages(messages)
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
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_1}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_2}"},
                    },
                    {"type": "text", "text": "Compare these"},
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
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
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": "Describe this"},
                ],
            },
            {"role": "assistant", "content": "I see an image."},
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
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
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert "tool_calls" in text_msgs[0]

    def test_invalid_image_url_rejected(self):
        """Invalid image URLs are rejected instead of silently skipped."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "invalid://not-real"}},
                    {"type": "text", "text": "test"},
                ],
            },
        ]
        with pytest.raises(InvalidRequestError):
            extract_images_from_messages(messages)

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
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(images) == 1

    def test_input_audio_base64_data_uri(self):
        """Messages with input_audio base64 data URI extract audio."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAABCxAgAEABAAZGF0YQAAAAA=",
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": "What do you hear?"},
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(audio) == 1
        assert len(images) == 0
        # Audio should be a BytesIO object
        assert hasattr(audio[0], "read")

    def test_input_audio_raw_base64(self):
        """Messages with raw base64 input_audio extract audio."""
        import base64

        raw_bytes = b"\x00\x01\x02\x03" * 16
        b64 = base64.b64encode(raw_bytes).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64,
                            "format": "wav",
                        },
                    },
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(audio) == 1
        assert hasattr(audio[0], "read")

    def test_input_audio_bytes_data(self):
        """Messages with bytes input_audio.data extract audio."""
        raw_bytes = b"\x00\x01\x02\x03" * 16

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": raw_bytes,
                            "format": "wav",
                        },
                    },
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(audio) == 1
        assert hasattr(audio[0], "read")

    def test_input_audio_string_path_rejected(self):
        """Non-base64 input_audio.data strings are rejected as path refs."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "/tmp/audio.wav",
                            "format": "wav",
                        },
                    },
                ],
            },
        ]
        with pytest.raises(InvalidRequestError):
            extract_images_from_messages(messages)

    def test_invalid_input_audio_rejected(self):
        """Invalid input_audio base64 is rejected."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "data:audio/wav;base64,!!!not_valid_base64!!!",
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": "test"},
                ],
            },
        ]
        with pytest.raises(InvalidRequestError):
            extract_images_from_messages(messages)

    def test_audio_mixed_with_images(self):
        """Audio and images in the same message both extracted."""
        img = _make_test_image(4, 4)
        b64 = _image_to_base64(img)
        import base64 as b64_mod

        raw_bytes = b"\x00\x01\x02\x03" * 16
        audio_b64 = b64_mod.b64encode(raw_bytes).decode()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": "Describe this image and audio"},
                ],
            },
        ]
        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(images) == 1
        assert len(audio) == 1
        # Text content should be preserved
        assert "Describe this image and audio" in text_msgs[0]["content"]


def test_video_input_is_rejected():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "data:video/mp4;base64,AAAA",
                        "fps": 3,
                    },
                },
            ],
        }
    ]

    with pytest.raises(InvalidRequestError, match="Video input is not supported"):
        extract_images_from_messages(messages)


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


# =============================================================================
# Tests: load_image decode cache
#
# Regression for the multi-turn agent TTFT cliff: an agent loop resends the
# same historical screenshots on every turn, and load_image re-ran the
# CPU-bound PNG/JPEG decode for every one of them each turn. Decoded images
# are now cached by content hash so repeated turns skip the re-decode.
# =============================================================================


def _unique_image(seed: int, width: int = 24, height: int = 24) -> Image.Image:
    """Build an RGB image whose pixel bytes are unique to ``seed``."""
    import random

    rng = random.Random(seed)
    data = bytes(rng.getrandbits(8) for _ in range(width * height * 3))
    return Image.frombytes("RGB", (width, height), data)


class TestLoadImageDecodeCache:
    """Decoded images are cached by content hash across load_image calls."""

    def setup_method(self):
        from omlx.utils.image import clear_image_decode_cache

        clear_image_decode_cache()

    def test_identical_image_decoded_once_across_calls(self):
        """Same bytes on a later turn must not re-open/re-decode the image."""
        uri = "data:image/png;base64," + _image_to_base64(_unique_image(1))
        real_open = Image.open
        calls = {"n": 0}

        def counting_open(*args, **kwargs):
            calls["n"] += 1
            return real_open(*args, **kwargs)

        with patch("omlx.utils.image.Image.open", side_effect=counting_open):
            first = load_image(uri)
            second = load_image(uri)

        assert calls["n"] == 1
        assert first.size == second.size == (24, 24)
        assert first.tobytes() == second.tobytes()

    def test_distinct_images_each_decoded(self):
        """Different bytes decode independently (no false cache hits)."""
        uri_a = "data:image/png;base64," + _image_to_base64(_unique_image(2))
        uri_b = "data:image/png;base64," + _image_to_base64(_unique_image(3))
        real_open = Image.open
        calls = {"n": 0}

        def counting_open(*args, **kwargs):
            calls["n"] += 1
            return real_open(*args, **kwargs)

        with patch("omlx.utils.image.Image.open", side_effect=counting_open):
            load_image(uri_a)
            load_image(uri_b)
            load_image(uri_a)  # cache hit, no new decode

        assert calls["n"] == 2

    def test_clear_forces_redecode(self):
        """clear_image_decode_cache() drops entries so the next load decodes."""
        uri = "data:image/png;base64," + _image_to_base64(_unique_image(4))
        real_open = Image.open
        calls = {"n": 0}

        def counting_open(*args, **kwargs):
            calls["n"] += 1
            return real_open(*args, **kwargs)

        with patch("omlx.utils.image.Image.open", side_effect=counting_open):
            load_image(uri)
            load_image(uri)
            from omlx.utils.image import clear_image_decode_cache

            clear_image_decode_cache()
            load_image(uri)

        assert calls["n"] == 2

    def test_cached_pixels_match_source(self):
        """A cache hit returns pixels identical to a cold decode."""
        src = _unique_image(5)
        uri = "data:image/png;base64," + _image_to_base64(src)
        cold = load_image(uri)
        warm = load_image(uri)
        assert cold.mode == warm.mode == "RGB"
        assert cold.tobytes() == warm.tobytes()


class TestDecodeCacheFollowup:
    @pytest.fixture(autouse=True)
    def isolate_cache(self):
        from omlx.utils.image import clear_image_decode_cache

        clear_image_decode_cache()
        yield
        clear_image_decode_cache()

    def test_rgb_storage_budget_evicts_at_four_bytes_per_pixel(self, monkeypatch):
        from omlx.utils import image as module

        monkeypatch.setattr(
            module,
            "_IMAGE_DECODE_CACHE_MAX_BYTES",
            (4 * 24 * 24 + 24 * struct.calcsize("P")),
        )
        sources = [
            "data:image/png;base64," + _image_to_base64(_unique_image(seed))
            for seed in (91, 92)
        ]
        load_image(sources[0])
        assert module._image_decode_cache_bytes == (
            4 * 24 * 24 + 24 * struct.calcsize("P")
        )
        load_image(sources[1])
        with patch.object(Image, "open", wraps=Image.open) as opened:
            load_image(sources[0])
        assert opened.call_count == 1

    def test_over_capacity_history_preserves_hits_and_image_order(self, monkeypatch):
        from omlx.utils import image as module

        # Three screenshots with room for only two decoded images.
        monkeypatch.setattr(
            module,
            "_IMAGE_DECODE_CACHE_MAX_BYTES",
            2 * (4 * 24 * 24 + 24 * struct.calcsize("P")),
        )
        originals = [_unique_image(seed) for seed in (101, 102, 103)]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + _image_to_base64(img)
                        },
                    },
                ],
            }
            for img in originals
        ]
        extract_images_from_messages(messages)
        for _ in range(3):
            with patch.object(Image, "open", wraps=Image.open) as opened:
                text, images, audio = extract_images_from_messages(messages)
            assert opened.call_count == 1
            assert [img.tobytes() for img in images] == [
                img.tobytes() for img in originals
            ]
            assert text == [{"role": "user", "content": "Describe"}] * 3
            assert audio == []
            assert module._image_decode_cache_bytes <= 2 * (
                4 * 24 * 24 + 24 * struct.calcsize("P")
            )

    def test_clear_during_decode_does_not_repopulate_cache(self):
        from omlx.utils import image as module

        entered, resume = Event(), Event()
        original = _unique_image(201)
        source = "data:image/png;base64," + _image_to_base64(original)
        real_open = Image.open

        def blocked_open(*args, **kwargs):
            entered.set()
            assert resume.wait(5)
            return real_open(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                with patch.object(Image, "open", side_effect=blocked_open):
                    future = pool.submit(load_image, source)
                    assert entered.wait(5)
                    module.clear_image_decode_cache()
                    resume.set()
                    image = future.result(timeout=5)
            finally:
                resume.set()
        assert image.tobytes() == original.tobytes()
        assert not module._image_decode_cache
        assert module._image_decode_cache_bytes == 0
        load_image(source)
        assert module._image_decode_cache

    def test_concurrent_duplicate_inserts_keep_exact_budget(self):
        from omlx.utils import image as module

        barrier = Barrier(4)
        real_open = Image.open
        original = _unique_image(301)
        source = "data:image/png;base64," + _image_to_base64(original)

        def concurrent_open(*args, **kwargs):
            barrier.wait(timeout=5)
            return real_open(*args, **kwargs)

        with (
            patch.object(Image, "open", side_effect=concurrent_open),
            ThreadPoolExecutor(max_workers=4) as pool,
        ):
            images = list(pool.map(load_image, [source] * 4))
        assert all(image.tobytes() == original.tobytes() for image in images)
        assert len(module._image_decode_cache) == 1
        assert module._image_decode_cache_bytes == (
            4 * 24 * 24 + 24 * struct.calcsize("P")
        )

    def test_oversized_image_is_returned_without_evicting_existing_hit(
        self, monkeypatch
    ):
        from omlx.utils import image as module

        monkeypatch.setattr(module, "_IMAGE_DECODE_CACHE_MAX_BYTES", 2500)
        small = "data:image/png;base64," + _image_to_base64(_unique_image(401))
        large = "data:image/png;base64," + _image_to_base64(_unique_image(402, 48, 48))
        load_image(small)
        assert load_image(large).size == (48, 48)
        with patch.object(Image, "open", wraps=Image.open) as opened:
            load_image(small)
        assert opened.call_count == 0
        assert len(module._image_decode_cache) == 1


# =============================================================================
# Tests: Image size validation and downscaling (Issue #3650)
# =============================================================================


class TestImageSizeAndDownscaling:
    """Tests for payload size limits, decompression bomb guards, and aspect downscaling."""

    @pytest.fixture(autouse=True)
    def isolate_cache(self):
        from omlx.utils.image import clear_image_decode_cache

        clear_image_decode_cache()
        yield
        clear_image_decode_cache()

    def test_rejects_oversized_payload(self, monkeypatch):
        """Images exceeding max payload bytes are rejected before decode."""
        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_bytes", lambda: 100)
        img = _make_test_image(64, 64, "blue")
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        with pytest.raises(InvalidRequestError, match="exceeds the maximum allowed limit"):
            load_image(uri)

    def test_rejects_oversized_encoded_length_early(self, monkeypatch):
        """Massive base64 strings are rejected early before b64decode."""
        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_bytes", lambda: 100)
        fake_b64 = "A" * 2000
        uri = f"data:image/png;base64,{fake_b64}"

        with pytest.raises(InvalidRequestError, match="exceeds the maximum allowed limit"):
            load_image(uri)

    def test_downscales_oversized_width_preserving_aspect(self, monkeypatch):
        """Wide image exceeding max side length is downscaled preserving aspect ratio."""
        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_side_length", lambda: 1024)
        img = _make_test_image(2048, 1024, "red")
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        loaded = load_image(uri)
        assert loaded.size == (1024, 512)

    def test_downscales_oversized_height_preserving_aspect(self, monkeypatch):
        """Tall image exceeding max side length is downscaled preserving aspect ratio."""
        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_side_length", lambda: 1024)
        img = _make_test_image(1024, 2048, "green")
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        loaded = load_image(uri)
        assert loaded.size == (512, 1024)

    def test_preserves_dimensions_within_limit(self, monkeypatch):
        """Images within limits are not resized."""
        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_side_length", lambda: 2048)
        img = _make_test_image(800, 600, "yellow")
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        loaded = load_image(uri)
        assert loaded.size == (800, 600)

    def test_downscaling_disabled_when_side_limit_zero(self, monkeypatch):
        """Setting max side length to 0 disables downscaling."""
        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_side_length", lambda: 0)
        img = _make_test_image(3000, 1500, "purple")
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        loaded = load_image(uri)
        assert loaded.size == (3000, 1500)

    def test_decompression_bomb_raises_invalid_request_error(self, monkeypatch):
        """Decompression bombs detected by Pillow raise InvalidRequestError."""
        from PIL import Image as PILImage

        monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", 50)
        img = _make_test_image(20, 20, "red")  # 400 pixels > 50
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        with pytest.raises(InvalidRequestError, match="decompression bomb detected"):
            load_image(uri)

    @staticmethod
    def _gradient_image(width, height, mode):
        """Image with a pattern that shows filtering, orientation, and alpha."""
        img = Image.new("RGB", (width, height))
        img.putdata(
            [
                ((x * 7) % 256, (y * 5) % 256, 0 if (x // 3) % 2 else 255)
                for y in range(height)
                for x in range(width)
            ]
        )
        if mode == "P":
            return img.convert("P", palette=Image.Palette.ADAPTIVE, colors=16)
        if mode == "P;transparency":
            pal = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=16)
            pal.info["transparency"] = 0
            return pal
        if mode == "RGBA":
            rgba = img.convert("RGBA")
            rgba.putalpha(Image.linear_gradient("L").resize(img.size))
            return rgba
        return img.convert(mode)

    @staticmethod
    def _reference_decode(img_bytes, max_side):
        """The previous loader: orient and convert at full size, then resize."""
        from PIL import ImageOps

        rgb = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes)))
        rgb = rgb.convert("RGB")
        rgb.thumbnail((max_side, max_side), resample=Image.Resampling.LANCZOS)
        return rgb

    @pytest.mark.parametrize(
        "mode", ["RGB", "L", "1", "P", "P;transparency", "RGBA", "LA", "I;16"]
    )
    @pytest.mark.parametrize("orientation", [None, 6])
    def test_resize_first_matches_resizing_after_conversion(
        self, monkeypatch, mode, orientation
    ):
        """Resizing before orientation and RGB conversion gives the same pixels."""
        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_side_length", lambda: 32)
        img = self._gradient_image(120, 80, mode)
        buf = io.BytesIO()
        if orientation is None:
            img.save(buf, format="PNG")
        else:
            exif = Image.Exif()
            exif[0x0112] = orientation
            img.save(buf, format="PNG", exif=exif)
        img_bytes = buf.getvalue()

        loaded = load_image(
            "data:image/png;base64," + base64.b64encode(img_bytes).decode()
        )
        expected = self._reference_decode(img_bytes, 32)
        assert loaded.size == expected.size
        assert loaded.size == ((32, 21) if orientation is None else (21, 32))
        if orientation is None:
            assert loaded.tobytes() == expected.tobytes()
        else:
            # A 90-degree turn swaps the order of the two resize passes, so
            # hard edges can round differently.
            diff = [abs(a - b) for a, b in zip(loaded.tobytes(), expected.tobytes())]
            assert max(diff) <= 8
            assert sum(diff) / len(diff) < 1

    def test_palette_image_is_filtered_not_sampled(self, monkeypatch):
        """Thin lines in a palette image survive downscaling as gray, not vanish."""
        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_side_length", lambda: 64)
        img = Image.new("L", (192, 192), 255)
        for x in range(0, 192, 6):
            img.paste(0, (x, 0, x + 1, 192))
        b64 = _image_to_base64(img.convert("P"))

        loaded = load_image(f"data:image/png;base64,{b64}")
        assert loaded.size == (64, 64)
        # NEAREST would land between the lines and return pure white.
        assert min(loaded.convert("L").getdata()) < 240

    def test_jpeg_decodes_at_reduced_scale(self, monkeypatch):
        """JPEGs use decoder-assisted downscaling and keep EXIF orientation."""
        from PIL import ImageFile

        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_side_length", lambda: 64)
        decoded = []
        original_load = ImageFile.ImageFile.load

        def recording_load(image):
            decoded.append(image.size)
            return original_load(image)

        monkeypatch.setattr(ImageFile.ImageFile, "load", recording_load)
        ramp = Image.linear_gradient("L")
        img = Image.merge(
            "RGB",
            (ramp, ramp.rotate(90), ramp.transpose(Image.Transpose.FLIP_TOP_BOTTOM)),
        ).resize((512, 256))
        exif = Image.Exif()
        exif[0x0112] = 6
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif, quality=95)
        img_bytes = buf.getvalue()

        loaded = load_image(
            "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()
        )
        assert loaded.size == (32, 64)
        # 1/8 DCT scaling gives exactly the 64x32 target before orientation.
        assert decoded and set(decoded) == {(64, 32)}

        expected = self._reference_decode(img_bytes, 64)
        diff = [abs(a - b) for a, b in zip(loaded.tobytes(), expected.tobytes())]
        assert sum(diff) / len(diff) < 8

    def test_images_in_pillow_warning_range_are_downscaled(self, monkeypatch):
        """Images between Pillow's warning and error limits still load."""
        from omlx.utils import image as module

        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 300)
        monkeypatch.setattr(module, "get_max_image_side_length", lambda: 10)
        b64 = _image_to_base64(_make_test_image(20, 20))  # 400 pixels

        with pytest.warns(Image.DecompressionBombWarning):
            loaded = load_image(f"data:image/png;base64,{b64}")
        assert loaded.size == (10, 10)

    def test_extract_images_from_messages_downscales_oversized(self, monkeypatch):
        """extract_images_from_messages downscales oversized images in messages."""
        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_side_length", lambda: 512)
        img = _make_test_image(1024, 512, "blue")
        b64 = _image_to_base64(img)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": "Describe"},
                ],
            }
        ]

        text_msgs, images, audio = extract_images_from_messages(messages)
        assert len(images) == 1
        assert images[0].size == (512, 256)

    def test_extract_images_from_messages_rejects_oversized_payload(self, monkeypatch):
        """extract_images_from_messages rejects images exceeding payload limits."""
        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_bytes", lambda: 100)
        img = _make_test_image(64, 64, "blue")
        b64 = _image_to_base64(img)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ]

        with pytest.raises(InvalidRequestError, match="exceeds the maximum allowed limit"):
            extract_images_from_messages(messages)

    def test_cache_stores_downscaled_image_and_accounts_accurately(self, monkeypatch):
        """Cache holds downscaled image and byte accounting reflects downscaled size."""
        from omlx.utils import image as module

        monkeypatch.setattr(module, "get_max_image_side_length", lambda: 100)
        img = _make_test_image(400, 200, "cyan")
        b64 = _image_to_base64(img)
        uri = f"data:image/png;base64,{b64}"

        loaded = load_image(uri)
        assert loaded.size == (100, 50)
        expected_bytes = module._decoded_pixel_bytes(loaded)
        assert module._image_decode_cache_bytes == expected_bytes

        # Subsequent fetch hits cache without redecoding
        with patch.object(Image, "open", wraps=Image.open) as opened:
            cached = load_image(uri)
        assert opened.call_count == 0
        assert cached.size == (100, 50)

    @pytest.mark.parametrize("cli_override", [False, True])
    def test_resolved_settings_control_image_processing(
        self, monkeypatch, tmp_path, cli_override
    ):
        from argparse import Namespace

        from omlx import settings as settings_module
        from omlx.utils.image import get_max_image_bytes, get_max_image_side_length

        monkeypatch.setattr(settings_module, "_global_settings", None)
        monkeypatch.setenv("OMLX_MAX_IMAGE_UPLOAD_SIZE", "20MB")
        monkeypatch.setenv("OMLX_MAX_IMAGE_SIDE_LENGTH", "1500")
        args = Namespace(max_image_upload_size="30MB", max_image_side_length=512)
        settings_module.init_settings(
            base_path=tmp_path, cli_args=args if cli_override else None
        )
        side = 512 if cli_override else 1500
        assert get_max_image_bytes() == (30 if cli_override else 20) * 1024 * 1024
        assert get_max_image_side_length() == side
        uri = "data:image/png;base64," + _image_to_base64(
            _make_test_image(3000, 1500, "blue")
        )
        assert load_image(uri).size == (side, side // 2)

    def test_uninitialized_settings_use_defaults(self, monkeypatch):
        from omlx import settings as settings_module
        from omlx.utils.image import get_max_image_bytes, get_max_image_side_length

        monkeypatch.setattr(settings_module, "_global_settings", None)
        assert get_max_image_bytes() == 50 * 1024 * 1024
        assert get_max_image_side_length() == 2048
