# SPDX-License-Identifier: Apache-2.0
"""
Tests for OpenAI-compatible Images API models.
"""

import pytest
from pydantic import ValidationError

from omlx.api.image_models import (
    ImageRequest,
    ImageResponse,
    ImageData,
)


class TestImageRequest:
    """Tests for ImageRequest Pydantic model."""

    def test_minimal_request(self):
        """Test minimal valid request."""
        request = ImageRequest(
            model="flux.1-schnell",
            prompt="A white cat",
        )

        assert request.model == "flux.1-schnell"
        assert request.prompt == "A white cat"
        assert request.n == 1
        assert request.size == "1024x1024"
        assert request.quality == "standard"
        assert request.response_format == "b64_json"

    def test_all_parameters(self):
        """Test request with all parameters."""
        request = ImageRequest(
            model="flux.1-dev",
            prompt="A white cat",
            n=2,
            size="1792x1024",
            quality="hd",
            response_format="url",
            style="vivid",
            negative_prompt="blurry",
            seed=42,
            num_inference_steps=30,
            guidance_scale=5.0,
        )

        assert request.n == 2
        assert request.size == "1792x1024"
        assert request.quality == "hd"
        assert request.style == "vivid"
        assert request.negative_prompt == "blurry"
        assert request.seed == 42
        assert request.num_inference_steps == 30
        assert request.guidance_scale == 5.0

    def test_invalid_n_too_high(self):
        """Test n validation (max 4)."""
        with pytest.raises(ValidationError):
            ImageRequest(
                model="flux.1-schnell",
                prompt="test",
                n=10,  # Exceeds max of 4
            )

    def test_invalid_n_zero(self):
        """Test n validation (min 1)."""
        with pytest.raises(ValidationError):
            ImageRequest(
                model="flux.1-schnell",
                prompt="test",
                n=0,
            )

    def test_invalid_size(self):
        """Test size validation."""
        with pytest.raises(ValidationError):
            ImageRequest(
                model="flux.1-schnell",
                prompt="test",
                size="999x999",  # Invalid size
            )

    def test_valid_sizes(self):
        """Test all valid sizes."""
        valid_sizes = ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]

        for size in valid_sizes:
            request = ImageRequest(
                model="flux.1-schnell",
                prompt="test",
                size=size,
            )
            assert request.size == size

    def test_i2i_parameters(self):
        """Test image-to-image parameters."""
        request = ImageRequest(
            model="flux.1-schnell",
            prompt="Make it blue",
            image="data:image/png;base64,abc123",
            strength=0.8,
        )

        assert request.image == "data:image/png;base64,abc123"
        assert request.strength == 0.8

    def test_strength_validation(self):
        """Test strength range validation (0.0-1.0)."""
        # Valid
        request = ImageRequest(
            model="flux.1-schnell",
            prompt="test",
            image="data:image/png;base64,abc",
            strength=0.5,
        )
        assert request.strength == 0.5

        # Invalid - too high
        with pytest.raises(ValidationError):
            ImageRequest(
                model="flux.1-schnell",
                prompt="test",
                image="data:image/png;base64,abc",
                strength=1.5,
            )


class TestImageData:
    """Tests for ImageData Pydantic model."""

    def test_b64_json_response(self):
        """Test base64 response format."""
        data = ImageData(b64_json="abc123")

        assert data.b64_json == "abc123"
        assert data.url is None

    def test_url_response(self):
        """Test URL response format."""
        data = ImageData(url="http://example.com/image.png")

        assert data.url == "http://example.com/image.png"
        assert data.b64_json is None

    def test_with_revised_prompt(self):
        """Test with revised prompt."""
        data = ImageData(
            b64_json="abc123",
            revised_prompt="A fluffy white cat",
        )

        assert data.revised_prompt == "A fluffy white cat"


class TestImageResponse:
    """Tests for ImageResponse Pydantic model."""

    def test_response_creation(self):
        """Test creating ImageResponse."""
        response = ImageResponse(
            created=1700000000,
            data=[
                ImageData(b64_json="abc123"),
            ],
        )

        assert response.created == 1700000000
        assert len(response.data) == 1

    def test_multiple_images(self):
        """Test response with multiple images."""
        response = ImageResponse(
            created=1700000000,
            data=[
                ImageData(b64_json="img1"),
                ImageData(b64_json="img2"),
            ],
        )

        assert len(response.data) == 2


class TestSizeParsing:
    """Tests for size to width/height parsing."""

    def test_parse_size(self):
        """Test parsing size string to width/height."""
        from omlx.api.image_models import parse_size

        assert parse_size("1024x1024") == (1024, 1024)
        assert parse_size("1792x1024") == (1792, 1024)
        assert parse_size("1024x1792") == (1024, 1792)
        assert parse_size("256x256") == (256, 256)
        assert parse_size("512x512") == (512, 512)