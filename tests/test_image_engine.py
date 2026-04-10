# SPDX-License-Identifier: Apache-2.0
"""
Tests for ImageEngine.

These tests verify the image generation engine functionality.
Tests marked with @pytest.mark.slow require real model loading.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import base64
from io import BytesIO

from omlx.engine.image import ImageEngine, ImageGenerationOutput


class TestImageEngine:
    """Unit tests for ImageEngine."""

    def test_init(self):
        """Test engine initialization."""
        engine = ImageEngine("flux.1-schnell")
        assert engine.model_name == "flux.1-schnell"
        assert engine._model is None

    def test_init_with_quantization(self):
        """Test engine initialization with quantization."""
        engine = ImageEngine("flux.1-schnell", quantize=8)
        assert engine.model_name == "flux.1-schnell"
        assert engine._kwargs.get("quantize") == 8

    @pytest.mark.asyncio
    async def test_start_lazy_import(self):
        """Test that start() lazily imports mflux."""
        # Skip this test if mflux is actually installed
        # (we can test with real mflux instead)
        try:
            import mflux
            pytest.skip("mflux is installed, skipping mock test")
        except ImportError:
            pass

        engine = ImageEngine("z-image-turbo")

        # Mock the module structure for testing without mflux installed
        mock_model_class = MagicMock()
        mock_module = MagicMock()
        mock_module.ZImageTurbo = mock_model_class

        with patch.dict("sys.modules", {"mflux.models.z_image": mock_module}):
            await engine.start()
            # Model should be loaded
            assert engine._model is not None

    @pytest.mark.asyncio
    async def test_stop_clears_model(self):
        """Test that stop() clears the model and frees memory."""
        engine = ImageEngine("flux.1-schnell")
        engine._model = MagicMock()

        await engine.stop()

        assert engine._model is None

    def test_get_stats(self):
        """Test get_stats returns correct structure."""
        engine = ImageEngine("flux.1-schnell")
        stats = engine.get_stats()

        assert "model_name" in stats
        assert "loaded" in stats
        assert stats["model_name"] == "flux.1-schnell"

    def test_get_stats_loaded(self):
        """Test get_stats when model is loaded."""
        engine = ImageEngine("flux.1-schnell")
        engine._model = MagicMock()

        stats = engine.get_stats()

        assert stats["loaded"] is True


class TestImageEngineGeneration:
    """Tests for image generation."""

    @pytest.mark.asyncio
    async def test_generate_image_returns_pil_image(self):
        """Test that generate_image returns PIL Image."""
        engine = ImageEngine("flux.1-schnell")

        # Mock mflux
        mock_image = MagicMock()
        mock_image.size = (1024, 1024)

        mock_model = MagicMock()
        mock_model.generate_image.return_value = mock_image

        with patch.object(engine, "_model", mock_model):
            result = await engine.generate_image(
                prompt="A white cat",
                width=1024,
                height=1024,
                num_inference_steps=20,
            )

            assert result is not None
            mock_model.generate_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_image_with_seed(self):
        """Test generation with specific seed."""
        engine = ImageEngine("flux.1-schnell")

        mock_model = MagicMock()
        mock_model.generate_image.return_value = MagicMock()

        with patch.object(engine, "_model", mock_model):
            await engine.generate_image(
                prompt="A white cat",
                seed=42,
            )

            call_kwargs = mock_model.generate_image.call_args[1]
            assert call_kwargs.get("seed") == 42

    @pytest.mark.asyncio
    async def test_generate_image_with_negative_prompt(self):
        """Test generation with negative prompt."""
        engine = ImageEngine("flux.1-schnell")

        mock_model = MagicMock()
        mock_model.generate_image.return_value = MagicMock()

        with patch.object(engine, "_model", mock_model):
            await engine.generate_image(
                prompt="A white cat",
                negative_prompt="blurry, low quality",
            )

            call_kwargs = mock_model.generate_image.call_args[1]
            assert "negative_prompt" in call_kwargs

    @pytest.mark.asyncio
    async def test_generate_image_i2i(self):
        """Test image-to-image generation."""
        engine = ImageEngine("flux.1-schnell")

        # Create a minimal base64 image
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="red")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode()

        mock_model = MagicMock()
        mock_model.generate_image.return_value = MagicMock()

        with patch.object(engine, "_model", mock_model):
            await engine.generate_image(
                prompt="Make it blue",
                image=f"data:image/png;base64,{base64_image}",
                strength=0.8,
            )


class TestImageGenerationOutput:
    """Tests for ImageGenerationOutput dataclass."""

    def test_output_creation(self):
        """Test creating ImageGenerationOutput."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        output = ImageGenerationOutput(
            image=img,
            prompt="test",
            width=100,
            height=100,
            num_inference_steps=20,
            seed=42,
        )

        assert output.prompt == "test"
        assert output.width == 100
        assert output.height == 100
        assert output.seed == 42

    def test_to_base64(self):
        """Test converting output to base64."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        output = ImageGenerationOutput(
            image=img,
            prompt="test",
            width=100,
            height=100,
            num_inference_steps=20,
            seed=42,
        )

        b64 = output.to_base64()

        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_to_bytes(self):
        """Test converting output to bytes."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        output = ImageGenerationOutput(
            image=img,
            prompt="test",
            width=100,
            height=100,
            num_inference_steps=20,
            seed=42,
        )

        data = output.to_bytes()

        assert isinstance(data, bytes)
        assert len(data) > 0