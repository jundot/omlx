# SPDX-License-Identifier: Apache-2.0
"""
Image generation engine for oMLX.

This module provides an engine for image generation (Text-to-Image, Image-to-Image)
using mflux or other MLX-based diffusion models.

Unlike LLM engines, ImageEngine doesn't support streaming or chat completion.
mflux is imported lazily inside start() to avoid module-level import errors
when mflux is not installed.
"""

import asyncio
import base64
import gc
import io
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx

from ..engine_core import get_mlx_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class ImageGenerationOutput:
    """
    Output from image generation.

    Contains the generated PIL Image and generation parameters.
    """

    image: Any  # mflux.GeneratedImage (contains PIL.Image in .image attribute)
    """The generated image (mflux GeneratedImage with PIL.Image in .image attribute)."""

    prompt: str
    """The prompt used for generation."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    num_inference_steps: int
    """Number of denoising steps used."""

    seed: Optional[int] = None
    """Random seed used (if any)."""

    negative_prompt: Optional[str] = None
    """Negative prompt used (if any)."""

    guidance_scale: float = 3.5
    """Guidance scale used."""

    def to_base64(self, format: str = "PNG") -> str:
        """
        Convert image to base64 string.

        Args:
            format: Image format (PNG, JPEG, etc.)

        Returns:
            Base64-encoded image string
        """
        # Extract PIL.Image from mflux GeneratedImage if needed
        pil_image = self._get_pil_image()
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def to_bytes(self, format: str = "PNG") -> bytes:
        """
        Convert image to bytes.

        Args:
            format: Image format (PNG, JPEG, etc.)

        Returns:
            Image as bytes
        """
        # Extract PIL.Image from mflux GeneratedImage if needed
        pil_image = self._get_pil_image()
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return buffer.getvalue()

    def _get_pil_image(self) -> Any:
        """
        Extract PIL.Image from the stored image object.

        mflux returns GeneratedImage which has .image attribute.
        If already a PIL.Image, return directly.

        Returns:
            PIL.Image.Image object
        """
        # Check if it's a mflux GeneratedImage (has .image attribute)
        if hasattr(self.image, "image") and hasattr(self.image, "model_config"):
            return self.image.image
        return self.image


class ImageEngine(BaseNonStreamingEngine):
    """
    Engine for image generation (Text-to-Image, Image-to-Image).

    This engine wraps mflux models and provides async methods
    for integration with the oMLX server.

    Supports:
    - Text-to-Image (T2I): Generate images from text prompts
    - Image-to-Image (I2I): Transform existing images based on prompts

    Supported models (via mflux):
    - Z-Image Turbo (default, fast)
    - Z-Image
    - FLUX.1-schnell, FLUX.1-dev
    - FLUX.2
    - FIBO
    - SeedVR2
    - Qwen Image
    """

    # Unified model alias mapping.
    #
    # This mapping provides aliases for common model names and config types.
    # It supports both user-friendly names (e.g., "z-image-turbo", "flux.1-schnell")
    # and config.json model_type values (e.g., "flux", "z_image").
    #
    # The actual model discovery is dynamic - new mflux models are automatically
    # detected via config_model_type from the model's config.json.
    #
    # To add a new alias:
    # Add an entry here: "alias-name": ("module.path", "ClassName")
    #
    # To support a new mflux model family without aliasing:
    # Just use the HuggingFace repo name directly - it will be auto-detected.
    MODEL_ALIAS_MAP = {
        # Z-Image (fastest)
        "z-image-turbo": ("mflux.models.z_image", "ZImageTurbo"),
        "zimage-turbo": ("mflux.models.z_image", "ZImageTurbo"),
        "z_image_turbo": ("mflux.models.z_image", "ZImageTurbo"),
        "z-image": ("mflux.models.z_image", "ZImage"),
        "zimage": ("mflux.models.z_image", "ZImageTurbo"),
        "z_image": ("mflux.models.z_image", "ZImageTurbo"),
        # FLUX.1 - uses Flux1 class from txt2img module
        "flux": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
        "flux1": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
        "flux_1": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
        "flux.1-schnell": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
        "flux1-schnell": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
        "flux_schnell": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
        "flux.1-dev": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
        "flux1-dev": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
        "flux_dev": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
        "flux.1": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
        # FLUX.2
        "flux2": ("mflux.models.flux2", "Flux2Klein"),
        "flux_2": ("mflux.models.flux2", "Flux2Klein"),
        "flux.2": ("mflux.models.flux2", "Flux2Klein"),
        # FIBO
        "fibo": ("mflux.models.fibo", "FIBO"),
        # SeedVR2
        "seedvr": ("mflux.models.seedvr2", "SeedVR2"),
        "seedvr2": ("mflux.models.seedvr2", "SeedVR2"),
        # Qwen Image
        "qwen-image": ("mflux.models.qwen", "QwenImage"),
        "qwen_image": ("mflux.models.qwen", "QwenImage"),
    }

    # Default model class for unknown names
    DEFAULT_MODEL = ("mflux.models.z_image", "ZImageTurbo")

    def __init__(self, model_name: str, config_model_type: str = "", **kwargs):
        """
        Initialize the image engine.

        Args:
            model_name: Model identifier (e.g., "flux.1-schnell", "z-image-turbo", or local path)
            config_model_type: Model type from config.json (used for class resolution)
            **kwargs: Additional model-specific parameters
                - quantize: Quantization bits (4, 8)
                - model_path: Custom model path
        """
        super().__init__()
        self._model_name = model_name
        self._model = None
        self._kwargs = kwargs
        self._config_model_type = config_model_type
        self._model_module, self._model_class_name = self._resolve_model_class(model_name, config_model_type)

    def _resolve_model_class(self, model_name: str, config_model_type: str = "") -> tuple:
        """Resolve model name to (module_path, class_name).

        Priority:
        1. config_model_type from config.json (most reliable for local models)
        2. Direct name mapping
        3. Fuzzy name matching
        """
        # Normalize model_type from config.json
        type_normalized = config_model_type.lower().replace("-", "_").replace(".", "_")

        # Try config_model_type first
        if type_normalized in self.MODEL_ALIAS_MAP:
            return self.MODEL_ALIAS_MAP[type_normalized]

        # Normalize model name for lookup
        name_lower = model_name.lower().replace("_", "-").replace(".", "-")

        # Direct name mapping
        if name_lower in self.MODEL_ALIAS_MAP:
            return self.MODEL_ALIAS_MAP[name_lower]

        # Fuzzy matching on model name
        if "turbo" in name_lower or "schnell" in name_lower:
            return self.MODEL_ALIAS_MAP["z-image-turbo"]
        if "flux" in name_lower and "2" in name_lower:
            return self.MODEL_ALIAS_MAP["flux2"]
        if "flux" in name_lower:
            return self.MODEL_ALIAS_MAP["flux"]
        if "qwen" in name_lower:
            return self.MODEL_ALIAS_MAP["qwen-image"]
        if "fibo" in name_lower:
            return self.MODEL_ALIAS_MAP["fibo"]
        if "seed" in name_lower:
            return self.MODEL_ALIAS_MAP["seedvr2"]
        if "z" in name_lower and "image" in name_lower:
            return self.MODEL_ALIAS_MAP["z-image-turbo"]

        # Default
        logger.warning(f"Unknown model '{model_name}' (type='{config_model_type}'), defaulting to ZImageTurbo")
        return self.DEFAULT_MODEL

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    async def start(self) -> None:
        """
        Start the engine (load model if not loaded).

        Model loading runs on the global MLX executor to avoid Metal
        command buffer races with concurrent operations.

        mflux is imported here (lazily) to avoid module-level errors
        when the package is not installed.
        """
        if self._model is not None:
            return

        logger.info(f"Starting image engine: {self._model_name}")

        def _load_sync():
            # Dynamic import of the model class
            import importlib
            import os
            from pathlib import Path

            try:
                module = importlib.import_module(self._model_module)
                model_class = getattr(module, self._model_class_name)
            except (ImportError, ModuleNotFoundError) as e:
                raise RuntimeError(
                    f"Failed to import mflux module '{self._model_module}'. "
                    f"Please ensure mflux is installed and supports this model. Error: {e}"
                ) from e
            except AttributeError as e:
                raise RuntimeError(
                    f"Model class '{self._model_class_name}' not found in module '{self._model_module}'. "
                    f"This may indicate a version mismatch or unsupported model. Error: {e}"
                ) from e

            # Build constructor kwargs
            init_kwargs = {}
            if "quantize" in self._kwargs:
                init_kwargs["quantize"] = self._kwargs["quantize"]

            # Check if model_name is a local path (directory exists)
            local_path = Path(self._model_name)
            if local_path.is_dir():
                # Local model directory - pass as model_path
                init_kwargs["model_path"] = self._model_name
            elif "/" in self._model_name:
                # HuggingFace repo name (e.g., "mlx-community/Flux-1.lite-8B-MLX-Q4")
                # Try to get base_model from HuggingFace metadata
                try:
                    # Respect HF_ENDPOINT environment variable
                    hf_endpoint = os.environ.get("HF_ENDPOINT")
                    if hf_endpoint:
                        os.environ["HF_ENDPOINT"] = hf_endpoint

                    from huggingface_hub import model_info
                    info = model_info(self._model_name)
                    # Infer base_model from tags (e.g., "flux-1.schnell", "flux-1.dev")
                    base_model = None
                    for tag in info.tags or []:
                        tag_lower = tag.lower()
                        if "schnell" in tag_lower:
                            base_model = "schnell"
                            break
                        elif "dev" in tag_lower and "flux" in tag_lower:
                            base_model = "dev"
                            break
                        elif "flux.2" in tag_lower or "flux2" in tag_lower:
                            base_model = "flux2"
                            break

                    # Use ModelConfig with inferred base_model
                    if base_model:
                        try:
                            from mflux.models.common.config.model_config import ModelConfig
                            model_config = ModelConfig.from_name(
                                model_name=self._model_name,
                                base_model=base_model,
                            )
                            return model_class(
                                quantize=init_kwargs.get("quantize"),
                                model_path=self._model_name,
                                model_config=model_config,
                            )
                        except Exception as e2:
                            logger.warning(f"Failed to create ModelConfig: {e2}, using model_path")
                except Exception as e:
                    logger.warning(f"Failed to get HF metadata: {e}, using model_path")

                # Fallback: pass HF repo name as model_path
                init_kwargs["model_path"] = self._model_name

            return model_class(**init_kwargs)

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(get_mlx_executor(), _load_sync)

        logger.info(f"Image engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is not None:
            logger.info(f"Stopping image engine: {self._model_name}")
            self._model = None

            # Clear MLX cache on the global executor to avoid Metal races
            gc.collect()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
            )
            logger.info(f"Image engine stopped: {self._model_name}")

    async def generate_image(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        image: Optional[str] = None,
        strength: float = 0.8,
    ) -> ImageGenerationOutput:
        """
        Generate an image from a text prompt (T2I) or transform an image (I2I).

        Args:
            prompt: Text description of the desired image
            negative_prompt: Things to exclude from the image
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps (more = higher quality, slower)
            guidance_scale: Strength of prompt guidance (higher = more prompt adherence)
            seed: Random seed for reproducibility
            image: Base64-encoded source image for I2I (None for T2I)
            strength: How much to transform the image (0.0-1.0, I2I only)

        Returns:
            ImageGenerationOutput containing the generated image
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        with self._active_lock:
            self._active_count += 1

        tmp_path = None
        try:
            # Use random seed if not provided
            if seed is None:
                seed = random.randint(0, 2**32 - 1)

            # Build generation kwargs for mflux API
            gen_kwargs = {
                "seed": seed,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
            }

            # Add optional parameters
            if guidance_scale is not None:
                gen_kwargs["guidance"] = guidance_scale
            if negative_prompt is not None:
                gen_kwargs["negative_prompt"] = negative_prompt

            # Handle I2I (image-to-image)
            if image is not None:
                # Decode base64 image and save to temp file
                # mflux expects image_path, not direct image
                import os
                import tempfile

                from PIL import Image as PILImage

                # Decode base64 image
                if image.startswith("data:"):
                    image = image.split(",", 1)[1]

                image_bytes = base64.b64decode(image)
                pil_image = PILImage.open(io.BytesIO(image_bytes))

                # Save to temp file (will be cleaned up in finally block)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    pil_image.save(tmp_path, format="PNG")
                    gen_kwargs["image_path"] = tmp_path
                    gen_kwargs["image_strength"] = strength

            # Run generation on MLX executor
            def _generate_sync():
                return self._model.generate_image(**gen_kwargs)

            loop = asyncio.get_running_loop()
            result_image = await loop.run_in_executor(
                get_mlx_executor(), _generate_sync
            )

            return ImageGenerationOutput(
                image=result_image,
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                seed=seed,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
            )

        finally:
            # Clean up I2I temp file
            if tmp_path is not None:
                try:
                    import os
                    os.unlink(tmp_path)
                except OSError:
                    pass

            # Decrement active count and clear cache if this was the last request
            if self._decrement_active():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.

        Returns:
            Dictionary containing engine statistics
        """
        return {
            "model_name": self._model_name,
            "model_class": self._model_class_name,
            "loaded": self._model is not None,
            "active_requests": self._active_count,
            "quantize": self._kwargs.get("quantize"),
        }