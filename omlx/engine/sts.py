# SPDX-License-Identifier: Apache-2.0
"""
STS (Speech-to-Speech) engine for oMLX.

This module provides an engine for audio processing (speech enhancement,
speech-to-speech conversion, source separation) using mlx-audio.
Unlike LLM engines, STS engines don't support streaming or chat completion.
mlx-audio is imported lazily inside start() to avoid module-level import errors
when mlx-audio is not installed.

Supported model families (mlx-audio >=0.4.0):
- DeepFilterNet: speech enhancement / noise removal
- MossFormer2: speech enhancement
- SAMAudio: text-guided sound/speech separation
- LFM2.5-Audio: multimodal speech-to-speech generation
"""

import asyncio
import gc
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import mlx.core as mx
import numpy as np

from ..engine_core import get_mlx_executor
from .audio_utils import DEFAULT_SAMPLE_RATE as _DEFAULT_SAMPLE_RATE
from .audio_utils import audio_to_wav_bytes as _audio_to_wav_bytes
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


def _detect_sts_family(model_name: str) -> str:
    """Detect STS model family from model name/path.

    Returns one of: "deepfilternet", "mossformer2", "sam_audio", "lfm2", "generic"
    """
    name_lower = model_name.lower()
    if "deepfilter" in name_lower:
        return "deepfilternet"
    if "mossformer" in name_lower:
        return "mossformer2"
    if "sam" in name_lower and "audio" in name_lower:
        return "sam_audio"
    if "lfm" in name_lower:
        return "lfm2"
    return "generic"


# ---------------------------------------------------------------------------
# Family-specific loaders
# ---------------------------------------------------------------------------


def _load_deepfilternet(model_name: str):
    """Load a DeepFilterNet model."""
    from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel

    # DeepFilterNet-mlx uses subfolder='v3' by default; pass model_name_or_path
    return DeepFilterNetModel.from_pretrained(model_name_or_path=model_name)


def _load_mossformer2(model_name: str):
    """Load a MossFormer2 SE model."""
    from mlx_audio.sts.models.mossformer2_se import MossFormer2SEModel

    return MossFormer2SEModel.from_pretrained(model_name)


def _load_sam_audio(model_name: str):
    """Load a SAMAudio model."""
    from mlx_audio.sts.models.sam_audio import SAMAudio

    return SAMAudio.from_pretrained(model_name)


def _load_lfm2(model_name: str):
    """Load LFM2.5-Audio model and processor."""
    from mlx_audio.sts.models.lfm_audio import LFM2AudioModel, LFM2AudioProcessor

    model = LFM2AudioModel.from_pretrained(model_name)
    processor = LFM2AudioProcessor.from_pretrained(model_name)
    return model, processor


_FAMILY_LOADERS = {
    "deepfilternet": _load_deepfilternet,
    "mossformer2": _load_mossformer2,
    "sam_audio": _load_sam_audio,
    "lfm2": _load_lfm2,
}

# ---------------------------------------------------------------------------
# Family-specific processors
# ---------------------------------------------------------------------------


def _process_deepfilternet(model, audio_path: str, **kwargs) -> bytes:
    """Enhance audio with DeepFilterNet (writes to temp file, returns WAV bytes)."""
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        model.enhance_file(str(audio_path), out_path)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def _process_mossformer2(model, audio_path: str, **kwargs) -> bytes:
    """Enhance audio with MossFormer2 SE."""
    enhanced = model.enhance(str(audio_path))
    sample_rate = getattr(model.config, "sample_rate", 48000)
    return _audio_to_wav_bytes(enhanced, int(sample_rate))


def _process_sam_audio(model, audio_path: str, **kwargs) -> bytes:
    """Separate audio with SAMAudio (text-guided)."""
    descriptions = kwargs.get("descriptions", ["speech"])
    result = model.separate(
        audios=[str(audio_path)],
        descriptions=descriptions,
    )
    # SeparationResult.target is List[mx.array] in batch mode
    target = result.target[0] if isinstance(result.target, list) else result.target
    sample_rate = getattr(
        getattr(model, "config", None), "sample_rate", _DEFAULT_SAMPLE_RATE
    )
    return _audio_to_wav_bytes(target, int(sample_rate))


def _process_lfm2(model_and_processor, audio_path: str, **kwargs) -> bytes:
    """Run speech-to-speech generation with LFM2.5-Audio."""
    from mlx_audio.sts.models.lfm_audio import ChatState, LFMModality

    model, processor = model_and_processor

    # Load audio
    from mlx_audio import audio_io

    audio_np, sr = audio_io.read(str(audio_path))
    audio_mx = mx.array(audio_np.flatten(), dtype=mx.float32)

    # Build chat state: user turn with audio, then start assistant turn
    chat_state = ChatState(processor)
    chat_state.new_turn("user")
    chat_state.add_audio(audio_mx, sample_rate=sr)
    chat_state.end_turn()
    chat_state.new_turn("assistant")

    # Generation parameters
    max_new_tokens = kwargs.get("max_new_tokens", 512)
    temperature = kwargs.get("temperature", 0.7)
    audio_temperature = kwargs.get("audio_temperature", 0.8)

    # Collect audio output frames (each is shape (num_codebooks,) or (1, num_codebooks))
    audio_frames = []
    for token, modality in model.generate_from_chat_state(
        chat_state,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        audio_temperature=audio_temperature,
    ):
        if modality == LFMModality.AUDIO_OUT:
            audio_frames.append(token)

    if not audio_frames:
        # No audio generated — return silence
        return _audio_to_wav_bytes(np.zeros(1600, dtype=np.float32), _DEFAULT_SAMPLE_RATE)

    # Stack frames: each frame is (num_codebooks,) -> stack to (T, num_codebooks)
    # then transpose to (num_codebooks, T) for decode_audio
    codes = mx.stack(audio_frames, axis=0)  # (T, num_codebooks) or (T, 1, num_codebooks)
    if codes.ndim == 3:
        codes = codes.squeeze(1)  # (T, num_codebooks)
    codes = codes.transpose(1, 0)  # (num_codebooks, T)
    codes = codes[None, :, :]  # (1, num_codebooks, T)
    waveform = processor.decode_audio(codes)

    # Output at 24kHz (LFM2 default)
    sample_rate = getattr(
        getattr(model, "config", None), "sample_rate", _DEFAULT_SAMPLE_RATE
    )
    return _audio_to_wav_bytes(waveform, int(sample_rate))


_FAMILY_PROCESSORS = {
    "deepfilternet": lambda model, path, **kw: _process_deepfilternet(model, path, **kw),
    "mossformer2": lambda model, path, **kw: _process_mossformer2(model, path, **kw),
    "sam_audio": lambda model, path, **kw: _process_sam_audio(model, path, **kw),
    "lfm2": lambda model, path, **kw: _process_lfm2(model, path, **kw),
}


class STSEngine(BaseNonStreamingEngine):
    """
    Engine for speech-to-speech / audio processing (Speech-to-Speech).

    This engine wraps mlx-audio STS models and provides async methods
    for integration with the oMLX server. Each model family has its own
    loading and processing logic since mlx-audio STS models do not share
    a unified utils API (unlike STT/TTS).

    Supported families:
    - deepfilternet: speech enhancement / noise removal
    - mossformer2: speech enhancement
    - sam_audio: text-guided sound separation
    - lfm2: multimodal speech-to-speech generation
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the STS engine.

        Args:
            model_name: HuggingFace model name or local path
            **kwargs: Additional model-specific parameters
        """
        self._model_name = model_name
        self._model = None  # For lfm2, this is (model, processor) tuple
        self._family = _detect_sts_family(model_name)
        self._kwargs = kwargs

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    async def start(self) -> None:
        """Start the engine (load model if not loaded).

        Model loading runs on the global MLX executor to avoid Metal
        command buffer races with concurrent BatchGenerator steps.
        mlx-audio is imported lazily inside the family-specific loaders
        to avoid module-level errors when mlx-audio is not installed.
        """
        if self._model is not None:
            return

        family = self._family
        logger.info(f"Starting STS engine: {self._model_name} (family={family})")

        loader = _FAMILY_LOADERS.get(family)
        if loader is None:
            raise ValueError(
                f"Unsupported STS model family: {family!r}. "
                f"Supported: {sorted(_FAMILY_LOADERS)}"
            )

        model_name = self._model_name

        def _load_sync():
            try:
                return loader(model_name)
            except ImportError as exc:
                raise ImportError(
                    "mlx-audio is required for STS inference. "
                    "Install it with: pip install 'omlx[audio]'"
                ) from exc

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(get_mlx_executor(), _load_sync)
        logger.info(f"STS engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is None:
            return

        logger.info(f"Stopping STS engine: {self._model_name}")
        self._model = None

        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info(f"STS engine stopped: {self._model_name}")

    async def process(self, audio_path: str, **kwargs) -> bytes:
        """
        Process an audio file through the STS model.

        For speech enhancement models (DeepFilterNet, MossFormer2), this
        enhances / denoises the audio. For SAMAudio, this separates target
        sounds. For LFM2, this runs speech-to-speech generation.

        Args:
            audio_path: Path to the audio file to process
            **kwargs: Additional model-specific parameters:
                - descriptions (list[str]): Target descriptions for SAMAudio
                  (default: ["speech"])
                - max_new_tokens (int): Max tokens for LFM2 (default: 512)
                - temperature (float): Sampling temperature for LFM2
                - audio_temperature (float): Audio sampling temp for LFM2

        Returns:
            WAV-encoded bytes (RIFF header + 16-bit mono PCM) of processed audio
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        import time

        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        logger.info(
            "STS process: model=%s, family=%s, file=%s (%d bytes)",
            self._model_name, self._family,
            os.path.basename(audio_path), file_size,
        )

        family = self._family
        processor_fn = _FAMILY_PROCESSORS.get(family)
        if processor_fn is None:
            raise ValueError(
                f"Unsupported STS model family: {family!r}. "
                f"Supported: {sorted(_FAMILY_PROCESSORS)}"
            )

        model = self._model
        t0 = time.monotonic()

        def _process_sync():
            return processor_fn(model, str(audio_path), **kwargs)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(get_mlx_executor(), _process_sync)

        elapsed = time.monotonic() - t0
        logger.info(
            "STS process done: model=%s, %.2fs, %d bytes output",
            self._model_name, elapsed, len(result),
        )
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "family": self._family,
        }

    def __repr__(self) -> str:
        s = "running" if self._model is not None else "stopped"
        return f"<STSEngine model={self._model_name} family={self._family} {s}>"
