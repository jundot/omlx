# SPDX-License-Identifier: Apache-2.0
"""
Speech tasks for LFM3-Audio and LFM3-Omni on the VLM engine's loaded model.

The model card selects each task with a system prompt: ASR serves
/v1/audio/transcriptions, TTS serves /v1/audio/speech and a spoken reply
serves /v1/audio/process. Chat requests keep the batched VLM path. Each task
runs mlx-vlm's single-prompt speech loop in one call on the engine's own MLX
executor, between scheduler steps.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import numpy as np
from mlx_vlm.generate.audio import generate_audio
from mlx_vlm.prompt_utils import apply_chat_template

from .audio_utils import audio_to_wav_bytes

if TYPE_CHECKING:
    from .vlm import VLMBatchedEngine

logger = logging.getLogger(__name__)

SPEECH_MODEL_TYPES = frozenset({"lfm3_audio", "lfm3_omni"})

# The speech languages of the model card, keyed by ISO 639-1 code. Prompts
# name a language in English.
LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "sv": "Swedish",
    "tr": "Turkish",
    "zh": "Chinese",
}
VOICES = ("female", "male")
ASR_PROMPT = "Perform ASR."
REPLY_PROMPT = "Respond with interleaved text and audio."
_SILENCE = np.zeros(1600, dtype=np.float32)


def language_name(language: str | None) -> str | None:
    """Return the English name of an ISO code or name the model speaks."""
    if not language:
        return None
    key = language.strip().lower().replace("_", "-")
    if key in LANGUAGES:
        return LANGUAGES[key]
    if key.split("-")[0] in LANGUAGES:
        return LANGUAGES[key.split("-")[0]]
    for name in LANGUAGES.values():
        if key == name.lower():
            return name
    if key in ("american english", "british english"):
        return key.title()
    raise ValueError(
        f"language {language!r} is not one of the model's speech languages: "
        + ", ".join(sorted(LANGUAGES))
    )


def tts_prompt(language: str | None, voice: str | None) -> str:
    """Build the TTS system prompt; English needs an American or British locale."""
    name = language_name(language) or "English"
    if name == "English":
        british = (language or "").strip().lower().replace("_", "-") in (
            "en-gb",
            "british english",
        )
        name = "British English" if british else "American English"
    gender = (voice or VOICES[0]).strip().lower()
    if gender not in VOICES:
        raise ValueError(f"voice {voice!r} is not one of {', '.join(VOICES)}")
    return f"Perform TTS in {name} using a {gender} voice."


class LFM3Speech:
    """ASR, TTS and spoken replies on a loaded LFM3-Audio or LFM3-Omni model."""

    def __init__(self, engine: "VLMBatchedEngine"):
        self._engine = engine

    def _run(
        self,
        system: str,
        text: str,
        audio: str | None,
        generation_mode: str,
        **kwargs: Any,
    ):
        model = self._engine._vlm_model
        processor = self._engine._processor
        prompt = apply_chat_template(
            processor,
            model.config,
            [{"role": "system", "content": system}, text],
            num_audios=1 if audio else 0,
        )
        return generate_audio(
            model,
            processor,
            prompt,
            audio=[audio] if audio else None,
            generation_mode=generation_mode,
            **kwargs,
        )

    async def _generate(self, kind: str, *args: Any, **kwargs: Any):
        engine_core = self._engine._engine
        if engine_core is None:
            raise RuntimeError("Engine not started. Call start() first.")
        start = time.monotonic()
        loop = asyncio.get_running_loop()
        # Keeps the pool from evicting the model while the speech loop runs.
        self._engine._speech_active += 1
        try:
            result = await loop.run_in_executor(
                engine_core._mlx_executor, lambda: self._run(*args, **kwargs)
            )
        finally:
            self._engine._speech_active -= 1
        logger.info(
            "LFM3 %s: model=%s, %d prompt tokens, %d slots, %.2fs",
            kind,
            self._engine.model_name,
            result.prompt_tokens,
            result.generation_tokens,
            time.monotonic() - start,
        )
        return result

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Transcribe a clip. ``prompt`` biasing is not supported and is ignored."""
        name = language_name(language)
        system = f"Perform ASR in {name}." if name else ASR_PROMPT
        options = {"max_tokens": max_tokens} if max_tokens else {}
        result = await self._generate(
            "transcribing", system, "", audio_path, "text", **options
        )
        return {
            "text": result.text.strip(),
            "language": name,
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
        }

    async def transcribe_stream(
        self, audio_path: str, **kwargs: Any
    ) -> AsyncIterator[dict]:
        """Yield the whole transcript as one chunk."""
        yield await self.transcribe(audio_path, **kwargs)

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
        ref_audio: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Speak ``text``. Sampling options apply to the speech codes."""
        if ref_audio:
            raise ValueError("LFM3 speaks with its own voices and does not clone one")
        system = tts_prompt(language, voice)
        options = {
            key: value
            for key, value in (
                ("audio_temperature", temperature),
                ("audio_top_k", top_k),
                ("audio_repetition_penalty", repetition_penalty),
                ("max_tokens", max_tokens),
            )
            if value is not None
        }
        result = await self._generate(
            "synthesizing speech", system, text, None, "audio", **options
        )
        if result.audio is None or result.audio.size == 0:
            raise RuntimeError("TTS model produced no audio output")
        return audio_to_wav_bytes(result.audio, result.sample_rate)

    async def process(self, audio_path: str, **kwargs: Any) -> bytes:
        """Answer one spoken turn with speech."""
        result = await self._generate(
            "replying", REPLY_PROMPT, "", audio_path, "interleaved"
        )
        if result.audio is None or result.audio.size == 0:
            return audio_to_wav_bytes(_SILENCE, result.sample_rate)
        return audio_to_wav_bytes(result.audio, result.sample_rate)
