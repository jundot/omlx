# SPDX-License-Identifier: Apache-2.0
"""
Realtime speech-to-speech sessions for /v1/realtime.

A session holds one spoken conversation with a voice model that the VLM engine
already loaded. The client streams 24 kHz PCM16 turns and decides when a turn
ends. The backend answers in speech a few slots per call on the engine's MLX
executor, so chat requests on the same model keep running in between.
"""

import asyncio
import contextlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..decode_activity import get_decode_activity

if TYPE_CHECKING:
    from .vlm import VLMBatchedEngine

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
MAX_INPUT_SECONDS = 120
MIN_COMMIT_SECONDS = 0.1
# Generation time per executor call; chat steps on the same model run between calls.
STEP_SECONDS = 0.04


class RealtimeVoiceError(Exception):
    """A client error with an OpenAI Realtime error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class VoiceStep:
    """Output of one executor call: transcript text, 24 kHz audio, slots used."""

    text: str = ""
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    slots: int = 0
    done: bool = False


class VoiceBackend(ABC):
    """One conversation with a voice model. Every method runs on the executor."""

    voices: tuple[str, ...] = ()

    @classmethod
    def supports(cls, model) -> bool:
        """True when the loaded checkpoint has the modules this backend needs."""
        return True

    @classmethod
    def voice_names(cls, config: dict) -> list[str]:
        """Voices a session can pick, read from the checkpoint's config.json."""
        return list(cls.voices)

    @property
    def default_voice(self) -> str | None:
        return self.voices[0] if self.voices else None

    @abstractmethod
    def begin_turn(
        self, audio: np.ndarray, instructions: str, voice: str | None
    ) -> int:
        """Prefill one user turn (24 kHz float32) and return its prompt tokens."""

    @abstractmethod
    def step(self, seconds: float, cancel: threading.Event) -> VoiceStep:
        """Generate for about ``seconds``, stopping early when ``cancel`` is set."""

    @abstractmethod
    def stop(self) -> None:
        """End the current reply where generation stopped."""

    @abstractmethod
    def truncate(self, played_seconds: float) -> None:
        """Keep only the part of the last reply that the client played."""

    @abstractmethod
    def close(self) -> None:
        """Release the conversation state."""


@dataclass
class ResponseResult:
    status: str
    input_tokens: int = 0
    output_slots: int = 0
    # Why a failed reply failed, for the client's error event.
    error: RealtimeVoiceError | None = None


TextCallback = Callable[[str], Awaitable[None]]
AudioCallback = Callable[[np.ndarray], Awaitable[None]]
DoneCallback = Callable[[ResponseResult], Awaitable[None]]


class RealtimeVoiceSession:
    """Turn buffering, one response at a time, cancellation and truncation."""

    def __init__(self, engine: "VLMBatchedEngine", backend: VoiceBackend):
        self._engine = engine
        self._backend = backend
        self._executor = engine._engine._mlx_executor
        self._activity_key = f"realtime:{id(self):x}"
        self.instructions = ""
        self.voice = backend.default_voice
        self.max_output_tokens: int | None = None
        self._buffer = bytearray()
        self._pending: np.ndarray | None = None
        self._locked = False
        self._cancel = threading.Event()
        self._task: asyncio.Task | None = None
        self._closed = False

    @property
    def voices(self) -> tuple[str, ...]:
        return self._backend.voices

    @property
    def responding(self) -> bool:
        return self._task is not None and not self._task.done()

    def update(
        self,
        instructions: str | None = None,
        voice: str | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        # The system prompt and voice open the conversation cache.
        if self._locked and (
            (instructions is not None and instructions != self.instructions)
            or (voice is not None and voice != self.voice)
        ):
            raise RealtimeVoiceError(
                "session_locked",
                "instructions and voice cannot change after the first turn",
            )
        if voice is not None and voice not in self.voices:
            raise RealtimeVoiceError(
                "invalid_value",
                f"voice {voice!r} is not one of {', '.join(self.voices)}",
            )
        if instructions is not None:
            self.instructions = instructions
        if voice is not None:
            self.voice = voice
        if max_output_tokens is not None:
            self.max_output_tokens = max_output_tokens or None

    def append(self, pcm16: bytes) -> None:
        if len(self._buffer) + len(pcm16) > MAX_INPUT_SECONDS * SAMPLE_RATE * 2:
            raise RealtimeVoiceError(
                "input_audio_buffer_too_large",
                f"a turn holds at most {MAX_INPUT_SECONDS} s of audio",
            )
        self._buffer += pcm16

    def clear(self) -> None:
        self._buffer.clear()

    def commit(self) -> None:
        if len(self._buffer) < MIN_COMMIT_SECONDS * SAMPLE_RATE * 2:
            raise RealtimeVoiceError(
                "input_audio_buffer_commit_empty",
                f"commit needs at least {int(MIN_COMMIT_SECONDS * 1000)} ms of audio",
            )
        usable = len(self._buffer) - len(self._buffer) % 2
        audio = np.frombuffer(bytes(self._buffer[:usable]), dtype="<i2")
        audio = audio.astype(np.float32) / 32768
        self._buffer.clear()
        # Length and level tell a clear spoken turn from echo or noise.
        level = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        logger.info(
            "Realtime voice turn: %.1f s of audio at %.0f dBFS",
            audio.size / SAMPLE_RATE,
            20 * np.log10(max(level, 1e-6)),
        )
        self._pending = (
            audio if self._pending is None else np.concatenate([self._pending, audio])
        )

    def start_response(
        self, on_text: TextCallback, on_audio: AudioCallback, on_done: DoneCallback
    ) -> asyncio.Task:
        if self.responding:
            raise RealtimeVoiceError(
                "conversation_already_has_active_response",
                "a response is already in progress",
            )
        if self._pending is None:
            raise RealtimeVoiceError(
                "input_audio_buffer_empty",
                "commit a spoken turn before response.create",
            )
        audio, self._pending = self._pending, None
        self._locked = True
        self._cancel.clear()
        self._task = asyncio.create_task(
            self._respond(audio, on_text, on_audio, on_done)
        )
        return self._task

    async def _respond(
        self,
        audio: np.ndarray,
        on_text: TextCallback,
        on_audio: AudioCallback,
        on_done: DoneCallback,
    ) -> None:
        loop = asyncio.get_running_loop()
        result = ResponseResult("completed")
        started = time.perf_counter()
        first_audio: float | None = None
        sent_seconds = 0.0
        # Buffer a client needs before playback so the reply never runs dry.
        needed_buffer = 0.0
        self._engine._speech_active += 1
        try:
            result.input_tokens = await loop.run_in_executor(
                self._executor,
                self._backend.begin_turn,
                audio,
                self.instructions,
                self.voice,
            )
            while True:
                if self._cancel.is_set():
                    result.status = "cancelled"
                    break
                get_decode_activity().publish(self._activity_key, 1)
                step = await loop.run_in_executor(
                    self._executor, self._backend.step, STEP_SECONDS, self._cancel
                )
                result.output_slots += step.slots
                if step.text:
                    await on_text(step.text)
                if step.audio.size:
                    now = time.perf_counter()
                    if first_audio is None:
                        first_audio = now
                    needed_buffer = max(needed_buffer, now - first_audio - sent_seconds)
                    sent_seconds += step.audio.size / SAMPLE_RATE
                    await on_audio(step.audio)
                if step.done:
                    break
                if (
                    self.max_output_tokens is not None
                    and result.output_slots >= self.max_output_tokens
                ):
                    result.status = "incomplete"
                    break
            if result.status != "completed":
                await loop.run_in_executor(self._executor, self._backend.stop)
        except asyncio.CancelledError:
            result.status = "cancelled"
            with contextlib.suppress(Exception):
                await loop.run_in_executor(self._executor, self._backend.stop)
            raise
        except RealtimeVoiceError as exc:
            result.status, result.error = "failed", exc
            with contextlib.suppress(Exception):
                await loop.run_in_executor(self._executor, self._backend.stop)
        except Exception:
            logger.exception("Realtime voice response failed")
            result.status = "failed"
            result.error = RealtimeVoiceError(
                "server_error", "the reply failed; see the server log"
            )
            with contextlib.suppress(Exception):
                await loop.run_in_executor(self._executor, self._backend.stop)
        finally:
            get_decode_activity().remove(self._activity_key)
            self._engine._speech_active -= 1
            if first_audio is not None:
                logger.info(
                    "Realtime voice reply %s: first audio %.0f ms, %.1f s audio "
                    "in %.1f s, gapless playback needs %.0f ms buffer",
                    result.status,
                    (first_audio - started) * 1000,
                    sent_seconds,
                    time.perf_counter() - started,
                    needed_buffer * 1000,
                )
            with contextlib.suppress(Exception):
                await on_done(result)

    def cancel(self) -> None:
        self._cancel.set()

    async def _settle(self) -> None:
        task = self._task
        if task is not None and not task.done():
            self.cancel()
            # asyncio.wait does not raise, so a cancellation of the caller
            # still propagates while the reply's own outcome is ignored.
            await asyncio.wait({task})
            if not task.cancelled():
                task.exception()

    async def truncate(self, audio_end_ms: int) -> None:
        if audio_end_ms < 0:
            raise RealtimeVoiceError(
                "invalid_value", "audio_end_ms must not be negative"
            )
        await self._settle()
        logger.info("Realtime voice reply truncated after %d ms", audio_end_ms)
        loop = asyncio.get_running_loop()
        self._engine._speech_active += 1
        try:
            await loop.run_in_executor(
                self._executor, self._backend.truncate, audio_end_ms / 1000
            )
        finally:
            self._engine._speech_active -= 1

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._settle()
            loop = asyncio.get_running_loop()
            with contextlib.suppress(Exception):
                await loop.run_in_executor(self._executor, self._backend.close)
        finally:
            self._engine.release_realtime_session(self)


class TranscriptStream:
    """Text deltas of a growing reply; an incomplete UTF-8 tail waits for its bytes."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.sent = ""

    def delta(self, ids: list[int]) -> str:
        text = self.tokenizer.decode(ids, skip_special_tokens=True).rstrip("\ufffd")
        if not text.startswith(self.sent):
            return ""
        delta, self.sent = text[len(self.sent) :], text
        return delta


def continuation_turn_text(
    tokenizer, messages: list[dict], im_end: str, **template_kwargs
) -> str:
    """Template text of a later user turn, after a reply that ended with ``im_end``."""
    sentinel = "\x00"
    text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": sentinel}, messages[-1]],
        add_generation_prompt=True,
        tokenize=False,
        **template_kwargs,
    )
    tail = text[text.index(sentinel) + 1 :]
    # The pending prefix already ends the previous reply with im_end.
    if not tail.startswith(im_end):
        raise RuntimeError("unexpected chat template layout after a reply")
    return tail[len(im_end) :]


def pcm16_bytes(audio: np.ndarray) -> bytes:
    """Encode float32 audio in [-1, 1] as little-endian PCM16."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()
