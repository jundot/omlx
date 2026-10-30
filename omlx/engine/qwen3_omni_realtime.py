# SPDX-License-Identifier: Apache-2.0
"""
Realtime voice backend for Qwen3-Omni.

The thinker writes the reply while the talker speaks it: the talker starts
after the first reply token and reads one reply token per speech frame,
waiting for the thinker when it runs out. The talker prompt and per-frame
inputs come from mlx-vlm; the sampling follows the reference talker (special
codec ids blocked, top-k, repetition penalty, greedy code predictor), and the
talker never reads the reply's final ``<|im_end|>``, which the reference
thinker does not embed either. code2wav decodes short windows of frames.
"""

import time

import mlx.core as mx
import numpy as np
from mlx_audio.utils import resample_audio
from mlx_vlm.models.cache import KVCache, make_prompt_cache

from .realtime_voice import (
    SAMPLE_RATE,
    RealtimeVoiceError,
    TranscriptStream,
    VoiceBackend,
    VoiceStep,
    continuation_turn_text,
)

# The model card's prompt for spoken interaction, without its video sentence:
# short conversational text without markup, so the talker reads it cleanly.
SYSTEM_PROMPT = (
    "You are Qwen-Omni, a smart voice assistant created by Alibaba Qwen. "
    "You are a virtual voice assistant with no gender or age.\n"
    "You are communicating with the user.\n"
    "In user messages, \u201cI/me/my/we/our\u201d refer to the user and "
    "\u201cyou/your\u201d refer to the assistant. In your replies, address the user "
    "as \u201cyou/your\u201d and yourself as \u201cI/me/my\u201d; never mirror the "
    "user\u2019s pronouns\u2014always shift perspective. Keep original pronouns only "
    "in direct quotes; if a reference is unclear, ask a brief clarifying question.\n"
    "Interact with users using short(no more than 50 words), brief, "
    "straightforward language, maintaining a natural tone.\n"
    "Never use formal phrasing, mechanical expressions, bullet points, overly "
    "structured language. \n"
    "Your output must consist only of the spoken content you want the user to "
    "hear. \n"
    "Do not include any descriptions of actions, emotions, sounds, or voice "
    "changes. \n"
    "Do not use asterisks, brackets, parentheses, or any other symbols to "
    "indicate tone or actions. \n"
    "You should communicate in the same language strictly as the user unless "
    "they request otherwise.\n"
    "When you are uncertain (e.g., you can't see/hear clearly, don't understand, "
    "or the user makes a comment rather than asking a question), use appropriate "
    "questions to guide the user to continue the conversation.\n"
    "Keep replies concise and conversational, as if talking face-to-face."
)
INPUT_RATE = 16000
FRAME_SECONDS = 0.08
SAMPLES_PER_FRAME = 1920
MAX_REPLY_TOKENS = 1024
MAX_FRAMES = 1500
# code2wav attends over at most 72 frames in the reference model.
CODE2WAV_CONTEXT = 25
MAX_CODE2WAV_CHUNK = 47
# The checkpoint's generation_config.json talker defaults.
TALKER_TEMPERATURE = 0.9
TALKER_TOP_K = 50
TALKER_REPETITION_PENALTY = 1.05
CODEC_SPECIAL_IDS = (2048, 3072)


def talker_sample(logits: mx.array, history: list[int], eos: int) -> mx.array:
    """Sample a codec token as the reference talker does."""
    scores = logits.astype(mx.float32)
    if history:
        ids = mx.array(sorted(set(history)))
        picked = scores[:, ids]
        scores[:, ids] = mx.where(
            picked > 0,
            picked / TALKER_REPETITION_PENALTY,
            picked * TALKER_REPETITION_PENALTY,
        )
    blocked = np.zeros(scores.shape[-1], dtype=bool)
    blocked[CODEC_SPECIAL_IDS[0] : CODEC_SPECIAL_IDS[1]] = True
    blocked[eos] = False
    scores = mx.where(mx.array(blocked)[None], -mx.inf, scores) / TALKER_TEMPERATURE
    kth = mx.sort(scores, axis=-1)[:, -TALKER_TOP_K][:, None]
    return mx.random.categorical(mx.where(scores < kth, -mx.inf, scores))


class Qwen3OmniVoiceBackend(VoiceBackend):
    """A spoken conversation with Qwen3-Omni (thinker, talker and code2wav)."""

    @classmethod
    def supports(cls, model) -> bool:
        # Thinker-only conversions drop the talker and code2wav.
        return bool(getattr(model, "has_talker", False))

    @classmethod
    def voice_names(cls, config: dict) -> list[str]:
        return list((config.get("talker_config") or {}).get("speaker_id") or {})

    def __init__(self, model, processor):
        config = model.config
        talker_config = config.talker_config
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.speakers = dict(talker_config.speaker_id)
        self.voices = tuple(self.speakers)
        self.thinker = model.thinker.language_model
        self.embed = self.thinker.model.embed_tokens
        self.talker = model.talker
        # mlx-vlm captures the talker's thinker hidden state at this index.
        self.accept_layer = talker_config.accept_hidden_layer + 1
        self.audio_token = config.thinker_config.audio_token_id
        self.im_start = config.im_start_token_id
        self.im_end = config.im_end_token_id
        self.codec_eos = talker_config.codec_eos_token_id
        special = mx.array(
            [
                [
                    config.tts_bos_token_id,
                    config.tts_eos_token_id,
                    config.tts_pad_token_id,
                ]
            ]
        )
        projected = self.talker.text_projection(self.embed(special))
        self.tts_bos, self.tts_eos, self.tts_pad = (
            projected[:, 0:1],
            projected[:, 1:2],
            projected[:, 2:3],
        )
        self.talker_layers = talker_config.text_config.num_hidden_layers
        self.limit = config.thinker_config.text_config.max_position_embeddings
        self.cache = None
        self.offset = 0
        self.snapshot = 0
        self.user_parts: list[mx.array] = []
        self.pending: list[int] = []
        self.done = True

    def _positions(self, length: int) -> mx.array:
        # Text and audio take plain positions; the helper would restart at 0 per turn.
        positions = mx.arange(self.offset, self.offset + length)[None, None, :]
        return mx.broadcast_to(positions, (3, 1, length))

    def _turn_text(self, system: str) -> str:
        user = {"role": "user", "content": [{"type": "audio"}]}
        messages = [{"role": "system", "content": system}, user]
        if self.cache is None:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )
        im_end = self.tokenizer.convert_ids_to_tokens(self.im_end)
        return continuation_turn_text(
            self.tokenizer, messages, im_end, enable_thinking=True
        )

    def begin_turn(
        self, audio: np.ndarray, instructions: str, voice: str | None
    ) -> int:
        system = f"{SYSTEM_PROMPT}\n\n{instructions}" if instructions else SYSTEM_PROMPT
        clip = np.asarray(resample_audio(audio, SAMPLE_RATE, INPUT_RATE), np.float32)
        inputs = self.processor(
            text=[self._turn_text(system)], audio=[clip], sampling_rate=INPUT_RATE
        )
        if self.cache is None:
            self.cache = make_prompt_cache(self.thinker)
            self.offset = 0
        ids = self.pending + np.asarray(inputs["input_ids"]).reshape(-1).tolist()
        if self.offset + len(ids) + MAX_REPLY_TOKENS > self.limit:
            raise RealtimeVoiceError(
                "conversation_too_long", "the conversation reached the model's context"
            )
        tokens = mx.array([ids])
        embeds = self.model.thinker.get_input_embeddings(
            tokens,
            input_features=mx.array(inputs["input_features"]),
            feature_attention_mask=mx.array(inputs["feature_attention_mask"]),
        ).inputs_embeds
        out = self.thinker(
            tokens,
            inputs_embeds=embeds,
            cache=self.cache,
            position_ids=self._positions(len(ids)),
            output_hidden_state_idx=self.accept_layer,
        )
        self.offset += len(ids)
        self.snapshot = self.offset
        starts = np.flatnonzero(np.asarray(ids) == self.im_start)
        user_start, assistant_start = int(starts[-2]), int(starts[-1])
        audio_mask = mx.array(np.asarray(ids) == self.audio_token)[None]
        self.user_parts.append(
            self.model._get_talker_user_parts(
                user_start, assistant_start, audio_mask, out.hidden_states, embeds
            )
        )
        self.assistant_prefix = embeds[:, assistant_start:]
        self.speaker = self.speakers[voice or self.voices[0]]
        self.pending = []
        self.reply: list[int] = []
        self.unfed: int | None = None
        self.thinker_done = False
        self._accept(int(mx.argmax(out.logits[:, -1, :], axis=-1).item()))
        self.talker_cache = None
        self.frames: list[mx.array] = []
        self.history: list[int] = []
        self.eos_fed = False
        self.decoded = 0
        self.chunks = 0
        self.transcript = TranscriptStream(self.tokenizer)
        self.done = not self.reply
        return len(ids)

    def _accept(self, token: int) -> None:
        """Record a sampled thinker token; the final <|im_end|> waits for the next turn."""
        if token == self.im_end or len(self.reply) >= MAX_REPLY_TOKENS:
            self.thinker_done = True
            self.unfed = None
            self.pending = [self.im_end]
        else:
            self.reply.append(token)
            self.unfed = token

    def _thinker_step(self) -> None:
        out = self.thinker(
            mx.array([[self.unfed]]), cache=self.cache, position_ids=self._positions(1)
        )
        self.offset += 1
        self._accept(int(mx.argmax(out.logits[:, -1, :], axis=-1).item()))

    def _start_talker(self) -> None:
        first = self.embed(mx.array([[self.reply[0]]]))
        assistant = mx.concatenate([self.assistant_prefix, first], axis=1)
        prefix, _, _ = self.model._get_talker_assistant_parts(
            0, 4, self.speaker, assistant, self.tts_pad, self.tts_bos, self.tts_eos
        )
        self.talker_cache = [KVCache() for _ in range(self.talker_layers)]
        logits, hidden = self.talker(
            input_ids=None,
            inputs_embeds=mx.concatenate([*self.user_parts, prefix], axis=1),
            past_key_values=self.talker_cache,
            use_cache=True,
        )
        self.past_hidden = hidden[:, -1:]
        self.codec_token = talker_sample(logits[:, -1, :], self.history, self.codec_eos)

    def _talker_frame(self) -> None:
        token = int(self.codec_token.item())
        if token == self.codec_eos:
            self._finish()
            return
        self.history.append(token)
        step = len(self.frames)
        while len(self.reply) < step + 2 and not self.thinker_done:
            self._thinker_step()
        if step + 1 < len(self.reply):
            trailing = self.talker.text_projection(
                self.embed(mx.array([[self.reply[step + 1]]]))
            )
        elif not self.eos_fed:
            trailing, self.eos_fed = self.tts_eos, True
        else:
            trailing = self.tts_pad
        embeds, codes = self.talker.prepare_inputs_for_generation(
            input_ids=self.codec_token[:, None],
            past_hidden=self.past_hidden,
            trailing_text_hidden=trailing,
            tts_pad_embed=self.tts_pad,
            generation_step=0,
            temperature=0,
        )
        logits, hidden = self.talker(
            input_ids=None,
            inputs_embeds=embeds,
            past_key_values=self.talker_cache,
            use_cache=True,
        )
        self.past_hidden = hidden[:, -1:]
        self.frames.append(codes)
        self.codec_token = talker_sample(logits[:, -1, :], self.history, self.codec_eos)

    def _finish(self) -> None:
        """End the reply; an unfed reply token joins the pending prefix."""
        if not self.thinker_done:
            self.pending = ([self.unfed] if self.unfed is not None else []) + [
                self.im_end
            ]
            self.thinker_done = True
        self.done = True

    def step(self, seconds: float, cancel) -> VoiceStep:
        out = VoiceStep()
        deadline = time.perf_counter() + seconds
        if not self.done and self.talker_cache is None:
            self._start_talker()
        while not self.done and not cancel.is_set():
            self._talker_frame()
            out.slots += 1
            if len(self.frames) >= MAX_FRAMES:
                self._finish()
            if time.perf_counter() >= deadline:
                break
        out.text = self.transcript.delta(self.reply)
        out.audio = self._decode(final=self.done)
        out.done = self.done
        return out

    def _decode(self, final: bool) -> np.ndarray:
        new = len(self.frames) - self.decoded
        # Small first chunks start playback early; later chunks amortize the context.
        if new == 0 or (new < min(4 + 2 * self.chunks, 12) and not final):
            return np.zeros(0, np.float32)
        audio = []
        while self.decoded < len(self.frames):
            end = min(len(self.frames), self.decoded + MAX_CODE2WAV_CHUNK)
            start = max(0, self.decoded - CODE2WAV_CONTEXT)
            window = mx.concatenate(self.frames[start:end], axis=0).T[None]
            wav = self.model.code2wav(codes=window).astype(mx.float32)
            wav = np.asarray(wav).reshape(-1)
            audio.append(wav[(self.decoded - start) * SAMPLES_PER_FRAME :])
            self.decoded = end
        self.chunks += 1
        return np.concatenate(audio)

    def stop(self) -> None:
        if not self.done:
            self._finish()

    def truncate(self, played_seconds: float) -> None:
        played = int(played_seconds / FRAME_SECONDS + 1e-6)
        # Nothing heard drops the whole reply, even before its first frame.
        if self.cache is None or (played and played >= len(self.frames)):
            return
        # Frame n reads reply token n + 1, so the played frames read reply[: played + 1].
        for cache in self.cache:
            cache.trim(self.offset - self.snapshot)
        self.offset = self.snapshot
        self.reply = self.reply[: played + 1] if played else []
        self.pending = self.reply + [self.im_end]
        self.frames = self.frames[:played]
        self.done = self.thinker_done = True

    def close(self) -> None:
        self.cache = None
        self.talker_cache = None
        self.user_parts = []
        self.pending = []
