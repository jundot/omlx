# SPDX-License-Identifier: Apache-2.0
"""
Realtime voice backend for LFM3-Audio and LFM3-Omni.

The slot loop follows mlx-vlm's ``stream_slots`` in interleaved mode: blocks
of six text tokens and twelve speech frames until ``<|text_end|>``, then
speech until an end-of-speech frame. Unlike ``stream_slots`` it keeps the
cache across turns, stops between executor calls, and decodes speech frames
incrementally.
"""

import time
from collections import deque

import mlx.core as mx
import numpy as np
from mlx_audio.utils import resample_audio
from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_vlm.models.lfm3_audio.depth import (
    embed_frames,
    normalize_eos,
    sample,
    sample_frame,
)

from .lfm3_speech import REPLY_PROMPT
from .realtime_voice import (
    SAMPLE_RATE,
    RealtimeVoiceError,
    TranscriptStream,
    VoiceBackend,
    VoiceStep,
    continuation_turn_text,
)

INPUT_RATE = 16000
FRAME_SECONDS = 0.08
MAX_REPLY_SLOTS = 2048
TEXT, AUDIO, INTERLEAVED = "text", "audio", "interleaved"
# stream_slots defaults that the checkpoints' audio_sampling does not override.
_SAMPLING = {
    "temperature": None,
    "top_k": None,
    "top_p": None,
    "min_p": None,
    "repetition_penalty": None,
    "audio_temperature": None,
    "audio_top_k": None,
    "audio_repetition_penalty": None,
    "audio_repetition_window": 64,
    "text_repetition_window": 32,
}


class CodecStream:
    """Incremental LiquidCodec decode: speech frames in, final 24 kHz samples out.

    The decoder is causal (conv layers and a sliding window), so it runs with a
    cache. The ISTFT overlap-add keeps the samples that later frames still add
    to, and trims ``(n_fft - hop) / 2`` samples at both ends like ``decode``.
    """

    def __init__(self, codec):
        config = codec.config
        self.codec = codec
        self.window_size = config.decoder_window
        self.stack = config.stft_frame_stack
        self.hop = config.stft_hop_length
        self.n_fft = config.stft_n_fft
        self.trim = (self.n_fft - self.hop) // 2
        self.window = np.asarray(
            codec.decoder.istft.window.astype(mx.float32), dtype=np.float32
        )
        self.reset()

    def reset(self) -> None:
        self.cache = [
            (
                ArraysCache(size=1)
                if layer.layer_type == "conv"
                else RotatingKVCache(max_size=self.window_size)
            )
            for layer in self.codec.decoder.lfm.layers
        ]
        self.audio = np.zeros(0, np.float32)
        self.envelope = np.zeros(0, np.float32)
        self.base = 0  # raw sample index of audio[0]
        self.stft_frames = 0
        self.emitted = self.trim  # next raw sample to emit

    def push(self, frames: list[mx.array]) -> np.ndarray:
        if not frames:
            return np.zeros(0, np.float32)
        codec, decoder = self.codec, self.codec.decoder
        codes = mx.stack(frames, axis=-1)[None].astype(mx.int32)
        x = codec.from_quantizer(codec.quantizer(codes))
        h = decoder.lfm(
            input_embeddings=mx.repeat(x, self.stack, axis=1), cache=self.cache
        )
        log_mag, angle = mx.split(decoder.lin(h).astype(mx.float32), 2, axis=-1)
        mag = mx.exp(log_mag)
        spec = mag * mx.cos(angle) + 1j * (mag * mx.sin(angle))
        stft = np.asarray(mx.fft.irfft(spec, n=self.n_fft, axis=-1)[0]) * self.window
        self._overlap_add(stft)
        return self._emit(self.stft_frames * self.hop)

    def flush(self) -> np.ndarray:
        if not self.stft_frames:
            return np.zeros(0, np.float32)
        end = (self.stft_frames - 1) * self.hop + self.n_fft - self.trim
        out = self._emit(end)
        self.reset()
        return out

    def _overlap_add(self, frames: np.ndarray) -> None:
        start = self.stft_frames * self.hop - self.base
        need = start + (len(frames) - 1) * self.hop + self.n_fft
        if len(self.audio) < need:
            grow = need - len(self.audio)
            self.audio = np.concatenate([self.audio, np.zeros(grow, np.float32)])
            self.envelope = np.concatenate([self.envelope, np.zeros(grow, np.float32)])
        square = self.window * self.window
        for i, frame in enumerate(frames):
            at = start + i * self.hop
            self.audio[at : at + self.n_fft] += frame
            self.envelope[at : at + self.n_fft] += square
        self.stft_frames += len(frames)

    def _emit(self, until: int) -> np.ndarray:
        if until <= self.emitted:
            return np.zeros(0, np.float32)
        a, b = self.emitted - self.base, until - self.base
        out = self.audio[a:b] / self.envelope[a:b]
        self.audio, self.envelope = self.audio[b:], self.envelope[b:]
        self.base = self.emitted = until
        return out.astype(np.float32)


class LFM3VoiceBackend(VoiceBackend):
    """A spoken conversation with LFM3-Audio or LFM3-Omni."""

    # Voice chat speaks with the checkpoint's one built-in voice.
    voices = ()

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.config = config = model.config
        self.backbone = model.language_model.model
        self.sampling = {**_SAMPLING, **model.audio_sampling}
        eos = config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.suppress = model.placeholder_token_ids()
        self.limit = config.lm_config.max_position_embeddings
        self.eos_frame = mx.full(
            (config.num_codebooks,), config.audio_eos_code, mx.int32
        )
        self.codec = CodecStream(model.codec())
        self.cache = None
        self.snapshot = None
        self.pending: list[tuple[int, mx.array | None]] = []
        self.reply: list[tuple[int, mx.array | None]] = []
        self.finished = True

    def _offset(self) -> int:
        return next(c.offset for c in self.cache if isinstance(c, KVCache))

    def _turn_text(self, system: str) -> str:
        """The chat-template text of one user turn up to the assistant prompt."""
        user = {"role": "user", "content": [{"type": "audio"}]}
        messages = [{"role": "system", "content": system}, user]
        if self.cache is None:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        im_end = self.tokenizer.convert_ids_to_tokens(self.config.eos_token_id[0])
        return continuation_turn_text(self.tokenizer, messages, im_end)

    def _embed(self, slots: list[tuple[int, mx.array | None]]) -> mx.array:
        rows = [
            (
                embed_frames(self.model.codebook_embedding, codes)
                if codes is not None
                else self.backbone.embed_tokens(mx.array([token]))[0]
            )
            for token, codes in slots
        ]
        return mx.stack(rows)

    def _turn_inputs(self, audio: np.ndarray, system: str) -> tuple[list[int], dict]:
        """Token ids of one user turn, opened as an interleaved reply, and its features."""
        clip = np.asarray(resample_audio(audio, SAMPLE_RATE, INPUT_RATE), np.float32)
        inputs = self.processor(text=self._turn_text(system), audio=[clip])
        ids = np.asarray(inputs["input_ids"]).reshape(-1).tolist()
        ids.append(self.config.mixed_start_token_id)
        features = {
            "input_features": inputs["input_features"],
            "feature_attention_mask": inputs["feature_attention_mask"],
        }
        return ids, features

    def begin_turn(
        self, audio: np.ndarray, instructions: str, voice: str | None
    ) -> int:
        system = f"{REPLY_PROMPT}\n\n{instructions}" if instructions else REPLY_PROMPT
        ids, features = self._turn_inputs(audio, system)
        if self.cache is None:
            self.cache = self.model.make_cache()
        prompt_tokens = len(self.pending) + len(ids)
        if self._offset() + prompt_tokens + MAX_REPLY_SLOTS > self.limit:
            raise RealtimeVoiceError(
                "conversation_too_long", "the conversation reached the model's context"
            )
        turn = self.model.get_input_embeddings(
            mx.array([ids]), **features
        ).inputs_embeds[0]
        if self.pending:
            turn = mx.concatenate([self._embed(self.pending), turn])
        self.hidden = self.backbone(None, self.cache, turn[None])[:, -1]
        self.pending = []
        self.snapshot = self._snapshot()
        self.mode = INTERLEAVED
        self.slots_left = self.config.interleaved_n_text
        self.speaking = self.text_done = False
        self.text_history = deque(maxlen=self.sampling["text_repetition_window"])
        self.audio_history = deque(maxlen=self.sampling["audio_repetition_window"])
        self.reply = []
        self.frames: list[mx.array] = []
        self.text_ids: list[int] = []
        self.transcript = TranscriptStream(self.tokenizer)
        self.finished = False
        self.codec.reset()
        return prompt_tokens

    def _slot(self) -> None:
        """Sample one slot and feed it back, as one ``stream_slots`` iteration."""
        config, s = self.config, self.sampling
        interleaving, finished = self.mode == INTERLEAVED, False
        if self.mode == AUDIO or (interleaving and self.speaking):
            history = (
                mx.stack(list(self.audio_history))[None] if self.audio_history else None
            )
            codes = sample_frame(
                self.model.depth_decoder,
                self.hidden,
                config.maskgit_schedule,
                s["audio_temperature"],
                s["audio_top_k"],
                s["audio_repetition_penalty"],
                history,
            )[0]
            codes = normalize_eos(codes, config.audio_eos_code)
            if codes[0].item() == config.audio_eos_code:
                self.mode = TEXT
            else:
                self.audio_history.append(codes)
                self.frames.append(codes)
            slot = (config.audio_out_token_id, codes)
        else:
            history = mx.array([list(self.text_history)]) if self.text_history else None
            token = sample(
                self.backbone.embed_tokens.as_linear(self.hidden),
                s["temperature"],
                s["top_k"],
                s["repetition_penalty"],
                history,
                self.suppress,
                s["top_p"],
                s["min_p"],
            ).item()
            self.text_history.append(token)
            if token == config.text_end_token_id:
                if interleaving:
                    self.text_done = self.speaking = True
                else:
                    finished = True
            elif token in self.eos_ids:
                finished = True
            elif interleaving:
                self.text_ids.append(token)
            elif token == config.audio_start_token_id:
                self.mode = AUDIO
            elif token == config.mixed_start_token_id:
                self.mode = INTERLEAVED
                self.slots_left = config.interleaved_n_text
                self.speaking = self.text_done = False
            else:
                self.text_ids.append(token)
            slot = (token, None)
        self.reply.append(slot)
        if interleaving and not self.text_done:
            self.slots_left -= 1
            if self.slots_left == 0:
                self.speaking = not self.speaking
                self.slots_left = (
                    config.interleaved_n_audio
                    if self.speaking
                    else config.interleaved_n_text
                )
        if finished:
            # The final token is fed with the next turn, as stream_slots never
            # feeds it. A reply that stops on <|text_end|> still needs <|im_end|>.
            im_end = config.eos_token_id[0]
            self.pending = [slot] if slot[0] == im_end else [slot, (im_end, None)]
            self.finished = True
        else:
            embed = self._embed([slot])
            self.hidden = self.backbone(None, self.cache, embed[None])[:, -1]

    def step(self, seconds: float, cancel) -> VoiceStep:
        out = VoiceStep()
        if self.finished:
            out.done = True
            return out
        deadline = time.perf_counter() + seconds
        while not self.finished and not cancel.is_set():
            self._slot()
            out.slots += 1
            if len(self.reply) >= MAX_REPLY_SLOTS:
                self.stop()
            if time.perf_counter() >= deadline:
                break
        out.text = self.transcript.delta(self.text_ids)
        audio = self.codec.push(self.frames)
        self.frames = []
        if self.finished:
            audio = np.concatenate([audio, self.codec.flush()])
            out.done = True
        out.audio = audio
        return out

    def _closing(self, slots: list[tuple[int, mx.array | None]]) -> list:
        """Close a cut reply: end open speech with an EOS frame, then <|im_end|>."""
        tail = []
        if any(codes is not None for _, codes in slots) and not (
            slots[-1][1] is not None
            and slots[-1][1][0].item() == self.config.audio_eos_code
        ):
            tail.append((self.config.audio_out_token_id, self.eos_frame))
        return tail + [(self.config.eos_token_id[0], None)]

    def stop(self) -> None:
        if self.finished:
            return
        self.pending = self._closing(self.reply)
        self.finished = True

    def truncate(self, played_seconds: float) -> None:
        if self.snapshot is None:
            return
        played = int(played_seconds / FRAME_SECONDS + 1e-6)
        heard, kept = 0, []
        for token, codes in self.reply:
            if heard >= played:
                break
            kept.append((token, codes))
            if codes is not None and codes[0].item() != self.config.audio_eos_code:
                heard += 1
        spoken = sum(
            1
            for _, codes in self.reply
            if codes is not None and codes[0].item() != self.config.audio_eos_code
        )
        # Nothing heard drops the whole reply, even before its first frame.
        if played and played >= spoken:
            return
        self._restore()
        self.pending = kept + self._closing(kept) if kept else self._closing([])
        self.reply = kept
        self.finished = True

    def _snapshot(self) -> list:
        snapshot = []
        for cache in self.cache:
            if isinstance(cache, KVCache):
                snapshot.append(cache.offset)
            elif isinstance(cache, RotatingKVCache):
                keys, values = cache.keys, cache.values
                if keys is not None:
                    # Single-token updates write the rotating buffer in place.
                    keys, values = mx.array(keys), mx.array(values)
                    mx.eval(keys, values)
                snapshot.append(((keys, values), cache.meta_state))
            else:
                snapshot.append(list(cache.cache))
        return snapshot

    def _restore(self) -> None:
        for cache, saved in zip(self.cache, self.snapshot, strict=True):
            if isinstance(cache, KVCache):
                cache.trim(cache.offset - saved)
            elif isinstance(cache, RotatingKVCache):
                (keys, values), meta = saved
                if keys is not None:
                    keys, values = mx.array(keys), mx.array(values)
                cache.keys, cache.values = keys, values
                cache.meta_state = meta
            else:
                cache.cache = list(saved)

    def close(self) -> None:
        self.cache = None
        self.snapshot = None
        self.pending = []
        self.reply = []
        self.codec.reset()
