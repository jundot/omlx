# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3-Audio checkpoint reference implementation.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

import time
from collections import deque

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from ..lfm3.language import LanguageModel as Lfm3LanguageModel
from .audio import AudioEncoder, AudioProjector
from .audio import sanitize as sanitize_audio
from .codec import load_codec
from .depth import DepthDecoder, embed_frames, normalize_eos, sample, sample_frame

TEXT, AUDIO, INTERLEAVED = "text", "audio", "interleaved"

# Training-export key prefixes -> module paths.
_LEGACY_PREFIXES = (
    ("lfm.model.", "language_model.model."),
    ("encoder.downsample.", "audio_tower.downsample."),
    ("encoder.lfm.lfm.", "audio_tower.lfm."),
    ("mlp.model.0.", "audio_projector.layer_norm."),
    ("mlp.model.1.", "audio_projector.linear_1."),
    ("mlp.model.3.", "audio_projector.linear_2."),
    ("depthformer.model.", "depth_decoder.model."),
    ("depth_linear.", "depth_decoder.depth_linear."),
    ("mask_embedding", "depth_decoder.mask_embedding"),
    ("depth_embeddings.", "depth_decoder.depth_embeddings."),
)


class LanguageModel(Lfm3LanguageModel):
    """The LFM3 backbone, whose text logits never pick a speech placeholder."""

    def __init__(self, config, placeholder_ids):
        super().__init__(config)
        self._placeholder_ids = list(placeholder_ids)

    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        output.logits[..., self._placeholder_ids] = -mx.inf
        return output


class Model(nn.Module):
    joint_audio_text_generation = True

    # the speech sampling the checkpoint's model card recommends
    audio_sampling = {
        "audio_temperature": 1.0,
        "audio_top_k": 128,
        "audio_repetition_penalty": 1.1,
        "audio_repetition_window": 4096,
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        hidden = config.lm_config.hidden_size
        self.language_model = LanguageModel(
            config.lm_config, self.placeholder_token_ids()
        )
        self.audio_tower = AudioEncoder(config.audio_encoder_config)
        self.audio_projector = AudioProjector(
            config.audio_encoder_config.hidden_size, hidden
        )
        self.codebook_embedding = nn.Embedding(
            config.num_codebooks * config.codebook_size, hidden
        )
        self.depth_decoder = DepthDecoder(
            config.depth_config, hidden, config.num_codebooks, config.codebook_size
        )

    def placeholder_token_ids(self):
        """Prompt-only ids the text head never samples."""
        return [self.config.audio_token_id, self.config.audio_out_token_id]

    @staticmethod
    def _merge(embeddings, input_ids, features, token_id):
        indices = np.flatnonzero(np.asarray(input_ids).reshape(-1) == token_id)
        count = 0 if features is None else features.shape[0]
        if len(indices) != count:
            raise ValueError(
                f"Placeholders {token_id} ({len(indices)}) and features ({count}) do not match"
            )
        if not count:
            return embeddings
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        flat[mx.array(indices)] = features.astype(embeddings.dtype)
        return flat.reshape(embeddings.shape)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        embeddings = self.language_model.model.embed_tokens(input_ids)
        return InputEmbeddingsFeatures(
            inputs_embeds=self.merge_speech(embeddings, input_ids, **kwargs)
        )

    def merge_speech(
        self,
        embeddings,
        input_ids,
        input_features=None,
        feature_attention_mask=None,
        audio_out_codes=None,
        **kwargs,
    ):
        """Scatter the encoded speech into ``<|audio|>`` and prior speech frames
        into ``<|audio_out|>`` positions of ``embeddings``."""
        config = self.config
        audio = frames = None
        # a prompt-cache hit drops the cached placeholders but keeps the features
        if input_features is not None and mx.any(input_ids == config.audio_token_id):
            input_features = mx.array(input_features)
            lengths = (
                mx.full((input_features.shape[0],), input_features.shape[-1])
                if feature_attention_mask is None
                else mx.array(feature_attention_mask).sum(-1)
            )
            audio = self.audio_projector(self.audio_tower(input_features, lengths))
        if audio_out_codes is not None:
            codes = mx.array(audio_out_codes).astype(mx.int32)
            # the codebooks share one table, so a bad code would read a neighbour's row
            if codes.size and (codes.min() < 0 or codes.max() >= config.codebook_size):
                raise ValueError(
                    f"audio_out_codes must be in [0, {config.codebook_size})"
                )
            frames = embed_frames(self.codebook_embedding, codes)
        embeddings = self._merge(embeddings, input_ids, audio, config.audio_token_id)
        return self._merge(embeddings, input_ids, frames, config.audio_out_token_id)

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids, cache=cache, inputs_embeds=features.inputs_embeds
        )

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        result = {}
        for name, value in weights.items():
            for old, new in _LEGACY_PREFIXES:
                if name.startswith(old):
                    name = new + name[len(old) :]
                    break
            if (
                name.startswith("language_model.")
                and name.endswith("conv.conv.weight")
                and value.shape[1] == 1
            ):
                value = value.transpose(0, 2, 1)
            result[name] = value
        return DepthDecoder.sanitize(sanitize_audio(result))

    @property
    def quant_predicate(self):
        # The speech modules are small and precision sensitive: at 4 bits the
        # depth decoder logits move by 8% and the speech features by 28%; the
        # backbone takes the bits.
        def predicate(path, module):
            return not path.startswith(
                (
                    "audio_tower",
                    "audio_projector",
                    "depth_decoder",
                    "codebook_embedding",
                )
            )

        return predicate

    def codec(self):
        """The Liquid codec, loaded on first use and kept out of the parameters."""
        codec = getattr(self, "_codec", None)
        if codec is None:
            dtype = self.language_model.model.embedding_norm.weight.dtype
            # load_model records the checkpoint directory as model.model_path
            codec = load_codec(self.model_path, dtype=dtype)
            object.__setattr__(self, "_codec", codec)
        return codec

    def stream_slots(
        self,
        input_ids,
        max_tokens=2048,
        generation_mode=None,
        temperature=None,
        top_k=None,
        top_p=None,
        min_p=None,
        repetition_penalty=None,
        audio_temperature=None,
        audio_top_k=None,
        audio_repetition_penalty=None,
        audio_repetition_window=64,
        text_repetition_window=32,
        suppress_mixed=False,
        suppress_audio_out=False,
        frame_order=None,
        **kwargs,
    ):
        """Yield one ``(token, codes)`` per generated slot, as the reference ``generate``.

        A text slot yields its token and ``None``; a speech slot yields
        ``audio_out_token_id`` and its 12 codes, an end-of-speech frame as all
        ``audio_eos_code``. ``generation_mode`` None starts from the prompt's last
        token; "audio" or "interleaved" appends the marker the mode needs.
        ``frame_order`` [frames, 12] fixes the
        MaskGIT reveal order, which is otherwise random even when greedy. Other
        keyword arguments (``input_features``, ``audio_out_codes``, ...) go to
        ``get_input_embeddings``. Batch size 1.
        """
        config = self.config
        ids = np.asarray(input_ids)
        if ids.ndim == 2 and ids.shape[0] != 1:
            raise ValueError("LFM3-Audio generates one prompt at a time")
        ids = ids.reshape(-1).tolist()
        eos_code, frame_token = config.audio_eos_code, config.audio_out_token_id
        markers = {
            AUDIO: (config.audio_start_token_id, frame_token),
            INTERLEAVED: (config.mixed_start_token_id,),
        }
        mode = generation_mode
        if mode is None:
            mode = (
                AUDIO
                if ids[-1] in markers[AUDIO]
                else INTERLEAVED if ids[-1] == config.mixed_start_token_id else TEXT
            )
            # a replayed end-of-speech frame resumes the text
            codes = kwargs.get("audio_out_codes")
            if ids[-1] == frame_token and codes is not None:
                if (np.asarray(codes)[-1] == eos_code).any():
                    mode = TEXT
        elif mode not in (TEXT, AUDIO, INTERLEAVED):
            raise ValueError(f"Unknown generation_mode {mode!r}")
        elif mode in markers and ids[-1] not in markers[mode]:
            ids.append(markers[mode][0])
        eos_ids = config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids])
        suppress = self.placeholder_token_ids()
        # an explicit text mode keeps the reply in text, as the reference does
        if suppress_mixed or generation_mode == TEXT:
            suppress.append(config.mixed_start_token_id)
        if suppress_audio_out or generation_mode == TEXT:
            suppress.append(config.audio_start_token_id)

        backbone = self.language_model.model
        input_ids = mx.array([ids])
        cache = self.make_cache()
        embeds = self.get_input_embeddings(input_ids, **kwargs).inputs_embeds
        hidden = backbone(None, cache, embeds)[:, -1]
        text_history = deque(maxlen=text_repetition_window)
        audio_history = deque(maxlen=audio_repetition_window)
        slots_left, frame = config.interleaved_n_text, 0
        speaking = text_done = False
        for step in range(max_tokens):
            interleaving, finished = mode == INTERLEAVED, False
            if mode == AUDIO or (interleaving and speaking):
                order = None
                if frame_order is not None:
                    order = mx.array(frame_order[frame]).reshape(1, -1)
                frame += 1
                history = mx.stack(list(audio_history))[None] if audio_history else None
                codes = sample_frame(
                    self.depth_decoder,
                    hidden,
                    config.maskgit_schedule,
                    audio_temperature,
                    audio_top_k,
                    audio_repetition_penalty,
                    history,
                    order,
                )[0]
                codes = normalize_eos(codes, eos_code)
                if codes[0].item() == eos_code:
                    mode = TEXT
                else:
                    audio_history.append(codes)
                embed = embed_frames(self.codebook_embedding, codes)
                yield frame_token, codes
            else:
                history = mx.array([list(text_history)]) if text_history else None
                token = sample(
                    backbone.embed_tokens.as_linear(hidden),
                    temperature,
                    top_k,
                    repetition_penalty,
                    history,
                    suppress,
                    top_p,
                    min_p,
                ).item()
                text_history.append(token)
                if token == config.text_end_token_id:
                    # ends the text of an interleaved block, else the turn
                    if interleaving:
                        text_done = speaking = True
                    else:
                        finished = True
                elif token in eos_ids:
                    finished = True
                elif interleaving:
                    pass  # markers are plain text inside a block
                elif token == config.audio_start_token_id:
                    mode = AUDIO
                elif token == config.mixed_start_token_id:
                    mode = INTERLEAVED
                    slots_left = config.interleaved_n_text
                    speaking = text_done = False
                embed = backbone.embed_tokens(mx.array([token]))
                yield token, None
            if interleaving and not text_done:
                slots_left -= 1
                if slots_left == 0:
                    speaking = not speaking
                    slots_left = (
                        config.interleaved_n_audio
                        if speaking
                        else config.interleaved_n_text
                    )
            if finished or step == max_tokens - 1:
                return
            hidden = backbone(None, cache, embed.reshape(1, 1, -1))[:, -1]

    def generate_audio(
        self,
        input_ids,
        tokenizer,
        generation_mode=None,
        verbose=False,
        **kwargs,
    ):
        """Text and 24 kHz speech for one prompt. Without ``generation_mode`` the
        model starts in text and opens speech or speech chat itself. Speech
        sampling defaults to ``audio_sampling``; keyword arguments override it."""
        from ...generate.audio import AudioGenerationResult

        config = self.config
        codec = self.codec()  # fail on a missing codec before sampling
        tokens, frames = [], []
        start = time.perf_counter()
        for token, codes in self.stream_slots(
            input_ids,
            generation_mode=generation_mode,
            **{**self.audio_sampling, **kwargs},
        ):
            if not tokens:
                prompt_time = time.perf_counter() - start
                start = time.perf_counter()
            tokens.append(token)
            if codes is not None and codes[0].item() != config.audio_eos_code:
                frames.append(codes)
        generation_time = time.perf_counter() - start
        codes = (
            mx.stack(frames, axis=-1)
            if frames
            else mx.zeros((config.num_codebooks, 0), mx.int32)
        )
        text = tokenizer.decode(
            [t for t in tokens if t != config.audio_out_token_id],
            skip_special_tokens=True,
        )
        prompt_tokens = np.asarray(input_ids).size
        stop_ids = np.atleast_1d(config.eos_token_id).tolist()
        stop_ids.append(config.text_end_token_id)
        result = AudioGenerationResult(
            text=text,
            token_ids=tokens,
            prompt_tokens=prompt_tokens,
            generation_tokens=len(tokens),
            total_tokens=prompt_tokens + len(tokens),
            prompt_tps=prompt_tokens / prompt_time if tokens else 0.0,
            generation_tps=len(tokens) / generation_time if tokens else 0.0,
            finish_reason="stop" if tokens and tokens[-1] in stop_ids else "length",
            audio=codec.decode(codes),
            sample_rate=codec.config.sampling_rate,
            audio_tokens=codes,
            peak_memory=mx.get_peak_memory() / 1e9,
        )
        if verbose:
            print("=" * 10)
            print(text)
            print("=" * 10)
            print(
                f"Prompt: {result.prompt_tokens} tokens, {result.prompt_tps:.3f} tokens-per-sec"
            )
            print(
                f"Generation: {result.generation_tokens} slots ({len(frames)} speech frames), "
                f"{result.generation_tps:.3f} slots-per-sec"
            )
            print(f"Peak memory: {result.peak_memory:.3f} GB")
        return result
