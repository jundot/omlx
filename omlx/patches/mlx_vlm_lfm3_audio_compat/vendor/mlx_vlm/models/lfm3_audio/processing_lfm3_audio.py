# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3-Audio checkpoint reference implementation.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer, log-mel features and speech placeholder expansion for LFM3-Audio."""

import json
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerFast
from transformers.feature_extraction_utils import BatchFeature

from ..base import install_auto_processor_patch, load_chat_template, to_mlx
from .audio import SAMPLE_RATE, extract_features, num_encoder_tokens

_CONFIG_FILES = ("processor_config.json", "preprocessor_config.json")


class Lfm3AudioFeatureExtractor:
    sampling_rate = SAMPLE_RATE


class Lfm3AudioProcessor:
    supports_multiple_audio = True
    audio_token = "<|audio|>"
    audio_out_token = "<|audio_out|>"
    # the end-of-speech code, one past the codec's 8 * 7 * 6 * 6 FSQ codes
    audio_eos_code = 2016

    def __init__(self, tokenizer, configs=None):
        self.tokenizer = tokenizer
        self.feature_extractor = Lfm3AudioFeatureExtractor()
        self.configs = configs or {}
        # plain text generation stops where the model would start to speak
        self.additional_eos_token_ids = tokenizer.convert_tokens_to_ids(
            ["<|audio_start|>", "<|mixed_start|>"]
        )

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        tokenizer = load_chat_template(
            PreTrainedTokenizerFast.from_pretrained(model_path, **kwargs), model_path
        )
        configs = {
            name: json.loads(path.read_text())
            for name in _CONFIG_FILES
            if (path := Path(model_path, name)).exists()
        }
        return cls(tokenizer, configs)

    @staticmethod
    def _expand(texts, token, counts):
        if sum(text.count(token) for text in texts) != len(counts):
            raise ValueError(f"Expected {len(counts)} {token} placeholders")
        counts = iter(counts)
        expanded = []
        for text in texts:
            parts = text.split(token)
            expanded.append(
                parts[0] + "".join(token * next(counts) + part for part in parts[1:])
            )
        return expanded

    def __call__(
        self,
        text=None,
        images=None,
        audio=None,
        audio_out_codes=None,
        padding=True,
        padding_side="left",
        return_tensors="mlx",
        add_special_tokens=False,
    ):
        """``audio``: 16 kHz mono clips, one per ``<|audio|>``. ``audio_out_codes``:
        one [frames, 12] code array per ``<|audio_out|>`` (prior speech, as
        ``generate_audio`` returns it transposed); each gets its end-of-speech frame.
        Speech history as waveforms needs the codec encoder, which is not ported."""
        if images is not None:
            raise ValueError("LFM3-Audio does not read images")
        texts = [text] if isinstance(text, str) else list(text)
        texts, data = self.expand_speech(texts, audio, audio_out_codes)
        data.update(
            self.tokenizer(
                texts,
                add_special_tokens=add_special_tokens,
                padding=padding,
                padding_side=padding_side,
                return_token_type_ids=False,
            )
        )
        if return_tensors == "mlx":
            return BatchFeature(data=to_mlx(data))
        return BatchFeature(data=data, tensor_type=return_tensors)

    @classmethod
    def expand_speech(cls, texts, audio=None, audio_out_codes=None):
        """Expand the speech placeholders of ``texts``: (texts, numpy model inputs)."""
        data = {}
        clips = [] if audio is None else audio
        clips = [clips] if isinstance(clips, np.ndarray) and clips.ndim == 1 else clips
        counts = []
        if len(clips):
            features, mask = extract_features(clips)
            data.update(input_features=features, feature_attention_mask=mask)
            counts = num_encoder_tokens(mask.sum(-1)).tolist()
        texts = cls._expand(texts, cls.audio_token, counts)
        codes = []
        for c in [] if audio_out_codes is None else audio_out_codes:
            c = np.asarray(c, np.int32)
            if c.ndim != 2 or c.shape[1] != 12:
                raise ValueError(
                    f"audio_out_codes clips must be [frames, 12], got {c.shape}"
                )
            codes.append(np.concatenate([c, np.full((1, 12), cls.audio_eos_code)]))
        texts = cls._expand(texts, cls.audio_out_token, [len(c) for c in codes])
        if codes:
            data["audio_out_codes"] = np.concatenate(codes).astype(np.int32)
        return texts, data

    def save_pretrained(self, save_directory, **kwargs):
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        for name, config in self.configs.items():
            Path(save_directory, name).write_text(json.dumps(config, indent=2) + "\n")


install_auto_processor_patch("lfm3_audio", Lfm3AudioProcessor)
