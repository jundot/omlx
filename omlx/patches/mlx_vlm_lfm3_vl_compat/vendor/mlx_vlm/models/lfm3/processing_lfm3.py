# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

from transformers import PreTrainedTokenizerFast

from ..base import install_auto_processor_patch, load_chat_template


class Lfm3Tokenizer:
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path, **kwargs)
        return load_chat_template(tokenizer, model_path)


install_auto_processor_patch("lfm3", Lfm3Tokenizer)
