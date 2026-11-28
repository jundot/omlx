# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Neywa Labs Inc.
# mlx-vlm: adapted from mlx_vlm/models/ernie5_5_moe/ernie5_5_moe.py at commit 36494b42.

import mlx.nn as nn

from .config import ModelConfig
from .language import LanguageModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config)
