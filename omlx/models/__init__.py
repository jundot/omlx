# SPDX-License-Identifier: Apache-2.0
"""
MLX Model wrappers for oMLX.

This module provides wrappers around mlx-lm and mlx-vlm
for integration with oMLX's model execution system.
"""

from omlx.models.llm import MLXLanguageModel
from omlx.models.embedding import MLXEmbeddingModel, EmbeddingOutput

__all__ = ["MLXLanguageModel", "MLXEmbeddingModel", "EmbeddingOutput"]
