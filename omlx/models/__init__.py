# SPDX-License-Identifier: Apache-2.0
"""
MLX Model wrappers for oMLX.

This module provides wrappers around mlx-lm and mlx-embeddings
for integration with oMLX's model execution system. Resolve wrappers on first
access so importing model metadata does not initialize the MLX runtime.
"""

_LAZY = {
    "MLXLanguageModel": "omlx.models.llm",
    "MLXEmbeddingModel": "omlx.models.embedding",
    "EmbeddingOutput": "omlx.models.embedding",
}


def __getattr__(name: str):
    import importlib

    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = ["MLXLanguageModel", "MLXEmbeddingModel", "EmbeddingOutput"]
