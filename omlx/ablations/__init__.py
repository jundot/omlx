# SPDX-License-Identifier: Apache-2.0
"""OMLX+ Ablation Suite — Hardware-targeted optimizations for Apple Silicon LLM inference."""

from .semantic_prefix import (
    SemanticPrefixIndex,
    install_semantic_prefix,
    remove_semantic_prefix,
    compute_semantic_hash,
    hamming_distance,
    are_semantically_similar,
)

__all__ = [
    "SemanticPrefixIndex",
    "install_semantic_prefix",
    "remove_semantic_prefix",
    "compute_semantic_hash",
    "hamming_distance",
    "are_semantically_similar",
]
