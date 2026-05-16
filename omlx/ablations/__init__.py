# SPDX-License-Identifier: Apache-2.0
"""
OMLX+ Ablation Suite — Hardware-targeted optimizations for Apple Silicon LLM inference.

Reference: https://github.com/jundot/omlx
"""

from .attention_cache import (
    AttentionGuidedCachePolicy,
    install_attention_guided_cache,
    remove_attention_guided_cache,
    score_blocks_from_attention,
)

__all__ = [
    "AttentionGuidedCachePolicy",
    "install_attention_guided_cache",
    "remove_attention_guided_cache",
    "score_blocks_from_attention",
]
