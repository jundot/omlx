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
from .asymmetric_kv import (
    install as install_asymmetric_kv,
    remove as remove_asymmetric_kv,
    get_stats as get_asymmetric_kv_stats,
)

__all__ = [
    # attention-guided eviction (#1273)
    "AttentionGuidedCachePolicy",
    "install_attention_guided_cache",
    "remove_attention_guided_cache",
    "score_blocks_from_attention",
    # asymmetric KV quantization (#1274)
    "install_asymmetric_kv",
    "remove_asymmetric_kv",
    "get_asymmetric_kv_stats",
]
