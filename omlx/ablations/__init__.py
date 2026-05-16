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
from .semantic_prefix import (
    SemanticPrefixIndex,
    install_semantic_prefix,
    remove_semantic_prefix,
    compute_semantic_hash,
    hamming_distance,
    are_semantically_similar,
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
    # semantic prefix matching (#1275)
    "SemanticPrefixIndex",
    "install_semantic_prefix",
    "remove_semantic_prefix",
    "compute_semantic_hash",
    "hamming_distance",
    "are_semantically_similar",
]
