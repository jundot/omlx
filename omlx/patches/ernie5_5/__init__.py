# SPDX-License-Identifier: Apache-2.0

from ...custom_kernels.attention import fast as attention_fast
from ...custom_kernels.sparse_index import fast as index_fast


def sparse_prefill_available() -> bool:
    """Whether MLA-DSA prefill chunks run on the native sparse kernels."""
    return (
        attention_fast.has_symbol("sparse_mla_attention")
        and index_fast.has_symbol("dsa_indexer_scores")
        and index_fast.has_symbol("dsa_topk_indices")
    )
