# SPDX-License-Identifier: Apache-2.0
"""Engine-side per-token logprob extraction (#1549).

Reduces a full-vocabulary logprob vector (computed for sampling) to a compact
``TokenLogprob`` so the large array can be freed immediately. Called only when a
request opts in via ``sampling_params.logprobs`` — the disabled path never
reaches here, keeping inference unaffected.
"""

import mlx.core as mx

from .request import TokenLogprob


def extract_token_logprob(
    logprobs_vec: mx.array, chosen_token_id: int, top_k: int
) -> TokenLogprob:
    """Reduce a full-vocab logprob vector to a compact ``TokenLogprob``.

    Args:
        logprobs_vec: 1-D log-probability vector over the vocabulary.
        chosen_token_id: The token actually sampled at this position.
        top_k: Number of top candidates to keep (already clamped to the
            effective server cap). ``0`` keeps only the chosen token's logprob.

    Returns:
        A ``TokenLogprob`` with the chosen logprob and the top-K candidates
        sorted by logprob descending.
    """
    chosen_lp = float(logprobs_vec[chosen_token_id].item())

    top_ids: list[int] = []
    top_lps: list[float] = []
    if top_k and top_k > 0:
        vocab = logprobs_vec.shape[-1]
        k = min(int(top_k), vocab)
        # Top-K (unordered) via argpartition, then sort that slice descending.
        part = mx.argpartition(-logprobs_vec, kth=k - 1)[:k]
        order = mx.argsort(-logprobs_vec[part])
        ordered = part[order]
        top_ids = ordered.tolist()
        top_lps = logprobs_vec[ordered].tolist()

    return TokenLogprob(
        token_id=int(chosen_token_id),
        logprob=chosen_lp,
        top_ids=top_ids,
        top_logprobs=top_lps,
    )
