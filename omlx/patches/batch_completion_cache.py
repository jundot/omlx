# SPDX-License-Identifier: Apache-2.0
"""Release old cache banks by layer when several requests finish together."""

from functools import wraps


def defer_terminal_cache_extraction(next_fn):
    """Scope deferral to response emission, never MTP activation or rollback."""

    @wraps(next_fn)
    def wrapped(batch, *args, **kwargs):
        if not getattr(type(batch), "_omlx_completion_cache_patched", False):
            return next_fn(batch, *args, **kwargs)
        if getattr(batch, "_omlx_terminal_caches", None) is not None:
            return next_fn(batch, *args, **kwargs)
        batch._omlx_terminal_caches = {}
        try:
            return next_fn(batch, *args, **kwargs)
        finally:
            del batch._omlx_terminal_caches

    return wrapped


def apply_batch_completion_cache_patch():
    """Install before the MTP filter wrapper, whose epilogue needs live caches."""
    from mlx_lm.generate import GenerationBatch

    if getattr(GenerationBatch, "_omlx_completion_cache_patched", False):
        return
    original_extract = GenerationBatch.extract_cache
    original_filter = GenerationBatch.filter

    @wraps(original_extract)
    def extract_cache(batch, idx):
        pending = getattr(batch, "_omlx_terminal_caches", None)
        if pending is None:
            return original_extract(batch, idx)
        # Response objects retain this list; filter fills it before next returns.
        return pending.setdefault(idx, [])

    @wraps(original_filter)
    def filter_cache(batch, keep):
        pending = getattr(batch, "_omlx_terminal_caches", None)
        if not pending:
            return original_filter(batch, keep)
        if set(pending) != set(range(len(batch.uids))) - set(keep):
            raise RuntimeError("Terminal cache rows do not match removed requests")

        caches = batch.prompt_cache
        survivors = []
        for layer in range(len(caches)):
            cache = caches[layer]
            for idx, row in pending.items():
                row.append(cache.extract(idx))
            if keep:
                cache.filter(keep)
                survivors.append(cache)
            caches[layer] = None
            del cache

        # The native method still owns UID/token/sampler/matcher filtering.
        # Its cache loop must not filter our already-filtered layers twice.
        batch.prompt_cache = []
        try:
            return original_filter(batch, keep)
        finally:
            batch.prompt_cache = survivors

    GenerationBatch.extract_cache = extract_cache
    GenerationBatch.filter = filter_cache
    GenerationBatch.next = defer_terminal_cache_extraction(GenerationBatch.next)
    GenerationBatch._omlx_completion_cache_patched = True
