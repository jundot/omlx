# SPDX-License-Identifier: Apache-2.0
"""End-to-end regression guards for V4 multi-session prefix cache.

The bug this commit chain fixes: when a DeepSeek V4 model handled a
prompt in session 1 and the same prompt arrived in session 2 (no
shared in-memory state, no continuation — just identical text), the
second session's first token came out garbage and the rest collapsed
into repetition. Disabling prefix cache made session 2 work; reloading
the model also worked. The cause was that omlx core stored only the
first two elements of every cache layer's state tuple, silently
dropping `PoolingCache.state[2]` (the `pooled` compressed-attention
buffer) on every save and reconstructing it as `None` on every load.

This module exercises the full extract → store → load → reconstruct
pipeline with a PoolingCache layer to confirm the third element
survives end-to-end. Each test is marked ``slow`` because the SSD
write path goes through a background thread; if a future regression
re-introduces the truncation, these tests will catch it before it
hits a real V4 model run.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def applied_patch():
    """Apply the deepseek_v4 patch so PoolingCache is importable."""
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    return True


def _wait_for_file(path, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def test_pooling_cache_round_trip_through_paged_ssd(applied_patch, tmp_path):
    """Full save → wait-for-disk → load round-trip via PagedSSDCacheManager.

    Mirrors what omlx scheduler does on prefill block boundary: take
    a layer state via the handler interface, hand it as an
    ``__nstate__`` marker to ``save_block``, then reconstruct on hit.
    """
    import mlx.core as mx

    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.patches.deepseek_v4.cache_handlers import PoolingCacheHandler

    manager = PagedSSDCacheManager(
        cache_dir=tmp_path / "v4_e2e", max_size_bytes=100 * 1024**2
    )
    block_hash = b"v4_e2e_pooling_cache"

    # Build a representative PoolingCache state. buf_kv / buf_gate are
    # zero-length placeholders (remainder cleared); pooled holds the
    # accumulated compressed sequence — this is the element the bug
    # used to drop.
    from mlx_lm.models.cache import PoolingCache

    ratio = 4
    cache = PoolingCache(ratio=ratio)
    pooled = mx.arange(1 * 16 * 12, dtype=mx.float32).reshape(1, 16, 12)
    mx.eval(pooled)
    cache.state = (None, None, pooled)

    # Serialize via the handler exactly as scheduler /
    # prefix_cache will after Commit 1.
    handler = PoolingCacheHandler()
    elements = handler.serialize_state(cache)
    layer_marker = ("__nstate__", "PoolingCache", list(elements))

    # Save and wait for the background writer to flush.
    manager.save_block(
        block_hash, [layer_marker], token_count=16, layer_cache_types=["PoolingCache"]
    )
    assert _wait_for_file(manager._get_file_path(block_hash))

    # Load and reconstruct.
    loaded = manager.load_block(block_hash)
    assert loaded is not None
    assert len(loaded) == 1
    marker = loaded[0]
    assert marker[0] == "__nstate__"
    assert marker[1] == "PoolingCache"
    restored_elements = marker[2]
    assert len(restored_elements) == 3

    # Critical: the third element survives byte-equal.
    rest_pooled = restored_elements[2]
    assert mx.max(mx.abs(rest_pooled - pooled)).item() == 0.0

    # And the handler can rebuild a PoolingCache from those elements.
    restored_cache = handler.deserialize_state(
        tuple(restored_elements), meta_state=ratio
    )
    assert restored_cache is not None
    assert restored_cache.ratio == ratio
    _, _, restored_pool_tensor = restored_cache.state
    assert mx.max(mx.abs(restored_pool_tensor - pooled)).item() == 0.0

    manager.close()


def test_two_session_simulation_pooled_preserved(applied_patch, tmp_path):
    """Simulate the original bug shape: session 1 saves cache for a prompt,
    session 2 (same prompt → cache hit) should reconstruct to a state
    byte-equal with what session 1 stored. Pre-fix, ``pooled`` came
    back ``None`` here and the model would prefill on a fresh
    sliding-window-only context — exactly the user-reported collapse.
    """
    import mlx.core as mx

    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.patches.deepseek_v4.cache_handlers import PoolingCacheHandler

    manager = PagedSSDCacheManager(
        cache_dir=tmp_path / "two_session", max_size_bytes=100 * 1024**2
    )
    block_hash = b"two_session_block___"

    from mlx_lm.models.cache import PoolingCache

    ratio = 4
    handler = PoolingCacheHandler()

    # Session 1: build a PoolingCache and store it.
    session1_cache = PoolingCache(ratio=ratio)
    pooled_s1 = mx.arange(1 * 24 * 16, dtype=mx.float32).reshape(1, 24, 16)
    mx.eval(pooled_s1)
    session1_cache.state = (None, None, pooled_s1)

    s1_elements = handler.serialize_state(session1_cache)
    manager.save_block(
        block_hash,
        [("__nstate__", "PoolingCache", list(s1_elements))],
        token_count=24,
        layer_cache_types=["PoolingCache"],
    )
    assert _wait_for_file(manager._get_file_path(block_hash))

    # Session 2: the same block hash is hit; reconstruct.
    loaded = manager.load_block(block_hash)
    assert loaded is not None
    s2_marker = loaded[0]
    s2_cache = handler.deserialize_state(tuple(s2_marker[2]), meta_state=ratio)
    assert s2_cache is not None

    # Session 2's pooled state must match session 1's exactly.
    _, _, s2_pooled = s2_cache.state
    assert (
        s2_pooled is not None
    ), "pooled element was dropped — V4 corruption regression"
    assert s2_pooled.shape == pooled_s1.shape
    assert mx.max(mx.abs(s2_pooled - pooled_s1)).item() == 0.0

    manager.close()
