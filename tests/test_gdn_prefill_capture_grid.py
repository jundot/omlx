# SPDX-License-Identifier: Apache-2.0
"""Prefill captures follow the gdn_snapshot_interval grid.

With interval N > 1 a store keeps recurrent state only on the N-block grid
and at its last full block, so prefill captures only those boundaries.
"""

from __future__ import annotations

from omlx.cache.prefix_cache import gdn_snapshot_interval_for
from omlx.scheduler import _prefill_boundary_kept

BS = 2048
PROMPT = 10 * BS + 300
FULL = [i * BS for i in range(1, 11)]


def _kept(interval, prompt_tokens=PROMPT):
    return [t for t in FULL if _prefill_boundary_kept(t, interval, BS, prompt_tokens)]


def test_capture_follows_snapshot_grid():
    assert _kept(4) == [4 * BS, 8 * BS, 10 * BS]  # grid + last full block


def test_interval_one_captures_every_boundary():
    assert _kept(1) == FULL


def test_unknown_prompt_length_captures_every_boundary():
    assert _kept(4, prompt_tokens=0) == FULL


def test_interval_applies_only_to_arrays_layouts():
    """Composite and rotating layers keep a snapshot at every block."""
    assert gdn_snapshot_interval_for(4, ["KVCache", "ArraysCache"]) == 4
    assert gdn_snapshot_interval_for(4, ["KVCache", "RotatingKVCache"]) == 1
    assert gdn_snapshot_interval_for(4, ["KVCache", "CacheList"]) == 1
    assert gdn_snapshot_interval_for(4, ["KVCache", "KVCache"]) == 1
    assert gdn_snapshot_interval_for(4, None) == 1
    assert gdn_snapshot_interval_for(1, ["KVCache", "ArraysCache"]) == 1
