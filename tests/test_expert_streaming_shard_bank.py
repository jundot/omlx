# SPDX-License-Identifier: Apache-2.0
"""ExpertBackingStore bounds and pin accounting.

- ``expert_run`` rejects out-of-range runs instead of silently clamping
  (clamping would serve the WRONG experts under the requested ids).
- ``pin_expert`` counts only newly wired pages — adjacent experts share
  the boundary page and must not double-charge ``pinned_bytes``.
"""
import json

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.expert_streaming.shard_bank import (
    ExpertBackingStore,
    _PAGE_SIZE,
)


def _store(tmp_path, e=8):
    w = mx.random.normal((e, 16, 8), dtype=mx.float16)
    key = "mlp.switch_mlp.gate_proj.weight"
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {key: w})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors"}})
    )
    return ExpertBackingStore(tmp_path), key, e


def test_expert_run_reads_exact_range(tmp_path):
    store, key, e = _store(tmp_path)
    try:
        rows = store.load_expert_run(key, 2, 3)
        assert len(rows) == 3
        for j, row in enumerate(rows):
            np.testing.assert_array_equal(
                np.asarray(row),
                np.asarray(store.load_expert_slice(key, 2 + j)),
            )
    finally:
        store.close()


def test_expert_run_rejects_out_of_range(tmp_path):
    store, key, e = _store(tmp_path)
    try:
        with pytest.raises(ValueError, match="out of range"):
            store.load_expert_run(key, -1, 1)
        with pytest.raises(ValueError, match="out of range"):
            store.load_expert_run(key, e, 1)
        with pytest.raises(ValueError, match="exceeds"):
            store.load_expert_run(key, e - 1, 2)
        with pytest.raises(ValueError, match="exceeds"):
            store.load_expert_run(key, 0, 0)
        # Edge values must be exact, never clamped.
        with pytest.raises(ValueError):
            store.load_expert_run(key, 0, e + 1)
    finally:
        store.close()




def test_pin_counts_unique_pages(tmp_path, monkeypatch):
    """Adjacent experts share the boundary page: pinning both must count
    the shared page once."""
    store, key, e = _store(tmp_path)
    try:
        locked_calls = []
        # Stub the mlock at the reader level so the accounting path runs
        # on every platform (real mlock needs privileges); the stub
        # reports one page locked per call.
        reader = store._reader_for_key(key, 0)
        orig = reader.pin_expert
        reader.pin_expert = lambda k, i: (
            locked_calls.append(i) or _PAGE_SIZE
        )
        try:
            n0 = store.pin_expert(key, 0)
            n1 = store.pin_expert(key, 1)
            assert n0 == _PAGE_SIZE and n1 == _PAGE_SIZE
            pages = set()
            for eid in (0, 1):
                off, end = reader.expert_byte_range(key, eid)
                pages.update(
                    range(off // _PAGE_SIZE, (end + _PAGE_SIZE - 1) // _PAGE_SIZE)
                )
            # Unique pages only — no boundary double charge.
            assert store.pinned_bytes == len(pages) * _PAGE_SIZE
        finally:
            reader.pin_expert = orig
        # Duplicate pin is a no-op.
        assert store.pin_expert(key, 0) == 0
        assert store.pinned_count == 2
    finally:
        store.close()
