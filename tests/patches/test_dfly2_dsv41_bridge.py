import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from omlx.patches.deepseek_v41_dfly2.dsv41_bridge import DSV41TargetBridge


def _as(x):
    mx.eval(x)
    return np.asarray(x)


def test_bridge_uses_anchor_minus_one_index_and_local_window():
    bridge = DSV41TargetBridge(target_layers=(28,), index_topk=2, local_window=2)
    # Position 4's indexer selects global positions 0 and 2. Draft anchor is 5.
    bridge.record(
        28,
        start=0,
        window_latents=mx.arange(6, dtype=mx.float32).reshape(1, 6, 1),
        global_positions=mx.array([0, 2], dtype=mx.int32),
        global_latents=mx.array([[[100.0], [200.0]]]),
        index_topk=mx.array([[[ -1, -1], [-1, -1], [-1, -1], [-1, -1], [0, 2], [0, 2]]]),
    )
    history = bridge.history(layer=28, anchor_position=5)
    np.testing.assert_array_equal(_as(history.indices), [[0, 2, 3, 4]])
    np.testing.assert_array_equal(_as(history.valid), [[True, True, True, True]])
    # Sparse picks take global rows; local-window rows take local target cache rows.
    np.testing.assert_array_equal(_as(history.latents).reshape(-1), [100, 200, 3, 4])


def test_bridge_dedupes_sparse_pick_that_is_also_in_window():
    bridge = DSV41TargetBridge(target_layers=(28,), index_topk=2, local_window=2)
    bridge.record(
        28,
        start=0,
        window_latents=mx.arange(5, dtype=mx.float32).reshape(1, 5, 1),
        global_positions=mx.array([0], dtype=mx.int32),
        global_latents=mx.array([[[100.0]]]),
        index_topk=mx.array([[[ -1, -1], [-1, -1], [-1, -1], [0, 2], [0, 2]]]),
    )
    history = bridge.history(layer=28, anchor_position=4)
    np.testing.assert_array_equal(_as(history.indices), [[0, 2, -1, 3]])
    np.testing.assert_array_equal(_as(history.valid), [[True, True, False, True]])
    np.testing.assert_array_equal(_as(history.latents).reshape(-1), [100, 2, 0, 3])


def test_bridge_rollback_removes_future_window_global_and_index_rows():
    bridge = DSV41TargetBridge(target_layers=(28,), index_topk=1, local_window=2)
    bridge.record(
        28,
        start=0,
        window_latents=mx.arange(6, dtype=mx.float32).reshape(1, 6, 1),
        global_positions=mx.array([0, 4], dtype=mx.int32),
        global_latents=mx.array([[[10.0], [40.0]]]),
        index_topk=mx.array([[[-1], [-1], [-1], [-1], [4], [4]]]),
    )
    before = bridge.history(layer=28, anchor_position=5)
    bridge.rollback(keep_positions=5)
    after = bridge.history(layer=28, anchor_position=5)
    np.testing.assert_array_equal(_as(after.indices), _as(before.indices))
    np.testing.assert_array_equal(_as(after.latents), _as(before.latents))
    bridge.rollback(keep_positions=4)
    # Anchor 5 can no longer use a target-index row for position 4.
    try:
        bridge.history(layer=28, anchor_position=5)
    except KeyError as exc:
        assert "anchor-1" in str(exc)
    else:
        raise AssertionError("rollback retained a future index record")


def test_bridge_rollback_removes_sparse_global_position_before_new_record_reuses_it():
    bridge = DSV41TargetBridge(target_layers=(28,), index_topk=1, local_window=0)
    bridge.record(
        28,
        start=0,
        window_latents=mx.array([[[1.0], [2.0], [3.0], [4.0], [5.0]]]),
        global_positions=mx.array([0, 4], dtype=mx.int32),
        global_latents=mx.array([[[10.0], [40.0]]]),
        index_topk=mx.array([[[-1], [-1], [-1], [-1], [4]]]),
    )
    bridge.rollback(keep_positions=4)
    # A later accepted target forward reuses absolute position 4. It must not
    # accidentally resolve through the rejected compressed row (40.0).
    bridge.record(
        28,
        start=4,
        window_latents=mx.array([[[400.0]]]),
        global_positions=mx.zeros((0,), dtype=mx.int32),
        global_latents=mx.zeros((1, 0, 1)),
        index_topk=mx.array([[[4]]]),
    )
    history = bridge.history(layer=28, anchor_position=5)
    np.testing.assert_array_equal(_as(history.indices), [[4]])
    np.testing.assert_array_equal(_as(history.latents).reshape(-1), [400.0])
