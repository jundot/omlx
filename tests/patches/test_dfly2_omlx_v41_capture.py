import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from omlx.patches.deepseek_v41_dfly2.dsv41_bridge import DSV41TargetBridge
from omlx.patches.deepseek_v41_dfly2.omlx_v41_capture import capture_attention_slice


def _as(value):
    mx.eval(value)
    return np.asarray(value)


def test_capture_converts_compressed_indexer_coordinates_to_absolute_target_positions():
    bridge = DSV41TargetBridge(target_layers=(28,), index_topk=2, local_window=1)
    # Native DSV41 indexer emits compressed-cache coordinates. With ratio=4,
    # index 2 denotes absolute target anchor position 8, not position 2.
    capture_attention_slice(
        bridge=bridge,
        layer=28,
        start=4,
        ratio=4,
        window_latents=mx.array([[[40.0], [50.0]]]),
        compressed_latents=mx.array([[[400.0], [800.0]]]),
        compressed_start=4,
        native_index_topk=mx.array([[[0, 2], [0, 2]]]),
    )
    history = bridge.history(layer=28, anchor_position=6)
    np.testing.assert_array_equal(_as(history.indices), [[-1, 8, 5]])
    np.testing.assert_array_equal(_as(history.latents).reshape(-1), [0.0, 800.0, 50.0])
    # Position 0 is unavailable in this request-local bridge and remains masked.
    np.testing.assert_array_equal(_as(history.valid), [[False, True, True]])


def test_capture_preserves_native_invalid_index_sentinel_when_scaling_coordinates():
    bridge = DSV41TargetBridge(target_layers=(28,), index_topk=2, local_window=0)
    capture_attention_slice(
        bridge=bridge,
        layer=28,
        start=0,
        ratio=2,
        window_latents=mx.array([[[1.0]]]),
        compressed_latents=mx.array([[[10.0]]]),
        compressed_start=0,
        native_index_topk=mx.array([[[-1, 0]]]),
    )
    history = bridge.history(layer=28, anchor_position=1)
    np.testing.assert_array_equal(_as(history.indices), [[-1, 0]])
    np.testing.assert_array_equal(_as(history.valid), [[False, True]])
