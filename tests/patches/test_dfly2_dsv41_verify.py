import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from omlx.patches.deepseek_v41_dfly2.dsv41_verify import greedy_prefix


def test_greedy_prefix_accepts_exact_target_prefix_and_does_not_rollback():
    rollbacks = []
    result = greedy_prefix(
        proposals=mx.array([[7, 8, 9]], dtype=mx.int32),
        target_logits=(
            mx.array([[0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0]]),
            mx.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0]]),
            mx.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]]),
        ),
        anchor_position=17,
        rollback=rollbacks.append,
    )
    assert result.accepted == 3
    assert result.rejected_position is None
    assert rollbacks == []
    np.testing.assert_array_equal(np.asarray(result.target_tokens), [[7, 8, 9]])


def test_greedy_prefix_rolls_back_first_rejected_target_position_only():
    rollbacks = []
    result = greedy_prefix(
        proposals=mx.array([[7, 8, 9]], dtype=mx.int32),
        target_logits=(
            mx.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0]]),
            mx.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 8.0]]),
            mx.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0]]),
        ),
        anchor_position=17,
        rollback=rollbacks.append,
    )
    assert result.accepted == 1
    assert result.rejected_position == 18
    # Candidate rows at positions >=18 are rejected; accepted row 17 remains.
    assert rollbacks == [18]
    np.testing.assert_array_equal(np.asarray(result.target_tokens), [[7, 7, 9]])
