import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from omlx.patches.deepseek_v41_dfly2.dfly2_sparse_attention import selected_target_attention


def _eval(x):
    mx.eval(x)
    return np.asarray(x)


def test_selected_target_attention_uses_only_valid_rows():
    # B=1,T=1,H=1,D=2,R=3,V=1. q=[1,0], so rows 0 and 1 score 1 and 0.
    query = mx.array([[[[1.0, 0.0]]]])
    keys = mx.array([[[[[1.0, 0.0]], [[0.0, 1.0]], [[100.0, 0.0]]]]])
    values = mx.array([[[[[2.0]], [[10.0]], [[999.0]]]]])
    out = selected_target_attention(query, keys, values, mx.array([[[True, True, False]]]))
    expected = (np.exp(1 / np.sqrt(2)) * 2.0 + 10.0) / (np.exp(1 / np.sqrt(2)) + 1.0)
    np.testing.assert_allclose(_eval(out), [[[[expected]]]], rtol=1e-5, atol=1e-5)


def test_selected_target_attention_returns_zero_for_all_padding():
    out = selected_target_attention(
        mx.ones((1, 1, 1, 2)),
        mx.ones((1, 1, 2, 1, 2)),
        mx.ones((1, 1, 2, 1, 3)),
        mx.array([[[False, False]]]),
    )
    np.testing.assert_array_equal(_eval(out), np.zeros((1, 1, 1, 3), dtype=np.float32))


def test_selected_target_attention_rejects_bad_mask_shape():
    with pytest.raises(ValueError, match="valid_mask"):
        selected_target_attention(
            mx.ones((1, 1, 1, 2)),
            mx.ones((1, 1, 2, 1, 2)),
            mx.ones((1, 1, 2, 1, 3)),
            mx.ones((1, 1, 2, 1)),
        )
