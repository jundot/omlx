"""K2 output-head calibration preserves hidden-state normalization."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.oq import OQImatrixCollector, _collect_k2_horizon_lm_head_imatrix
from omlx.patches.k2_horizon.k2_horizon_model import Model, ModelArgs
from test_k2_horizon import small_config


@pytest.mark.parametrize("head", ["untied", "tied", "missing"])
@pytest.mark.parametrize("four_dimensional", [False, True])
def test_head_calibration_without_logits(monkeypatch, head, four_dimensional):
    mx.random.seed(212)
    model = Model(ModelArgs.from_dict(small_config(tie_word_embeddings=head == "tied")))
    if head == "missing":
        del model.lm_head
    hidden = mx.random.normal((1, 2, 3, 64) if four_dimensional else (1, 2, 64))
    expected = model.model.norm(hidden.mean(axis=2) if four_dimensional else hidden)
    mx.eval(expected)

    def no_logits(*args, **kwargs):
        raise AssertionError("Calibration must not invoke a vocabulary projection")

    monkeypatch.setattr(nn.Linear, "__call__", no_logits)
    collector = OQImatrixCollector()
    collector.install(model)
    try:
        assert _collect_k2_horizon_lm_head_imatrix(model, hidden, collector) == (
            head == "untied"
        )
        if head == "untied":
            entry = collector.entries["lm_head"]
            np.testing.assert_allclose(
                entry.in_sum2,
                np.square(np.asarray(expected)).sum(axis=(0, 1)),
                rtol=1e-6,
            )
            assert entry.counts.tolist() == [2]
        else:
            assert collector.entries == {}
    finally:
        collector.restore(model)
