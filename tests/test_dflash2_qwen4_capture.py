# SPDX-License-Identifier: Apache-2.0
"""DFlash2 reads the hyper-connection mean of completed qwen4 layers."""

from __future__ import annotations

import mlx.core as mx
import pytest

from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config, qwen4_language


def test_dflash2_capture_is_the_hc_mean_of_completed_layers(monkeypatch):
    """The DFlash2 drafter reads the completed layer output averaged over the
    hyper-connection streams, not the mixer's mixed representation."""
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    model.model._omlx_dflash_capture_mean = True

    recorded = []
    original = qwen4_language.Qwen4ExpDecoderLayer.__call__

    def spy(self, hidden_states, *args, **kwargs):
        out = original(self, hidden_states, *args, **kwargs)
        recorded.append(out)
        return out

    monkeypatch.setattr(qwen4_language.Qwen4ExpDecoderLayer, "__call__", spy)
    output = model(
        mx.array([[2, 3, 4]], dtype=mx.int32),
        cache=model.make_cache(),
        return_hidden=True,
        capture_layer_ids=[0, 1],
    )
    captured = output.hidden_states[:-1]
    mx.eval(output.logits, captured)
    assert len(captured) == 2

    hc, hidden = model.model.args.hc_count, model.model.args.hidden_size
    for index, layer_output in enumerate(recorded):
        if isinstance(layer_output, tuple):
            hyper, branch, gate = layer_output[:3]
            completed = qwen4_language._hc_write(hyper, branch, gate)
        else:
            completed = layer_output
        expected = completed.reshape(*completed.shape[:-1], hc, hidden).mean(
            axis=-2
        )
        assert captured[index].shape == expected.shape
        assert mx.array_equal(captured[index], expected).item()
