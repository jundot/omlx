# SPDX-License-Identifier: Apache-2.0
"""Per-layer eager dispatch in Qwen4ExpModel: fires for decode/verify rows only and never changes outputs."""
from __future__ import annotations

import sys

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config


def _model():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    config = _tiny_config()
    mx.random.seed(1)
    model = LanguageModel(config.text_config, config)
    mx.eval(model.parameters())
    return model, config


def _count_dispatches(monkeypatch, language, rows: int):
    calls = []
    real = mx.async_eval

    def spy(*a, **k):
        # Count only the decoder-layer loop's dispatches; mlx-lm's gated-delta kernel also async_evals.
        if sys._getframe(1).f_code.co_filename == language.__file__:
            calls.append(len(a))
        return real(*a, **k)

    monkeypatch.setattr(language.mx, "async_eval", spy)
    model, config = _model()
    inputs = mx.random.randint(0, config.text_config.vocab_size, (1, rows))
    out = model(inputs)
    out = getattr(out, "logits", out)
    mx.eval(out)
    return len(calls), out


@pytest.mark.parametrize("rows,expected", [(1, True), (4, True), (64, True), (65, False), (200, False)])
def test_dispatches_once_per_layer_for_small_row_counts(monkeypatch, rows, expected):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    monkeypatch.setattr(language, "_EAGER_DISPATCH", True)
    count, _ = _count_dispatches(monkeypatch, language, rows)
    layers = _tiny_config().text_config.num_hidden_layers
    assert count == (layers if expected else 0)


def test_kill_switch_and_bitwise_equality(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language

    monkeypatch.setattr(language, "_EAGER_DISPATCH", False)
    off_count, off = _count_dispatches(monkeypatch, language, 4)
    monkeypatch.setattr(language, "_EAGER_DISPATCH", True)
    on_count, on = _count_dispatches(monkeypatch, language, 4)
    assert off_count == 0 and on_count > 0
    assert mx.array_equal(on.view(mx.uint32) if on.dtype == mx.float32 else on.view(mx.uint16),
                          off.view(mx.uint32) if off.dtype == mx.float32 else off.view(mx.uint16)).item()
