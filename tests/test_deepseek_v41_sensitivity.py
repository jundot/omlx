"""V4.1 layer sensitivity preserves the full-forward cache and weights."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.patches.deepseek_v41.cache import DeepseekV41Cache
from omlx.patches.deepseek_v41.sensitivity import measure_sensitivity


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 64, bias=False)
        self.proj.weight = mx.sin(mx.arange(4096).reshape(64, 64)).astype(mx.bfloat16)

    def __call__(self, h, pre, cache, shared, start, image_mask):
        old = cache[1]
        if old is not None:
            h = h + old
        h = h + shared.get("h", mx.zeros_like(h))
        value = self.proj(h)
        cache[1] = value
        shared["h"] = value
        return value, pre


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [Block(), Block()]

    def make_cache(self):
        return [DeepseekV41Cache(), DeepseekV41Cache()]

    def __call__(self, x, cache):
        h = mx.sin(x[..., None] + mx.arange(64)).astype(mx.bfloat16)
        pre = mx.ones_like(h)
        shared = {}
        for block, state in zip(self.layers, cache):
            h, pre = block(h, pre, state, shared, 0, None)
        return h


@pytest.mark.parametrize("cancel_phase", [None, "capture", "measure"])
def test_measure_sensitivity_restores_layers_weights_and_snapshots(
    monkeypatch, cancel_phase
):
    import omlx.oq as oq

    with mx.stream(mx.cpu):
        model = SimpleNamespace(language_model=Core())
        core = model.language_model
        originals = list(core.layers)
        projections = [layer.proj for layer in originals]
        tokens = mx.array([[1, 2], [3, 4]])
        monkeypatch.setattr(oq, "_load_calibration_data", lambda *args: tokens)
        expected = core(tokens, cache=core.make_cache())
        mx.eval(expected)

        def progress(phase, done, total):
            if phase == cancel_phase:
                raise RuntimeError("cancelled")

        if cancel_phase:
            with pytest.raises(RuntimeError, match="cancelled"):
                measure_sensitivity(
                    model, None, num_samples=2, seq_length=2, progress=progress
                )
        else:
            scores = measure_sensitivity(model, None, num_samples=2, seq_length=2)
            repeated = measure_sensitivity(model, None, num_samples=2, seq_length=2)
            assert scores == repeated
            assert set(scores) == {0, 1}
            assert all(0 < value < 1 for value in scores.values())
        assert all(a is b for a, b in zip(core.layers, originals))
        assert all(layer.proj is proj for layer, proj in zip(core.layers, projections))
        actual = core(tokens, cache=core.make_cache())
        assert mx.array_equal(actual, expected).item()


def test_cancellation_during_second_projection_restores_first():
    from omlx.patches.deepseek_v41.sensitivity import _quantized_block

    with mx.stream(mx.cpu):
        block = nn.Module()
        block.first = nn.Linear(64, 64, bias=False)
        block.second = nn.Linear(64, 64, bias=False)
        original = (block.first, block.second)
        calls = []

        def cancel(done, total):
            calls.append((done, total))
            if len(calls) == 2:
                assert block.first is not original[0]
                raise RuntimeError("cancelled")

        with (
            pytest.raises(RuntimeError, match="cancelled"),
            _quantized_block(block, 3, progress=cancel),
        ):
            raise AssertionError("Cancelled quantization must not enter the body")
        assert block.first is original[0]
        assert block.second is original[1]
        assert len(calls) == 2
