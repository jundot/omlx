# SPDX-License-Identifier: Apache-2.0
"""Real tiny-model coverage for capacity-preserving singleton decoding."""

import copy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_vlm.models.cache import BatchKVCache
from mlx_vlm.models.qwen3_5.config import TextConfig
from mlx_vlm.models.qwen3_5.language import LanguageModel, Qwen3_5Model

from omlx.patches.prism_hadamard.decode import CapacityPreservingModel


def _model():
    mx.random.seed(7)
    return LanguageModel(
        TextConfig(
            model_type="qwen3_5_text",
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=32,
            full_attention_interval=2,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
            max_position_embeddings=2048,
            rope_parameters={
                "type": "default",
                "mrope_section": [1, 1, 0],
                "rope_theta": 10000,
                "partial_rotary_factor": 0.25,
            },
        ),
        config=SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=29,
            video_token_id=30,
            vision_start_token_id=31,
        ),
    )


@pytest.mark.parametrize(
    "batch,padding,steps", [(1, 0, 1), (1, 2, 1), (2, 0, 1), (1, 0, 3)]
)
def test_decode_matches_upstream_and_preserves_capacity(batch, padding, steps):
    reference = _model()
    model = copy.deepcopy(reference)
    model.model.__class__ = CapacityPreservingModel
    seed = reference.make_cache()
    mx.eval(reference(mx.array([[1, 2, 3, 4]]), cache=seed).logits)
    mx.eval(model(mx.array([[1, 2, 3, 4]]), cache=model.make_cache()).logits)
    cache = [type(c).merge([copy.deepcopy(c) for _ in range(batch)]) for c in seed]
    if padding:
        for c in cache:
            if isinstance(c, BatchKVCache):
                c.keys = mx.pad(c.keys, [(0, 0), (0, 0), (padding, 0), (0, 0)])
                c.values = mx.pad(c.values, [(0, 0), (0, 0), (padding, 0), (0, 0)])
                c._idx += padding
                c.left_padding = mx.array([padding] * batch)
    mx.eval([c.state for c in cache])
    expected_cache = copy.deepcopy(cache)
    fast = batch == 1 and padding == 0 and steps == 1
    capacities = []
    for _ in range(4):
        inputs = mx.full((batch, steps), 5, dtype=mx.int32)
        expected = reference(inputs, cache=expected_cache).logits
        actual = model(inputs, cache=cache).logits
        mx.eval(
            expected,
            actual,
            [c.state for c in cache],
            [c.state for c in expected_cache],
        )
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5
        )
        for got, want in zip(cache, expected_cache):
            if isinstance(got, BatchKVCache):
                assert got._idx == want._idx
                np.testing.assert_array_equal(
                    np.asarray(got.offset), np.asarray(want.offset)
                )
                np.testing.assert_allclose(
                    np.asarray(got.keys[:, :, : got._idx]),
                    np.asarray(want.keys[:, :, : want._idx]),
                    rtol=1e-5,
                    atol=1e-5,
                )
                np.testing.assert_allclose(
                    np.asarray(got.values[:, :, : got._idx]),
                    np.asarray(want.values[:, :, : want._idx]),
                    rtol=1e-5,
                    atol=1e-5,
                )
        capacities.append(cache[reference.model.fa_idx].keys.shape[2])
    if fast:
        assert len(set(capacities)) == 1
        assert capacities[-1] > cache[reference.model.fa_idx]._idx


def test_prefill_and_hidden_capture_keep_upstream_behavior():
    reference = _model()
    model = copy.deepcopy(reference)
    model.model.__class__ = CapacityPreservingModel
    expected_cache = reference.make_cache()
    cache = model.make_cache()
    inputs = mx.array([[1, 2, 3]])
    expected = reference(inputs, cache=expected_cache, capture_layer_ids=[0])
    actual = model(inputs, cache=cache, capture_layer_ids=[0])
    mx.eval(
        expected.logits, actual.logits, expected.hidden_states, actual.hidden_states
    )
    np.testing.assert_allclose(
        np.asarray(actual.logits), np.asarray(expected.logits), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(actual.hidden_states[0]),
        np.asarray(expected.hidden_states[0]),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "extra_kwargs,borrow",
    [
        ({"gdn_sink": None}, True),
        ({"gdn_sink": []}, False),
        ({"future_sink": []}, False),
        ({"future_option": None}, False),
    ],
)
def test_runtime_kwargs_forwarded_without_borrowing_capture_cache(
    monkeypatch, extra_kwargs, borrow
):
    """New runtime arguments reach upstream, including empty capture sinks."""
    model = _model()
    seed = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache=seed).logits)
    cache = [type(c).merge([c]) for c in seed]
    model.model.__class__ = CapacityPreservingModel
    marker = object()
    calls = []

    def upstream(self, inputs, **kwargs):
        calls.append(kwargs)
        for name, value in extra_kwargs.items():
            assert kwargs[name] is value
            if isinstance(value, list):
                value.append(marker)
        return marker

    monkeypatch.setattr(Qwen3_5Model, "__call__", upstream)
    result = model.model(mx.array([[4]]), cache=cache, **extra_kwargs)
    assert result is marker
    assert len(calls) == 1
    forwarded_cache = calls[0]["cache"]
    if borrow:
        from mlx_vlm.models.cache import KVCache

        row = forwarded_cache[model.model.fa_idx]
        assert type(row) is KVCache
        assert row.keys is cache[model.model.fa_idx].keys
    else:
        assert forwarded_cache is cache
    for value in extra_kwargs.values():
        if isinstance(value, list):
            assert value == [marker]
