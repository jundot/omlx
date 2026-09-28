# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the opt-in single-launch Qwen4 multi-row GDN decode.

With ``OMLX_QWEN4_FUSED_GDN=1`` two- and three-row one-token decode must
reproduce the stock chain bit for bit (layer output, conv state, recurrent
state), and every other shape (B1 decode, speculative verify, prefill) must
keep its existing route.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen3_5 import language as q35
from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
from mlx_vlm.speculative.cache_state import start_speculative_cache

from omlx.patches import qwen4_gdn_fused_decode as fused_mod
from omlx.patches import qwen35_gdn_prework as prework_mod
from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

HV, DV, DK, C = fused_mod.HV, fused_mod.DV, fused_mod.DK, fused_mod.CONV_DIM
# Shipped oQ4e allocations: default q4/g64 qkv with q5/g128 gates, and the
# q6/g64 layers.
ALLOCATIONS = {
    "q4": {"in_proj_qkv": (4, 64), "": (5, 128)},
    "q6": {
        "in_proj_qkv": (6, 64),
        "in_proj_z": (6, 64),
        "in_proj_a": (6, 64),
        "in_proj_b": (6, 64),
        "": (5, 128),
    },
}


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    apply_mlx_vlm_qwen4_exp_compat_patch()
    cls = q35.Qwen3_5GatedDeltaNet
    monkeypatch.setattr(cls, "__call__", cls.__call__)
    monkeypatch.setattr(
        Qwen3_5BatchInvariantForward,
        "_gated_delta",
        Qwen3_5BatchInvariantForward._gated_delta,
    )
    monkeypatch.setattr(prework_mod, "_PATCHED", False)
    assert prework_mod.apply_qwen35_gdn_prework_patch()


def _layer(allocation="q4", seed=0):
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedDeltaNet

    config = SimpleNamespace(
        hidden_size=2560,
        linear_num_value_heads=HV,
        linear_num_key_heads=16,
        linear_key_head_dim=DK,
        linear_value_head_dim=DV,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
    )
    mx.random.seed(seed)
    layer = Qwen4ExpGatedDeltaNet(config)
    layer.conv1d.weight = mx.random.normal(layer.conv1d.weight.shape) * 0.3
    layer.A_log = mx.random.normal((HV,)) * 0.5
    layer.dt_bias = mx.random.normal((HV,)) * 0.5
    layer.norm.weight = 1 + mx.random.normal((DV,)) * 0.1
    layer.set_dtype(mx.bfloat16)
    signatures = ALLOCATIONS[allocation]

    def predicate(path, module):
        if not isinstance(module, nn.Linear):
            return False
        bits, group = signatures.get(path, signatures[""])
        return {"bits": bits, "group_size": group}

    nn.quantize(layer, class_predicate=predicate)
    layer.eval()
    mx.eval(layer.parameters())
    return layer


def _cache(batch, seed):
    mx.random.seed(seed)
    cache = ArraysCache(size=2)
    cache[0] = (mx.random.normal((batch, 3, C)) * 0.5).astype(mx.bfloat16)
    cache[1] = mx.random.normal((batch, HV, DV, DK)) * 0.05
    return cache


def _inputs(batch, length, seed):
    mx.random.seed(seed)
    return mx.random.normal((batch, length, 2560)).astype(mx.bfloat16)


def _count_fused(monkeypatch):
    calls = []
    step = fused_mod.fused_step

    def record(*args, **kwargs):
        calls.append(args[0].shape)
        return step(*args, **kwargs)

    monkeypatch.setattr(fused_mod, "fused_step", record)
    return calls


def _decode(layer, cache, steps, batch, seed):
    outputs = []
    for step in range(steps):
        outputs.append(layer(_inputs(batch, 1, seed + step), cache=cache))
        mx.eval(outputs[-1], cache.state)
    return outputs


def _assert_equal(expected, observed):
    for (_, a), (_, b) in zip(tree_flatten(expected), tree_flatten(observed)):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert mx.array_equal(a, b).item()


@pytest.mark.parametrize("allocation", ["q4", "q6"])
@pytest.mark.parametrize("batch", [2, 3])
def test_multi_row_decode_matches_stock_bit_exact(monkeypatch, allocation, batch):
    layer = _layer(allocation)
    reference_cache = _cache(batch, 1)
    cache = copy.deepcopy(reference_cache)
    monkeypatch.delenv("OMLX_QWEN4_FUSED_GDN", raising=False)
    expected = _decode(layer, reference_cache, 6, batch, 100)

    monkeypatch.setenv("OMLX_QWEN4_FUSED_GDN", "1")
    calls = _count_fused(monkeypatch)
    observed = _decode(layer, cache, 6, batch, 100)
    assert calls == [(batch, 1, C)] * 6
    _assert_equal(expected, observed)
    _assert_equal(reference_cache.state, cache.state)


@pytest.mark.parametrize("batch", [2, 3])
def test_multi_row_decode_with_exhausted_left_padding(monkeypatch, batch):
    # A batched cache keeps its left_padding array after the pads are consumed;
    # with no mask the stock chain ignores it.
    layer = _layer("q6", seed=8)
    reference_cache = _cache(batch, 3)
    reference_cache.left_padding = mx.array([-4] * batch)
    cache = copy.deepcopy(reference_cache)
    monkeypatch.delenv("OMLX_QWEN4_FUSED_GDN", raising=False)
    expected = _decode(layer, reference_cache, 3, batch, 200)

    monkeypatch.setenv("OMLX_QWEN4_FUSED_GDN", "1")
    calls = _count_fused(monkeypatch)
    observed = _decode(layer, cache, 3, batch, 200)
    assert calls == [(batch, 1, C)] * 3
    _assert_equal(expected, observed)
    _assert_equal(reference_cache.state, cache.state)
    assert reference_cache.left_padding.tolist() == cache.left_padding.tolist()


def test_b1_decode_keeps_existing_route(monkeypatch):
    layer = _layer("q4", seed=6)
    monkeypatch.setenv("OMLX_QWEN4_FUSED_GDN", "1")
    calls = _count_fused(monkeypatch)
    # Both B1 arms: the three-kernel decode and (left padding) the stock chain.
    padded = _cache(1, 2)
    padded.left_padding = mx.array([-4])
    for cache in (_cache(1, 1), padded):
        _decode(layer, cache, 2, 1, 1)
    assert calls == []


@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_verify_keeps_existing_route(monkeypatch, batch, length):
    from mlx_vlm.models.qwen4_exp.language import _VERIFIER

    layer = _layer("q4", seed=3)
    inputs = _inputs(batch, length, 7)
    monkeypatch.setenv("OMLX_QWEN4_FUSED_GDN", "1")
    calls = _count_fused(monkeypatch)
    for speculative in (True, False):
        cache = _cache(batch, 2)
        transaction = start_speculative_cache([cache], length) if speculative else None
        mx.eval(_VERIFIER._gated_delta(layer, inputs, None, cache), cache.state)
        if transaction is not None:
            transaction.commit([length] * batch)
    assert calls == []


def test_unsupported_shapes_and_default_off_keep_the_existing_route(monkeypatch):
    layer = _layer("q4", seed=6)
    calls = _count_fused(monkeypatch)

    monkeypatch.delenv("OMLX_QWEN4_FUSED_GDN", raising=False)
    mx.eval(layer(_inputs(2, 1, 1), cache=_cache(2, 1)))
    assert calls == []

    monkeypatch.setenv("OMLX_QWEN4_FUSED_GDN", "1")
    mx.eval(layer(_inputs(4, 1, 1), cache=_cache(4, 1)))  # too many rows
    mx.eval(layer(_inputs(2, 8, 1), cache=_cache(2, 1)))  # prefill-sized
    empty = ArraysCache(size=2)  # first chunk: no recurrent state yet
    mx.eval(layer(_inputs(2, 1, 1), cache=empty))
    ragged = _cache(2, 1)
    ragged.lengths = mx.array([1, 1])
    mx.eval(layer(_inputs(2, 1, 1), cache=ragged))
    mask = mx.array([[True], [False]])
    mx.eval(layer(_inputs(2, 1, 1), mask=mask, cache=_cache(2, 1)))
    speculating = _cache(2, 1)
    transaction = start_speculative_cache([speculating], 1)
    mx.eval(layer(_inputs(2, 1, 1), cache=speculating))
    transaction.abort()
    assert calls == []
