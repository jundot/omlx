# SPDX-License-Identifier: Apache-2.0
"""The ordinary FP16/BF16 route must engage only for its tested cache/shape ABI."""

from __future__ import annotations

import copy

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen3_5 import language
from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
from mlx_vlm.speculative.cache_state import start_speculative_cache

from omlx.patches import qwen35_gdn_prework as prework

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.fixture(autouse=True)
def restore_hooks(monkeypatch):
    cls = language.Qwen3_5GatedDeltaNet
    monkeypatch.setattr(cls, "__call__", cls.__call__)
    monkeypatch.setattr(cls, "_omlx_gdn_prework_patched", False, raising=False)
    verifier = Qwen3_5BatchInvariantForward
    monkeypatch.setattr(verifier, "_gated_delta", verifier._gated_delta)
    monkeypatch.setattr(prework, "_PATCHED", False)
    monkeypatch.setattr(prework, "_QWEN35_DECODE_ENGAGED_LOGGED", False)


def _module(dtype=mx.float16, hidden_size=2048, num_k_heads=16, num_v_heads=32):
    # Keep the real GDN class, convolution, normalization and recurrence,
    # with cheap deterministic projections instead of large random weights.
    module = language.Qwen3_5GatedDeltaNet.__new__(language.Qwen3_5GatedDeltaNet)
    nn.Module.__init__(module)
    module.hidden_size = hidden_size
    module.num_k_heads, module.num_v_heads = num_k_heads, num_v_heads
    module.head_k_dim = module.head_v_dim = 128
    module.key_dim, module.value_dim = num_k_heads * 128, num_v_heads * 128
    module.conv_dim = 2 * module.key_dim + module.value_dim
    module.conv_kernel_size = 4
    module.conv1d = nn.Conv1d(
        module.conv_dim, module.conv_dim, 4, groups=module.conv_dim, bias=False
    )
    module.conv1d.weight = (mx.random.normal((module.conv_dim, 4, 1)) * 0.2).astype(
        dtype
    )
    module.A_log = mx.zeros((num_v_heads,), dtype=dtype)
    module.dt_bias = mx.ones((num_v_heads,), dtype=dtype)
    module.norm = language.Qwen3_5RMSNormGated(128, eps=1e-6)
    module.norm.weight = (1 + 0.1 * mx.random.normal((128,))).astype(dtype)
    conv_dim, value_dim = module.conv_dim, module.value_dim
    module.in_proj_qkv = lambda x: mx.tile(
        x, (1, 1, (conv_dim + hidden_size - 1) // hidden_size)
    )[..., :conv_dim]
    module.in_proj_z = lambda x: mx.tile(
        x, (1, 1, (value_dim + hidden_size - 1) // hidden_size)
    )[..., :value_dim]
    module.in_proj_b = lambda x: x[..., :num_v_heads] * 0.2
    module.in_proj_a = lambda x: x[..., num_v_heads : 2 * num_v_heads] * 0.2
    module.out_proj = lambda x: x[..., :hidden_size]
    module.eval()
    return module


def _cache():
    cache = ArraysCache(size=2)
    cache[0] = (mx.random.normal((1, 3, 8192)) * 0.2).astype(mx.float16)
    cache[1] = mx.random.normal((1, 32, 128, 128)) * 0.01
    return cache


def _record_kernel(monkeypatch):
    calls = []
    kernel = prework.gdn_prework_fused

    def record(*args, **kwargs):
        calls.append((args[0].shape, args[0].dtype))
        return kernel(*args, **kwargs)

    monkeypatch.setattr(prework, "gdn_prework_fused", record)
    return calls


def _assert_equal(actual, expected):
    mx.eval(actual, expected)
    for a, e in zip(actual, expected):
        assert a.dtype == e.dtype
        assert a.shape == e.shape
        assert mx.array_equal(a, e).item()
        assert mx.all(mx.isfinite(a)).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("scale", [0.0, 0.001, 0.1, 1.0, 5.0, 16.0])
@pytest.mark.parametrize("strided", [False, True])
def test_prework_matches_current_decode_convolution(scale, strided, dtype):
    mx.random.seed(71)
    module = _module(dtype)
    width = 8192 * (2 if strided else 1)
    qkv = (mx.random.normal((1, 1, width)) * scale).astype(dtype)
    state = (mx.random.normal((1, 3, width)) * scale).astype(dtype)
    if strided:
        qkv, state = qkv[..., ::2], state[..., ::2]
    conv_input = mx.concatenate([state, qkv], axis=1)
    conv = nn.silu(module._causal_conv1d_decode(conv_input))
    q, k, v = mx.split(conv, [2048, 4096], axis=-1)
    q, k = module._normalize_qk(q.reshape(1, 1, 16, 128), k.reshape(1, 1, 16, 128))
    expected = (q, k, v.reshape(1, 1, 32, 128), mx.contiguous(conv_input[:, -3:]))
    actual = prework.gdn_prework_fused(
        qkv,
        state,
        module.conv1d.weight,
        mx.array(128**-1, dtype=dtype),
        mx.array(128**-0.5, dtype=dtype),
        16,
        32,
        128,
        128,
    )
    _assert_equal(actual, expected)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "hidden_size,num_k_heads,num_v_heads",
    [(2048, 16, 32), (5120, 16, 48), (1024, 8, 16)],
)
def test_decode_matches_stock_over_prefill_decode_and_restored_cache(
    monkeypatch, dtype, hidden_size, num_k_heads, num_v_heads
):
    mx.random.seed(117)
    module = _module(dtype, hidden_size, num_k_heads, num_v_heads)
    stock = type(module).__call__
    expected_cache, actual_cache = ArraysCache(2), ArraysCache(2)
    assert prework.apply_qwen35_gdn_prework_patch()
    calls = _record_kernel(monkeypatch)
    advance = actual_cache.advance
    advances = []

    def record_advance(n):
        advances.append(n)
        advance(n)

    monkeypatch.setattr(actual_cache, "advance", record_advance)
    for length in [8] + [1] * 16 + [3, 1]:
        inputs = mx.random.normal((1, length, hidden_size)).astype(dtype)
        expected = stock(module, inputs, cache=expected_cache)
        actual = module(inputs, cache=actual_cache)
        _assert_equal((actual, *actual_cache.state), (expected, *expected_cache.state))
    assert len(calls) == 17
    assert advances == [8] + [1] * 16 + [3, 1]

    # Exercise a prefix cache snapshot with fresh cache-object metadata.
    expected_cache = copy.deepcopy(expected_cache)
    restored = ArraysCache(2)
    restored.state = copy.deepcopy(actual_cache.state)
    inputs = mx.random.normal((1, 1, hidden_size)).astype(dtype)
    actual = module(inputs, cache=restored)
    expected = stock(module, inputs, cache=expected_cache)
    _assert_equal((actual, *restored.state), (expected, *expected_cache.state))
    assert len(calls) == 18


@pytest.mark.parametrize(
    "case",
    [
        "batch",
        "prefill",
        "bf16_input",
        "fp32_input",
        "mask",
        "no_cache",
        "cold_conv",
        "cold_recurrent",
        "conv_shape",
        "conv_dtype",
        "recurrent_shape",
        "recurrent_dtype",
        "lengths",
        "padding",
        "speculation",
        "subclass",
        "cache_subclass",
        "training",
        "head_dim",
        "conv_bias",
        "conv_weight_dtype",
        "projected_dtype",
        "projected_shape",
        "cpu",
    ],
)
def test_decode_falls_back_before_cache_mutation(monkeypatch, case):
    module, cache = _module(), _cache()
    inputs, mask = mx.zeros((1, 1, 2048), dtype=mx.float16), None
    if case == "batch":
        inputs = mx.zeros((2, 1, 2048), dtype=mx.float16)
    elif case == "prefill":
        inputs = mx.zeros((1, 2, 2048), dtype=mx.float16)
    elif case in ("bf16_input", "fp32_input"):
        inputs = inputs.astype(mx.bfloat16 if case == "bf16_input" else mx.float32)
    elif case == "mask":
        mask = mx.array([[True]])
    elif case == "no_cache":
        cache = None
    elif case == "cold_conv":
        cache[0] = None
    elif case == "cold_recurrent":
        cache[1] = None
    elif case == "conv_shape":
        cache[0] = cache[0][:, :2]
    elif case == "conv_dtype":
        cache[0] = cache[0].astype(mx.bfloat16)
    elif case == "recurrent_shape":
        cache[1] = cache[1][:, :16]
    elif case == "recurrent_dtype":
        cache[1] = cache[1].astype(mx.float16)
    elif case == "lengths":
        cache.lengths = mx.array([1])
    elif case == "padding":
        cache.left_padding = mx.array([0])
    elif case == "speculation":
        cache.start_speculation(1)
    elif case == "subclass":
        module.__class__ = type("SpecialGDN", (type(module),), {})
    elif case == "cache_subclass":
        cache.__class__ = type("SpecialCache", (ArraysCache,), {})
    elif case == "training":
        module.train()
    elif case == "head_dim":
        module.head_k_dim = 64
    elif case == "conv_bias":
        module.conv1d.bias = mx.zeros((8192,), dtype=mx.float16)
    elif case == "conv_weight_dtype":
        module.conv1d.weight = module.conv1d.weight.astype(mx.float32)
    elif case == "projected_dtype":
        module.in_proj_qkv = lambda x: mx.zeros((1, 1, 8192), dtype=mx.float32)
    elif case == "projected_shape":
        module.in_proj_qkv = lambda x: mx.zeros((1, 1, 4096), dtype=mx.float16)
    elif case == "cpu":
        monkeypatch.setattr(mx, "default_device", lambda: mx.cpu)

    seen = []
    states = None if cache is None else tuple(cache.state)

    def stock(self, actual_inputs, mask=None, cache=None):
        seen.append((self, actual_inputs, mask, cache))
        if cache is not None:
            assert all(a is b for a, b in zip(cache.state, states))
        return "stock"

    monkeypatch.setattr(language.Qwen3_5GatedDeltaNet, "__call__", stock)
    assert prework.apply_qwen35_gdn_prework_patch()
    calls = _record_kernel(monkeypatch)
    assert module(inputs, mask=mask, cache=cache) == "stock"
    assert len(seen) == 1 and seen[0] == (module, inputs, mask, cache)
    assert calls == []


@pytest.mark.parametrize(
    "kwargs", [{"gdn_sink": []}, {"target_verify": True}, {"extra": 1}]
)
def test_decode_forwards_runtime_extensions(monkeypatch, kwargs):
    module, cache = _module(), _cache()
    seen = []

    def stock(self, inputs, mask=None, cache=None, **extensions):
        seen.append(extensions)
        return "stock"

    monkeypatch.setattr(language.Qwen3_5GatedDeltaNet, "__call__", stock)
    assert prework.apply_qwen35_gdn_prework_patch()
    calls = _record_kernel(monkeypatch)
    assert (
        module(mx.zeros((1, 1, 2048), dtype=mx.float16), cache=cache, **kwargs)
        == "stock"
    )
    assert seen == [kwargs] and calls == []


def test_decode_does_not_commit_or_retry_after_fused_failure(monkeypatch):
    module, cache = _module(), _cache()
    before = tuple(cache.state)
    assert prework.apply_qwen35_gdn_prework_patch()
    calls = _record_kernel(monkeypatch)

    def fail_norm(*args):
        raise RuntimeError("norm failed")

    module.norm = fail_norm
    with pytest.raises(RuntimeError, match="norm failed"):
        module(mx.zeros((1, 1, 2048), dtype=mx.float16), cache=cache)
    assert len(calls) == 1
    assert all(a is b for a, b in zip(before, cache.state))


def test_decode_speculation_keeps_stock_history_and_commit(monkeypatch):
    module = _module()
    stock = type(module).__call__
    cache = _cache()
    reference_cache = copy.deepcopy(cache)
    transaction = start_speculative_cache([cache], 1)
    reference_transaction = start_speculative_cache([reference_cache], 1)
    inputs = mx.random.normal((1, 1, 2048)).astype(mx.float16)
    expected = stock(module, inputs, cache=reference_cache)
    assert prework.apply_qwen35_gdn_prework_patch()
    calls = _record_kernel(monkeypatch)
    actual = module(inputs, cache=cache)
    _assert_equal((actual, *cache.state), (expected, *reference_cache.state))
    transaction.commit([1])
    reference_transaction.commit([1])
    _assert_equal(cache.state, reference_cache.state)
    assert calls == []
