from __future__ import annotations

import inspect
import logging
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cluster import deployment
from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast


@pytest.fixture(scope="module")
def dm():
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    return sys.modules["mlx_lm.models.deepseek_v4"]


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)


class _KVNorm:
    def __init__(self):
        self.weight = _Tensor((512,), mx.bfloat16)

    def get(self, key, default=None):
        return self.weight if key == "weight" else default


class _RoPE:
    def __init__(self):
        self.freqs = _Tensor((256,), mx.float32)

    def _get_freqs(self, head_dim, inverse):
        assert head_dim == 512 and inverse is False
        return self.freqs


def _eligible_fixture(monkeypatch, dm, heads=24, tokens=1024):
    monkeypatch.setattr(dm, "_DEEPSEEK_V4_ATTN_FINALIZER_PREFILL", True)
    monkeypatch.setattr(dm, "is_dspark_verify_armed", lambda: False)
    monkeypatch.setattr(glm_fast, "is_native_available", lambda: True)
    monkeypatch.setattr(
        glm_fast,
        "has_symbol",
        lambda name: name in {"ds4_q_head_rms_rope", "ds4_kv_rms_rope"},
    )
    attn = SimpleNamespace(
        training=False,
        head_dim=512,
        kv_norm=_KVNorm(),
        rope=_RoPE(),
        config=SimpleNamespace(rms_norm_eps=1e-6),
    )
    q_raw = _Tensor((1, tokens, heads, 512), mx.bfloat16)
    kv_raw = _Tensor((1, tokens, 512), mx.bfloat16)
    return attn, q_raw, kv_raw


@pytest.mark.parametrize("heads", (24, 32, 40))
def test_proven_m1024_head_shapes_are_eligible_only_when_enabled(
    monkeypatch, dm, heads
):
    attn, q_raw, kv_raw = _eligible_fixture(monkeypatch, dm, heads)
    assert dm._attention_finalizer_native_inputs(attn, q_raw, kv_raw, 8192)

    monkeypatch.setattr(dm, "_DEEPSEEK_V4_ATTN_FINALIZER_PREFILL", False)
    assert dm._attention_finalizer_native_inputs(attn, q_raw, kv_raw, 8192) is None


def test_single_m3_h64_m2048_prefill_is_eligible(monkeypatch, dm):
    attn, q_raw, kv_raw = _eligible_fixture(
        monkeypatch, dm, heads=64, tokens=2048
    )
    assert dm._attention_finalizer_native_inputs(attn, q_raw, kv_raw, 8192)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda attn, q, kv: setattr(attn, "training", True),
        lambda attn, q, kv: setattr(q, "shape", (1, 1023, 24, 512)),
        lambda attn, q, kv: setattr(q, "shape", (1, 1024, 16, 512)),
        lambda attn, q, kv: setattr(q, "dtype", mx.float16),
        lambda attn, q, kv: setattr(kv, "shape", (1, 512, 512)),
        lambda attn, q, kv: setattr(kv, "dtype", mx.float16),
        lambda attn, q, kv: setattr(attn, "head_dim", 256),
        lambda attn, q, kv: setattr(
            attn.kv_norm, "weight", _Tensor((256,), mx.bfloat16)
        ),
        lambda attn, q, kv: setattr(
            attn.kv_norm, "weight", _Tensor((512,), mx.float16)
        ),
        lambda attn, q, kv: setattr(attn.rope, "freqs", _Tensor((32,), mx.float32)),
        lambda attn, q, kv: setattr(attn.rope, "freqs", _Tensor((256,), mx.bfloat16)),
        lambda attn, q, kv: setattr(attn.config, "rms_norm_eps", 0.0),
    ),
)
def test_training_shape_dtype_weight_frequency_and_eps_rejections_fall_back(
    monkeypatch, dm, mutation
):
    attn, q_raw, kv_raw = _eligible_fixture(monkeypatch, dm)
    mutation(attn, q_raw, kv_raw)
    assert dm._attention_finalizer_native_inputs(attn, q_raw, kv_raw, 0) is None


@pytest.mark.parametrize("offset", (-1, 2**31, 1.5, mx.array(0)))
def test_non_scalar_or_out_of_range_offsets_fall_back(monkeypatch, dm, offset):
    attn, q_raw, kv_raw = _eligible_fixture(monkeypatch, dm)
    assert dm._attention_finalizer_native_inputs(attn, q_raw, kv_raw, offset) is None


def test_verification_and_partial_native_capability_fall_back(monkeypatch, dm):
    attn, q_raw, kv_raw = _eligible_fixture(monkeypatch, dm)
    monkeypatch.setattr(dm, "is_dspark_verify_armed", lambda: True)
    assert dm._attention_finalizer_native_inputs(attn, q_raw, kv_raw, 0) is None

    monkeypatch.setattr(dm, "is_dspark_verify_armed", lambda: False)
    monkeypatch.setattr(
        glm_fast, "has_symbol", lambda name: name == "ds4_q_head_rms_rope"
    )
    assert dm._attention_finalizer_native_inputs(attn, q_raw, kv_raw, 0) is None


@pytest.mark.parametrize("heads", (24, 40, 64))
def test_m6_verification_route_requires_its_independent_gate(
    monkeypatch, dm, heads
):
    attn, _q_raw, _kv_raw = _eligible_fixture(monkeypatch, dm, heads)
    q_raw = _Tensor((1, 6, heads, 512), mx.bfloat16)
    kv_raw = _Tensor((1, 6, 512), mx.bfloat16)
    monkeypatch.setattr(dm, "_DEEPSEEK_V4_ATTN_FINALIZER_PREFILL", False)
    monkeypatch.setattr(dm, "_DEEPSEEK_V4_ATTN_FINALIZER_VERIFY", True)
    monkeypatch.setattr(dm, "is_dspark_verify_armed", lambda: True)
    assert dm._attention_finalizer_native_inputs(attn, q_raw, kv_raw, 8192)

    monkeypatch.setattr(dm, "_DEEPSEEK_V4_ATTN_FINALIZER_VERIFY", False)
    assert dm._attention_finalizer_native_inputs(attn, q_raw, kv_raw, 8192) is None


def test_pair_preflight_completes_before_either_native_node(monkeypatch, dm, caplog):
    attn, q_raw, kv_raw = _eligible_fixture(monkeypatch, dm, heads=40)
    stock_calls = []
    native_calls = []
    q_result, kv_result = object(), object()
    monkeypatch.setattr(dm.mx, "contiguous", lambda value: value)
    monkeypatch.setattr(
        dm,
        "_stock_attention_qkv_finalizer",
        lambda *args: stock_calls.append(args),
    )
    monkeypatch.setattr(
        glm_fast,
        "ds4_q_head_rms_rope",
        lambda *args, **kwargs: native_calls.append(("q", args, kwargs)) or q_result,
    )
    monkeypatch.setattr(
        glm_fast,
        "ds4_kv_rms_rope",
        lambda *args, **kwargs: native_calls.append(("kv", args, kwargs)) or kv_result,
    )
    monkeypatch.setattr(dm, "_DEEPSEEK_V4_ATTN_FINALIZER_PREFILL_LOGGED", False)
    caplog.set_level(logging.INFO, logger=dm.__name__)

    result = dm._finalize_attention_qkv(attn, q_raw, kv_raw, 8192)

    assert result == (q_result, kv_result)
    assert stock_calls == []
    assert [call[0] for call in native_calls] == ["q", "kv"]
    assert native_calls[0][1] == (q_raw, attn.rope.freqs, 8192, 1e-6)
    assert native_calls[1][1] == (
        kv_raw,
        attn.kv_norm.weight,
        attn.rope.freqs,
        8192,
        1e-6,
    )
    records = [
        record
        for record in caplog.records
        if "RMSNorm+RoPE finalizers" in record.message
    ]
    assert len(records) == 1


def test_failed_pair_preflight_uses_stock_without_native_enqueue(monkeypatch, dm):
    attn, q_raw, kv_raw = _eligible_fixture(monkeypatch, dm)
    stock_result = (object(), object())
    calls = []
    monkeypatch.setattr(
        glm_fast, "ds4_q_head_rms_rope", lambda *args: calls.append(args)
    )
    monkeypatch.setattr(glm_fast, "ds4_kv_rms_rope", lambda *args: calls.append(args))
    monkeypatch.setattr(
        dm, "_stock_attention_qkv_finalizer", lambda *args: stock_result
    )
    monkeypatch.setattr(glm_fast, "is_native_available", lambda: False)

    result = dm._finalize_attention_qkv(attn, q_raw, kv_raw, 0)

    assert result is stock_result
    assert calls == []


def test_candidate_pair_preserves_stock_arrays_exactly(monkeypatch, dm):
    q = mx.array([[1.0, -2.0]], dtype=mx.bfloat16)
    kv = mx.array([[3.0, -4.0]], dtype=mx.bfloat16)
    q_expected, kv_expected = q + q, kv + kv
    native_inputs = (mx.ones((1,), dtype=mx.bfloat16), mx.ones((1,)), 1e-6)
    monkeypatch.setattr(
        dm, "_attention_finalizer_native_inputs", lambda *_args: native_inputs
    )
    monkeypatch.setattr(glm_fast, "ds4_q_head_rms_rope", lambda *_args: q + q)
    monkeypatch.setattr(glm_fast, "ds4_kv_rms_rope", lambda *_args: kv + kv)
    monkeypatch.setattr(dm, "_DEEPSEEK_V4_ATTN_FINALIZER_PREFILL_LOGGED", True)
    attn = SimpleNamespace()

    q_actual, kv_actual = dm._finalize_attention_qkv(attn, q, kv, 0)
    mx.eval(q_expected, kv_expected, q_actual, kv_actual)

    assert bool(mx.array_equal(q_expected, q_actual).item())
    assert bool(mx.array_equal(kv_expected, kv_actual).item())


def test_all_attention_classes_use_one_pair_seam_and_no_retry(dm):
    source = inspect.getsource(dm)
    assert source.count("_finalize_attention_qkv(self, q_raw, kv_raw, offset)") == 3

    helper = inspect.getsource(dm._finalize_attention_qkv)
    preflight = helper.index("_attention_finalizer_native_inputs")
    q_native = helper.index("glm_fast.ds4_q_head_rms_rope")
    kv_native = helper.index("glm_fast.ds4_kv_rms_rope")
    assert preflight < q_native < kv_native
    assert "except" not in helper[preflight:]


def test_cluster_hostfile_propagates_explicit_default_and_override(monkeypatch):
    monkeypatch.delenv("OMLX_DSV4_ATTN_FINALIZER_PREFILL", raising=False)
    assert "OMLX_DSV4_ATTN_FINALIZER_PREFILL=0" in deployment._hostfile_envs()

    monkeypatch.setenv("OMLX_DSV4_ATTN_FINALIZER_PREFILL", "1")
    envs = deployment._hostfile_envs()
    assert envs.count("OMLX_DSV4_ATTN_FINALIZER_PREFILL=1") == 1
    assert "OMLX_DSV4_ATTN_FINALIZER_PREFILL=0" not in envs

    monkeypatch.delenv("OMLX_DSV4_ATTN_FINALIZER_VERIFY", raising=False)
    assert "OMLX_DSV4_ATTN_FINALIZER_VERIFY=0" in deployment._hostfile_envs()
    monkeypatch.setenv("OMLX_DSV4_ATTN_FINALIZER_VERIFY", "1")
    assert "OMLX_DSV4_ATTN_FINALIZER_VERIFY=1" in deployment._hostfile_envs()
