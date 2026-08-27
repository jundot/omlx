"""Strict production gates for the exact DS4 output projection chain."""

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

apply_deepseek_v4_patch()
dm = sys.modules["mlx_lm.models.deepseek_v4"]


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Projection:
    group_size = 32
    bits = 8
    mode = "mxfp8"

    def __init__(self, weight_shape, scale_shape):
        self.values = {
            "weight": _Tensor(weight_shape, mx.uint32),
            "scales": _Tensor(scale_shape, mx.uint8),
        }

    def get(self, key, default=None):
        return self.values.get(key, default)


def _eligible(monkeypatch, **overrides):
    monkeypatch.setattr(
        dm, "_DEEPSEEK_V4_OUTPUT_CHAIN_PREFILL", overrides.pop("enabled", True)
    )
    monkeypatch.setattr(
        dm,
        "_DEEPSEEK_V4_OUTPUT_CHAIN_EQUAL_TP",
        overrides.pop("equal_enabled", True),
    )
    verify = overrides.pop("verify", False)
    monkeypatch.setattr(
        dm,
        "is_dspark_verify_armed",
        lambda: verify,
    )
    fingerprint = overrides.pop("fingerprint", True)
    missing_symbol = overrides.pop("missing_symbol", None)
    monkeypatch.setattr(dm, "_dsv4f_exact_config", lambda *_args: fingerprint)
    monkeypatch.setattr(glm_fast, "is_native_available", lambda: True)
    monkeypatch.setattr(
        glm_fast,
        "has_symbol",
        lambda name: name != missing_symbol,
    )
    device = overrides.pop("device", "Apple M3 Ultra")
    monkeypatch.setattr(dm.mx, "device_info", lambda: {"device_name": device})
    k = overrides.pop("k", 1536)
    o_a = _Projection(
        overrides.pop("o_a_weight", (8, 1024, k // 4)),
        overrides.pop("o_a_scales", (8, 1024, k // 32)),
    )
    o_b = _Projection(
        overrides.pop("o_b_weight", (4096, 2048)),
        overrides.pop("o_b_scales", (4096, 256)),
    )
    o_a.bits = overrides.pop("o_a_bits", 8)
    attn = SimpleNamespace(
        training=overrides.pop("training", False),
        config=object(),
        wo_a=o_a,
        wo_b=o_b,
    )
    prepared = _Tensor(
        overrides.pop("shape", (1, 8, 1024, k)),
        overrides.pop("dtype", mx.bfloat16),
    )
    assert not overrides
    return dm._attention_output_chain_native_inputs(attn, prepared)


@pytest.mark.parametrize("k", (1536, 2048, 2560, 4096))
def test_signed_tp_and_single_shapes_are_eligible(monkeypatch, k):
    assert _eligible(monkeypatch, k=k) is not None


def test_single_m3_m2048_shape_is_eligible(monkeypatch):
    assert _eligible(monkeypatch, k=4096, shape=(1, 8, 2048, 4096)) is not None


def test_equal_k2048_chain_is_m3_only(monkeypatch):
    assert _eligible(monkeypatch, k=2048, device="Apple M5 Max") is None
    assert _eligible(monkeypatch, k=2048) is not None


def test_equal_k2048_has_independent_default_gate(monkeypatch):
    assert _eligible(monkeypatch, k=2048, enabled=False) is not None
    assert _eligible(monkeypatch, k=2048, equal_enabled=False) is None


@pytest.mark.parametrize(
    "override",
    (
        {"enabled": False},
        {"training": True},
        {"verify": True},
        {"fingerprint": False},
        {"shape": (1, 8, 512, 1536)},
        {"dtype": mx.float16},
        {"o_a_bits": 4},
        {"o_a_weight": (8, 1024, 512)},
        {"o_b_weight": (2048, 2048)},
        {"missing_symbol": "ds4_output_projection_chain"},
    ),
)
def test_every_unqualified_contract_falls_back_before_graph(monkeypatch, override):
    assert _eligible(monkeypatch, **override) is None


def test_chain_preflight_precedes_stock_projection_graph():
    source = inspect.getsource(dm._project_attention_output)
    preflight = source.index("_project_attention_output_chain")
    native_return = source.index("return native_chain", preflight)
    stock_graph = source.index("_project_attention_oa", native_return)
    assert preflight < native_return < stock_graph


def test_production_chain_uses_confirmed_bm64_bk32_bn64_variant():
    source = inspect.getsource(dm._project_attention_output_chain)
    assert "variant=1" in source
