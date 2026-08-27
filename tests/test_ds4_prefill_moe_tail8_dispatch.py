"""Strict default-off dispatch gates for the exact DS4 M3 tail8 path."""

import inspect
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cluster import deployment
from omlx.patches.deepseek_v4 import switch_layers as sl


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Projection:
    def __init__(self, weight_shape, scale_shape):
        self.values = {
            "weight": _Tensor(weight_shape, mx.uint32),
            "scales": _Tensor(scale_shape, mx.uint8),
        }

    def __getitem__(self, key):
        return self.values[key]


def _eligible(monkeypatch, **overrides):
    monkeypatch.setattr(sl, "_DEEPSEEK_MXFP4_TAIL8", overrides.pop("enabled", True))
    monkeypatch.setattr(
        sl,
        "_DEEPSEEK_MXFP4_TAIL8_EQUAL_TP",
        overrides.pop("equal_enabled", False),
    )
    combined = overrides.pop("combined", False)
    monkeypatch.setattr(sl, "_DEEPSEEK_MXFP4_COMBINED", combined)
    verify = overrides.pop("verify", False)
    monkeypatch.setattr(sl, "is_dspark_verify_armed", lambda: verify)
    nax = overrides.pop("nax", False)
    monkeypatch.setattr(sl, "is_nax_available", lambda: nax)
    missing = overrides.pop("missing_symbol", None)
    monkeypatch.setattr(sl.glm_fast, "has_symbol", lambda name: name != missing)
    monkeypatch.setattr(sl, "QuantizedSwitchLinear", _Projection)

    intermediate = 768 if combined else 1024
    layer = SimpleNamespace(
        training=overrides.pop("training", False),
        _omlx_dsv4f_exact_config=overrides.pop("fingerprint", combined),
        _omlx_dsv4f_moe_tp=overrides.pop("tp", (2, 0, (3, 5))),
        up_proj=_Projection(
            (256, intermediate, 512), (256, intermediate, 128)
        ),
        gate_proj=_Projection(
            (256, intermediate, 512), (256, intermediate, 128)
        ),
        down_proj=_Projection(
            (256, 4096, intermediate // 8),
            (256, 4096, intermediate // 32),
        ),
        activation=SimpleNamespace(limit=10.0, fp32=False),
    )
    device = overrides.pop("device", "Apple M3 Ultra")
    monkeypatch.setattr(sl.mx, "device_info", lambda: {"device_name": device})
    block_plan = (
        _Tensor((448, 3), mx.int32),
        _Tensor((1,), mx.int32),
        overrides.pop("variant", 2),
    )
    request_shape = overrides.pop("request_shape", (1, 1024, 4096))
    indices = _Tensor(overrides.pop("indices_shape", (1, 1024, 6)), mx.uint32)
    x_sorted = _Tensor(
        overrides.pop("sorted_shape", (6144, 1, 4096)), mx.float16
    )
    assert not overrides
    return sl.SwitchGLU._can_use_mxfp4_tail8_prefill(
        layer,
        request_shape,
        indices,
        x_sorted,
        mx.bfloat16,
        ("mxfp4", "mxfp4", "mxfp4"),
        True,
        block_plan,
    )


def test_combined_switch_accepts_only_exact_m3_rank0_three_five_slice(monkeypatch):
    assert _eligible(monkeypatch, combined=True, enabled=False)
    assert not _eligible(
        monkeypatch, combined=True, enabled=False, tp=(2, 1, (3, 5))
    )
    assert not _eligible(
        monkeypatch, combined=True, enabled=False, fingerprint=False
    )
    assert not _eligible(
        monkeypatch, combined=True, enabled=False, verify=True
    )


def test_exact_m1024_m3_shape_is_eligible_only_when_explicitly_enabled(monkeypatch):
    assert _eligible(monkeypatch)
    assert not _eligible(monkeypatch, enabled=False)


def test_equal_tp2_m3_shape_has_independent_default_gate(monkeypatch):
    assert _eligible(
        monkeypatch,
        enabled=False,
        equal_enabled=True,
        tp=(2, 0, ()),
    )
    assert not _eligible(
        monkeypatch,
        enabled=False,
        equal_enabled=True,
        tp=(2, 0, ()),
        device="Apple M5 Max",
    )
    assert not _eligible(
        monkeypatch,
        enabled=False,
        equal_enabled=False,
        tp=(2, 0, ()),
    )


@pytest.mark.parametrize(
    "override",
    (
        {"training": True},
        {"nax": True},
        {"request_shape": (1, 1, 4096), "indices_shape": (1, 1, 6)},
        {"request_shape": (2, 512, 4096), "indices_shape": (2, 512, 6)},
        {"variant": 1},
        {"missing_symbol": "deepseek_mxfp4_gather_qmm_blocks_tail8"},
        {
            "missing_symbol": (
                "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8"
            )
        },
    ),
)
def test_decode_nax_batch_and_partial_native_availability_fall_back(
    monkeypatch, override
):
    assert not _eligible(monkeypatch, **override)


def test_dispatch_checks_complete_contract_before_enqueuing_either_kernel():
    source = inspect.getsource(sl.SwitchGLU.__call__)
    eligibility = source.index("use_tail8_prefill =")
    pair_call = source.index("gather_qmm_pair_swiglu_blocks_tail8")
    down_call = source.index("gather_qmm_blocks_tail8")
    assert eligibility < pair_call < down_call
    assert "except" not in source[eligibility:down_call]


def test_cluster_hostfile_carries_explicit_default_off_value(monkeypatch):
    monkeypatch.delenv("OMLX_DSV4_MOE_TAIL8", raising=False)
    assert "OMLX_DSV4_MOE_TAIL8=0" in deployment._hostfile_envs()
    monkeypatch.setenv("OMLX_DSV4_MOE_TAIL8", "1")
    assert "OMLX_DSV4_MOE_TAIL8=1" in deployment._hostfile_envs()
    monkeypatch.delenv("OMLX_DSV4_MOE_TAIL8_EQUAL_TP", raising=False)
    assert "OMLX_DSV4_MOE_TAIL8_EQUAL_TP=1" in deployment._hostfile_envs()
    monkeypatch.setenv("OMLX_DSV4_MOE_TAIL8_EQUAL_TP", "0")
    assert "OMLX_DSV4_MOE_TAIL8_EQUAL_TP=0" in deployment._hostfile_envs()
    monkeypatch.delenv("OMLX_DSV4_COMBINED_MOE_PREFILL", raising=False)
    assert "OMLX_DSV4_COMBINED_MOE_PREFILL=0" in deployment._hostfile_envs()
    monkeypatch.setenv("OMLX_DSV4_COMBINED_MOE_PREFILL", "1")
    assert "OMLX_DSV4_COMBINED_MOE_PREFILL=1" in deployment._hostfile_envs()
