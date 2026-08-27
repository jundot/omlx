"""Strict runtime gates for the exact DS4F M5 rank-1 NAX MoE path."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.cluster import deployment
from omlx.patches.deepseek_v4 import switch_layers as sl


SYMBOL = "deepseek_mxfp4_gather_qmm_blocks_nax"


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _Projection:
    group_size = 32
    bits = 4
    mode = "mxfp4"

    def __init__(self, weight_shape, scale_shape):
        self.values = {
            "weight": _Tensor(weight_shape, mx.uint32),
            "scales": _Tensor(scale_shape, mx.uint8),
        }
        self.num_experts = weight_shape[0]

    def __getitem__(self, key):
        return self.values[key]

    def __contains__(self, key):
        return key in self.values

    def get(self, key, default=None):
        return self.values.get(key, default)


def _eligible(monkeypatch, **overrides):
    monkeypatch.setattr(
        sl, "_DEEPSEEK_MXFP4_NAX_BLOCKS", overrides.pop("enabled", True)
    )
    monkeypatch.setattr(
        sl,
        "_DEEPSEEK_MXFP4_COMBINED",
        overrides.pop("combined", False),
    )
    verify = overrides.pop("verify", False)
    monkeypatch.setattr(sl, "is_dspark_verify_armed", lambda: verify)
    nax_available = overrides.pop("nax_available", True)
    monkeypatch.setattr(
        sl,
        "is_nax_available",
        lambda: nax_available,
    )
    device_name = overrides.pop("device_name", "Apple M5 Max")
    monkeypatch.setattr(sl.mx, "device_info", lambda: {"device_name": device_name})
    missing_symbol = overrides.pop("missing_symbol", None)
    monkeypatch.setattr(
        sl.glm_fast, "has_symbol", lambda name: name != missing_symbol
    )
    artifact = overrides.pop("artifact", True)
    monkeypatch.setattr(
        sl.glm_fast,
        "ds4_projection_nax_kernels_built",
        lambda: artifact,
    )
    artifact_device = overrides.pop("artifact_device", True)
    monkeypatch.setattr(
        sl.glm_fast,
        "ds4_projection_nax_device_available",
        lambda: artifact_device,
    )
    monkeypatch.setattr(sl, "QuantizedSwitchLinear", _Projection)

    up_weight_shape = overrides.pop("up_weight_shape", (256, 1280, 512))
    up_scale_shape = overrides.pop("up_scale_shape", (256, 1280, 128))
    layer = SimpleNamespace(
        training=overrides.pop("training", False),
        _omlx_dsv4f_exact_config=overrides.pop("fingerprint", True),
        _omlx_dsv4f_moe_tp=overrides.pop("tp", (2, 1, (3, 5))),
        up_proj=_Projection(up_weight_shape, up_scale_shape),
        gate_proj=_Projection(up_weight_shape, up_scale_shape),
        down_proj=_Projection(
            overrides.pop("down_weight_shape", (256, 4096, 160)),
            overrides.pop("down_scale_shape", (256, 4096, 40)),
        ),
        activation=SimpleNamespace(
            limit=overrides.pop("activation_limit", 10.0),
            fp32=overrides.pop("activation_fp32", False),
        ),
    )
    indices = _Tensor(
        overrides.pop("indices_shape", (1, 1024, 6)),
        overrides.pop("indices_dtype", mx.uint32),
    )
    scores = _Tensor(
        overrides.pop("scores_shape", (1, 1024, 6)),
        overrides.pop("scores_dtype", mx.float32),
    )
    x_sorted = _Tensor(
        overrides.pop("sorted_shape", (6144, 1, 4096)),
        overrides.pop("sorted_dtype", mx.bfloat16),
    )
    request_shape = overrides.pop("request_shape", (1, 1024, 4096))
    original_dtype = overrides.pop("original_dtype", mx.bfloat16)
    assert not overrides
    return sl.SwitchGLU._can_use_mxfp4_nax_blocks_prefill(
        layer,
        request_shape,
        indices,
        scores,
        x_sorted,
        original_dtype,
    )


def test_only_exact_enabled_m5_rank1_3x5_shape_is_eligible(monkeypatch):
    assert _eligible(monkeypatch)


def test_exact_enabled_m5_rank1_equal_shape_is_eligible(monkeypatch):
    assert _eligible(
        monkeypatch,
        tp=(2, 1, (4, 4)),
        up_weight_shape=(256, 1024, 512),
        up_scale_shape=(256, 1024, 128),
        down_weight_shape=(256, 4096, 128),
        down_scale_shape=(256, 4096, 32),
    )


def test_runtime_empty_equal_vector_and_int32_indices_are_eligible(monkeypatch):
    assert _eligible(
        monkeypatch,
        tp=(2, 1, ()),
        indices_dtype=mx.int32,
        up_weight_shape=(256, 1024, 512),
        up_scale_shape=(256, 1024, 128),
        down_weight_shape=(256, 4096, 128),
        down_scale_shape=(256, 4096, 32),
    )


def test_combined_flag_does_not_enable_equal_shape(monkeypatch):
    assert not _eligible(
        monkeypatch,
        enabled=False,
        combined=True,
        tp=(2, 1, (4, 4)),
        up_weight_shape=(256, 1024, 512),
        up_scale_shape=(256, 1024, 128),
        down_weight_shape=(256, 4096, 128),
        down_scale_shape=(256, 4096, 32),
    )


def test_combined_switch_selects_the_same_exact_m5_rank1_contract(monkeypatch):
    assert _eligible(monkeypatch, enabled=False, combined=True)
    assert not _eligible(
        monkeypatch, enabled=False, combined=True, tp=(2, 0, (3, 5))
    )


@pytest.mark.parametrize(
    "override",
    (
        {"enabled": False},
        {"training": True},
        {"verify": True},
        {"fingerprint": False},
        {"tp": (2, 0, (3, 5))},
        {"tp": (2, 1, (4, 4))},
        {"device_name": "Apple M3 Ultra"},
        {"nax_available": False},
        {"artifact": False},
        {"artifact_device": False},
        {"missing_symbol": SYMBOL},
        {"request_shape": (1, 512, 4096), "indices_shape": (1, 512, 6)},
        {"sorted_shape": (6144, 1, 4096), "sorted_dtype": mx.float16},
        {"original_dtype": mx.float16},
        {"scores_shape": (1, 1024, 1)},
        {"indices_dtype": mx.int16},
        {"scores_dtype": mx.bfloat16},
        {"activation_limit": 0.0},
        {"activation_fp32": True},
        {"up_weight_shape": (256, 1024, 512)},
        {"down_weight_shape": (256, 4096, 128)},
    ),
)
def test_every_unqualified_pairing_falls_back_before_candidate_graph(
    monkeypatch, override
):
    assert not _eligible(monkeypatch, **override)


def test_preflight_precedes_block_plan_and_all_candidate_invocations():
    source = inspect.getsource(sl.SwitchGLU.__call__)
    preflight = source.index("use_nax_blocks_prefill =")
    legacy_cast = source.index(
        "if use_f16_moe and not use_nax_blocks_prefill:", preflight
    )
    plan = source.index("nax_block_plan = _build_mxfp4_blocks", preflight)
    pair_call = source.index(f"glm_fast.{SYMBOL}", plan)
    down_call = source.rindex(f"glm_fast.{SYMBOL}")
    assert preflight < legacy_cast < plan < pair_call < down_call
    assert "except" not in source[preflight:down_call]


def test_combined_m5_path_uses_separate_nax_projections_not_fused_pair():
    source = inspect.getsource(sl.SwitchGLU.__call__)
    assert source.count(f"glm_fast.{SYMBOL}") == 3
    assert "deepseek_mxfp4_gather_qmm_pair_blocks_nax" not in source


def test_exact_config_and_tp_fingerprints_are_attached_by_model_and_shard():
    source = (
        Path(__file__).parents[1]
        / "omlx/patches/deepseek_v4/deepseek_v4_model.py"
    ).read_text()
    assert "_omlx_dsv4f_exact_config = _dsv4f_exact_config(" in source
    assert "config, 4" in source
    assert "_omlx_dsv4f_exact_config = False" in source
    assert "_omlx_dsv4f_moe_tp" in source
    assert "tuple(moe_shard_weights or ())" in source


def test_bm32_plan_covers_zero_ragged_full_and_multi_block_router_counts():
    # A realistic fixed-6144 distribution with one inactive expert, exact
    # boundary cases, one 33-row expert, and the remaining experts at 24/25.
    counts = [0, 1, 31, 32, 33] + [25] * 23 + [24] * (256 - 28)
    assert sum(counts) == 6144
    assert {0, 1, 31, 32, 33}.issubset(counts)
    sorted_indices = mx.array(
        [expert for expert, count in enumerate(counts) for _ in range(count)],
        dtype=mx.uint32,
    )
    block_meta, block_count = sl._build_mxfp4_blocks(sorted_indices, 256, 32)
    mx.eval(block_meta, block_count)
    count = int(block_count.item())
    rows = np.asarray(block_meta[:count])
    by_expert: dict[int, list[int]] = {}
    for _start, expert, block_rows in rows.tolist():
        by_expert.setdefault(int(expert), []).append(int(block_rows))

    assert block_meta.shape == (448, 3)
    assert count == 256
    assert 0 not in by_expert
    assert by_expert[1] == [1]
    assert by_expert[2] == [31]
    assert by_expert[3] == [32]
    assert sorted(by_expert[4]) == [1, 32]
    assert all(
        1 <= block_rows <= 32
        for values in by_expert.values()
        for block_rows in values
    )


def test_cluster_hostfile_carries_explicit_default_off_value(monkeypatch):
    monkeypatch.delenv("OMLX_DSV4_NAX_MOE_BLOCKS", raising=False)
    assert "OMLX_DSV4_NAX_MOE_BLOCKS=0" in deployment._hostfile_envs()
    monkeypatch.setenv("OMLX_DSV4_NAX_MOE_BLOCKS", "1")
    assert "OMLX_DSV4_NAX_MOE_BLOCKS=1" in deployment._hostfile_envs()
