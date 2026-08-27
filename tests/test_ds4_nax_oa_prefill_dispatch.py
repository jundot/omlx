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


class _Projection:
    def __init__(self):
        self.group_size = 32
        self.bits = 8
        self.mode = "mxfp8"
        self.values = {
            "weight": _Tensor((8, 1024, 640), mx.uint32),
            "scales": _Tensor((8, 1024, 80), mx.uint8),
            "biases": None,
        }
        self.calls = []
        self.result = object()

    def get(self, key, default=None):
        return self.values.get(key, default)

    def __getitem__(self, key):
        return self.values[key]

    def __call__(self, value):
        self.calls.append(value)
        return self.result


def _eligible_fixture(monkeypatch, dm):
    monkeypatch.setattr(dm, "_DEEPSEEK_V4_NAX_OA_PREFILL", True)
    monkeypatch.setattr(glm_fast, "is_native_available", lambda: True)
    monkeypatch.setattr(glm_fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(glm_fast, "ds4_projection_nax_kernels_built", lambda: True)
    monkeypatch.setattr(glm_fast, "ds4_projection_nax_device_available", lambda: True)
    projection = _Projection()
    attn = SimpleNamespace(training=False, wo_a=projection)
    prepared = _Tensor((1, 8, 1024, 2560), mx.bfloat16)
    return attn, projection, prepared


def test_exact_confirmed_m5_shape_is_default_off(monkeypatch, dm):
    attn, _projection, prepared = _eligible_fixture(monkeypatch, dm)
    assert dm._can_use_nax_oa_prefill(attn, prepared)

    monkeypatch.setattr(dm, "_DEEPSEEK_V4_NAX_OA_PREFILL", False)
    assert not dm._can_use_nax_oa_prefill(attn, prepared)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda attn, projection, prepared: setattr(attn, "training", True),
        lambda attn, projection, prepared: setattr(
            prepared, "shape", (1, 8, 1024, 1536)
        ),
        lambda attn, projection, prepared: setattr(prepared, "dtype", mx.float16),
        lambda attn, projection, prepared: setattr(projection, "group_size", 64),
        lambda attn, projection, prepared: setattr(projection, "bits", 4),
        lambda attn, projection, prepared: setattr(projection, "mode", "affine"),
        lambda attn, projection, prepared: projection.values.__setitem__(
            "biases", _Tensor((1024,), mx.bfloat16)
        ),
        lambda attn, projection, prepared: projection.values.__setitem__(
            "weight", _Tensor((8, 1024, 320), mx.uint32)
        ),
        lambda attn, projection, prepared: projection.values.__setitem__(
            "scales", _Tensor((8, 1024, 40), mx.uint8)
        ),
    ),
)
def test_training_rank0_dtype_and_weight_contracts_fall_back(monkeypatch, dm, mutation):
    attn, projection, prepared = _eligible_fixture(monkeypatch, dm)
    mutation(attn, projection, prepared)
    assert not dm._can_use_nax_oa_prefill(attn, prepared)


@pytest.mark.parametrize(
    ("capability", "value"),
    (
        ("is_native_available", False),
        ("has_symbol", False),
        ("ds4_projection_nax_kernels_built", False),
        ("ds4_projection_nax_device_available", False),
    ),
)
def test_missing_extension_symbol_artifact_or_m5_capability_falls_back(
    monkeypatch, dm, capability, value
):
    attn, projection, prepared = _eligible_fixture(monkeypatch, dm)
    if capability == "has_symbol":
        monkeypatch.setattr(glm_fast, capability, lambda name: value)
    else:
        monkeypatch.setattr(glm_fast, capability, lambda: value)

    result = dm._project_attention_oa(attn, prepared)

    assert result is projection.result
    assert projection.calls == [prepared]


def test_dispatch_uses_confirmed_nax_variant_and_logs_once(monkeypatch, dm, caplog):
    attn, projection, prepared = _eligible_fixture(monkeypatch, dm)
    native_result = object()
    calls = []
    monkeypatch.setattr(dm.mx, "contiguous", lambda value: value)

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return native_result

    monkeypatch.setattr(glm_fast, "ds4_projection_mxfp8_qmm", native)
    monkeypatch.setattr(dm, "_DEEPSEEK_V4_NAX_OA_PREFILL_LOGGED", False)
    caplog.set_level(logging.INFO, logger=dm.__name__)

    first = dm._project_attention_oa(attn, prepared)
    second = dm._project_attention_oa(attn, prepared)

    assert first is native_result and second is native_result
    assert projection.calls == []
    assert len(calls) == 2
    args, kwargs = calls[0]
    assert args == (
        prepared,
        projection["weight"],
        projection["scales"],
    )
    assert kwargs == {"variant": 0, "use_nax": True, "nax_variant": 0}
    records = [
        record for record in caplog.records if "NAX O-A prefill tile" in record.message
    ]
    assert len(records) == 1


def test_candidate_preserves_the_stock_projection_array_exactly(monkeypatch, dm):
    x = mx.array([[1.0, -2.0, 3.0]], dtype=mx.bfloat16)

    class ExactProjection(_Projection):
        def __call__(self, value):
            return value + value

    projection = ExactProjection()
    attn = SimpleNamespace(training=False, wo_a=projection)
    monkeypatch.setattr(dm, "_can_use_nax_oa_prefill", lambda *_args: True)
    monkeypatch.setattr(
        glm_fast,
        "ds4_projection_mxfp8_qmm",
        lambda value, *_args, **_kwargs: projection(value),
    )

    reference = projection(x)
    candidate = dm._project_attention_oa(attn, x)
    mx.eval(reference, candidate)

    assert bool(mx.array_equal(reference, candidate).item())


def test_verify_branch_precedes_prefill_native_dispatch_and_has_no_retry(dm):
    output_source = inspect.getsource(dm._project_attention_output)
    assert output_source.index("if is_dspark_verify_armed()") < output_source.index(
        "_project_attention_oa"
    )

    helper_source = inspect.getsource(dm._project_attention_oa)
    eligibility = helper_source.index("if not _can_use_nax_oa_prefill")
    native = helper_source.index("glm_fast.ds4_projection_mxfp8_qmm")
    assert eligibility < native
    assert "except" not in helper_source[eligibility:native]


def test_cluster_hostfile_propagates_default_off_and_operator_override(monkeypatch):
    monkeypatch.delenv("OMLX_DSV4_NAX_OA_PREFILL", raising=False)
    assert "OMLX_DSV4_NAX_OA_PREFILL=0" in deployment._hostfile_envs()

    monkeypatch.setenv("OMLX_DSV4_NAX_OA_PREFILL", "1")
    envs = deployment._hostfile_envs()
    assert envs.count("OMLX_DSV4_NAX_OA_PREFILL=1") == 1
    assert "OMLX_DSV4_NAX_OA_PREFILL=0" not in envs
