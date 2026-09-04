# SPDX-License-Identifier: Apache-2.0
"""Tests for Ling's trained per-layer SwiGLU clamp.

Ling-3.0-flash ships ``expert_swiglu_limit_list`` and
``share_expert_swiglu_limit_list`` in config.json and is *trained* with
those clamps. Without them the late layers run unclamped; measured on
Ling-3.0-flash that costs 17 points of HumanEval (88.41% -> 71.34%).
"""

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.bailing_hybrid import apply_bailing_hybrid_patch  # noqa: E402
from omlx.patches.bailing_hybrid.swiglu_clamp import (  # noqa: E402
    bind_limits,
    layer_swiglu_limit,
)

# The live module may be oMLX's vendored copy or an mlx-lm build that already
# ships bailing_hybrid; apply() resolves whichever and installs the clamp on
# it. Its return value only says which, so it is not a skip condition.
apply_bailing_hybrid_patch()

bh = pytest.importorskip("mlx_lm.models.bailing_hybrid")

HID = 128
N_LAYERS = 6

CFG = dict(
    model_type="bailing_hybrid",
    vocab_size=256,
    hidden_size=HID,
    intermediate_size=256,
    moe_intermediate_size=64,
    num_hidden_layers=N_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=4,
    num_experts=8,
    num_experts_per_tok=2,
    num_shared_experts=1,
    n_group=1,
    topk_group=1,
    first_k_dense_replace=1,
    layer_group_size=6,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    routed_scaling_factor=1.0,
    head_dim=32,
    kv_lora_rank=32,
    qk_rope_head_dim=16,
    qk_nope_head_dim=32,
    v_head_dim=32,
    moe_shared_expert_intermediate_size=64,
    # Required by the vendored ModelArgs; CI exercises that path (the
    # bundled mlx-lm build supplies its own bailing_hybrid locally).
    group_norm_size=1,
)


def _shared_limit(mlp):
    """Limit as stored by either build.

    The vendored copy takes ``swiglu_limit`` as a constructor argument; the
    installed path tags the module with a private attribute instead.
    """
    for name in ("swiglu_limit", "_omlx_swiglu_limit"):
        value = getattr(mlp, name, None)
        if value:
            return value
    return None


def _with_limits():
    cfg = dict(CFG)
    cfg["expert_swiglu_limit_list"] = [0] * (N_LAYERS - 2) + [4, 4]
    cfg["share_expert_swiglu_limit_list"] = [0] * (N_LAYERS - 2) + [5, 7]
    return cfg


class TestLimitResolution:
    def test_absent_or_zero_is_none(self):
        assert layer_swiglu_limit(None, 0) is None
        assert layer_swiglu_limit([], 0) is None
        assert layer_swiglu_limit([0, 0], 1) is None
        # index past the end must not raise
        assert layer_swiglu_limit([4], 5) is None

    def test_nonzero_returns_float(self):
        assert layer_swiglu_limit([0, 4], 1) == 4.0


class TestClampMath:
    def test_matches_reference_formula(self):
        import mlx.nn as nn

        limit = 4.0
        gate = mx.array([[-20.0, -1.0, 0.0, 1.0, 20.0]])
        up = mx.array([[-30.0, -2.0, 0.5, 2.0, 30.0]])
        got = bh.clamped_swiglu(gate, up, limit)
        want = mx.minimum(nn.silu(gate), limit) * mx.clip(up, -limit, limit)
        assert bool(mx.allclose(got, want, atol=1e-6).item())

    def test_clamp_actually_binds(self):
        # unclamped silu(20) * 30 would be ~600
        got = bh.clamped_swiglu(
            mx.array([[20.0]]), mx.array([[30.0]]), 4.0
        )
        assert float(got.item()) <= 4.0 * 4.0 + 1e-4

    def test_switchglu_signature_order(self):
        """SwitchGLU calls activation(x_up, x_gate) — silu must hit gate."""
        act = bh.ClampedSwiGLU(4.0)
        up, gate = mx.array([[3.0]]), mx.array([[-10.0]])
        # silu(-10)*3 ~ -1.4e-3. Applying silu to `up` instead would give
        # silu(3) * clip(-10) ~ -11, so this discriminates the two orders.
        assert abs(float(act(up, gate).item())) < 0.01


class TestWiring:
    def test_limits_bind_to_late_layers_only(self):
        model = bh.Model(bh.ModelArgs.from_dict(_with_limits()))
        routed, shared = [], []
        for idx, layer in enumerate(model.model.layers):
            sm = getattr(layer.mlp, "switch_mlp", None)
            if sm is not None and isinstance(sm.activation, bh.ClampedSwiGLU):
                routed.append((idx, sm.activation.limit))
            se = getattr(layer.mlp, "shared_experts", None)
            if se is not None and _shared_limit(se):
                shared.append((idx, _shared_limit(se)))
        assert routed == [(N_LAYERS - 2, 4.0), (N_LAYERS - 1, 4.0)]
        assert shared == [(N_LAYERS - 2, 5.0), (N_LAYERS - 1, 7.0)]

    def test_dense_layers_ignore_routed_expert_limits(self):
        cfg = dict(CFG)
        cfg["first_k_dense_replace"] = 2
        cfg["expert_swiglu_limit_list"] = [4, 4] + [0] * (N_LAYERS - 2)
        model = bh.Model(bh.ModelArgs.from_dict(cfg))

        for layer in model.model.layers[:2]:
            assert getattr(layer.mlp, "switch_mlp", None) is None
            assert getattr(layer.mlp, "swiglu_limit", None) is None

    def test_installed_path_ignores_dense_layer_limits(self):
        from types import SimpleNamespace

        dense = SimpleNamespace()
        routed = SimpleNamespace(activation=None)
        shared = SimpleNamespace()
        moe = SimpleNamespace(switch_mlp=routed, shared_experts=shared)
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(mlp=dense), SimpleNamespace(mlp=moe)]
            )
        )
        config = SimpleNamespace(
            expert_swiglu_limit_list=[4, 4],
            share_expert_swiglu_limit_list=[5, 5],
        )
        module = SimpleNamespace(
            ClampedSwiGLU=lambda limit: SimpleNamespace(limit=limit)
        )

        assert bind_limits(module, model, config) == 2
        assert not hasattr(dense, "_omlx_swiglu_limit")
        assert routed.activation.limit == 4.0
        assert shared._omlx_swiglu_limit == 5.0

    def test_without_limits_model_is_unclamped(self):
        model = bh.Model(bh.ModelArgs.from_dict(dict(CFG)))
        for layer in model.model.layers:
            sm = getattr(layer.mlp, "switch_mlp", None)
            if sm is not None:
                assert not isinstance(sm.activation, bh.ClampedSwiGLU)
            se = getattr(layer.mlp, "shared_experts", None)
            if se is not None:
                assert _shared_limit(se) is None

    def test_forward_is_finite_with_limits(self):
        from mlx.utils import tree_flatten

        model = bh.Model(bh.ModelArgs.from_dict(_with_limits()))
        weights = {
            k: mx.random.normal(v.shape).astype(v.dtype) * 0.02
            for k, v in dict(tree_flatten(model.parameters())).items()
        }
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        out = model(mx.array([[3, 15, 42, 7]]), cache=model.make_cache())
        assert bool(mx.all(mx.isfinite(out)).item())
