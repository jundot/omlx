from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from omlx.patches.deepseek_v41_dfly2.dfly2_layer import _linear, _moe, _swiglu


def _slow_moe(x, weights, config):
    router = _linear(x.astype(mx.float32), weights["ffn.gate.weight"].astype(mx.float32)) + weights["ffn.gate.bias"].astype(mx.float32)
    scores = mx.sqrt(mx.logaddexp(mx.zeros_like(router), router))
    picked = mx.argsort(scores + weights["ffn.gate.bias_vl"].astype(mx.float32), axis=-1)[..., -config.num_experts_per_tok :]
    picked_scores = mx.take_along_axis(scores, picked, axis=-1)
    picked_scores = picked_scores / mx.sum(picked_scores, axis=-1, keepdims=True)

    out = mx.zeros_like(x)
    for expert in range(config.num_experts):
        gate = _linear(x, weights[f"ffn.experts.{expert}.w1.weight"])
        up = _linear(x, weights[f"ffn.experts.{expert}.w3.weight"])
        value = _linear(_swiglu(gate, up, config.swiglu_limit), weights[f"ffn.experts.{expert}.w2.weight"])
        selected = mx.sum((picked == expert).astype(x.dtype) * picked_scores.astype(x.dtype), axis=-1, keepdims=True)
        out = out + selected * value

    shared_gate = _linear(x, weights["ffn.shared_experts.w1.weight"])
    shared_up = _linear(x, weights["ffn.shared_experts.w3.weight"])
    shared = _linear(_swiglu(shared_gate, shared_up, config.swiglu_limit), weights["ffn.shared_experts.w2.weight"])
    return out * config.router_scaling_factor + shared, picked


def _weights(config):
    rng = np.random.default_rng(41)

    def sample(*shape):
        return mx.array(rng.standard_normal(shape, dtype=np.float32) * np.float32(0.05))

    weights = {
        "ffn.gate.weight": sample(config.num_experts, config.dim),
        "ffn.gate.bias": sample(config.num_experts),
        # Restrict all routes to experts 0 and 1, leaving six experts empty.
        "ffn.gate.bias_vl": mx.array([3.0, 2.0] + [-3.0] * (config.num_experts - 2)),
    }
    for expert in range(config.num_experts):
        weights.update(
            {
                f"ffn.experts.{expert}.w1.weight": sample(config.intermediate_size, config.dim),
                f"ffn.experts.{expert}.w2.weight": sample(config.dim, config.intermediate_size),
                f"ffn.experts.{expert}.w3.weight": sample(config.intermediate_size, config.dim),
            }
        )
    weights.update(
        {
            "ffn.shared_experts.w1.weight": sample(config.intermediate_size, config.dim),
            "ffn.shared_experts.w2.weight": sample(config.dim, config.intermediate_size),
            "ffn.shared_experts.w3.weight": sample(config.intermediate_size, config.dim),
        }
    )
    return weights, rng


def test_moe_dispatch_matches_all_experts_reference_with_empty_experts():
    config = SimpleNamespace(
        dim=16,
        intermediate_size=12,
        num_experts=8,
        num_experts_per_tok=2,
        route_norm=True,
        router_scaling_factor=1.5,
        moe_router_score_func="sqrtsoftplus",
        swiglu_limit=0.25,
    )
    weights, rng = _weights(config)
    x = mx.array(rng.standard_normal((1, 6, config.dim), dtype=np.float32))

    expected, picked = _slow_moe(x, weights, config)
    actual = _moe(x, weights, config)
    mx.eval(expected, actual, picked)

    picked = np.asarray(picked)
    assert set(np.unique(picked)).issubset({0, 1})
    assert set(range(2, config.num_experts)).isdisjoint(np.unique(picked))
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-7)
