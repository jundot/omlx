from __future__ import annotations

import mlx.core as mx

from omlx.patches.mlx_lm_mtp.norm_repair import repair_legacy_head_norms


def _v(mean: float, n: int = 8) -> mx.array:
    return mx.full((n,), mean)


def _payload(head_input_layernorm=0.099, pre_fc_embed=-0.480, pre_fc_hidden=-0.227):
    # Backbone norms are already +1-shifted; the head norms model the raw
    # zero-centered values of the OptiQ side-car from issue #4960.
    return {
        "model.layers.0.self_attn.q_norm.weight": _v(1.20),
        "model.layers.1.self_attn.q_norm.weight": _v(1.20),
        "model.layers.0.self_attn.k_norm.weight": _v(1.20),
        "model.layers.1.self_attn.k_norm.weight": _v(1.20),
        "model.layers.0.input_layernorm.weight": _v(1.20),
        "model.layers.1.input_layernorm.weight": _v(1.20),
        "model.layers.0.post_attention_layernorm.weight": _v(0.93),
        "model.layers.1.post_attention_layernorm.weight": _v(0.93),
        "model.norm.weight": _v(2.14),
        "mtp.layers.0.input_layernorm.weight": _v(head_input_layernorm),
        "mtp.layers.0.self_attn.q_norm.weight": _v(0.75),
        "mtp.layers.0.self_attn.k_norm.weight": _v(0.75),
        "mtp.layers.0.post_attention_layernorm.weight": _v(0.50),
        "mtp.pre_fc_norm_embedding.weight": _v(pre_fc_embed),
        "mtp.pre_fc_norm_hidden.weight": _v(pre_fc_hidden),
        # Raw mtp.norm measures ~1.3-1.9 on real checkpoints; below that the
        # repair could not converge in one pass.
        "mtp.norm.weight": _v(1.30),
        "mtp.layers.0.self_attn.q_proj.weight": _v(0.0, (8, 4)),
    }


def test_raw_sidecar_norms_get_shifted():
    weights, fixed = repair_legacy_head_norms(_payload())
    # input_layernorm via the backbone anchor, the two pre_fc norms via their
    # negative means, plus the anchored head norms that sit ~1 below their
    # backbone counterparts.
    assert fixed == 7
    assert float(weights["mtp.layers.0.input_layernorm.weight"].mean()) == 1.099
    assert float(weights["mtp.pre_fc_norm_embedding.weight"].mean()) == 0.52
    assert float(weights["mtp.pre_fc_norm_hidden.weight"].mean()) == 0.77


def test_converted_sidecar_is_untouched():
    payload = _payload(
        head_input_layernorm=1.099, pre_fc_embed=0.52, pre_fc_hidden=0.77
    )
    # The anchored head norms below are also already above their margins.
    for k in (
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.norm.weight",
    ):
        payload[k] = _v(2.0)
    weights, fixed = repair_legacy_head_norms(payload)
    assert fixed == 0
    assert float(weights["mtp.layers.0.input_layernorm.weight"].mean()) == 1.099


def test_repair_is_idempotent():
    weights, _ = repair_legacy_head_norms(_payload())
    _, fixed = repair_legacy_head_norms(weights)
    assert fixed == 0
