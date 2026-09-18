# SPDX-License-Identifier: Apache-2.0
"""MTP-head norm conventions must survive an already-converted checkpoint.

The head's ``pre_fc_norm_hidden`` / ``pre_fc_norm_embedding`` gammas sit
below 0.5 in BOTH the raw-HF and the MLX (+1) convention — the head damps
the target hidden and the next-token embedding before ``fc`` fuses them.
The per-key "mean < 0.5 means raw-HF" test therefore cannot classify
them, and shifts an already-shifted weight a second time.

mlx-vlm 0.7.0 hid this by skipping ``Model.sanitize`` for MLX-format
shards; 0.7.1 sanitizes unconditionally, so the misfire went live and
cost roughly half of MTP draft acceptance while leaving the backbone
untouched. These weights must follow the head's per-layer norms, which
ARE magnitude-discriminable.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.mlx_vlm_mtp import qwen35_moe_vlm_runtime
from omlx.patches.mtp_head_norm_convention import (
    fc_norm_is_raw_hf,
    head_layer_norms_are_raw_hf,
    is_fc_norm,
)

DIM = 8


def _norm(mean: float) -> mx.array:
    return mx.full((DIM,), mean, dtype=mx.float32)


def _model():
    qwen35_moe_vlm_runtime.apply()
    from mlx_vlm.models.qwen3_5_moe import Model

    text_config = SimpleNamespace(
        tie_word_embeddings=False,
        num_experts=0,
        num_hidden_layers=0,
    )
    return SimpleNamespace(
        config=SimpleNamespace(text_config=text_config),
        sanitize=Model.sanitize.__get__(
            SimpleNamespace(config=SimpleNamespace(text_config=text_config))
        ),
    )


def _head_weights(layer_norm_mean: float) -> dict:
    """A head whose per-layer norms declare the checkpoint's convention."""
    return {
        # per-layer head norms: the discriminable signal
        "mtp.layers.0.input_layernorm.weight": _norm(layer_norm_mean),
        "mtp.layers.0.post_attention_layernorm.weight": _norm(layer_norm_mean),
        "mtp.layers.0.self_attn.q_norm.weight": _norm(layer_norm_mean),
        "mtp.layers.0.self_attn.k_norm.weight": _norm(layer_norm_mean),
        # the two that magnitude cannot classify
        "mtp.pre_fc_norm_hidden.weight": _norm(0.4937),
        "mtp.pre_fc_norm_embedding.weight": _norm(0.2734),
    }


def _sanitized(weights):
    return _model().sanitize(dict(weights))


def _mean(weights, key):
    return float(mx.mean(weights[key].astype(mx.float32)).item())


def test_converted_checkpoint_keeps_fc_norms():
    """Per-layer norms already shifted => fc norms are already shifted too."""
    out = _sanitized(_head_weights(layer_norm_mean=0.9049))

    assert _mean(out, "language_model.mtp.pre_fc_norm_hidden.weight") == pytest.approx(
        0.4937, abs=1e-4
    ), "already-converted fc norm was shifted a second time"
    assert _mean(
        out, "language_model.mtp.pre_fc_norm_embedding.weight"
    ) == pytest.approx(0.2734, abs=1e-4)


def test_raw_hf_checkpoint_still_shifts_fc_norms():
    """Per-layer norms zero-centered => the whole head is raw-HF."""
    out = _sanitized(_head_weights(layer_norm_mean=-0.0951))

    assert _mean(out, "language_model.mtp.pre_fc_norm_hidden.weight") == pytest.approx(
        1.4937, abs=1e-4
    ), "raw-HF fc norm must still receive the +1 shift"
    assert _mean(
        out, "language_model.mtp.pre_fc_norm_embedding.weight"
    ) == pytest.approx(1.2734, abs=1e-4)


def test_per_layer_norms_keep_their_own_per_key_decision():
    """The mixed-bundle heuristic is untouched for discriminable keys."""
    weights = _head_weights(layer_norm_mean=-0.0951)
    # JANG-style mixed bundle: mtp.norm already in the MLX convention
    weights["mtp.norm.weight"] = _norm(2.9253)
    out = _sanitized(weights)

    assert _mean(
        out, "language_model.mtp.layers.0.input_layernorm.weight"
    ) == pytest.approx(0.9049, abs=1e-4)
    assert _mean(out, "language_model.mtp.norm.weight") == pytest.approx(
        2.9253, abs=1e-4
    ), "an already-shifted mtp.norm must not be shifted again"


def test_is_fc_norm_matches_only_the_undiscriminable_pair():
    assert is_fc_norm("language_model.mtp.pre_fc_norm_hidden.weight")
    assert is_fc_norm("mtp.pre_fc_norm_embedding.weight")
    assert not is_fc_norm("mtp.norm.weight")
    assert not is_fc_norm("mtp.layers.0.input_layernorm.weight")


def test_verdict_reads_the_head_not_the_backbone():
    """Backbone norms must not vote: they are always converted correctly."""
    weights = {
        "model.language_model.layers.0.input_layernorm.weight": _norm(1.0312),
        "mtp.layers.0.input_layernorm.weight": _norm(-0.0951),
        "mtp.layers.0.self_attn.q_norm.weight": _norm(-0.2328),
    }
    assert head_layer_norms_are_raw_hf(weights) is True


def test_verdict_is_none_without_readable_head_norms():
    """No evidence means callers keep their prior behaviour."""
    assert head_layer_norms_are_raw_hf({}) is None
    assert (
        head_layer_norms_are_raw_hf(
            {"model.language_model.layers.0.input_layernorm.weight": _norm(1.03)}
        )
        is None
    )


def test_dense_and_mlx_lm_sanitizers_agree_with_the_moe_one():
    """All four sanitizer copies share one convention decision."""
    from omlx.patches.mlx_lm_mtp import qwen35_model
    from omlx.patches.mlx_vlm_mtp import qwen35_moe_vlm_model, qwen35_vlm_model

    for module in (qwen35_model, qwen35_moe_vlm_model, qwen35_vlm_model):
        src = module.__file__
        body = open(src).read()
        assert "_fc_norm_is_raw(" in body, f"{src} still decides pre_fc by magnitude"
        assert "mtp_head_norm_convention" in body, f"{src} does not share the decision"


def test_sign_decides_the_unambiguous_cases():
    """Only the 0..0.5 band needs the head's opinion."""
    # negative gamma is raw-HF no matter what the head says
    for verdict in (True, False, None):
        assert fc_norm_is_raw_hf(_norm(-0.44), verdict) is True
    # at or above the legacy cutoff it is already converted
    for verdict in (True, False, None):
        assert fc_norm_is_raw_hf(_norm(0.5393), verdict) is False


def test_ambiguous_band_defers_to_the_head():
    assert fc_norm_is_raw_hf(_norm(0.4937), False) is False
    assert fc_norm_is_raw_hf(_norm(0.4937), True) is True
    # no evidence at all keeps the legacy cutoff's answer
    assert fc_norm_is_raw_hf(_norm(0.4937), None) is True


def test_tie_among_head_norms_counts_as_converted():
    """Shifting is the destructive direction, so it needs a real majority."""
    weights = {
        "mtp.layers.0.self_attn.q_norm.weight": _norm(0.75),
        "mtp.layers.0.self_attn.k_norm.weight": _norm(0.74),
        "mtp.layers.0.input_layernorm.weight": _norm(0.04),
        "mtp.layers.0.post_attention_layernorm.weight": _norm(0.21),
    }
    assert head_layer_norms_are_raw_hf(weights) is False
