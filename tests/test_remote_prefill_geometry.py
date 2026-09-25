# SPDX-License-Identifier: Apache-2.0
"""A handoff must have the local model's KV geometry, read from the model's own config."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from omlx.remote_prefill import wire
from omlx.remote_prefill.geometry import check_manifest, model_geometry
from omlx.remote_prefill.receiver import HandoffError


def _model(**args):
    return SimpleNamespace(args=SimpleNamespace(**args))


def _attention(index=0, heads=2, total_heads=2, head_size=16):
    return wire.LayerExport(
        index=index,
        kind="attention",
        shape=(2, heads, 16, 2 * head_size),
        dims=("block", "head", "token", "kv_head_dim"),
        dtype="bfloat16",
        heads=heads,
        total_heads=total_heads,
        head_size=head_size,
    )


def _mla(index=0, latent_size=32, rope_size=8):
    return wire.LayerExport(
        index=index,
        kind="mla",
        shape=(2, 16, latent_size + rope_size),
        dims=("block", "token", "latent"),
        dtype="bfloat16",
        latent_size=latent_size,
        rope_size=rope_size,
    )


def _manifest(*layers, tp_size=1):
    return wire.Manifest(
        handoff="",
        model="m",
        prompt_tokens=30,
        first_token=0,
        token_sha256="",
        block_size=16,
        tp_rank=0,
        tp_size=tp_size,
        layers=layers,
        frames=1,
    )


def test_attention_geometry_comes_from_heads_and_head_dim():
    geometry, reason = model_geometry(
        _model(num_attention_heads=4, num_key_value_heads=2, hidden_size=64), 3
    )
    assert reason == "" and (geometry.kind, geometry.heads, geometry.head_size) == (
        "attention",
        2,
        16,
    )
    check_manifest(_manifest(_attention(0), _attention(1), _attention(2)), geometry)


def test_an_explicit_head_dim_wins_and_nested_text_configs_are_read():
    text = {"num_attention_heads": 8, "num_key_value_heads": 8, "head_dim": 128}
    geometry, _ = model_geometry(
        SimpleNamespace(args=SimpleNamespace(text_config=text, hidden_size=64)), 1
    )
    assert (geometry.heads, geometry.head_size) == (8, 128)


def test_mla_geometry_is_the_latent_and_rope_sizes():
    geometry, _ = model_geometry(
        _model(num_attention_heads=4, kv_lora_rank=32, qk_rope_head_dim=8), 1
    )
    assert (geometry.kind, geometry.latent_size, geometry.rope_size) == ("mla", 32, 8)
    check_manifest(_manifest(_mla()), geometry)
    with pytest.raises(
        HandoffError, match="latent 64 and rope 8; the local model has 32 and 8"
    ):
        check_manifest(_manifest(_mla(latent_size=64)), geometry)


def test_a_model_without_readable_geometry_is_not_used():
    geometry, reason = model_geometry(SimpleNamespace(), 2)
    assert geometry is None and "config" in reason


@pytest.mark.parametrize(
    ("layers", "tp_size", "message"),
    [
        (
            (_attention(heads=4, total_heads=4),),
            1,
            "4 KV heads of 16; the local model has 2 of 16",
        ),
        (
            (_attention(head_size=32),),
            1,
            "2 KV heads of 32; the local model has 2 of 16",
        ),
        ((_mla(),), 1, "an mla layer; the local model has attention"),
        ((_attention(0), _attention(1)), 1, "2 layers; the local model has 1"),
    ],
)
def test_handoffs_of_another_geometry_are_refused(layers, tp_size, message):
    geometry, _ = model_geometry(
        _model(num_attention_heads=4, num_key_value_heads=2, hidden_size=64), 1
    )
    with pytest.raises(HandoffError, match=message):
        check_manifest(_manifest(*layers, tp_size=tp_size), geometry)


def test_ranks_holding_a_share_of_the_heads_are_checked_by_the_total():
    geometry, _ = model_geometry(
        _model(num_attention_heads=4, num_key_value_heads=2, hidden_size=64), 1
    )
    check_manifest(_manifest(_attention(heads=1, total_heads=2), tp_size=2), geometry)
