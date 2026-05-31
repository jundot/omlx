# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.qwen3_5_attention."""

from __future__ import annotations


def test_patched_attention_signature_matches_mlx_vlm_target_verify():
    import inspect

    from omlx.patches.qwen3_5_attention import _build_replacement_call

    sig = inspect.signature(_build_replacement_call())
    params = list(sig.parameters.keys())
    assert params == [
        "self",
        "x",
        "mask",
        "cache",
        "position_ids",
        "position_embeddings",
        "target_verify",
    ]


def test_patched_attention_delegates_target_verify_to_upstream():
    from omlx.patches.qwen3_5_attention import _build_replacement_call

    calls = []

    def original(
        self,
        x,
        mask=None,
        cache=None,
        position_ids=None,
        position_embeddings=None,
        target_verify=False,
    ):
        calls.append(
            (self, x, mask, cache, position_ids, position_embeddings, target_verify)
        )
        return "upstream"

    replacement = _build_replacement_call(original)
    owner = object()
    position_embeddings = ("cos", "sin")
    result = replacement(
        owner,
        "x",
        mask="mask",
        cache="cache",
        position_ids="position_ids",
        position_embeddings=position_embeddings,
        target_verify=True,
    )

    assert result == "upstream"
    assert calls == [
        (
            owner,
            "x",
            "mask",
            "cache",
            "position_ids",
            position_embeddings,
            True,
        )
    ]
