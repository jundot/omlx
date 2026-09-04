# SPDX-License-Identifier: Apache-2.0
"""Tests for the mlx-lm gemma4 sanitize patch (merged MTP checkpoints).

The text-only mlx-lm gemma4 adapter (DFlash targets, VLM->LLM fallback)
has no binding site for the merged ``language_model.mtp.*`` head, so its
sanitize must discard those keys like it discards vision/audio towers.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx_lm.models.gemma4")

from omlx.patches.mlx_lm_mtp import gemma4_text_model


def _sanitize(weights):
    from mlx_lm.models import gemma4 as lm_gemma4

    stub = SimpleNamespace(language_model=SimpleNamespace(sanitize=lambda w: w))
    return lm_gemma4.Model.sanitize(stub, weights)


def test_apply_is_idempotent():
    assert gemma4_text_model.apply()
    assert gemma4_text_model.apply()


def test_sanitize_strips_merged_mtp_keys():
    assert gemma4_text_model.apply()
    weights = {
        "language_model.mtp.model.embed_tokens.weight": 1,
        "language_model.mtp.pre_projection.weight": 2,
        "model.language_model.mtp.post_projection.weight": 3,
        "language_model.model.embed_tokens.weight": 4,
    }
    out = _sanitize(weights)
    assert not any(".mtp." in k or k.startswith("mtp.") for k in out)
    # Backbone keys survive untouched.
    assert "language_model.model.embed_tokens.weight" in out


def test_sanitize_keeps_stock_behavior_without_mtp_keys():
    assert gemma4_text_model.apply()
    weights = {
        "language_model.model.norm.weight": 1,
        "model.vision_tower.patch_embed.weight": 2,
    }
    out = _sanitize(weights)
    assert "language_model.model.norm.weight" in out
    # Stock sanitize still drops multimodal towers.
    assert not any("vision_tower" in k for k in out)
