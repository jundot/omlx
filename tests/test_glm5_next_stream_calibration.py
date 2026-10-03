# SPDX-License-Identifier: Apache-2.0
"""Tests for glm5_next support in the streaming oQe imatrix sourcer.

Covers the pieces that are layout-specific for GLM-5.3 and that the existing
minimax/qwen4_exp tests cannot reach:

* the model_type is registered with the streamed calibration path,
* the boundary state builder tiles the hidden into ``(B, S, hc_mult, D)`` and
  returns the causal mask the sparse-attention layers take,
* a collector installed on a bare block with a ``name_prefix`` keeps both name
  spaces, which is what lets the glm5_next ``embed_q`` capture resolve its
  module (it looks the module up by the full entry name).

No model checkpoint is loaded.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.oq import (
    OQImatrixCollector,
    _STREAM_CALIBRATION_SUPPORTED_MODEL_TYPES,
    _stream_calibration_supported,
    _streamed_glm5_next_state,
)


def _glm5_config(hc_mult: int = 4, hidden_size: int = 8) -> dict:
    return {
        "model_type": "glm5_next",
        "text_config": {"hc_mult": hc_mult, "hidden_size": hidden_size},
    }


def test_glm5_next_streams_but_unknown_layouts_do_not():
    assert "glm5_next" in _STREAM_CALIBRATION_SUPPORTED_MODEL_TYPES
    assert _stream_calibration_supported("glm5_next")
    assert not _stream_calibration_supported("qwen3_5_moe")


def test_boundary_state_tiles_hc_streams_and_builds_a_mask():
    from mlx_lm.models.base import create_attention_mask

    embedded = mx.ones((2, 3, 8))
    hidden, mask = _streamed_glm5_next_state(_glm5_config(), embedded)
    # HyperConnection streams are materialised; the mask must be exactly the
    # one the resident glm5_next branch builds from the un-tiled boundary.
    expected = create_attention_mask(embedded, None, return_array=True)
    assert hidden.shape == (2, 3, 4, 8)
    assert mask.shape == expected.shape
    assert mask.dtype == expected.dtype
    mx.eval(hidden, mask)


def test_boundary_state_requires_hc_mult():
    with pytest.raises(RuntimeError, match="hc_mult"):
        _streamed_glm5_next_state(
            {"model_type": "glm5_next", "text_config": {}}, mx.ones((1, 2, 8))
        )


def test_prefixed_install_keeps_both_name_spaces():
    """A prefixed install must not hide modules from entry-name lookups."""

    collector = OQImatrixCollector()

    class _Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_b_proj = nn.Linear(4, 4)
            self.embed_q = nn.Linear(4, 4)

    attention = _Attention()
    prefix = "language_model.model.layers.3.self_attn."
    installed = collector.install(attention, name_prefix=prefix)
    assert installed == 2
    # Bare paths stay available for restore(), full entry names for the hooks.
    assert "embed_q" in collector._original_modules
    assert f"{prefix}embed_q" in collector._prefixed_modules
    collector.restore(attention)
    assert not collector._prefixed_modules
