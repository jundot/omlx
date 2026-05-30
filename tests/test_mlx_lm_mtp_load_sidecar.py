# SPDX-License-Identifier: Apache-2.0
"""Tests for mtp.safetensors sidecar load + detection."""

from __future__ import annotations

import pytest

from omlx.utils.model_loading import _checkpoint_has_mtp_weights


class TestLoadSidecarPatch:
    def test_apply_idempotent(self):
        from omlx.patches.mlx_lm_mtp import load_sidecar

        assert load_sidecar.apply() in (True, False)
        assert load_sidecar.apply() is True


class TestCheckpointDetectsOptiQLayout:
    def test_opti_q_style_index_without_mtp_keys(self, tmp_path):
        import json

        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.embed_tokens.weight": "model-00001-of-00006.safetensors",
                    }
                }
            )
        )
        try:
            import mlx.core as mx

            mx.save_safetensors(
                str(tmp_path / "mtp.safetensors"),
                {"mtp.fc.weight": mx.zeros((2, 4))},
            )
        except ImportError:
            pytest.skip("mlx not available")

        assert _checkpoint_has_mtp_weights(tmp_path) is True
