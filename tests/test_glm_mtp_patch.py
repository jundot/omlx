# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.mlx_lm_mtp.glm_moe_dsa_model (GLM-5.2 external MTP head).

Mirrors the structural style of test_mlx_lm_mtp_patch.py / test_glm_moe_dsa_patch.py:
registration, method presence, idempotency, and the external-checkpoint arming
path — no full-size model instantiation.
"""

from __future__ import annotations

import json

import pytest


def _apply_glm_base():
    try:
        from omlx.patches.glm_moe_dsa import apply_glm_moe_dsa_patch
    except ImportError:
        pytest.skip("omlx.patches.glm_moe_dsa not importable")
    apply_glm_moe_dsa_patch()
    import sys

    glm = sys.modules.get("mlx_lm.models.glm_moe_dsa")
    if glm is None or not hasattr(glm, "Model"):
        pytest.skip("glm_moe_dsa module not registered (mlx_lm absent?)")
    return glm


class TestApply:
    def test_apply_requires_registered_glm_module(self, monkeypatch):
        import sys

        from omlx.patches.mlx_lm_mtp import glm_moe_dsa_model

        monkeypatch.delitem(sys.modules, "mlx_lm.models.glm_moe_dsa", raising=False)
        assert glm_moe_dsa_model.apply() is False

    def test_apply_registers_head_and_hooks(self):
        glm = _apply_glm_base()
        from omlx.patches.mlx_lm_mtp import glm_moe_dsa_model

        assert glm_moe_dsa_model.apply() is True
        assert hasattr(glm, "GlmMoeDsaMTP")
        for attr in ("mtp_forward", "make_mtp_cache", "attach_glm_mtp"):
            assert hasattr(glm.Model, attr), attr
        # Backbone + outer __call__ carry the idempotency marker.
        assert getattr(glm.GlmMoeDsaModel.__call__, "_omlx_glm_mtp_marker", False)
        assert getattr(glm.Model.__dict__["__call__"], "_omlx_glm_mtp_marker", False)

    def test_apply_idempotent(self):
        glm = _apply_glm_base()
        from omlx.patches.mlx_lm_mtp import glm_moe_dsa_model

        assert glm_moe_dsa_model.apply() is True
        call_before = glm.Model.__dict__["__call__"]
        assert glm_moe_dsa_model.apply() is True
        assert glm.Model.__dict__["__call__"] is call_before

    def test_orchestrator_includes_glm_submodule(self):
        _apply_glm_base()
        from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

        assert apply_mlx_lm_mtp_patch() is True


class TestExternalCheckpointArming:
    def test_find_glm_mtp_checkpoint(self, tmp_path):
        from omlx.patches.mlx_lm_mtp.glm_moe_dsa_model import find_glm_mtp_checkpoint

        assert find_glm_mtp_checkpoint(str(tmp_path)) is None
        mtp = tmp_path / "mtp"
        mtp.mkdir()
        assert find_glm_mtp_checkpoint(str(tmp_path)) is None  # config.json requis
        (mtp / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa_mtp"}))
        assert find_glm_mtp_checkpoint(str(tmp_path)) == str(mtp)

    def test_stash_set_and_consumed_once(self):
        from omlx.patches.mlx_lm_mtp import glm_moe_dsa_model as g

        g.set_glm_mtp_path("/nonexistent/mtp")

        class _NoAttach:
            pass

        model = _NoAttach()
        # Modèle sans attach_glm_mtp → warning + modèle inchangé, stash consommé.
        assert g.maybe_attach_glm_mtp(model) is model
        # Second appel : stash vidé → no-op strict.
        assert g.maybe_attach_glm_mtp(model) is model

    def test_attach_failure_is_fail_open(self):
        glm = _apply_glm_base()
        from omlx.patches.mlx_lm_mtp import glm_moe_dsa_model as g

        assert g.apply() is True

        class _FakeModel:
            attach_glm_mtp = glm.Model.attach_glm_mtp

        g.set_glm_mtp_path("/nonexistent/mtp")
        model = _FakeModel()
        # Chemin inexistant → l'attach lève en interne, maybe_attach n'explose
        # pas et rend le modèle inchangé (serving sans MTP).
        assert g.maybe_attach_glm_mtp(model) is model
        assert getattr(model, "_omlx_mtp_decode_enabled", False) is False

    def test_pre_load_arms_glm_mtp(self, tmp_path, monkeypatch):
        _apply_glm_base()
        from omlx.model_settings import ModelSettings
        from omlx.patches.mlx_lm_mtp import glm_moe_dsa_model as g
        from omlx.utils.model_loading import maybe_apply_pre_load_patches

        model_dir = tmp_path / "GLM-5.2-test"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}))
        mtp = model_dir / "mtp"
        mtp.mkdir()
        (mtp / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa_mtp"}))

        settings = ModelSettings(mtp_enabled=True)
        maybe_apply_pre_load_patches(str(model_dir), model_settings=settings)
        assert g._GLM_MTP_PATH == str(mtp)

        # mtp_enabled=False → stash non armé (reset par le pré-load suivant).
        settings_off = ModelSettings(mtp_enabled=False)
        maybe_apply_pre_load_patches(str(model_dir), model_settings=settings_off)
        assert g._GLM_MTP_PATH is None
