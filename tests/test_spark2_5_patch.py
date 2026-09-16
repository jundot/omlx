from __future__ import annotations

import sys
from unittest.mock import MagicMock

from omlx.utils import model_loading
from omlx.utils.model_loading import maybe_apply_pre_load_patches


def _write_config(tmp_path, model_type: str) -> str:
    (tmp_path / "config.json").write_text(f'{{"model_type": "{model_type}"}}')
    return str(tmp_path)


def test_pre_load_dispatch_applies_spark2_5_patch(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.mlx_lm_mtp",
        MagicMock(set_mtp_active=MagicMock()),
    )

    apply_mock = MagicMock(return_value=True)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.spark2_5",
        MagicMock(apply_spark2_5_patch=apply_mock),
    )

    path = _write_config(tmp_path, "spark2_5")
    maybe_apply_pre_load_patches(path)

    apply_mock.assert_called_once_with()


def test_pre_load_dispatch_skips_spark2_5_for_other_models(tmp_path, monkeypatch):
    monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.mlx_lm_mtp",
        MagicMock(set_mtp_active=MagicMock()),
    )

    apply_mock = MagicMock(return_value=True)
    monkeypatch.setitem(
        sys.modules,
        "omlx.patches.spark2_5",
        MagicMock(apply_spark2_5_patch=apply_mock),
    )

    path = _write_config(tmp_path, "llama")
    maybe_apply_pre_load_patches(path)

    apply_mock.assert_not_called()


def test_spark2_5_patch_registers_model_module():
    from omlx.patches.spark2_5 import apply_spark2_5_patch

    assert apply_spark2_5_patch() is True

    import mlx_lm.models.spark2_5 as spark2_5

    assert spark2_5.Model is not None
    assert spark2_5.ModelArgs is not None
