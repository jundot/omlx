"""dflash_draft_for: which draft a model's settings point the DFlash loader at."""

from pathlib import Path
from types import SimpleNamespace

from omlx.utils.model_loading import dflash_draft_for


def _settings(draft, enabled=True):
    return SimpleNamespace(dflash_enabled=enabled, dflash_draft_model=draft)


def test_home_relative_draft_path_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    draft = tmp_path / "models" / "qwen-dflash2"
    draft.mkdir(parents=True)
    assert dflash_draft_for(_settings("~/models/qwen-dflash2"), None) == str(draft)
    assert Path(dflash_draft_for(_settings("~/models/qwen-dflash2"), None)).is_dir()


def test_repo_ids_and_absolute_paths_pass_through_unchanged():
    assert (
        dflash_draft_for(_settings("z-lab/Qwen3.8-27B-DFlash2"), None)
        == "z-lab/Qwen3.8-27B-DFlash2"
    )
    assert (
        dflash_draft_for(_settings("/models/qwen-dflash2"), None)
        == "/models/qwen-dflash2"
    )


def test_disabled_or_unset_draft():
    assert dflash_draft_for(_settings("~/models/x", enabled=False), None) is None
    assert dflash_draft_for(None, None) is None
    assert dflash_draft_for(_settings(None), None) is None
