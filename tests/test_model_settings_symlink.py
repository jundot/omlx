"""Atomic JSON writes must preserve symlinked config files (#1958).

Users keep ~/.omlx config in a git repo and `ln` it into place. settings.json
wrote in place and kept the link; model_settings/profiles/templates used a
temp-file+rename that clobbered the link with a real file. All now route
through _atomic_write_json, which writes through to the link target.
"""

import json
import os

from omlx.model_settings import (
    ModelSettings,
    ModelSettingsManager,
    _atomic_write_json,
)


def test_atomic_write_json_writes_through_symlink(tmp_path):
    real = tmp_path / "repo" / "data.json"
    real.parent.mkdir()
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)

    _atomic_write_json(link, {"x": 1}, indent=2)

    assert link.is_symlink(), "rename clobbered the symlink with a real file"
    assert os.path.realpath(link) == str(real)
    assert json.loads(real.read_text()) == {"x": 1}
    # no stray temp file left behind next to the resolved target
    assert not (real.parent / "data.tmp").exists()


def test_atomic_write_json_plain_file_unchanged(tmp_path):
    # Refactor regression guard: non-symlink path still writes atomically.
    target = tmp_path / "plain.json"
    _atomic_write_json(target, {"y": 2}, indent=2)
    assert not target.is_symlink()
    assert json.loads(target.read_text()) == {"y": 2}


def test_set_settings_preserves_symlinked_model_settings(tmp_path):
    # Repro of #1958: model_settings.json as a symlink into a config repo.
    real = tmp_path / "repo" / "model_settings.json"
    real.parent.mkdir()
    real.write_text('{"version": 1, "models": {}}')

    cfg = tmp_path / "dotomlx"
    cfg.mkdir()
    link = cfg / "model_settings.json"
    link.symlink_to(real)

    mgr = ModelSettingsManager(cfg)
    mgr.set_settings("my-model", ModelSettings(temperature=0.5))

    assert link.is_symlink(), "save broke the symlink (#1958)"
    assert os.path.realpath(link) == str(real)
    assert "my-model" in real.read_text()
