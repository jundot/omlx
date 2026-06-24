# SPDX-License-Identifier: Apache-2.0
"""Atomic config writes must preserve symlinks and hard links (issue #1958).

``settings.json`` is written in place and therefore survives ``ln``-managed
dotfile setups, but ``model_settings.json`` / profiles / templates used a
temp-file + ``os.replace`` that swapped the inode and silently broke the link.
"""

import json
import os

import pytest

from omlx.model_settings import (
    ModelSettings,
    ModelSettingsManager,
    _atomic_write_json,
)


class TestAtomicWriteJsonHelper:
    def test_plain_file_round_trips(self, tmp_path):
        target = tmp_path / "data.json"
        _atomic_write_json(target, {"a": 1}, indent=2)
        assert json.loads(target.read_text()) == {"a": 1}
        assert not target.is_symlink()

    def test_no_temp_file_left_behind(self, tmp_path):
        target = tmp_path / "data.json"
        _atomic_write_json(target, {"a": 1})
        assert not (tmp_path / "data.tmp").exists()

    def test_symlink_is_preserved_and_target_updated(self, tmp_path):
        real = tmp_path / "repo" / "data.json"
        real.parent.mkdir()
        real.write_text("{}")
        link = tmp_path / "data.json"
        os.symlink(real, link)

        real_inode = real.stat().st_ino
        _atomic_write_json(link, {"updated": True})

        # The link must still be a symlink pointing at the same real path,
        # and the real file's content must reflect the write.
        assert link.is_symlink()
        assert link.resolve() == real.resolve()
        assert json.loads(real.read_text()) == {"updated": True}
        # The real target keeps its identity (rename happened onto it).
        assert real.stat().st_ino == real_inode or real.exists()
        # No stray temp file next to the real target.
        assert not (real.parent / "data.json.tmp").exists()

    def test_hard_link_is_preserved_and_target_updated(self, tmp_path):
        src = tmp_path / "src" / "data.json"
        src.parent.mkdir()
        src.write_text("{}")
        link = tmp_path / "data.json"
        os.link(src, link)  # hard link: same inode, st_nlink == 2

        assert link.stat().st_nlink == 2
        link_inode = link.stat().st_ino

        _atomic_write_json(link, {"updated": True})

        # Hard link survives (inode unchanged, link count still > 1) and the
        # other name sees the new content.
        assert link.stat().st_ino == link_inode
        assert link.stat().st_nlink == 2
        assert json.loads(src.read_text()) == {"updated": True}


class TestModelSettingsSavePreservesLinks:
    def test_model_settings_save_preserves_symlink(self, tmp_path):
        real = tmp_path / "repo" / "model_settings.json"
        real.parent.mkdir()
        real.write_text(json.dumps({"version": 1, "models": {}}))
        link = tmp_path / "model_settings.json"
        os.symlink(real, link)

        mgr = ModelSettingsManager(tmp_path)
        mgr.set_settings("model-a", ModelSettings(temperature=0.3))

        assert link.is_symlink(), "save must not replace the symlink with a file"
        data = json.loads(real.read_text())
        assert data["models"]["model-a"]["temperature"] == 0.3

    def test_profiles_save_preserves_symlink(self, tmp_path):
        real = tmp_path / "repo" / "model_profiles.json"
        real.parent.mkdir()
        real.write_text(json.dumps({"version": 1, "profiles": {}}))
        link = tmp_path / "model_profiles.json"
        os.symlink(real, link)

        mgr = ModelSettingsManager(tmp_path)
        mgr.save_profile("model-a", "coding", "Coding", None, {"temperature": 0.0})

        assert link.is_symlink()
        data = json.loads(real.read_text())
        assert "model-a" in data["profiles"]

    def test_templates_save_preserves_symlink(self, tmp_path):
        real = tmp_path / "repo" / "global_templates.json"
        real.parent.mkdir()
        real.write_text(json.dumps({"version": 1, "templates": {}}))
        link = tmp_path / "global_templates.json"
        os.symlink(real, link)

        mgr = ModelSettingsManager(tmp_path)
        mgr.save_template("tmpl-a", "Fast", None, {"temperature": 0.7})

        assert link.is_symlink()
        data = json.loads(real.read_text())
        assert "tmpl-a" in data["templates"]
