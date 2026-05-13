# SPDX-License-Identifier: Apache-2.0
"""Tests for profile/template CRUD on ModelSettingsManager."""


import json

import pytest

from omlx.model_profiles import InvalidProfileNameError
from omlx.model_settings import ModelSettings, ModelSettingsManager


@pytest.fixture
def mgr(tmp_path):
    return ModelSettingsManager(tmp_path)


class TestProfilesCRUD:
    def test_list_profiles_empty_by_default(self, mgr):
        assert mgr.list_profiles("model-a") == []

    def test_save_and_list_profile(self, mgr):
        mgr.save_profile(
            model_id="model-a",
            name="coding",
            display_name="Coding",
            description="det.",
            settings={"temperature": 0.0, "top_p": 0.95, "is_pinned": True},
        )
        profiles = mgr.list_profiles("model-a")
        assert len(profiles) == 1
        assert profiles[0]["name"] == "coding"
        assert profiles[0]["display_name"] == "Coding"
        # is_pinned is excluded
        assert "is_pinned" not in profiles[0]["settings"]
        assert profiles[0]["settings"]["temperature"] == 0.0

    def test_save_profile_rejects_duplicate_name(self, mgr):
        mgr.save_profile("m", "coding", "Coding", None, {"temperature": 0.0})
        with pytest.raises(ValueError, match="already exists"):
            mgr.save_profile("m", "coding", "Coding", None, {"temperature": 0.1})

    def test_save_profile_rejects_invalid_name(self, mgr):
        with pytest.raises(InvalidProfileNameError):
            mgr.save_profile("m", "Has Space", "x", None, {})

    def test_get_profile_returns_none_for_missing(self, mgr):
        assert mgr.get_profile("m", "nope") is None

    def test_update_profile_metadata(self, mgr):
        mgr.save_profile("m", "coding", "Coding", None, {"temperature": 0.0})
        mgr.update_profile(
            "m", "coding",
            display_name="Coding v2",
            description="new desc",
            settings={"temperature": 0.2},
        )
        p = mgr.get_profile("m", "coding")
        assert p["display_name"] == "Coding v2"
        assert p["description"] == "new desc"
        assert p["settings"]["temperature"] == 0.2

    def test_rename_profile(self, mgr):
        mgr.save_profile("m", "coding", "Coding", None, {"temperature": 0.0})
        mgr.update_profile("m", "coding", new_name="coding-v2")
        assert mgr.get_profile("m", "coding") is None
        assert mgr.get_profile("m", "coding-v2") is not None

    def test_rename_to_existing_fails(self, mgr):
        mgr.save_profile("m", "a", "A", None, {})
        mgr.save_profile("m", "b", "B", None, {})
        with pytest.raises(ValueError, match="already exists"):
            mgr.update_profile("m", "a", new_name="b")

    def test_delete_profile(self, mgr):
        mgr.save_profile("m", "coding", "Coding", None, {"temperature": 0.0})
        assert mgr.delete_profile("m", "coding") is True
        assert mgr.get_profile("m", "coding") is None

    def test_delete_missing_returns_false(self, mgr):
        assert mgr.delete_profile("m", "nope") is False

    def test_profiles_persist_across_instances(self, tmp_path):
        m1 = ModelSettingsManager(tmp_path)
        m1.save_profile("m", "coding", "Coding", None, {"temperature": 0.0})
        m2 = ModelSettingsManager(tmp_path)
        assert m2.get_profile("m", "coding") is not None

    def test_rename_cascade_persists_to_disk(self, tmp_path):
        m1 = ModelSettingsManager(tmp_path)
        m1.save_profile("m", "coding", "Coding", None, {"temperature": 0.0})
        m1.apply_profile("m", "coding")
        m1.update_profile("m", "coding", new_name="coding-v2")
        m2 = ModelSettingsManager(tmp_path)
        assert m2.get_settings("m").active_profile_name == "coding-v2"

    def test_delete_cascade_persists_to_disk(self, tmp_path):
        m1 = ModelSettingsManager(tmp_path)
        m1.save_profile("m", "coding", "Coding", None, {"temperature": 0.0})
        m1.apply_profile("m", "coding")
        m1.delete_profile("m", "coding")
        m2 = ModelSettingsManager(tmp_path)
        assert m2.get_settings("m").active_profile_name is None


class TestApplyProfile:
    def test_apply_sets_settings_and_active_name(self, mgr):
        mgr.save_profile("m", "coding", "Coding", None,
                         {"temperature": 0.0, "top_p": 0.95})
        applied = mgr.apply_profile("m", "coding")
        assert applied is not None
        assert applied.temperature == 0.0
        assert applied.top_p == 0.95
        assert applied.active_profile_name == "coding"

        # Persisted
        again = mgr.get_settings("m")
        assert again.active_profile_name == "coding"
        assert again.temperature == 0.0

    def test_apply_merges_leaves_unset_fields_alone(self, mgr):
        # Pre-existing settings
        pre = ModelSettings(temperature=0.9, top_p=0.5, top_k=40)
        mgr.set_settings("m", pre)
        mgr.save_profile("m", "coding", "Coding", None, {"temperature": 0.0})
        mgr.apply_profile("m", "coding")
        s = mgr.get_settings("m")
        assert s.temperature == 0.0          # overwritten
        assert s.top_p == 0.5                # preserved
        assert s.top_k == 40                 # preserved

    def test_apply_missing_profile_returns_none(self, mgr):
        assert mgr.apply_profile("m", "nope") is None


class TestProfileFieldFiltering:
    def test_save_filters_excluded_fields(self, mgr):
        mgr.save_profile("m", "p", "P", None, {
            "temperature": 0.5,
            "is_pinned": True,
            "is_default": True,
            "display_name": "ignored",
            "unknown_key": "x",
        })
        p = mgr.get_profile("m", "p")
        assert p["settings"] == {"temperature": 0.5}


class TestTemplatesCRUD:
    def test_list_templates_includes_qwen36_builtins(self, mgr):
        # Built-in templates surface in the merged list even when no user
        # templates exist. The CRUD tests below all use names that don't
        # collide with the built-ins.
        names = {t["name"] for t in mgr.list_templates()}
        assert {
            "qwen36-thinking-general",
            "qwen36-thinking-coding",
            "qwen36-instruct-general",
            "qwen36-instruct-reasoning",
        } <= names

    def test_save_template_universal_only(self, mgr):
        mgr.save_template(
            name="coding",
            display_name="Coding",
            description="d",
            settings={
                "temperature": 0.0,
                "turboquant_kv_enabled": True,
                "is_pinned": True,
            },
        )
        t = mgr.get_template("coding")
        assert t is not None
        assert t["settings"] == {"temperature": 0.0}

    def test_save_template_rejects_duplicate(self, mgr):
        mgr.save_template("coding", "Coding", None, {"temperature": 0.0})
        with pytest.raises(ValueError, match="already exists"):
            mgr.save_template("coding", "Coding", None, {"temperature": 0.1})

    def test_save_template_rejects_invalid_name(self, mgr):
        with pytest.raises(InvalidProfileNameError):
            mgr.save_template("Has Space", "x", None, {})

    def test_update_template(self, mgr):
        mgr.save_template("coding", "Coding", None, {"temperature": 0.0})
        mgr.update_template(
            "coding",
            display_name="Coding v2",
            settings={"temperature": 0.2, "turboquant_kv_enabled": True},
        )
        t = mgr.get_template("coding")
        assert t["display_name"] == "Coding v2"
        assert t["settings"] == {"temperature": 0.2}

    def test_rename_template(self, mgr):
        mgr.save_template("coding", "Coding", None, {"temperature": 0.0})
        mgr.update_template("coding", new_name="coding-v2")
        assert mgr.get_template("coding") is None
        assert mgr.get_template("coding-v2") is not None

    def test_delete_template(self, mgr):
        mgr.save_template("coding", "Coding", None, {"temperature": 0.0})
        assert mgr.delete_template("coding") is True
        assert mgr.get_template("coding") is None

    def test_delete_missing_returns_false(self, mgr):
        assert mgr.delete_template("nope") is False

    def test_templates_persist_across_instances(self, tmp_path):
        m1 = ModelSettingsManager(tmp_path)
        m1.save_template("coding", "Coding", None, {"temperature": 0.0})
        m2 = ModelSettingsManager(tmp_path)
        assert m2.get_template("coding") is not None


class TestBuiltinTemplates:
    """Built-in contract — the four Qwen3.6 defaults ship inside the package
    (omlx/default_global_templates.json), are merged in at read time, and are
    NEVER written to the user's disk. Users who want different defaults
    create their own templates with different names.
    """

    def test_builtins_visible_without_writing_file(self, tmp_path):
        ModelSettingsManager(tmp_path)
        # The user-owned file must NOT be created just for built-ins.
        assert not (tmp_path / "global_templates.json").exists()

    def test_list_includes_all_four_qwen36_builtins(self, tmp_path):
        mgr = ModelSettingsManager(tmp_path)
        names = {t["name"] for t in mgr.list_templates()}
        assert "qwen36-thinking-general" in names
        assert "qwen36-thinking-coding" in names
        assert "qwen36-instruct-general" in names
        assert "qwen36-instruct-reasoning" in names

    def test_builtins_flagged_is_builtin_true(self, tmp_path):
        mgr = ModelSettingsManager(tmp_path)
        for t in mgr.list_templates():
            if t["name"].startswith("qwen36-"):
                assert t["is_builtin"] is True

    @pytest.mark.parametrize("name,expected", [
        ("qwen36-thinking-general", {
            "max_context_window": 131072,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "repetition_penalty": 1.0,
            "enable_thinking": True,
        }),
        ("qwen36-thinking-coding", {
            "max_context_window": 131072,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0,
            "enable_thinking": True,
        }),
        ("qwen36-instruct-general", {
            "max_context_window": 131072,
            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "repetition_penalty": 1.0,
            "enable_thinking": False,
        }),
        ("qwen36-instruct-reasoning", {
            "max_context_window": 131072,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 1.5, "repetition_penalty": 1.0,
            "enable_thinking": False,
        }),
    ])
    def test_builtin_values_match_qwen36_recipe(self, tmp_path, name, expected):
        mgr = ModelSettingsManager(tmp_path)
        t = mgr.get_template(name)
        assert t is not None
        assert t["settings"] == expected
        assert t["is_builtin"] is True

    def test_save_with_builtin_name_rejected(self, tmp_path):
        mgr = ModelSettingsManager(tmp_path)
        with pytest.raises(ValueError, match="built-in"):
            mgr.save_template("qwen36-thinking-general", "x", None,
                              {"temperature": 0.1})

    def test_upsert_with_builtin_name_rejected(self, tmp_path):
        mgr = ModelSettingsManager(tmp_path)
        with pytest.raises(ValueError, match="built-in"):
            mgr.upsert_template("qwen36-thinking-general", "x", None,
                                {"temperature": 0.1})

    def test_update_builtin_rejected(self, tmp_path):
        mgr = ModelSettingsManager(tmp_path)
        with pytest.raises(ValueError, match="built-in"):
            mgr.update_template("qwen36-instruct-general",
                                settings={"temperature": 0.3})

    def test_rename_into_builtin_name_rejected(self, tmp_path):
        mgr = ModelSettingsManager(tmp_path)
        mgr.save_template("custom", "Custom", None, {"temperature": 0.1})
        with pytest.raises(ValueError, match="built-in"):
            mgr.update_template("custom", new_name="qwen36-thinking-general")

    def test_delete_builtin_rejected(self, tmp_path):
        mgr = ModelSettingsManager(tmp_path)
        with pytest.raises(ValueError, match="built-in"):
            mgr.delete_template("qwen36-thinking-coding")

    def test_user_template_coexists_with_builtins(self, tmp_path):
        mgr = ModelSettingsManager(tmp_path)
        mgr.save_template("my-tuned", "My Tuned", "experiment",
                          {"temperature": 0.3, "top_p": 0.9})

        ts = {t["name"]: t for t in mgr.list_templates()}
        assert "my-tuned" in ts
        assert ts["my-tuned"]["is_builtin"] is False
        # All four built-ins still present alongside the user one.
        assert "qwen36-thinking-general" in ts
        assert ts["qwen36-thinking-general"]["is_builtin"] is True

    def test_user_templates_persist_builtins_not_written_to_disk(self, tmp_path):
        m1 = ModelSettingsManager(tmp_path)
        m1.save_template("custom", "Custom", None, {"temperature": 0.1})

        # The on-disk file holds ONLY the user template — no qwen36-* entries.
        on_disk = json.loads((tmp_path / "global_templates.json").read_text())
        assert set(on_disk["templates"].keys()) == {"custom"}

        # But a fresh manager still surfaces the built-ins via the merged view.
        m2 = ModelSettingsManager(tmp_path)
        names = {t["name"] for t in m2.list_templates()}
        assert "custom" in names
        assert "qwen36-thinking-general" in names
