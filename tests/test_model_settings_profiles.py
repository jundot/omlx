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
    def test_list_templates_empty_by_default(self, mgr):
        # Shipped builtins were retired in favor of the client-side preset
        # bundle (`omlx/admin/static/omlx_preset.json`); the server's
        # /api/profile-templates surface now exposes user templates only.
        assert mgr.list_templates() == []

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


class TestTemplatesPersistence:
    """The on-disk template file holds only user-created entries. Built-in
    seed templates were retired in favor of the client-side preset bundle
    (`omlx/admin/static/omlx_preset.json`); /api/profile-templates is now a
    pure user-store surface."""

    def test_no_file_created_when_empty(self, tmp_path):
        ModelSettingsManager(tmp_path)
        # With no user templates and no shipped builtins, the manager must
        # not create the templates file proactively.
        assert not (tmp_path / "global_templates.json").exists()

    def test_user_template_persists_only_itself(self, tmp_path):
        m1 = ModelSettingsManager(tmp_path)
        m1.save_template("custom", "Custom", None, {"temperature": 0.1})

        on_disk = json.loads((tmp_path / "global_templates.json").read_text())
        assert set(on_disk["templates"].keys()) == {"custom"}

        m2 = ModelSettingsManager(tmp_path)
        names = {t["name"] for t in m2.list_templates()}
        assert names == {"custom"}
        # No `is_builtin` is emitted now that builtins are retired; preset
        # vs user classification lives on the client (preset bundle), not
        # on this response.
        assert "is_builtin" not in m2.get_template("custom")


# ==================== Exposed profile models ====================


def _save_exposed_profile(manager, model_id="qwen-base", name="thinking", settings=None):
    return manager.save_profile(
        model_id=model_id,
        name=name,
        display_name=name.title(),
        description=None,
        settings=settings if settings is not None else {"temperature": 0.6, "enable_thinking": True},
        expose_as_model=True,
    )


class TestExposedProfilePersistence:
    def test_save_profile_can_expose_profile_as_model(self, tmp_path):
        manager = ModelSettingsManager(tmp_path)

        profile = _save_exposed_profile(manager)

        assert profile["expose_as_model"] is True

        reloaded = ModelSettingsManager(tmp_path)
        exposed = reloaded.list_exposed_profile_models()
        assert len(exposed) == 1
        assert exposed[0]["model_id"] == "qwen-base:thinking"
        assert exposed[0]["source_model_id"] == "qwen-base"

    def test_list_profiles_includes_derived_model_id(self, mgr):
        _save_exposed_profile(mgr)

        profiles = mgr.list_profiles("qwen-base")

        assert [p["model_id"] for p in profiles] == ["qwen-base:thinking"]

    def test_list_profiles_derives_has_engine_fields(self, mgr):
        """The server classifies engine-construction overrides so UIs can
        warn on exposure without mirroring the field list."""
        _save_exposed_profile(mgr, name="thinking")
        _save_exposed_profile(
            mgr,
            name="accelerated",
            settings={"temperature": 0.6, "dflash_enabled": True},
        )

        flags = {p["name"]: p["has_engine_fields"] for p in mgr.list_profiles("qwen-base")}

        assert flags == {"thinking": False, "accelerated": True}

    def test_alias_drives_advertised_model_id(self, mgr):
        """A base-model alias renames the advertised exposed ID, mirroring
        how /v1/models lists the base model under its alias."""
        mgr.set_settings("qwen-base", ModelSettings(model_alias="gpt-4"))
        _save_exposed_profile(mgr)

        exposed = mgr.list_exposed_profile_models()
        assert [p["model_id"] for p in exposed] == ["gpt-4:thinking"]
        profiles = mgr.list_profiles("qwen-base")
        assert [p["model_id"] for p in profiles] == ["gpt-4:thinking"]

        # Both the alias form and the directory-name form resolve.
        assert mgr.get_exposed_profile_source_model_id("gpt-4:thinking") == "qwen-base"
        assert mgr.get_exposed_profile_source_model_id("qwen-base:thinking") == "qwen-base"

    def test_unexposed_profile_is_not_a_model(self, mgr):
        mgr.save_profile(
            model_id="qwen-base",
            name="thinking",
            display_name="Thinking",
            description=None,
            settings={"temperature": 0.6},
        )

        assert mgr.list_exposed_profile_models() == []
        assert mgr.get_exposed_profile_source_model_id("qwen-base:thinking") is None

    def test_rename_keeps_exposure_and_updates_model_id(self, mgr):
        _save_exposed_profile(mgr)

        mgr.update_profile("qwen-base", "thinking", new_name="reasoning")

        exposed = mgr.list_exposed_profile_models()
        assert [p["model_id"] for p in exposed] == ["qwen-base:reasoning"]
        assert mgr.get_exposed_profile_source_model_id("qwen-base:thinking") is None

    def test_delete_profile_removes_exposed_model(self, mgr):
        _save_exposed_profile(mgr)

        mgr.delete_profile("qwen-base", "thinking")

        assert mgr.list_exposed_profile_models() == []
        assert mgr.get_exposed_profile_source_model_id("qwen-base:thinking") is None


class TestExposedProfileRequestSettings:
    def test_request_settings_overlay_base_without_mutating_it(self, mgr):
        mgr.set_settings("qwen-base", ModelSettings(temperature=0.2, top_p=0.8))
        _save_exposed_profile(mgr)

        settings = mgr.get_settings_for_request(
            "qwen-base:thinking",
            resolved_model_id="qwen-base",
        )

        assert settings.temperature == 0.6
        assert settings.top_p == 0.8
        assert settings.enable_thinking is True
        assert mgr.get_settings("qwen-base").temperature == 0.2
        assert mgr.get_settings("qwen-base").active_profile_name is None

    def test_request_settings_handle_provider_prefix(self, mgr):
        _save_exposed_profile(mgr)

        settings = mgr.get_settings_for_request(
            "omlx/qwen-base:thinking",
            resolved_model_id="qwen-base",
        )

        assert settings.temperature == 0.6
        assert mgr.get_exposed_profile_source_model_id("omlx/qwen-base:thinking") == "qwen-base"

    def test_engine_construction_fields_are_not_overlaid(self, mgr):
        """Exposed profiles overlay request-time fields only — engine
        knobs in the profile stay at the base model's values."""
        mgr.set_settings("qwen-base", ModelSettings(temperature=0.2))
        _save_exposed_profile(
            mgr,
            settings={"temperature": 0.9, "dflash_enabled": True},
        )

        settings = mgr.get_settings_for_request(
            "qwen-base:thinking",
            resolved_model_id="qwen-base",
        )

        assert settings.temperature == 0.9
        assert settings.dflash_enabled is False

    def test_request_settings_fall_back_to_resolved_physical_model(self, mgr):
        mgr.set_settings(
            "qwen-base", ModelSettings(temperature=0.2, model_alias="my-alias")
        )

        settings = mgr.get_settings_for_request(
            "my-alias",
            resolved_model_id="qwen-base",
        )

        assert settings.temperature == 0.2

    def test_alias_form_of_exposed_profile_serves_overlay(self, mgr):
        """Requests to <alias>:<profile> get the profile overlay, same as
        the directory-name form."""
        mgr.set_settings(
            "qwen-base", ModelSettings(temperature=0.2, model_alias="gpt-4")
        )
        _save_exposed_profile(mgr)

        settings = mgr.get_settings_for_request(
            "gpt-4:thinking",
            resolved_model_id="qwen-base",
        )

        assert settings.temperature == 0.6
        assert settings.enable_thinking is True
