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
        assert profiles[0]["api_name"] == "coding"
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
            "m",
            "coding",
            display_name="Coding v2",
            description="new desc",
            settings={"temperature": 0.2},
        )
        p = mgr.get_profile("m", "coding")
        assert p["display_name"] == "Coding v2"
        assert p["api_name"] == "coding"
        assert p["description"] == "new desc"
        assert p["settings"]["temperature"] == 0.2

    def test_update_profile_api_name(self, mgr):
        mgr.save_profile("m", "p-abc", "Fast Chat", None, {"temperature": 0.0})
        mgr.update_profile("m", "p-abc", api_name="fast-chat-api")

        p = mgr.get_profile("m", "p-abc")
        assert p["name"] == "p-abc"
        assert p["display_name"] == "Fast Chat"
        assert p["api_name"] == "fast-chat-api"

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

    def test_existing_profiles_get_display_name_based_api_name(self, tmp_path):
        profiles_file = tmp_path / "model_profiles.json"
        profiles_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "m": {
                            "p-abc123": {
                                "name": "p-abc123",
                                "display_name": "Fast Chat",
                                "description": None,
                                "created_at": "2026-06-01T00:00:00+00:00",
                                "updated_at": "2026-06-01T00:00:00+00:00",
                                "settings": {"temperature": 0.1},
                                "source_template": None,
                            }
                        }
                    },
                }
            )
        )

        manager = ModelSettingsManager(tmp_path)

        profile = manager.get_profile("m", "p-abc123")
        assert profile["name"] == "p-abc123"
        assert profile["display_name"] == "Fast Chat"
        assert profile["api_name"] == "fast-chat"

    def test_profile_api_name_migration_dedupes_collisions(self, tmp_path):
        profiles_file = tmp_path / "model_profiles.json"
        profiles_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "m": {
                            "p-a": {
                                "name": "p-a",
                                "display_name": "Fast Chat",
                                "created_at": "2026-06-01T00:00:00+00:00",
                                "updated_at": "2026-06-01T00:00:00+00:00",
                                "settings": {},
                            },
                            "p-b": {
                                "name": "p-b",
                                "display_name": "Fast Chat",
                                "created_at": "2026-06-01T00:00:00+00:00",
                                "updated_at": "2026-06-01T00:00:00+00:00",
                                "settings": {},
                            },
                        }
                    },
                }
            )
        )

        manager = ModelSettingsManager(tmp_path)
        api_names = [p["api_name"] for p in manager.list_profiles("m")]

        assert api_names == ["fast-chat", "fast-chat-2"]

    def test_profile_api_name_migration_avoids_random_internal_name_fallback(
        self, tmp_path
    ):
        profiles_file = tmp_path / "model_profiles.json"
        profiles_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "m": {
                            "p-random": {
                                "name": "p-random",
                                "display_name": "思考モード",
                                "created_at": "2026-06-01T00:00:00+00:00",
                                "updated_at": "2026-06-01T00:00:00+00:00",
                                "settings": {},
                            }
                        }
                    },
                }
            )
        )

        manager = ModelSettingsManager(tmp_path)

        profile = manager.get_profile("m", "p-random")
        assert profile["api_name"] == "profile"

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
        mgr.save_profile(
            "m", "coding", "Coding", None, {"temperature": 0.0, "top_p": 0.95}
        )
        applied = mgr.apply_profile("m", "coding")
        assert applied is not None
        assert applied.temperature == 0.0
        assert applied.top_p == 0.95
        assert applied.active_profile_name == "coding"

        # Persisted
        again = mgr.get_settings("m")
        assert again.active_profile_name == "coding"
        assert again.temperature == 0.0

    def test_apply_resets_absent_universal_and_preserves_model_specific(self, mgr):
        pre = ModelSettings(
            temperature=0.9,
            top_p=0.5,
            top_k=40,
            max_tokens=512,
            dflash_enabled=True,
            ttl_seconds=300,
            is_pinned=True,
        )
        mgr.set_settings("m", pre)
        mgr.save_profile("m", "coding", "Coding", None, {"temperature": 0.0})
        mgr.apply_profile("m", "coding")
        s = mgr.get_settings("m")
        assert s.temperature == 0.0  # overwritten
        # Universal fields absent from the profile reset to defaults
        assert s.top_p is None
        assert s.top_k is None
        assert s.max_tokens is None
        # Model-specific fields keep additive overlay (preset/template chips
        # materialize universal-only profiles and must not disturb them)
        assert s.dflash_enabled is True
        # Excluded fields are never touched by profiles
        assert s.ttl_seconds == 300
        assert s.is_pinned is True

    def test_apply_round_trip_clears_removed_universal_fields(self, mgr):
        mgr.save_profile(
            "m",
            "p",
            "P",
            None,
            {"max_context_window": 8192, "max_tokens": 512, "temperature": 0.7},
        )
        mgr.apply_profile("m", "p")
        s = mgr.get_settings("m")
        assert s.max_context_window == 8192
        assert s.max_tokens == 512

        mgr.update_profile("m", "p", settings={"temperature": 0.7})
        mgr.apply_profile("m", "p")
        s = mgr.get_settings("m")
        assert s.max_context_window is None
        assert s.max_tokens is None
        assert s.temperature == 0.7

    def test_apply_round_trip_clears_removed_kwargs(self, mgr):
        mgr.save_profile(
            "m",
            "p",
            "P",
            None,
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "forced_ct_kwargs": ["enable_thinking"],
            },
        )
        mgr.apply_profile("m", "p")
        s = mgr.get_settings("m")
        assert s.chat_template_kwargs == {"enable_thinking": True}
        assert s.forced_ct_kwargs == ["enable_thinking"]

        mgr.update_profile("m", "p", settings={"temperature": 0.5})
        mgr.apply_profile("m", "p")
        s = mgr.get_settings("m")
        assert s.chat_template_kwargs is None
        assert s.forced_ct_kwargs is None

    def test_apply_overlays_model_specific_fields_when_present(self, mgr):
        mgr.set_settings("m", ModelSettings(turboquant_kv_enabled=False))
        mgr.save_profile("m", "p", "P", None, {"turboquant_kv_enabled": True})
        mgr.apply_profile("m", "p")
        assert mgr.get_settings("m").turboquant_kv_enabled is True

    def test_apply_tolerates_legacy_empty_string_values(self, tmp_path):
        profiles_file = tmp_path / "model_profiles.json"
        profiles_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "m": {
                            "p": {
                                "name": "p",
                                "display_name": "P",
                                "created_at": "2026-06-01T00:00:00+00:00",
                                "updated_at": "2026-06-01T00:00:00+00:00",
                                "settings": {
                                    "max_context_window": "",
                                    "max_tokens": None,
                                    "temperature": 0.3,
                                },
                            }
                        }
                    },
                }
            )
        )
        manager = ModelSettingsManager(tmp_path)
        manager.apply_profile("m", "p")
        s = manager.get_settings("m")
        assert s.max_context_window is None
        assert s.max_tokens is None
        assert s.temperature == 0.3

    def test_apply_missing_profile_returns_none(self, mgr):
        assert mgr.apply_profile("m", "nope") is None


class TestProfileFieldFiltering:
    def test_save_filters_excluded_fields(self, mgr):
        mgr.save_profile(
            "m",
            "p",
            "P",
            None,
            {
                "temperature": 0.5,
                "is_pinned": True,
                "is_default": True,
                "display_name": "ignored",
                "unknown_key": "x",
            },
        )
        p = mgr.get_profile("m", "p")
        assert p["settings"] == {"temperature": 0.5}

    def test_save_and_update_drop_none_and_empty_string_values(self, mgr):
        mgr.save_profile(
            "m",
            "p",
            "P",
            None,
            {"temperature": 0.5, "max_tokens": None, "max_context_window": ""},
        )
        assert mgr.get_profile("m", "p")["settings"] == {"temperature": 0.5}

        mgr.update_profile(
            "m",
            "p",
            settings={"top_p": 0.9, "max_tokens": None, "reasoning_parser": ""},
        )
        assert mgr.get_profile("m", "p")["settings"] == {"top_p": 0.9}

    def test_save_template_drops_none_and_empty_string_values(self, mgr):
        mgr.save_template(
            "t",
            "T",
            None,
            {"temperature": 0.1, "max_tokens": None, "reasoning_parser": ""},
        )
        assert mgr.get_template("t")["settings"] == {"temperature": 0.1}


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


def _save_exposed_profile(
    manager, model_id="qwen-base", name="thinking", settings=None
):
    return manager.save_profile(
        model_id=model_id,
        name=name,
        display_name=name.title(),
        description=None,
        settings=(
            settings
            if settings is not None
            else {"temperature": 0.6, "enable_thinking": True}
        ),
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

        flags = {
            p["name"]: p["has_engine_fields"] for p in mgr.list_profiles("qwen-base")
        }

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
        assert (
            mgr.get_exposed_profile_source_model_id("qwen-base:thinking") == "qwen-base"
        )

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

    def test_rename_keeps_exposure_and_preserves_model_id(self, mgr):
        _save_exposed_profile(mgr)

        mgr.update_profile("qwen-base", "thinking", new_name="reasoning")

        exposed = mgr.list_exposed_profile_models()
        assert [p["model_id"] for p in exposed] == ["qwen-base:thinking"]
        assert (
            mgr.get_exposed_profile_source_model_id("qwen-base:thinking") == "qwen-base"
        )

    def test_api_name_drives_exposed_model_id(self, mgr):
        mgr.save_profile(
            model_id="qwen-base",
            name="p-abc123",
            display_name="Fast Chat",
            description=None,
            settings={"temperature": 0.6},
            api_name="fast-chat",
            expose_as_model=True,
        )

        exposed = mgr.list_exposed_profile_models()
        assert [p["name"] for p in exposed] == ["p-abc123"]
        assert [p["api_name"] for p in exposed] == ["fast-chat"]
        assert [p["model_id"] for p in exposed] == ["qwen-base:fast-chat"]
        assert mgr.get_exposed_profile_source_model_id("qwen-base:p-abc123") is None
        assert (
            mgr.get_exposed_profile_source_model_id("qwen-base:fast-chat")
            == "qwen-base"
        )

    def test_update_api_name_updates_exposed_model_id(self, mgr):
        _save_exposed_profile(mgr)

        mgr.update_profile("qwen-base", "thinking", api_name="reasoning")

        exposed = mgr.list_exposed_profile_models()
        assert [p["model_id"] for p in exposed] == ["qwen-base:reasoning"]
        assert mgr.get_exposed_profile_source_model_id("qwen-base:thinking") is None
        assert (
            mgr.get_exposed_profile_source_model_id("qwen-base:reasoning")
            == "qwen-base"
        )

    def test_delete_profile_removes_exposed_model(self, mgr):
        _save_exposed_profile(mgr)

        mgr.delete_profile("qwen-base", "thinking")

        assert mgr.list_exposed_profile_models() == []
        assert mgr.get_exposed_profile_source_model_id("qwen-base:thinking") is None

    def test_exposed_profile_rejects_alias_collision(self, mgr):
        mgr.set_settings("other", ModelSettings(model_alias="qwen-base:thinking"))

        with pytest.raises(ValueError, match="conflicts with model alias"):
            _save_exposed_profile(mgr)


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
        assert (
            mgr.get_exposed_profile_source_model_id("omlx/qwen-base:thinking")
            == "qwen-base"
        )

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

    def test_runtime_settings_include_engine_fields_without_mutating_base(self, mgr):
        mgr.set_settings(
            "qwen-base",
            ModelSettings(temperature=0.2, mtp_enabled=False),
        )
        _save_exposed_profile(
            mgr,
            settings={"temperature": 0.9, "mtp_enabled": True},
        )

        runtime = mgr.get_exposed_profile_runtime_settings_for_request(
            "qwen-base:thinking"
        )

        assert runtime is not None
        source_model_id, settings = runtime
        assert source_model_id == "qwen-base"
        assert settings.temperature == 0.9
        assert settings.mtp_enabled is True
        assert mgr.get_settings("qwen-base").temperature == 0.2
        assert mgr.get_settings("qwen-base").mtp_enabled is False

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
