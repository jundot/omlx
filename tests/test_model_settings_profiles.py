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
    @pytest.mark.parametrize(
        "scheme,bits", [("affine4", 4), ("affine8", 8)]
    )
    def test_affine_profile_save_load_and_apply(self, tmp_path, scheme, bits):
        manager = ModelSettingsManager(tmp_path)
        settings = {"turboquant_kv_scheme": scheme, "turboquant_kv_bits": bits}
        manager.save_profile("m", "affine", scheme.title(), None, settings)
        manager = ModelSettingsManager(tmp_path)
        assert manager.get_profile("m", "affine")["settings"] == settings
        applied = manager.apply_profile("m", "affine")
        assert applied.turboquant_kv_scheme == scheme
        assert applied.turboquant_kv_bits == bits

    def test_invalid_profile_update_preserves_saved_profile(self, mgr):
        mgr.save_profile("m", "kv", "KV", None, {"turboquant_kv_scheme": "affine4"})
        original = mgr.get_profile("m", "kv")
        with pytest.raises(ValueError, match="affine4 requires"):
            mgr.update_profile("m", "kv", settings={
                "turboquant_kv_scheme": "affine4", "turboquant_kv_bits": 3,
            })
        assert mgr.get_profile("m", "kv") == original

    def test_partial_profile_validates_against_base_settings(self, mgr):
        mgr.save_profile("m", "bits", "Bits", None, {"turboquant_kv_bits": 3})
        mgr.set_settings("m", ModelSettings(turboquant_kv_scheme="affine4"))
        with pytest.raises(ValueError, match="affine4 requires"):
            mgr.apply_profile("m", "bits")
        assert mgr.get_settings("m").turboquant_kv_scheme == "affine4"
        assert mgr.get_settings("m").turboquant_kv_bits == 4
        with pytest.raises(ValueError, match="affine4 requires"):
            mgr.save_profile("m", "bad", "Bad", None, {"turboquant_kv_bits": 3})

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

    @pytest.mark.parametrize("bits", [4, 8])
    def test_legacy_turboquant_profiles_gain_explicit_scheme(self, tmp_path, bits):
        profiles_file = tmp_path / "model_profiles.json"
        profiles_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "m": {
                            "legacy": {
                                "name": "legacy",
                                "display_name": "Legacy TurboQuant",
                                "created_at": "2026-06-01T00:00:00+00:00",
                                "updated_at": "2026-06-01T00:00:00+00:00",
                                "settings": {
                                    "turboquant_kv_enabled": True,
                                    "turboquant_kv_bits": bits,
                                },
                            }
                        }
                    },
                }
            )
        )

        manager = ModelSettingsManager(tmp_path)

        assert manager.get_profile("m", "legacy")["settings"] == {
            "turboquant_kv_enabled": True,
            "turboquant_kv_bits": bits,
            "turboquant_kv_scheme": "turboquant",
        }
        persisted = json.loads(profiles_file.read_text())
        assert (
            persisted["profiles"]["m"]["legacy"]["settings"][
                "turboquant_kv_scheme"
            ]
            == "turboquant"
        )

    def test_sampling_only_profile_does_not_gain_kv_scheme(self, tmp_path):
        profiles_file = tmp_path / "model_profiles.json"
        profiles_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "m": {
                            "sampling": {
                                "name": "sampling",
                                "display_name": "Sampling",
                                "created_at": "2026-06-01T00:00:00+00:00",
                                "updated_at": "2026-06-01T00:00:00+00:00",
                                "settings": {"temperature": 0.2},
                            }
                        }
                    },
                }
            )
        )

        manager = ModelSettingsManager(tmp_path)

        assert manager.get_profile("m", "sampling")["settings"] == {
            "temperature": 0.2
        }

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

    def test_apply_resolves_vlm_mtp_processor_conflict(self, tmp_path):
        manager = ModelSettingsManager(tmp_path)
        manager.set_settings(
            "m",
            ModelSettings(
                vlm_mtp_enabled=True,
                vlm_mtp_draft_model="qwen-mtp-drafter",
            ),
        )
        manager.save_profile(
            "m",
            "penalty",
            "Penalty",
            None,
            {"presence_penalty": 1.5},
        )

        applied = manager.apply_profile("m", "penalty")

        assert applied is not None
        assert applied.presence_penalty == 1.5
        assert applied.vlm_mtp_enabled is False
        assert applied.active_profile_name == "penalty"

        persisted = ModelSettingsManager(tmp_path).get_settings("m")
        assert persisted.presence_penalty == 1.5
        assert persisted.vlm_mtp_enabled is False
        assert persisted.active_profile_name == "penalty"

    @pytest.mark.parametrize("bits", [4, 8])
    def test_apply_legacy_turboquant_profile_overrides_affine_base(
        self, tmp_path, bits
    ):
        profiles_file = tmp_path / "model_profiles.json"
        profiles_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "m": {
                            "legacy": {
                                "name": "legacy",
                                "display_name": "Legacy TurboQuant",
                                "created_at": "2026-06-01T00:00:00+00:00",
                                "updated_at": "2026-06-01T00:00:00+00:00",
                                "settings": {
                                    "turboquant_kv_enabled": True,
                                    "turboquant_kv_bits": bits,
                                },
                            }
                        }
                    },
                }
            )
        )
        manager = ModelSettingsManager(tmp_path)
        manager.set_settings(
            "m",
            ModelSettings(
                turboquant_kv_enabled=True,
                turboquant_kv_scheme="affine4",
                turboquant_kv_bits=4,
            ),
        )

        applied = manager.apply_profile("m", "legacy")

        assert applied.turboquant_kv_scheme == "turboquant"
        assert applied.turboquant_kv_bits == bits

    def test_sampling_only_profile_inherits_affine_engine_settings(self, mgr):
        mgr.set_settings(
            "m",
            ModelSettings(
                turboquant_kv_enabled=True,
                turboquant_kv_scheme="affine8",
                turboquant_kv_bits=8,
            ),
        )
        mgr.save_profile("m", "sampling", "Sampling", None, {"temperature": 0.2})

        applied = mgr.apply_profile("m", "sampling")

        assert applied.turboquant_kv_enabled is True
        assert applied.turboquant_kv_scheme == "affine8"
        assert applied.turboquant_kv_bits == 8

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

    def test_qwen_ane_prefill_fields_round_trip_as_model_specific(self, mgr):
        settings = {
            "qwen35_ane_prefill_enabled": True,
            "qwen35_ane_prefill_sequence_length": 2048,
            "qwen35_ane_prefill_tail_padding_min_tokens": 1357,
            "qwen35_ane_prefill_fraction": 0.53,
            "qwen35_ane_prefill_max_layers": 64,
            "qwen35_ane_prefill_dual_ane": True,
            "qwen35_ane_prefill_gdn": True,
            "qwen35_ane_prefill_gdn_fraction": 0.50,
            "qwen35_ane_prefill_gdn_max_layers": 48,
        }

        mgr.save_profile("m", "ane", "ANE", None, settings)
        assert mgr.get_profile("m", "ane")["settings"] == settings

        mgr.apply_profile("m", "ane")
        applied = mgr.get_settings("m")
        assert applied.qwen35_ane_prefill_enabled is True
        assert applied.qwen35_ane_prefill_sequence_length == 2048
        assert applied.qwen35_ane_prefill_tail_padding_min_tokens == 1357
        assert applied.qwen35_ane_prefill_fraction == 0.53
        assert applied.qwen35_ane_prefill_max_layers == 64
        assert applied.qwen35_ane_prefill_dual_ane is True
        assert applied.qwen35_ane_prefill_gdn is True
        assert applied.qwen35_ane_prefill_gdn_fraction == 0.50
        assert applied.qwen35_ane_prefill_gdn_max_layers == 48

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

    def test_legacy_turboquant_exposed_profile_overrides_affine_base(self, tmp_path):
        profiles_file = tmp_path / "model_profiles.json"
        profiles_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "qwen-base": {
                            "legacy": {
                                "name": "legacy",
                                "display_name": "Legacy TurboQuant",
                                "api_name": "legacy",
                                "created_at": "2026-06-01T00:00:00+00:00",
                                "updated_at": "2026-06-01T00:00:00+00:00",
                                "settings": {
                                    "turboquant_kv_enabled": True,
                                    "turboquant_kv_bits": 8,
                                },
                                "expose_as_model": True,
                            }
                        }
                    },
                }
            )
        )
        manager = ModelSettingsManager(tmp_path)
        manager.set_settings(
            "qwen-base",
            ModelSettings(
                turboquant_kv_enabled=True,
                turboquant_kv_scheme="affine4",
                turboquant_kv_bits=4,
            ),
        )

        runtime = manager.get_exposed_profile_runtime_settings_for_request(
            "qwen-base:legacy"
        )
        listed = manager.list_exposed_profile_models()[0]

        assert runtime is not None
        _, settings = runtime
        assert settings.turboquant_kv_scheme == "turboquant"
        assert settings.turboquant_kv_bits == 8
        assert listed["settings"]["turboquant_kv_scheme"] == "affine4"
        assert manager.list_profiles("qwen-base")[0]["has_engine_fields"] is True

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


class TestTemplateApplication:
    def test_latest_template_reuses_copy_and_preserves_engine_settings(self, mgr):
        mgr.set_settings("m", ModelSettings(mtp_enabled=True))
        mgr.save_template("coding", "Coding", None, {"temperature": 0.2, "top_p": 0.8})
        first = mgr.apply_template("m", "coding")
        mgr.update_template("coding", settings={"temperature": 0.9})
        second = mgr.apply_template("m", "coding")
        assert first.active_profile_name == second.active_profile_name
        assert second.temperature == 0.9
        assert second.top_p is None
        assert second.mtp_enabled is True
        assert len(mgr.list_profiles("m")) == 1

    def test_renamed_copy_keeps_identity_and_exposure(self, mgr, tmp_path):
        mgr.save_template("coding", "Coding", None, {"temperature": 0.2})
        mgr.apply_template("m", "coding")
        mgr.update_profile("m", "coding", new_name="copy", expose_as_model=True)
        before = mgr.get_profile("m", "copy")
        mgr.update_template("coding", new_name="renamed", settings={"temperature": 0.9})
        applied = mgr.apply_template("m", "renamed")
        assert applied.active_profile_name == "copy"
        assert applied.temperature == 0.9
        copy = mgr.get_profile("m", "copy")
        assert copy["source_template"] == "renamed"
        assert copy["api_name"] == before["api_name"]
        assert copy["expose_as_model"] is True
        restored = ModelSettingsManager(tmp_path)
        assert restored.get_profile("m", "copy") == copy
        assert restored.get_settings("m").temperature == 0.9

    def test_delete_detaches_copies_without_changing_runtime(self, mgr, tmp_path):
        mgr.save_template("coding", "Coding", None, {"temperature": 0.2})
        mgr.apply_template("m", "coding")
        mgr.apply_template("other", "coding")
        mgr.delete_template("coding")
        restored = ModelSettingsManager(tmp_path)
        for model in ("m", "other"):
            assert restored.get_profile(model, "coding")["source_template"] is None
            assert restored.get_settings(model).temperature == 0.2
            assert restored.get_settings(model).active_profile_name == "coding"
        assert restored.apply_template("m", "coding") is None

    def test_rejected_settings_do_not_create_or_update_copy(self, mgr):
        mgr.save_template("coding", "Coding", None, {"temperature": 0.2})

        def reject(settings):
            raise ValueError("Rejected")

        with pytest.raises(ValueError, match="Rejected"):
            mgr.apply_template("m", "coding", settings_sanitizer=reject)
        assert mgr.list_profiles("m") == []
        mgr.apply_template("m", "coding")
        original = mgr.get_profile("m", "coding")
        mgr.update_template("coding", settings={"temperature": 0.9})
        with pytest.raises(ValueError, match="Rejected"):
            mgr.apply_template("m", "coding", settings_sanitizer=reject)
        assert mgr.get_profile("m", "coding") == original
        assert mgr.get_settings("m").temperature == 0.2

    @pytest.mark.parametrize("operation", ["apply", "rename", "delete"])
    def test_second_file_write_failure_rolls_back(
        self, mgr, tmp_path, monkeypatch, operation
    ):
        mgr.save_template("coding", "Coding", None, {"temperature": 0.2})
        mgr.apply_template("m", "coding")
        mgr.update_template("coding", settings={"temperature": 0.9})
        before = mgr.get_profile("m", "coding")

        def fail():
            raise OSError("Disk full")

        monkeypatch.setattr(
            mgr, "_save" if operation == "apply" else "_save_templates", fail
        )
        with pytest.raises(OSError, match="Disk full"):
            if operation == "apply":
                mgr.apply_template("m", "coding")
            elif operation == "rename":
                mgr.update_template("coding", new_name="new")
            else:
                mgr.delete_template("coding")
        restored = ModelSettingsManager(tmp_path)
        for instance in (mgr, restored):
            assert instance.get_profile("m", "coding") == before
            assert instance.get_template("coding") is not None
            assert instance.get_template("new") is None
            assert instance.get_settings("m").temperature == 0.2


def _write_legacy_profile_storage(tmp_path):
    mgr = ModelSettingsManager(tmp_path)
    mgr.save_template("coding", "Coding", None, {"temperature": 0.9})
    mgr.save_profile("m", "coding", "Coding", None, {"temperature": 0.1})
    mgr.save_profile(
        "m",
        "orphan",
        "Orphan",
        None,
        {"temperature": 0.3},
        source_template="deleted",
        expose_as_model=True,
    )
    mgr.save_profile(
        "other", "copy", "Copy", None, {"temperature": 0.2}, source_template="coding"
    )
    mgr.set_settings("m", ModelSettings(active_profile_name="missing", temperature=0.4))
    mgr.set_settings(
        "other", ModelSettings(active_profile_name="copy", temperature=0.2)
    )
    documents = {
        path.name: json.loads(path.read_bytes())
        for path in (mgr.settings_file, mgr.profiles_file, mgr.templates_file)
    }
    documents["model_settings.json"]["models"]["m"]["unknown"] = 7
    documents["model_profiles.json"]["profiles"]["m"]["orphan"]["unknown"] = 8
    for name, document in documents.items():
        (tmp_path / name).write_text(json.dumps(document))
    return documents


def test_startup_repairs_only_missing_references_with_exact_backups(tmp_path):
    expected = _write_legacy_profile_storage(tmp_path)
    originals = {name: (tmp_path / name).read_bytes() for name in expected}
    mgr = ModelSettingsManager(tmp_path)
    expected["model_profiles.json"]["profiles"]["m"]["orphan"]["source_template"] = None
    expected["model_settings.json"]["models"]["m"]["active_profile_name"] = None
    for name, document in expected.items():
        assert json.loads((tmp_path / name).read_bytes()) == document
    assert mgr.get_settings("m").temperature == 0.4
    assert mgr.get_settings("other").active_profile_name == "copy"
    assert mgr.get_profile("other", "copy")["settings"]["temperature"] == 0.2
    backups = list(tmp_path.glob("profile-reference-backup-*"))
    assert len(backups) == 1
    for name, content in originals.items():
        assert (backups[0] / name).read_bytes() == content

    snapshots = {
        name: ((tmp_path / name).read_bytes(), (tmp_path / name).stat().st_mtime_ns)
        for name in expected
    }
    ModelSettingsManager(tmp_path)
    assert list(tmp_path.glob("profile-reference-backup-*")) == backups
    for name, snapshot in snapshots.items():
        assert (
            (tmp_path / name).read_bytes(),
            (tmp_path / name).stat().st_mtime_ns,
        ) == snapshot


def test_fresh_storage_does_not_create_repair_files(tmp_path):
    ModelSettingsManager(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "filename", ["model_settings.json", "model_profiles.json", "global_templates.json"]
)
@pytest.mark.parametrize("damage", ["invalid_json", "unknown_version"])
def test_unreadable_storage_does_not_trigger_reference_repair(
    tmp_path, filename, damage, caplog
):
    documents = _write_legacy_profile_storage(tmp_path)
    path = tmp_path / filename
    if damage == "invalid_json":
        path.write_text("{")
    else:
        document = documents[filename]
        document["version"] = 999
        path.write_text(json.dumps(document))
    originals = {name: (tmp_path / name).read_bytes() for name in documents}
    ModelSettingsManager(tmp_path)
    assert "Skipped profile reference repair" in caplog.text
    assert not list(tmp_path.glob("profile-reference-backup-*"))
    for name, content in originals.items():
        assert (tmp_path / name).read_bytes() == content


@pytest.mark.parametrize("failure", ["backup", "second_write", "corrupt_backup"])
def test_reference_repair_failure_retains_original_documents(
    tmp_path, monkeypatch, failure, caplog
):
    documents = _write_legacy_profile_storage(tmp_path)
    originals = {name: (tmp_path / name).read_bytes() for name in documents}
    write = ModelSettingsManager._write_profile_repair

    def fail_write(path, content):
        in_backup = path.parent != tmp_path
        if path.name == "model_profiles.json" and in_backup and failure == "backup":
            raise OSError("Backup unavailable")
        if (
            path.name == "model_settings.json"
            and not in_backup
            and failure == "second_write"
        ):
            raise OSError("Settings write unavailable")
        if in_backup and failure == "corrupt_backup":
            content = b"corrupt backup"
        write(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(
            ModelSettingsManager, "_write_profile_repair", staticmethod(fail_write)
        )
        ModelSettingsManager(tmp_path)
        backup = next(tmp_path.glob("profile-reference-backup-*"))
        first_backup = backup / "model_settings.json"
        modified_at = first_backup.stat().st_mtime_ns
        ModelSettingsManager(tmp_path)
        assert list(tmp_path.glob("profile-reference-backup-*")) == [backup]
        assert first_backup.stat().st_mtime_ns == modified_at
        assert "Profile reference repair failed" in caplog.text
        for name, content in originals.items():
            assert (tmp_path / name).read_bytes() == content

    if failure == "corrupt_backup":
        ModelSettingsManager(tmp_path)
        assert first_backup.read_bytes() == b"corrupt backup"
        assert list(tmp_path.glob("profile-reference-backup-*")) == [backup]
        for name, content in originals.items():
            assert (tmp_path / name).read_bytes() == content
        return

    ModelSettingsManager(tmp_path)
    assert list(tmp_path.glob("profile-reference-backup-*")) == [backup]
    assert first_backup.stat().st_mtime_ns == modified_at
    for name, content in originals.items():
        assert (backup / name).read_bytes() == content
    repaired = json.loads((tmp_path / "model_settings.json").read_bytes())
    assert repaired["models"]["m"]["active_profile_name"] is None


def test_reference_repair_rollback_failure_reports_backup(tmp_path, monkeypatch, caplog):
    documents = _write_legacy_profile_storage(tmp_path)
    originals = {name: (tmp_path / name).read_bytes() for name in documents}
    write = ModelSettingsManager._write_profile_repair

    def fail_write(path, content):
        if path.parent == tmp_path:
            if path.name == "model_settings.json":
                raise OSError("Settings write unavailable")
            if content == originals[path.name]:
                raise OSError("Rollback unavailable")
        write(path, content)

    monkeypatch.setattr(
        ModelSettingsManager, "_write_profile_repair", staticmethod(fail_write)
    )
    with pytest.raises(OSError, match="Rollback unavailable"):
        ModelSettingsManager(tmp_path)
    backup = next(tmp_path.glob("profile-reference-backup-*"))
    assert str(backup) in caplog.text
    for name, content in originals.items():
        assert (backup / name).read_bytes() == content
