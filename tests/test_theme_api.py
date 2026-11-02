# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the external theme API."""

import asyncio
import json

from omlx.settings import GlobalSettings


def _state(response):
    return json.loads(response.body) if hasattr(response, "body") else response


def _apply(gs, monkeypatch, light=None, dark=None, revision=0):
    from omlx.admin import routes as admin_routes

    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
    request = admin_routes.ThemeApplyRequest(
        light=light or {}, dark=dark or {}, revision=revision
    )
    return asyncio.run(admin_routes.apply_theme(request, if_match=None, is_admin=True))


def test_schema_exposes_the_whole_contract():
    from omlx.admin.routes import get_theme_schema

    schema = asyncio.run(get_theme_schema(is_admin=True))
    assert schema["schema_version"] == 1
    assert len(schema["tokens"]) == 22
    assert {group["id"] for group in schema["groups"]} == {
        "surfaces",
        "text",
        "borders",
        "controls",
        "status",
        "accents",
    }
    rules = schema["contrast_rules"]
    assert any(rule["foreground"] == "text-primary" for rule in rules)
    assert schema["tokens"][0]["default"]["light"].startswith("#")


def test_validate_reports_structured_errors_without_saving(tmp_path, monkeypatch):
    from omlx.admin import routes as admin_routes

    gs = GlobalSettings(base_path=tmp_path)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
    result = asyncio.run(
        admin_routes.validate_theme(
            admin_routes.ThemePayload(
                light={"bg-primary": "#111111", "text-primary": "#222222"},
                dark={},
            ),
            is_admin=True,
        )
    )
    assert result["ok"] is False
    assert any(error["code"] == "contrast_below_floor" for error in result["errors"])
    assert any(error["surface"] == "bg-primary" for error in result["errors"])
    assert gs.ui.custom_theme is None


def test_put_is_revision_guarded_and_reset_clears(tmp_path, monkeypatch):
    from omlx.admin import routes as admin_routes

    gs = GlobalSettings(base_path=tmp_path)
    theme = {"light": {"bg-primary": "#fdf6e3"}, "dark": {"text-danger": "#f87171"}}
    first = _apply(gs, monkeypatch, **theme, revision=0)
    assert first["ok"] is True
    assert first["revision"] == 1
    assert gs.ui.custom_theme == theme
    assert GlobalSettings.load(base_path=tmp_path).ui.custom_theme == theme

    conflict = _apply(gs, monkeypatch, **theme, revision=0)
    assert conflict.status_code == 409
    assert _state(conflict)["error"]["code"] == "revision_conflict"

    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
    reset = asyncio.run(admin_routes.reset_theme(revision=1, is_admin=True))
    assert reset["ok"] is True
    assert reset["revision"] == 2
    assert gs.ui.custom_theme is None
    assert GlobalSettings.load(base_path=tmp_path).ui.custom_theme is None


def test_put_rejects_low_contrast_without_bumping_revision(tmp_path, monkeypatch):
    gs = GlobalSettings(base_path=tmp_path)
    response = _apply(
        gs,
        monkeypatch,
        light={"bg-primary": "#111111", "text-primary": "#222222"},
        revision=0,
    )
    assert response.status_code == 422
    assert _state(response)["errors"][0]["code"] == "contrast_below_floor"
    assert gs.ui.custom_theme_revision == 0


def test_put_refuses_translucent_and_non_string_colors(tmp_path, monkeypatch):
    gs = GlobalSettings(base_path=tmp_path)
    response = _apply(
        gs, monkeypatch, light={"text-primary": "#00000000", "link-color": 7}
    )
    assert response.status_code == 422
    errors = _state(response)["errors"]
    assert [error["code"] for error in errors] == ["invalid_hex", "invalid_hex"]
    assert gs.ui.custom_theme is None


def test_failed_save_restores_the_theme_and_revision(tmp_path, monkeypatch):
    from omlx.admin import routes as admin_routes

    gs = GlobalSettings(base_path=tmp_path)

    async def fail(_settings):
        raise OSError("disk full")

    monkeypatch.setattr(admin_routes, "_save_global_settings_async", fail)
    try:
        _apply(gs, monkeypatch, light={"bg-primary": "#fdf6e3"})
    except admin_routes.HTTPException as exc:
        assert exc.status_code == 500
    else:
        raise AssertionError("save failure was not reported")
    assert gs.ui.custom_theme is None
    assert gs.ui.custom_theme_revision == 0


def test_draft_apply_and_undo(tmp_path, monkeypatch):
    from omlx.admin import routes as admin_routes

    gs = GlobalSettings(base_path=tmp_path)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
    draft = asyncio.run(
        admin_routes.create_theme_draft(
            admin_routes.ThemeDraftRequest(
                light={"bg-primary": "#fdf6e3"},
                dark={},
                note="warm paper",
                source="agent",
            ),
            is_admin=True,
        )
    )
    assert draft["ok"] is True
    assert draft["draft"]["base_revision"] == 0
    assert draft["draft"]["note"] == "warm paper"
    assert gs.ui.custom_theme is None

    applied = asyncio.run(
        admin_routes.apply_theme_draft(
            admin_routes.ThemeRevisionRequest(revision=0), is_admin=True
        )
    )
    assert applied["ok"] is True
    assert applied["revision"] == 1
    assert gs.ui.custom_theme == {"light": {"bg-primary": "#fdf6e3"}, "dark": {}}
    assert gs.ui.custom_theme_draft is None

    undone = asyncio.run(
        admin_routes.undo_theme(
            admin_routes.ThemeRevisionRequest(revision=1), is_admin=True
        )
    )
    assert undone["ok"] is True
    assert undone["revision"] == 2
    assert gs.ui.custom_theme is None
    assert gs.ui.custom_theme_history == []


def test_draft_discard_and_stale_base(tmp_path, monkeypatch):
    from omlx.admin import routes as admin_routes

    gs = GlobalSettings(base_path=tmp_path)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
    asyncio.run(
        admin_routes.create_theme_draft(
            admin_routes.ThemeDraftRequest(light={"bg-primary": "#fdf6e3"}),
            is_admin=True,
        )
    )
    gs.ui.custom_theme_revision = 1
    response = asyncio.run(
        admin_routes.apply_theme_draft(
            admin_routes.ThemeRevisionRequest(revision=1), is_admin=True
        )
    )
    assert response.status_code == 409

    discarded = asyncio.run(admin_routes.delete_theme_draft(is_admin=True))
    assert discarded["ok"] is True
    assert gs.ui.custom_theme_draft is None


def test_recipe_expands_seeds_to_a_valid_theme():
    from omlx.admin import routes as admin_routes
    from omlx.theme_contract import CUSTOM_THEME_TOKENS

    result = asyncio.run(
        admin_routes.expand_theme_recipe(
            admin_routes.ThemeRecipeRequest(
                light={
                    "background": "#fdf6e3",
                    "surface": "#f7efd9",
                    "text": "#073642",
                    "accent": "#cb4b16",
                },
                dark={
                    "background": "#002b36",
                    "surface": "#073642",
                    "text": "#eee8d5",
                    "accent": "#268bd2",
                },
            ),
            is_admin=True,
        )
    )
    assert result["ok"] is True
    assert set(result["tokens"]["light"]) == set(CUSTOM_THEME_TOKENS)
    assert all(item["ok"] for item in result["contrast"] if item["blocking"])


def test_recipe_borders_stay_close_to_the_background():
    from omlx.theme_contract import contrast_ratio
    from omlx.theme_recipe import build_mode

    tokens = build_mode(
        {"background": "#ffffff", "text": "#111111", "accent": "#2563eb"}, "light"
    )
    # Faint borders are a hairline, not a second text color.
    assert contrast_ratio(tokens["border-faint"], "#ffffff") < 1.5
    assert contrast_ratio(tokens["border-faint"], "#ffffff") < contrast_ratio(
        tokens["border-normal"], "#ffffff"
    )


def test_contrast_errors_suggest_a_passing_color(tmp_path, monkeypatch):
    from omlx.admin import routes as admin_routes
    from omlx.theme_contract import contrast_ratio

    result = asyncio.run(
        admin_routes.validate_theme(
            admin_routes.ThemePayload(light={"link-color": "#9db8f0"}),
            is_admin=True,
        )
    )
    errors = [e for e in result["errors"] if e["code"] == "contrast_below_floor"]
    assert errors
    for error in errors:
        surface = result["resolved"]["light"][error["surface"]]
        assert contrast_ratio(error["suggestion"]["value"], surface) >= 4.5


def test_draft_with_low_contrast_is_refused(tmp_path, monkeypatch):
    from omlx.admin import routes as admin_routes

    gs = GlobalSettings(base_path=tmp_path)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)
    response = asyncio.run(
        admin_routes.create_theme_draft(
            admin_routes.ThemeDraftRequest(light={"text-secondary": "#bbbbbb"}),
            is_admin=True,
        )
    )
    assert response.status_code == 422
    assert gs.ui.custom_theme_draft is None
