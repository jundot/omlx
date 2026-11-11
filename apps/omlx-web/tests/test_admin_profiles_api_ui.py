# SPDX-License-Identifier: Apache-2.0
"""Dashboard wiring for model profiles and settings snapshot actions."""

import json
from pathlib import Path


def test_dashboard_profile_ui_round_trips_expose_as_model_flag():
    root = Path(__file__).resolve().parents[1]
    js = (root / "omlx_web/static/js/dashboard.js").read_text()
    html = (
        root / "omlx_web/templates/dashboard/_modal_model_settings.html"
    ).read_text()
    settings_html = (root / "omlx_web/templates/dashboard/_settings.html").read_text()
    dashboard_html = (root / "omlx_web/templates/dashboard.html").read_text()
    en = (root / "omlx_web/i18n/en.json").read_text()

    # api_name is the exposed model ID suffix, so it's validated as a
    # slug and rejected on bad input.
    assert "isValidProfileName" in js
    assert "api_name" in js
    assert "modal.model_settings.profiles.invalid_name" in en
    # Profile chips keep the display name visible in both API-on and
    # API-off states. API state is shown as a badge and edited from the
    # edit form, not toggled directly from the chip row.
    assert "profileTooltip(p)" in html
    assert "p.display_name || p.name" in html
    assert "p.expose_as_model ? (p.api_name || p.name)" not in html
    assert "expose_as_model: !p.expose_as_model" not in html
    assert "p.has_engine_fields" in html
    # Editing updates display_name/api_name/description/exposure without
    # renaming the internal profile key.
    assert "updateProfileFromEdit" in js
    assert "api_name: apiName" in js
    assert "description: description" in js
    assert "expose_as_model: exposeAsModel" in js
    edit_method = js.split("updateProfileFromEdit(p) {", 1)[1].split(
        "updateProfileSettingsFromForm(p)", 1
    )[0]
    assert "settings:" not in edit_method
    assert "updateProfileSettingsFromForm(p)" in js
    assert "updateProfileFromEdit(p)" in html
    assert "updateProfileSettingsFromForm(p)" in html
    assert "_editDescription" in html
    assert "_editExposeAsModel" in html
    assert "profileTooltip(profile)" in settings_html
    assert "model.exposed_profiles" in settings_html
    assert "profile.api_name || profile.name" in settings_html
    assert settings_html.index("profile.has_engine_fields") < settings_html.index(
        "profile.api_name || profile.name"
    )
    assert 'x-text="model.settings.active_profile_name"' not in settings_html
    assert "profileTooltip" in js
    assert "whitespace-pre-line" in dashboard_html
    assert "modal.model_settings.profiles.expose_as_model" in html
    assert "modal.model_settings.profiles.exposed_as" in en
    assert "modal.model_settings.profiles.expose_engine_fields_hint" in en


def test_dashboard_wires_snapshot_actions():
    root = Path(__file__).resolve().parents[1]
    js = (root / "omlx_web/static/js/dashboard.js").read_text()
    modal = (
        root / "omlx_web/templates/dashboard/_modal_model_settings.html"
    ).read_text()
    apply_modal = (
        root / "omlx_web/templates/dashboard/_modal_settings_apply.html"
    ).read_text()
    dashboard = (root / "omlx_web/templates/dashboard.html").read_text()
    en = json.loads((root / "omlx_web/i18n/en.json").read_text())
    ko = json.loads((root / "omlx_web/i18n/ko.json").read_text())

    assert "dashboard/_modal_settings_apply.html" in dashboard
    for mode in ("reset", "optimal", "recipe"):
        assert f"openSettingsApply('{mode}')" in modal
    assert "/settings/${path}`" in js
    assert "_settingsActionRequest('POST', 'reset')" in js
    assert "_settingsActionRequest('GET', 'optimal')" in js
    assert "_settingsActionRequest('POST', 'recipe'" in js
    assert "applyOptimalCandidate(item.benchmark_id)" in apply_modal
    assert "by_pp" in js and "by_tg" in js
    for key in (
        "modal.model_settings.actions.reset_confirm",
        "modal.model_settings.actions.group_pp",
        "modal.model_settings.actions.group_tg",
        "modal.model_settings.actions.none_body",
        "js.error.settings_apply_failed",
    ):
        assert key in en and key in ko
