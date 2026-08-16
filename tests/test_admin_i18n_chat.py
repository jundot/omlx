"""Regression tests for admin chat localization coverage."""

import json
from pathlib import Path

from omlx.admin import routes as admin_routes


def _chat_template() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "omlx/admin/templates/chat.html").read_text()


def test_locale_loader_uses_english_fallback_for_missing_keys(tmp_path, monkeypatch):
    (tmp_path / "en.json").write_text(
        json.dumps(
            {
                "chat.model_tab": "Model",
                "chat.future_key": "English fallback",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ru.json").write_text(
        json.dumps({"chat.model_tab": "Модель"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(admin_routes, "_i18n_dir", tmp_path)
    monkeypatch.setattr(
        admin_routes,
        "_en_locale",
        {
            "chat.model_tab": "Model",
            "chat.future_key": "English fallback",
        },
    )

    locale = admin_routes._load_locale("ru")

    assert locale["chat.model_tab"] == "Модель"
    assert locale["chat.future_key"] == "English fallback"


def test_chat_sidebar_model_settings_use_i18n_keys():
    html = _chat_template()

    expected_keys = [
        "chat.model_tab",
        "chat.profile_tab",
        "chat.active_profile",
        "chat.active_model",
        "modal.model_settings.temperature",
        "modal.model_settings.max_tokens",
        "modal.model_settings.top_p",
        "modal.model_settings.repetition_penalty_short",
        "chat.thinking_mode.on_limited",
        "chat.model_settings_advanced_hint",
        "chat.stats.token_generation",
        "chat.status.prefilling_percent",
        "chat.web_search_on",
        "chat.web_search_off",
        "chat.max_tool_rounds",
        "chat.max_tool_rounds_hint",
        "chat.error.max_tool_rounds",
        "chat.status.searching_web",
        "chat.status.fetching_page",
    ]
    for key in expected_keys:
        assert key in html

    hardcoded = [
        ">MODEL<",
        ">PROFILE<",
        ">Save Setting<",
        ">Active Profile<",
        ">Active Model<",
        ">Temperature<",
        ">Max Tokens<",
        ">Rep Penalty<",
        ">Pres Penalty<",
        ">Thinking tokens<",
        ">Token Gen (t/s)<",
        ">Duration (s)<",
        'placeholder="Enter instructions for this session..."',
        'placeholder="Profile name..."',
    ]
    for literal in hardcoded:
        assert literal not in html
