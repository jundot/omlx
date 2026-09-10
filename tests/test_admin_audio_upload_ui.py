import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_TEMPLATE = ROOT / "omlx/admin/templates/dashboard/_settings.html"
DASHBOARD_JS = ROOT / "omlx/admin/static/js/dashboard.js"

AUDIO_UPLOAD_I18N_KEYS = {
    "settings.advanced.uploads",
    "settings.advanced.max_audio_upload_size",
    "settings.advanced.max_audio_upload_size_hint",
}


def test_settings_template_exposes_audio_upload_limit_under_advanced_uploads():
    template = SETTINGS_TEMPLATE.read_text()

    advanced_start = template.index("<!-- Advanced Settings")
    uploads_start = template.index("settings.advanced.uploads")
    streaming_start = template.index("settings.advanced.streaming")

    assert advanced_start < uploads_start < streaming_start
    assert 'x-model.trim="globalSettings.server.max_audio_upload_size"' in template
    assert 'placeholder="100MB"' in template


def test_dashboard_defaults_and_posts_audio_upload_limit():
    script = DASHBOARD_JS.read_text()
    payload_start = script.index("async saveGlobalSettings()")
    payload_end = script.index("async saveModelSettings()", payload_start)
    payload = script[payload_start:payload_end]

    assert "max_audio_upload_size: '100MB'" in script
    assert "max_audio_upload_size:" in payload
    assert "Maximum Audio Upload Size" in payload


def test_audio_upload_i18n_keys_exist_in_every_locale():
    i18n_dir = ROOT / "omlx/admin/i18n"

    for locale_path in sorted(i18n_dir.glob("*.json")):
        translations = json.loads(locale_path.read_text())
        missing_keys = AUDIO_UPLOAD_I18N_KEYS - translations.keys()
        assert not missing_keys, f"{locale_path.name} is missing {sorted(missing_keys)}"
