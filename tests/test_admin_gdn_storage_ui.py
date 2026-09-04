import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GDN_I18N_KEYS = {
    "settings.advanced.gdn_cache_policy",
    "settings.advanced.gdn_cache_policy_hint",
    "settings.advanced.gdn_snapshot_storage",
    "settings.advanced.gdn_snapshot_storage_hint",
    "settings.advanced.gdn_snapshot_storage_auto",
    "settings.advanced.gdn_snapshot_storage_ssd",
    "settings.advanced.gdn_snapshot_storage_embedded",
    "settings.advanced.gdn_pending_write_limit",
    "settings.advanced.gdn_pending_write_limit_hint",
    "settings.advanced.gdn_snapshot_state_precision",
    "settings.advanced.gdn_snapshot_state_precision_hint",
    "settings.advanced.gdn_snapshot_state_precision_warning",
    "settings.advanced.gdn_snapshot_state_precision_rht_int16",
    "settings.advanced.gdn_snapshot_state_precision_fp32",
    "settings.advanced.gdn_snapshot_state_precision_bf16",
    "settings.advanced.gdn_snapshot_state_precision_int8",
    "settings.advanced.gdn_snapshot_state_precision_rht_int8",
}


def test_settings_template_exposes_gdn_storage_policy_and_codecs():
    template = (ROOT / "omlx/admin/templates/dashboard/_settings.html").read_text()

    advanced_start = template.index("<!-- Advanced Settings")
    gdn_policy_start = template.index("settings.advanced.gdn_cache_policy")

    assert gdn_policy_start > advanced_start
    assert 'value="auto"' in template
    assert 'value="ssd_sidecar"' in template
    assert 'value="embedded"' in template
    assert 'value="rht_int16"' in template
    assert 'value="fp32"' in template
    assert 'value="rht_int8"' in template
    assert "globalSettings.cache.gdn_snapshot_state_dtype !== 'fp32'" in template
    assert "text-red-600" in template


def test_gdn_cache_policy_i18n_keys_exist_in_every_locale():
    i18n_dir = ROOT / "omlx/admin/i18n"

    for locale_path in sorted(i18n_dir.glob("*.json")):
        translations = json.loads(locale_path.read_text())
        missing_keys = GDN_I18N_KEYS - translations.keys()
        assert not missing_keys, f"{locale_path.name} is missing {sorted(missing_keys)}"


def test_dashboard_posts_canonical_gdn_fields_only():
    script = (ROOT / "omlx/admin/static/js/dashboard.js").read_text()
    payload_start = script.index("async saveGlobalSettings()")
    payload_end = script.index("async saveModelSettings()", payload_start)
    payload = script[payload_start:payload_end]

    assert "gdn_snapshot_storage:" in payload
    assert "gdn_snapshot_state_dtype:" in payload
    assert "gdn_ssd_pending_max_size:" in payload
    assert "gdn_ssd_split_enabled:" not in payload


def test_precision_hint_keeps_fp32_as_the_default_in_every_locale():
    """The rebased policy keeps fp32 as the persisted default (the v0.6.0
    lossy-default incident), so no locale may advertise RHT + int16 as the
    default."""
    i18n_dir = ROOT / "omlx/admin/i18n"
    forbidden = ("near-lossless default", "거의 무손실인 기본값")
    required_any = ("FP32가 정확한 기본값", "FP32 is the exact default")

    for locale_path in sorted(i18n_dir.glob("*.json")):
        translations = json.loads(locale_path.read_text())
        hint = translations["settings.advanced.gdn_snapshot_state_precision_hint"]
        for phrase in forbidden:
            assert phrase not in hint, f"{locale_path.name} claims a lossy default"
        assert any(
            phrase in hint for phrase in required_any
        ), f"{locale_path.name} does not name FP32 as the default"
