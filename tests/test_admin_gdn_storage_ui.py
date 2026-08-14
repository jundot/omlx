from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_settings_template_exposes_gdn_storage_policy_and_codecs():
    template = (
        ROOT / "omlx/admin/templates/dashboard/_settings.html"
    ).read_text()

    assert 'value="auto"' in template
    assert 'value="ssd_sidecar"' in template
    assert 'value="embedded"' in template
    assert 'value="rht_int16"' in template
    assert 'value="fp32"' in template
    assert 'value="rht_int8"' in template


def test_dashboard_posts_canonical_gdn_fields_only():
    script = (ROOT / "omlx/admin/static/js/dashboard.js").read_text()
    payload_start = script.index("async saveGlobalSettings()")
    payload_end = script.index("async saveModelSettings()", payload_start)
    payload = script[payload_start:payload_end]

    assert "gdn_snapshot_storage:" in payload
    assert "gdn_sidecar_state_dtype:" in payload
    assert "gdn_ssd_pending_max_size:" in payload
    assert "gdn_ssd_split_enabled:" not in payload
