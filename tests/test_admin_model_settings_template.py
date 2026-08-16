"""Regression tests for admin model-settings UI gates."""

import json
from pathlib import Path


def _model_settings_template() -> str:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "omlx/admin/templates/dashboard/_modal_model_settings.html"
    ).read_text()


def _dashboard_script() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "omlx/admin/static/js/dashboard.js").read_text()


def _status_template() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "omlx/admin/templates/dashboard/_status.html").read_text()


def _section(html: str, start_marker: str, end_marker: str) -> str:
    return html.split(start_marker, 1)[1].split(end_marker, 1)[0]


def test_lightning_mtp_and_turboquant_are_not_ui_mutexed():
    html = _model_settings_template()

    turboquant = _section(
        html,
        "<!-- TurboQuant KV Cache -->",
        "<!-- IndexCache (DSA models only) -->",
    )
    lightning_mtp = _section(
        html,
        "<!-- Lightning MTP (built-in MTP head speculative decoding) -->",
        "<!-- Experimental Features -->",
    )

    assert "modelSettings.mtp_enabled" not in turboquant
    assert "modelSettings.turboquant_kv_enabled" not in lightning_mtp


def test_vlm_mtp_still_conflicts_with_turboquant():
    html = _model_settings_template()
    vlm_mtp = _section(
        html,
        "<!-- VLM MTP",
        "<!-- Performance",
    )

    assert "modelSettings.turboquant_kv_enabled" in vlm_mtp


def test_reasoning_effort_has_presets_and_custom_input():
    """Common strings stay convenient while model-specific values remain usable."""
    html = _model_settings_template()

    marker = "<template x-if=\"entry.type === 'reasoning_effort'\">"
    section = html.split(marker, 2)[2].split(
        "<template x-if=\"entry.type === 'enable_thinking'\">", 1
    )[0]

    assert 'x-show="!entry.custom"' in section
    assert 'x-show="entry.custom"' in section
    assert 'x-model="entry.customValue"' in section
    assert 'x-model="entry.custom"' in section
    assert 'x-model="entry.force"' in section
    assert 'class="flex items-center gap-3"' in section
    assert 'placeholder="0.9"' in section
    assert "<datalist" not in section

    order = ["low", "medium", "high", "xhigh", "max"]
    positions = [section.index(f'value="{value}"') for value in order]
    assert positions == sorted(positions)


def test_reasoning_effort_add_guard_covers_custom_entries():
    """A generic custom row cannot duplicate the dedicated effort key."""
    html = _model_settings_template()

    guard = (
        "e.type === 'reasoning_effort' || "
        "(e.type === 'custom' && e.key && e.key.trim() === 'reasoning_effort')"
    )
    assert guard in html


def test_reasoning_effort_reload_restores_preset_or_custom_editor():
    """Stored values must never fall through to a generic custom kwarg row."""
    script = _dashboard_script()

    branch = script.split("} else if (key === 'reasoning_effort') {", 1)[1].split(
        "} else {", 1
    )[0]
    assert "REASONING_EFFORT_PRESETS.has(value)" in branch
    assert "type: 'reasoning_effort'" in branch
    assert "value: isPreset ? value : 'low'" in branch
    assert "custom: !isPreset" in branch
    assert "customValue: isPreset ? '' : String(value)" in branch


def test_model_settings_feature_labels_use_i18n_keys():
    modal_html = _model_settings_template()
    status_html = _status_template()

    assert "{{ t('modal.model_settings.reasoning_parser') }}" in modal_html
    assert "{{ t('modal.model_settings.specprefill') }}" in modal_html
    assert "{{ t('modal.model_settings.dflash') }}" in modal_html
    assert "{{ t('status.active_models.dflash_label') }}" in status_html

    assert ">Reasoning Parser</label>" not in modal_html
    assert ">SpecPrefill</span>" not in modal_html
    assert ">DFlash</span>" not in modal_html
    assert ">DFlash</span>" not in status_html


def test_model_settings_feature_i18n_keys_exist_in_every_locale():
    root = Path(__file__).resolve().parents[1]
    i18n_dir = root / "omlx/admin/i18n"
    keys = {
        "modal.model_settings.reasoning_parser",
        "modal.model_settings.specprefill",
        "modal.model_settings.dflash",
        "status.active_models.dflash_label",
    }

    for locale_path in sorted(i18n_dir.glob("*.json")):
        translations = json.loads(locale_path.read_text())
        missing_keys = keys - translations.keys()
        assert not missing_keys, f"{locale_path.name} is missing {sorted(missing_keys)}"
