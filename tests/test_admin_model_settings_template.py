"""Regression tests for admin model-settings UI gates."""

import json
from pathlib import Path


def _model_settings_template() -> str:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "omlx/admin/templates/dashboard/_modal_model_settings.html"
    ).read_text()


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


def test_reasoning_effort_offers_max_after_high():
    html = _model_settings_template()

    high_option = '<option value="high">'
    max_option = '<option value="max">max</option>'

    assert high_option in html
    assert max_option in html
    assert html.index(high_option) < html.index(max_option)


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


def test_model_settings_feature_i18n_keys_exist_in_english_and_chinese():
    root = Path(__file__).resolve().parents[1]
    keys = {
        "modal.model_settings.reasoning_parser",
        "modal.model_settings.specprefill",
        "modal.model_settings.dflash",
        "status.active_models.dflash_label",
    }

    for language in ("en", "zh"):
        translations = json.loads(
            (root / f"omlx/admin/i18n/{language}.json").read_text()
        )
        missing_keys = keys - translations.keys()
        assert not missing_keys, f"{language}.json is missing {sorted(missing_keys)}"
