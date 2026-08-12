"""Regression tests for admin model-settings UI gates."""

import json
from pathlib import Path


def _model_settings_template() -> str:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "omlx/admin/templates/dashboard/_modal_model_settings.html"
    ).read_text()


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


def test_mid_prefill_toggle_is_nested_under_turboquant_parent() -> None:
    html = _model_settings_template()
    turboquant = _section(
        html,
        "<!-- TurboQuant KV Cache -->",
        "<!-- IndexCache (DSA models only) -->",
    )

    parent_gate = 'x-show="modelSettings.turboquant_kv_enabled"'
    child_key = "modelSettings.turboquant_mid_prefill"
    assert parent_gate in turboquant
    assert child_key in turboquant
    assert turboquant.index(parent_gate) < turboquant.index(child_key)
    assert 'id="turboquant-mid-prefill-label"' in turboquant
    assert 'role="switch"' in turboquant
    assert 'aria-labelledby="turboquant-mid-prefill-label"' in turboquant
    assert (
        ":aria-checked=\"modelSettings.turboquant_mid_prefill ? 'true' : 'false'\""
        in turboquant
    )


def test_mid_prefill_dashboard_state_payload_profile_and_reset_strings() -> None:
    root = Path(__file__).resolve().parents[1]
    js = (root / "omlx/admin/static/js/dashboard.js").read_text()

    assert "turboquant_mid_prefill: s.turboquant_mid_prefill || false" in js
    assert "out.turboquant_mid_prefill = !!ms.turboquant_kv_enabled" in js
    assert "&& !!ms.turboquant_mid_prefill;" in js
    assert "turboquant_mid_prefill: this.modelSettings.turboquant_kv_enabled" in js
    assert "&& !!this.modelSettings.turboquant_mid_prefill" in js
    assert "this.modelSettings.turboquant_mid_prefill = false;" in js
    assert "'turboquant_mid_prefill'," in js
    assert "this.profileFields.model_specific" in js


def test_all_live_locales_include_mid_prefill_copy_and_correct_parent_semantics() -> (
    None
):
    root = Path(__file__).resolve().parents[1]
    locale_paths = sorted((root / "omlx/admin/i18n").glob("*.json"))
    assert len(locale_paths) == 9

    for path in locale_paths:
        strings = json.loads(path.read_text())
        assert strings["modal.model_settings.turboquant_mid_prefill"]
        assert strings["modal.model_settings.turboquant_mid_prefill_hint"]
        parent_hint = strings["modal.model_settings.turboquant_kv_hint"].lower()
        assert "after prefill" in parent_hint
        assert "does not convert during cold prefill" in parent_hint

    english = json.loads((root / "omlx/admin/i18n/en.json").read_text())
    assert (
        english["modal.model_settings.turboquant_mid_prefill"]
        == "Convert under prefill pressure"
    )
    assert english["modal.model_settings.turboquant_mid_prefill_hint"] == (
        "When a full prefill chunk cannot fit, convert the growing KV cache once "
        "and continue with TurboQuant. Requires this to be the only loaded model. "
        "Adds a one-time pause and may slow the rest of prefill."
    )


def test_reasoning_effort_offers_max_after_high():
    html = _model_settings_template()

    high_option = '<option value="high">'
    max_option = '<option value="max">max</option>'

    assert high_option in html
    assert max_option in html
    assert html.index(high_option) < html.index(max_option)
