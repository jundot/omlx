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


def test_apply_profile_surfaces_server_validation_error():
    script = _dashboard_script()
    method = script.split("async applyProfileToForm(profile) {", 1)[1].split(
        "async applyTemplateToForm(template) {", 1
    )[0]

    assert "this.profileError = '';" in method
    assert "const data = await r.json().catch(() => ({}));" in method
    assert "this.profileError = data.detail || 'Failed to apply profile';" in method
    assert "this.profileError = String(e);" in method


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
        "modal.model_settings.qwen_ane",
        "modal.model_settings.qwen_ane_hint",
        "modal.model_settings.qwen_ane_prompt_block",
        "modal.model_settings.qwen_ane_mlp_fraction",
        "modal.model_settings.qwen_ane_mlp_layers",
        "modal.model_settings.qwen_ane_dual",
        "modal.model_settings.qwen_ane_dual_hint",
        "modal.model_settings.qwen_ane_gdn",
        "modal.model_settings.qwen_ane_gdn_hint",
        "modal.model_settings.qwen_ane_gdn_fraction",
        "modal.model_settings.qwen_ane_gdn_layers",
        "modal.model_settings.qwen_ane_tune",
        "modal.model_settings.qwen_ane_tune_hint",
        "modal.model_settings.qwen_ane_tune_start",
        "modal.model_settings.qwen_ane_tune_again",
        "modal.model_settings.qwen_ane_tune_overrides",
        "modal.model_settings.qwen_ane_tune_allow_cpu",
        "modal.model_settings.qwen_ane_tune_allow_cpu_gate",
        "modal.model_settings.qwen_ane_tune_allow_cpu_down",
        "modal.model_settings.qwen_ane_tune_allow_ane_gdn",
        "modal.model_settings.qwen_ane_tune_allow_cpu_gdn",
        "modal.model_settings.qwen_ane_tune_allow_cpu_scheduler",
        "modal.model_settings.qwen_ane_tune_cancel",
        "modal.model_settings.qwen_ane_tune_apply",
        "modal.model_settings.qwen_ane_tune_applying",
        "modal.model_settings.qwen_ane_tune_applied",
        "modal.model_settings.qwen_ane_tune_preparing",
        "modal.model_settings.qwen_ane_tune_test",
        "modal.model_settings.qwen_ane_tune_throughput",
        "modal.model_settings.qwen_ane_tail_padding",
    }

    for locale_path in sorted(i18n_dir.glob("*.json")):
        translations = json.loads(locale_path.read_text())
        missing_keys = keys - translations.keys()
        assert not missing_keys, f"{locale_path.name} is missing {sorted(missing_keys)}"


def test_qwen_ane_model_specific_controls_are_fully_wired():
    html = _model_settings_template()
    script = _dashboard_script()
    fields = {
        "qwen35_ane_prefill_enabled",
        "qwen35_ane_prefill_sequence_length",
        "qwen35_ane_prefill_tail_padding_min_tokens",
        "qwen35_ane_prefill_fraction",
        "qwen35_ane_prefill_max_layers",
        "qwen35_ane_prefill_dual_ane",
        "qwen35_ane_prefill_gdn",
        "qwen35_ane_prefill_gdn_fraction",
        "qwen35_ane_prefill_gdn_max_layers",
    }

    assert 'x-if="isQwen35AnePrefillModel(selectedModel)"' in html
    assert "'qwen3_5', 'qwen3_6', 'qwen3_8'" in script
    for field in fields:
        assert f"modelSettings.{field}" in html
        assert f"{field}:" in script

    assert 'x-model.number="modelSettings.qwen35_ane_prefill_fraction"' in html
    assert 'x-model.number="modelSettings.qwen35_ane_prefill_gdn_fraction"' in html
    assert 'min="0.05" max="0.90" step="0.005"' in html
    assert 'placeholder="0.53"' in html
    assert 'placeholder="0.5"' in html
    assert "measured optimum" not in html


def test_qwen_ane_numeric_controls_accept_arbitrary_valid_values():
    html = _model_settings_template()
    section = _section(
        html,
        "<!-- Qwen 3.5/3.6/3.8 private ANE/GPU prompt processing -->",
        "<!-- TurboQuant KV Cache -->",
    )

    for field in (
        "qwen35_ane_prefill_sequence_length",
        "qwen35_ane_prefill_tail_padding_min_tokens",
        "qwen35_ane_prefill_fraction",
        "qwen35_ane_prefill_cpu_fraction",
        "qwen35_ane_prefill_cpu_down_fraction",
        "qwen35_ane_prefill_cpu_gdn_fraction",
        "qwen35_ane_prefill_cpu_threads",
        "qwen35_ane_prefill_gdn_fraction",
    ):
        binding = f'x-model.number="modelSettings.{field}"'
        before, after = section.split(binding, 1)
        assert before.rsplit("<", 1)[-1].startswith("input ")
        assert "</select>" not in after.split(">", 1)[0]

    assert 'min="1024" step="64"' in section
    assert 'min="0" max="64" step="1"' in section


def test_qwen_ane_web_tuner_is_wired_to_transient_benchmark_and_apply():
    html = _model_settings_template()
    script = _dashboard_script()

    assert "startANETuning()" in html
    assert "cancelANETuning()" in html
    assert "applyANETuningRecommendation()" in html
    assert "aneTuningRecommendationText()" in html
    assert "aneTuningResultText(result)" in html
    assert "aneTuning.status?.termination_reason" in html
    assert "aneTuning.status?.results || []" in html
    assert 'x-model="aneTuningOverrides.allowCpu"' in html
    assert 'x-model="aneTuningOverrides.allowAneGdn"' in html
    assert 'x-model="aneTuningOverrides.allowCpuGdn"' in html
    assert "'/admin/api/bench/ane-tune/start'" in script
    assert "/admin/api/bench/ane-tune/${encodeURIComponent(tuningId)}/results" in script
    assert "/admin/api/bench/ane-tune/${encodeURIComponent(tuningId)}/cancel" in script
    assert "qwen35_ane_prefill_fraction = Number(recommendation.mlp_fraction)" in script
    assert "qwen35_ane_prefill_gdn_fraction = Number(" in script
    assert "qwen35_ane_prefill_cpu_enabled = !!recommendation.cpu_enabled" in script
    assert "qwen35_ane_prefill_cpu_fraction = Number(" in script
    assert "allow_cpu: this.aneTuningOverrides.allowCpu" in script
    assert "allow_ane_gdn: this.aneTuningOverrides.allowAneGdn" in script
    assert "allow_cpu_gdn: this.aneTuningOverrides.allowCpu" in script
    assert "qwen35_ane_prefill_cpu_down_fraction = Number(" in script
    assert "qwen35_ane_prefill_cpu_gdn_fraction = Number(" in script
    assert "recommendation.cpu_shared_resource" in script
    assert "if (result?.processing_tps === null" in script
    assert "result?.latency_ms !== null" in script


def test_qwen_ane_arbitrary_inputs_are_validated_before_save():
    script = _dashboard_script()

    assert "validateQwenAneSettings()" in script
    assert "ANE prompt block must be a multiple of 64." in script
    assert "MLP ANE and CPU fractions must total less than 1.0." in script
    assert "GDN ANE and CPU fractions must total less than 1.0." in script
    assert "CPU worker count must be between 0 and 64." in script
    assert "const qwenAneValidationError = this.validateQwenAneSettings()" in script
    assert "qwen35_ane_prefill_fraction: Number(" in script


def test_qwen_ane_web_defaults_match_configured_profile():
    script = _dashboard_script()
    state = script.split("buildModelSettingsState(model, settings) {", 1)[1].split(
        "_resetPresetApplicableFields()", 1
    )[0]

    assert "qwen35_ane_prefill_sequence_length: s.qwen35_ane_prefill_sequence_length || 2048" in state
    assert "qwen35_ane_prefill_fraction: s.qwen35_ane_prefill_fraction ?? 0.53" in state
    assert "qwen35_ane_prefill_max_layers: s.qwen35_ane_prefill_max_layers || 64" in state
    assert "qwen35_ane_prefill_dual_ane: s.qwen35_ane_prefill_dual_ane !== false" in state
    assert "qwen35_ane_prefill_gdn: s.qwen35_ane_prefill_gdn !== false" in state
    assert "qwen35_ane_prefill_gdn_fraction: s.qwen35_ane_prefill_gdn_fraction ?? 0.5" in state
    assert "qwen35_ane_prefill_gdn_max_layers: s.qwen35_ane_prefill_gdn_max_layers ?? 48" in state
    assert "qwen35_ane_prefill_cpu_enabled: s.qwen35_ane_prefill_cpu_enabled || false" in state
    assert "qwen35_ane_prefill_cpu_fraction: s.qwen35_ane_prefill_cpu_fraction ?? 0.135" in state
    assert "qwen35_ane_prefill_cpu_down_fraction: s.qwen35_ane_prefill_cpu_down_fraction ?? 0" in state
    assert "qwen35_ane_prefill_cpu_gdn_fraction: s.qwen35_ane_prefill_cpu_gdn_fraction ?? 0" in state
    assert "qwen35_ane_prefill_cpu_threads: s.qwen35_ane_prefill_cpu_threads ?? 8" in state
    assert "qwen35_ane_prefill_cpu_shared_resource: s.qwen35_ane_prefill_cpu_shared_resource !== false" in state


def test_js_embedded_translations_escape_apostrophes():
    # A t() value dropped into a single-quoted Alpine JS string breaks the
    # whole expression as soon as a translation contains an apostrophe (or a
    # trailing backslash). Every quoted embed must run the JS-escape replace
    # chain instead of interpolating the raw translation.
    import re

    unsafe = re.findall(
        r"'\{\{ t\('[a-z_.0-9]+'\) \}\}'", _model_settings_template()
    )
    assert unsafe == []
