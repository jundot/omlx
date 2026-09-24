"""Regression tests for admin model-settings UI gates."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


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
    return (
        root / "omlx/admin/templates/dashboard/blocks/_active_models.html"
    ).read_text()


def _section(html: str, start_marker: str, end_marker: str) -> str:
    return html.split(start_marker, 1)[1].split(end_marker, 1)[0]


def test_lightning_mtp_and_turboquant_are_not_ui_mutexed():
    html = _model_settings_template()

    turboquant = _section(
        html,
        "<!-- TurboQuant KV Cache -->",
        "<!-- MoE Expert Offload -->",
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

    assert 'x-for="value in selectedModel?.reasoning_effort_options || []"' in section
    assert ':value="value" x-text="value"' in section
    assert "!selectedModel?.reasoning_effort_custom && !entry.custom" in section


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

    assert "x-if=\"selectedModel?.ane_prefill_backend === 'qwen'\"" in html
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
    assert "aneTuning.status?.message" in html
    assert "aneTuningProgressPercent()" in html
    assert "aneTuning.status?.termination_reason" in html
    assert "!aneTuning.running && aneTuning.status?.recommendation" in html
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


def test_qwen_ane_arbitrary_inputs_are_validated_before_save():
    script = _dashboard_script()

    assert "validateQwenAneSettings()" in script
    # The messages are localised now, so assert the keys the validator returns
    # rather than the English copy.
    assert "window.t('js.error.ane_prompt_block_multiple')" in script
    assert "window.t('js.error.mlp_ane_cpu_total')" in script
    assert "window.t('js.error.gdn_ane_cpu_total')" in script
    assert "window.t('js.error.cpu_workers_range')" in script
    assert "const qwenAneValidationError = this.validateQwenAneSettings()" in script
    assert "qwen35_ane_prefill_fraction: Number(" in script


def test_qwen_ane_web_defaults_match_configured_profile():
    script = _dashboard_script()
    state = script.split("buildModelSettingsState(model, settings) {", 1)[1].split(
        "_resetPresetApplicableFields()", 1
    )[0]

    assert (
        "qwen35_ane_prefill_sequence_length: s.qwen35_ane_prefill_sequence_length || 2048"
        in state
    )
    assert (
        "qwen35_ane_prefill_fraction: s.qwen35_ane_prefill_fraction ?? model?.ane_prefill_default_fraction ?? 0.53"
        in state
    )
    assert (
        "qwen35_ane_prefill_max_layers: s.qwen35_ane_prefill_max_layers || 64" in state
    )
    assert (
        "qwen35_ane_prefill_dual_ane: s.qwen35_ane_prefill_dual_ane !== false" in state
    )
    assert "qwen35_ane_prefill_gdn: s.qwen35_ane_prefill_gdn !== false" in state
    assert (
        "qwen35_ane_prefill_gdn_fraction: s.qwen35_ane_prefill_gdn_fraction ?? 0.5"
        in state
    )
    assert (
        "qwen35_ane_prefill_gdn_max_layers: s.qwen35_ane_prefill_gdn_max_layers ?? 48"
        in state
    )
    assert (
        "qwen35_ane_prefill_cpu_enabled: s.qwen35_ane_prefill_cpu_enabled || false"
        in state
    )
    assert (
        "qwen35_ane_prefill_cpu_fraction: s.qwen35_ane_prefill_cpu_fraction ?? 0.135"
        in state
    )
    assert (
        "qwen35_ane_prefill_cpu_down_fraction: s.qwen35_ane_prefill_cpu_down_fraction ?? 0"
        in state
    )
    assert (
        "qwen35_ane_prefill_cpu_gdn_fraction: s.qwen35_ane_prefill_cpu_gdn_fraction ?? 0"
        in state
    )
    assert (
        "qwen35_ane_prefill_cpu_threads: s.qwen35_ane_prefill_cpu_threads ?? 8" in state
    )
    assert (
        "qwen35_ane_prefill_cpu_shared_resource: s.qwen35_ane_prefill_cpu_shared_resource !== false"
        in state
    )


def test_js_embedded_translations_escape_apostrophes():
    # A t() value dropped into a single-quoted Alpine JS string breaks the
    # whole expression as soon as a translation contains an apostrophe (or a
    # trailing backslash). Every quoted embed must run the JS-escape replace
    # chain instead of interpolating the raw translation.
    import re

    unsafe = re.findall(r"'\{\{ t\('[a-z_.0-9]+'\) \}\}'", _model_settings_template())
    assert unsafe == []


def test_oq_a8_toggle_is_gated_to_qwen35_models():
    """The kernels only exist for this checkpoint family, so the UI hides them."""
    html = _model_settings_template()
    section = _section(
        html,
        "<!-- Qwen 3.5/3.6/3.8 oQ INT8-activation prefill kernels -->",
        "<!-- Qwen 3.5/3.6/3.8 private ANE/GPU prompt processing -->",
    )
    assert 'x-if="isQwenOqA8Model(selectedModel)"' in section
    assert "modelSettings.qwen35_oq_a8_enabled" in section
    # The detail controls only appear once the feature is on.
    assert 'x-show="modelSettings.qwen35_oq_a8_enabled"' in section
    assert "modelSettings.qwen35_oq_a8_min_tokens" in section


def test_oq_a8_offers_no_kernel_choice():
    """The tile is not a user-facing choice: the dispatcher picks it per bit
    width from a measured default, so the modal exposes only the toggle and
    the token floor."""
    html = _model_settings_template()
    section = _section(
        html,
        "<!-- Qwen 3.5/3.6/3.8 oQ INT8-activation prefill kernels -->",
        "<!-- Qwen 3.5/3.6/3.8 private ANE/GPU prompt processing -->",
    )
    assert "modelSettings.qwen35_oq_a8_variant" not in section


def test_oq_a8_settings_are_registered_in_the_dashboard_script():
    js = _dashboard_script()
    for field in ("qwen35_oq_a8_enabled", "qwen35_oq_a8_min_tokens"):
        # Profile-field registry, modal defaults, server load, and save payload.
        assert js.count(field) >= 4, field
    assert "validateQwenOqA8Settings()" in js
    # The modal has to explain the ANE clash itself rather than let the save
    # come back as a bare 400 with the toggle already flipped.
    validator = js.split("validateQwenOqA8Settings()", 1)[1].split("},", 1)[0]
    assert "qwen35_ane_prefill_enabled" in validator
    # The tile must not leak into any of those four places, not even as a
    # hidden default the modal never renders.
    assert "qwen35_oq_a8_variant" not in js


def test_oq_a8_labels_use_i18n_keys():
    html = _model_settings_template()
    assert "{{ t('modal.model_settings.qwen_oq_a8') }}" in html
    assert ">Qwen INT8 Activation Prefill<" not in html


def test_oq_a8_i18n_keys_exist_in_every_locale():
    root = Path(__file__).resolve().parents[1]
    i18n_dir = root / "omlx/admin/i18n"
    keys = {
        "modal.model_settings.qwen_oq_a8",
        "modal.model_settings.qwen_oq_a8_hint",
        "modal.model_settings.qwen_oq_a8_min_tokens",
    }
    for path in sorted(i18n_dir.glob("*.json")):
        catalog = json.loads(path.read_text())
        missing = keys - set(catalog)
        assert not missing, f"{path.name} is missing {sorted(missing)}"


def test_profile_api_toggle_state_uses_i18n_keys():
    """Both expose-as-API toggles localise their ON/OFF state.

    The state was hardcoded before the migration; `API` itself stays a Latin
    literal because it is the product's own acronym for an API-addressable
    model, and this PR leaves acronyms untranslated.
    """
    html = _model_settings_template()

    # New-profile form and the inline edit form share the one key pair.
    assert html.count("t('modal.model_settings.profiles.expose_as_model_on')") == 2
    assert html.count("t('modal.model_settings.profiles.expose_as_model_off')") == 2
    assert "'ON'" not in html
    assert "'OFF'" not in html

    assert [line.strip() for line in html.splitlines()].count("API") == 2


def test_profile_api_toggle_i18n_keys_exist_in_every_locale():
    root = Path(__file__).resolve().parents[1]
    i18n_dir = root / "omlx/admin/i18n"
    english = {
        "modal.model_settings.profiles.expose_as_model_on": "ON",
        "modal.model_settings.profiles.expose_as_model_off": "OFF",
    }
    for path in sorted(i18n_dir.glob("*.json")):
        catalog = json.loads(path.read_text())
        missing = set(english) - set(catalog)
        assert not missing, f"{path.name} is missing {sorted(missing)}"
        if path.name == "zh.json":
            # Simplified Chinese carries its own labels; every other locale
            # keeps the English fallback until its own translation lands.
            assert catalog["modal.model_settings.profiles.expose_as_model_on"] == "开"
            assert catalog["modal.model_settings.profiles.expose_as_model_off"] == "关"
            continue
        if path.name == "cs.json":
            # Czech carries its own labels (ZAP/VYP), same exemption as zh.
            assert catalog["modal.model_settings.profiles.expose_as_model_on"] == "ZAP"
            assert catalog["modal.model_settings.profiles.expose_as_model_off"] == "VYP"
            continue
        for key, value in english.items():
            assert (
                catalog[key] == value
            ), f"{path.name} {key} is not the English fallback"


def test_moe_expert_offload_toggle_blocks_speculative_decoding():
    """Offload is incompatible with speculative verification paths."""
    html = _model_settings_template()
    section = _section(html, "<!-- MoE Expert Offload -->", "<!-- IndexCache")
    assert "modelSettings.moe_expert_offload_enabled" in section
    assert "modelSettings.moe_expert_offload_resident_percent" in section
    assert ":disabled" in section
    for key in ("mtp_enabled", "vlm_mtp_enabled", "dflash_enabled"):
        assert f"modelSettings.{key}" in section


def test_profile_editor_behavior():
    """Execute the dashboard methods, including async response and edit races."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const test = require('node:test');

const source = fs.readFileSync('omlx/admin/static/js/dashboard.js', 'utf8');
// The dashboard resolves its copy through window.t, so feed the component the
// shipped English catalog exactly as base.html does; the assertions below then
// read the strings the app really renders.
const catalog = JSON.parse(fs.readFileSync('omlx/admin/i18n/en.json', 'utf8'));
function setup(fetch) {
    const context = {
        localStorage: {getItem: () => null},
        THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
        window: {t: key => (catalog[key] !== undefined ? catalog[key] : key)},
        navigator: {language: 'en'}, document: {}, fetch,
    };
    const state = vm.runInNewContext(source + '\n dashboard;', context)();
    state.selectedModel = {id: 'model-a'};
    state.models = [state.selectedModel];
    state.loadProfilesForModel = async () => {};
    return state;
}

test('Template apply uses server identity and effective settings', async () => {
    const requests = [];
    const state = setup(async (url, options) => {
        requests.push([url, options.method]);
        return {ok: true, json: async () => ({settings: {active_profile_name: 'copy-2', temperature: 0.9}})};
    });
    state.profiles = [{name:'coding', settings:{temperature:0.1}}];
    await state.applyTemplateToForm({name:'coding', settings:{temperature:0.2}});
    assert.deepEqual(requests, [['/admin/api/models/model-a/profile-templates/coding/apply', 'POST']]);
    assert.equal(state.activeProfileName, 'copy-2');
    assert.equal(state.modelSettings.temperature, 0.9);
});

test('Failed template application keeps the form and surfaces the error', async () => {
    const state = setup(async () => ({ok:false, status:400, json:async()=>({detail:'Rejected'})}));
    state.modelSettings.temperature = 0.1;
    await state.applyTemplateToForm({name:'coding'});
    assert.equal(state.profileError, 'Rejected');
    assert.equal(state.modelSettings.temperature, 0.1);
});

test('Save as new invokes the corresponding apply action', async () => {
    for (const global of [false, true]) {
        const state = setup(async (url, options) => ({
            ok: true, json: async () => ({[global ? 'template' : 'profile']: JSON.parse(options.body)}),
        }));
        let applied;
        state.loadTemplates = async () => {};
        const apply = global ? 'applyTemplateToForm' : 'applyProfileToForm';
        state[apply] = async profile => { applied = profile; };
        if (global) {
            state.newTemplate = {display_name:'Coding', description:''};
            await state.createTemplate();
        } else {
            state.newProfile = {display_name:'Coding', api_name:'coding', description:''};
            await state.createProfile();
        }
        assert.equal(applied.display_name, 'Coding');
    }
});

test('Focus refresh updates clean forms and preserves edits made while fetching', async () => {
    const state = setup();
    state.showModelSettingsModal = true;
    state._modelSettingsBaseline = JSON.stringify(state.modelSettings);
    let opened = 0;
    state.openModelSettings = async()=>{opened++;};
    state.loadModels = async()=>{};
    await state.refreshOpenModelSettings();
    assert.equal(opened, 1);
    state.loadModels = async()=>{state.modelSettings.temperature = 0.3;};
    await state.refreshOpenModelSettings();
    assert.equal(opened, 1);
    await state.refreshOpenModelSettings();
    assert.equal(state.modelSettings.temperature, 0.3);
    assert.equal(opened, 1);
});

test('Apply response cannot replace another model editor', async () => {
    let resolve;
    const state = setup(() => new Promise(r => {resolve=r;}));
    const pending = state.applyTemplateToForm({name:'coding'});
    state.selectedModel = {id:'model-b'};
    state.modelSettings.temperature = 0.4;
    resolve({ok:true, json:async()=>({settings:{temperature:0.9}})});
    await pending;
    assert.equal(state.modelSettings.temperature, 0.4);
});

test('Only a current linked copy marks a global template active', () => {
    const state = setup();
    state.templates = [{name:'coding',settings:{temperature:0.9}}];
    const independent = {name:'coding',settings:{temperature:0.1}};
    const copy = {name:'copy',source_template:'coding',settings:{temperature:0.9}};
    state.profiles = [independent, copy];
    state.activeProfileName = 'coding';
    assert.equal(state.activeTemplateName, null);
    assert.equal(state.visibleModelProfiles.length, 1);
    state.activeProfileName = 'copy';
    assert.equal(state.activeTemplateName, 'coding');
    copy.settings.temperature = 0.2;
    assert.equal(state.activeTemplateName, null);
    assert.equal(state.visibleModelProfiles.length, 2);
    copy.settings.temperature = 0.9;
    copy.expose_as_model = true;
    assert.equal(state.visibleModelProfiles.length, 2);
    state.templates = [];
    assert.equal(state.visibleModelProfiles.length, 2);
});


test('Template matching ignores nested dictionary ordering', () => {
    const state = setup();
    state.templates = [{name:'coding', settings:{chat_template_kwargs:{enable_thinking:true,custom:1}}}];
    const copy = {name:'copy', source_template:'coding', settings:{chat_template_kwargs:{custom:1,enable_thinking:true}}};
    assert.equal(state.matchingProfileTemplate(copy).name, 'coding');
    copy.settings.chat_template_kwargs.custom = 2;
    assert.equal(state.matchingProfileTemplate(copy), null);
});
"""
    result = subprocess.run(
        [node, "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


DASHBOARD_BLOCK_IDS = (
    "serving_stats",
    "usage_history",
    "active_models",
    "cache_observability",
    "api_endpoints",
    "claude_code",
    "applications",
    "engine_versions",
)


def test_dashboard_layout_template_contract():
    """Every block renders as a parked GridStack item with a tray pill."""
    from jinja2 import Environment, FileSystemLoader

    root = Path(__file__).resolve().parents[1]
    admin = root / "omlx/admin"
    env = Environment(loader=FileSystemLoader(str(admin / "templates")))
    env.globals.update(t=lambda key: key, static=lambda path: path)
    status = env.get_template("dashboard/_status.html").render()
    dashboard = (admin / "templates/dashboard.html").read_text()
    dashboard_js = _dashboard_script()

    for block_id in DASHBOARD_BLOCK_IDS:
        item = (
            f'class="grid-stack-item dash-block dash-block-parked" gs-id="{block_id}"'
        )
        assert status.count(item) == 1, block_id
        assert f"dashRemoveBlock('{block_id}')" in status
        # Alpine directives stay on the wrapper: GridStack clones the pill into
        # <body> while dragging, outside any x-data scope.
        pill = (
            f"""<div class="contents" x-show="!dashPlaced('{block_id}')">\n"""
            '                            <div class="dash-tray-pill grid-stack-item"'
            f' gs-w="12" gs-h="1" gs-min-w="6" data-block="{block_id}">'
        )
        assert pill in status, block_id
        assert f"status.layout.block.{block_id}" in status

    assert "css/gridstack.min.css" in dashboard
    assert "js/gridstack-all.js" in dashboard
    assert "js/dashboard_layout.js" in dashboard
    assert "dashboardWidthClass : 'max-w-7xl'" in dashboard
    assert 'class="max-w-7xl mx-auto px-4' not in dashboard

    assert "ui_dashboard_layout" in dashboard_js
    assert "columnMax: lib.COLUMNS" in dashboard_js

    locales = sorted((admin / "i18n").glob("*.json"))
    assert locales
    keys = [
        "status.layout.customize",
        "status.layout.width_full",
        "status.layout.save",
        *[f"status.layout.block.{block_id}" for block_id in DASHBOARD_BLOCK_IDS],
    ]
    for locale in locales:
        data = json.loads(locale.read_text(encoding="utf-8"))
        for key in keys:
            assert key in data, f"{locale.name} missing {key}"


def test_dashboard_layout_normalizer():
    """dashboard_layout.js clamps hand-edited layouts to the grid contract."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard layout tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync('omlx/admin/static/js/dashboard_layout.js', 'utf8');
const context = { window: {} };
vm.runInNewContext(source, context);
const lib = context.window.DashboardLayout;
// Objects built inside the vm realm have a different Object.prototype, so
// strict deepEqual rejects them; compare plain JSON copies instead.
const plain = value => JSON.parse(JSON.stringify(value));

assert.equal(lib.COLUMNS, 24);
assert.equal(lib.MIN_W, 6);
const def = lib.defaultLayout();
assert.equal(def.blocks.length, 8);
assert.ok(def.blocks.every((b, i) => b.x === 0 && b.y === i && b.w === 24));
assert.deepEqual(plain(lib.normalizeLayout(null)), plain(def));
assert.deepEqual(plain(lib.normalizeLayout({ blocks: 'nope' })), plain(def));

const messy = lib.normalizeLayout({
    width: 'huge',
    blocks: [
        { id: 'serving_stats', x: 20, y: '3', w: 12 },   // x clamped to 12
        { id: 'unknown', x: 0, y: 0, w: 24 },            // dropped
        { id: 'serving_stats', x: 0, y: 0, w: 24 },      // duplicate dropped
        { id: 'engine_versions', x: 0, y: -4, w: 2 },    // w -> 6, y -> 0
        { id: 'applications', x: 0, y: 1, w: 99 },       // w -> 24
    ],
});
assert.equal(messy.width, 'default');
assert.deepEqual(plain(messy.blocks), [
    { id: 'serving_stats', x: 12, y: 3, w: 12 },
    { id: 'engine_versions', x: 0, y: 0, w: 6 },
    { id: 'applications', x: 0, y: 1, w: 24 },
]);
assert.deepEqual(plain(lib.normalizeLayout({ width: 'full', blocks: [] }).blocks), []);
assert.equal(lib.widthClass('wide'), 'max-w-[90rem]');
assert.equal(lib.widthClass('bogus'), 'max-w-7xl');
"""
    result = subprocess.run(
        [node, "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
