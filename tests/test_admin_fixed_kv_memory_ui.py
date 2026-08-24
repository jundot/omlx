"""Contract tests for the fixed KV-cache launch preflight UI."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = (ROOT / "omlx/admin/templates/dashboard.html").read_text()
SETTINGS_MODAL = (
    ROOT / "omlx/admin/templates/dashboard/_modal_model_settings.html"
).read_text()
LAUNCH_MODAL = (
    ROOT / "omlx/admin/templates/dashboard/_modal_model_launch.html"
).read_text()
JAVASCRIPT = (ROOT / "omlx/admin/static/js/dashboard.js").read_text()


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_dashboard_includes_launch_preflight_modal():
    assert '{% include "dashboard/_modal_model_launch.html" %}' in DASHBOARD
    assert 'x-show="showModelLaunchModal"' in LAUNCH_MODAL
    assert '@click="confirmModelLaunch()"' in LAUNCH_MODAL


def test_manual_load_opens_preflight_before_posting():
    public_load = _between(
        JAVASCRIPT,
        "async loadModel(modelId) {",
        "async fetchModelMemoryEstimate(modelId, maxContextWindow) {",
    )
    confirmed_load = _between(
        JAVASCRIPT,
        "async performConfirmedModelLoad(modelId) {",
        "async confirmModelLaunch() {",
    )

    assert "await this.openModelLaunch(model)" in public_load
    assert "/load`" not in public_load
    assert "/load`" in confirmed_load
    assert "{ method: 'POST' }" in confirmed_load


def test_non_text_engines_keep_direct_load_path():
    public_load = _between(
        JAVASCRIPT,
        "modelSupportsFixedKVCache(model) {",
        "async fetchModelMemoryEstimate(modelId, maxContextWindow) {",
    )

    assert "engineType === 'batched' || engineType === 'vlm'" in public_load
    assert "configType !== 'diffusion_gemma'" in public_load
    assert "if (!this.modelUsesFixedKVLaunchPreflight(model))" in public_load
    assert "await this.performConfirmedModelLoad(modelId)" in public_load


def test_per_model_toggle_disables_preflight_and_is_saved():
    assert 'modelSettings.fixed_kv_cache_enabled' in SETTINGS_MODAL
    assert '@click="toggleModelFixedKVCache()"' in SETTINGS_MODAL
    assert "model?.settings?.fixed_kv_cache_enabled !== false" in JAVASCRIPT
    assert "fixed_kv_cache_enabled: s.fixed_kv_cache_enabled !== false" in JAVASCRIPT
    assert (
        "fixed_kv_cache_enabled: this.modelSettings.fixed_kv_cache_enabled !== false"
        in JAVASCRIPT
    )
    assert "this.modelSettings.fixed_kv_cache_enabled === false" in JAVASCRIPT
    assert "modal.model_settings.fixed_kv_cache_disabled" in SETTINGS_MODAL


def test_estimate_request_uses_context_override_and_no_client_kv_formula():
    estimate_fetch = _between(
        JAVASCRIPT,
        "async fetchModelMemoryEstimate(modelId, maxContextWindow) {",
        "queueModelMemoryEstimate() {",
    )

    assert "/memory-estimate${suffix}" in estimate_fetch
    assert "params.set('max_context_window', String(context))" in estimate_fetch
    assert "per_session_kv_bytes *" not in JAVASCRIPT
    assert "num_hidden_layers *" not in JAVASCRIPT
    assert "num_key_value_heads *" not in JAVASCRIPT


def test_context_changes_debounce_both_previews():
    assert 'x-model.number="modelSettings.max_context_window"' in SETTINGS_MODAL
    assert '@input="queueModelMemoryEstimate()"' in SETTINGS_MODAL
    assert 'x-model.number="launchContextWindow"' in LAUNCH_MODAL
    assert '@input="queueLaunchMemoryEstimate()"' in LAUNCH_MODAL
    assert "}, 300);" in JAVASCRIPT


def test_launch_saves_context_then_calls_private_load_method():
    confirm = _between(
        JAVASCRIPT,
        "async confirmModelLaunch() {",
        "async unloadModel(modelId) {",
    )

    assert "body: JSON.stringify({ max_context_window: context })" in confirm
    assert "await this.performConfirmedModelLoad(modelId)" in confirm
    assert "this.launchMemoryEstimate?.fits === true" in JAVASCRIPT
    assert ':disabled="!canConfirmModelLaunch()"' in LAUNCH_MODAL


def test_breakdown_shows_server_fields_and_committed_state():
    combined = SETTINGS_MODAL + LAUNCH_MODAL
    required = {
        "weights_bytes",
        "per_session_kv_bytes",
        "requested_session_slots",
        "reserved_session_slots",
        "fixed_kv_cache_bytes",
        "other_fixed_bytes",
        "estimated_total_bytes",
        "unified_memory_bytes",
        "available_memory_bytes",
        "projected_remaining_bytes",
        "model_context_limit",
        "fit_reason",
        "components",
    }
    missing = {field for field in required if field not in combined + JAVASCRIPT}
    assert not missing
    assert "memory.preview.committed" in JAVASCRIPT
    assert "launchModelComplete" in LAUNCH_MODAL
    assert "bg-blue-500" in combined
    assert "bg-amber-500" in combined
    assert "bg-purple-500" in combined


def test_memory_copy_exists_in_every_dashboard_locale():
    required = {
        "model_launch.section_label",
        "model_launch.description",
        "model_launch.context_window",
        "model_launch.context_hint",
        "model_launch.confirm",
        "model_launch.launching",
        "model_launch.success",
        "memory.preview.section_label",
        "memory.preview.settings_hint",
        "memory.preview.launch_hint",
        "memory.preview.loading",
        "memory.preview.error",
        "memory.preview.estimated",
        "memory.preview.committed",
        "memory.preview.weights",
        "memory.preview.fixed_kv",
        "memory.preview.other_fixed",
        "memory.preview.estimated_total",
        "memory.preview.per_session_kv",
        "memory.preview.session_pool",
        "memory.preview.session_pool_value",
        "memory.preview.unified_memory",
        "memory.preview.available_now",
        "memory.preview.projected_remaining",
        "memory.preview.unavailable",
        "modal.model_settings.fixed_kv_cache",
        "modal.model_settings.fixed_kv_cache_hint",
        "modal.model_settings.fixed_kv_cache_disabled",
    }
    for locale_path in sorted((ROOT / "omlx/admin/i18n").glob("*.json")):
        locale = json.loads(locale_path.read_text())
        missing = {key for key in required if not locale.get(key)}
        assert not missing, f"{locale_path.name}: missing {sorted(missing)}"
