"""Regression tests for attaching MTP/assistant drafters from the dashboard.

Clicking Load on an MTP/Assistant drafter must attach it to its chat model
(enable VLM MTP with the drafter on the target, then load the target)
instead of surfacing the standalone-load 500 the server returns for
drafter checkpoints.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCALES = ["en", "es", "fr", "ja", "ko", "pt-BR", "ru", "zh-TW", "zh"]


def _read(*parts) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def _dashboard_js() -> str:
    return _read("omlx", "admin", "static", "js", "dashboard.js")


def _method_block(js: str, signature: str, following_signature: str) -> str:
    return js.split(signature, 1)[1].split(following_signature, 1)[0]


def test_load_model_routes_attachable_drafters():
    js = _dashboard_js()
    body = _method_block(js, "async loadModel(modelId) {", "async unloadModel(")

    assert "isAttachableDrafter(model)" in body
    assert "attachDrafter(model)" in body


def test_attachable_drafter_kinds_are_mtp_and_assistant_only():
    js = _dashboard_js()
    body = _method_block(js, "isAttachableDrafter(model) {", "resolveDrafterTarget(")

    assert "helper_kind === 'mtp'" in body
    assert "helper_kind === 'assistant'" in body
    assert "dflash" not in body


def test_drafter_pairing_prefers_reference_then_name_heuristic():
    js = _dashboard_js()
    body = _method_block(js, "resolveDrafterTarget(drafter) {", "async attachDrafter(")

    # Priority 1: a chat model whose settings already reference the drafter,
    # matched by model id, on-disk path, or Hub repo id.
    assert "vlm_mtp_draft_model" in body
    assert "model_path" in body
    assert "source_repo_id" in body
    # Only unambiguous matches resolve — several candidates mean "manual".
    assert body.count("length === 1") == 3


def test_drafter_target_names_strip_mtp_and_assistant_tokens():
    js = _dashboard_js()
    body = _method_block(js, "function drafterTargetNames(name) {", "function dashboard()")

    # Case-insensitive token drop, joined back exactly.
    assert "toLowerCase()" in body
    assert "!== 'mtp'" in body
    assert "!== 'assistant'" in body


def test_attach_drafter_enables_vlm_mtp_and_loads_target():
    js = _dashboard_js()
    body = _method_block(js, "async attachDrafter(drafter) {", "async loadModel(modelId) {")

    assert "vlm_mtp_enabled: true" in body
    assert "vlm_mtp_draft_model: drafter.id" in body
    # A loaded-but-unpinned target is auto-unloaded by the settings PUT and
    # only pinned models auto-reload — the dashboard must re-load it itself.
    assert "auto_unloaded" in body
    assert "auto_reloaded" in body
    assert "js.error.drafter_no_target" in body


def test_drafter_no_target_error_localized_in_all_locales():
    for locale in LOCALES:
        data = json.loads(_read("omlx", "admin", "i18n", f"{locale}.json"))
        message = data.get("js.error.drafter_no_target")
        assert message, locale
        assert "{id}" in message, locale


def test_manager_rows_badge_helper_models():
    template = _read("omlx", "admin", "templates", "dashboard", "_models.html")

    assert "managerModelInfo(model.name)?.is_helper" in template
    assert "settings.models.table.helper_badge" in template
