"""Regression tests for admin speculative draft-model filters."""

from pathlib import Path


def _dashboard_js() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "omlx/admin/static/js/dashboard.js").read_text()


def _method_block(js: str, signature: str, following_signature: str) -> str:
    return js.split(signature, 1)[1].split(following_signature, 1)[0]


def test_specprefill_draft_filter_allows_mtp_preserved_models():
    js = _dashboard_js()
    body = _method_block(
        js,
        "isSpecPrefillDraftModel(model) {",
        "draftModelCandidates(",
    )

    assert "!this.isDflashDraftModel(model)" in body
    assert "!this.isVlmMtpDraftModel(model)" in body
    assert "mtp($|[-_/\\s])" not in body


def test_vlm_mtp_draft_filter_keeps_mtp_fallback():
    js = _dashboard_js()
    body = _method_block(
        js,
        "isVlmMtpDraftModel(model) {",
        "isSpecPrefillDraftModel(",
    )

    assert "VLM_MTP_DRAFTER_CONFIG_MODEL_TYPES.has(configType)" in body
    assert "assistant|(^|[-_/\\s])mtp" in body
