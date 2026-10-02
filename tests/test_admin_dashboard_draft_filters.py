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


def test_dflash_draft_filter_claims_muse_glimmer_assistant():
    js = _dashboard_js()
    body = _method_block(
        js,
        "isDflashDraftModel(model) {",
        "isVlmMtpDraftModel(",
    )

    # Meta's Muse Glimmer DFlash drafter is named "-assistant" and carries
    # no "dflash" name token; routing keys on config_model_type.
    assert "DFLASH_DRAFTER_CONFIG_MODEL_TYPES.has(configType)" in body
    # DFlash 2 checkpoints are versioned ("-DFlash2"), so the name heuristic
    # must accept an optional numeric suffix after the "dflash" token.
    assert "dflash[0-9]*($|[-_/\\s])" in body
    assert "'muse_glimmer_assistant'" in js.split(
        "DFLASH_DRAFTER_CONFIG_MODEL_TYPES = new Set([", 1
    )[1].split("])", 1)[0]


def test_vlm_mtp_set_does_not_claim_muse_glimmer_assistant():
    js = _dashboard_js()
    vlm_mtp_set = js.split(
        "VLM_MTP_DRAFTER_CONFIG_MODEL_TYPES = new Set([", 1
    )[1].split("])", 1)[0]
    assert "muse_glimmer_assistant" not in vlm_mtp_set
