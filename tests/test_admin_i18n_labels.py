"""Locale coverage for the engine-type, community-bench, and oQ About labels."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = ROOT / "omlx/admin/i18n"

# Keys introduced in the i18n pass over:
#   _modal_model_settings.html (engine-type options),
#   _bench.html (community upload block),
#   _models.html (oQ About panel).
NEW_KEYS = {
    # Engine-type dropdown
    "modal.model_settings.engine_type_llm",
    "modal.model_settings.engine_type_vlm",
    "modal.model_settings.engine_type_embedding",
    "modal.model_settings.engine_type_reranker",
    "modal.model_settings.engine_type_audio_stt",
    "modal.model_settings.engine_type_audio_tts",
    "modal.model_settings.engine_type_audio_sts",
    # Community benchmark upload block
    "bench.community.section_label",
    "bench.community.uploading",
    # oQ About panel
    "models.oq.about_title",
    "models.oq.about_subtitle",
    "models.oq.about_full_name",
    "models.oq.about_quote",
    "models.oq.about_portable",
    "models.oq.about_no_custom_loader",
    "models.oq.about_mixed_heading",
    "models.oq.about_mixed_body",
    "models.oq.about_what_heading",
    "models.oq.about_layer_sensitivity",
    "models.oq.about_layer_sensitivity_body",
    "models.oq.about_mixed_precision",
    "models.oq.about_mixed_precision_body",
    "models.oq.about_architecture_rules",
    "models.oq.about_architecture_rules_body",
    "models.oq.about_oqe_heading",
    "models.oq.about_oqe_body",
    "models.oq.about_table_path",
    "models.oq.about_table_bit_alloc",
    "models.oq.about_table_affine",
    "models.oq.about_table_best_use",
    "models.oq.about_oq_row_bit",
    "models.oq.about_oq_row_affine",
    "models.oq.about_oq_row_best",
    "models.oq.about_oqe_row_bit",
    "models.oq.about_oqe_row_affine",
    "models.oq.about_oqe_row_best",
    "models.oq.about_imatrix_heading",
    "models.oq.about_step1_title",
    "models.oq.about_step1_body",
    "models.oq.about_step2_title",
    "models.oq.about_step2_body",
    "models.oq.about_step3_title",
    "models.oq.about_step3_body",
    "models.oq.about_step4_title",
    "models.oq.about_step4_body",
    "models.oq.about_fallback_note",
}


def _locales():
    for locale_path in sorted(I18N_DIR.glob("*.json")):
        yield locale_path.name, json.loads(locale_path.read_text(encoding="utf-8"))


def test_new_labels_exist_in_every_locale():
    for name, translations in _locales():
        missing = NEW_KEYS - translations.keys()
        assert not missing, f"{name} is missing {sorted(missing)}"


def test_zh_provides_real_translations():
    zh = json.loads((I18N_DIR / "zh.json").read_text(encoding="utf-8"))
    # Spot-check the non-trivial translations shipped with zh.json.
    assert zh["bench.community.section_label"] == "社区基准上传"
    assert zh["bench.community.uploading"] == "上传中..."
    assert zh["modal.model_settings.engine_type_audio_stt"] == "音频 STT"
    assert zh["models.oq.about_title"] == "关于 oQ 量化"
    assert "回退" in zh["models.oq.about_fallback_note"]
    assert "imatrix" in zh["models.oq.about_fallback_note"]
    # Untranslated-by-design entries stay identical to English.
    assert zh["modal.model_settings.engine_type_llm"] == "LLM"
    assert zh["modal.model_settings.engine_type_embedding"] == "Embedding"


def test_oq_panel_templates_use_only_known_keys():
    """Every t() key used by the oQ About panel must exist in the catalogs."""
    html = (
        ROOT / "omlx/admin/templates/dashboard/_models.html"
    ).read_text(encoding="utf-8")
    section = html.split("<!-- About oQ Quantization -->", 1)[1].split(
        "<!-- ==================== UPLOADER TAB", 1
    )[0]
    en = json.loads((I18N_DIR / "en.json").read_text(encoding="utf-8"))
    used = set()
    for part in section.split("{{ t('")[1:]:
        used.add(part.split("'", 1)[0])
    assert used, "no t() calls found in oQ About panel"
    missing = {k for k in used if "t('" in k or not k}
    assert not missing, f"malformed keys: {sorted(missing)}"
    assert used <= en.keys(), f"unknown keys in template: {sorted(used - en.keys())}"