# SPDX-License-Identifier: Apache-2.0
"""Tests for model discovery functionality."""

import json
import logging
from pathlib import Path

import pytest

from omlx.model_discovery import (
    DiscoveredModel,
    _is_adapter_dir,
    _is_helper_checkpoint,
    _is_unsupported_model,
    _read_model_context_length,
    _register_model,
    _resolve_hf_cache_entry,
    detect_model_type,
    discover_models,
    discover_models_from_dirs,
    estimate_model_size,
    format_size,
    is_helper_config_model_type,
    is_helper_model_config,
    model_directory_access_error,
)


class TestIsHelperConfigModelType:
    """Tests for helper (dFlash / Assistant / Draft) model_type detection."""

    @pytest.mark.parametrize(
        "config_model_type",
        ["gemma4_assistant", "qwen3_5_mtp", "foo_mtp", "bar_assistant", "QWEN3_5_MTP"],
    )
    def test_helper_types(self, config_model_type):
        assert is_helper_config_model_type(config_model_type) is True

    @pytest.mark.parametrize(
        "config_model_type",
        ["qwen3", "gemma4_text", "gemma4", "llama", "qwen2_5_vl", "", None, 123],
    )
    def test_non_helper_types(self, config_model_type):
        assert is_helper_config_model_type(config_model_type) is False


class TestIsHelperModelConfig:
    """Tests for full-config drafter detection (model_type / architecture / config-block)."""

    def test_dflash_draft_via_architecture(self):
        # DFlash drafts declare a plain qwen3 model_type but a DFlashDraftModel arch.
        config = {"model_type": "qwen3", "architectures": ["DFlashDraftModel"]}
        assert is_helper_model_config(config) is True

    def test_dflash_draft_via_config_block(self):
        config = {"model_type": "qwen3", "dflash_config": {"block_size": 16}}
        assert is_helper_model_config(config) is True

    def test_assistant_via_model_type(self):
        config = {
            "model_type": "gemma4_assistant",
            "architectures": ["Gemma4Assistant"],
        }
        assert is_helper_model_config(config) is True

    def test_mtp_via_model_type(self):
        config = {"model_type": "qwen3_5_mtp"}
        assert is_helper_model_config(config) is True

    @pytest.mark.parametrize(
        "config",
        [
            {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
            },
            {"model_type": "llama"},
            {},
            {"architectures": None},
            {"model_type": 123, "architectures": 123},
        ],
    )
    def test_non_helper_configs(self, config):
        assert is_helper_model_config(config) is False


class TestDetectModelType:
    """Tests for detect_model_type function."""

    def test_detect_llm_model(self, tmp_path):
        """Test detection of LLM model."""
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_qwen_model(self, tmp_path):
        """Test detection of Qwen model as LLM."""
        config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_embedding_model_by_type(self, tmp_path):
        """Test detection of embedding model by model_type."""
        config = {
            "model_type": "bert",
            "architectures": ["BertModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_embedding_model_by_architecture(self, tmp_path):
        """Test detection of embedding model by architecture."""
        config = {
            "model_type": "unknown",
            "architectures": ["XLMRobertaModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_modernbert_embedding(self, tmp_path):
        """Test detection of ModernBERT as embedding model."""
        config = {
            "model_type": "modernbert",
            "architectures": ["ModernBertModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_reranker_model(self, tmp_path):
        """Test detection of reranker model by architecture."""
        config = {
            "model_type": "modernbert",
            "architectures": ["ModernBertForSequenceClassification"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "reranker"

    def test_detect_xlm_roberta_reranker(self, tmp_path):
        """Test detection of XLM-RoBERTa reranker."""
        config = {
            "model_type": "xlm-roberta",
            "architectures": ["XLMRobertaForSequenceClassification"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "reranker"

    def test_detect_jina_reranker_without_name_heuristic(self, tmp_path):
        """JinaForRanking should detect as reranker without requiring 'rerank' in directory name."""
        model_dir = tmp_path / "jina-v3-mlx"
        model_dir.mkdir()
        config = {
            "model_type": "qwen3",
            "architectures": ["JinaForRanking"],
        }
        (model_dir / "config.json").write_text(json.dumps(config))
        assert detect_model_type(model_dir) == "reranker"

    def test_detect_causal_lm_reranker(self, tmp_path):
        """Test detection of CausalLM-based reranker (e.g., Qwen3-Reranker)."""
        reranker_dir = tmp_path / "Qwen3-Reranker-0.6B-mxfp8"
        reranker_dir.mkdir()
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }
        (reranker_dir / "config.json").write_text(json.dumps(config))
        assert detect_model_type(reranker_dir) == "reranker"

    def test_detect_causal_lm_embedding(self, tmp_path):
        """Test detection of CausalLM-based embedding (e.g., Qwen3-Embedding)."""
        embed_dir = tmp_path / "Qwen3-Embedding-8B-mxfp8"
        embed_dir.mkdir()
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }
        (embed_dir / "config.json").write_text(json.dumps(config))
        assert detect_model_type(embed_dir) == "embedding"

    def test_causal_lm_without_reranker_or_embedding_name_is_llm(self, tmp_path):
        """Test that Qwen3ForCausalLM without 'reranker' or 'embedding' in name is LLM."""
        llm_dir = tmp_path / "Qwen3-0.6B"
        llm_dir.mkdir()
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }
        (llm_dir / "config.json").write_text(json.dumps(config))
        assert detect_model_type(llm_dir) == "llm"

    def test_detect_qwen2_causal_lm_embedding(self, tmp_path):
        """Qwen2ForCausalLM with an embedding-named dir is an embedding (#686).

        jina-code-embeddings ships a Qwen2ForCausalLM architecture without
        lm_head weights, so it must classify as an embedding model rather than
        a chat LLM.
        """
        embed_dir = tmp_path / "jina-code-embeddings-1.5b"
        embed_dir.mkdir()
        config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
        }
        (embed_dir / "config.json").write_text(json.dumps(config))
        assert detect_model_type(embed_dir) == "embedding"

    def test_qwen2_causal_lm_without_embedding_name_is_llm(self, tmp_path):
        """Qwen2ForCausalLM without an embedding-named dir stays an LLM (#686).

        Regression guard: adding Qwen2ForCausalLM to the embedding-arch set must
        not reclassify ordinary Qwen2/Qwen2.5 chat models, which is gated by the
        _is_causal_lm_embedding directory-name heuristic.
        """
        llm_dir = tmp_path / "Qwen2.5-7B-Instruct"
        llm_dir.mkdir()
        config = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
        }
        (llm_dir / "config.json").write_text(json.dumps(config))
        assert detect_model_type(llm_dir) == "llm"

    def test_detect_lfm2_text_model_is_llm(self, tmp_path):
        """LFM2 text checkpoints share model_type with non-text variants."""
        llm_dir = tmp_path / "LFM2-1.2B"
        llm_dir.mkdir()
        config = {
            "model_type": "lfm2",
            "architectures": ["Lfm2ForCausalLM"],
        }
        (llm_dir / "config.json").write_text(json.dumps(config))
        assert detect_model_type(llm_dir) == "llm"

    def test_detect_lfm2_5_moe_text_model_is_llm(self, tmp_path):
        """LiquidAI/LFM2.5-8B-A1B is a text-generation MoE LLM."""
        llm_dir = tmp_path / "LFM2.5-8B-A1B"
        llm_dir.mkdir()
        config = {
            "model_type": "lfm2_moe",
            "architectures": ["Lfm2MoeForCausalLM"],
        }
        (llm_dir / "config.json").write_text(json.dumps(config))
        assert detect_model_type(llm_dir) == "llm"

    def test_detect_qwen3_vl_reranker(self, tmp_path):
        """Qwen3VLForConditionalGeneration + 'reranker' in dir name → reranker."""
        reranker_dir = tmp_path / "Qwen3-VL-Reranker-2B-4bit"
        reranker_dir.mkdir()
        config = {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "vision_config": {"hidden_size": 1024},
        }
        (reranker_dir / "config.json").write_text(json.dumps(config))
        assert detect_model_type(reranker_dir) == "reranker"

    def test_detect_qwen3_vl_embedding(self, tmp_path):
        """Qwen3VLForConditionalGeneration + 'embedding' in dir name → embedding."""
        embed_dir = tmp_path / "Qwen3-VL-Embedding-2B"
        embed_dir.mkdir()
        config = {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "vision_config": {"hidden_size": 1024},
        }
        (embed_dir / "config.json").write_text(json.dumps(config))
        assert detect_model_type(embed_dir) == "embedding"

    def test_detect_qwen3_vl_without_rerank_or_embed_name_is_vlm(self, tmp_path):
        """Plain Qwen3-VL without rerank/embed hints still classifies as VLM."""
        vlm_dir = tmp_path / "Qwen3-VL-2B-Instruct"
        vlm_dir.mkdir()
        config = {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "vision_config": {"hidden_size": 1024},
        }
        (vlm_dir / "config.json").write_text(json.dumps(config))
        assert detect_model_type(vlm_dir) == "vlm"

    def test_missing_config_defaults_to_llm(self, tmp_path):
        """Test that missing config.json defaults to LLM."""
        assert detect_model_type(tmp_path) == "llm"

    def test_invalid_json_defaults_to_llm(self, tmp_path):
        """Test that invalid JSON defaults to LLM."""
        (tmp_path / "config.json").write_text("not valid json")
        assert detect_model_type(tmp_path) == "llm"

    def test_empty_config_defaults_to_llm(self, tmp_path):
        """Test that empty config defaults to LLM."""
        (tmp_path / "config.json").write_text("{}")
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_vlm_by_model_type(self, tmp_path):
        """Test detection of VLM model by model_type."""
        config = {
            "model_type": "qwen2_5_vl",
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "vision_config": {"hidden_size": 1152},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_diffusion_gemma_as_vlm(self, tmp_path):
        """DiffusionGemma is served by mlx-vlm even without vision_config."""
        config = {
            "model_type": "diffusion_gemma",
            "canvas_length": 256,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_cohere2_moe_as_vlm_without_vision_config(self, tmp_path):
        """Cohere2 MoE is text-only but implemented by mlx-vlm."""
        config = {
            "model_type": "cohere2_moe",
            "architectures": ["Cohere2MoeForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_minimax_m3_vl_as_vlm(self, tmp_path):
        """MiniMax M3 VL is served by mlx-vlm."""
        config = {
            "model_type": "minimax_m3_vl",
            "architectures": ["MiniMaxM3VLForConditionalGeneration"],
            "vision_config": {"hidden_size": 1280},
            "text_config": {"hidden_size": 6144},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_minimax_m3_text_as_vlm_native_text(self, tmp_path):
        """MiniMax M3 text-only checkpoints are implemented by mlx-vlm."""
        config = {
            "model_type": "minimax_m3",
            "architectures": ["MiniMaxM3ForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_vlm_by_architecture(self, tmp_path):
        """Test detection of VLM model by architecture name."""
        config = {
            "model_type": "unknown_vlm",
            "architectures": ["LlavaForConditionalGeneration"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_vlm_by_vision_config(self, tmp_path):
        """Test detection of VLM model by vision_config + text_config."""
        config = {
            "model_type": "some_new_vlm",
            "vision_config": {"hidden_size": 1024},
            "text_config": {"hidden_size": 2048},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_vlm_gemma3(self, tmp_path):
        """Test detection of Gemma3 as VLM."""
        config = {
            "model_type": "gemma3",
            "architectures": ["Gemma3ForConditionalGeneration"],
            "vision_config": {"hidden_size": 1152},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_vlm_gemma4(self, tmp_path):
        """Test detection of Gemma4 as VLM."""
        config = {
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "vision_config": {"hidden_size": 1152},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_vlm_gemma4_unified(self, tmp_path):
        """Test detection of Gemma4 unified as VLM."""
        config = {
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "vision_config": {"hidden_size": 1152},
            "audio_config": {"feature_size": 128},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_text_only_gemma3_as_llm(self, tmp_path):
        """Text-only quant of Gemma3 (no vision_config) should be LLM."""
        config = {
            "model_type": "gemma3",
            "architectures": ["Gemma3ForConditionalGeneration"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_text_only_gemma4_as_llm(self, tmp_path):
        """Text-only quant of Gemma4 (no vision_config) should be LLM."""
        config = {
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_gemma4_unified_without_vision_config_as_vlm(self, tmp_path):
        """Gemma4 unified is always VLM even without vision_config."""
        config = {
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_vlm_qwen3_5_moe(self, tmp_path):
        """Test detection of Qwen3.5 MoE as VLM."""
        config = {
            "model_type": "qwen3_5_moe",
            "vision_config": {"depth": 32, "hidden_size": 1280},
            "text_config": {"hidden_size": 4096},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_text_only_qwen3_5_moe_as_llm(self, tmp_path):
        """Text-only quant of qwen3_5_moe (no vision_config) should be LLM."""
        config = {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_qwen3_causal_lm_is_llm(self, tmp_path):
        """Qwen3 with CausalLM architecture should be LLM, not embedding."""
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_qwen3_embedding_by_architecture(self, tmp_path):
        """Qwen3 with TextEmbedding architecture should be embedding."""
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForTextEmbedding"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "embedding"

    def test_detect_qwen3_no_architecture_defaults_to_llm(self, tmp_path):
        """Qwen3 without architectures field should default to LLM."""
        config = {
            "model_type": "qwen3",
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_gemma3_text_without_embedding_arch_is_llm(self, tmp_path):
        """gemma3_text with non-embedding architecture should be LLM."""
        config = {
            "model_type": "gemma3_text",
            "architectures": ["Gemma3TextForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_gemma3_text_model_without_sentence_transformers_modules_is_llm(self, tmp_path):
        """gemma3_text base transformer without sentence-transformers modules stays LLM."""
        config = {
            "model_type": "gemma3_text",
            "architectures": ["Gemma3TextModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_sentence_transformers_gemma3_text_as_embedding(self, tmp_path):
        """gemma3_text sentence-transformers exports should be detected as embeddings."""
        config = {
            "model_type": "gemma3_text",
            "architectures": ["Gemma3TextModel"],
        }
        modules = [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.models.Transformer",
            },
            {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.models.Pooling",
            },
            {
                "idx": 2,
                "name": "2",
                "path": "2_Normalize",
                "type": "sentence_transformers.models.Normalize",
            },
        ]
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "modules.json").write_text(json.dumps(modules))
        assert detect_model_type(tmp_path) == "embedding"

    def test_transformer_only_modules_json_is_not_embedding(self, tmp_path):
        """modules.json with only Transformer (no Pooling/Normalize) should not be embedding."""
        config = {
            "model_type": "gemma3_text",
            "architectures": ["Gemma3TextModel"],
        }
        modules = [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.models.Transformer",
            },
        ]
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "modules.json").write_text(json.dumps(modules))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_vlm_model_type_requires_vision_config(self, tmp_path):
        """VLM_MODEL_TYPES match without vision_config should fall back to LLM."""
        config = {
            "model_type": "gemma3",
            # No VLM architecture, no vision_config — text-only derivative
            "architectures": ["SomeTextOnlyArch"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_vlm_molmo2_via_vit_config(self, tmp_path):
        """Molmo2 bf16 ships ``vit_config`` (not ``vision_config``) — must classify as VLM."""
        config = {
            "model_type": "molmo2",
            "architectures": ["Molmo2ForConditionalGeneration"],
            "vit_config": {"hidden_size": 1024, "num_layers": 24},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_vlm_fastvlm_via_mm_vision_tower(self, tmp_path):
        """FastVLM bf16 ships ``mm_vision_tower`` (older LLaVA style) — must classify as VLM."""
        config = {
            "model_type": "llava_qwen2",
            "architectures": ["LlavaQwen2ForCausalLM"],
            "mm_vision_tower": "openai/clip-vit-large-patch14-336",
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "vlm"

    def test_detect_text_only_quant_with_empty_mm_vision_tower_as_llm(self, tmp_path):
        """``mm_vision_tower`` evidence check is non-empty-only — empty string falls back to LLM."""
        config = {
            "model_type": "llava_qwen2",
            "architectures": ["LlavaQwen2ForCausalLM"],
            "mm_vision_tower": "",
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_text_only_quant_no_vision_evidence_as_llm(self, tmp_path):
        """A VLM_ARCHITECTURE match without any vision sub-config evidence is a text-only quant."""
        config = {
            # qwen3_5_moe is in VLM_MODEL_TYPES, and the unsloth Qwen3.5-122B
            # text-only quant ships only the chat backbone arch.
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5_MoEForCausalLM"],
            # No vision_config / vit_config / mm_vision_tower.
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_lfm_text_moe_family_as_llm_not_audio_sts(self, tmp_path):
        """LFM text MoE (lfm*_moe + *ForCausalLM) must not use mlx-audio STS."""
        config = {
            "model_type": "lfm2_moe",
            "architectures": ["Lfm2MoeForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_lfm_future_moe_variant_as_llm_not_audio_sts(self, tmp_path):
        """Any lfm*_moe text type with *ForCausalLM uses LLM, not mlx-audio STS."""
        config = {
            "model_type": "lfm2_5_moe",
            "architectures": ["Lfm2MoeForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "llm"

    def test_detect_lfm_audio_architecture_as_sts(self, tmp_path):
        """mlx-audio LFM STS remains classified via LFM2AudioModel architecture."""
        config = {
            "model_type": "lfm_audio",
            "architectures": ["LFM2AudioModel"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "audio_sts"

    def test_detect_lfm_audio_model_type_as_sts(self, tmp_path):
        """lfm_audio model_type is STS even without a recognized architecture."""
        config = {
            "model_type": "lfm_audio",
            "architectures": [],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "audio_sts"

    def test_detect_unknown_lfm_prefix_without_causal_lm_as_sts(self, tmp_path):
        """Unknown mlx-audio lfm* types without CausalLM still use prefix fallback."""
        config = {
            "model_type": "lfm_custom_audio",
            "architectures": ["SomeLegacyLFMAudioWrapper"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert detect_model_type(tmp_path) == "audio_sts"


class TestEstimateModelSize:
    """Tests for estimate_model_size function."""

    def test_estimate_from_safetensors(self, tmp_path):
        """Test size estimation from safetensors files."""
        # Create dummy safetensors files
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"0" * 1000)
        (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"0" * 2000)

        size = estimate_model_size(tmp_path)
        # 3000 bytes + 5% overhead
        assert size == int(3000 * 1.05)

    def test_estimate_from_single_safetensors(self, tmp_path):
        """Test size estimation from single safetensors file."""
        (tmp_path / "model.safetensors").write_bytes(b"0" * 5000)

        size = estimate_model_size(tmp_path)
        assert size == int(5000 * 1.05)

    def test_estimate_from_bin_files(self, tmp_path):
        """Test size estimation from .bin files when no safetensors."""
        # Create dummy bin files
        (tmp_path / "pytorch_model.bin").write_bytes(b"0" * 5000)

        size = estimate_model_size(tmp_path)
        # 5000 bytes + 5% overhead
        assert size == int(5000 * 1.05)

    def test_skip_optimizer_files(self, tmp_path):
        """Test that optimizer files are skipped."""
        (tmp_path / "model.bin").write_bytes(b"0" * 1000)
        (tmp_path / "optimizer.bin").write_bytes(b"0" * 2000)

        size = estimate_model_size(tmp_path)
        # Only model.bin (1000 bytes) + 5% overhead
        assert size == int(1000 * 1.05)

    def test_skip_training_files(self, tmp_path):
        """Test that training files are skipped."""
        (tmp_path / "model.bin").write_bytes(b"0" * 1000)
        (tmp_path / "training_args.bin").write_bytes(b"0" * 500)

        size = estimate_model_size(tmp_path)
        assert size == int(1000 * 1.05)

    def test_no_weights_raises_error(self, tmp_path):
        """Test that missing weights raises ValueError."""
        with pytest.raises(ValueError, match="No model weights found"):
            estimate_model_size(tmp_path)

    def test_nested_safetensors(self, tmp_path):
        """Test size estimation from nested safetensors."""
        # Create subdirectory with safetensors
        subdir = tmp_path / "weights"
        subdir.mkdir()
        (subdir / "model.safetensors").write_bytes(b"0" * 3000)

        size = estimate_model_size(tmp_path)
        assert size == int(3000 * 1.05)

    def test_prefers_safetensors_over_bin(self, tmp_path):
        """Test that safetensors are preferred over bin files."""
        (tmp_path / "model.safetensors").write_bytes(b"0" * 1000)
        (tmp_path / "model.bin").write_bytes(b"0" * 5000)

        size = estimate_model_size(tmp_path)
        # Should use safetensors size only
        assert size == int(1000 * 1.05)


class TestDiscoverModels:
    """Tests for discover_models function."""

    def test_discover_single_model(self, tmp_path):
        """Test discovery of a single model."""
        model_dir = tmp_path / "llama-3b"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "llama-3b" in models
        assert models["llama-3b"].model_type == "llm"
        assert models["llama-3b"].engine_type == "batched"

    def test_register_model_skips_duplicate_id(self, tmp_path, caplog):
        """Collision guard: a second model with an already-registered model_id is
        skipped (first registration kept), not silently overwritten, and the
        collision is logged naming both the kept and the skipped path."""
        # A real, registerable model dir — without the guard this WOULD overwrite.
        second = tmp_path / "dup"
        second.mkdir()
        (second / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (second / "model.safetensors").write_bytes(b"0" * 1000)

        original = DiscoveredModel(
            model_id="dup",
            model_path="/first/dup",
            model_type="llm",
            engine_type="batched",
            estimated_size=123,
        )
        models = {"dup": original}
        with caplog.at_level(logging.WARNING):
            _register_model(models, second, "dup")

        # Guard kept the first registration rather than overwriting it.
        assert models["dup"] is original
        assert models["dup"].model_path == "/first/dup"
        # ...and surfaced the collision, naming both the kept and the skipped path.
        assert "Duplicate model_id 'dup'" in caplog.text
        assert "/first/dup" in caplog.text
        assert str(second) in caplog.text

    def test_discover_model_dir_is_itself_a_model(self, tmp_path):
        """Test that pointing directly at a model directory works."""
        model_dir = tmp_path / "Qwen3-5-9B-MLX-4bit"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(model_dir)
        assert len(models) == 1
        assert "Qwen3-5-9B-MLX-4bit" in models
        assert models["Qwen3-5-9B-MLX-4bit"].model_type == "llm"
        assert models["Qwen3-5-9B-MLX-4bit"].engine_type == "batched"

    def test_discover_no_fallback_when_subdirs_have_models(self, tmp_path):
        """Fallback should not trigger when subdirectory models exist,
        even if model_dir itself has config.json."""
        # model_dir itself looks like a model
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (tmp_path / "model.safetensors").write_bytes(b"0" * 1000)

        # but it also has a valid model subdirectory
        sub = tmp_path / "qwen-7b"
        sub.mkdir()
        (sub / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        (sub / "model.safetensors").write_bytes(b"0" * 2000)

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "qwen-7b" in models
        assert tmp_path.name not in models

    def test_discover_multiple_models(self, tmp_path):
        """Test discovery of multiple models."""
        # Create first LLM model
        llm_dir = tmp_path / "llama-3b"
        llm_dir.mkdir()
        (llm_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (llm_dir / "model.safetensors").write_bytes(b"0" * 1000)

        # Create second LLM model
        llm2_dir = tmp_path / "qwen-7b"
        llm2_dir.mkdir()
        (llm2_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        (llm2_dir / "model.safetensors").write_bytes(b"0" * 2000)

        models = discover_models(tmp_path)
        assert len(models) == 2
        assert models["llama-3b"].engine_type == "batched"
        assert models["qwen-7b"].engine_type == "batched"

    def test_discover_embedding_model(self, tmp_path):
        """Test discovery of embedding model with correct engine type."""
        emb_dir = tmp_path / "bge-small"
        emb_dir.mkdir()
        (emb_dir / "config.json").write_text(
            json.dumps({"model_type": "bert", "architectures": ["BertModel"]})
        )
        (emb_dir / "model.safetensors").write_bytes(b"0" * 500)

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert models["bge-small"].model_type == "embedding"
        assert models["bge-small"].engine_type == "embedding"

    def test_discover_reranker_model(self, tmp_path):
        """Test discovery of reranker model with correct engine type."""
        reranker_dir = tmp_path / "bge-reranker"
        reranker_dir.mkdir()
        (reranker_dir / "config.json").write_text(
            json.dumps({
                "model_type": "modernbert",
                "architectures": ["ModernBertForSequenceClassification"]
            })
        )
        (reranker_dir / "model.safetensors").write_bytes(b"0" * 500)

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert models["bge-reranker"].model_type == "reranker"
        assert models["bge-reranker"].engine_type == "reranker"

    def test_skip_invalid_directories(self, tmp_path):
        """Test that directories without config.json are skipped."""
        # Valid model
        valid_dir = tmp_path / "valid-model"
        valid_dir.mkdir()
        (valid_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (valid_dir / "model.safetensors").write_bytes(b"0" * 1000)

        # Invalid model (no config.json)
        invalid_dir = tmp_path / "invalid-model"
        invalid_dir.mkdir()
        (invalid_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "valid-model" in models
        assert "invalid-model" not in models

    def test_skip_hidden_directories(self, tmp_path):
        """Test that hidden directories are skipped."""
        # Hidden directory
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (hidden_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert len(models) == 0

    def test_skip_files(self, tmp_path):
        """Test that files are skipped (only directories processed)."""
        # Create a file at top level
        (tmp_path / "README.md").write_text("readme")

        # Create valid model
        model_dir = tmp_path / "llama-3b"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert len(models) == 1

    def test_nonexistent_directory_raises_error(self, tmp_path):
        """Test that nonexistent directory raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            discover_models(tmp_path / "nonexistent")

    def test_file_instead_of_directory_raises_error(self, tmp_path):
        """Test that file path raises ValueError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")
        with pytest.raises(ValueError, match="not a directory"):
            discover_models(file_path)

    def test_unreadable_directory_is_skipped(self, tmp_path, monkeypatch):
        """Test that unreadable model roots do not crash discovery."""
        original_iterdir = Path.iterdir

        def fake_iterdir(path):
            if path == tmp_path:
                raise PermissionError("Operation not permitted")
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", fake_iterdir)

        assert discover_models(tmp_path) == {}
        assert "not readable" in model_directory_access_error(tmp_path)

    def test_unreadable_directory_skipped_in_multi_dir_discovery(
        self, tmp_path, monkeypatch
    ):
        """Test that a readable directory still wins when another root is unreadable."""
        unreadable = tmp_path / "unreadable"
        unreadable.mkdir()
        readable = tmp_path / "readable"
        readable.mkdir()
        model_dir = readable / "valid-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        original_iterdir = Path.iterdir

        def fake_iterdir(path):
            if path == unreadable:
                raise PermissionError("Operation not permitted")
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", fake_iterdir)

        models = discover_models_from_dirs([unreadable, readable])
        assert list(models) == ["valid-model"]

    def test_model_with_weight_error_skipped(self, tmp_path):
        """Test that models with no weights are skipped."""
        # Model with config but no weights
        model_dir = tmp_path / "no-weights"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

        models = discover_models(tmp_path)
        assert len(models) == 0

    def test_discovered_model_fields(self, tmp_path):
        """Test that DiscoveredModel has all expected fields."""
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        model = models["test-model"]

        assert model.model_id == "test-model"
        assert model.model_path == str(model_dir)
        assert model.model_type == "llm"
        assert model.engine_type == "batched"
        assert model.estimated_size == int(1000 * 1.05)


class TestFormatSize:
    """Tests for format_size function."""

    def test_format_bytes(self):
        """Test formatting bytes."""
        assert format_size(500) == "500.00B"

    def test_format_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_size(1024) == "1.00KB"
        assert format_size(2048) == "2.00KB"

    def test_format_megabytes(self):
        """Test formatting megabytes."""
        assert format_size(1024 * 1024) == "1.00MB"
        assert format_size(5 * 1024 * 1024) == "5.00MB"

    def test_format_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_size(1024 * 1024 * 1024) == "1.00GB"
        assert format_size(32 * 1024 * 1024 * 1024) == "32.00GB"

    def test_format_terabytes(self):
        """Test formatting terabytes."""
        assert format_size(1024 * 1024 * 1024 * 1024) == "1.00TB"

    def test_format_petabytes(self):
        """Test formatting petabytes."""
        assert format_size(1024 * 1024 * 1024 * 1024 * 1024) == "1.00PB"


class TestAdapterDetection:
    """Tests for LoRA/PEFT adapter detection."""

    def test_adapter_dir_detected(self, tmp_path):
        """Directory with adapter_config.json is detected as adapter."""
        (tmp_path / "adapter_config.json").write_text("{}")
        assert _is_adapter_dir(tmp_path) is True

    def test_normal_model_not_adapter(self, tmp_path):
        """Normal model directory is not detected as adapter."""
        (tmp_path / "config.json").write_text('{"model_type": "llama"}')
        (tmp_path / "model.safetensors").write_bytes(b"0" * 1000)
        assert _is_adapter_dir(tmp_path) is False

    def test_discover_skips_lora_adapter(self, tmp_path):
        """discover_models should skip LoRA adapter directories."""
        # Normal model
        model_dir = tmp_path / "llama-3b"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        # LoRA adapter (has both config.json and adapter_config.json)
        adapter_dir = tmp_path / "my-lora"
        adapter_dir.mkdir()
        (adapter_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        (adapter_dir / "adapter_config.json").write_text("{}")
        (adapter_dir / "adapters.safetensors").write_bytes(b"0" * 100)

        models = discover_models(tmp_path)
        assert "llama-3b" in models
        assert "my-lora" not in models

    def test_discover_skips_nested_lora_adapter(self, tmp_path):
        """discover_models should skip LoRA adapters in org folders."""
        org_dir = tmp_path / "my-org"
        org_dir.mkdir()

        # Normal model under org
        model_dir = org_dir / "llama-3b"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (model_dir / "model.safetensors").write_bytes(b"0" * 1000)

        # LoRA adapter under org
        adapter_dir = org_dir / "my-lora"
        adapter_dir.mkdir()
        (adapter_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        (adapter_dir / "adapter_config.json").write_text("{}")

        models = discover_models(tmp_path)
        assert "llama-3b" in models
        assert "my-lora" not in models


class TestDiscoveredModel:
    """Tests for DiscoveredModel dataclass."""

    def test_create_discovered_model(self):
        """Test creating a DiscoveredModel."""
        model = DiscoveredModel(
            model_id="test-model",
            model_path="/path/to/model",
            model_type="llm",
            engine_type="batched",
            estimated_size=1024 * 1024 * 1024,  # 1GB
        )

        assert model.model_id == "test-model"
        assert model.model_path == "/path/to/model"
        assert model.model_type == "llm"
        assert model.engine_type == "batched"
        assert model.estimated_size == 1024 * 1024 * 1024

    def test_discovered_model_embedding(self):
        """Test DiscoveredModel for embedding type."""
        model = DiscoveredModel(
            model_id="emb-model",
            model_path="/path/to/emb",
            model_type="embedding",
            engine_type="embedding",
            estimated_size=500 * 1024 * 1024,
        )

        assert model.model_type == "embedding"
        assert model.engine_type == "embedding"


class TestReadModelContextLength:
    """Tests for _read_model_context_length — the discovery-time
    config.json reader that backs #1308's API exposure fix."""

    def _write(self, tmp_path, config=None, tokenizer_config=None):
        if config is not None:
            (tmp_path / "config.json").write_text(json.dumps(config))
        if tokenizer_config is not None:
            (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))

    def test_max_position_embeddings_wins(self, tmp_path):
        self._write(tmp_path, config={"max_position_embeddings": 262144})
        assert _read_model_context_length(tmp_path) == 262144

    def test_alternate_keys(self, tmp_path):
        for key in ("max_seq_len", "max_seq_length", "seq_length", "n_positions"):
            sub = tmp_path / key
            sub.mkdir()
            self._write(sub, config={key: 8192})
            assert _read_model_context_length(sub) == 8192, key

    def test_top_level_takes_precedence_over_nested(self, tmp_path):
        self._write(tmp_path, config={
            "max_position_embeddings": 200000,
            "text_config": {"max_position_embeddings": 32768},
        })
        assert _read_model_context_length(tmp_path) == 200000

    def test_text_config_fallback(self, tmp_path):
        """VLM wrappers stash the language head's ctx in text_config."""
        self._write(tmp_path, config={"text_config": {"max_position_embeddings": 131072}})
        assert _read_model_context_length(tmp_path) == 131072

    def test_language_config_fallback(self, tmp_path):
        """Some Qwen-style configs use language_config instead."""
        self._write(tmp_path, config={"language_config": {"max_position_embeddings": 65536}})
        assert _read_model_context_length(tmp_path) == 65536

    def test_tokenizer_max_length_fallback(self, tmp_path):
        """When config.json doesn't expose a length, tokenizer_config wins."""
        self._write(tmp_path, config={"model_type": "llama"}, tokenizer_config={"model_max_length": 4096})
        assert _read_model_context_length(tmp_path) == 4096

    def test_tokenizer_sentinel_rejected(self, tmp_path):
        """Transformers' int(1e30) "no cap" sentinel must NOT leak through."""
        self._write(
            tmp_path,
            config={"model_type": "llama"},
            tokenizer_config={"model_max_length": int(1e30)},
        )
        assert _read_model_context_length(tmp_path) is None

    def test_no_config_returns_none(self, tmp_path):
        assert _read_model_context_length(tmp_path) is None

    def test_invalid_types_rejected(self, tmp_path):
        """A string or zero in the config must not be returned as a length."""
        self._write(tmp_path, config={"max_position_embeddings": "long"})
        assert _read_model_context_length(tmp_path) is None
        sub = tmp_path / "zero"
        sub.mkdir()
        self._write(sub, config={"max_position_embeddings": 0})
        assert _read_model_context_length(sub) is None

    def test_malformed_config_is_silent(self, tmp_path):
        """A broken config.json must not raise — discovery should keep going."""
        (tmp_path / "config.json").write_text("{not valid json")
        assert _read_model_context_length(tmp_path) is None


class TestTwoLevelDiscovery:
    """Tests for two-level model discovery."""

    def _make_model(self, path: Path, model_type: str = "llama"):
        """Helper to create a valid model directory."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"model_type": model_type}))
        (path / "model.safetensors").write_bytes(b"0" * 1000)

    def test_two_level_org_folder(self, tmp_path):
        """Test discovery through organization folders."""
        self._make_model(tmp_path / "mlx-community" / "llama-3b")
        self._make_model(tmp_path / "mlx-community" / "qwen-7b")

        models = discover_models(tmp_path)
        assert len(models) == 2
        assert "llama-3b" in models
        assert "qwen-7b" in models

    def test_mixed_flat_and_org(self, tmp_path):
        """Test mix of flat models and organization folders."""
        # Flat model at level 1
        self._make_model(tmp_path / "mistral-7b")
        # Org folder with models at level 2
        self._make_model(tmp_path / "Qwen" / "Qwen3-8B")

        models = discover_models(tmp_path)
        assert len(models) == 2
        assert "mistral-7b" in models
        assert "Qwen3-8B" in models

    def test_multiple_org_folders(self, tmp_path):
        """Test multiple organization folders."""
        self._make_model(tmp_path / "mlx-community" / "llama-3b")
        self._make_model(tmp_path / "Qwen" / "Qwen3-8B")
        self._make_model(tmp_path / "GLM" / "glm-4")

        models = discover_models(tmp_path)
        assert len(models) == 3

    def test_empty_org_folder_skipped(self, tmp_path):
        """Test that empty org folders are silently skipped."""
        self._make_model(tmp_path / "valid-model")
        (tmp_path / "empty-org").mkdir()

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "valid-model" in models

    def test_org_folder_hidden_children_skipped(self, tmp_path):
        """Test that hidden subdirs inside org folders are skipped."""
        org = tmp_path / "mlx-community"
        self._make_model(org / "llama-3b")
        self._make_model(org / ".hidden-model")

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "llama-3b" in models

    def test_org_folder_invalid_children_skipped(self, tmp_path):
        """Test that children without config.json in org folders are skipped."""
        org = tmp_path / "mlx-community"
        self._make_model(org / "llama-3b")
        # Child without config.json
        no_config = org / "broken-model"
        no_config.mkdir(parents=True)
        (no_config / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert len(models) == 1

    def test_two_level_model_path_is_correct(self, tmp_path):
        """Test that model_path points to the actual model dir, not the org."""
        self._make_model(tmp_path / "mlx-community" / "llama-3b")

        models = discover_models(tmp_path)
        assert models["llama-3b"].model_path == str(
            tmp_path / "mlx-community" / "llama-3b"
        )


class TestDiscoverModelsFromDirs:
    """Tests for discover_models_from_dirs function."""

    def _make_model(self, path, model_type="llama"):
        """Helper to create a mock model directory."""
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "model_type": model_type,
            "architectures": ["LlamaForCausalLM"],
        }
        (path / "config.json").write_text(json.dumps(config))
        # Create a small safetensors file
        (path / "model.safetensors").write_bytes(b"\x00" * 100)

    def test_multiple_dirs(self, tmp_path):
        """Test discovering models from multiple directories."""
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        self._make_model(dir_a / "model-1")
        self._make_model(dir_b / "model-2")

        models = discover_models_from_dirs([dir_a, dir_b])
        assert "model-1" in models
        assert "model-2" in models
        assert len(models) == 2

    def test_first_directory_wins_on_conflict(self, tmp_path):
        """Test that first directory takes priority on model_id conflicts."""
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        self._make_model(dir_a / "same-model")
        self._make_model(dir_b / "same-model")

        models = discover_models_from_dirs([dir_a, dir_b])
        assert len(models) == 1
        assert models["same-model"].model_path == str(dir_a / "same-model")

    def test_empty_list(self, tmp_path):
        """Test with empty directory list."""
        models = discover_models_from_dirs([])
        assert models == {}

    def test_nonexistent_directory_skipped(self, tmp_path):
        """Test that non-existent directories are skipped with warning."""
        dir_a = tmp_path / "dir_a"
        self._make_model(dir_a / "model-1")
        nonexistent = tmp_path / "does_not_exist"

        models = discover_models_from_dirs([dir_a, nonexistent])
        assert "model-1" in models
        assert len(models) == 1

    def test_mixed_valid_invalid_dirs(self, tmp_path):
        """Test with a mix of valid, empty, and non-existent directories."""
        dir_valid = tmp_path / "valid"
        dir_empty = tmp_path / "empty"
        dir_empty.mkdir(parents=True)
        self._make_model(dir_valid / "model-1")

        models = discover_models_from_dirs(
            [dir_valid, dir_empty, tmp_path / "nonexistent"]
        )
        assert "model-1" in models
        assert len(models) == 1


class TestUnsupportedModels:
    """Tests for _is_unsupported_model() — audio models are now supported."""

    def test_whisper_not_unsupported(self, tmp_path):
        """Whisper is now an audio_stt model, not unsupported."""
        config = {
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert _is_unsupported_model(tmp_path) is False

    def test_whisper_model_type_not_unsupported(self, tmp_path):
        """Whisper by model_type alone is not unsupported."""
        config = {
            "model_type": "whisper",
            "architectures": ["SomeCustomWhisperArch"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert _is_unsupported_model(tmp_path) is False

    def test_tts_not_unsupported(self, tmp_path):
        """qwen3_tts is now an audio_tts model, not unsupported."""
        config = {
            "model_type": "qwen3_tts",
            "architectures": ["Qwen3TTSForConditionalGeneration"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert _is_unsupported_model(tmp_path) is False

    def test_multimodal_with_audio_not_unsupported(self, tmp_path):
        """Multimodal model with nested audio_config is NOT unsupported."""
        config = {
            "model_type": "minicpmo",
            "architectures": ["MiniCPMO"],
            "vision_config": {"model_type": "siglip_vision_model"},
            "audio_config": {"model_type": "whisper"},
            "tts_config": {"model_type": "minicpmtts"},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert _is_unsupported_model(tmp_path) is False

    def test_llm_not_unsupported(self, tmp_path):
        """Regular LLM is not unsupported."""
        config = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert _is_unsupported_model(tmp_path) is False

    def test_audio_models_included_in_discovery(self, tmp_path):
        """Audio models are now discovered (not skipped) with correct types."""
        # Create a normal LLM model
        llm_dir = tmp_path / "llama-3b"
        llm_dir.mkdir()
        (llm_dir / "config.json").write_text(
            json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
        )
        (llm_dir / "model.safetensors").write_bytes(b"0" * 1000)

        # Create a whisper ASR model
        asr_dir = tmp_path / "whisper-large-v3"
        asr_dir.mkdir()
        (asr_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "whisper",
                    "architectures": ["WhisperForConditionalGeneration"],
                }
            )
        )
        (asr_dir / "model.safetensors").write_bytes(b"0" * 2000)

        # Create a TTS model
        tts_dir = tmp_path / "Qwen3-TTS"
        tts_dir.mkdir()
        (tts_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen3_tts"})
        )
        (tts_dir / "model.safetensors").write_bytes(b"0" * 1500)

        models = discover_models(tmp_path)
        assert len(models) == 3
        assert "llama-3b" in models
        assert "whisper-large-v3" in models
        assert models["whisper-large-v3"].model_type == "audio_stt"
        assert "Qwen3-TTS" in models
        assert models["Qwen3-TTS"].model_type == "audio_tts"


class TestHfCacheDiscovery:
    """Tests for HF Hub cache entry resolution and discovery."""

    FAKE_COMMIT = "abc123def456"

    def _make_model(self, path: Path, model_type: str = "llama"):
        """Helper to create a valid model directory."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"model_type": model_type}))
        (path / "model.safetensors").write_bytes(b"0" * 1000)

    def _make_hf_cache_entry(self, parent: Path, org: str, name: str):
        """Helper to create a bare HF Hub cache directory layout (no model files)."""
        entry = parent / f"models--{org}--{name}"
        refs = entry / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text(self.FAKE_COMMIT)
        snapshot = entry / "snapshots" / self.FAKE_COMMIT
        snapshot.mkdir(parents=True)
        return entry, snapshot

    def _make_hf_cache_model(self, parent: Path, org: str, name: str, model_type: str = "llama"):
        """Helper to create an HF cache entry with a valid model in the snapshot."""
        _, snapshot = self._make_hf_cache_entry(parent, org, name)
        (snapshot / "config.json").write_text(json.dumps({"model_type": model_type}))
        (snapshot / "model.safetensors").write_bytes(b"0" * 1000)

    def test_resolve_valid_entry(self, tmp_path):
        """Valid HF cache entry resolves to snapshot path and repo metadata."""
        entry, snapshot = self._make_hf_cache_entry(tmp_path, "mlx-community", "Qwen3-8B-4bit")

        result = _resolve_hf_cache_entry(entry)
        assert result is not None
        assert result.snapshot_path == snapshot
        assert result.model_id == "mlx-community--Qwen3-8B-4bit"
        assert result.source_repo_id == "mlx-community/Qwen3-8B-4bit"

    def test_resolve_regular_dir_returns_none(self, tmp_path):
        """Regular directory without models-- prefix returns None."""
        regular = tmp_path / "mlx-community"
        regular.mkdir()
        assert _resolve_hf_cache_entry(regular) is None

    def test_resolve_unnamespaced_repo(self, tmp_path):
        """models--Name (no org separator) is supported."""
        entry, _ = self._make_hf_cache_entry(tmp_path, "", "gpt2")
        entry.rename(tmp_path / "models--gpt2")
        entry = tmp_path / "models--gpt2"
        snapshot = entry / "snapshots" / self.FAKE_COMMIT

        result = _resolve_hf_cache_entry(entry)
        assert result is not None
        assert result.snapshot_path == snapshot
        assert result.model_id == "gpt2"
        assert result.source_repo_id == "gpt2"

    def test_resolve_missing_refs_main_returns_none(self, tmp_path):
        """Missing refs and snapshots returns None."""
        entry = tmp_path / "models--mlx-community--Qwen3-8B"
        entry.mkdir(parents=True)
        assert _resolve_hf_cache_entry(entry) is None

    def test_resolve_missing_snapshot_returns_none(self, tmp_path):
        """Valid refs/main but missing snapshot directory returns None."""
        entry = tmp_path / "models--mlx-community--Qwen3-8B"
        refs = entry / "refs"
        refs.mkdir(parents=True)
        (refs / "main").write_text("deadbeef")
        assert _resolve_hf_cache_entry(entry) is None

    def test_resolve_strips_whitespace_from_refs(self, tmp_path):
        """Trailing newline in refs/main is stripped (matches real HF cache)."""
        entry, snapshot = self._make_hf_cache_entry(tmp_path, "mlx-community", "Qwen3-8B")
        # Overwrite with trailing newline (like real HF cache)
        (entry / "refs" / "main").write_text(self.FAKE_COMMIT + "\n")

        result = _resolve_hf_cache_entry(entry)
        assert result is not None
        assert result.snapshot_path == snapshot

    def test_discover_hf_cache_model(self, tmp_path):
        """HF cache entries are discovered as models."""
        self._make_hf_cache_model(tmp_path, "mlx-community", "Qwen3-8B-4bit")

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "mlx-community--Qwen3-8B-4bit" in models
        assert models["mlx-community--Qwen3-8B-4bit"].model_type == "llm"
        assert models["mlx-community--Qwen3-8B-4bit"].source_type == "hf_cache"
        assert (
            models["mlx-community--Qwen3-8B-4bit"].source_repo_id
            == "mlx-community/Qwen3-8B-4bit"
        )

    def test_discover_multiple_hf_cache_models(self, tmp_path):
        """Multiple HF cache entries are all discovered."""
        self._make_hf_cache_model(tmp_path, "mlx-community", "Qwen3-8B-4bit")
        self._make_hf_cache_model(tmp_path, "mlx-community", "Mistral-7B-v0.3")

        models = discover_models(tmp_path)
        assert len(models) == 2
        assert "mlx-community--Qwen3-8B-4bit" in models
        assert "mlx-community--Mistral-7B-v0.3" in models

    def test_hf_cache_model_path_points_to_snapshot(self, tmp_path):
        """model_path points to the snapshot dir, not the cache entry."""
        self._make_hf_cache_model(tmp_path, "mlx-community", "Qwen3-8B-4bit")

        models = discover_models(tmp_path)
        assert models["mlx-community--Qwen3-8B-4bit"].model_path == str(
            tmp_path / "models--mlx-community--Qwen3-8B-4bit" / "snapshots" / self.FAKE_COMMIT
        )

    def test_hf_cache_without_config_json_skipped(self, tmp_path):
        """HF cache entries without config.json in snapshot are skipped."""
        self._make_hf_cache_entry(tmp_path, "mlx-community", "NoConfig")

        models = discover_models(tmp_path)
        assert len(models) == 0

    def test_hf_cache_non_mlx_repo_skipped(self, tmp_path):
        """HF cache entries without MLX metadata or repo-name hints are skipped."""
        self._make_hf_cache_model(tmp_path, "acme", "PlainModel")

        models = discover_models(tmp_path)
        assert len(models) == 0

    def test_hf_cache_bad_helper_schema_is_skipped(self, tmp_path):
        """A malformed helper-looking schema must not abort HF cache discovery."""
        _, snapshot = self._make_hf_cache_entry(tmp_path, "acme", "BadSchema")
        (snapshot / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "architectures": 123})
        )
        (snapshot / "model.safetensors").write_bytes(b"0" * 1000)

        models = discover_models(tmp_path)
        assert models == {}

    def test_hf_cache_mlx_metadata_is_discovered(self, tmp_path):
        """HF cache entries with safetensors format=mlx metadata are discovered."""
        np = pytest.importorskip("numpy")
        safetensors_numpy = pytest.importorskip("safetensors.numpy")

        entry, snapshot = self._make_hf_cache_entry(tmp_path, "acme", "PlainModel")
        (snapshot / "config.json").write_text(json.dumps({"model_type": "llama"}))
        safetensors_numpy.save_file(
            {"weight": np.zeros((1,), dtype=np.float32)},
            snapshot / "model.safetensors",
            metadata={"format": "mlx"},
        )

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "acme--PlainModel" in models
        assert models["acme--PlainModel"].source_repo_id == "acme/PlainModel"

    def test_hf_cache_dflash_draft_is_discovered(self, tmp_path):
        """HF cache DFlash draft checkpoints are discovered despite pt-format
        safetensors and a repo name with no 'mlx' token (#1643)."""
        np = pytest.importorskip("numpy")
        safetensors_numpy = pytest.importorskip("safetensors.numpy")

        entry, snapshot = self._make_hf_cache_entry(
            tmp_path, "z-lab", "Qwen3.6-27B-DFlash"
        )
        (snapshot / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3",
                    "architectures": ["DFlashDraftModel"],
                    "dflash_config": {},
                }
            )
        )
        # pt-format safetensors (NOT mlx) so only the DFlash architecture — not
        # safetensors metadata or the repo name — can qualify this entry.
        safetensors_numpy.save_file(
            {"weight": np.zeros((1,), dtype=np.float32)},
            snapshot / "model.safetensors",
            metadata={"format": "pt"},
        )

        models = discover_models(tmp_path)
        assert "z-lab--Qwen3.6-27B-DFlash" in models
        # The shared DiscoveredModel construction path must flag the draft as
        # a helper so it lands in the draft picker, not the chat-model list.
        assert models["z-lab--Qwen3.6-27B-DFlash"].is_helper is True

    def test_is_helper_checkpoint_on_disk_edge_cases(self, tmp_path):
        """The on-disk wrapper qualifies a drafter config and never raises on
        unreadable entries (it runs outside discover_models' per-model guard).

        Marker-family coverage lives in TestIsHelperModelConfig; this pins
        only the wrapper's own file-handling behavior.
        """
        draft = tmp_path / "draft"
        draft.mkdir()
        (draft / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "architectures": ["DFlashDraftModel"]})
        )
        assert _is_helper_checkpoint(draft) is True

        # plain (non-draft) model config stays excluded
        plain = tmp_path / "plain"
        plain.mkdir()
        (plain / "config.json").write_text(
            json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
        )
        assert _is_helper_checkpoint(plain) is False

        # no config.json at all
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _is_helper_checkpoint(empty) is False

        # malformed JSON must not raise
        bad_json = tmp_path / "bad-json"
        bad_json.mkdir()
        (bad_json / "config.json").write_text("{not json")
        assert _is_helper_checkpoint(bad_json) is False

        # non-UTF-8 bytes must not raise: UnicodeDecodeError is a ValueError,
        # not a JSONDecodeError, and an escape here aborts the whole scan
        bad_bytes = tmp_path / "bad-bytes"
        bad_bytes.mkdir()
        (bad_bytes / "config.json").write_bytes(b'\xff\xfe{"a": 1}')
        assert _is_helper_checkpoint(bad_bytes) is False

        # valid JSON with an unexpected schema must not raise either
        bad_schema = tmp_path / "bad-schema"
        bad_schema.mkdir()
        (bad_schema / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "architectures": 123})
        )
        assert _is_helper_checkpoint(bad_schema) is False

    def test_mixed_flat_and_hf_cache(self, tmp_path):
        """Mix of flat models and HF cache entries."""
        self._make_model(tmp_path / "mistral-7b")
        self._make_hf_cache_model(tmp_path, "mlx-community", "Qwen3-8B-4bit")

        models = discover_models(tmp_path)
        assert len(models) == 2
        assert "mistral-7b" in models
        assert "mlx-community--Qwen3-8B-4bit" in models

    def test_mixed_org_and_hf_cache(self, tmp_path):
        """Mix of org folders and HF cache entries."""
        self._make_model(tmp_path / "Qwen" / "Qwen3-8B", model_type="qwen2")
        self._make_hf_cache_model(tmp_path, "mlx-community", "Mistral-7B")

        models = discover_models(tmp_path)
        assert len(models) == 2
        assert "Qwen3-8B" in models
        assert "mlx-community--Mistral-7B" in models

    def test_hf_cache_does_not_fall_through_to_org_scan(self, tmp_path):
        """HF cache entries don't get scanned as org folders."""
        self._make_hf_cache_model(tmp_path, "mlx-community", "Qwen3-8B-4bit")

        models = discover_models(tmp_path)
        assert len(models) == 1
        assert "mlx-community--Qwen3-8B-4bit" in models
