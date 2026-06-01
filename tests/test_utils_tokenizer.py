# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.tokenizer module."""

import pytest

from omlx.utils.tokenizer import (
    apply_qwen3_fix,
    get_tokenizer_config,
    is_gemma4_model,
    is_harmony_model,
    is_qwen3_model,
    resolve_vocab_size,
)


class TestIsHarmonyModel:
    """Test cases for is_harmony_model function."""

    def test_harmony_model_via_config_model_type(self):
        """Test detection via config.model_type == 'gpt_oss'."""
        config = {"model_type": "gpt_oss"}
        assert is_harmony_model("some-model", config) is True

    def test_harmony_model_via_name_gpt_oss(self):
        """Test detection via model name containing 'gpt-oss'."""
        assert is_harmony_model("gpt-oss-1.0", None) is True
        assert is_harmony_model("GPT-OSS-v2", None) is True
        assert is_harmony_model("my-gpt-oss-model", None) is True

    def test_harmony_model_via_name_gptoss(self):
        """Test detection via model name containing 'gptoss'."""
        assert is_harmony_model("gptoss", None) is True
        assert is_harmony_model("GPTOSS-large", None) is True
        assert is_harmony_model("my-gptoss", None) is True

    def test_not_harmony_model(self):
        """Test non-Harmony models return False."""
        assert is_harmony_model("llama-3.1-8b", None) is False
        assert is_harmony_model("qwen2.5-32b", None) is False
        assert is_harmony_model("mistral-7b", None) is False

    def test_not_harmony_with_different_model_type(self):
        """Test non-Harmony model type returns False."""
        config = {"model_type": "llama"}
        assert is_harmony_model("some-model", config) is False

    def test_harmony_model_empty_name(self):
        """Test with empty model name."""
        assert is_harmony_model("", None) is False

    def test_harmony_model_none_config(self):
        """Test with None config."""
        assert is_harmony_model("gpt-oss", None) is True
        assert is_harmony_model("llama", None) is False

    def test_harmony_model_empty_config(self):
        """Test with empty config dict."""
        assert is_harmony_model("gpt-oss", {}) is True
        assert is_harmony_model("llama", {}) is False


class TestIsGemma4Model:
    """Test cases for is_gemma4_model function."""

    def test_gemma4_model_via_config_model_type(self):
        config = {"model_type": "gemma4"}
        assert is_gemma4_model("some-model", config) is True

    def test_gemma4_model_via_name(self):
        assert is_gemma4_model("google/gemma-4b", None) is True
        assert is_gemma4_model("GEMMA-4-27B", None) is True
        assert is_gemma4_model("my-gemma4-model", None) is True

    def test_not_gemma4_model(self):
        assert is_gemma4_model("gemma-3-27b", None) is False
        assert is_gemma4_model("llama-3.1-8b", None) is False

    def test_not_gemma4_with_different_model_type(self):
        config = {"model_type": "gemma"}
        assert is_gemma4_model("some-model", config) is False


class TestIsQwen3Model:
    """Test cases for is_qwen3_model function."""

    def test_qwen3_lowercase(self):
        """Test detection with lowercase 'qwen3'."""
        assert is_qwen3_model("qwen3-8b") is True
        assert is_qwen3_model("my-qwen3-model") is True
        assert is_qwen3_model("qwen3") is True

    def test_qwen3_mixed_case(self):
        """Test detection with mixed case 'Qwen3'."""
        assert is_qwen3_model("Qwen3-8B") is True
        assert is_qwen3_model("My-Qwen3-Model") is True
        assert is_qwen3_model("Qwen3") is True

    def test_not_qwen3(self):
        """Test non-Qwen3 models return False."""
        assert is_qwen3_model("qwen2.5-32b") is False
        assert is_qwen3_model("Qwen2-7B") is False
        assert is_qwen3_model("llama-3.1") is False
        assert is_qwen3_model("qwen-7b") is False

    def test_qwen3_empty_name(self):
        """Test with empty model name."""
        assert is_qwen3_model("") is False

    def test_qwen3_partial_match(self):
        """Test that partial matches don't trigger false positives."""
        # 'qwen30' should NOT match as Qwen3
        # However, current implementation will match it since 'qwen3' is in 'qwen30'
        # This test documents the current behavior
        assert is_qwen3_model("qwen30-model") is True  # Contains 'qwen3'


class TestGetTokenizerConfig:
    """Test cases for get_tokenizer_config function."""

    def test_basic_config(self):
        """Test basic config generation."""
        config = get_tokenizer_config("llama-3.1-8b")
        assert "trust_remote_code" in config
        assert config["trust_remote_code"] is False

    def test_config_with_trust_remote_code(self):
        """Test config with trust_remote_code enabled."""
        config = get_tokenizer_config("some-model", trust_remote_code=True)
        assert config["trust_remote_code"] is True

    def test_qwen3_model_config(self):
        """Test Qwen3 model gets eos_token fix."""
        config = get_tokenizer_config("qwen3-8b")
        assert config["eos_token"] == "<|im_end|>"

    def test_non_qwen3_model_no_eos_fix(self):
        """Test non-Qwen3 models don't get eos_token."""
        config = get_tokenizer_config("llama-3.1-8b")
        assert "eos_token" not in config

    def test_qwen3_with_trust_remote_code(self):
        """Test Qwen3 model with trust_remote_code."""
        config = get_tokenizer_config("Qwen3-72B", trust_remote_code=True)
        assert config["trust_remote_code"] is True
        assert config["eos_token"] == "<|im_end|>"


class TestApplyQwen3Fix:
    """Test cases for apply_qwen3_fix function."""

    def test_apply_fix_to_qwen3(self):
        """Test applying Qwen3 fix."""
        config = {"trust_remote_code": True}
        result = apply_qwen3_fix(config, "qwen3-8b")
        assert result["eos_token"] == "<|im_end|>"
        assert result["trust_remote_code"] is True

    def test_no_fix_for_non_qwen3(self):
        """Test no fix applied for non-Qwen3 models."""
        config = {"trust_remote_code": True}
        result = apply_qwen3_fix(config, "llama-3.1-8b")
        assert "eos_token" not in result
        assert result["trust_remote_code"] is True

    def test_apply_fix_modifies_original(self):
        """Test that apply_qwen3_fix modifies the original config."""
        config = {"trust_remote_code": True}
        result = apply_qwen3_fix(config, "qwen3-8b")
        # The function modifies in place and returns the same dict
        assert config is result
        assert config["eos_token"] == "<|im_end|>"

    def test_apply_fix_overwrites_existing_eos(self):
        """Test that apply_qwen3_fix overwrites existing eos_token."""
        config = {"eos_token": "<|endoftext|>"}
        result = apply_qwen3_fix(config, "qwen3-8b")
        assert result["eos_token"] == "<|im_end|>"

    def test_apply_fix_empty_config(self):
        """Test applying fix to empty config."""
        config = {}
        result = apply_qwen3_fix(config, "qwen3-8b")
        assert result["eos_token"] == "<|im_end|>"

    def test_apply_fix_preserves_other_keys(self):
        """Test that apply_qwen3_fix preserves other config keys."""
        config = {
            "trust_remote_code": True,
            "use_fast": True,
            "padding_side": "left",
        }
        result = apply_qwen3_fix(config, "qwen3-8b")
        assert result["trust_remote_code"] is True
        assert result["use_fast"] is True
        assert result["padding_side"] == "left"
        assert result["eos_token"] == "<|im_end|>"


def _stub_lm_head(vocab_size: int):
    """Build a stub lm_head whose weight has shape (vocab_size, hidden)."""
    from types import SimpleNamespace

    return SimpleNamespace(weight=SimpleNamespace(shape=(vocab_size, 16)))


class TestResolveVocabSize:
    """Test cases for resolve_vocab_size function.

    Covers the three-level resolution order:
      1. lm_head.weight.shape[0] (authoritative)
      2. text_config.vocab_size (inner LM vocab on VLM composite)
      3. config.vocab_size / args.vocab_size (top-level fallback)
    """

    def test_returns_none_for_none_model(self):
        """resolve_vocab_size(None) returns None without crashing."""
        assert resolve_vocab_size(None) is None

    def test_lm_head_authoritative_pure_llm(self):
        """Pure LLM with model.lm_head: returns lm_head's first dim."""
        from types import SimpleNamespace

        model = SimpleNamespace(
            lm_head=_stub_lm_head(32000),
            config=SimpleNamespace(vocab_size=32000),
        )
        assert resolve_vocab_size(model) == 32000

    def test_lm_head_authoritative_via_language_model(self):
        """Raw mlx-vlm wrapping: model.language_model.lm_head."""
        from types import SimpleNamespace

        model = SimpleNamespace(
            language_model=SimpleNamespace(lm_head=_stub_lm_head(151552)),
            config=SimpleNamespace(vocab_size=257152),
        )
        assert resolve_vocab_size(model) == 151552

    def test_lm_head_authoritative_via_underscore_language_model(self):
        """oMLX VLM-adapter wrapping: model._language_model.lm_head."""
        from types import SimpleNamespace

        model = SimpleNamespace(
            _language_model=SimpleNamespace(lm_head=_stub_lm_head(151552)),
            config=SimpleNamespace(vocab_size=257152),
        )
        assert resolve_vocab_size(model) == 151552

    def test_lm_head_overrides_wrong_top_level_vocab_size(self):
        """The bug case: GLM-4.6V-style mismatch where ModelConfig
        defaults to 257152 but actual logits vocab is 151552. The
        lm_head probe wins over the top-level config value.
        """
        from types import SimpleNamespace

        model = SimpleNamespace(
            language_model=SimpleNamespace(lm_head=_stub_lm_head(151552)),
            config=SimpleNamespace(
                vocab_size=257152,
                text_config=SimpleNamespace(vocab_size=151552),
            ),
        )
        assert resolve_vocab_size(model) == 151552

    def test_text_config_fallback_when_lm_head_missing(self):
        """No lm_head exposed anywhere: fall back to text_config.vocab_size."""
        from types import SimpleNamespace

        model = SimpleNamespace(
            config=SimpleNamespace(
                vocab_size=257152,
                text_config=SimpleNamespace(vocab_size=151552),
            ),
        )
        assert resolve_vocab_size(model) == 151552

    def test_text_config_dict_fallback(self):
        """text_config can be a dict (some configs use dataclass-as-dict)."""
        from types import SimpleNamespace

        model = SimpleNamespace(
            config=SimpleNamespace(
                vocab_size=None,
                text_config={"vocab_size": 151552},
            ),
        )
        assert resolve_vocab_size(model) == 151552

    def test_top_level_config_fallback(self):
        """No lm_head, no text_config: top-level config.vocab_size."""
        from types import SimpleNamespace

        model = SimpleNamespace(config=SimpleNamespace(vocab_size=32000))
        assert resolve_vocab_size(model) == 32000

    def test_args_attr_fallback(self):
        """Older mlx-lm models expose vocab via args, not config."""
        from types import SimpleNamespace

        model = SimpleNamespace(args=SimpleNamespace(vocab_size=32000))
        assert resolve_vocab_size(model) == 32000

    def test_malformed_shape_falls_back_gracefully(self):
        """A weight whose .shape is non-subscriptable (e.g. a bare object)
        must not crash; the resolver should fall back to config.

        Bad shape lives under `_language_model.lm_head` (the first wrapping
        path probed) so the test directly exercises the try/except guard
        for shape=object() rather than falling through earlier paths
        whose shape=None happens to short-circuit safely.
        """
        from types import SimpleNamespace

        model = SimpleNamespace(
            _language_model=SimpleNamespace(
                lm_head=SimpleNamespace(weight=SimpleNamespace(shape=object()))
            ),
            config=SimpleNamespace(vocab_size=32000),
        )
        assert resolve_vocab_size(model) == 32000

    def test_none_shape_falls_back_gracefully(self):
        """Weight present but .shape is None: skip lm_head, fall back."""
        from types import SimpleNamespace

        model = SimpleNamespace(
            lm_head=SimpleNamespace(weight=SimpleNamespace(shape=None)),
            config=SimpleNamespace(vocab_size=32000),
        )
        assert resolve_vocab_size(model) == 32000

    def test_empty_shape_falls_back_gracefully(self):
        """Weight present but .shape is empty: skip lm_head, fall back."""
        from types import SimpleNamespace

        model = SimpleNamespace(
            lm_head=SimpleNamespace(weight=SimpleNamespace(shape=())),
            config=SimpleNamespace(vocab_size=32000),
        )
        assert resolve_vocab_size(model) == 32000

    def test_returns_none_when_all_sources_unavailable(self):
        """No lm_head, no config, no args: returns None."""
        from types import SimpleNamespace

        model = SimpleNamespace()
        assert resolve_vocab_size(model) is None
