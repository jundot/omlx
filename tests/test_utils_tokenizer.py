# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.tokenizer module."""

import json

from omlx.utils.tokenizer import (
    apply_qwen3_fix,
    create_streaming_detokenizer,
    get_tokenizer_config,
    is_gemma4_model,
    is_harmony_model,
    is_qwen3_model,
    repair_misconverted_unlimited_ocr_tokenizer,
)


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _spm_decoder(strip_space=True):
    decoders = [
        {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
    ]
    if strip_space:
        decoders.append({"type": "Strip", "content": " ", "start": 1, "stop": 0})
    return {"type": "Sequence", "decoders": decoders}


class _ByteFallbackTokenizer:
    clean_up_tokenization_spaces = False
    vocab = {
        "<pad>": 0,
        "<0xEC>": 1,
        "<0x9E>": 2,
        "<0xA0>": 3,
    }

    def decode(self, token_ids, skip_special_tokens: bool = True):
        table = {
            0: b"",
            1: bytes([0xEC]),
            2: bytes([0x9E]),
            3: bytes([0xA0]),
        }
        raw = b"".join(table[token_id] for token_id in token_ids)
        if not raw:
            return ""
        if raw == bytes([0xEC, 0x9E, 0xA0]):
            return "\uc7a0"
        return "\ufffd" * sum(1 for token_id in token_ids if token_id != 0)


class _BpeTokenizer:
    clean_up_tokenization_spaces = False
    vocab = {"A": 0, "B": 1}

    def decode(self, token_ids, skip_special_tokens: bool = True):
        reverse = {token_id: token for token, token_id in self.vocab.items()}
        return "".join(reverse[token_id] for token_id in token_ids)


class BPEStreamingDetokenizer:
    __module__ = "mlx_vlm.tokenizer_utils"

    def reset(self):
        pass


class _MlxVlmBpeTokenizer:
    clean_up_tokenization_spaces = False

    def __init__(self, vocab):
        self.vocab = vocab
        self.detokenizer = BPEStreamingDetokenizer()

    def decode(self, token_ids, skip_special_tokens: bool = True):
        reverse = {token_id: token for token, token_id in self.vocab.items()}
        return "".join(reverse[token_id] for token_id in token_ids)


class _ExplicitNoDetokenizer:
    detokenizer = None

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return ""


def _bpe_byte_chars(*byte_values):
    from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer

    BPEStreamingDetokenizer.make_byte_decoder()
    byte_encoder = {
        byte_value: char
        for char, byte_value in BPEStreamingDetokenizer._byte_decoder.items()
    }
    return [byte_encoder[byte_value] for byte_value in byte_values]


def _make_misconverted_unlimited_ocr_tokenizer(
    tmp_path,
    *,
    model_type="unlimited-ocr",
):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    ni_bytes = _bpe_byte_chars(0xE4, 0xBD, 0xA0)
    vocab = {
        "Ġ": 0,
        "Ċ": 1,
        "A": 2,
        "B": 3,
        ni_bytes[0]: 4,
        ni_bytes[1]: 5,
        ni_bytes[2]: 6,
    }
    backend = Tokenizer(
        models.BPE(
            vocab=vocab,
            merges=[],
            fuse_unk=True,
            byte_fallback=True,
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        prepend_scheme="always",
        split=False,
    )
    backend.decoder = decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(" ", 1, 0),
        ]
    )
    backend.save(str(tmp_path / "tokenizer.json"))
    _write_json(tmp_path / "config.json", {"model_type": model_type})
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tmp_path / "tokenizer.json"))
    return tokenizer, ni_bytes


class TestCreateStreamingDetokenizer:
    def test_uses_spm_decoder_from_tokenizer_json(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": _spm_decoder()})

        detokenizer = create_streaming_detokenizer(
            _ByteFallbackTokenizer(),
            model_path=tmp_path,
        )
        assert detokenizer is not None

        parts = []
        for token_id in [1, 2, 3]:
            detokenizer.add_token(token_id)
            parts.append(detokenizer.last_segment)

        assert "".join(parts) == "\uc7a0"

    def test_uses_bpe_decoder_from_tokenizer_json(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": {"type": "ByteLevel"}})

        detokenizer = create_streaming_detokenizer(
            _BpeTokenizer(),
            model_path=tmp_path,
        )

        assert type(detokenizer).__name__ == "BPEStreamingDetokenizer"

    def test_replaces_mlx_vlm_bpe_detokenizer_from_tokenizer_json(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": {"type": "ByteLevel"}})
        chars = _bpe_byte_chars(0xEC, 0x9E, 0xA0, 0x20)
        tokenizer = _MlxVlmBpeTokenizer(
            {
                chars[0]: 0,
                chars[1]: 1,
                chars[2]: 2,
                chars[3] + "A": 3,
            }
        )

        detokenizer = create_streaming_detokenizer(tokenizer, model_path=tmp_path)

        assert type(detokenizer).__module__ == "mlx_lm.tokenizer_utils"
        parts = []
        for token_id in [0, 1, 2, 3]:
            detokenizer.add_token(token_id)
            parts.append(detokenizer.last_segment)
        detokenizer.finalize()
        parts.append(detokenizer.last_segment)

        assert "".join(parts) == "\uc7a0 A"

    def test_mlx_vlm_bpe_replacement_buffers_incomplete_utf8(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": {"type": "ByteLevel"}})
        lead_byte, space = _bpe_byte_chars(0xEC, 0x20)
        tokenizer = _MlxVlmBpeTokenizer({lead_byte: 0, space: 1})

        detokenizer = create_streaming_detokenizer(tokenizer, model_path=tmp_path)

        detokenizer.add_token(0)
        detokenizer.add_token(1)

        assert detokenizer.last_segment == ""

    def test_explicit_none_detokenizer_without_model_path_stays_none(self):
        assert create_streaming_detokenizer(_ExplicitNoDetokenizer()) is None

    def test_missing_tokenizer_json_uses_naive_fallback(self, tmp_path):
        detokenizer = create_streaming_detokenizer(
            _ByteFallbackTokenizer(),
            model_path=tmp_path,
        )

        assert type(detokenizer).__name__ in {
            "NaiveStreamingDetokenizer",
            "_CompatNaiveStreamingDetokenizer",
        }
        for token_id in [1, 2, 3]:
            detokenizer.add_token(token_id)
        detokenizer.finalize()

        assert detokenizer.text == "\uc7a0"


class TestRepairMisconvertedUnlimitedOCRTokenizer:
    def test_repairs_prompt_encoding_and_utf8_decode(self, tmp_path):
        tokenizer, _ = _make_misconverted_unlimited_ocr_tokenizer(tmp_path)

        assert tokenizer.encode(" A\nB", add_special_tokens=False) == [2, 3]
        assert tokenizer.decode([4, 5, 6]) != "你"

        repaired = repair_misconverted_unlimited_ocr_tokenizer(
            tokenizer,
            model_path=tmp_path,
        )

        assert repaired is True
        assert tokenizer.encode(" A\nB", add_special_tokens=False) == [0, 2, 1, 3]
        assert tokenizer.decode([0, 2, 1, 3]) == " A\nB"
        assert tokenizer.decode([4, 5, 6]) == "你"
        assert tokenizer.backend_tokenizer.model.fuse_unk is False
        assert tokenizer.backend_tokenizer.model.byte_fallback is False

    def test_uses_fresh_bpe_detokenizer_for_misconverted_export(self, tmp_path):
        tokenizer, _ = _make_misconverted_unlimited_ocr_tokenizer(tmp_path)

        first = create_streaming_detokenizer(tokenizer, model_path=tmp_path)
        second = create_streaming_detokenizer(tokenizer, model_path=tmp_path)

        assert type(first).__module__ == "mlx_lm.tokenizer_utils"
        assert type(first).__name__ == "BPEStreamingDetokenizer"
        assert first is not second

        parts = []
        for token_id in [0, 2, 4, 5, 6, 1]:
            first.add_token(token_id)
            parts.append(first.last_segment)
        first.finalize()
        parts.append(first.last_segment)

        assert "".join(parts) == "A你\n"

    def test_leaves_non_unlimited_model_untouched(self, tmp_path):
        tokenizer, _ = _make_misconverted_unlimited_ocr_tokenizer(
            tmp_path,
            model_type="llama",
        )

        repaired = repair_misconverted_unlimited_ocr_tokenizer(
            tokenizer,
            model_path=tmp_path,
        )

        assert repaired is False
        assert tokenizer.encode(" A\nB", add_special_tokens=False) == [2, 3]

    def test_leaves_canonical_unlimited_tokenizer_untouched(self, tmp_path):
        tokenizer, _ = _make_misconverted_unlimited_ocr_tokenizer(tmp_path)
        tokenizer_content = json.loads((tmp_path / "tokenizer.json").read_text())
        tokenizer_content["pre_tokenizer"] = {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        }
        tokenizer_content["decoder"] = {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        }
        _write_json(tmp_path / "tokenizer.json", tokenizer_content)

        repaired = repair_misconverted_unlimited_ocr_tokenizer(
            tokenizer,
            model_path=tmp_path,
        )

        assert repaired is False


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

    def test_gemma4_unified_model_via_config_model_type(self):
        config = {"model_type": "gemma4_unified"}
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


class TestLFM2ToolParserConfig:
    """Test cases for the scoped LFM2 Pythonic tool parser fix."""

    @staticmethod
    def _write_lfm2_text_model(tmp_path, chat_template=None):
        _write_json(
            tmp_path / "config.json",
            {
                "model_type": "lfm2",
                "architectures": ["LFM2ForCausalLM"],
            },
        )
        if chat_template is not None:
            _write_json(
                tmp_path / "tokenizer_config.json",
                {"chat_template": chat_template},
            )

    def test_lfm2_moe_text_model_gets_pythonic_tool_parser(self, tmp_path):
        _write_json(
            tmp_path / "config.json",
            {
                "model_type": "lfm2_moe",
                "architectures": ["LFM2MoeForCausalLM"],
            },
        )
        _write_json(
            tmp_path / "tokenizer_config.json",
            {"chat_template": "<|tool_call_start|>x<|tool_call_end|>"},
        )

        config = get_tokenizer_config(str(tmp_path))

        assert config["tool_parser_type"] == "pythonic"

    def test_lfm2_audio_architecture_excluded(self, tmp_path):
        _write_json(
            tmp_path / "config.json",
            {
                "model_type": "lfm2",
                "architectures": ["LFM2AudioModel"],
            },
        )
        _write_json(
            tmp_path / "tokenizer_config.json",
            {"chat_template": "<|tool_call_start|>x<|tool_call_end|>"},
        )

        config = get_tokenizer_config(str(tmp_path))

        assert "tool_parser_type" not in config

    def test_lfm_audio_model_type_excluded(self, tmp_path):
        _write_json(
            tmp_path / "config.json",
            {
                "model_type": "lfm2_audio",
                "architectures": ["LFM2ForCausalLM"],
            },
        )
        _write_json(
            tmp_path / "tokenizer_config.json",
            {"chat_template": "<|tool_call_start|>x<|tool_call_end|>"},
        )

        config = get_tokenizer_config(str(tmp_path))

        assert "tool_parser_type" not in config

    def test_lfm2_text_model_gets_pythonic_tool_parser(self, tmp_path):
        self._write_lfm2_text_model(
            tmp_path,
            "<|tool_call_start|>[call(arg='x')]<|tool_call_end|>",
        )

        config = get_tokenizer_config(str(tmp_path), trust_remote_code=True)

        assert config["trust_remote_code"] is True
        assert config["tool_parser_type"] == "pythonic"

    def test_lfm2_text_model_without_markers_gets_parser(self, tmp_path):
        self._write_lfm2_text_model(tmp_path, "plain template")

        config = get_tokenizer_config(str(tmp_path))

        assert config["tool_parser_type"] == "pythonic"

    def test_non_lfm2_model_with_markers_does_not_get_parser(self, tmp_path):
        _write_json(
            tmp_path / "config.json",
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
            },
        )
        _write_json(
            tmp_path / "tokenizer_config.json",
            {"chat_template": "<|tool_call_start|>x<|tool_call_end|>"},
        )

        config = get_tokenizer_config(str(tmp_path))

        assert "tool_parser_type" not in config


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

    def test_laguna_config_enables_mistral_regex_fix(self, tmp_path):
        """Laguna's Mistral-derived tokenizer needs the corrected regex."""
        _write_json(
            tmp_path / "config.json",
            {
                "model_type": "laguna",
                "architectures": ["LagunaForCausalLM"],
            },
        )

        config = get_tokenizer_config(str(tmp_path))

        assert config["fix_mistral_regex"] is True

    def test_laguna_config_pins_laguna_tool_parser(self, tmp_path):
        """Laguna templates contain <arg_key>, which mlx-lm's template
        sniffing misreads as glm47; the vendored parser must be pinned."""
        _write_json(
            tmp_path / "config.json",
            {
                "model_type": "laguna",
                "architectures": ["LagunaForCausalLM"],
            },
        )

        config = get_tokenizer_config(str(tmp_path))

        assert config["tool_parser_type"] == "laguna"

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


class TestMistralCommonTokenizerConfig:
    """get_tokenizer_config forces the HF-native backend for mistral-common repos.

    transformers routes repos that ship tekken.json (Devstral 2 / Mistral
    Small Tekken exports) to MistralCommonBackend, whose rendered chat
    template cannot be re-encoded faithfully — control tokens become literal
    text and every prompt built via render-then-encode is corrupted. Passing
    fix_mistral_regex selects TokenizersBackend instead (gate shipped in
    transformers 5.12.1, the pyproject floor).
    """

    def _make_mistral_repo(self, tmp_path):
        _write_json(tmp_path / "tekken.json", {"version": "v13"})
        _write_json(tmp_path / "tokenizer.json", {"version": "1.0"})
        return tmp_path

    def test_mistral_common_repo_gets_fix_mistral_regex(self, tmp_path):
        repo = self._make_mistral_repo(tmp_path)
        config = get_tokenizer_config(str(repo))
        assert config["fix_mistral_regex"] is True

    def test_symlinked_repo_detected(self, tmp_path):
        """Served model dirs are frequently symlinks; is_file() must follow."""
        real = tmp_path / "real"
        real.mkdir()
        self._make_mistral_repo(real)
        link = tmp_path / "link"
        link.symlink_to(real)
        config = get_tokenizer_config(str(link))
        assert config["fix_mistral_regex"] is True

    def test_plain_repo_without_tekken_json_untouched(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"version": "1.0"})
        config = get_tokenizer_config(str(tmp_path))
        assert "fix_mistral_regex" not in config

    def test_audio_only_export_without_tokenizer_json_untouched(self, tmp_path):
        """Voxtral-shaped exports ship tekken.json with no HF-native
        tokenizer.json; there is no TokenizersBackend to select — do not
        inject the kwarg."""
        _write_json(tmp_path / "tekken.json", {"version": "v13"})
        config = get_tokenizer_config(str(tmp_path))
        assert "fix_mistral_regex" not in config

    def test_hf_repo_id_untouched(self):
        """Remote repo ids are not local paths; detection stays conservative."""
        config = get_tokenizer_config("mlx-community/Devstral-Small-2-24B-4bit")
        assert "fix_mistral_regex" not in config

    def test_other_family_fixes_unaffected(self, tmp_path):
        repo = self._make_mistral_repo(tmp_path)
        config = get_tokenizer_config(str(repo))
        assert "eos_token" not in config
