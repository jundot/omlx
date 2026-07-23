# SPDX-License-Identifier: Apache-2.0
"""
Tokenizer utilities for oMLX.

This module provides shared tokenizer configuration and fixes that are used
across multiple modules in the codebase.
"""

import json
import logging
from collections.abc import Callable
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def unwrap_tokenizer(tokenizer):
    """Unwrap mlx-lm TokenizerWrapper to a HuggingFace PreTrainedTokenizer.

    xgrammar accepts HuggingFace ``PreTrainedTokenizer`` /
    ``PreTrainedTokenizerFast`` but NOT the raw ``tokenizers.Tokenizer``
    nor the mlx-lm ``TokenizerWrapper``.  This helper peels exactly one
    layer of mlx-lm wrapping while keeping the HuggingFace object intact.
    """
    try:
        from transformers import PreTrainedTokenizerBase

        if isinstance(tokenizer, PreTrainedTokenizerBase):
            return tokenizer
    except ImportError:
        pass
    if hasattr(tokenizer, "_tokenizer"):
        inner = tokenizer._tokenizer
        try:
            from transformers import PreTrainedTokenizerBase

            if isinstance(inner, PreTrainedTokenizerBase):
                return inner
        except ImportError:
            pass
        return inner
    return tokenizer


def resolve_vocab_size(model: Any) -> int | None:
    """Extract vocab_size from a model's config/args, handling nested configs.

    Tries ``model.config.vocab_size``, then ``model.args.vocab_size``,
    then ``text_config.vocab_size`` for VLM composite models (e.g. Qwen3.5).

    Args:
        model: An MLX model object (LLM, VLM, or any object with config/args).

    Returns:
        The vocabulary size, or None if it cannot be determined.
    """
    if model is None:
        return None
    for attr in ("config", "args"):
        config = getattr(model, attr, None)
        if config is None:
            continue
        vs = getattr(config, "vocab_size", None)
        if isinstance(vs, int):
            return vs
        text_cfg = getattr(config, "text_config", None)
        if isinstance(text_cfg, dict):
            vs = text_cfg.get("vocab_size")
        elif text_cfg is not None:
            vs = getattr(text_cfg, "vocab_size", None)
        if isinstance(vs, int):
            return vs
    return None


def is_harmony_model(model_name: str, config: dict[str, Any] | None = None) -> bool:
    """
    Check if the model uses Harmony format.

    Harmony format is used by gpt-oss models with special tokens like
    <|start|>, <|channel|>, <|message|>, <|end|>, <|return|>, <|call|>.

    Detection priority:
    1. model_type == "gpt_oss" in config.json
    2. Fallback: model_name contains "gpt-oss" or "gptoss" (case-insensitive)

    Args:
        model_name: The model name or path.
        config: Optional model config dict (from config.json).

    Returns:
        True if the model uses Harmony format.
    """
    # Primary detection: config.model_type
    if config is not None:
        model_type = config.get("model_type", "")
        if model_type == "gpt_oss":
            logger.debug(f"Harmony model detected via config.model_type: {model_name}")
            return True

    # Fallback detection: model name pattern
    if model_name:
        name_lower = model_name.lower()
        if "gpt-oss" in name_lower or "gptoss" in name_lower:
            logger.debug(f"Harmony model detected via model name pattern: {model_name}")
            return True

    return False


def is_gemma4_model(model_name: str, config: dict[str, Any] | None = None) -> bool:
    """
    Check if the model is a Gemma 4 model.

    Detection priority:
    1. Gemma 4 model_type in config.json
    2. Fallback: model_name contains "gemma-4" or "gemma4" (case-insensitive)
    """
    if config is not None:
        model_type = config.get("model_type", "")
        # diffusion_gemma shares Gemma 4's wire protocol (channel markers,
        # call:name{...} tool calls), so it uses the same parser/extractor.
        if model_type in {"gemma4", "gemma4_unified", "diffusion_gemma"}:
            logger.debug(f"Gemma 4 model detected via config.model_type: {model_name}")
            return True

    if model_name:
        name_lower = model_name.lower()
        if "gemma-4" in name_lower or "gemma4" in name_lower:
            logger.debug(f"Gemma 4 model detected via model name pattern: {model_name}")
            return True

    return False


def is_qwen3_model(model_name: str) -> bool:
    """
    Check if the model is a Qwen3 model.

    Args:
        model_name: The model name or path.

    Returns:
        True if the model is a Qwen3 model.
    """
    model_lower = model_name.lower()
    return "qwen3" in model_lower or "Qwen3" in model_name


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _find_tokenizer_json(
    tokenizer: Any,
    model_path: str | Path | None = None,
) -> Path | None:
    candidates: list[str | Path] = []
    if model_path:
        candidates.append(model_path)

    tokenizer_path = getattr(tokenizer, "name_or_path", None)
    if tokenizer_path:
        candidates.append(tokenizer_path)

    for candidate in candidates:
        candidate_path = Path(candidate).expanduser()
        tokenizer_file = candidate_path / "tokenizer.json"
        if tokenizer_file.exists():
            return tokenizer_file

        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(str(candidate), "tokenizer.json")
        except Exception:
            cached = None

        if cached and isinstance(cached, str):
            cached_path = Path(cached)
            if cached_path.exists():
                return cached_path

    return None


@lru_cache(maxsize=128)
def _detokenizer_factory_from_tokenizer_json(
    tokenizer_file: str,
) -> Callable[[Any], Any] | None:
    tokenizer_content = _read_json_file(Path(tokenizer_file))
    if not tokenizer_content or "decoder" not in tokenizer_content:
        return None

    try:
        from mlx_lm.tokenizer_utils import (
            BPEStreamingDetokenizer,
            SPMStreamingDetokenizer,
            _is_bpe_decoder,
            _is_spm_decoder,
            _is_spm_decoder_no_space,
        )
    except ImportError:
        return None

    decoder = tokenizer_content["decoder"]
    if _is_spm_decoder(decoder):
        return SPMStreamingDetokenizer
    if _is_spm_decoder_no_space(decoder):
        return partial(SPMStreamingDetokenizer, trim_space=False)
    if _is_bpe_decoder(decoder):
        return BPEStreamingDetokenizer
    return None


def _is_unsafe_mlx_vlm_bpe_detokenizer(detokenizer: Any) -> bool:
    detokenizer_type = type(detokenizer)
    return (
        detokenizer_type.__module__ == "mlx_vlm.tokenizer_utils"
        and detokenizer_type.__name__ == "BPEStreamingDetokenizer"
    )


def _create_decoder_aware_detokenizer(
    tokenizer: Any,
    tokenizer_file: Path | None,
) -> Any | None:
    if tokenizer_file is None:
        return None

    factory = _detokenizer_factory_from_tokenizer_json(str(tokenizer_file))
    if factory is None:
        return None

    try:
        return factory(tokenizer)
    except Exception as exc:
        logger.debug(
            "Failed to create decoder-aware detokenizer from %s: %s",
            tokenizer_file,
            exc,
        )
        return None


class _CompatNaiveStreamingDetokenizer:
    """Naive fallback for raw tokenizers that lack mlx-lm's probe APIs."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._tokenizer.decode([0])
        self.reset()

    def reset(self) -> None:
        self.offset = 0
        self.tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token: int) -> None:
        self._current_tokens.append(token)
        self.tokens.append(token)

    def finalize(self) -> None:
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self) -> str:
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
            if self._current_text.endswith("\ufffd") or (
                bool(getattr(self._tokenizer, "clean_up_tokenization_spaces", False))
                and len(self._current_text) > 0
                and self._current_text[-1] == " "
            ):
                self._current_text = self._current_text[:-1]
        if self._current_text and self._current_text[-1] == "\n":
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text

    @property
    def last_segment(self) -> str:
        text = self.text
        segment = text[self.offset :]
        self.offset = len(text)
        return segment


def create_streaming_detokenizer(
    tokenizer: Any,
    model_path: str | Path | None = None,
) -> Any | None:
    """Create a fresh streaming detokenizer for one request.

    mlx-lm's TokenizerWrapper exposes the correct per-model detokenizer, but
    raw VLM/DFlash tokenizers may not.  In that case, mirror mlx-lm's
    tokenizer.json decoder detection before falling back to the naive decoder.
    """
    has_existing_attr = True
    try:
        detokenizer = tokenizer.detokenizer
    except AttributeError:
        has_existing_attr = False
        detokenizer = None
    except Exception as exc:
        has_existing_attr = False
        detokenizer = None
        logger.debug("Failed to read tokenizer.detokenizer: %s", exc)

    if detokenizer is not None:
        if _is_unsafe_mlx_vlm_bpe_detokenizer(detokenizer):
            tokenizer_file = _find_tokenizer_json(tokenizer, model_path)
            decoder_aware_detokenizer = _create_decoder_aware_detokenizer(
                tokenizer,
                tokenizer_file,
            )
            if decoder_aware_detokenizer is not None:
                return decoder_aware_detokenizer
            logger.debug(
                "Using existing mlx-vlm BPE detokenizer because no "
                "decoder-aware replacement is available"
            )
        return detokenizer

    tokenizer_file = _find_tokenizer_json(tokenizer, model_path)
    decoder_aware_detokenizer = _create_decoder_aware_detokenizer(
        tokenizer,
        tokenizer_file,
    )
    if decoder_aware_detokenizer is not None:
        return decoder_aware_detokenizer

    if has_existing_attr:
        return None

    try:
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
    except ImportError:
        return None

    try:
        return NaiveStreamingDetokenizer(tokenizer)
    except Exception as exc:
        logger.debug("Failed to create naive streaming detokenizer: %s", exc)

    try:
        return _CompatNaiveStreamingDetokenizer(tokenizer)
    except Exception as compat_exc:
        logger.debug(
            "Failed to create compatibility naive streaming detokenizer: %s",
            compat_exc,
        )
        return None


def _is_lfm2_text_lm(model_name: str) -> bool:
    """Return True for local LFM2 text causal LM checkpoints."""
    config_path = Path(model_name) / "config.json"
    config = _read_json_file(config_path)
    if config is None:
        return False

    model_type = str(config.get("model_type") or "").lower().replace("-", "_")
    architectures = [
        str(arch) for arch in config.get("architectures", []) if isinstance(arch, str)
    ]
    architectures_lower = [arch.lower() for arch in architectures]

    if model_type in {"lfm_audio", "lfm2_audio"}:
        return False
    if any(key in config for key in ("audio_config", "tts_config", "stt_config")):
        return False
    if any("audio" in arch for arch in architectures_lower):
        return False
    if not any("forcausallm" in arch for arch in architectures_lower):
        return False

    return model_type.startswith("lfm2") or any(
        arch.lower().startswith("lfm2") for arch in architectures
    )


def _is_laguna_model(model_name: str) -> bool:
    """Return True only for a local checkpoint declaring ``model_type: laguna``."""
    config = _read_json_file(Path(model_name) / "config.json")
    return config is not None and config.get("model_type") == "laguna"


def get_tokenizer_config(
    model_name: str,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """
    Get tokenizer configuration with model-specific fixes.

    This function centralizes tokenizer configuration to ensure consistent
    behavior across different modules.

    Args:
        model_name: The model name or path.
        trust_remote_code: Whether to trust remote code.

    Returns:
        Dictionary of tokenizer configuration options.
    """
    config: dict[str, Any] = {"trust_remote_code": trust_remote_code}

    # Apply Qwen3 fix if needed
    if is_qwen3_model(model_name):
        config["eos_token"] = "<|im_end|>"
        logger.debug("Qwen3 detected: setting eos_token to <|im_end|>")

    if _is_lfm2_text_lm(model_name):
        config.setdefault("tool_parser_type", "pythonic")
        logger.debug("LFM2 text LM detected: setting tool_parser_type to pythonic")

    if _is_laguna_model(model_name):
        # Laguna's Mistral-derived tokenizer ships the legacy regex that
        # Transformers identifies as tokenization-incorrect without this flag.
        config["fix_mistral_regex"] = True
        # mlx-lm's template sniffing sees Laguna's <arg_key> markers and picks
        # the glm47 parser; pin the vendored Laguna parser instead.
        config.setdefault("tool_parser_type", "laguna")
        logger.debug(
            "Laguna detected: enabling the Mistral tokenizer regex fix and "
            "the laguna tool parser"
        )

    return config


def apply_qwen3_fix(
    tokenizer_config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """
    Apply Qwen3 tokenizer fix to an existing config.

    Qwen3 has a known issue where eos_token changed from <|im_end|> to
    <|endoftext|>, but the chat template still uses <|im_end|>. This
    function applies the fix if needed.

    Args:
        tokenizer_config: Existing tokenizer configuration dict.
        model_name: The model name or path.

    Returns:
        Updated tokenizer configuration with Qwen3 fix applied if needed.
    """
    if is_qwen3_model(model_name):
        tokenizer_config["eos_token"] = "<|im_end|>"
        logger.debug("Qwen3 detected: setting eos_token to <|im_end|>")

    return tokenizer_config
