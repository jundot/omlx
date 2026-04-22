# SPDX-License-Identifier: Apache-2.0
"""
Tokenizer utilities for oMLX.

This module provides shared tokenizer configuration and fixes that are used
across multiple modules in the codebase.
"""

import logging
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
    if hasattr(tokenizer, '_tokenizer'):
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
    for attr in ('config', 'args'):
        config = getattr(model, attr, None)
        if config is None:
            continue
        vs = getattr(config, 'vocab_size', None)
        if isinstance(vs, int):
            return vs
        text_cfg = getattr(config, 'text_config', None)
        if isinstance(text_cfg, dict):
            vs = text_cfg.get('vocab_size')
        elif text_cfg is not None:
            vs = getattr(text_cfg, 'vocab_size', None)
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
    1. model_type == "gemma4" in config.json
    2. Fallback: model_name contains "gemma-4" or "gemma4" (case-insensitive)
    """
    if config is not None:
        model_type = config.get("model_type", "")
        if model_type == "gemma4":
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


def is_mistral_model(model_name: str) -> bool:
    """
    Check if the model is a Mistral model that needs fix_mistral_regex.

    Mistral-Small and some other Mistral models have a known tokenizer
    regex issue that requires the fix_mistral_regex=True flag.

    Also includes Qwen3.5-Claude models which use Mistral-based tokenizers.

    Args:
        model_name: The model name or path.

    Returns:
        True if the model needs the mistral regex fix.
    """
    model_lower = model_name.lower()
    # Check for Mistral-Small variants and other affected Mistral models
    # Also includes Qwen3.5-Claude models which use Mistral tokenizers
    return (
        "mistral-small" in model_lower or
        "mistral" in model_lower or
        "qwen3.5" in model_lower and "claude" in model_lower
    )


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

    # Gemma 4 models: Set extra_special_tokens to empty dict to avoid AttributeError
    # Gemma 4 configs have extra_special_tokens as a list, but transformers expects a dict.
    # Setting it to an empty dict overrides the config value and prevents:
    # "AttributeError: 'list' object has no attribute 'keys'"
    if is_gemma4_model(model_name):
        config["extra_special_tokens"] = {}
        # Override tokenizer_class to GemmaTokenizer (Gemma 2/3 tokenizer) for compatibility
        # with transformers <5.4.0. Gemma 4 support was added in transformers 5.5.0+.
        # GemmaTokenizer is backward compatible and works with Gemma 4 models.
        config["tokenizer_class"] = "GemmaTokenizer"
        logger.debug("Gemma 4 detected: overriding extra_special_tokens to empty dict, tokenizer_class to GemmaTokenizer")

    # Fix for Qwen3.5/3.6-A3B models with broken tokenizer_class
    # These MLX-quantized models have "tokenizer_class": "TokenizersBackend" which
    # doesn't exist in transformers. We need to override it to "Qwen2Tokenizer".
    if ("Qwen3.5" in model_name or "Qwen3.6" in model_name) and "A3B" in model_name:
        config["tokenizer_class"] = "Qwen2Tokenizer"
        logger.debug(f"{model_name}: overriding tokenizer_class to Qwen2Tokenizer")

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
