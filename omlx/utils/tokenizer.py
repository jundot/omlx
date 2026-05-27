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


def collect_stop_token_ids(tokenizer: Any) -> set[int]:
    """Collect every EOS / end-of-turn token id the model treats as terminal.

    Mirrors the scheduler's ``_get_stop_tokens`` logic minus protocol-specific
    additions (Harmony stops are scheduler-layer concerns). Used by
    ``api.grammar.create_grammar_compiler`` to wire xgrammar's
    ``TokenizerInfo.stop_token_ids`` so the GrammarMatcher's bitmask treats
    every real EOS as terminal.

    Why this matters: Gemma 4 declares ``<end_of_turn>`` (token 106) plus a
    second EOS variant (50) only in ``generation_config.json`` ---
    ``tokenizer.eos_token_id`` is the canonical ``<eos>`` alone. Without
    surfacing the generation_config set to xgrammar, the matcher masks
    those tokens out of the bitmask once a structured-output state expects
    string content; the model loops on repeating characters inside the
    ``<string>`` state until it hits ``max_tokens`` (observed 2026-05-28:
    21k tokens / 6 min on a single chat completion).

    Resolution order (union):
    1. ``tokenizer.eos_token_id`` (int or list).
    2. ``tokenizer.eos_token_ids`` (some MLX tokenizers expose the plural).
    3. ``{tokenizer.name_or_path}/generation_config.json`` ``eos_token_id``
       (int or list), if the path resolves locally.

    Returns an empty set if no stop token can be derived. xgrammar treats
    that as "use tokenizer's eos only", which is the pre-fix behavior.
    """
    stops: set[int] = set()

    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        if isinstance(eos, int):
            stops.add(eos)
        else:
            try:
                stops.update(int(t) for t in eos)
            except TypeError:
                pass

    eos_plural = getattr(tokenizer, "eos_token_ids", None)
    if eos_plural is not None:
        if isinstance(eos_plural, int):
            stops.add(eos_plural)
        else:
            try:
                stops.update(int(t) for t in eos_plural)
            except TypeError:
                pass

    model_path = getattr(tokenizer, "name_or_path", None)
    if model_path:
        try:
            import json
            import os

            gc_path = os.path.join(model_path, "generation_config.json")
            if os.path.exists(gc_path):
                with open(gc_path) as f:
                    gc = json.load(f)
                gc_eos = gc.get("eos_token_id")
                if gc_eos is not None:
                    if isinstance(gc_eos, int):
                        stops.add(gc_eos)
                    else:
                        try:
                            stops.update(int(t) for t in gc_eos)
                        except TypeError:
                            pass
        except Exception as e:
            logger.debug(
                "collect_stop_token_ids: failed reading generation_config: %s", e
            )

    return stops


def resolve_vocab_size(model: Any) -> int | None:
    """Extract vocab_size from a model, preferring the authoritative source.

    Resolution order:
    1. The ``lm_head`` weight's first dimension (authoritative — this is the
       exact vocabulary the model emits logits over).
    2. ``text_config.vocab_size`` when present (the inner language model's
       vocab on VLM composite configs).
    3. ``model.config.vocab_size`` / ``model.args.vocab_size`` (top-level).

    Why lm_head and text_config come first for VLMs: several mlx-vlm
    ``ModelConfig`` dataclasses (e.g. glm4v, glm4v_moe, gemma3) hard-code a
    top-level ``vocab_size`` default that does not match the inner LM vocab
    when ``config.json`` omits the top-level key.  For example, GLM-4.6V has
    ``text_config.vocab_size=151552`` but ``ModelConfig.vocab_size=257152``
    as a dataclass default.  Code that sizes logits-aligned buffers (e.g.
    grammar bitmasks) from the top-level value produces a shape mismatch
    against the real (151552) logits.

    Args:
        model: An MLX model object (LLM, VLM, or any object with config/args).

    Returns:
        The vocabulary size, or None if it cannot be determined.
    """
    if model is None:
        return None

    # 1. lm_head weight — authoritative for any model that exposes one.
    #    VLM adapters wrap the language model under ``_language_model``;
    #    raw mlx-lm/mlx-vlm models expose ``lm_head`` directly or under
    #    ``language_model``.
    for path in (
        ("_language_model", "lm_head"),
        ("language_model", "lm_head"),
        ("lm_head",),
    ):
        obj: Any = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        weight = getattr(obj, "weight", None) if obj is not None else None
        shape = getattr(weight, "shape", None)
        try:
            first_dim = shape[0] if shape is not None else None
        except (TypeError, IndexError):
            first_dim = None
        if isinstance(first_dim, int):
            return int(first_dim)

    # 2 & 3. Config-based fallbacks.
    for attr in ('config', 'args'):
        config = getattr(model, attr, None)
        if config is None:
            continue
        text_cfg = getattr(config, 'text_config', None)
        if isinstance(text_cfg, dict):
            vs = text_cfg.get('vocab_size')
        elif text_cfg is not None:
            vs = getattr(text_cfg, 'vocab_size', None)
        else:
            vs = None
        if isinstance(vs, int):
            return vs
        vs = getattr(config, 'vocab_size', None)
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
