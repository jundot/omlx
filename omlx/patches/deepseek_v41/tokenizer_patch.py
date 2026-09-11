# SPDX-License-Identifier: Apache-2.0
"""Tokenizer + chat-template hooks for DeepSeek-V4.1.

HF ``tokenizer_config.json`` ships ``add_bos_token: false``, but the official
encoder always prefixes ``<｜begin▁of▁sentence｜>``. Without BOS, greedy
completion collapses after the first token.

Mirrors the V4 tokenizer load patch: wrap ``mlx_lm.tokenizer_utils.load``
*and* rebind ``mlx_lm.utils._load_tokenizer`` (already imported at module
load time).

BOS handling: ensure encode prepends bos_token_id when missing, without
doubling when the chat template already embedded the BOS string.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from omlx.patches.deepseek_v41.predicates import is_deepseek_v41

logger = logging.getLogger(__name__)
_PATCHED = False

_BOS_STRING = "<｜begin▁of▁sentence｜>"


def _register_chat_template_module() -> None:
    if "mlx_lm.chat_templates.deepseek_v41" not in sys.modules:
        from . import chat_template_v41 as _ct

        sys.modules["mlx_lm.chat_templates.deepseek_v41"] = _ct
        sys.modules["mlx_lm.chat_templates.deepseek_v41_text"] = _ct


def _is_deepseek_v41_model(model_path) -> bool:
    try:
        cfg_path = Path(model_path) / "config.json"
        if not cfg_path.is_file():
            return False
        cfg = json.loads(cfg_path.read_text())
        mt = cfg.get("model_type")
        if mt is None and isinstance(cfg.get("text_config"), dict):
            mt = cfg["text_config"].get("model_type")
        return is_deepseek_v41(mt)
    except Exception:
        return False


def _ensure_bos_on_encode(wrapper) -> None:
    """Wrap inner encode so raw prompts get a leading BOS exactly once."""
    inner = getattr(wrapper, "_tokenizer", None)
    if inner is None or getattr(inner, "_omlx_dsv41_bos_wrapped", False):
        return
    bos_id = getattr(inner, "bos_token_id", None)
    if bos_id is None:
        return
    orig_encode = inner.encode

    def encode(text, *args, **kwargs):
        add_special = kwargs.get("add_special_tokens", True)
        ids = orig_encode(text, *args, **kwargs)
        if not add_special:
            return ids
        # list-ify for mutation across HF return types
        out = list(ids)
        if out and out[0] == bos_id:
            return ids
        # Chat-template strings already contain the BOS marker; encoding
        # that marker yields bos_id as the first token. Raw prompts don't.
        return [bos_id] + out

    inner.encode = encode  # type: ignore[method-assign]
    inner._omlx_dsv41_bos_wrapped = True


def apply_tokenizer_patch() -> bool:
    """Inject V4.1 chat template + BOS-on-encode into TokenizerWrapper load."""
    global _PATCHED
    if _PATCHED:
        return False

    try:
        import mlx_lm.tokenizer_utils as _tu
    except Exception as e:  # pragma: no cover
        logger.warning("V4.1 tokenizer patch: mlx_lm.tokenizer_utils missing: %s", e)
        return False

    _register_chat_template_module()
    orig_load = _tu.load

    def patched_load(model_path, tokenizer_config_extra=None, eos_token_ids=None):
        wrapper = orig_load(
            model_path,
            tokenizer_config_extra=tokenizer_config_extra,
            eos_token_ids=eos_token_ids,
        )
        if not _is_deepseek_v41_model(model_path):
            return wrapper

        from . import chat_template_v41 as _ct

        _ensure_bos_on_encode(wrapper)
        # Avoid mlx auto-enabling thinking via has_thinking.
        try:
            wrapper.has_thinking = False
        except Exception:
            pass

        if getattr(wrapper, "_chat_template", None) is None:
            wrapper._chat_template = _ct.apply_chat_template
            wrapper.has_chat_template = True
            logger.info(
                "Injected deepseek_v41 chat_template into TokenizerWrapper for %s",
                model_path,
            )
        return wrapper

    _tu.load = patched_load
    try:
        import mlx_lm.utils as _mu

        if hasattr(_mu, "_load_tokenizer"):
            _mu._load_tokenizer = patched_load
    except Exception as e:
        logger.warning(
            "Could not patch mlx_lm.utils._load_tokenizer (V4.1 chat_template "
            "injection may not fire): %s",
            e,
        )

    _PATCHED = True
    logger.info(
        "mlx_lm.tokenizer_utils.load wrapped "
        "(injects chat_template + BOS encode for deepseek_v41)"
    )
    return True


__all__ = ["apply_tokenizer_patch"]
