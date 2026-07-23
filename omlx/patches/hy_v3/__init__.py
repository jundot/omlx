# SPDX-License-Identifier: Apache-2.0
"""Tencent Hy3 (HunYuan V3, ``model_type: hy_v3``) support for mlx-lm.

Brings ml-explore/mlx-lm#1211 into oMLX without modifying the pinned mlx-lm
package. The upstream PR adds:

* ``mlx_lm.models.hy_v3`` — the 80-layer MoE decoder (192 routed experts +
  1 shared, sigmoid router with expert bias, qk-norm, MTP weights skipped
  in ``sanitize()``).
* ``mlx_lm.tool_parsers.hy_v3`` / ``hy_v3_opensource`` — the Hy3 tool-call
  format. The released checkpoints rename every special token with an
  ``":opensource"`` suffix at the same token ids (``<tool_calls>`` →
  ``<tool_calls:opensource>``), so the release needs the ``_opensource``
  sentinels while Hy3-preview uses the plain ones.
* Tool-parser inference and think-token detection in ``tokenizer_utils``:
  Hy3 templates contain ``<arg_key`` (like GLM 4.7) but also ``<tool_sep``,
  so the hy_v3 check must run before the ``<arg_key>`` → ``glm47`` branch,
  and ``<think:opensource>``/``</think:opensource>`` must be recognised as
  a thinking pair.

All hooks are upstream-first: once mlx-lm merges #1211 they become no-ops.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PR_HEAD_SHA = "b7635e9cff3a953a89505aee1e029073810f777f"
PR_URL = "https://github.com/ml-explore/mlx-lm/pull/1211"

_APPLIED = False

_VENDORED_MODULES = (
    # (qualname, vendored filename) — hy_v3 tool parser must be registered
    # before hy_v3_opensource, which does ``from .hy_v3 import parse_tool_call``.
    ("mlx_lm.models.hy_v3", "hy_v3_model.py", "mlx_lm.models"),
    ("mlx_lm.tool_parsers.hy_v3", "hy_v3_tool_parser.py", "mlx_lm.tool_parsers"),
    (
        "mlx_lm.tool_parsers.hy_v3_opensource",
        "hy_v3_opensource_tool_parser.py",
        "mlx_lm.tool_parsers",
    ),
)


def _register_module(qualname: str, filename: str, package: str) -> None:
    if qualname in sys.modules:
        return

    file_path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[qualname] = module
    spec.loader.exec_module(module)

    parent = importlib.import_module(package)
    setattr(parent, qualname.rsplit(".", 1)[1], module)
    logger.info("Registered %s from %s", qualname, filename)


def _patch_infer_tool_parser() -> None:
    """Run the hy_v3 template check ahead of mlx-lm's inference.

    Hy3 chat templates contain ``<arg_key`` (which mlx-lm maps to ``glm47``)
    *and* ``<tool_sep``; upstream #1211 disambiguates by checking the pair
    first. Mirror that here by wrapping ``_infer_tool_parser``.
    """
    import mlx_lm.tokenizer_utils as tu

    original = tu._infer_tool_parser
    if getattr(original, "_omlx_hy_v3", False):
        return

    def _infer_tool_parser_with_hy_v3(chat_template):
        if isinstance(chat_template, str) and (
            "<tool_sep" in chat_template and "<arg_key" in chat_template
        ):
            return (
                "hy_v3_opensource" if ":opensource" in chat_template else "hy_v3"
            )
        return original(chat_template)

    _infer_tool_parser_with_hy_v3._omlx_hy_v3 = True
    tu._infer_tool_parser = _infer_tool_parser_with_hy_v3


def _patch_infer_thinking() -> None:
    """Recognise the release tokenizer's ``<think:opensource>`` pair."""
    import mlx_lm.tokenizer_utils as tu

    infer = getattr(tu, "_infer_thinking", None)
    if infer is None or getattr(infer, "_omlx_hy_v3", False):
        return

    def _infer_thinking_with_hy_v3(tokenizer):
        result = infer(tokenizer)
        if result and result[0] is not None:
            return result
        vocab = tokenizer.get_vocab()
        start, end = "<think:opensource>", "</think:opensource>"
        if start in vocab and end in vocab:
            return (start, end, (vocab[start],), (vocab[end],))
        return result

    _infer_thinking_with_hy_v3._omlx_hy_v3 = True
    tu._infer_thinking = _infer_thinking_with_hy_v3


def apply_hy_v3_patch() -> bool:
    """Register Hy3 support when the pinned mlx-lm does not provide it."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        logger.debug("mlx_lm not importable - hy_v3 patch skipped")
        return False

    try:
        importlib.import_module("mlx_lm.models.hy_v3")
        upstream = True
    except ImportError:
        upstream = False

    if not upstream:
        for qualname, filename, package in _VENDORED_MODULES:
            _register_module(qualname, filename, package)
        _patch_infer_tool_parser()
        _patch_infer_thinking()
        _APPLIED = True
        logger.info("Hy3 mlx-lm patch applied (PR 1211 head %s)", PR_HEAD_SHA[:8])
        return True

    _APPLIED = True
    logger.debug("mlx_lm.models.hy_v3 already available upstream")
    return False


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_hy_v3_patch", "is_applied", "PR_HEAD_SHA", "PR_URL"]
