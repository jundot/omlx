# SPDX-License-Identifier: Apache-2.0
"""Patch ``mlx_lm.tokenizer_utils`` to fall back when transformers does not
yet recognize the ``deepseek_v4`` model_type.

PR 1192 itself does not modify ``tokenizer_utils.py`` — instead its README
asks the user to install transformers PR 45643 from source. PR 1189 took
the alternative path of adding a try/except fallback inside
``tokenizer_utils.load`` that catches the AttributeError /ValueError raised
when ``transformers.AutoTokenizer.from_pretrained`` cannot infer
``max_position_embeddings`` from a generic ``PreTrainedConfig``.

We adopt PR 1189's fallback approach because:

1. transformers 5.7.0 (released 2026-04-28) does NOT include the
   deepseek_v4 model_type — it ships *before* PR 45643 was merged
   (2026-05-02). Until the next transformers release lands on PyPI,
   ``AutoTokenizer.from_pretrained`` on a deepseek_v4 model will fail
   with ``AttributeError: 'PreTrainedConfig' object has no attribute
   'max_position_embeddings'``.
2. Asking users to ``pip install`` transformers from a specific PR is
   an operational footgun.
3. The fallback is forward-compatible: when transformers eventually
   ships native support, the ``try`` succeeds and the fallback never
   runs.

Strategy: replace ``mlx_lm.tokenizer_utils.AutoTokenizer`` with a thin
wrapper whose ``from_pretrained`` attempts the upstream call first and,
on the specific exception signature, retries with an empty
``PreTrainedConfig()`` injected. mlx-lm's ``load`` function does
``AutoTokenizer.from_pretrained(model_path, ...)`` via the module-level
attribute, so attribute-replacement is enough — we don't need to touch
the ``load`` function body.
"""

from __future__ import annotations

import logging
import warnings

import mlx_lm.tokenizer_utils as _tu

logger = logging.getLogger(__name__)
_PATCHED = False


def _build_wrapper():
    """Build the AutoTokenizer wrapper that adds the deepseek_v4 fallback.

    We capture the original ``transformers.AutoTokenizer`` so we can call
    its ``from_pretrained`` in the happy path. The wrapper exposes only
    ``from_pretrained`` because that is the sole entry point mlx-lm
    uses; any other attribute access is forwarded transparently.
    """
    from transformers import AutoTokenizer as _UpstreamAutoTokenizer

    # PreTrainedConfig is the base class transformers uses when it
    # cannot find a specific config class for a model_type.
    from transformers import PreTrainedConfig

    class _DeepSeekV4AwareAutoTokenizer:
        """Thin wrapper around transformers.AutoTokenizer.

        Adds a fallback for the deepseek_v4 / max_position_embeddings
        error that occurs when transformers has not yet shipped the
        deepseek_v4 model_type. Forward-compatible: when transformers
        adds native support, the ``try`` succeeds and the except branch
        is never hit.
        """

        @staticmethod
        def from_pretrained(model_path, *args, **kwargs):
            try:
                return _UpstreamAutoTokenizer.from_pretrained(
                    model_path, *args, **kwargs
                )
            except (AttributeError, ValueError) as e:
                message = str(e)
                # Only fall back on the specific deepseek_v4 / missing
                # max_position_embeddings signature. Everything else
                # re-raises unchanged.
                if "config" in kwargs or (
                    "deepseek_v4" not in message
                    and "max_position_embeddings" not in message
                ):
                    raise
                warnings.warn(
                    "Falling back to generic tokenizer config because "
                    "Transformers does not recognize this model config: "
                    f"{e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return _UpstreamAutoTokenizer.from_pretrained(
                    model_path,
                    *args,
                    config=PreTrainedConfig(),
                    **kwargs,
                )

        # Forward any other attribute access (e.g. .register, .from_config)
        # to the upstream AutoTokenizer untouched. mlx-lm registers a
        # NewlineTokenizer this way.
        def __getattr__(self, name):
            return getattr(_UpstreamAutoTokenizer, name)

    # Forward class-level attribute access too, so callers that use
    # ``AutoTokenizer.register(...)`` instead of an instance work.
    class _Meta(type):
        def __getattr__(cls, name):
            return getattr(_UpstreamAutoTokenizer, name)

    return _Meta(
        "AutoTokenizer",
        (_DeepSeekV4AwareAutoTokenizer,),
        {},
    )


def apply_tokenizer_patch() -> bool:
    """Replace ``mlx_lm.tokenizer_utils.AutoTokenizer`` with the wrapper.

    Idempotent. Only the binding inside ``mlx_lm.tokenizer_utils`` is
    swapped — global ``transformers.AutoTokenizer`` is untouched, so
    code outside mlx-lm is unaffected.
    """
    global _PATCHED
    if _PATCHED:
        return False

    wrapper = _build_wrapper()
    _tu.AutoTokenizer = wrapper
    _PATCHED = True
    logger.info(
        "mlx_lm.tokenizer_utils.AutoTokenizer wrapped "
        "(deepseek_v4 / max_position_embeddings fallback active)"
    )
    return True


_LOAD_PATCHED = False


def _register_chat_template_and_parser_modules() -> None:
    """Register our chat_template_v4 / tool_parser_v4 as if they lived
    inside mlx-lm. mlx-lm's tokenizer_config dispatcher does
    ``importlib.import_module(f"mlx_lm.chat_templates.{type}")`` and
    ``importlib.import_module(f"mlx_lm.tool_parsers.{type}")``, so
    putting our modules under those qualified names keeps the upstream
    code path intact.
    """
    import sys

    # chat_templates.deepseek_v4
    if "mlx_lm.chat_templates.deepseek_v4" not in sys.modules:
        from . import chat_template_v4 as _ct

        sys.modules["mlx_lm.chat_templates.deepseek_v4"] = _ct
    # tool_parsers.deepseek_v4
    if "mlx_lm.tool_parsers.deepseek_v4" not in sys.modules:
        from . import tool_parser_v4 as _tp

        sys.modules["mlx_lm.tool_parsers.deepseek_v4"] = _tp


def _is_deepseek_v4_model(model_path) -> bool:
    """Return True if ``model_path/config.json`` declares deepseek_v4."""
    import json
    from pathlib import Path

    p = Path(model_path) / "config.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("model_type") == "deepseek_v4"
    except Exception:
        return False


def apply_load_patch() -> bool:
    """Wrap ``mlx_lm.tokenizer_utils.load`` so DeepSeek V4 models get the
    DSML chat_template + tool_parser injected even when the published
    ``tokenizer_config.json`` ships a minimal jinja with no tool grammar.

    Approach: call upstream ``load``, then on the returned
    ``TokenizerWrapper`` overwrite ``_chat_template``, ``_tool_parser``,
    ``_tool_call_start``, ``_tool_call_end``, and the cached
    ``_tool_call_*_tokens`` so they reflect our V4 modules. Touching
    ``_attribute`` directly is hacky, but mlx-lm is pinned by commit so
    the layout is stable, and going through the public init kwargs
    would require either rewriting tokenizer_config.json on disk or
    re-running ``load`` recursively — both worse trade-offs.
    """
    global _LOAD_PATCHED
    if _LOAD_PATCHED:
        return False

    _register_chat_template_and_parser_modules()

    orig_load = _tu.load

    def patched_load(model_path, tokenizer_config_extra=None, eos_token_ids=None):
        wrapper = orig_load(
            model_path,
            tokenizer_config_extra=tokenizer_config_extra,
            eos_token_ids=eos_token_ids,
        )

        if not _is_deepseek_v4_model(model_path):
            return wrapper

        from . import chat_template_v4 as _ct
        from . import tool_parser_v4 as _tp

        # Skip if the published tokenizer_config already wired up V4 by
        # itself — leave whatever the user / publisher chose alone.
        if wrapper._chat_template is None:
            wrapper._chat_template = _ct.apply_chat_template
            wrapper.has_chat_template = True
        if wrapper._tool_parser is None:
            wrapper._tool_parser = _tp.parse_tool_call
            wrapper._tool_call_start = _tp.tool_call_start
            wrapper._tool_call_end = _tp.tool_call_end
            try:
                wrapper._tool_call_start_tokens = tuple(
                    wrapper._tokenizer.encode(
                        _tp.tool_call_start, add_special_tokens=False
                    )
                )
                wrapper._tool_call_end_tokens = tuple(
                    wrapper._tokenizer.encode(
                        _tp.tool_call_end, add_special_tokens=False
                    )
                )
            except Exception as e:
                logger.warning(
                    "Could not encode DSML tool-call markers as tokens: %s", e
                )
        logger.info(
            "Injected DeepSeek V4 DSML chat_template + tool_parser into "
            "TokenizerWrapper for %s",
            model_path,
        )
        return wrapper

    _tu.load = patched_load
    # mlx_lm.utils does ``from .tokenizer_utils import load as
    # _load_tokenizer`` at module import, so patching only the module
    # attribute on tokenizer_utils misses the already-bound reference
    # on mlx_lm.utils. Replace both bindings so any call site
    # (mlx_lm.utils.load, direct mlx_lm.tokenizer_utils.load) hits the
    # patched function.
    try:
        import mlx_lm.utils as _mu

        if hasattr(_mu, "_load_tokenizer"):
            _mu._load_tokenizer = patched_load
    except Exception as e:
        logger.warning(
            "Could not patch mlx_lm.utils._load_tokenizer (V4 chat_template "
            "injection may not fire): %s",
            e,
        )
    _LOAD_PATCHED = True
    logger.info(
        "mlx_lm.tokenizer_utils.load wrapped "
        "(injects DSML chat_template + tool_parser for deepseek_v4)"
    )
    return True
