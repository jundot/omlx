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
                if (
                    "config" in kwargs
                    or (
                        "deepseek_v4" not in message
                        and "max_position_embeddings" not in message
                    )
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
