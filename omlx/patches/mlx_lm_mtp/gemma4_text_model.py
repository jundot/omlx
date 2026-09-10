# SPDX-License-Identifier: Apache-2.0
"""Lightning MTP for merged gemma4 checkpoints on the mlx-lm path.

``oq.combine_gemma4_assistant_mtp`` merges Google's separate
``gemma4_assistant`` model under ``language_model.mtp.*``. The head only
ever had a binding site on the mlx-vlm side, so a checkpoint served
through mlx-lm -- a text-only quant, a DFlash target, the VLM->LLM
fallback -- discarded it. With MTP on this attaches it and supplies what
it reads from a backbone forward: the per-layer-type K/V banks and the
pre-norm hidden. ``mlx_lm_gemma4_assistant.draft_step`` drives it, shared
with the mlx-vlm path.

Hidden is captured BEFORE the trunk RMSNorm, which
``_trunk_norm_module`` re-applies -- a post-norm capture would
double-norm the head's input and show up only as a weak accept rate.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_MTP_KEY_PREFIXES = ("mtp.", "model.mtp.", "language_model.mtp.")

_active = threading.local()


class _NormRecorder:
    """Stand in for the trunk RMSNorm and keep its input."""

    def __init__(self, norm):
        self._norm = norm
        self.pre_norm = None

    def __call__(self, x):
        self.pre_norm = x
        return self._norm(x)


def _patch_model_args(mod: Any) -> None:
    """Carry ``mtp_assistant_config`` past ``BaseModelArgs.from_dict``.

    ``from_dict`` keeps only keys matching the dataclass signature, so the
    merged assistant config is dropped before the head can be sized.
    """
    cls = mod.ModelArgs
    if getattr(cls, "_omlx_mtp_from_dict_patched", False):
        return
    original = cls.from_dict.__func__

    def from_dict(inner_cls, params):
        args = original(inner_cls, params)
        assistant = params.get("mtp_assistant_config")
        if assistant is not None:
            args.mtp_assistant_config = assistant
        return args

    cls.from_dict = classmethod(from_dict)
    cls._omlx_mtp_from_dict_patched = True


def _patch_decoder_layer(mod: Any) -> None:
    cls = mod.DecoderLayer
    if getattr(cls, "_omlx_kv_sink_patched", False):
        return
    original = cls.__call__

    def __call__(self, *args, **kwargs):
        h, kvs, offset = original(self, *args, **kwargs)
        sink = getattr(_active, "sink", None)
        if sink is not None and kvs is not None:
            sink[self.layer_type] = kvs
        return h, kvs, offset

    cls.__call__ = __call__
    cls._omlx_kv_sink_patched = True


def _patch_text_model(mod: Any) -> None:
    """Accept ``shared_kv_sink`` and expose the pre-norm hidden."""
    cls = mod.Gemma4TextModel
    if getattr(cls, "_omlx_hidden_hook_patched", False):
        return
    original = cls.__call__

    def __call__(self, *args, **kwargs):
        sink = kwargs.pop("shared_kv_sink", None)
        want_hidden = bool(kwargs.pop("return_hidden", False))
        if sink is None and not want_hidden:
            return original(self, *args, **kwargs)

        recorder = _NormRecorder(self.norm) if want_hidden else None
        previous = getattr(_active, "sink", None)
        _active.sink = sink
        if recorder is not None:
            self.norm = recorder
        try:
            out = original(self, *args, **kwargs)
        finally:
            if recorder is not None:
                self.norm = recorder._norm
            _active.sink = previous

        if recorder is not None:
            self._omlx_last_pre_norm = recorder.pre_norm
        return out

    cls.__call__ = __call__
    cls._omlx_hidden_hook_patched = True


def _patch_inner_model(mod: Any) -> None:
    """Attach the head and serve ``mtp_forward`` on ``gemma4_text.Model``."""
    cls = mod.Model
    if getattr(cls, "_omlx_mtp_attach_patched", False):
        return

    # Bound at patch time: both run per decode step, where a function-level
    # import would pay the import machinery on every draft.
    from ..mlx_lm_gemma4_assistant import draft_step, query_position

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, args):
        original_init(self, args)
        assistant = getattr(args, "mtp_assistant_config", None)
        from . import get_mtp_depth, is_mtp_active

        enabled = bool(assistant) and is_mtp_active()
        self._omlx_mtp_decode_enabled = enabled
        if not enabled:
            return
        from ..mlx_lm_gemma4_assistant import build_draft_model

        self.mtp = build_draft_model(assistant)
        # Function refs read weights at call time, so binding pre-load is safe.
        self.mtp.bind(self)
        self._omlx_mtp_chain = True
        self._omlx_mtp_depth = get_mtp_depth()

    def __call__(self, inputs, cache=None, **kwargs):
        want_hidden = bool(kwargs.pop("return_hidden", False))
        kwargs.pop("n_confirmed", None)  # no GDN split on gemma4
        sink = kwargs.pop("shared_kv_sink", None)
        if not want_hidden and sink is None:
            return original_call(self, inputs, cache=cache, **kwargs)

        # The engine asks for return_hidden without supplying a sink.
        if sink is None and getattr(self, "mtp", None) is not None:
            sink = {}

        inner = dict(kwargs)
        if sink is not None:
            inner["shared_kv_sink"] = sink
        if want_hidden:
            inner["return_hidden"] = True
        out = self.model(inputs, cache=cache, **inner)

        if sink is not None and getattr(self, "mtp", None) is not None:
            # mtp_forward is a separate top-level call, so this outlives the
            # forward; see the module docstring for why that is safe.
            self._omlx_mtp_shared_kv = sink
            self._omlx_mtp_cache_ref = cache
            self._omlx_mtp_kv_offset = query_position(self)

        if not want_hidden:
            return self._omlx_logits(out)
        # Set on the inner Gemma4TextModel by _patch_text_model, not here.
        pre_norm = getattr(self.model, "_omlx_last_pre_norm", None)
        return self._omlx_logits(out), pre_norm

    def _omlx_logits(self, hidden):
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(hidden)
        else:
            out = self.lm_head(hidden)
        if self.final_logit_softcapping is not None:
            out = mod.logit_softcap(self.final_logit_softcapping, out)
        return out

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        """Drive the assistant head; the mlx-vlm path drives it the same way."""
        del mtp_cache, logits_keep  # stateless head, one output position
        return draft_step(self, hidden_states, next_token_ids, return_hidden)

    def make_mtp_cache(self):
        """The assistant head is stateless."""
        return []

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Trim the rejected tail after a depth-k verify.

        Every layer is asked before any is trimmed, because a half
        rolled-back cache is unrecoverable. The cache holds one entry per
        distinct K/V owner, not one per layer -- E2B/E4B share the last
        ``num_kv_shared_layers``, so a per-layer count refuses every
        rollback there.
        """
        layers = self.model.layers
        previous_kvs = getattr(self.model, "previous_kvs", None)
        expected = len(set(previous_kvs)) if previous_kvs else len(layers)
        if len(cache) != expected:
            return False
        trim_n = num_drafts - accepted
        if trim_n <= 0:
            return True
        live = [c for c in cache if c is not None]
        if not live:
            return False
        for c in live:
            if not (hasattr(c, "is_trimmable") and c.is_trimmable()):
                return False
        for c in live:
            c.trim(trim_n)
        return True

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls._omlx_logits = _omlx_logits
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_partial_rollback = mtp_partial_rollback
    cls._omlx_mtp_attach_patched = True


def _is_head_key(key: str) -> bool:
    return key.startswith(_MTP_KEY_PREFIXES) or ".mtp." in key


def _align_head_dtype(weights: dict) -> dict:
    """Store the assistant head at the backbone's float dtype.

    The head ships bfloat16, so on a float16 build every head matmul
    promotes to float32. Safe in either direction: the head only proposes,
    and every token emitted is one the backbone verified.
    """
    from collections import Counter

    import mlx.core as mx

    counts: Counter = Counter()
    for key, value in weights.items():
        if _is_head_key(key):
            continue
        if value.dtype in (mx.float16, mx.bfloat16):
            counts[value.dtype] += 1
    if not counts:
        return weights
    target = counts.most_common(1)[0][0]

    aligned = {}
    for key, value in weights.items():
        if (
            _is_head_key(key)
            and value.dtype in (mx.float16, mx.bfloat16)
            and value.dtype != target
        ):
            value = value.astype(target)
        aligned[key] = value
    return aligned


def _patch_outer_model(mod: Any) -> None:
    """Teach ``gemma4.Model``, which is the object the engine holds.

    The engine splits its lookups: ``_model_has_mtp_module`` walks
    ``language_model`` for ``.mtp``, but ``hasattr(model, "mtp_forward")``
    and the ``return_hidden`` forward both run against the outer wrapper,
    whose ``__call__`` lists its parameters and so rejects the kwarg.
    """
    cls = mod.Model
    if getattr(cls, "_omlx_mtp_outer_patched", False):
        return
    original = cls.__call__

    def __call__(self, inputs, cache=None, **kwargs):
        if not (kwargs.get("return_hidden") or kwargs.get("shared_kv_sink")):
            for key in ("return_hidden", "shared_kv_sink", "n_confirmed"):
                kwargs.pop(key, None)
            return original(self, inputs, cache=cache, **kwargs)
        return self.language_model(inputs, cache=cache, **kwargs)

    def mtp_forward(self, *args, **kwargs):
        return self.language_model.mtp_forward(*args, **kwargs)

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def sanitize(self, weights):
        """Drop the merged head, or align its dtype to the backbone.

        Attachment already happened in ``__init__``, so the head's own
        presence is the condition — no second read of the active flag.
        """
        if getattr(self.language_model, "mtp", None) is None:
            weights = {
                k: v
                for k, v in weights.items()
                if not k.removeprefix("model.").startswith(_MTP_KEY_PREFIXES)
            }
        else:
            weights = _align_head_dtype(weights)
        return original_sanitize(self, weights)

    original_sanitize = cls.sanitize
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.sanitize = sanitize
    cls._omlx_mtp_outer_patched = True


def apply() -> bool:
    """Patch mlx-lm's gemma4 pair. False when mlx-lm is not importable."""
    try:
        from mlx_lm.models import gemma4 as lm_gemma4
        from mlx_lm.models import gemma4_text as lm_gemma4_text
    except ImportError:
        logger.debug("mlx_lm.models.gemma4 not importable; MTP patch skipped")
        return False

    _patch_model_args(lm_gemma4_text)
    _patch_decoder_layer(lm_gemma4_text)
    _patch_text_model(lm_gemma4_text)
    _patch_inner_model(lm_gemma4_text)
    _patch_outer_model(lm_gemma4)

    # head_dim 512 global layers fuse only at L=1, so a multi-row verify
    # drops to the unfused pass. Absent, the verify is slower, not wrong.
    try:
        from ..gemma4_verify_attention import apply_mlx_lm as _apply_verify_attn

        _apply_verify_attn()
    except Exception as exc:
        logger.debug("gemma4 verify attention unavailable on mlx-lm: %s", exc)

    logger.debug("mlx-lm gemma4 assistant MTP patch applied")
    return True
