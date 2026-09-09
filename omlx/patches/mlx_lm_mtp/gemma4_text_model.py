# SPDX-License-Identifier: Apache-2.0
"""Lightning MTP for merged gemma4 checkpoints on the mlx-lm path.

Google ships the gemma4 draft head as a separate ``gemma4_assistant``
model, which ``oq.combine_gemma4_assistant_mtp`` merges under
``language_model.mtp.*``. The head was only ever attached on the mlx-vlm
side (``mlx_vlm_mtp.gemma4_vlm_runtime``), so a checkpoint served through
mlx-lm — a text-only quant, a DFlash target, the VLM->LLM fallback — had
no binding site and this module simply discarded the head. It still does
when MTP is off; when MTP is on it attaches the head and supplies the two
things the head reads from a backbone forward — the per-layer-type K/V
banks (``shared_kv_sink``) and the pre-norm hidden. ``draft_step`` in
``mlx_lm_gemma4_assistant`` then drives it, shared with the mlx-vlm path.

Hidden is captured BEFORE the trunk RMSNorm. ``_trunk_norm_module``
re-applies it for any model that does not set
``_omlx_mtp_head_hidden_normed``, so capturing post-norm would double-norm
the head's inputs — silently, as a weak acceptance rate.

Both outputs come from wrapping gemma4's layer loop rather than copying
it: each ``DecoderLayer`` already returns its ``kvs``, and swapping
``norm`` for a recorder captures its input. Reading the K/V back out of
the cache instead would be wrong, because ``RotatingKVCache.state``
returns the raw ring and a wrapped sliding layer would hand the head
out-of-order banks.

The sink and the stash both carry state out of a forward, and neither can
cross requests: MLX runs on the one-worker executor in
``omlx.engine_core`` (issue #85), and MTP is admitted only for singleton
batches. The sink is a ``threading.local`` anyway, which costs nothing and
bounds a failure that would otherwise draft over another request's K/V.
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
            from ..mlx_lm_gemma4_assistant import query_position

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
        from ..mlx_lm_gemma4_assistant import draft_step

        return draft_step(self, hidden_states, next_token_ids, return_hidden)

    def make_mtp_cache(self):
        """The assistant head is stateless."""
        return []

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Trim the rejected tail after a depth-k verify over [confirmed, d1..dk].

        Gemma 4 is attention-only, so unlike qwen35 there is no recurrent
        state to restore and replay: every layer drops
        ``num_drafts - accepted`` positions. Every layer is asked whether it
        can trim before any of them is trimmed, because a half-rolled-back
        cache is unrecoverable.

        A sliding layer whose ring has wrapped still trims: ``cache_rollback``
        arms an undo log around the verify forward, so ``is_trimmable()``
        answers for that snapshot rather than for the ring. A layer carrying
        neither returns False and the caller takes the standard step.
        """
        layers = self.model.layers
        if len(cache) != len(layers):
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

    The head ships bfloat16 and the combine step preserves that, so in a
    float16 build every head matmul meets float16 activations and MLX
    promotes the result to float32 — the head then runs at twice the
    memory traffic it needs, on the decode hot path. Mirrors the
    bfloat16 -> float16 normalization deepseek_v4_model already applies to
    its own MTP metadata.

    Either direction is safe to cast, for a reason specific to drafting:
    the head only proposes. Every token the engine emits is one the
    backbone verified, so head precision moves the acceptance rate and can
    never move the output. bfloat16 -> float16 is in any case lossless
    here (float16 carries 10 mantissa bits against bfloat16's 7, and the
    head's largest weight is 23.1 against a 65504 ceiling); float16 ->
    bfloat16, which needs a checkpoint whose head and backbone were
    written by different tools, drops 3 mantissa bits and is still worth
    it against a float32 hot path.

    The target is the dtype most of the backbone is stored in, not the
    first one seen: in a quantized checkpoint most tensors are packed
    uint32 with float scales beside them, and dict order would otherwise
    let one unrepresentative tensor choose for the whole model.
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
    logger.debug("mlx-lm gemma4 assistant MTP patch applied")
    return True
