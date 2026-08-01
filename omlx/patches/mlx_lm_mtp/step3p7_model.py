# SPDX-License-Identifier: Apache-2.0
"""Native MTP (Multi-Token Prediction) monkey-patch for Step-3.7-Flash.

Neither StepFun's own reference `modeling_step3p7.py` nor oMLX's vendored
`mlx_lm.models.step3p7`/`step3p5` implement Step-3.7's MTP draft layer at
all -- `step3p5.Model.sanitize()` actively discards any weight key for a
layer index >= num_hidden_layers (and anything containing ".mtp"), by
design, matching upstream mlx-lm's stance that Step-3.5/3.7 support ships
without MTP. This patch adds it, following oMLX's own proven pattern for
DeepSeek-V4 (`patches/mlx_lm_mtp/deepseek_v4_model.py`) and GLM-5.2
(`glm_moe_dsa_model.py`) -- both already-working implementations of the
exact same DeepSeek-V3-style nextn design Step-3.7 uses (same enorm/hnorm/
eh_proj/shared_head naming).

The forward pass below is not inferred from tensor shapes -- it is a
direct port of StepFun's own reference C++ implementation
(stepfun-ai/llama.cpp, branch `step3p5-mtp`, `src/models/step35-iswa.cpp`):

    inp_tokens  = embed_tokens(next_token)          # trunk's shared table
    prev_hidden = trunk's post-final-norm hidden     # == Step3p5Model.__call__'s return value
    inp_normed  = enorm(inp_tokens)                  # RMSNorm
    hid_normed  = hnorm(prev_hidden)                 # RMSNorm
    x           = eh_proj(concat(inp_normed, hid_normed, axis=-1))
    x           = one full Step3p5DecoderLayer(x)    # same attn+MLP code as every other layer
    out         = shared_head.norm(x)                # layer-45's own norm (present in our checkpoint)
    logits      = shared_head.head(out)               # layer-45's own output proj (present in our checkpoint)

StepFun's own runtime uses only the first (and in our checkpoint, only
extracted) MTP layer -- depth-1, no chaining -- matching DeepSeek-V4's
"legacy depth-1 cycle" in oMLX's own MTP framework, not GLM's chained
depth-k path. Layer 45 is NOT in step3p5's MoE range (`range(1,
num_hidden_layers)` excludes 45 exactly as it excludes 0), so its MLP is
the plain dense `Step3p5MLP`, matching what was actually extracted from
the upstream checkpoint (see `attach_step3p7_mtp.py`).

Weight keys as saved by `attach_step3p7_mtp.py` (language_model.-prefixed,
matching this checkpoint's VLM-wrapper naming):

    language_model.model.layers.45.eh_proj.{weight,scales,biases}
    language_model.model.layers.45.enorm.weight
    language_model.model.layers.45.hnorm.weight
    language_model.model.layers.45.input_layernorm.weight
    language_model.model.layers.45.self_attn.{q,k,v,o,g}_proj.{weight,scales,biases}
    language_model.model.layers.45.self_attn.{q,k}_norm.weight
    language_model.model.layers.45.post_attention_layernorm.weight
    language_model.model.layers.45.mlp.{gate,up,down}_proj.{weight,scales,biases}
    language_model.model.layers.45.transformer.shared_head.norm.weight
    language_model.model.layers.45.transformer.shared_head.output.{weight,scales,biases}

`sanitize()` remaps these onto a single `self.mtp` attribute (not a list --
depth is always 1 here) rather than extending `self.layers`, which would
require excluding index 45 from every ordinary (non-speculative) decode
pass. This mirrors DeepSeek-V4/GLM's own `mtp.<idx>.*` convention exactly.

Verified end-to-end on a REAP-pruned + mixed-precision-quantized Step-3.7-Flash checkpoint
(VLM-shaped: `language_model.`-prefixed keys), via oMLX's own built-in throughput benchmark,
mtp_enabled true vs false, same checkpoint/prompts, A/B across reloads:

    pp 4096 / tg 128  : 52.7 -> 64.4 tok/s  (1.22x)
    pp 8192 / tg 128  : 51.2 -> 63.3 tok/s  (1.24x)
    pp 16384 / tg 128 : 48.8 -> 61.1 tok/s  (1.25x)
    pp 32768 / tg 128 : 42.6 -> 55.2 tok/s  (1.30x)

Speedup grows with context; costs +0.4GB peak memory (the draft head). ~1.25x is the expected
magnitude for this checkpoint, not a ceiling: StepFun's own runtime is depth-1 (one draft token
per verify cycle), which caps the theoretical best at 2x, so this reflects a moderate draft
acceptance rate with both the trunk and the draft head aggressively quantized.

Getting from "loads without error" to "actually measurably faster" required finding two
completely silent bugs -- both looked identical from the outside (clean load, correct output,
zero throughput gain), and neither raised an exception anywhere:

1. `step3p5.ModelArgs` is a dataclass, and `BaseModelArgs.from_dict` keeps only declared fields.
   `num_nextn_predict_layers` isn't one (upstream Step-3.5/3.7 support ships without MTP), so it
   was silently dropped every time. `Model.__init__` then read `getattr(args,
   "num_nextn_predict_layers", 0)` as 0 and never attached `self.mtp` at all -- the model
   generated perfectly with the entire MTP path inert. Fixed below in `_patch_model_args()`,
   which wraps `from_dict` to restore the attribute the dataclass discarded -- the same fix shape
   `deepseek_v4_model._patch_model_args` and `glm_moe_dsa_model` already use for their own
   architectures.
2. Separately, `utils/model_loading.py`'s `_nextn_weight_prefixes()` builds bare
   `model.layers.{n}.` prefixes only, so `_checkpoint_has_mtp_weights()` reports zero MTP weight
   tensors present for any VLM-shaped checkpoint (`language_model.model.layers.{n}.*`) -- the
   settings UI refuses to let `mtp_enabled` be toggled on at all for such a checkpoint, citing
   "config declares mtp layers but the weight files contain neither mtp.* tensors nor native
   nextn layers". This is a real upstream gap, not specific to Step-3.7 -- it affects any
   VLM-shaped checkpoint using the nextn (extra-decoder-layer) MTP layout. Companion fix included
   in this same change: emit `language_model.model.layers.{n}.` and
   `model.language_model.layers.{n}.` variants too.

`mtp_clamp_accept`/`mtp_partial_rollback` (draft-rejection cache rollback) are simplified versions
of `deepseek_v4_model.py`'s pattern -- they check `is_trimmable()` only, without that file's
rotated-cache `_mtp_undo` fallback. More conservative (may decline a rollback DeepSeek-V4's fuller
check would allow), not incorrect; the throughput numbers above reflect this as-is.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _is_our_method(cls: Any, attr: str, marker: str) -> bool:
    existing = cls.__dict__.get(attr)
    return getattr(existing, marker, False)


def apply() -> bool:
    """Apply the Step-3.7 MTP model-side patches.

    Must run after oMLX's own step3p7/step3p5 base patches have registered
    their modules (patches/step3p7/__init__.py, and step3p5 is upstream
    mlx-lm so it's always present). Call from
    patches/mlx_lm_mtp/__init__.py's apply_mlx_lm_mtp_patch(), alongside
    the existing qwen35/deepseek_v4/glm_moe_dsa/nemotron_h calls.
    """
    step3p5 = sys.modules.get("mlx_lm.models.step3p5")
    step3p7 = sys.modules.get("mlx_lm.models.step3p7")
    if step3p5 is None or not hasattr(step3p5, "Model"):
        logger.debug("step3p5 module not registered; skipping MTP patch")
        return False
    if step3p7 is None or not hasattr(step3p7, "Model"):
        logger.debug("step3p7 module not registered; skipping MTP patch")
        return False

    _patch_model_args(step3p5)
    _register_mtp_block(step3p5)
    _patch_step3p5_model(step3p5)
    _patch_step3p7_wrapper(step3p7)

    if not hasattr(step3p5.Model, "_omlx_mtp_patched"):
        step3p5.Model._omlx_mtp_patched = "patch"
        logger.info("Step-3.7 MTP model patch applied")
    return True


# ---------------------------------------------------------------------------
# ModelArgs — preserve num_nextn_predict_layers through from_dict.
# ---------------------------------------------------------------------------


def _patch_model_args(step3p5: Any) -> None:
    """Carry ``num_nextn_predict_layers`` onto the constructed ModelArgs.

    ``step3p5.ModelArgs`` is a dataclass and ``BaseModelArgs.from_dict``
    keeps only keys that are declared fields. ``num_nextn_predict_layers``
    is not one of them (upstream Step-3.5/3.7 support ships without MTP),
    so it is silently dropped -- ``getattr(args, "num_nextn_predict_layers",
    0)`` in the patched ``__init__`` then reads 0, ``self.mtp`` is never
    attached, and the model loads and generates perfectly while MTP never
    engages. That failure is completely silent: no error, correct output,
    and (measured) zero speedup, which is exactly how it presented before
    this was found.

    Same fix shape as ``deepseek_v4_model._patch_model_args`` /
    ``glm_moe_dsa_model``'s equivalent: wrap ``from_dict``, let the original
    build the args, then set the attribute the dataclass discarded.

    step3p7's own ``ModelArgs.__post_init__`` builds its ``text_config`` via
    ``TextConfig.from_dict(...)`` where ``TextConfig is step3p5.ModelArgs``,
    so patching here catches the nested VLM-wrapper path too -- that nested
    text_config object is exactly what ``Step3p5Model``/``Model.__init__``
    receive as ``args``.
    """
    args_cls = step3p5.ModelArgs
    if "_omlx_mtp_args_patched" in args_cls.__dict__:
        return

    original_from_dict = args_cls.from_dict.__func__

    def patched_from_dict(cls, params):
        args = original_from_dict(cls, params)
        try:
            n_mtp = int(params.get("num_nextn_predict_layers", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            n_mtp = 0
        if not hasattr(args, "num_nextn_predict_layers"):
            args.num_nextn_predict_layers = n_mtp
        return args

    args_cls.from_dict = classmethod(patched_from_dict)
    args_cls._omlx_mtp_args_patched = True


# ---------------------------------------------------------------------------
# MTPBlock — enorm/hnorm/eh_proj + one reused Step3p5DecoderLayer + shared_head.
# ---------------------------------------------------------------------------


def _register_mtp_block(step3p5: Any) -> None:
    if hasattr(step3p5, "MTPBlock"):
        return

    import mlx.core as mx
    import mlx.nn as nn

    Step3p5DecoderLayer = step3p5.Step3p5DecoderLayer
    ZeroCenteredRMSNorm = step3p5.ZeroCenteredRMSNorm

    class MTPBlock(nn.Module):
        """Step-3.7's single MTP draft layer (index == num_hidden_layers).

        Ported from stepfun-ai/llama.cpp's `llm_build_step35_iswa`
        (branch step3p5-mtp, src/models/step35-iswa.cpp): normalize the
        next token's embedding and the trunk's post-final-norm hidden
        state separately, concatenate, project back to hidden_size, run
        one full decoder layer, then a dedicated output head.
        """

        def __init__(self, args, layer_idx: int):
            super().__init__()
            dim = args.hidden_size
            self.enorm = ZeroCenteredRMSNorm(dim, eps=args.rms_norm_eps)
            self.hnorm = ZeroCenteredRMSNorm(dim, eps=args.rms_norm_eps)
            self.eh_proj = nn.Linear(2 * dim, dim, bias=False)
            # layer_idx == num_hidden_layers, outside moe_layers_idx's
            # range(1, num_hidden_layers) exactly as layer 0 is -- dense
            # Step3p5MLP, matching the actual extracted checkpoint.
            self.block = Step3p5DecoderLayer(args, layer_idx)
            self.shared_head_norm = ZeroCenteredRMSNorm(dim, eps=args.rms_norm_eps)
            self.shared_head_head = nn.Linear(dim, args.vocab_size, bias=False)

        def __call__(self, prev_hidden, embed_tokens, input_ids, mask, cache):
            e = self.enorm(embed_tokens(input_ids))
            h = self.hnorm(prev_hidden)
            x = mx.concatenate([e, h], axis=-1)
            x = self.eh_proj(x)
            x = self.block(x, mask=mask, cache=cache)
            return x

        def head(self, x):
            return self.shared_head_head(self.shared_head_norm(x))

    step3p5.MTPBlock = MTPBlock


# ---------------------------------------------------------------------------
# step3p5.Model — attach self.mtp, add return_hidden/mtp_forward/make_mtp_cache,
# extend sanitize to keep+remap the layer-45 keys instead of dropping them.
# ---------------------------------------------------------------------------


def _patch_step3p5_model(step3p5: Any) -> None:
    cls = step3p5.Model
    init_wrapped = getattr(cls, "_omlx_mtp_init_wrapped", False)
    call_owned = _is_our_method(cls, "__call__", "_omlx_mtp_call_marker")
    if init_wrapped and call_owned:
        return

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    original_init = cls.__init__

    def __init__(self, args):
        original_init(self, args)
        n_mtp = 1 if int(getattr(args, "num_nextn_predict_layers", 0) or 0) > 0 else 0
        from . import is_mtp_active  # patches/mlx_lm_mtp package

        mtp_decode_enabled = bool(n_mtp > 0 and is_mtp_active())
        self._omlx_mtp_decode_enabled = mtp_decode_enabled
        if mtp_decode_enabled:
            n_main = args.num_hidden_layers
            self.mtp = step3p5.MTPBlock(args, n_main)
            # StepFun's own runtime is depth-1 (see module docstring) --
            # no chaining, unlike GLM's depth-k path.
            self._omlx_mtp_chain = False
            self._omlx_mtp_depth = 1
            self._omlx_mtp_head_clone = False

    def __call__(self, inputs, cache=None, return_hidden: bool = False, n_confirmed: int = 0):
        # n_confirmed: part of the patched-backbone interface batch_generator
        # expects (see deepseek_v4_model.__call__ for the same parameter).
        # Step-3.7 keeps all decode state in per-layer KVCache/RotatingKVCache
        # objects; draft rejection rolls back via cache.trim, so this is
        # accepted and unused, matching DeepSeek-V4's own comment.
        h = self.model(inputs, cache)  # already post-final-norm (Step3p5Model.__call__ returns self.norm(h))
        logits = self.lm_head(h)
        if return_hidden:
            return logits, h
        return logits

    def make_mtp_cache(self):
        if not hasattr(self, "mtp"):
            return None
        # Match make_cache()'s own per-layer dispatch: the MTP block's
        # inner Step3p5DecoderLayer sets its own is_sliding from layer_idx
        # (layer_types[] lookup or the layer_idx % 2 == 0 fallback), so
        # mirror that here rather than assuming one cache type.
        is_sliding = self.mtp.block.is_sliding
        return [
            RotatingKVCache(max_size=self.args.sliding_window) if is_sliding else KVCache()
        ]

    def mtp_forward(self, h, input_ids, cache=None, return_hidden: bool = False, logits_keep: int = 0):
        """Run the single MTP block + its own head on the trunk's hidden state.

        `h` is the trunk's post-final-norm hidden (same convention
        `__call__(return_hidden=True)` returns) -- Step-3.7's reference
        implementation documents this hidden as what `hnorm` consumes.
        """
        if cache is None:
            cache = [None]
        layer_cache = cache[0]
        mask_input = h
        mask = create_attention_mask(
            mask_input, layer_cache, window_size=self.args.sliding_window
        )

        out = self.mtp(h, self.model.embed_tokens, input_ids, mask, layer_cache)

        logits_source = out
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:]
        logits = self.mtp.head(logits_source)
        if return_hidden:
            return logits, out
        return logits

    def _cache_can_trim(c) -> bool:
        is_trimmable = getattr(c, "is_trimmable", None)
        if callable(is_trimmable):
            return is_trimmable()
        return False

    def mtp_clamp_accept(self, cache, accepted: int, num_drafts: int) -> int:
        for m in range(accepted, -1, -1):
            n = num_drafts - m
            if n <= 0 or all(_cache_can_trim(c) for c in cache):
                return m
        return 0

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        n = num_drafts - accepted
        if n <= 0:
            return True
        for c in cache:
            if c.trim(n) != n:
                logger.warning("Step-3.7 MTP rollback trim shortfall on %s", type(c).__name__)
                return False
        return True

    original_sanitize = cls.sanitize

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        n_main = self.args.num_hidden_layers
        has_mtp = hasattr(self, "mtp")

        # Pull out this checkpoint's layer-45 (== n_main) keys before the
        # original sanitize runs -- its own layer-index filter does not
        # reliably catch language_model.-prefixed keys (parts[2] lands on
        # "layers", not the digit, for that prefix shape), but keeping the
        # extraction explicit here rather than relying on that quirk.
        mtp_prefix_candidates = (
            f"model.layers.{n_main}.",
            f"language_model.model.layers.{n_main}.",
        )
        # StepFun's own runtime is depth-1 (see module docstring): only
        # layer n_main is ever attached/used. The upstream shard our
        # extraction script pulled from also carries layers n_main+1 and
        # n_main+2 (unused there too), so those keys can be present in a
        # checkpoint that over-extracted. The ORIGINAL sanitize's own
        # ">= num_hidden_layers" filter is supposed to catch them but does
        # not reliably match language_model.-prefixed keys (parts[2] lands
        # on the literal "layers" segment, not the digit, for that prefix
        # shape) -- so anything beyond n_main is dropped explicitly here
        # rather than trusted to that filter.
        import re

        extra_layer_re = re.compile(r"(?:language_model\.)?model\.layers\.(\d+)\.")

        mtp_weights: Dict[str, Any] = {}
        rest: Dict[str, Any] = {}
        for k, v in weights.items():
            matched = next((p for p in mtp_prefix_candidates if k.startswith(p)), None)
            if matched is not None:
                mtp_weights[k[len(matched):]] = v
                continue
            m = extra_layer_re.match(k)
            if m is not None and int(m.group(1)) > n_main:
                continue
            rest[k] = v

        new_weights = dict(original_sanitize(self, rest))

        if not has_mtp or not mtp_weights:
            return new_weights

        rename = {
            "eh_proj.": "mtp.eh_proj.",
            "enorm.": "mtp.enorm.",
            "hnorm.": "mtp.hnorm.",
            "transformer.shared_head.norm.": "mtp.shared_head_norm.",
            "transformer.shared_head.output.": "mtp.shared_head_head.",
            "input_layernorm.": "mtp.block.input_layernorm.",
            "post_attention_layernorm.": "mtp.block.post_attention_layernorm.",
            "self_attn.": "mtp.block.self_attn.",
            "mlp.": "mtp.block.mlp.",
        }
        for k, v in mtp_weights.items():
            for old, new in rename.items():
                if k.startswith(old):
                    new_weights[new + k[len(old):]] = v
                    break
            else:
                logger.warning("Step-3.7 MTP: unmapped weight key %r, dropped", k)

        return new_weights

    if not init_wrapped:
        cls.__init__ = __init__
        cls._omlx_mtp_init_wrapped = True
    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_forward = mtp_forward
    cls.mtp_clamp_accept = mtp_clamp_accept
    cls.mtp_partial_rollback = mtp_partial_rollback
    cls.sanitize = sanitize


# ---------------------------------------------------------------------------
# step3p7.Model — thin wrapper delegates to self.language_model.
# ---------------------------------------------------------------------------


def _patch_step3p7_wrapper(step3p7: Any) -> None:
    cls = step3p7.Model
    if _is_our_method(cls, "__call__", "_omlx_mtp_call_marker"):
        return

    def __call__(self, inputs, cache=None, return_hidden: bool = False, n_confirmed: int = 0):
        return self.language_model(inputs, cache, return_hidden=return_hidden, n_confirmed=n_confirmed)

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    def mtp_forward(self, h, input_ids, cache=None, return_hidden: bool = False, logits_keep: int = 0):
        return self.language_model.mtp_forward(
            h, input_ids, cache=cache, return_hidden=return_hidden, logits_keep=logits_keep
        )

    def mtp_clamp_accept(self, cache, accepted: int, num_drafts: int) -> int:
        return self.language_model.mtp_clamp_accept(cache, accepted, num_drafts)

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        return self.language_model.mtp_partial_rollback(cache, accepted, num_drafts)

    @property
    def _omlx_mtp_decode_enabled(self):
        return getattr(self.language_model, "_omlx_mtp_decode_enabled", False)

    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_forward = mtp_forward
    cls.mtp_clamp_accept = mtp_clamp_accept
    cls.mtp_partial_rollback = mtp_partial_rollback
    cls._omlx_mtp_decode_enabled = _omlx_mtp_decode_enabled
