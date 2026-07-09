# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 (glm_moe_dsa) native MTP head for the depth-k chain cycle.

GLM-5.2's checkpoint carries one extra decoder layer (index
``num_hidden_layers``) that implements DeepSeek-V3-style multi-token
prediction: the trunk's post-norm hidden and the embedding of the next
token are each RMSNorm'd (``hnorm`` / ``enorm``), concatenated, projected
back to ``hidden_size`` (``eh_proj``), and run through one full
``GlmMoeDsaDecoderLayer`` (own DSA indexer + 256-expert MoE); logits come
from ``shared_head.norm`` + the shared ``lm_head``. The layer is trained
as the speculative draft model, so it slots straight into the chain cycle.

oMLX registers the GLM base model via ``omlx/patches/glm_moe_dsa/``
(landing it in ``sys.modules['mlx_lm.models.glm_moe_dsa']``); this patch
sits on top and adds the MTP module + ``mtp_forward`` / ``make_mtp_cache``
plus the sanitize handling that keeps and remaps the extra layer's
weights (``model.layers.<n>.*`` -> ``mtp.<i>.*``). Apply order matches
deepseek_v4_model: the caller runs ``apply()`` after the base patch has
registered the module.

Convention notes (draft-quality critical):

- The trunk hidden the head consumes is the *post-final-norm* hidden —
  the same tensor the trunk's own lm_head reads. ``batch_generator``
  applies the trunk norm on 3D hidden rows (``_HEAD_HIDDEN_POST_NORM``),
  so the patched backbone ``__call__`` returns the raw pre-norm hidden.
- Chained depth>1 steps feed the head block's raw residual output back
  as the next ``h``; ``hnorm`` re-normalises it inside ``mtp_forward``.
- The head runs its own DSA indexer (the checkpoint ships indexer
  weights for the MTP layer, and the freq/offset pattern marks it
  "full"). ``index_share_for_mtp_iteration`` reuse of trunk indices is a
  possible future draft-cost optimisation, not a correctness knob.

Rollback: every cache in play (latent KV + indexer KV, head and trunk)
is a plain trimmable ``KVCache``, so draft rejection uses stock ``trim``
— no rotating-cache undo, no clamp, and the chain runs directly on the
persistent head cache (``_omlx_mtp_head_clone = False``).
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
    """Apply the GLM MTP model-side patches when the base patch is active."""
    glm = sys.modules.get("mlx_lm.models.glm_moe_dsa")
    if glm is None or not hasattr(glm, "Model"):
        logger.debug(
            "glm_moe_dsa module not registered; skipping MTP patch (this is "
            "expected for non-GLM models)"
        )
        return False

    _patch_model_args(glm)
    _register_mtp_block(glm)
    _patch_glm_model_call(glm)
    _patch_model(glm)

    if not hasattr(glm.Model, "_omlx_mtp_patched"):
        glm.Model._omlx_mtp_patched = "patch"
        logger.info("GLM-5.2 MTP model patch applied")
    return True


# ---------------------------------------------------------------------------
# ModelArgs — carry num_nextn_predict_layers and extend indexer_types to
# cover MTP layers.
# ---------------------------------------------------------------------------

def _patch_model_args(glm: Any) -> None:
    """Wrap ``ModelArgs.from_dict`` so MTP layers are constructible.

    ``num_nextn_predict_layers`` isn't a declared field (BaseModelArgs
    filters unknown keys), and ``indexer_types`` only spans the backbone,
    so ``GlmMoeDsaAttention(config, n_main + i)`` would IndexError. The
    wrapper stashes the count and appends indexer types for the MTP
    layers using the same freq/offset formula as ``__post_init__``.
    """
    args_cls = glm.ModelArgs
    if "_omlx_mtp_args_patched" in args_cls.__dict__:
        return

    original_from_dict = args_cls.from_dict.__func__

    def patched_from_dict(cls, params):
        args = original_from_dict(cls, params)
        n_mtp = int(params.get("num_nextn_predict_layers", 0) or 0)
        args.num_nextn_predict_layers = n_mtp
        if n_mtp > 0 and isinstance(args.indexer_types, list):
            n_main = int(args.num_hidden_layers)
            freq = max(int(getattr(args, "index_topk_freq", 1) or 1), 1)
            offset = int(getattr(args, "index_skip_topk_offset", 2) or 0)
            # Copy before extending — the list is shared with the caller's
            # config dict and must not grow in place.
            types = list(args.indexer_types)
            while len(types) < n_main + n_mtp:
                i = len(types)
                types.append(
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                )
            args.indexer_types = types
        return args

    args_cls.from_dict = classmethod(patched_from_dict)
    args_cls._omlx_mtp_args_patched = True


# ---------------------------------------------------------------------------
# GlmMTPBlock — register on the module.
# ---------------------------------------------------------------------------

def _register_mtp_block(glm: Any) -> None:
    if hasattr(glm, "GlmMTPBlock"):
        return

    import mlx.core as mx
    import mlx.nn as nn

    GlmMoeDsaDecoderLayer = glm.GlmMoeDsaDecoderLayer

    class GlmMTPBlock(nn.Module):
        """One MTP layer: enorm/hnorm fusion + a full GLM decoder layer.

        ``norm`` holds the checkpoint's ``shared_head.norm`` — applied to
        the block output before the shared lm_head (mtp_forward does the
        lm_head so ``logits_keep`` can shrink the vocab matmul).
        """

        def __init__(self, config, layer_idx: int):
            super().__init__()
            dim = config.hidden_size
            self.enorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.hnorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.eh_proj = nn.Linear(2 * dim, dim, bias=False)
            self.norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.block = GlmMoeDsaDecoderLayer(config, layer_idx)

        def __call__(self, h, embed_tokens, input_ids, mask, cache):
            e = self.enorm(embed_tokens(input_ids))
            x = self.eh_proj(mx.concatenate([e, self.hnorm(h)], axis=-1))
            x, _ = self.block(x, mask, cache, None)
            return x

    glm.GlmMTPBlock = GlmMTPBlock


# ---------------------------------------------------------------------------
# GlmMoeDsaModel — return_raw_hidden support.
# ---------------------------------------------------------------------------

def _patch_glm_model_call(glm: Any) -> None:
    """Replace ``GlmMoeDsaModel.__call__`` to optionally return the raw hidden."""
    cls = glm.GlmMoeDsaModel
    if _is_our_method(cls, "__call__", "_omlx_mtp_call_marker"):
        return

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    def __call__(self, x, cache=None, return_raw_hidden: bool = False):
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        prev_topk_indices = None
        for i in range(self.num_layers):
            h, prev_topk_indices = self.layers[self.start_idx + i](
                h, mask, cache[i], prev_topk_indices
            )

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        out = self.norm(h)
        if return_raw_hidden:
            return out, h
        return out

    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__


# ---------------------------------------------------------------------------
# Model — wrap __init__, replace __call__, add mtp_forward / make_mtp_cache,
# wrap sanitize with the MTP weight remapping.
# ---------------------------------------------------------------------------

def _patch_model(glm: Any) -> None:
    cls = glm.Model
    init_wrapped = getattr(cls, "_omlx_mtp_init_wrapped", False)
    call_owned = _is_our_method(cls, "__call__", "_omlx_mtp_call_marker")
    if init_wrapped and call_owned:
        return

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache

    from omlx.patches.glm_moe_dsa.deepseek_v32 import (
        _use_load_fused_wk_weights_proj,
    )

    original_init = cls.__init__
    original_sanitize = cls.sanitize

    def __init__(self, config):
        original_init(self, config)
        n_mtp = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
        # Gated on the MTP active-flag so mtp_enabled=False produces a
        # model indistinguishable from stock (same pattern as qwen35 /
        # deepseek_v4).
        from . import is_mtp_active

        mtp_decode_enabled = bool(n_mtp > 0 and is_mtp_active())
        self._omlx_mtp_decode_enabled = mtp_decode_enabled
        if mtp_decode_enabled:
            n_main = config.num_hidden_layers
            self.mtp = [glm.GlmMTPBlock(config, n_main + i) for i in range(n_mtp)]
            from . import get_mtp_depth

            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = get_mtp_depth()
            self._omlx_mtp_head_clone = False
            # Marginal cost prior for the adaptive depth controller: with
            # 8-of-256 routing each extra verify row pulls an almost
            # disjoint expert set (~55% of step bytes), measured ~35 ms
            # per row on GLM-5.2 vs the dense-backbone 7 ms default.
            self._omlx_mtp_marginal_ms = 35.0

    def __call__(
        self,
        inputs,
        cache=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        # ``n_confirmed`` only matters for models with module-level
        # recurrent state (Qwen3.5's GatedDeltaNet). GLM keeps all decode
        # state in trimmable KVCaches, so it is accepted and unused.
        if return_hidden:
            out, h_raw = self.model(inputs, cache, return_raw_hidden=True)
            return self.lm_head(out), h_raw
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def make_mtp_cache(self):
        """Two plain KVCaches per MTP block: latent KV + DSA indexer KV.

        A flat list (not a CacheList) so the chain's ``_mtp_head_trim_to``
        sees each cache's ``offset``/``trim`` directly; ``mtp_forward``
        re-slices pairs per block.
        """
        if not hasattr(self, "mtp"):
            return None
        caches = []
        for _ in self.mtp:
            caches.append(KVCache())
            caches.append(KVCache())
        return caches

    def mtp_forward(
        self,
        h,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        """Run the MTP block(s) + shared_head norm + shared lm_head.

        ``h`` is the trunk's post-norm hidden for the history fold, or the
        head's own raw output for chained draft steps — ``hnorm`` inside
        the block normalises either. ``return_hidden`` also returns the
        block's raw residual output (the next chain step's ``h``);
        ``logits_keep`` limits the norm+lm_head tail to the last N
        positions (0 = all).
        """
        if cache is None:
            cache = [None] * (2 * len(self.mtp))

        mask = create_attention_mask(h, cache[0], return_array=True)

        last_block = None
        for i, mtp_block in enumerate(self.mtp):
            layer_cache = cache[2 * i : 2 * i + 2]
            if layer_cache[0] is None:
                layer_cache = None
            h = mtp_block(
                h, self.model.embed_tokens, input_ids, mask, layer_cache
            )
            last_block = mtp_block

        logits_source = h
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:]
        logits = self.lm_head(last_block.norm(logits_source))
        if return_hidden:
            return logits, h
        return logits

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Trim the verify window back to ``accepted`` drafts on every layer.

        Every GLM cache is a CacheList of plain KVCaches (latent KV +
        indexer KV), so a straight trim covers any partial accept.
        """
        n = num_drafts - accepted
        if n <= 0:
            return True
        if not all(c.is_trimmable() for c in cache):
            return False
        for c in cache:
            if c.trim(n) != n:
                logger.warning(
                    "GLM MTP rollback trim shortfall on %s", type(c).__name__
                )
                return False
        return True

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Keep + remap the checkpoint's extra MTP layer.

        Raw-HF/FP8 checkpoints store the MTP layer as
        ``model.layers.<n_main + i>.*``. The stock sanitize strips layers
        >= ``num_hidden_layers`` and runs its per-layer transforms
        (FP8 dequant, expert stacking, fused gate_up, kv_b -> embed_q /
        unembed_out split, indexer fusion) over backbone indices only —
        so the layer count is temporarily bumped to cover the MTP layers,
        letting every transform apply to them identically, and the
        transformed keys are then renamed under ``mtp.<i>.*``.

        Pre-remapped checkpoints (oQ output) pass through the stock
        sanitize untouched apart from the load-time indexer fusion, which
        is replicated for the mtp prefix because the stock loop only
        walks ``model.layers.<l>``.
        """
        has_mtp = hasattr(self, "mtp")
        n_mtp = len(self.mtp) if has_mtp else 0
        n_main = int(self.args.num_hidden_layers)

        has_mtp_weights = any(
            k.startswith("mtp.") or k.startswith(f"model.layers.{n_main}.")
            for k in weights
        )
        if has_mtp and not has_mtp_weights:
            # Graceful fallback for checkpoints that stripped the head.
            try:
                del self.mtp
            except AttributeError:
                pass
            self._omlx_mtp_decode_enabled = False
            has_mtp = False
            n_mtp = 0

        if not has_mtp:
            # Stock sanitize drops raw nextn layers; stray pre-remapped
            # mtp.* keys (head-ful checkpoint loaded with MTP off) are
            # dropped here so the strict load doesn't trip on them.
            weights = {k: v for k, v in weights.items() if not k.startswith("mtp.")}
            return original_sanitize(self, weights)

        has_nextn_keys = any(
            any(k.startswith(f"model.layers.{n_main + i}.") for i in range(n_mtp))
            for k in weights
        )

        if has_nextn_keys:
            self.args.num_hidden_layers = n_main + n_mtp
            try:
                weights = original_sanitize(self, weights)
            finally:
                self.args.num_hidden_layers = n_main

            remapped: Dict[str, Any] = {}
            special = {
                "eh_proj.weight": "eh_proj.weight",
                "enorm.weight": "enorm.weight",
                "hnorm.weight": "hnorm.weight",
                "shared_head.norm.weight": "norm.weight",
            }
            for k, v in weights.items():
                nk = k
                for i in range(n_mtp):
                    prefix = f"model.layers.{n_main + i}."
                    if not k.startswith(prefix):
                        continue
                    rest = k[len(prefix):]
                    if rest.startswith(("shared_head.head.", "embed_tokens.")):
                        # Duplicates of the shared lm_head / embeddings some
                        # nextn checkpoints carry; the modules are shared.
                        nk = None
                    elif rest in special:
                        nk = f"mtp.{i}.{special[rest]}"
                    else:
                        nk = f"mtp.{i}.block.{rest}"
                    break
                if nk is not None:
                    remapped[nk] = v
            return remapped

        # Load-time indexer wk/weights_proj fusion for the mtp prefix
        # (mirrors the stock backbone loop; gate depends on the
        # checkpoint's quantization spec). Must run BEFORE the stock
        # sanitize: when the backbone loop fuses anything it rebuilds the
        # dict dropping every ``.self_attn.indexer.wk.`` /
        # ``.weights_proj.`` key by substring — which would strip the MTP
        # block's unfused keys too. The fused key name doesn't match
        # those substrings, so it passes through untouched.
        if _use_load_fused_wk_weights_proj(self.args):
            for i in range(n_mtp):
                prefix = f"mtp.{i}.block.self_attn.indexer"
                if f"{prefix}.wk.weight" not in weights:
                    continue
                for suffix in ("weight", "scales", "biases"):
                    wk_key = f"{prefix}.wk.{suffix}"
                    wp_key = f"{prefix}.weights_proj.{suffix}"
                    if wk_key in weights and wp_key in weights:
                        weights[f"{prefix}.wk_weights_proj.{suffix}"] = (
                            mx.concatenate(
                                [weights.pop(wk_key), weights.pop(wp_key)],
                                axis=0,
                            )
                        )
        return original_sanitize(self, weights)

    if not init_wrapped:
        cls.__init__ = __init__
        cls._omlx_mtp_init_wrapped = True
    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_partial_rollback = mtp_partial_rollback
    cls.sanitize = sanitize
