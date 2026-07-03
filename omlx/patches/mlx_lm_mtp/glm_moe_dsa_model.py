# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 (glm_moe_dsa) native MTP head for the mlx_lm_mtp draft/verify cycle.

GLM-5.2's MTP layer is DeepSeek-V3-style: at position ``i`` it fuses the
backbone's pre-norm hidden state (``hnorm``) with the embedding of token
``i+1`` (``enorm``) through a concat + ``eh_proj``, then runs one full
``GlmMoeDsaDecoderLayer`` (full DSA indexer + MoE) and shares the backbone's
``lm_head``. Zhipu explicitly designed it as the draft model for speculative
decoding.

Unlike Qwen3.5 / DeepSeek-V4-Flash, common quantized GLM-5.2 checkpoints
(e.g. mxfp4) strip the MTP tensors, so the head ships as a *separate*
checkpoint (``model_type: glm_moe_dsa_mtp``, ~11 GB Q4). This patch therefore
does NOT hook ``Model.__init__``/``sanitize`` — the head is attached
post-load from a sibling ``<model_dir>/mtp/`` directory (see
``maybe_attach_glm_mtp``), which keeps the strict main-checkpoint load
untouched and the model byte-identical to stock when MTP is off.

The BatchGenerator MTP dispatch is model-agnostic: it only needs
``model(inputs, cache, return_hidden=True)`` -> ``(logits, hidden)``,
``model.mtp_forward(hidden, next_ids, mtp_cache)``, ``model.make_mtp_cache()``
and the ``_omlx_mtp_decode_enabled`` instance marker — all provided here.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Stash set at pre-load time (maybe_apply_pre_load_patches) and consumed at
# post-load time (apply_post_load_transforms). Same single-MLX-executor
# serialization argument as ``mlx_lm_mtp._MTP_ACTIVE``.
_GLM_MTP_PATH: Optional[str] = None


def set_glm_mtp_path(path: Optional[str]) -> None:
    global _GLM_MTP_PATH
    _GLM_MTP_PATH = path


def find_glm_mtp_checkpoint(model_path: str) -> Optional[str]:
    """Return the MTP head checkpoint dir for a GLM model, or None.

    Convention: a ``mtp/`` subdirectory (or symlink) inside the model
    directory holding ``config.json`` (``model_type: glm_moe_dsa_mtp``) +
    ``*.safetensors``.
    """
    cand = Path(model_path) / "mtp"
    if (cand / "config.json").exists():
        return str(cand)
    return None


def apply() -> bool:
    """Patch the (oMLX-vendored) ``mlx_lm.models.glm_moe_dsa`` module.

    Must run after ``apply_glm_moe_dsa_patch`` registered the module and
    before ``mlx_lm.load()``. Idempotent via class markers.
    """
    glm = sys.modules.get("mlx_lm.models.glm_moe_dsa")
    if glm is None or not hasattr(glm, "Model"):
        logger.debug(
            "glm_moe_dsa module not registered; skipping GLM MTP patch "
            "(expected for non-GLM models)"
        )
        return False

    _register_mtp_module(glm)
    _patch_backbone_call(glm)
    _patch_model(glm)

    if not getattr(glm.Model, "_omlx_glm_mtp_patched", False):
        glm.Model._omlx_glm_mtp_patched = True
        logger.info("GLM-5.2 MTP model patch applied (external head)")
    return True


def _cache_classes(glm: Any):
    CacheList = getattr(glm, "CacheList", None)
    KVCache = getattr(glm, "KVCache", None)
    if CacheList is None or KVCache is None:
        from mlx_lm.models.cache import CacheList, KVCache  # type: ignore
    return CacheList, KVCache


# ---------------------------------------------------------------------------
# GlmMoeDsaMTP — the head module (external checkpoint).
# ---------------------------------------------------------------------------


def _register_mtp_module(glm: Any) -> None:
    if hasattr(glm, "GlmMoeDsaMTP"):
        return

    import mlx.nn as nn

    class GlmMoeDsaMTP(nn.Module):
        """GLM-5.2 MTP head: enorm/hnorm -> concat -> eh_proj -> one
        ``GlmMoeDsaDecoderLayer`` (full indexer) -> norm. Shares the
        backbone's ``embed_tokens`` and ``lm_head`` at call time."""

        def __init__(self, config: Any, layer_idx: int):
            super().__init__()
            dim = config.hidden_size
            self.enorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.hnorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.eh_proj = nn.Linear(dim * 2, dim, bias=False)
            self.layers = [glm.GlmMoeDsaDecoderLayer(config, layer_idx)]
            self.norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)

    glm.GlmMoeDsaMTP = GlmMoeDsaMTP


# ---------------------------------------------------------------------------
# Backbone — optional pre-norm hidden return.
# ---------------------------------------------------------------------------


def _patch_backbone_call(glm: Any) -> None:
    cls = glm.GlmMoeDsaModel
    existing = cls.__dict__.get("__call__")
    if getattr(existing, "_omlx_glm_mtp_marker", False):
        return

    import mlx.core as mx

    create_attention_mask = glm.create_attention_mask
    original_call = cls.__call__

    def __call__(
        self,
        x,
        cache=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        # ``n_confirmed`` is accepted for the patched-backbone interface and
        # unused: GLM keeps all decode state in KV caches, rejection rolls
        # back via cache_rollback (same rationale as DeepSeek-V4).
        if not return_hidden:
            return original_call(self, x, cache)

        # Mirror of the vendored body, returning the pre-norm hidden too.
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

        return self.norm(h), h

    __call__._omlx_glm_mtp_marker = True
    cls.__call__ = __call__


# ---------------------------------------------------------------------------
# Model — return_hidden passthrough + mtp_forward / make_mtp_cache / attach.
# ---------------------------------------------------------------------------


def _patch_model(glm: Any) -> None:
    cls = glm.Model
    existing = cls.__dict__.get("__call__")
    if getattr(existing, "_omlx_glm_mtp_marker", False):
        return

    import mlx.core as mx
    import mlx.nn as nn

    create_attention_mask = glm.create_attention_mask
    CacheList, KVCache = _cache_classes(glm)
    original_call = cls.__call__

    def __call__(
        self,
        inputs,
        cache=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
        **kwargs,
    ):
        if not return_hidden:
            return original_call(self, inputs, cache, **kwargs)
        normed, h = self.model(inputs, cache, return_hidden=True)
        return self.lm_head(normed), h

    def make_mtp_cache(self):
        # The MTP layer runs a full indexer -> CacheList(attn KV, indexer KV).
        if getattr(self, "mtp", None) is None:
            return None
        return [CacheList(KVCache(), KVCache())]

    def mtp_forward(self, hidden, next_ids, mtp_cache):
        e = self.model.embed_tokens(next_ids)
        fused = self.mtp.eh_proj(
            mx.concatenate([self.mtp.enorm(e), self.mtp.hnorm(hidden)], axis=-1)
        )
        c = mtp_cache[0]
        mask = create_attention_mask(fused, c[0] if c[0] else None, return_array=True)
        x, prev = fused, None
        for layer in self.mtp.layers:
            x, prev = layer(x, mask, c, prev)
        return self.lm_head(self.mtp.norm(x))

    def attach_glm_mtp(self, mtp_path: str):
        """Build + load the MTP head from its separate checkpoint and mark
        the instance MTP-decode-enabled for the BatchGenerator dispatch."""
        import glob
        import json

        cfgf = json.load(open(Path(mtp_path) / "config.json"))
        tcfg = cfgf.get("text_config", cfgf)
        qcfg = cfgf.get("quantization_config") or tcfg.get("quantization") or {}
        margs = glm.ModelArgs.from_dict(tcfg)
        layer_idx = next(
            i
            for i in range(margs.num_hidden_layers)
            if i >= margs.first_k_dense_replace and margs.indexer_types[i] == "full"
        )
        head = glm.GlmMoeDsaMTP(margs, layer_idx)
        weights = {}
        for f in glob.glob(str(Path(mtp_path) / "*.safetensors")):
            weights.update(mx.load(f))
        # The vendored GLM MoE fuses gate_proj+up_proj into gate_up_proj
        # (see deepseek_v32.sanitize); the external head checkpoint stores
        # them split — replicate the fusion here (concat on the expert
        # output axis, forward splits in half on axis=-1).
        fuse_prefixes = sorted(
            {
                k.rsplit(".gate_proj.", 1)[0]
                for k in weights
                if ".switch_mlp.gate_proj." in k
            }
        )
        for prefix in fuse_prefixes:
            for k in ("weight", "scales", "biases"):
                g = f"{prefix}.gate_proj.{k}"
                u = f"{prefix}.up_proj.{k}"
                if g in weights and u in weights:
                    weights[f"{prefix}.gate_up_proj.{k}"] = mx.concatenate(
                        [weights.pop(g), weights.pop(u)], axis=1
                    )
        nn.quantize(
            head,
            group_size=qcfg.get("group_size", 64),
            bits=qcfg.get("bits", 4),
            class_predicate=lambda p, m: (p + ".scales") in weights,
        )
        head.load_weights(list(weights.items()), strict=True)
        mx.eval(head.parameters())
        self.mtp = head
        self._omlx_mtp_decode_enabled = True
        self._omlx_mtp_aligned_cache = True
        logger.info(
            "GLM MTP head attached from %s (layer_idx=%d, %d tensors)",
            mtp_path,
            layer_idx,
            len(weights),
        )
        return self

    __call__._omlx_glm_mtp_marker = True
    cls.__call__ = __call__
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_forward = mtp_forward
    cls.attach_glm_mtp = attach_glm_mtp


# ---------------------------------------------------------------------------
# Post-load attach entry point (called from apply_post_load_transforms).
# ---------------------------------------------------------------------------


def maybe_attach_glm_mtp(model: Any, model_settings: Any = None) -> Any:
    """Attach the external GLM MTP head when armed at pre-load time.

    No-op unless ``set_glm_mtp_path`` stashed a checkpoint path for this
    load (which already encodes the model_type + settings gate).
    """
    global _GLM_MTP_PATH
    path = _GLM_MTP_PATH
    _GLM_MTP_PATH = None
    if not path:
        return model
    if not hasattr(model, "attach_glm_mtp"):
        logger.warning(
            "GLM MTP path stashed but model has no attach_glm_mtp "
            "(patch not applied?) — skipping"
        )
        return model
    try:
        model.attach_glm_mtp(path)
    except Exception:
        logger.exception("GLM MTP head attach failed — continuing without MTP")
    return model
