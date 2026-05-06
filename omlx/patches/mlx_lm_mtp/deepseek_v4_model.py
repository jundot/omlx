# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch for Blaizzy/mlx-lm#15 — DeepSeek-V4-Flash native MTP.

PR 15 is currently DRAFT. The shape mirrors PR 990 (Qwen3.5/3.6) but the
MTP head architecture is heavier: each ``MTPBlock`` wraps a full
``DeepseekV4Block`` plus per-block ``HyperHead`` and projection layers
(``e_proj``, ``h_proj``, ``enorm``, ``hnorm``, ``norm``).

oMLX already injects the DeepSeek-V4 base model itself via
``omlx/patches/deepseek_v4/`` (which lands the model class into
``sys.modules['mlx_lm.models.deepseek_v4']``); this patch sits on top and
adds the MTP head + ``mtp_forward`` / ``make_mtp_cache``. Apply order:
caller (``patches/mlx_lm_mtp/__init__.py``) runs ``apply()`` after the
base DeepSeek-V4 patch has registered the module.

The DeepSeek-V4 backbone has 4D hidden states (``B, S, hc_mult, hidden``)
because of the Hyper-head broadcasting. Both the patched ``__call__``
(with ``return_hidden=True``) and ``mtp_forward`` accept / produce 4D
tensors; the ``BatchGenerator`` MTP dispatch handles the dimension
difference via a small adapter (see ``batch_generator._slice_hidden``).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PATCHED = False


def apply() -> bool:
    """Apply PR 15 model-side patches when the DeepSeek-V4 base patch is active."""
    global _PATCHED
    if _PATCHED:
        return True

    dsv4 = sys.modules.get("mlx_lm.models.deepseek_v4")
    if dsv4 is None or not hasattr(dsv4, "Model"):
        # Base DeepSeek-V4 patch hasn't registered the module yet. This
        # branch only hits when MTP is enabled on a non-DeepSeek model —
        # log and skip cleanly.
        logger.debug(
            "DeepSeek-V4 module not registered; skipping MTP patch (this is "
            "expected for non-DeepSeek models)"
        )
        return False

    # Idempotency check.
    if hasattr(dsv4.Model, "_omlx_mtp_patched"):
        _PATCHED = True
        return True

    _patch_model_args(dsv4)
    _register_mtp_block(dsv4)
    _patch_deepseek_v4_model_call(dsv4)
    _patch_model(dsv4)

    _PATCHED = True
    dsv4.Model._omlx_mtp_patched = "patch"
    logger.info("DeepSeek-V4 MTP model patch applied (PR 15)")
    return True


# ---------------------------------------------------------------------------
# ModelArgs — extend compress_ratios to cover MTP layers.
# ---------------------------------------------------------------------------

def _patch_model_args(dsv4: Any) -> None:
    """Wrap ``ModelArgs.from_dict`` so MTP layers get a default compress_ratio.

    PR 15 widens the compress_ratios list from ``num_hidden_layers`` to
    ``num_hidden_layers + num_nextn_predict_layers`` (default 0, no
    compression). The original ``__post_init__`` raises if the list length
    doesn't match num_hidden_layers; we extend the list before that check
    by wrapping ``from_dict``.
    """
    args_cls = dsv4.ModelArgs
    if hasattr(args_cls, "_omlx_mtp_args_patched"):
        return

    original_from_dict = args_cls.from_dict.__func__

    def patched_from_dict(cls, params):
        # Pre-extend compress_ratios in the params dict so __post_init__
        # accepts the wider list. We're conservative: only act when MTP
        # layers are declared and the existing list is short by exactly
        # num_nextn_predict_layers.
        params = dict(params)
        n_main = int(params.get("num_hidden_layers", 0) or 0)
        n_mtp = int(params.get("num_nextn_predict_layers", 0) or 0)
        ratios = params.get("compress_ratios")
        if n_mtp > 0 and ratios is not None:
            ratios = list(ratios)
            target = n_main + n_mtp
            if len(ratios) == n_main:
                ratios = ratios + [0] * n_mtp
            params["compress_ratios"] = ratios[:target]
        return original_from_dict(cls, params)

    args_cls.from_dict = classmethod(patched_from_dict)
    args_cls._omlx_mtp_args_patched = True


# ---------------------------------------------------------------------------
# MTPBlock — register on the module.
# ---------------------------------------------------------------------------

def _register_mtp_block(dsv4: Any) -> None:
    """Define ``MTPBlock`` and attach it to the module."""
    if hasattr(dsv4, "MTPBlock"):
        return

    import mlx.core as mx
    import mlx.nn as nn

    DeepseekV4Block = dsv4.DeepseekV4Block
    HyperHead = dsv4.HyperHead

    class MTPBlock(nn.Module):
        """One MTP layer in DeepSeek-V4's stack.

        Fuses the previous-layer hidden ``h`` (4D, broadcast to
        ``hc_mult`` Hyper-head copies) with the embedding of the
        next-position token ``input_ids``, then runs a full
        ``DeepseekV4Block`` over the result. The block's own
        ``hc_head`` collapses Hyper-head copies back to ``hidden_size``
        before the shared lm_head produces logits.
        """

        def __init__(self, config, layer_idx: int):
            super().__init__()
            dim = config.hidden_size
            self.block = DeepseekV4Block(config, layer_idx)
            self.e_proj = nn.Linear(dim, dim, bias=False)
            self.h_proj = nn.Linear(dim, dim, bias=False)
            self.enorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.hnorm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.norm = nn.RMSNorm(dim, eps=config.rms_norm_eps)
            self.hc_head = HyperHead(config)

        def __call__(
            self,
            h,
            embed_tokens,
            input_ids,
            mask,
            cache,
        ):
            e = embed_tokens(input_ids)
            e = self.enorm(e)
            h_norm = self.hnorm(h)
            x = self.e_proj(e)[:, :, None, :] + self.h_proj(h_norm)
            x = mx.contiguous(x)
            x = self.block(x, mask, cache, input_ids)
            return x

    dsv4.MTPBlock = MTPBlock


# ---------------------------------------------------------------------------
# DeepseekV4Model — return_raw_hidden support.
# ---------------------------------------------------------------------------

def _patch_deepseek_v4_model_call(dsv4: Any) -> None:
    """Replace ``DeepseekV4Model.__call__`` to optionally return the raw 4D hidden."""
    cls = dsv4.DeepseekV4Model
    if hasattr(cls, "_omlx_mtp_patched"):
        return

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    CacheList = dsv4.CacheList

    def __call__(
        self,
        inputs,
        cache=None,
        return_raw_hidden: bool = False,
    ):
        h = self.embed_tokens(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.args.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)

        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList) else first_cache
        )
        mask = create_attention_mask(
            h[:, :, 0, :],
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            h = layer(h, mask, layer_cache, inputs)

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            cache_item = cache[-1]
            if isinstance(cache_item, CacheList):
                cache_item = cache_item[0]
            if cache_item is not None:
                cache_item.keys = mx.depends(cache_item.keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        out = self.norm(self.hc_head(h))
        if return_raw_hidden:
            return out, h
        return out

    cls.__call__ = __call__
    cls._omlx_mtp_patched = True


# ---------------------------------------------------------------------------
# Model — wrap __init__, replace __call__, add mtp_forward / make_mtp_cache,
# replace sanitize with the PR 15 body that handles MTP weight remapping.
# ---------------------------------------------------------------------------

def _patch_model(dsv4: Any) -> None:
    cls = dsv4.Model
    if hasattr(cls, "_omlx_mtp_patched"):
        return

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    # oMLX's DeepSeek-V4 fork uses a ``CacheList`` of (RotatingKVCache,
    # PoolingCache, [PoolingCache]) instead of upstream's
    # ``DeepseekV4Cache`` wrapper. The make_mtp_cache + mtp_forward bodies
    # below use the oMLX-side cache layout to stay compatible with the
    # already-patched DeepseekV4Model.__call__ above.
    CacheList = dsv4.CacheList
    PoolingCache = dsv4.PoolingCache
    RotatingKVCache = dsv4.RotatingKVCache
    SparseCompressedAttention = getattr(dsv4, "SparseCompressedAttention", None)

    original_init = cls.__init__

    def __init__(self, config):
        original_init(self, config)
        n_mtp = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
        # See qwen35_model._patch_model: gated on the MTP active-flag so
        # mtp_enabled=False produces a model indistinguishable from stock.
        from . import is_mtp_active

        if n_mtp > 0 and is_mtp_active():
            n_main = config.num_hidden_layers
            self.mtp = [dsv4.MTPBlock(config, n_main + i) for i in range(n_mtp)]

    def __call__(
        self,
        inputs,
        cache=None,
        return_hidden: bool = False,
    ):
        if return_hidden:
            h, h_raw = self.model(inputs, cache, return_raw_hidden=True)
            return self.lm_head(h), h_raw
        h = self.model(inputs, cache)
        return self.lm_head(h)

    def make_mtp_cache(self):
        """Build per-MTP-block caches. Mirrors ``Model.make_cache`` but for the
        MTP stack. PR 15's MTP layers default to ``compress_ratio=0`` so the
        common case is a plain RotatingKVCache, but we honor the same
        SparseCompressedAttention / CacheList layout as the backbone for any
        config that ever assigns a non-zero compress_ratio to MTP layers.
        """
        if not hasattr(self, "mtp"):
            return None
        caches = []
        sw = self.args.sliding_window
        for mtp_block in self.mtp:
            attn = mtp_block.block.attn
            ratio = getattr(attn, "compress_ratio", 0)
            if ratio == 0:
                caches.append(RotatingKVCache(max_size=sw))
            elif (
                SparseCompressedAttention is not None
                and isinstance(attn, SparseCompressedAttention)
            ):
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=sw),
                        PoolingCache(ratio),
                        PoolingCache(ratio),
                    )
                )
            else:
                caches.append(
                    CacheList(
                        RotatingKVCache(max_size=sw),
                        PoolingCache(ratio),
                    )
                )
        return caches

    def mtp_forward(self, h, input_ids, cache=None):
        """Run the chained MTP blocks + final hc_head/norm/lm_head on a 4D hidden.

        Mirrors PR 15: each MTP block fuses ``h`` with the embedded
        ``input_ids`` through its ``e_proj``/``h_proj`` projection and
        passes the result through a ``DeepseekV4Block``. The last block's
        ``hc_head`` collapses the Hyper-head dimension before ``norm`` and
        the shared ``lm_head`` produce logits.
        """
        if cache is None:
            cache = [None] * len(self.mtp)

        first_cache = cache[0]
        mask_cache = (
            first_cache[0] if isinstance(first_cache, CacheList) else first_cache
        )
        mask_input = h[:, :, 0, :] if h.ndim == 4 else h
        mask = create_attention_mask(
            mask_input,
            mask_cache,
            window_size=self.args.sliding_window,
            return_array=True,
        )

        last_block = None
        for mtp_block, layer_cache in zip(self.mtp, cache):
            h = mtp_block(
                h, self.model.embed_tokens, input_ids, mask, layer_cache
            )
            last_block = mtp_block

        out = last_block.hc_head(h)
        out = last_block.norm(out)
        return self.lm_head(out)

    def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Combined oMLX-base + PR 15 sanitize.

        oMLX's stock sanitize strips ``mtp.*`` and remaps the FP4 expert
        weights / Hyper-head names. PR 15 keeps ``mtp.*`` when an MTP head
        is present, nests block-internal weights under ``.block.``, and
        stacks routed expert weights for MTP layers as well as backbone
        layers.
        """
        n_layers = self.args.num_hidden_layers
        has_mtp = hasattr(self, "mtp")
        has_mtp_weights = any(k.startswith("mtp.") for k in weights)
        # Disable MTP module if weights are absent (e.g. quantized checkpoints
        # that stripped them). Mirrors PR 15's graceful fallback.
        if has_mtp and not has_mtp_weights:
            try:
                del self.mtp
            except AttributeError:
                pass
            has_mtp = False

        new_weights: Dict[str, Any] = {}
        for k, v in weights.items():
            if k.startswith("mtp."):
                if not has_mtp:
                    continue
                new_weights[k] = v
                continue
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                try:
                    if int(parts[1]) >= n_layers:
                        continue
                except ValueError:
                    pass
            new_weights[k] = v
        weights = new_weights

        # FP4 dequant pre-pass (oMLX-specific). Identical to the
        # un-patched body — safe to keep as-is.
        new_weights = {}
        for k, v in weights.items():
            if "tid2eid" in k:
                new_weights[k] = v.astype(mx.int32)

            if not k.endswith(".scale"):
                if k not in new_weights:
                    new_weights[k] = v
                continue

            wk = k[: -len(".scale")] + ".weight"
            weight = weights.get(wk)
            if weight is None:
                new_weights[k] = v
                continue
            if (
                ".ffn.experts." in wk
                and ".shared_experts." not in wk
                and weight.dtype in (mx.int8, mx.uint8)
                and v.shape[-1] * 16 == weight.shape[-1]
            ):
                new_weights[k + "s"] = v
                new_weights[wk] = weight.view(mx.uint32)
            elif weight.dtype == mx.uint8:
                new_weights[k + "s"] = mx.repeat(mx.repeat(v, 4, -1), 128, 0)
                new_weights[wk] = weight.view(mx.uint32)
            else:
                new_weights[k] = v
        weights = new_weights

        top_remap = {
            "embed.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "hc_head_fn": "model.hc_head.fn",
            "hc_head_base": "model.hc_head.base",
            "hc_head_scale": "model.hc_head.scale",
        }
        for old, new in top_remap.items():
            if old in weights:
                weights[new] = weights.pop(old)

        # Block-internal weight key remapping. Adds PR 15's ``mtp.*`` →
        # ``mtp.<idx>.block.*`` nesting on top of the existing remap.
        remapped = {}
        w_remap = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        mtp_block_subs = (
            "attn.", "ffn.", "attn_norm.", "ffn_norm.",
            "hc_attn_", "hc_ffn_",
        )
        for k, v in weights.items():
            nk = "model." + k if k.startswith("layers.") else k
            # MTP block: nest block-internal weights under .block.
            if nk.startswith("mtp."):
                parts = nk.split(".", 2)  # ["mtp", "<idx>", "<rest>"]
                if len(parts) == 3:
                    rest = parts[2]
                    if any(rest.startswith(s) for s in mtp_block_subs):
                        nk = f"mtp.{parts[1]}.block.{rest}"
                    for param in ("fn", "base", "scale"):
                        if rest == f"hc_head_{param}":
                            nk = f"mtp.{parts[1]}.hc_head.{param}"
            nk = nk.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for sub in ("attn", "ffn"):
                for param in ("fn", "base", "scale"):
                    nk = nk.replace(f".hc_{sub}_{param}", f".{sub}_hc.{param}")
            for old, new in w_remap.items():
                nk = nk.replace(f".shared_experts.{old}.", f".shared_experts.{new}.")
            remapped[nk] = v
        weights = remapped

        # Stack routed expert weights for backbone layers.
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.ffn.experts"
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                for suffix in ("weight", "scales"):
                    key0 = f"{prefix}.0.{src}.{suffix}"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[
                            f"model.layers.{layer_idx}.ffn.switch_mlp.{dst}.{suffix}"
                        ] = mx.stack(stacked)

        # Reshape wo_a from nn.Linear (2D) to MultiLinear (3D) for all layers.
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}.attn.wo_a"
            for key in (f"{prefix}.weight", f"{prefix}.scales", f"{prefix}.biases"):
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key].reshape(
                        self.args.o_groups, self.args.o_lora_rank, -1
                    )

        # Stack routed expert weights for MTP layers (PR 15).
        if has_mtp:
            for mtp_idx in range(self.args.num_nextn_predict_layers):
                prefix = f"mtp.{mtp_idx}.block.ffn.experts"
                for src, dst in (
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ):
                    for suffix in ("weight", "scales"):
                        key0 = f"{prefix}.0.{src}.{suffix}"
                        if key0 in weights:
                            stacked = [
                                weights.pop(f"{prefix}.{e}.{src}.{suffix}")
                                for e in range(self.args.n_routed_experts)
                            ]
                            weights[
                                f"mtp.{mtp_idx}.block.ffn.switch_mlp.{dst}.{suffix}"
                            ] = mx.stack(stacked)

        return weights

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.sanitize = sanitize
    cls._omlx_mtp_patched = True
