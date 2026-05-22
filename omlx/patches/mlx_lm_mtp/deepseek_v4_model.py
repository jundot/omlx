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
    if "_omlx_mtp_patched" in dsv4.Model.__dict__:
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
    if "_omlx_mtp_args_patched" in args_cls.__dict__:
        return

    original_from_dict = args_cls.from_dict.__func__

    def patched_from_dict(cls, params):
        # Build args via the base ``from_dict`` (which runs ``__post_init__``
        # and may truncate ``compress_ratios`` back to ``num_hidden_layers``).
        # Then re-extend the ratio list to cover MTP layers so MTPBlock's
        # ``DeepseekV4Block(..., layer_idx=n_main+i)`` lookup succeeds.
        args = original_from_dict(cls, params)
        n_main = int(getattr(args, "num_hidden_layers", 0) or 0)
        n_mtp = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
        if n_mtp > 0 and hasattr(args, "compress_ratios"):
            ratios = list(args.compress_ratios)
            if len(ratios) < n_main + n_mtp:
                ratios = ratios + [0] * (n_main + n_mtp - len(ratios))
            args.compress_ratios = ratios
        return args

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
    if "_omlx_mtp_patched" in cls.__dict__:
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
    if "_omlx_mtp_patched" in cls.__dict__:
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
            "embed.scales": "model.embed_tokens.scales",
            "embed.biases": "model.embed_tokens.biases",
            "norm.weight": "model.norm.weight",
            "head.weight": "lm_head.weight",
            "head.scales": "lm_head.scales",
            "head.biases": "lm_head.biases",
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
            nk = nk.replace(".mlp.switch_mlp.", ".ffn.switch_mlp.")
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

        # Native JAnq runtime integration
        is_jangtq = any(".tq_packed" in k for k in weights.keys())
        if is_jangtq:
            import sys
            import logging
            logger = logging.getLogger("mlx_lm")
            if "/Users/true/Library/Python/3.14/lib/python/site-packages" not in sys.path:
                sys.path.append("/Users/true/Library/Python/3.14/lib/python/site-packages")
            
            try:
                from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear
                from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort
                from jang_tools.turboquant.fused_gate_up_kernel import (
                    fused_gate_up_swiglu_matmul,
                    make_fused_gate_up_swiglu_decode,
                )
                from jang_tools.turboquant.gather_tq_kernel import make_gather_tq_decode_per_row
                from jang_tools.turboquant.hadamard_kernel import hadamard_rotate_metal
            except ImportError as e:
                msg = f"Failed to import JAnq runtime modules: {e}. Please ensure jang_tools is installed."
                logger.error(msg)
                print(msg)
                raise

            import json
            from pathlib import Path
            jang_cfg_path = Path(getattr(self, "model_path", "")) / "jang_config.json"
            if jang_cfg_path.exists():
                with open(jang_cfg_path) as f:
                    jang_cfg = json.load(f)
            else:
                jang_cfg = {}
            mxtq_seed = jang_cfg.get("mxtq_seed", 42)
            
            msg = f"JANGTQ model detected. Hydrating with native JAnq/jang_tools runtime (seed={mxtq_seed})...."
            logger.info(msg)
            print(msg)

            for layer_idx in range(n_layers):
                layer = self.layers[layer_idx]
                for prefix in (f"model.layers.{layer_idx}.ffn.switch_mlp", f"layers.{layer_idx}.ffn.switch_mlp"):
                    for dst in ("gate_proj", "up_proj", "down_proj"):
                        tq_packed_key = f"{prefix}.{dst}.tq_packed"
                        
                        pk_key = tq_packed_key if tq_packed_key in weights else (f"model.{tq_packed_key}" if f"model.{tq_packed_key}" in weights else None)
                        if pk_key is not None:
                            nm_key = pk_key.replace(".tq_packed", ".tq_norms")
                            bt_key = pk_key.replace(".tq_packed", ".tq_bits")
                            
                            packed = weights.pop(pk_key)
                            norms = weights.pop(nm_key)
                            bits_arr = weights.pop(bt_key)
                            bits = int(bits_arr.item())
                            
                            num_experts, out_features, packed_cols = packed.shape
                            vals_per_u32 = 32 // bits
                            in_features = packed_cols * vals_per_u32
                            
                            new_module = TurboQuantSwitchLinear(
                                in_features=in_features,
                                out_features=out_features,
                                num_experts=num_experts,
                                bits=bits,
                                bias=False,
                                seed=mxtq_seed,
                            )
                            new_module.packed = packed
                            new_module.norms = norms
                            
                            setattr(layer.ffn.switch_mlp, dst, new_module)
                            
                            # Expose via standard parameter names for model.load_weights
                            target_pk_key = pk_key.replace(".tq_packed", ".packed")
                            target_nm_key = nm_key.replace(".tq_norms", ".norms")
                            weights[target_pk_key] = packed
                            weights[target_nm_key] = norms

            if not hasattr(SwitchGLU, "_orig_call"):
                SwitchGLU._orig_call = SwitchGLU.__call__

            _decode_compiled = {}

            def _get_compiled_decode(in_f, out_f, bits, k, swiglu_limit=0.0, dp_bits=None):
                if dp_bits is None:
                    dp_bits = bits
                limit_milli = int(round(float(swiglu_limit or 0.0) * 1000.0))
                cache_key = (in_f, out_f, bits, dp_bits, k, limit_milli)
                if cache_key in _decode_compiled:
                    return _decode_compiled[cache_key]
                fused_gu = make_fused_gate_up_swiglu_decode(
                    in_f, out_f, bits, k, swiglu_limit=swiglu_limit
                )
                gather_dn = make_gather_tq_decode_per_row(out_f, in_f, dp_bits, k)

                def _mlp(x_flat, pg, ng, pu, nu, pd, nd, cb_gate, cb_down, signs_in, signs_dn, idx_flat):
                    x_rot = hadamard_rotate_metal(x_flat, signs_in)
                    x_act = fused_gu(x_rot, pg, ng, pu, nu, cb_gate, idx_flat)
                    x_act_rot = hadamard_rotate_metal(x_act, signs_dn)
                    return gather_dn(x_act_rot, pd, nd, cb_down, idx_flat)

                _decode_compiled[cache_key] = mx.compile(_mlp)
                return _decode_compiled[cache_key]

            def _dsv4_fused_switchglu_call(self_sg, x, indices):
                gp = self_sg.gate_proj
                up = self_sg.up_proj
                dp = self_sg.down_proj
                if not isinstance(gp, TurboQuantSwitchLinear) or not isinstance(up, TurboQuantSwitchLinear):
                    return SwitchGLU._orig_call(self_sg, x, indices)
                activation = getattr(self_sg, "activation", None)
                swiglu_limit = getattr(activation, "swiglu_limit", 0.0) or 0.0
                x_sq = x
                while x_sq.ndim > 2 and x_sq.shape[-2] == 1:
                    x_sq = x_sq.squeeze(-2)
                x_flat = x_sq.reshape(-1, gp.in_features)
                batch = x_flat.shape[0]
                k = indices.shape[-1] if indices.ndim > 0 else 1
                can_fast = batch == 1 and k > 0 and indices.ndim >= 1 and indices.size < 64
                if can_fast and not getattr(self_sg, "training", False):
                    idx_flat = indices.reshape(-1).astype(mx.uint32)
                    compiled_mlp = _get_compiled_decode(
                        gp.in_features, gp.out_features, gp.bits, k, swiglu_limit,
                        dp_bits=dp.bits,
                    )
                    y = compiled_mlp(
                        x_flat.astype(mx.float32),
                        gp.packed, gp.norms, up.packed, up.norms,
                        dp.packed, dp.norms,
                        gp.codebook, dp.codebook,
                        gp.signs, dp.signs, idx_flat,
                    )
                    out = y.reshape(*indices.shape[:-1], k, 1, gp.in_features)
                    if out.dtype != x.dtype:
                        out = out.astype(x.dtype)
                    return out.squeeze(-2)

                x_exp = mx.expand_dims(x, (-2, -3))
                do_sort = indices.size >= 64
                idx = indices
                inv_order = None
                if do_sort:
                    x_exp, idx, inv_order = _gather_sort(x_exp, indices)
                if getattr(self_sg, "training", False):
                    idx = mx.stop_gradient(idx)
                x_act = fused_gate_up_swiglu_matmul(
                    x_exp,
                    gp.packed, gp.norms,
                    up.packed, up.norms,
                    gp.codebook, gp.signs,
                    idx,
                    bits=gp.bits,
                    swiglu_limit=swiglu_limit,
                )
                x_out = dp(x_act, idx, sorted_indices=do_sort)
                if do_sort:
                    x_out = _scatter_unsort(x_out, inv_order, indices.shape)
                return x_out.squeeze(-2)

            SwitchGLU.__call__ = _dsv4_fused_switchglu_call
            
            import gc
            gc.collect()
            mx.metal.clear_cache()

        return weights

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.sanitize = sanitize
    cls._omlx_mtp_patched = True
