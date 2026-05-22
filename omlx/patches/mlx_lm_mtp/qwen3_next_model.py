# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch for mlx-lm ``qwen3_next`` native MTP (Qwen3-Coder-Next).

Follows the same pattern as ``qwen35_model.py`` (PR 990) but adapted for
``mlx_lm.models.qwen3_next`` where class names use the ``Qwen3Next`` prefix
and the GatedDeltaNet uses combined ``in_proj_qkvz`` / ``in_proj_ba``
projections instead of separate ``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_b`` /
``in_proj_a``.  The ``Model`` class in ``qwen3_next`` is a single unified
outer model (no VLM wrapper), so no separate ``_patch_outer_model`` is needed.

What this patch installs (all on classes from ``mlx_lm.models.qwen3_next``):

- ``ModelArgs``: runtime ``mtp_num_hidden_layers`` instance attribute via a
  thin ``from_dict`` wrapper.
- ``Qwen3NextGatedDeltaNet``: ``_process_chunk`` helper + ``__call__``
  replacement that supports the ``n_confirmed`` argument (splits forward into
  confirmed prefix and draft suffix with SSM/conv snapshots for rollback).
- ``Qwen3NextDecoderLayer``: passes ``n_confirmed`` through to the linear-attn
  sublayer.
- ``Qwen3NextModel``: returns *pre-norm* hidden states (MTP head needs them).
- ``Model``: ``__init__`` wraps to attach a fresh ``MTPModule`` when the
  args declared one.  ``__call__`` accepts ``return_hidden`` and
  ``n_confirmed``.  ``mtp_forward`` and ``make_mtp_cache`` added.  ``sanitize``
  keeps the ``mtp.*`` keys when an MTP head exists (and handles MoE expert
  stacking).  ``quant_predicate`` keeps the fusion projection in full precision.
- ``MTPDecoderLayer`` and ``MTPModule`` registered on the module.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PATCHED = False


def apply() -> bool:
    """Apply MTP model-side patches to ``mlx_lm.models.qwen3_next``. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from mlx_lm.models import qwen3_next as q3n
    except ImportError:
        logger.debug("mlx_lm.models.qwen3_next not importable; skipping MTP patch")
        return False

    # Skip if upstream already merged MTP support.
    if hasattr(q3n.Model, "mtp_forward") and not hasattr(
        q3n.Model, "_omlx_mtp_patched"
    ):
        _PATCHED = True
        q3n.Model._omlx_mtp_patched = "upstream"
        return True

    _patch_model_args(q3n)
    _register_mtp_classes(q3n)
    _patch_gated_delta_net(q3n)
    _patch_decoder_layer(q3n)
    _patch_qwen3_next_model(q3n)
    _patch_outer_model(q3n)

    _PATCHED = True
    q3n.Model._omlx_mtp_patched = "patch"
    logger.info("Qwen3-Next MTP model patch applied")
    return True


# ---------------------------------------------------------------------------
# ModelArgs.from_dict — surface mtp_num_hidden_layers as instance attr.
# ---------------------------------------------------------------------------

def _patch_model_args(q3n) -> None:
    args_cls = q3n.ModelArgs
    if hasattr(args_cls, "_omlx_mtp_from_dict_patched"):
        return

    original_from_dict = args_cls.from_dict.__func__

    def patched_from_dict(cls, params):
        instance = original_from_dict(cls, params)
        instance.mtp_num_hidden_layers = int(
            params.get("mtp_num_hidden_layers", 0) or 0
        )
        return instance

    args_cls.from_dict = classmethod(patched_from_dict)
    args_cls._omlx_mtp_from_dict_patched = True


# ---------------------------------------------------------------------------
# MTPDecoderLayer + MTPModule — register on the qwen3_next module.
# ---------------------------------------------------------------------------

def _register_mtp_classes(q3n) -> None:
    if hasattr(q3n, "MTPModule"):
        return

    import mlx.core as mx
    import mlx.nn as nn

    Attention = q3n.Qwen3NextAttention
    SparseMoeBlock = q3n.Qwen3NextSparseMoeBlock
    MLP = q3n.Qwen3NextMLP
    create_attention_mask = q3n.create_attention_mask

    class MTPDecoderLayer(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.self_attn = Attention(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            if args.num_experts > 0:
                self.mlp = SparseMoeBlock(args)
            else:
                self.mlp = MLP(args.hidden_size, args.intermediate_size)

        def __call__(self, x, mask=None, cache=None):
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
            return h + self.mlp(self.post_attention_layernorm(h))

    class MTPModule(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [
                MTPDecoderLayer(args) for _ in range(args.mtp_num_hidden_layers)
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
            embeds = embed_tokens(next_token_ids)
            e = self.pre_fc_norm_embedding(embeds)
            h = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([e, h], axis=-1))

            if cache is None:
                cache = [None] * len(self.layers)

            mask = create_attention_mask(fused, cache[0] if cache else None)
            for layer, c in zip(self.layers, cache):
                fused = layer(fused, mask, c)

            return self.norm(fused)

    q3n.MTPDecoderLayer = MTPDecoderLayer
    q3n.MTPModule = MTPModule


# ---------------------------------------------------------------------------
# Qwen3NextGatedDeltaNet — _process_chunk helper + __call__ replacement.
# ---------------------------------------------------------------------------

def _patch_gated_delta_net(q3n) -> None:
    cls = q3n.Qwen3NextGatedDeltaNet
    if "_omlx_mtp_patched" in cls.__dict__:
        return

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import gated_delta_update

    def _process_chunk(
        self,
        qkv_chunk,
        a_chunk,
        b_chunk,
        conv_state,
        ssm_state,
        ssm_mask=None,
        lengths=None,
    ):
        B, S_chunk = qkv_chunk.shape[:2]
        conv_in = mx.concatenate([conv_state, qkv_chunk], axis=1)
        n_keep = self.conv_kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, S_chunk)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_conv_state = mx.take_along_axis(conv_in, positions, axis=1)
        else:
            new_conv_state = mx.contiguous(conv_in[:, -n_keep:])
        conv_out = nn.silu(self.conv1d(conv_in))

        q, k, v = [
            t.reshape(B, S_chunk, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, new_ssm_state = gated_delta_update(
            q,
            k,
            v,
            a_chunk,
            b_chunk,
            self.A_log,
            self.dt_bias,
            ssm_state,
            ssm_mask,
            use_kernel=not self.training,
        )
        return out, new_conv_state, new_ssm_state

    def __call__(
        self,
        inputs,
        mask=None,
        cache=None,
        n_confirmed: int = 0,
    ):
        B, S, _ = inputs.shape

        q, k, v, z, b, a = self.fix_query_key_value_ordering(
            self.in_proj_qkvz(inputs), self.in_proj_ba(inputs)
        )
        qkv = mx.concatenate(
            [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1
        )

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        ssm_state = cache[1] if cache else None

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)

        if n_confirmed > 0 and n_confirmed < S:
            mask_c = mask[:, :n_confirmed] if mask is not None else None
            mask_d = mask[:, n_confirmed:] if mask is not None else None
            out_c, conv_c, ssm_c = self._process_chunk(
                qkv[:, :n_confirmed],
                a[:, :n_confirmed],
                b[:, :n_confirmed],
                conv_state,
                ssm_state,
                mask_c,
            )
            if cache is not None:
                cache.rollback_state = (conv_c, ssm_c)
            out_d, conv_f, ssm_f = self._process_chunk(
                qkv[:, n_confirmed:],
                a[:, n_confirmed:],
                b[:, n_confirmed:],
                conv_c,
                ssm_c,
                mask_d,
            )
            out = mx.concatenate([out_c, out_d], axis=1)
        else:
            lengths = cache.lengths if cache is not None else None
            out, conv_f, ssm_f = self._process_chunk(
                qkv, a, b, conv_state, ssm_state, mask, lengths=lengths
            )

        if cache is not None:
            cache[0] = conv_f
            cache[1] = ssm_f
            cache.advance(S)

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        return out

    cls._process_chunk = _process_chunk
    cls.__call__ = __call__
    cls._omlx_mtp_patched = True


# ---------------------------------------------------------------------------
# Qwen3NextDecoderLayer — pass n_confirmed to linear attn.
# ---------------------------------------------------------------------------

def _patch_decoder_layer(q3n) -> None:
    cls = q3n.Qwen3NextDecoderLayer
    if "_omlx_mtp_patched" in cls.__dict__:
        return

    def __call__(self, x, mask=None, cache=None, n_confirmed: int = 0):
        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x), mask, cache, n_confirmed=n_confirmed
            )
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

    cls.__call__ = __call__
    cls._omlx_mtp_patched = True


# ---------------------------------------------------------------------------
# Qwen3NextModel — return pre-norm hidden, accept n_confirmed.
# ---------------------------------------------------------------------------

def _patch_qwen3_next_model(q3n) -> None:
    cls = q3n.Qwen3NextModel
    if "_omlx_mtp_patched" in cls.__dict__:
        return

    create_attention_mask = q3n.create_attention_mask
    create_ssm_mask = q3n.create_ssm_mask

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        n_confirmed: int = 0,
    ):
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(
                hidden_states, mask=mask, cache=c, n_confirmed=n_confirmed
            )

        return hidden_states

    cls.__call__ = __call__
    cls._omlx_mtp_patched = True


# ---------------------------------------------------------------------------
# Model — wrap __init__, replace __call__, add mtp_forward / make_mtp_cache,
# refresh sanitize / quant_predicate.
# ---------------------------------------------------------------------------

def _patch_outer_model(q3n) -> None:
    cls = q3n.Model
    if "_omlx_mtp_patched" in cls.__dict__:
        return

    from mlx_lm.models.cache import KVCache

    original_init = cls.__init__

    def __init__(self, args):
        original_init(self, args)
        n_mtp = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        from . import is_mtp_active

        if n_mtp > 0 and is_mtp_active():
            self.mtp = q3n.MTPModule(args)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        hidden = self.model(
            inputs,
            cache,
            input_embeddings=input_embeddings,
            n_confirmed=n_confirmed,
        )
        normed = self.model.norm(hidden)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(normed)
        else:
            out = self.lm_head(normed)
        if return_hidden:
            return out, hidden
        return out

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
        mtp_out = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
        )
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(mtp_out)
        return self.lm_head(mtp_out)

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    def sanitize(self, weights):
        import mlx.core as mx

        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and getattr(v, "shape", (1,))[-1] != 1
            for k, v in weights.items()
        )
        should_shift_norm_weights = has_unsanitized_conv1d

        if not hasattr(self, "mtp"):
            weights = {k: v for k, v in weights.items() if "mtp." not in k}
        elif not any("mtp." in k for k in weights):
            raise ValueError(
                "Native MTP is enabled for this model but the converted "
                "weights are missing the mtp.* tensors. Default mlx-lm "
                "converters strip them; you need a converter that preserves "
                "MTP weights. To recover without re-converting, "
                "open the model's settings in the oMLX admin UI and toggle "
                "'Native MTP' off, then retry."
            )

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # MoE expert stacking (per-expert → switch_mlp).
        is_moe = "model.layers.0.mlp.experts.0.up_proj.weight" in weights
        if is_moe:
            for l in range(self.args.num_hidden_layers):
                prefix = f"model.layers.{l}.mlp"
                for n in ["up_proj", "down_proj", "gate_proj"]:
                    to_join = [
                        weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                        for e in range(self.args.num_experts)
                    ]
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(to_join)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if should_shift_norm_weights and any(
                k.endswith(sfx) for sfx in norm_keys
            ):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            if path.endswith("mtp.fc"):
                return False
            return True

        if (
            self.args.num_experts <= 0
            and int(getattr(self.args, "mtp_num_hidden_layers", 0) or 0) <= 0
        ):
            return None
        return predicate

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.sanitize = sanitize
    cls.quant_predicate = property(quant_predicate)
    cls._omlx_mtp_patched = True
