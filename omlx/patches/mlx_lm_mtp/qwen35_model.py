# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch for ml-explore/mlx-lm PR #990 — Qwen3.5/3.6 native MTP.

Adds an MTP head to ``mlx_lm.models.qwen3_5.TextModel`` (the language-model
half) and a pass-through on ``mlx_lm.models.qwen3_5.Model`` (the VLM-outer
wrapper). The mechanism mirrors the patch idiom in
``omlx/patches/gated_delta_advance.py``: replace class methods on a one-shot,
idempotent basis tracked by a module flag.

Important: the class names below match what mlx-lm 0.31.x actually exports.
Earlier drafts of this patch used ``Qwen3_5GatedDeltaNet`` / ``Qwen3_5DecoderLayer``
matching the source-PR's class names; mlx-lm names them ``GatedDeltaNet`` and
``DecoderLayer``. The patch targets the runtime names.

What this patch installs (all on classes from ``mlx_lm.models.qwen3_5``):

- ``TextModelArgs``: a runtime ``mtp_num_hidden_layers`` instance attribute
  via a thin ``from_dict`` wrapper. We can't add a real dataclass field at
  runtime without rebuilding the class, so we attach the value to the
  instance after ``from_dict`` runs and rely on duck-typing in the rest of
  the code.
- ``GatedDeltaNet``: ``_process_chunk`` helper added, ``__call__`` body
  replaced to support the ``n_confirmed`` argument. When ``n_confirmed`` is
  between 1 and ``S - 1``, the prefix is processed first and the
  ``(conv_state, ssm_state)`` snapshot is stored on the cache as
  ``rollback_state`` so a rejected draft can restore it.
- ``DecoderLayer``: ``__call__`` passes ``n_confirmed`` through to the
  linear-attention sublayer.
- ``Qwen3_5TextModel``: ``__call__`` accepts ``n_confirmed`` and returns
  *pre-norm* hidden states (the MTP head needs them).
- ``TextModel``: ``__init__`` wrapped to attach a fresh ``MTPModule`` when
  the args declared one. ``__call__`` accepts ``return_hidden`` and
  ``n_confirmed`` and decouples pre- vs post-norm hidden / logits.
  ``mtp_forward`` and ``make_mtp_cache`` added. ``sanitize`` keeps the
  ``mtp.*`` keys when an MTP head exists. ``quant_predicate`` keeps the
  fusion projection in full precision.
- ``Model`` (VLM outer wrapper): ``__call__`` and ``mtp_forward`` /
  ``make_mtp_cache`` pass through to the language model. This is what lets
  ``mtp_enabled`` work for both LM (``model_type=qwen3_5``) and VLM
  (``model_type=qwen3_5_vl``) checkpoints — the VLM's ``language_model``
  is the patched ``TextModel`` instance.
- ``MTPDecoderLayer`` and ``MTPModule`` are registered as new attributes
  on ``mlx_lm.models.qwen3_5`` so the patched ``TextModel.__init__`` can
  find them.

The patch is intentionally limited to ``mlx_lm.models.qwen3_5``; mlx-vlm's
``mlx_vlm.models.qwen3_5.language`` is a separate copy and is not touched
(oMLX's existing ``gated_delta_advance.py`` already covers that side).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PATCHED = False


def apply() -> bool:
    """Apply PR 990 model-side patches to mlx_lm.models.qwen3_5. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from mlx_lm.models import qwen3_5 as q35
    except ImportError:
        logger.debug("mlx_lm.models.qwen3_5 not importable; skipping MTP patch")
        return False

    # Skip if upstream already merged PR 990: TextModel already has mtp_forward.
    if hasattr(q35.TextModel, "mtp_forward") and not hasattr(
        q35.TextModel, "_omlx_mtp_patched"
    ):
        _PATCHED = True
        q35.TextModel._omlx_mtp_patched = "upstream"
        return True

    _patch_text_model_args(q35)
    _register_mtp_classes(q35)
    _patch_gated_delta_net(q35)
    _patch_decoder_layer(q35)
    _patch_qwen3_5_text_model(q35)
    _patch_text_model(q35)
    _patch_outer_model(q35)
    _patch_qwen3_5_moe()

    _PATCHED = True
    q35.TextModel._omlx_mtp_patched = "patch"
    logger.info("Qwen3.5/3.6 MTP model patch applied (PR 990)")
    return True


# ---------------------------------------------------------------------------
# TextModelArgs.from_dict — surface mtp_num_hidden_layers as instance attr.
# ---------------------------------------------------------------------------

def _patch_text_model_args(q35: Any) -> None:
    """Wrap ``TextModelArgs.from_dict`` to retain ``mtp_num_hidden_layers``.

    We can't add a dataclass field at runtime without rebuilding the class,
    so we read the raw value from the source dict and ``setattr`` on the
    instance. Subsequent code reads ``args.mtp_num_hidden_layers`` and gets
    the right thing. ``BaseModelArgs.from_dict`` ignores unknown keys, so
    plain construction would discard the value otherwise.
    """
    args_cls = q35.TextModelArgs
    if hasattr(args_cls, "_omlx_mtp_from_dict_patched"):
        return

    original_from_dict = args_cls.from_dict.__func__  # unwrap classmethod

    def patched_from_dict(cls, params):
        instance = original_from_dict(cls, params)
        # Default to 0 when missing so the rest of the code can branch on it.
        instance.mtp_num_hidden_layers = int(
            params.get("mtp_num_hidden_layers", 0) or 0
        )
        return instance

    args_cls.from_dict = classmethod(patched_from_dict)
    args_cls._omlx_mtp_from_dict_patched = True


# ---------------------------------------------------------------------------
# MTPDecoderLayer + MTPModule — register on the qwen3_5 module.
# ---------------------------------------------------------------------------

def _register_mtp_classes(q35: Any) -> None:
    """Attach ``MTPDecoderLayer`` / ``MTPModule`` to the qwen3_5 module."""
    if hasattr(q35, "MTPModule"):
        return

    import mlx.core as mx
    import mlx.nn as nn

    # qwen3_5 imports these as module attributes via ``from .qwen3_next
    # import ... as ...`` so we can read them from the module directly.
    Attention = q35.Attention
    SparseMoeBlock = q35.SparseMoeBlock
    MLP = q35.MLP
    create_attention_mask = q35.create_attention_mask

    class MTPDecoderLayer(nn.Module):
        """Full-attention transformer layer used inside the MTP head.

        Unlike the standard ``DecoderLayer`` (which routes through
        ``GatedDeltaNet`` for "linear" layers), the MTP head only uses
        full attention. MoE config is honored when num_experts > 0.
        """

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
        """Multi-Token Prediction head from PR #990.

        Predicts token t+2 by fusing the backbone's pre-norm hidden state
        at position t with the embedding of the sampled main token t+1.
        """

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

    q35.MTPDecoderLayer = MTPDecoderLayer
    q35.MTPModule = MTPModule


# ---------------------------------------------------------------------------
# GatedDeltaNet — _process_chunk helper + __call__ replacement.
# ---------------------------------------------------------------------------

def _patch_gated_delta_net(q35: Any) -> None:
    """Replace ``GatedDeltaNet.__call__`` with an n_confirmed-aware body.

    The original body inlines the conv1d / gated_delta_update path against
    the full sequence. PR 990 splits that path into ``_process_chunk`` so
    the same helper can run on the prefix (n_confirmed tokens) and the
    suffix (draft tokens) separately, snapshotting the SSM/conv state in
    between for rollback on draft rejection.
    """
    cls = q35.GatedDeltaNet
    if "_omlx_mtp_patched" in cls.__dict__:
        return

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.nn.layers.distributed import sum_gradients
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
        inputs: Any,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
        n_confirmed: int = 0,
    ):
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

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

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out

    cls._process_chunk = _process_chunk
    cls.__call__ = __call__
    cls._omlx_mtp_patched = True


# ---------------------------------------------------------------------------
# DecoderLayer — pass n_confirmed to linear attn.
# ---------------------------------------------------------------------------

def _patch_decoder_layer(q35: Any) -> None:
    cls = q35.DecoderLayer
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
# Qwen3_5TextModel — return pre-norm hidden, accept n_confirmed.
# ---------------------------------------------------------------------------

def _patch_qwen3_5_text_model(q35: Any) -> None:
    cls = q35.Qwen3_5TextModel
    if "_omlx_mtp_patched" in cls.__dict__:
        return

    create_attention_mask = q35.create_attention_mask
    create_ssm_mask = q35.create_ssm_mask

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

        # PR 990: return pre-norm hidden so the MTP head can fuse it. The
        # wrapping ``TextModel.__call__`` applies ``self.model.norm`` on top
        # to produce logits.
        return hidden_states

    cls.__call__ = __call__
    cls._omlx_mtp_patched = True


# ---------------------------------------------------------------------------
# TextModel — wrap __init__, replace __call__, add mtp_forward / make_mtp_cache,
# refresh sanitize / quant_predicate.
# ---------------------------------------------------------------------------

def _patch_text_model(q35: Any) -> None:
    cls = q35.TextModel
    if "_omlx_mtp_patched" in cls.__dict__:
        return

    from mlx_lm.models.cache import KVCache

    original_init = cls.__init__

    def __init__(self, args):
        original_init(self, args)
        n_mtp = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        # Only attach the MTP head when the active-flag is set. With the
        # flag off the model is indistinguishable from a stock no-MTP
        # build: ``hasattr(self, "mtp")`` is False, sanitize strips
        # ``mtp.*`` weights, and BatchGenerator's _is_mtp_eligible bails
        # out because the inner ``language_model`` has no ``mtp``.
        from . import is_mtp_active

        if n_mtp > 0 and is_mtp_active():
            self.mtp = q35.MTPModule(args)

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
        # Full PR 990 replacement of TextModel.sanitize. We can't call the
        # original because mlx-lm's stock body unconditionally strips the
        # ``mtp.*`` keys (``weights = {k: v for k, v in weights.items() if
        # "mtp." not in k}``), which would defeat the whole patch. The
        # logic below mirrors PR 990's body line-for-line:
        #   - keep mtp.* when the model has an mtp head, drop otherwise
        #   - shift norm weights by +1 only on raw-HF checkpoints (detected
        #     via unsanitized conv1d shapes); already-converted MLX models
        #     must not be shifted again even when MTP weights are present
        #   - extend the norm-shift set with MTP-specific norm names so a
        #     genuine raw-HF MTP checkpoint also gets the right shift
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
                "MTP weights (or a Qwen3.6 / DeepSeek-V4 checkpoint that "
                "already preserves them). To recover without re-converting, "
                "open the model's settings in the oMLX admin UI and toggle "
                "'Native MTP' off, then retry."
            )

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            # MTP-specific norms (not covered by the patterns above)
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
            # Keep the MTP fusion projection in full precision.
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


# ---------------------------------------------------------------------------
# Model (VLM-style outer wrapper) — pass-through for return_hidden, n_confirmed,
# mtp_forward, make_mtp_cache. Lets ``mtp_enabled`` work for VLM checkpoints
# too (their ``language_model`` is an instance of the patched ``TextModel``).
# ---------------------------------------------------------------------------

def _patch_outer_model(q35: Any) -> None:
    cls = q35.Model
    if "_omlx_mtp_patched" in cls.__dict__:
        return

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            return_hidden=return_hidden,
            n_confirmed=n_confirmed,
        )

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache):
        return self.language_model.mtp_forward(
            hidden_states, next_token_ids, mtp_cache
        )

    def make_mtp_cache(self):
        return self.language_model.make_mtp_cache()

    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls._omlx_mtp_patched = True


# ---------------------------------------------------------------------------
# qwen3_5_moe.Model — sanitize replacement to handle MTP MoE weight formats.
# ---------------------------------------------------------------------------

def _patch_qwen3_5_moe() -> None:
    """Replace ``qwen3_5_moe.Model.sanitize`` to handle MTP MoE weight formats.

    PR 990 extends the MoE sanitize to detect whether MTP layers ship their
    expert weights in the fused ``gate_up_proj`` form (Qwen3.6) or as
    per-expert tensors (Qwen3.5) and convert each to the unified
    ``switch_mlp`` layout. Without this, loading a Qwen3.5/3.6 MoE
    checkpoint that has MTP weights fails on key mismatches.
    """
    try:
        from mlx_lm.models import qwen3_5_moe as moe
    except ImportError:
        # Some installs don't ship the MoE module; that's fine — dense
        # Qwen3.5 still works through the regular qwen3_5 patch.
        logger.debug("mlx_lm.models.qwen3_5_moe not importable; skipping MoE patch")
        return

    cls = moe.Model
    if "_omlx_mtp_patched" in cls.__dict__:
        return

    import mlx.core as mx

    def _unfuse_experts(weights, prefix):
        gate_up_key = f"{prefix}.experts.gate_up_proj"
        if gate_up_key not in weights:
            return
        gate_up = weights.pop(gate_up_key)
        mid = gate_up.shape[-2] // 2
        weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
        weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
        weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
            f"{prefix}.experts.down_proj"
        )

    def _stack_per_expert(weights, prefix, num_experts):
        if f"{prefix}.experts.0.gate_proj.weight" not in weights:
            return
        for n in ("gate_proj", "up_proj", "down_proj"):
            weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(
                [
                    weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                    for e in range(num_experts)
                ]
            )

    def sanitize(self, weights):
        new_weights = {}
        for key, value in weights.items():
            if key.startswith("vision_tower") or key.startswith("model.visual"):
                continue
            if key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif not key.startswith("language_model."):
                key = "language_model." + key
            new_weights[key] = value

        # Backbone MoE layers always use fused gate_up_proj (Qwen3.5/3.6).
        for l in range(self.language_model.args.num_hidden_layers):
            _unfuse_experts(new_weights, f"language_model.model.layers.{l}.mlp")

        # MTP layers: fused (Qwen3.6) or per-expert (Qwen3.5). Detect once.
        mtp_num = int(
            getattr(self.language_model.args, "mtp_num_hidden_layers", 0) or 0
        )
        if mtp_num > 0:
            has_any_mtp = any(
                k.startswith("language_model.mtp.") for k in new_weights
            )
            if not has_any_mtp:
                logger.debug(
                    "mtp_num_hidden_layers=%d but no MTP weights found; "
                    "model may have been quantized without preserve_mtp",
                    mtp_num,
                )
            else:
                num_experts = self.language_model.args.num_experts
                mtp_is_fused = (
                    "language_model.mtp.layers.0.mlp.experts.gate_up_proj"
                    in new_weights
                )
                for layer_idx in range(mtp_num):
                    prefix = f"language_model.mtp.layers.{layer_idx}.mlp"
                    # Idempotent: oQ outputs already store experts in
                    # switch_mlp form (mlx-vlm sanitize patch unfuses MTP
                    # experts before quantization). Skip the load-time
                    # unfuse/stack in that case.
                    if f"{prefix}.switch_mlp.gate_proj.weight" in new_weights:
                        continue
                    if mtp_is_fused:
                        _unfuse_experts(new_weights, prefix)
                    else:
                        _stack_per_expert(new_weights, prefix, num_experts)

        return self.language_model.sanitize(new_weights)

    cls.sanitize = sanitize
    cls._omlx_mtp_patched = True
