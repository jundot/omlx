# SPDX-License-Identifier: Apache-2.0
"""Native MTP monkey-patch for Intern-S2-Mobius (a Qwen3.5 architecture sibling).

Intern-S2-Mobius ships an MTP draft head with the same shape PR #990 defines for
Qwen3.5/3.6: slot i fuses the backbone's final hidden state at position i with
the embedding of the sampled token i+1 and predicts token i+2. This patch
re-expresses our proven ``interns2_mobius`` head onto oMLX's generic MTP driver
contract (``mtp_forward`` / ``make_mtp_cache`` / ``mtp_partial_rollback`` plus the
per-layer ``cache.rollback_state`` snapshot the ``batch_generator`` drives), so
the existing continuous-batching / paged-cache engine runs the draft/verify cycle
unchanged.

Rollback uses oMLX's native ``rollback_state`` + ``_mtp_draft_stash`` idiom: the
GatedDeltaNet verify forward runs the ``[confirmed, drafts]`` window as one chunk,
keeps zero-copy references to its pre-forward ``(conv_state, ssm_state)`` and the
projected ``(qkv, a, b)`` inputs, and on a partial reject replays the kept prefix
through ``_process_chunk``. This is the same masked-replay computation our stock
mlx-lm head used, mapped onto the driver's cache slots.

The vendored ``interns2_mobius_model`` grows the head classes but attaches
``self.mtp`` only when MTP decode is active, so plain generation is byte-identical.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MODULE_NAME = "mlx_lm.models.interns2_mobius"


def _is_our_method(cls: Any, name: str, marker: str) -> bool:
    fn = cls.__dict__.get(name)
    return fn is not None and getattr(fn, marker, False)


def _gdn_rollback_to(gdn: Any, cache: Any, keep: int) -> None:
    """Rewind a GatedDeltaNet verify window so only its first ``keep`` tokens
    count, from the zero-copy pre-forward refs stashed on the cache.

    Full-width masked replay (our proven scheme, mapped onto oMLX's cache slots):
    the SSM re-runs the stashed window with a keep-mask -- the same kernel shape
    as the original forward, so the recurrent state is bitwise-equal to a run
    that only ever saw the kept tokens -- and the conv state is sliced at the
    kept offset. A narrower sliced replay (mask=None) rounds differently at
    keep=1 in bf16; the masked full-width replay does not.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.gated_delta import gated_delta_update

    conv_0, ssm_0 = cache.rollback_state
    qkv_s, a_s, b_s = cache._mtp_draft_stash
    B, S = qkv_s.shape[:2]
    n_pad = gdn.conv_kernel_size - 1

    conv_in = mx.concatenate([conv_0, qkv_s], axis=1)
    cache[0] = mx.contiguous(conv_in[:, keep : keep + n_pad])

    conv_out = nn.silu(gdn.conv1d(conv_in))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
            [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
            [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    keep_mask = (mx.arange(S) < keep)[None]
    _, cache[1] = gated_delta_update(
        q, k, v, a_s, b_s, gdn.A_log, gdn.dt_bias, ssm_0, keep_mask
    )


def apply() -> bool:
    """Apply the Intern-S2-Mobius MTP patch. Idempotent and self-healing."""
    import sys

    mod = sys.modules.get(_MODULE_NAME)
    if mod is None:
        logger.debug(
            "%s not registered; skipping Intern-S2-Mobius MTP patch", _MODULE_NAME
        )
        return False

    _patch_gated_delta_net(mod)
    _patch_decoder_layer(mod)
    _patch_backbone_model(mod)
    _patch_model(mod)
    logger.info("Intern-S2-Mobius MTP model patch applied")
    return True


# ---------------------------------------------------------------------------
# GatedDeltaNet — n_confirmed-aware __call__ + _process_chunk helper.
# ---------------------------------------------------------------------------


def _patch_gated_delta_net(mod: Any) -> None:
    cls = mod.InternS2MobiusGatedDeltaNet
    if _is_our_method(cls, "__call__", "_omlx_mtp_call_marker"):
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

    def __call__(self, x, mask=None, cache=None, n_confirmed: int = 0):
        B, S, _ = x.shape

        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype
            )
        ssm_state = cache[1] if cache is not None else None

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)

        if n_confirmed > 0 and n_confirmed < S:
            # MTP verify forward: run the whole [confirmed, drafts] window as
            # one chunk and stash the zero-copy pre-forward refs so a partial
            # reject can replay the kept prefix through _process_chunk.
            out, conv_f, ssm_f = self._process_chunk(
                qkv, a, b, conv_state, ssm_state, mask
            )
            if cache is not None:
                cache.rollback_state = (conv_state, ssm_state)
                cache._mtp_draft_stash = (qkv, a, b)
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
        return self.out_proj(out.reshape(B, S, -1))

    cls._process_chunk = _process_chunk
    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__


# ---------------------------------------------------------------------------
# DecoderLayer — thread n_confirmed to the linear-attention layer.
# ---------------------------------------------------------------------------


def _patch_decoder_layer(mod: Any) -> None:
    cls = mod.InternS2MobiusDecoderLayer
    if _is_our_method(cls, "__call__", "_omlx_mtp_call_marker"):
        return

    def __call__(self, x, meta_block, mask=None, cache=None, n_confirmed: int = 0):
        if self.is_linear:
            h_in = self.input_layernorm(x)
            if n_confirmed:
                r = self.linear_attn(h_in, mask, cache, n_confirmed=n_confirmed)
            else:
                r = self.linear_attn(h_in, mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r

        normed = self.post_attention_layernorm(h)
        B, S, D = normed.shape
        normed_2d = normed.reshape(-1, D)
        out = meta_block(normed_2d, self.shared_slot, self.mlp(normed_2d))
        return h + out.reshape(B, S, D)

    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__


# ---------------------------------------------------------------------------
# InternS2MobiusModel (backbone) — accept + thread n_confirmed.
# ---------------------------------------------------------------------------


def _patch_backbone_model(mod: Any) -> None:
    cls = mod.InternS2MobiusModel
    if _is_our_method(cls, "__call__", "_omlx_mtp_call_marker"):
        return

    create_attention_mask = mod.create_attention_mask
    create_ssm_mask = mod.create_ssm_mask

    def __call__(self, inputs, cache=None, n_confirmed: int = 0):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(
                h,
                self.meta_mlp[i % self.num_blocks],
                mask=mask,
                cache=c,
                n_confirmed=n_confirmed,
            )

        return self.norm(h)

    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__


# ---------------------------------------------------------------------------
# Model — attach MTP head, return_hidden / n_confirmed __call__, MTP hooks.
# ---------------------------------------------------------------------------


def _patch_model(mod: Any) -> None:
    cls = mod.Model
    init_wrapped = getattr(cls, "_omlx_mtp_init_wrapped", False)
    call_owned = _is_our_method(cls, "__call__", "_omlx_mtp_call_marker")
    if init_wrapped and call_owned:
        return

    import mlx.core as mx  # noqa: F401  (kept for parity / future use)
    from mlx_lm.models.cache import KVCache

    original_init = cls.__init__
    original_sanitize = cls.sanitize

    def __init__(self, args):
        original_init(self, args)
        n_mtp = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        from . import get_mtp_depth, is_mtp_active

        mtp_decode_enabled = bool(n_mtp > 0 and is_mtp_active())
        self._omlx_mtp_decode_enabled = mtp_decode_enabled
        # The head consumes the backbone's post-norm hidden (our reference
        # convention). Model.__call__ returns that post-norm hidden and marks
        # it so the driver's trunk-norm fold is a no-op.
        self._omlx_mtp_head_hidden_normed = True
        if mtp_decode_enabled:
            self.mtp = mod.InternS2MobiusMTP(args)
            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = get_mtp_depth()

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden: bool = False,
        n_confirmed: int = 0,
    ):
        hidden = self.model(inputs, cache, n_confirmed=n_confirmed)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(hidden)
        else:
            out = self.lm_head(hidden)
        if return_hidden:
            return out, hidden
        return out

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        token_embed = self.model.embed_tokens(next_token_ids)
        mtp_out = self.mtp(token_embed, hidden_states, mtp_cache)
        logits_source = mtp_out
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:, :]
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(logits_source)
        else:
            logits = self.lm_head(logits_source)
        if return_hidden:
            return logits, mtp_out
        return logits

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        """Roll the backbone cache back to ``accepted`` drafts after a depth-k
        verify forward over ``[confirmed, d1..dk]``.

        Full-attention KV layers trim; GatedDeltaNet layers restore the
        zero-copy pre-forward refs and replay the kept prefix (confirmed +
        accepted drafts) through ``_process_chunk``.
        """
        layers = self.model.layers
        if len(cache) != len(layers):
            return False
        trim_n = num_drafts - accepted
        if trim_n <= 0:
            return True
        keep = 1 + accepted
        for layer, c in zip(layers, cache):
            if getattr(layer, "is_linear", False):
                if getattr(c, "rollback_state", None) is None:
                    return False
                if getattr(c, "_mtp_draft_stash", None) is None:
                    return False
            else:
                if not (hasattr(c, "is_trimmable") and c.is_trimmable()):
                    return False
        for layer, c in zip(layers, cache):
            if getattr(layer, "is_linear", False):
                _gdn_rollback_to(layer.linear_attn, c, keep)
                c.rollback_state = None
                c._mtp_draft_stash = None
            else:
                c.trim(trim_n)
        return True

    def sanitize(self, weights):
        # Preserve the (already fused + quantized) mtp.* head keys when a head
        # is attached; the backbone sanitize otherwise drops them. Stash and
        # restore so the backbone's conv1d/expert-fusion passes never touch
        # them (the head has no conv1d and ships pre-fused experts).
        mtp_saved = {}
        if hasattr(self, "mtp"):
            mtp_saved = {
                k: weights.pop(k) for k in list(weights) if k.startswith("mtp.")
            }
        weights = original_sanitize(self, weights)
        weights.update(mtp_saved)
        return weights

    if not init_wrapped:
        cls.__init__ = __init__
        cls._omlx_mtp_init_wrapped = True
    __call__._omlx_mtp_call_marker = True
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_partial_rollback = mtp_partial_rollback
    cls.sanitize = sanitize
