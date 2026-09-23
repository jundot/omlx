# SPDX-License-Identifier: Apache-2.0
"""Native MTP (Multi-Token Prediction) monkey-patch for MiMo-V2 (mlx-lm).

MiMo-V2 ships ``model.mtp.layers.0..N`` heads (three in MiMo-V2.5/2.6):
each is a self-contained draft step — enorm(embedding) + hnorm(trunk
hidden) fused through ``eh_proj``, one SWA decoder layer with a dense FFN,
then ``final_layernorm`` into the shared lm_head. The blocks chain
sequentially (block i drafts token t+i+1), mirroring the inkling
multi-block head pattern: a per-cycle round counter routes fold/chain
calls to the matching block, and the head marks ``_omlx_mtp_head_prenorm``
because ``hnorm`` normalizes its hidden input internally — so the backbone
hands out the raw pre-norm trunk hidden (``return_hidden`` support lives
in the vendored ``mlx_lm.models.mimo_v2`` model itself).

Stock ``Model.sanitize`` strips ``model.mtp.*``; the patched sanitize
keeps them when the head is attached. Contract mirrors qwen35_model.py:

- ``Model.__init__``: attach ``self.mtp`` when the config declares
  ``num_nextn_predict_layers > 0`` and ``is_mtp_active()``.
- ``mtp_forward`` / ``make_mtp_cache`` / ``mtp_begin_cycle``: the
  BatchGenerator MTP contract.

All patches are idempotent via marker-based identity checks.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MTP_WEIGHT_PREFIX = "model.mtp."


def apply() -> bool:
    """Apply the MiMo-V2 MTP monkey-patches to mlx-lm.

    Returns True when the patch set is installed on
    ``mlx_lm.models.mimo_v2`` (or already was), False when the module is
    not importable (the vendored registration has not run).
    """
    try:
        import mlx_lm.models.mimo_v2 as mimo
    except ImportError:
        return False

    if not hasattr(mimo, "Attention") or not hasattr(mimo, "MiMoV2Model"):
        return False

    _register_mtp_classes(mimo)
    _patch_model(mimo)
    return True


def _register_mtp_classes(mimo: Any) -> None:
    """Attach ``MTPDecoderLayer`` / ``MTPModule`` to the mimo_v2 module."""
    if hasattr(mimo, "MTPModule"):
        return

    import mlx.core as mx
    import mlx.nn as nn

    Attention = mimo.Attention  # noqa: N806
    MLP = mimo.MLP  # noqa: N806

    class MTPDecoderLayer(nn.Module):
        """One draft step: enorm/hnorm + eh_proj + SWA attention + dense FFN.

        MiMo's MTP blocks use sliding-window attention (window 128, SWA
        head geometry, learnable sink bias) and a dense FFN instead of the
        backbone's MoE — per the MiMo-V2 technical report this keeps each
        block at ~0.33B params. Each block owns its full norm stack,
        matching the checkpoint key layout.
        """

        def __init__(self, config: Any):
            super().__init__()
            h = config.hidden_size
            eps = getattr(config, "layernorm_epsilon", 1e-5)
            self.enorm = nn.RMSNorm(h, eps=eps)
            self.hnorm = nn.RMSNorm(h, eps=eps)
            self.eh_proj = nn.Linear(h * 2, h, bias=False)
            self.self_attn = Attention(config, is_sliding_window=True)
            self.input_layernorm = nn.RMSNorm(h, eps=eps)
            self.pre_mlp_layernorm = nn.RMSNorm(h, eps=eps)
            self.mlp = MLP(config)
            self.final_layernorm = nn.RMSNorm(h, eps=eps)

        def __call__(
            self,
            hidden_states: mx.array,
            next_token_ids: mx.array,
            embed_tokens: nn.Embedding,
            cache: Any | None = None,
        ) -> mx.array:
            e = self.enorm(embed_tokens(next_token_ids))
            h = self.hnorm(hidden_states)
            x = self.eh_proj(mx.concatenate([e, h], axis=-1))
            x = x + self.self_attn(self.input_layernorm(x), cache=cache)
            return x + self.mlp(self.pre_mlp_layernorm(x))

    class MTPModule(nn.Module):
        """Sequential multi-block MTP head (one block per draft depth).

        ``__call__`` routes by a per-cycle round counter: the fold call
        (committed tokens) runs block 0, each chain step runs the next
        block, clamped at the last. ``begin_cycle`` resets the counter and
        is invoked by BatchGenerator before every fold.
        """

        def __init__(self, config: Any, num_blocks: int):
            super().__init__()
            self.layers = [MTPDecoderLayer(config) for _ in range(num_blocks)]
            self._round = 0

        def begin_cycle(self, cache: Any = None, depth: int = 1) -> None:
            del cache, depth
            self._round = 0

        def __call__(
            self,
            hidden_states: mx.array,
            next_token_ids: mx.array,
            embed_tokens: nn.Embedding,
            cache: Any | None = None,
        ) -> mx.array:
            idx = min(self._round, len(self.layers) - 1)
            self._round += 1
            if cache is None:
                cache = [None] * len(self.layers)
            block = self.layers[idx]
            x = block(hidden_states, next_token_ids, embed_tokens, cache[idx])
            return block.final_layernorm(x)

    mimo.MTPDecoderLayer = MTPDecoderLayer
    mimo.MTPModule = MTPModule


def _patch_model(mimo: Any) -> None:
    cls = mimo.Model
    backbone_cls = mimo.MiMoV2Model

    init_wrapped = getattr(backbone_cls, "_omlx_mtp_init_wrapped", False)
    if init_wrapped and hasattr(cls, "mtp_forward"):
        return

    from mlx_lm.models.cache import RotatingKVCache

    # The head lives on the backbone: checkpoint keys are
    # ``model.mtp.layers.*``, so the subtree must sit on ``Model.model``
    # for mlx_lm's load_weights to bind it. The outer Model exposes it via
    # a pass-through property (BatchGenerator's _model_has_mtp_module) and
    # carries the decode-time markers itself (its detector only walks
    # ``model`` / ``language_model``).
    if not init_wrapped:
        original_init = backbone_cls.__init__

        def patched_init(self, args: Any):  # noqa: N807
            original_init(self, args)
            n_mtp = int(getattr(args, "num_nextn_predict_layers", 0) or 0)
            from . import is_mtp_active

            mtp_decode_enabled = bool(n_mtp > 0 and is_mtp_active())
            self._omlx_mtp_decode_enabled = mtp_decode_enabled
            # The head normalizes its hidden input internally (hnorm), so
            # the fold must hand it the raw pre-norm trunk hidden.
            self._omlx_mtp_head_prenorm = True
            if mtp_decode_enabled:
                self.mtp = mimo.MTPModule(args, n_mtp)
                from . import get_mtp_depth

                self._omlx_mtp_chain = True
                self._omlx_mtp_depth = min(get_mtp_depth(), n_mtp)
                # Head blocks 1..N advance on speculative chain entries that
                # a post-hoc trim cannot undo once their window-128
                # RotatingKVCache has rotated (and before rotation the trim
                # would still leave rejected-cycle entries behind), so the
                # chain drafts on a per-cycle clone and the persistent head
                # cache stays committed-only (fold-fed).
                self._omlx_mtp_head_clone = True

        backbone_cls.__init__ = patched_init
        backbone_cls._omlx_mtp_init_wrapped = True

    original_sanitize = cls.sanitize

    def sanitize(self, weights):
        if not getattr(self.model, "_omlx_mtp_decode_enabled", False):
            return original_sanitize(self, weights)
        kept = {k: v for k, v in weights.items() if k.startswith(_MTP_WEIGHT_PREFIX)}
        out = original_sanitize(self, weights)
        out.update(kept)
        return out

    def mtp_forward(
        self,
        hidden_states: Any,
        next_token_ids: Any,
        mtp_cache: Any | None = None,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        head_hidden = self.model.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
        )
        logits_source = head_hidden
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:, :]
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(logits_source)
        else:
            logits = self.lm_head(logits_source)
        if return_hidden:
            return logits, head_hidden
        return logits

    def make_mtp_cache(self):
        window = int(self.args.sliding_window_size)
        self.model.mtp.begin_cycle()
        return [
            RotatingKVCache(max_size=window, keep=0)
            for _ in self.model.mtp.layers
        ]

    def mtp_begin_cycle(self, cache: Any = None, depth: int = 1) -> None:
        self.model.mtp.begin_cycle(cache, depth)

    def mtp_partial_rollback(
        self, cache: Any, accepted: int, num_drafts: int
    ) -> bool:
        """Roll the backbone cache back to ``accepted`` drafts after a
        depth-k verify forward over ``[confirmed, d1..dk]``.

        MiMo is pure softmax attention — every layer simply discards the
        ``num_drafts - accepted`` rejected positions. Full-attention
        ``KVCache`` layers trim natively; sliding-window ``RotatingKVCache``
        layers refuse stock trim once rotated but are trimmable through the
        MTP undo stash armed by ``batch_generator._call_backbone``. Without
        this hook ``_chain_rollback`` refuses every partial accept at
        depth > 1 and every verify cycle falls back to a full re-prefill.
        """
        if len(cache) != len(self.layers):
            return False
        trim_n = num_drafts - accepted
        if trim_n <= 0:
            return True
        for c in cache:
            if not (hasattr(c, "is_trimmable") and c.is_trimmable()):
                return False
        for c in cache:
            c.trim(trim_n)
        return True

    cls.mtp = property(lambda self: self.model.mtp)
    cls.sanitize = sanitize
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_begin_cycle = mtp_begin_cycle
    cls.mtp_partial_rollback = mtp_partial_rollback
    # Mirror the decode-time markers onto the outer wrapper: the
    # eligibility / depth / pre-norm detectors inspect the outer instance
    # or language_model, never ``model.model``.
    original_cls_init = cls.__init__

    def outer_init(self, args: Any):
        original_cls_init(self, args)
        inner = self.model
        self._omlx_mtp_decode_enabled = bool(
            getattr(inner, "_omlx_mtp_decode_enabled", False)
        )
        self._omlx_mtp_head_prenorm = True
        if getattr(inner, "_omlx_mtp_chain", False):
            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = getattr(inner, "_omlx_mtp_depth", 1)
            if getattr(inner, "_omlx_mtp_head_clone", False):
                self._omlx_mtp_head_clone = True

    cls.__init__ = outer_init
