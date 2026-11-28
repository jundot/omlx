# SPDX-License-Identifier: Apache-2.0
"""ERNIE 5.5 as the host of its bundled DSpark draft in oMLX's MTP loop."""

from collections import deque

import mlx.core as mx

from ..deepseek_v41.mtp import AcceptanceDepthController
from ..mlx_lm_mtp import cache_rollback
from ..mlx_lm_mtp.deepseek_v4_dspark import (
    _PRIME_CTX_ATTR,
    DSparkContextCache,
    capture_prompt,
    take_primed,
)
from ..mlx_lm_mtp.prompt_priming import owned_capture

_VERIFY_ROWS = 6


class DraftContext(list):
    def __init__(self, stages, window):
        super().__init__(DSparkContextCache(window) for _ in range(stages))
        self.taps = deque(maxlen=window)


class DSparkHost:
    """Lightning MTP hooks of the ERNIE language model.

    Each request verifies its block on its own caches, alone or in one forward
    with other requests' blocks. Every verify row attends as its own one-token
    decode step and the other ops compute each row independently, so greedy
    output with MTP equals decoding without it.
    """

    _omlx_mtp_multi_request = True
    _omlx_mtp_independent_verify = True

    @property
    def args(self):
        return self.config

    def configure_mtp(self, enabled, depth=7):
        enabled = bool(enabled) and "dspark" in self
        self._omlx_mtp_decode_enabled = enabled
        self._omlx_dspark_decode_enabled = enabled
        self._omlx_mtp_chain = True
        self._omlx_mtp_depth = max(1, min(depth, self.config.dspark_block_size))

    def get_mtp_module(self):
        return self.dspark if self._omlx_mtp_decode_enabled else None

    def make_mtp_cache(self):
        return DraftContext(len(self.dspark.layers), self.config.sliding_window)

    def capture_draft_context(self, rows, taps, cache):
        if rows.shape[1] == 1 and self._keep_draft_tap(taps, cache):
            return
        owned_capture(self, self._settle_draft_taps)
        capture_prompt(self, rows, taps, cache, schedule=False)

    def _keep_draft_tap(self, tap, cache):
        offset = mx.array(cache[0][0].offset)
        kept = []

        def keep():
            ctx = getattr(self, _PRIME_CTX_ATTR, None)
            if ctx is not None:
                ctx.caches.taps.append((tap, offset))
                kept.append(ctx)

        owned_capture(self, keep)
        return bool(kept)

    def _settle_draft_taps(self):
        ctx = getattr(self, _PRIME_CTX_ATTR, None)
        end = None if ctx is None else self.dspark.project_taps(ctx.caches)
        if end is not None:
            ctx.expected_target_offset = end

    def make_mtp_depth_controller(self, depth):
        return AcceptanceDepthController(depth)

    def mtp_shared_verify_rows(self):
        return _VERIFY_ROWS

    def mtp_verify_rows_ok(self, row_caches, width):
        return True

    def mtp_verify_rows(self, inputs, row_caches):
        cache_rollback.set_undo_armed(True)
        try:
            out = self(
                inputs,
                cache=[list(layer) for layer in zip(*row_caches)],
                return_hidden=True,
            )
        finally:
            cache_rollback.set_undo_armed(False)
        return out.logits, out.hidden_states[0], row_caches

    def dspark_append_context(self, hidden_rows, cache, *, start_offset=None):
        self.dspark.append_context([hidden_rows], [cache], start_offset)

    def dspark_forward(self, hidden_rows, anchor, cache=None, *, draft_length=None):
        head = self.dspark_hidden_rows(
            [hidden_rows], [anchor], [cache], draft_length=draft_length
        )
        return self.dspark_logits(head), head

    def dspark_hidden_rows(self, hidden_rows, anchors, caches, *, draft_length=None):
        self.dspark.append_context(hidden_rows, caches)
        width = draft_length or self.config.dspark_block_size
        anchors = mx.concatenate(anchors)
        return self.dspark.block(self.model.embed_tokens, anchors, caches, width)

    def dspark_logits(self, hidden):
        return self.dspark.logits(hidden)

    def dspark_step_logits(self, block, index, previous):
        bias, _ = self.dspark.markov_head(previous)
        return self.multimax(block[:, index] + bias)

    def mtp_take_primed(self, cache, main_token):
        self._settle_draft_taps()
        return take_primed(self, cache, main_token)

    def mtp_partial_rollback(self, cache, accepted, num_drafts):
        for layer, c in zip(self.layers, cache):
            layer.attn.rollback(c, accepted + 1, num_drafts - accepted)
        return True
