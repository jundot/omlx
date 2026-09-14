# SPDX-License-Identifier: MIT
"""DSpark integration with oMLX's shared speculative acceptance loop."""

import mlx.core as mx


class AcceptanceDepthController:
    """Choose verification shapes from token acceptance, never wall-clock time.

    V4.1's quantized backbone can round differently at different row counts.
    Timing-based scheduling would therefore let system load change greedy text.
    A rejection retains one lookahead beyond the accepted prefix; fully accepted
    blocks grow by one until the configured depth is reached.
    """

    def __init__(self, depth):
        self.max_depth = max(1, int(depth))
        self.cur = self.max_depth
        self._warmup = False
        self.exit_streak = 0
        self.t = {}

    def observe(self, used, accepted, cycle_ms, time_sample=True):
        accepted = max(0, min(int(accepted), int(used)))
        self.cur = min(self.max_depth, accepted + 1)

    def should_exit(self):
        # Timing-based handoff would also change the subsequent numeric path.
        return False


class DSparkMixin:
    def _omlx_prefill(self, input_ids, cache=None, **kwargs):
        """Scheduler cache-only entry; normal forward retains full logits."""
        return self(input_ids, cache=cache, _ced_prefill=True, **kwargs)

    @property
    def args(self):
        return self._config

    def configure_mtp(self, enabled, depth=5):
        if enabled and not self._config.preserve_mtp:
            raise ValueError("DSpark decoding requires preserved MTP weights")
        self._omlx_mtp_decode_enabled = bool(enabled)
        self._omlx_dspark_decode_enabled = bool(enabled)
        self._omlx_mtp_chain = True
        self._omlx_mtp_head_clone = False
        self._omlx_mtp_rowwise_unsupported = True
        self._omlx_mtp_depth = max(1, min(depth, self._config.dspark_block_size))

    def make_mtp_depth_controller(self, depth):
        return AcceptanceDepthController(depth)

    def make_mtp_cache(self):
        return self.make_dspark_cache()

    def dspark_append_context(self, main_hidden, cache, *, start_offset=None):
        first = self.mtp[0]
        projected = first.main_norm(first.main_proj(main_hidden))
        for stage, item in zip(self.mtp, cache):
            stage.attn.append_context(projected, item, start_offset=start_offset)

    def dspark_forward(self, main_hidden, anchor_ids, cache=None, *, draft_length=None):
        from .dspark import proposal_forward

        cache = self.make_mtp_cache() if cache is None else cache
        self.dspark_append_context(main_hidden, cache)
        return proposal_forward(self, anchor_ids, cache, draft_length)

    def dspark_markov(self, token_ids):
        return self.mtp[-1].markov_head(token_ids)

    def mtp_forward(
        self, hidden, input_ids, cache=None, return_hidden=False, logits_keep=0
    ):
        logits, states = self.dspark_forward(hidden, input_ids, cache)
        if logits_keep:
            logits, states = logits[:, -logits_keep:], states[:, -logits_keep:]
        return (logits, states) if return_hidden else logits

    def mtp_take_primed(self, cache, main_token):
        from ..mlx_lm_mtp.deepseek_v4_dspark import take_primed

        return take_primed(self, cache, main_token)

    def mtp_partial_rollback(self, cache, accepted, num_drafts):
        if not 0 <= accepted <= num_drafts:
            return False
        stash = getattr(cache[0], "_mtp_draft_stash", None)
        if stash is None:
            return accepted == num_drafts
        inputs, snapshots, before, verify_states = stash
        if inputs.shape[1] != num_drafts + 1:
            return False
        if any(item.size() != before + inputs.shape[1] for item in cache):
            return False
        cache[0]._mtp_draft_stash = None
        if accepted == num_drafts:
            return True
        # Commit the already-computed causal prefix, including projections that
        # became an incomplete compression group after rejecting later tokens.
        count = accepted + 1
        end = before + count
        history = None
        if self._hasher is not None:
            _, history = self._hasher(inputs[:, :count], snapshots[0][0][6])
        for i, (item, snapshot, state) in enumerate(
            zip(cache, snapshots, verify_states)
        ):
            _, padding, lengths = snapshot
            window_end = min(before, self._config.window_size) + count
            item[1] = state["window"][
                :, max(0, window_end - self._config.window_size) : window_end
            ]
            ratio = item.compress_ratio
            if ratio:
                item[2] = item[2][:, : end // ratio]
                item[3] = item[3][:, : end // ratio]
                if ratio > 1:
                    kv, gate = state["compressor"]
                    projected_end = before % ratio + count
                    cutoff = projected_end // ratio * ratio
                    item[4] = kv[:, cutoff:projected_end]
                    item[5] = gate[:, cutoff:projected_end]
            item[0] = mx.array([end], mx.int32)
            if i == 0 and history is not None:
                item[6] = mx.array(history, mx.int64)
            item.left_padding, item.lengths = padding, lengths
            item.advance(count)
        mx.eval([item.state for item in cache])
        return True

    def __call__(
        self, input_ids, cache=None, inputs_embeds=None, token_types=None, **kwargs
    ):
        from ..mlx_lm_mtp.deepseek_v4_dspark import capture_prompt

        cache = self.make_cache() if cache is None else cache
        return_hidden = bool(kwargs.pop("return_hidden", False))
        target_verify = kwargs.pop("target_verify", False)
        n_confirmed = kwargs.pop("n_confirmed", 0)
        verify = bool(target_verify or n_confirmed)
        active = getattr(self, "_omlx_dspark_decode_enabled", False)
        capture_dspark = kwargs.pop("return_dspark_hidden", False)
        capture = return_hidden or bool(capture_dspark)
        if kwargs.get("_ced_prefill", False) and self._config.ced_prefill:
            if input_ids.shape[0] != 1:
                raise ValueError("CED scheduler prefill requires one request row")
            if capture or verify:
                raise ValueError("CED prefill cannot supply full hidden/verify states")
        prime = active and not capture and not verify and input_ids.shape[0] == 1
        verify_states = None
        if verify:
            if input_ids.shape[0] != 1 or inputs_embeds is not None:
                raise ValueError("DSpark verification requires one text decode row")
            before = cache[0].size()
            snapshots = [(list(c.cache), c.left_padding, c.lengths) for c in cache]
            verify_states = [{} for _ in cache]
        result = self._forward(
            input_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            token_types=token_types,
            return_dspark_hidden=capture or prime,
            mtp_verify_states=verify_states,
            **kwargs,
        )
        if verify:
            cache[0]._mtp_draft_stash = (input_ids, snapshots, before, verify_states)
        if prime:
            logits, hidden = result
            # The draft ring retains one window; reset across omitted spans
            # using the existing absolute-position prompt-capture contract.
            capture_prompt(self, input_ids[:, -hidden.shape[1] :], hidden, cache)
            return logits
        return result
