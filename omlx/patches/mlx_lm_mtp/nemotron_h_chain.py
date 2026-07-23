# SPDX-License-Identifier: Apache-2.0
"""Depth-k Lightning MTP chain enablement for nemotron_h.

Layered on top of nemotron_h_model (the PR-990 depth-1 base patch). Ports
the qwen35_model chain contract to the Nemotron-H hybrid trunk:

- ``Mixer.__call__`` (verify windows, ``0 < n_confirmed < S``): run the whole
  window as ONE pristine chunk, keep zero-copy *pre-forward* (conv, ssm)
  state refs in ``cache.rollback_state`` and the raw mixer input in
  ``cache._mtp_draft_stash``. Nothing extra is paid on full accepts.
- ``Model.mtp_forward``: adds ``return_hidden`` / ``logits_keep`` (chain
  drafting interface). Legacy 3-positional calls behave as before.
- ``Model.mtp_partial_rollback``: after a depth-k verify over
  ``[confirmed, d1..dk]`` with ``m`` accepted drafts — attention KV layers
  trim ``k - m``; Mamba layers restore the pre-forward refs and replay the
  kept ``1 + m`` tokens from the stash through the pristine mixer forward
  (a handful of tiny kernels, paid only on rejections). The replay is
  bit-identical to the original forward for those positions (same stock
  ssm path, same single-chunk math).
- Stamps ``_omlx_mtp_chain`` / ``_omlx_mtp_depth`` (from ``get_mtp_depth()``,
  i.e. the ``mtp_num_draft_tokens`` model setting; nemotron_h defaults to a
  fixed depth-1 cycle — the stock head is depth-1 trained) on MTP-bearing
  instances, plus ``_omlx_mtp_head_hidden_normed`` — nemotron's
  ``return_hidden`` hidden is already post-``norm_f``, so
  ``_trunk_norm_module`` returns identity for this model (qwen returns
  pre-norm).

Greedy speculative decoding is lossless: outputs are identical to the
depth-1/no-MTP greedy stream; only throughput changes.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_MARKER = "_omlx_nh_chain"


def _is_ours(cls, attr):
    fn = cls.__dict__.get(attr)
    return getattr(fn, _MARKER, False)


def apply() -> bool:
    try:
        from mlx_lm.models import nemotron_h as nh
    except Exception:
        logger.debug("mlx_lm nemotron_h not importable; chain patch skipped")
        return False

    # Require the base MTP patch (provides n_confirmed plumbing + MTPModule).
    if not getattr(nh.Model, "_omlx_nh_mtp_init", False):
        logger.debug("nemotron_h base MTP patch missing; chain patch skipped")
        return False

    _patch_mixer(nh)
    _patch_ssm_sequential()
    _patch_conv_capture(nh)
    _patch_mtp_forward(nh)
    _patch_partial_rollback(nh)
    _patch_init_markers(nh)
    logger.info(
        "nemotron_h MTP chain patch applied "
        "(depth-k drafting, sequential fused verify, replay-free rollback)"
    )
    return True


# --------------------------------------------------------------------------- #
# Mixer: chain verify semantics.
#
# Verify windows keep the BATCHED projections (in_proj/conv/out_proj stream
# each layer's weights once for the whole window) but run the SSM scan
# SEQUENTIALLY with the fused single-step decode kernel:
#   - measured faster than ssm_attn's chunked graph at window sizes 2-8
#     (0.35 vs 0.63 ms/layer at S=2 -> ~11 ms/cycle across 40 layers), and
#   - numerically identical to the plain decode path (same kernel), and
#   - yields PER-POSITION (conv, ssm) states as a free byproduct, so a
#     partial rollback is a pure ref restore — no replay, no re-streaming.
# The per-position capture is armed around the window call; if anything
# about the call shape is unexpected (mask set, no state, capture missed a
# layer) the code falls back to the raw-input stash + replay path.
# Thread-local: two models decoding concurrently in one process must not
# interleave arm/read across threads.
# --------------------------------------------------------------------------- #
class _Capture(threading.local):
    def __init__(self):
        self.armed = False
        self.ssm = None
        self.conv = None


_capture = _Capture()


def _patch_mixer(nh):
    Mixer = nh.NemotronHMamba2Mixer
    if _is_ours(Mixer, "__call__"):
        return
    base_call = Mixer.__call__  # the base-patched version (two-chunk split)

    def __call__(self, hidden_states, mask, cache=None, n_confirmed=0):
        S = hidden_states.shape[1]
        if cache is not None and 0 < n_confirmed < S:
            conv0, ssm0 = cache[0], cache[1]  # zero-copy pre-forward refs
            _capture.armed = mask is None
            _capture.ssm = None
            _capture.conv = None
            try:
                out = base_call(self, hidden_states, mask, cache)
            finally:
                armed = _capture.armed
                ssm_states = _capture.ssm
                conv_tails = _capture.conv
                _capture.armed = False
                _capture.ssm = None
                _capture.conv = None
            if (
                armed
                and ssm_states is not None
                and conv_tails is not None
                and len(ssm_states) == S
                and len(conv_tails) == S
            ):
                cache._mtp_pos_states = list(zip(conv_tails, ssm_states))
            else:
                cache._mtp_pos_states = None
            # Raw-input stash: replay fallback when per-position capture
            # didn't engage (masked window, exotic path).
            cache.rollback_state = (conv0, ssm0)
            cache._mtp_draft_stash = hidden_states
            return out
        return base_call(self, hidden_states, mask, cache)

    __call__._omlx_nh_mtp = True   # base patch idempotency: don't re-wrap
    setattr(__call__, _MARKER, True)
    Mixer.__call__ = __call__


_ssm_seq_applied = False


def _patch_ssm_sequential():
    """Wrap mlx_lm ssm_update: on armed chain-verify windows run the fused
    single-step kernel per position, capturing each state.

    Layering note: the SSD prefill patch also wraps ssm_update. Both wrappers
    fall through cleanly (SSD only claims T>=32 unmasked prefill; this one
    only claims armed tiny windows), so either application order works. A
    module flag — not a function attribute — guards idempotency, because a
    later SSD re-wrap would hide the attribute.
    """
    global _ssm_seq_applied
    import mlx.core as mx
    from mlx_lm.models import ssm as ssm_mod

    if _ssm_seq_applied:
        return
    _ssm_seq_applied = True
    inner = ssm_mod.ssm_update  # whatever is installed (incl. SSD prefill wrap)
    kernel_step = ssm_mod.ssm_update_kernel

    def ssm_update(
        hidden_states,
        A_log,
        B,
        C,
        D,
        dt,
        dt_bias,
        state=None,
        time_step_limit=(0.001, 100.0),
        mask=None,
        lengths=None,
    ):
        S = hidden_states.shape[1]
        if (
            _capture.armed
            and 1 < S < 32
            and state is not None
            and mask is None
            and lengths is None
            and mx.default_device() == mx.gpu
        ):
            ys = []
            states = []
            s = state
            for i in range(S):
                y, s = kernel_step(
                    hidden_states[:, i : i + 1],
                    A_log,
                    B[:, i],
                    C[:, i],
                    D,
                    dt[:, i : i + 1],
                    dt_bias,
                    s,
                    time_step_limit,
                )
                ys.append(y)
                states.append(s)
            _capture.ssm = states
            return mx.concatenate(ys, axis=1), s
        return inner(
            hidden_states, A_log, B, C, D, dt, dt_bias, state,
            time_step_limit, mask, lengths,
        )

    setattr(ssm_update, _MARKER, True)
    ssm_mod.ssm_update = ssm_update
    try:
        from mlx_lm.models import nemotron_h as nh_mod

        nh_mod.ssm_update = ssm_update
    except Exception:
        pass


def _patch_conv_capture(nh):
    """Wrap Mixer._conv: on armed windows also record the per-position conv
    cache tails (last conv_kernel-1 columns of [prev_tail | window input])."""
    import mlx.core as mx

    Mixer = nh.NemotronHMamba2Mixer
    if getattr(Mixer.__dict__.get("_conv"), _MARKER, False):
        return
    orig_conv = Mixer._conv

    def _conv(self, conv_input, cache=None, mask=None):
        if (
            _capture.armed
            and cache is not None
            and mask is None
            and getattr(cache, "lengths", None) is None
            and conv_input.shape[1] > 1
        ):
            n_keep = self.conv_kernel_size - 1
            prev = cache[0]
            if prev is None:
                prev = mx.zeros(
                    (conv_input.shape[0], n_keep, self.conv_dim),
                    dtype=conv_input.dtype,
                )
            S = conv_input.shape[1]
            tails = []
            for i in range(S):
                seq = mx.concatenate([prev, conv_input[:, : i + 1]], axis=1)
                tails.append(seq[:, -n_keep:, :])
            _capture.conv = tails
        return orig_conv(self, conv_input, cache, mask)

    setattr(_conv, _MARKER, True)
    Mixer._conv = _conv


# --------------------------------------------------------------------------- #
# Model.mtp_forward: return_hidden / logits_keep (chain drafting interface)
# --------------------------------------------------------------------------- #
def _patch_mtp_forward(nh):
    Model = nh.Model
    if _is_ours(Model, "mtp_forward"):
        return

    def mtp_forward(
        self, hidden, next_ids, mtp_cache, return_hidden=False, logits_keep=None
    ):
        out = self.mtp(hidden, next_ids, self.backbone.embeddings, mtp_cache)
        sliced = out[:, -logits_keep:] if logits_keep else out
        logits = self.lm_head(sliced)
        if return_hidden:
            return logits, out
        return logits

    mtp_forward._omlx_nh_mtp = True
    setattr(mtp_forward, _MARKER, True)
    Model.mtp_forward = mtp_forward


# --------------------------------------------------------------------------- #
# Model.mtp_partial_rollback: trim KV, restore+replay Mamba state
# --------------------------------------------------------------------------- #
def _patch_partial_rollback(nh):
    Model = nh.Model
    if _is_ours(Model, "mtp_partial_rollback"):
        return

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        layers = self.backbone.layers
        trim_n = num_drafts - accepted
        keep = 1 + accepted  # confirmed token + accepted drafts

        # Validate everything before mutating anything (partial rollbacks
        # desync per-layer cache lengths and corrupt later forwards).
        plan = []
        cur = 0
        for layer in layers:
            bt = layer.block_type
            if bt == "M":
                if cur >= len(cache):
                    return False
                c = cache[cur]
                cur += 1
                if trim_n > 0 and (
                    getattr(c, "rollback_state", None) is None
                    or getattr(c, "_mtp_draft_stash", None) is None
                ):
                    return False
                plan.append(("M", layer, c))
            elif bt == "*":
                if cur >= len(cache):
                    return False
                c = cache[cur]
                cur += 1
                if trim_n > 0 and not (
                    hasattr(c, "is_trimmable") and c.is_trimmable()
                ):
                    return False
                plan.append(("*", layer, c))
            # 'E' / '-' blocks are stateless: no cache entry.
        if cur != len(cache):
            return False

        if trim_n <= 0:
            # Full accept: nothing to undo; drop the stashes so the refs
            # don't pin the pre-forward arrays.
            for kind, _, c in plan:
                if kind == "M":
                    c.rollback_state = None
                    c._mtp_draft_stash = None
                    c._mtp_pos_states = None
            return True

        for kind, layer, c in plan:
            if kind == "M":
                pos = getattr(c, "_mtp_pos_states", None)
                if pos is not None and len(pos) >= keep:
                    # Fast path: the sequential verify captured per-position
                    # (conv, ssm) states — restore refs, no recompute at all.
                    conv_m, ssm_m = pos[keep - 1]
                    c[0] = conv_m
                    c[1] = ssm_m
                else:
                    # Fallback: restore pre-window refs and replay the kept
                    # prefix through the pristine mixer forward.
                    conv0, ssm0 = c.rollback_state
                    stash = c._mtp_draft_stash
                    c[0] = conv0
                    c[1] = ssm0
                    replay = stash[:, :keep]
                    c.rollback_state = None
                    c._mtp_draft_stash = None
                    c._mtp_pos_states = None
                    layer.mixer(replay, None, c)
                    continue
                c.rollback_state = None
                c._mtp_draft_stash = None
                c._mtp_pos_states = None
            else:
                c.trim(trim_n)
        return True

    setattr(mtp_partial_rollback, _MARKER, True)
    Model.mtp_partial_rollback = mtp_partial_rollback


# --------------------------------------------------------------------------- #
# __init__ wrap: stamp chain markers on MTP-bearing instances
# --------------------------------------------------------------------------- #
def _patch_init_markers(nh):
    Model = nh.Model
    if getattr(Model, "_omlx_nh_chain_init", False):
        return
    orig_init = Model.__init__

    def __init__(self, args):
        orig_init(self, args)
        if hasattr(self, "mtp"):
            from . import get_mtp_depth

            self._omlx_mtp_chain = True
            self._omlx_mtp_depth = get_mtp_depth()
            # return_hidden hidden is post-norm_f already: the chain's
            # trunk-norm hook must be identity for this model.
            self._omlx_mtp_head_hidden_normed = True

    Model.__init__ = __init__
    Model._omlx_nh_chain_init = True
