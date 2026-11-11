# SPDX-License-Identifier: Apache-2.0
"""Block-drafter hooks for plain-attention target models.

Qwen3, Qwen3-MoE, Laguna and MiMo V2 (mlx-lm) and Muse Glimmer (mlx-vlm)
checkpoints draft with DFlash on the batched engines. Their forward returns
logits only, so this patch lets one forward also return chosen decoder-layer
outputs (prefill seeding and verify), feeds the drafter from ordinary
one-token steps, and rolls a rejected verify back by trimming the KV caches.

Layer outputs are recorded by wrapping the decoder layer class and running
the model's own forward, so its logits transform (Muse's output scale and
softcap) and any replacement of the inner model keep working. Speculation
stays single-row: rows that accept different counts would need a ragged KV
rollback.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

_CAPTURE = threading.local()


class _Sink:
    """Keeps the outputs of the wanted decoder layers, by call order."""

    def __init__(self, wanted: set[int]):
        self.wanted = wanted
        self.count = 0
        self.outputs: dict[int, Any] = {}

    def add(self, hidden: Any) -> None:
        if self.count in self.wanted:
            self.outputs[self.count] = hidden
        self.count += 1


def record(hidden: Any) -> None:
    """Report a decoder layer output from a loop that skips the layer call.

    MiMo's fused decode loop runs the layer math inline.
    """
    sink = getattr(_CAPTURE, "sink", None)
    if sink is not None:
        sink.add(hidden)


def _wrap_layer(layer_cls: Any) -> None:
    if getattr(layer_cls, "_omlx_drafter_capture_wrapped", False):
        return
    original_call = layer_cls.__call__

    def __call__(self, *args, **kwargs):
        out = original_call(self, *args, **kwargs)
        sink = getattr(_CAPTURE, "sink", None)
        if sink is not None:
            sink.add(out)
        return out

    layer_cls.__call__ = __call__
    layer_cls._omlx_drafter_capture_wrapped = True


def _patch_model(model_cls: Any) -> None:
    if getattr(model_cls, "_omlx_drafter_target", False):
        return
    original_call = model_cls.__call__

    def __call__(
        self,
        inputs=None,
        cache=None,
        *args,
        return_hidden: bool = False,
        n_confirmed: int = 0,
        capture_layer_ids=None,
        **kwargs,
    ):
        drafter = getattr(self, "_omlx_drafter", None)
        scope = getattr(drafter, "scope_uids", None)
        # An ordinary decode step keeps the committed token in the drafter
        # context.
        observe = bool(
            capture_layer_ids is None
            and not return_hidden
            and scope
            and inputs is not None
            and inputs.ndim == 2
            and tuple(inputs.shape) == (len(scope), 1)
        )
        if observe:
            capture_layer_ids = list(drafter.target_layer_ids)
        if not capture_layer_ids and not return_hidden:
            return original_call(self, inputs, cache, *args, **kwargs)
        last = len(self.model.layers) - 1
        sink = _Sink({*(capture_layer_ids or ()), last})
        _CAPTURE.sink = sink
        try:
            out = original_call(self, inputs, cache, *args, **kwargs)
        finally:
            _CAPTURE.sink = None
        captures = [sink.outputs[i] for i in capture_layer_ids or ()]
        if observe:
            drafter.observe(scope, captures)
            return out
        logits = getattr(out, "logits", out)
        if return_hidden:
            # The last layer's output stands in for the head's hidden; block
            # drafters read only the captures.
            return SimpleNamespace(
                logits=logits,
                hidden_states=captures + [sink.outputs[last]],
                gdn_states=None,
            )
        return SimpleNamespace(logits=logits, hidden_states=captures)

    def mtp_partial_rollback(self, cache, accepted: int, num_drafts: int) -> bool:
        rejected = int(num_drafts) - int(accepted)
        if rejected <= 0:
            return True
        return all(c.trim(rejected) == rejected for c in cache if c is not None)

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        # mlx-vlm's drafter bind checks for this hook; one row trims.
        accepted = int(accepted)
        self.mtp_partial_rollback(caches, accepted, int(block_size) - 1)
        return accepted

    model_cls.__call__ = __call__
    # MiMo keeps its own rollback, which checks the caches can trim first.
    if not hasattr(model_cls, "mtp_partial_rollback"):
        model_cls.mtp_partial_rollback = mtp_partial_rollback
    model_cls.rollback_speculative_cache = rollback_speculative_cache
    model_cls._omlx_drafter_single_row = True
    model_cls._omlx_drafter_target = True


def apply(model: Any) -> bool:
    """Install the drafter hooks on ``model``'s classes. Idempotent."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return False
    for layer_cls in {type(layer) for layer in layers}:
        _wrap_layer(layer_cls)
    _patch_model(type(model))
    return True
