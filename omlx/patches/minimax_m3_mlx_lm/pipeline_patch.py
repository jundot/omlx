# SPDX-License-Identifier: Apache-2.0
"""Teach MiniMax-M3 to run its layers across more than one Mac.

Spreading the weight files over two Macs achieves nothing on its own. The
vendored model's forward is ``for layer in self.layers`` — every layer it has,
locally, with no notion of another machine. Give a rank only its own shards and
it fails on the tensors it still tries to build; mlx-lm refuses earlier than
that, with *"The model does not support pipelining but a pipeline_group was
provided"*.

Three things are needed, and mlx-lm already defines what they look like:

1. ``pipeline(group)`` — blank the layers this rank does not own so they are
   never built or loaded. ``PipelineMixin`` does exactly this and is reused
   rather than reimplemented.
2. A forward that walks only ``start_idx .. start_idx + num_layers``.
3. ``recv_like`` / ``send`` at the stage boundary, so activations cross the
   wire once per token instead of every layer.
4. A ``make_cache`` over the stage. mlx-lm builds a prompt cache the moment the
   model is loaded, and the vendored one walks the blanked list from (1) — so
   without this no rank whose stage starts past layer 0 ever reaches ready.

Ranks are numbered in reverse: rank 0 holds the *last* layers, so execution
flows from the highest rank down to rank 0, which produces the output. That is
``PipelineMixin``'s convention and the planner's assignment follows it.

Applied by monkey-patching the vendored class rather than editing the vendor
copy, so the vendored tree stays a faithful mirror of upstream and this stays
one deletable file if MiniMax gains pipelining upstream.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

# The one implementation of "keep the send in the lazy graph", shared with the
# Nemotron-H pipeline path. Duplicating it is how the MiniMax port acquired
# ``cache[-1].keys``, which does not exist on MiniMax's second cache family.
from omlx.cluster.pipeline_compat import _cache_dependency

logger = logging.getLogger(__name__)

# The layer range this process was assigned by the planner, if any.
#
# ``PipelineMixin.pipeline(group)`` derives the split from rank and world size
# alone — 60 layers across 2 Macs is always 30/30. It never sees the plan, so
# an uneven split chosen to protect a workstation is silently discarded. That
# is not academic: a MacBook planned for 14 layers (56 GiB) loaded 30 layers
# (112 GiB) against a 109 GiB addressable limit, and died.
#
# A process global because mlx-lm constructs the model inside ``load()`` and
# calls ``pipeline(group)`` itself — there is no instance to configure first.
_ASSIGNED_STAGE: tuple[int, int] | None = None


def set_assigned_stage(start_layer: int, end_layer: int) -> None:
    """Pin the layer range this rank must hold, overriding the even split."""

    global _ASSIGNED_STAGE
    _ASSIGNED_STAGE = (int(start_layer), int(end_layer))


def clear_assigned_stage() -> None:
    global _ASSIGNED_STAGE
    _ASSIGNED_STAGE = None


def assigned_stage() -> tuple[int, int] | None:
    return _ASSIGNED_STAGE


def effective_stage(
    layer_count: int, rank: int, world_size: int
) -> tuple[int, int]:
    """The range a rank will actually hold, assignment or not.

    Mirrors ``PipelineMixin``'s reverse split when nothing is pinned, so a
    memory guard can check what will really be loaded rather than what was
    planned.
    """

    if _ASSIGNED_STAGE is not None:
        return _ASSIGNED_STAGE
    per_rank = layer_count // max(1, world_size)
    extra = layer_count - per_rank * max(1, world_size)
    if rank < extra:
        per_rank += 1
    start = (world_size - rank - 1) * per_rank
    return start, start + per_rank


def _keep_send_in_graph(cache_entry: Any, value: Any) -> None:
    """Anchor a pipeline send in the lazy graph, whichever cache this layer has.

    Without the dependency the KV writes and the send can be reordered and the
    peer reads a stale activation.

    MiniMax-M3 has two cache families. Sparse-index layers get a
    ``MiniMaxM3KVCache``, which owns no ``keys`` of its own — it wraps a plain
    ``KVCache`` in ``.kv_cache``, the idiom the vendored code itself uses. The
    port of this line from ``deepseek_v32`` reached for ``cache[-1].keys``
    directly, so every rank except rank 0 raised ``AttributeError`` on its
    first token: with the default sparse frequency the *last* layer of a
    non-final stage is always a sparse one.
    """

    if cache_entry is None:
        return
    _cache_dependency(getattr(cache_entry, "kv_cache", cache_entry), value, mx)


def _pipelined_make_cache(self: Any) -> list:
    """``LanguageModel.make_cache`` for a rank that holds a slice of the layers.

    The vendored version walks ``self.layers`` — the list ``pipeline()`` pads
    with ``None`` for the layers this rank does not own, to keep the global
    numbering weight loading needs. Every rank whose stage does not start at
    layer 0 therefore raised ``AttributeError: 'NoneType' has no attribute
    'self_attn'`` inside ``make_prompt_cache`` during ``load_default()`` — and
    in the reverse ordering that is always rank 0, the rank that serves HTTP,
    so a two-Mac deployment never reached ready.

    Walking ``pipeline_layers`` instead gives a list exactly ``num_layers``
    long in *local* order — the order ``_pipelined_call`` indexes it in. That
    ordering is the point: skipping the ``None`` entries would leave a list of
    global length whose entries no longer line up with the layers, handing a
    sparse-index layer a plain ``KVCache``. It has no ``update_index_and_fetch``,
    so ``use_sparse_mask`` goes False and the model silently degrades to dense
    attention past the sparse threshold. Wrong output, no error.
    """

    from mlx_vlm.models.minimax_m3_vl.language import KVCache, MiniMaxM3KVCache

    inner = self.model
    layers = inner.pipeline_layers
    if any(layer is None for layer in layers):
        raise RuntimeError(
            "MiniMax-M3 pipeline stage "
            f"{inner.start_idx}-{inner.end_idx} contains an unbuilt layer; "
            "the cache cannot be typed from it."
        )
    return [
        MiniMaxM3KVCache() if layer.self_attn.has_sparse_index else KVCache()
        for layer in layers
    ]


def _pipelined_call(
    self: Any,
    inputs: mx.array,
    inputs_embeds: mx.array | None = None,
    mask: mx.array | None = None,
    cache: Any = None,
    capture_layer_ids: list[int] | None = None,
    hidden_sink: list | None = None,
    position_ids: mx.array | None = None,
) -> mx.array:
    """The vendored forward, walking only this rank's layers.

    Keeps the vendored signature exactly: the mlx-vlm tree calls this with
    ``position_ids`` and the capture hooks, and dropping them would break
    single-node serving, which shares this same class.
    """

    h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

    pipeline_rank = getattr(self, "pipeline_rank", 0)
    pipeline_size = getattr(self, "pipeline_size", 1)
    start_idx = getattr(self, "start_idx", 0)
    # Number of layers this rank actually holds. On a single node this is all
    # of them, so the un-pipelined path is unchanged.
    num_layers = getattr(self, "num_layers", None)
    if num_layers is None:
        num_layers = len(self.layers) - start_idx

    if cache is None:
        cache = [None] * num_layers
    if mask is None:
        # Imported here, not at module scope: the vendored module is only
        # importable once the mlx-vlm compat patch has run.
        from mlx_vlm.models.minimax_m3_vl.language import create_attention_mask

        mask = create_attention_mask(
            h, cache[0] if cache and cache[0] is not None else cache
        )

    # Receive from the rank holding the preceding layers. Embedding above ran
    # only to give recv_like a shape to fill; its value is discarded here.
    if pipeline_rank < pipeline_size - 1:
        h = mx.distributed.recv_like(h, (pipeline_rank + 1))

    capture_set = set(capture_layer_ids) if capture_layer_ids else set()
    for i in range(num_layers):
        layer = self.layers[start_idx + i]
        h = layer(h, mask, cache[i], position_ids=position_ids)
        if hidden_sink is not None and (start_idx + i) in capture_set:
            hidden_sink.append(h)

    # Hand off to the rank holding the following layers.
    if pipeline_rank != 0:
        h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
        if cache:
            _keep_send_in_graph(cache[-1], h)

    # Only rank 0 computed the final hidden state; every rank needs it to
    # return logits, so it is gathered rather than left on one node.
    if pipeline_size > 1:
        h = mx.distributed.all_gather(h)[: h.shape[0]]

    if hidden_sink is not None and not capture_set:
        hidden_sink.append(h)

    return self.norm(h)


def apply_minimax_m3_pipeline_patch() -> bool:
    """Give the vendored MiniMax model mlx-lm's pipelining contract."""

    try:
        from mlx_lm.models.pipeline import PipelineMixin
        from mlx_vlm.models.minimax_m3_vl.language import (
            LanguageModel,
            MiniMaxM3Model,
        )
    except ImportError as exc:
        logger.warning("Cannot pipeline MiniMax-M3: %s", exc)
        return False

    if getattr(MiniMaxM3Model, "_omlx_pipelined", False):
        return True

    def pipeline(self: Any, group: Any) -> None:
        """Take this rank's stage — the planner's, when it named one.

        Falls back to ``PipelineMixin``'s even split when nothing is pinned, so
        behaviour outside oMLX's launcher is unchanged.
        """

        assigned = assigned_stage()
        if assigned is None:
            PipelineMixin.pipeline(self, group)
        else:
            start, end = assigned
            self.pipeline_rank = group.rank()
            self.pipeline_size = group.size()
            self.start_idx = max(0, min(start, len(self.layers)))
            self.end_idx = max(self.start_idx, min(end, len(self.layers)))
            self.layers = self.layers[: self.end_idx]
            # Keep the layer numbers stable so weight loading still matches.
            self.layers[: self.start_idx] = [None] * self.start_idx
        self.num_layers = len(self.layers) - self.start_idx
        logger.info(
            "MiniMax-M3 rank %s holds layers %s-%s (%s of them)",
            self.pipeline_rank, self.start_idx, self.end_idx, self.num_layers,
        )

    # Architecture capability contract consumed by the worker's pre-load
    # guard. This method reads assigned_stage() itself, so an uneven plan is
    # the range the loader will actually construct.
    pipeline._omlx_honors_pipeline_assignment = True
    MiniMaxM3Model.pipeline = pipeline
    MiniMaxM3Model.pipeline_layers = PipelineMixin.pipeline_layers
    MiniMaxM3Model.__call__ = _pipelined_call
    # The cache has to be built from the layers this rank holds, not from the
    # None-padded list. Unpipelined this is the identical list, so single-node
    # serving — which runs this same class — is unchanged.
    LanguageModel.make_cache = _pipelined_make_cache
    # Defaults for the single-node path, which constructs the class directly
    # and never calls pipeline().
    MiniMaxM3Model.pipeline_rank = 0
    MiniMaxM3Model.pipeline_size = 1
    MiniMaxM3Model.start_idx = 0
    MiniMaxM3Model.end_idx = None
    MiniMaxM3Model.num_layers = None
    MiniMaxM3Model._omlx_pipelined = True
    logger.info("MiniMax-M3 pipelining installed")
    return True
