# SPDX-License-Identifier: Apache-2.0
"""Streaming fused-QKV support for MiMo V2.5 FP8 checkpoints.

MiMo V2.5 stores attention as a single fused ``qkv_proj.weight`` that is
already sharded for tensor parallelism, plus a block-128 ``weight_scale_inv``
companion. The model has no fused ``qkv_proj`` module: ``Model.sanitize``
dequantizes the fused tensor per shard and splits it into
``q_proj``/``k_proj``/``v_proj``.

oQ's streaming quantizer cannot rely on that sanitize step. It discovers the
key mapping by running sanitize over recorder proxies, and that recorder can
only replay single-source unary ops (reshape/slice/astype/...). The fused
dequant needs ``mx.from_fp8``, per-shard ``mx.pad``, and a *binary* multiply
against the scale tensor. The recorder does not merely fail to record the
multiply, it silently omits it, so feeding the scale through sanitize would
yield a plan that quantizes unscaled weights.

So the split moves one layer down, to where the weight and its scale are both
in hand: this module registers *virtual* ``q_proj``/``k_proj``/``v_proj``
tensors on the lazy index, hiding the fused weight and its scale. Sanitize
then sees the split tensors already present, its fused branch is inert, and
the plan records three ordinary passthrough entries. The real dequant happens
on materialization, entirely outside the recorder.

The geometry and the arithmetic come from :mod:`fused_qkv_layout`, which the
vendored model also drives, so there is exactly one implementation.
"""

from __future__ import annotations

import importlib
import logging
import weakref

import mlx.core as mx

from .fused_qkv_layout import (
    FUSED_QKV_BLOCK_SIZE,
    detect_fused_qkv_tp,
    fused_qkv_keys,
    fused_qkv_layout_matches,
    fused_qkv_split_shapes,
    layer_head_geometry,
    split_fused_qkv,
)

logger = logging.getLogger(__name__)

_MODULE_NAME = "mlx_lm.models.mimo_v2"

_PARTS = ("q_proj", "k_proj", "v_proj")


def _model_args_class():
    """The ``ModelArgs`` for MiMo, from whichever module provides mimo_v2.

    Only ``ModelArgs`` is read from that module. Everything about the fused
    layout lives in :mod:`fused_qkv_layout`, so this keeps working when
    upstream mlx-lm ships its own mimo_v2 and the vendored copy is no longer
    what the name resolves to.
    """
    from . import apply_mimo_v2_patch

    apply_mimo_v2_patch()
    return importlib.import_module(_MODULE_NAME).ModelArgs


def _is_fused_qkv_mimo(config) -> bool:
    if str(config.get("model_type", "")).lower() != "mimo_v2":
        return False
    layout = config.get("attention_projection_layout")
    # Absent on older configs; the tensor scan below is the real gate.
    return layout is None or str(layout).lower() == "fused_qkv"


class _ShardedQKVSplitter:
    """Dequantize one fused qkv at a time and hand out its three slices.

    The plan visits ``q_proj``, ``k_proj`` and ``v_proj`` of a layer
    consecutively, so a single-entry cache lets all three share one dequant.
    Each slice is dropped from the cache as it is handed over, and the entry
    is discarded once all three have been served, so at most one layer's
    unserved attention weights stay resident.

    The index is held weakly on purpose. This object is reachable *from* the
    index (through the materializer closures stored in its virtual table), so
    a strong reference would form a cycle and defeat the deliberate
    ``del all_weights`` / ``mx.clear_cache()`` release in the streaming loop,
    leaving the whole header index for the garbage collector to find later.
    """

    def __init__(self, index):
        self._index_ref = weakref.ref(index)
        self._key = None
        self._parts = None
        self._served = set()
        self._evicted = set()

    def _index(self):
        index = self._index_ref()
        if index is None:
            raise RuntimeError(
                "fused-QKV materializer outlived its tensor index; the index "
                "must stay alive while its virtual tensors are being read"
            )
        return index

    def _release(self) -> None:
        if self._parts is not None:
            self._evicted.add(self._key)
            self._parts = None
            self._key = None
            self._served = set()
            mx.clear_cache()

    def _load(self, qkv_key, scale_key, geometry) -> None:
        if qkv_key in self._evicted:
            logger.warning(
                "MiMo fused-QKV: re-reading %s. q/k/v are no longer visited "
                "consecutively, so every layer now costs three loads and "
                "three dequants instead of one",
                qkv_key,
            )
        index = self._index()
        # load_source returns fully evaluated arrays; no mx.eval needed here.
        qkv_raw = index.load_source(qkv_key)
        scale_raw = index.load_source(scale_key)
        parts = list(split_fused_qkv(qkv_raw, scale_raw, **geometry))
        mx.eval(*parts)
        del qkv_raw, scale_raw
        mx.clear_cache()
        self._key = qkv_key
        self._parts = parts
        self._served = set()

    def part(self, qkv_key, scale_key, geometry, which):
        cached = (
            self._key == qkv_key
            and self._parts is not None
            and self._parts[which] is not None
        )
        if not cached:
            self._release()
            self._load(qkv_key, scale_key, geometry)

        part = self._parts[which]
        # Hand over sole ownership: the consumer quantizes this slice while
        # the next two are still being produced, and holding a second
        # reference would keep it alive across those passes.
        self._parts[which] = None
        self._served.add(which)
        if len(self._served) == len(_PARTS):
            self._release()
        return part

    def materializer(self, qkv_key, scale_key, geometry, which):
        def materialize():
            return self.part(qkv_key, scale_key, geometry, which)

        return materialize


def register(index, config) -> int:
    """Expose fused qkv tensors as virtual split q/k/v on ``index``.

    Returns the number of layers registered (0 when this checkpoint has no
    fused qkv, which is the common case, including oQ outputs and the
    calibration proxy: both inherit the config but ship split tensors).
    """
    if not _is_fused_qkv_mimo(config):
        return 0

    # Candidate keys come from the config rather than a scan of the index.
    # That confines detection to the text backbone's own layers, so the MTP
    # head (``model.mtp.layers.0.*``, which reuses SWA geometry while sitting
    # at layer index 0) and the vision/audio towers can never reach the
    # geometry check below. Scanning plus a layer-index regex would misread
    # those and abort the run over tensors sanitize is about to discard.
    n_layers = int(config.get("num_hidden_layers") or 0)
    if n_layers <= 0:
        return 0

    candidates = []
    for layer_idx in range(n_layers):
        prefix, qkv_key, scale_key = fused_qkv_keys(layer_idx)
        qkv_shape = index.source_shape(qkv_key)
        scale_shape = index.source_shape(scale_key)
        if qkv_shape is not None and scale_shape is not None:
            candidates.append(
                (layer_idx, prefix, qkv_key, scale_key, qkv_shape, scale_shape)
            )
    if not candidates:
        return 0

    args = _model_args_class().from_dict(config)
    tp = detect_fused_qkv_tp(args, index.source_shape)
    splitter = _ShardedQKVSplitter(index)

    for layer_idx, prefix, qkv_key, scale_key, qkv_shape, scale_shape in candidates:
        n_h, n_kv, hd, vhd = layer_head_geometry(args, layer_idx)

        # Refuse rather than guess: the declared geometry must reproduce the
        # on-disk shapes exactly, or we do not understand this checkpoint and
        # must not dequantize it.
        if not fused_qkv_layout_matches(
            n_h, n_kv, hd, vhd, tp, qkv_shape, scale_shape
        ):
            raise ValueError(
                f"fused-qkv geometry mismatch for {qkv_key}: weight rows="
                f"{qkv_shape[0]}, scale rows={scale_shape[0]} "
                f"(block {FUSED_QKV_BLOCK_SIZE}) do not match the TP={tp} "
                f"layout implied by the config for layer {layer_idx}"
            )

        shapes = fused_qkv_split_shapes(n_h, n_kv, hd, vhd, tp, qkv_shape[1])
        geometry = {"tp": tp, "n_h": n_h, "n_kv": n_kv, "hd": hd, "vhd": vhd}
        for which, (part, shape) in enumerate(zip(_PARTS, shapes)):
            index.register_virtual(
                f"{prefix}.{part}.weight",
                shape,
                "BF16",
                splitter.materializer(qkv_key, scale_key, geometry, which),
                hides=(qkv_key, scale_key),
            )

    logger.info(
        "MiMo fused-QKV: %d layers exposed as streaming split q/k/v (TP=%d)",
        len(candidates),
        tp,
    )
    return len(candidates)
