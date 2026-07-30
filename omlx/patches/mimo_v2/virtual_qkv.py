# SPDX-License-Identifier: Apache-2.0
"""Streaming fused-QKV support for MiMo V2.5 FP8 checkpoints.

MiMo V2.5 FP8 checkpoints store attention projections as a single fused
``qkv_proj.weight`` that is already sharded for tensor parallelism, plus a
block-128 ``weight_scale_inv`` companion. The model itself has no fused
``qkv_proj`` module: ``Model.sanitize`` dequantizes the fused tensor per shard
and splits it into ``q_proj``/``k_proj``/``v_proj``.

oQ's streaming quantizer cannot rely on that sanitize step. It discovers the
key mapping by running sanitize over recorder proxies, and that recorder can
only replay single-source unary ops (reshape/slice/astype/...). The fused
dequant needs ``mx.from_fp8``, per-shard ``mx.pad``, and a *binary* multiply
against the scale tensor — none of which the recorder can carry. Feeding the
scale through sanitize therefore either crashes at replay or, worse, silently
drops the scale multiply.

So the split is moved one layer down, to where the weight and its scale are
both in hand: this module registers *virtual* ``q_proj``/``k_proj``/``v_proj``
tensors on the lazy index, hiding the fused weight and its scale. Sanitize
then sees the split tensors already present, its fused-qkv branch is inert,
and the plan records three ordinary passthrough entries. The real dequant
happens on materialization, entirely outside the recorder.

The geometry and the arithmetic both come from the vendored model module, so
there is exactly one implementation of this layout. See
``mimo_v2_model.py`` (``split_fused_qkv`` and friends).
"""

from __future__ import annotations

import importlib
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_MODULE_NAME = "mlx_lm.models.mimo_v2"

_PARTS = ("q_proj", "k_proj", "v_proj")


def _model_module():
    """Import the vendored MiMo module, registering the patch if needed.

    Imports the *registered* ``mlx_lm.models.mimo_v2`` rather than the file
    next door: loading the same source twice would produce two unrelated
    ``ModelArgs`` classes and two copies of the layout constants.
    """
    from . import apply_mimo_v2_patch

    apply_mimo_v2_patch()
    return importlib.import_module(_MODULE_NAME)


def _is_fused_qkv_mimo(config) -> bool:
    if str(config.get("model_type", "")).lower() != "mimo_v2":
        return False
    layout = config.get("attention_projection_layout")
    # Absent on older configs; the tensor scan below is the real gate.
    return layout is None or str(layout).lower() == "fused_qkv"


class _ShardedQKVSplitter:
    """Dequantize one fused qkv at a time and hand out its three slices.

    The plan visits ``q_proj``, ``k_proj`` and ``v_proj`` of a layer
    consecutively, so a single-entry cache is enough to make all three share
    one dequant. The entry is dropped as soon as all three have been served,
    which keeps at most one layer's worth of bf16 attention weights resident.
    """

    def __init__(self, index, split_fn):
        self._index = index
        self._split = split_fn
        self._key = None
        self._parts = None
        self._served = set()

    def _release(self) -> None:
        if self._parts is not None:
            self._parts = None
            self._key = None
            self._served = set()
            mx.clear_cache()

    def _parts_for(self, qkv_key, scale_key, geometry):
        if self._key == qkv_key and self._parts is not None:
            return self._parts
        self._release()
        qkv_raw = self._index.load_source(qkv_key)
        scale_raw = self._index.load_source(scale_key)
        mx.eval(qkv_raw, scale_raw)
        parts = self._split(qkv_raw, scale_raw, **geometry)
        mx.eval(*parts)
        del qkv_raw, scale_raw
        mx.clear_cache()
        self._key = qkv_key
        self._parts = parts
        self._served = set()
        return parts

    def materializer(self, qkv_key, scale_key, geometry, which):
        def materialize():
            part = self._parts_for(qkv_key, scale_key, geometry)[which]
            self._served.add(which)
            if len(self._served) == len(_PARTS):
                self._release()
            return part

        return materialize


def register(index, config) -> int:
    """Expose fused qkv tensors as virtual split q/k/v on ``index``.

    Returns the number of layers registered (0 when this checkpoint has no
    fused qkv, which is the common case — including oQ outputs and the
    calibration proxy, both of which inherit the config but ship split
    tensors).
    """
    if not _is_fused_qkv_mimo(config):
        return 0

    # Candidate keys are built from the config rather than scanned out of the
    # index. That is deliberate: it confines detection to the text backbone's
    # own layers, so the MTP head (``model.mtp.layers.0.*``, which reuses SWA
    # geometry while sitting at layer index 0) and the vision/audio towers can
    # never reach the geometry check below. Scanning plus a layer-index regex
    # would misread those and abort the run over tensors sanitize is about to
    # discard.
    n_layers = int(config.get("num_hidden_layers") or 0)
    if n_layers <= 0:
        return 0

    def keys_for(layer_idx):
        prefix = f"model.layers.{layer_idx}.self_attn"
        qkv_key = f"{prefix}.qkv_proj.weight"
        return prefix, qkv_key, f"{qkv_key}_scale_inv"

    candidates = []
    for layer_idx in range(n_layers):
        _, qkv_key, scale_key = keys_for(layer_idx)
        if (
            index.source_shape(qkv_key) is not None
            and index.source_shape(scale_key) is not None
        ):
            candidates.append(layer_idx)
    if not candidates:
        return 0

    module = _model_module()
    args = module.ModelArgs.from_dict(config)
    block = module.FUSED_QKV_BLOCK_SIZE
    tp = module.detect_fused_qkv_tp(args, index.source_shape)
    splitter = _ShardedQKVSplitter(index, module.split_fused_qkv)

    for layer_idx in candidates:
        prefix, qkv_key, scale_key = keys_for(layer_idx)
        qkv_shape = index.source_shape(qkv_key)
        scale_shape = index.source_shape(scale_key)
        n_h, n_kv, hd, vhd = module.layer_head_geometry(args, layer_idx)

        # Refuse rather than guess: the declared geometry must reproduce the
        # on-disk shapes exactly, or we do not understand this checkpoint and
        # must not dequantize it.
        rows, padded_rows = module.fused_qkv_shard_rows(n_h, n_kv, hd, vhd, tp)
        if rows * tp != qkv_shape[0] or padded_rows * tp != scale_shape[0] * block:
            raise ValueError(
                f"fused-qkv geometry mismatch for {qkv_key}: "
                f"weight rows={qkv_shape[0]} scale rows={scale_shape[0]} "
                f"but TP={tp} geometry predicts {rows * tp} / "
                f"{padded_rows * tp // block}"
            )

        shapes = module.fused_qkv_split_shapes(n_h, n_kv, hd, vhd, tp, qkv_shape[1])
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
