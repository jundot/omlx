# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the SSD expert-streaming test domain.

The PR's streaming tests grew by copy-paste; this module single-sources
the pieces several files need:

* ``write_safetensors`` — minimal safetensors writer taking real array
  bytes OR header-only ``(shape, dtype, nbytes)`` specs, plus optional
  ``__metadata__``;
* ``write_moe_checkpoint`` — a header-only qwen-style MoE checkpoint
  (config + zero-pad shard + index) for converter/estimator tests;
* ``quantized_glu_store`` — a real quantized SwitchGLU written under a
  caller-chosen key prefix, plus the ``CheckpointExpertStore`` /
  ``_GLUStoreView`` pair reading it;
* ``v41_backing`` — ``V41StreamingBacking`` assembly incl. the
  ``_expert_streaming_backing`` wiring the engine performs in prod;
* ``bind_streaming_probe`` — the Scheduler LRU heap-growth probe every
  scheduler stand-in must carry since PR #3468;
* ``ensure_spy`` — ``slots.ensure`` row-count recorder;
* ``closer`` — teardown fixture replacing ``try/finally: obj.close()``.
"""

import json
import struct
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

_NP_TO_ST = {
    np.dtype(np.bool_): "BOOL",
    np.dtype(np.uint8): "U8",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint16): "U16",
    np.dtype(np.int16): "I16",
    np.dtype(np.uint32): "U32",
    np.dtype(np.int32): "I32",
    np.dtype(np.uint64): "U64",
    np.dtype(np.int64): "I64",
    np.dtype(np.float16): "F16",
    np.dtype(np.float32): "F32",
    np.dtype(np.float64): "F64",
}
_ST_TO_NP = {v: k for k, v in _NP_TO_ST.items()}


def write_safetensors(path, tensors, metadata=None):
    """Write a minimal safetensors file at ``path``.

    Each value in ``tensors`` may be:

    * an array (numpy or anything ``np.asarray`` consumes) — raw bytes,
      safetensors dtype inferred;
    * ``(array, dtype)`` — raw bytes stamped as ``dtype`` (the array is
      cast when a numpy equivalent exists);
    * ``(shape, dtype, nbytes)`` — a header entry backed by ``nbytes``
      of zeros, enough for header-only readers (``ExpertBackingStore``,
      the conversion estimator) that never touch the payload.

    ``metadata`` lands in the ``__metadata__`` header block (string
    values, the cold-tier label convention).
    """
    header = {}
    if metadata:
        header["__metadata__"] = dict(metadata)
    blob = bytearray()
    for name, spec in tensors.items():
        if isinstance(spec, tuple) and len(spec) == 3:
            shape, dtype, _nbytes = spec
            data = b"\x00" * _nbytes
        elif isinstance(spec, tuple):
            arr, dtype = spec
            np_dtype = _ST_TO_NP.get(dtype)
            if np_dtype is not None:
                arr = np.ascontiguousarray(arr).astype(np_dtype)
            data = np.ascontiguousarray(arr).tobytes()
            shape = arr.shape
        else:
            arr = np.asarray(spec)
            dtype = _NP_TO_ST[arr.dtype]
            data = arr.tobytes()
            shape = arr.shape
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob += data
    hb = json.dumps(header).encode()
    Path(path).write_bytes(struct.pack("<Q", len(hb)) + hb + bytes(blob))


def write_moe_checkpoint(
    path, model_type="qwen4_exp", *, fused=False, layers=2, n_experts=4,
    hidden=32, moe=16,
):
    """Fabricate a header-only MoE checkpoint (config + zero-pad shard
    + index): enough for the conversion estimator and backing-store
    readers, with no real weights materialized."""
    path = Path(path)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "num_hidden_layers": layers,
                "num_experts": n_experts,
                "hidden_size": hidden,
                "moe_intermediate_size": moe,
            }
        )
    )
    projs = (
        [("gate_up_proj", (n_experts, 2 * moe, hidden))]
        if fused
        else [
            ("gate_proj", (n_experts, moe, hidden)),
            ("up_proj", (n_experts, moe, hidden)),
        ]
    ) + [("down_proj", (n_experts, hidden, moe))]
    tensors = {}
    for layer in range(layers):
        for proj, shape in projs:
            key = f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight"
            tensors[key] = (shape, "BF16", int(np.prod(shape)) * 2)
    write_safetensors(path / "model.safetensors", tensors)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )


def quantized_glu_store(
    path, prefix, *, experts=4, dims=32, inter=32, group_size=32, bits=4
):
    """Quantized ``experts``-expert SwitchGLU whose ``prefix``-rooted
    tensors are written to ``path/model.safetensors``, plus the
    ``CheckpointExpertStore``/``_GLUStoreView`` pair reading them.

    Returns ``(glu, store, view)`` — the caller keeps ``store`` alive for
    the cache/view lifetime (register it on ``closer``).
    """
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    from omlx.patches.moe_expert_offload import (
        CheckpointExpertStore,
        _GLUStoreView,
    )

    glu = SwitchGLU(dims, inter, experts)
    nn.quantize(glu, group_size=group_size, bits=bits)
    tensors = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(glu, proj)
        for field in ("weight", "scales", "biases"):
            if lin.get(field) is not None:
                tensors[f"{prefix}.{proj}.{field}"] = lin[field]
    write_safetensors(path / "model.safetensors", tensors)
    store = CheckpointExpertStore(path)
    return glu, store, _GLUStoreView(store, prefix)


def v41_backing(disk, **kwargs):
    """Assemble ``V41StreamingBacking`` over a disk-loaded model's
    offloaded MoE layers and wire ``disk._expert_streaming_backing`` —
    the attribute the engine's hook chain resolves in production, so a
    test exercises the same lookup the scheduler performs."""
    from omlx.patches.deepseek_v41.streaming_backing import V41StreamingBacking

    layers = [
        (i, layer.ffn.experts.slots)
        for i, layer in enumerate(disk.language_model.layers)
    ]
    backing = V41StreamingBacking(disk._moe_offload_plan, layers, **kwargs)
    disk._expert_streaming_backing = backing
    return backing


def bind_streaming_probe(ns):
    """Bind the app-level LRU heap-growth probe a real Scheduler carries.

    PR #3468: ``_record_chunk_transient`` reads the app-level LRU heap
    growth; a stand-in must carry the same probe as a real Scheduler
    (returns 0 with no stashed cache, matching the production default).
    """
    from omlx.scheduler import Scheduler

    ns._streaming_lru_cache = getattr(ns, "_streaming_lru_cache", None)
    ns._streaming_lru_bytes_last = getattr(ns, "_streaming_lru_bytes_last", None)
    ns._streaming_lru_heap_growth = Scheduler._streaming_lru_heap_growth.__get__(
        ns, Scheduler
    )


@contextmanager
def ensure_spy(slots):
    """Wrap ``slots.ensure`` to record each call's row count; the
    original is restored on exit."""
    calls = []
    orig = slots.ensure

    def spy(idx, phase=None):
        calls.append(int(idx.shape[0]))
        return orig(idx, phase=phase)

    slots.ensure = spy
    try:
        yield calls
    finally:
        slots.ensure = orig


@pytest.fixture
def closer():
    """Register closeable objects torn down LIFO at test end — replaces
    ``obj = …; try: … finally: obj.close()`` frames."""
    objs = []

    def _defer(obj):
        objs.append(obj)
        return obj

    yield _defer
    for obj in reversed(objs):
        obj.close()
