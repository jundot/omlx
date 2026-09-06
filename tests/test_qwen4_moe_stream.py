# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp MoE expert weight streaming loader.

Covers the canonical key parser, the StreamingArtifact manifest index, the
weights-list interception (``stream_weight_items``), the idempotent enforcer
provider registration, and -- gated on the real artifact + native extension --
a bit-exact round-trip of a wrapped expert tensor through the actual
``QuantizedSwitchLinear.load_weights`` seam and ``gather_qmm`` forward.

The native-extension / real-artifact tests skip cleanly when either is absent so
the suite still runs in environments without the built ``_ext`` or the 46.87 GB
artifact.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import pytest

from omlx.custom_kernels.qwen4_moe_stream import fast, loader

# Real 46.87 GB streaming artifact. Machine-specific; override with
# OMLX_QWEN4_MOE_STREAM_ARTIFACT. The @_has_artifact guard skips the
# artifact-dependent tests cleanly when it is absent, so the default path is
# only an example, not a requirement.
_ARTIFACT = os.environ.get(
    "OMLX_QWEN4_MOE_STREAM_ARTIFACT",
    "/Users/alytaphoenix/.omlx/models/Vontra/"
    "Qwen3.8-Flash-Next-MLX-oQ2-MTP/moe_experts_streaming.artifact",
)

_native = pytest.mark.skipif(
    not fast.is_native_available(), reason="qwen4_moe_stream native ext unavailable"
)
_has_artifact = pytest.mark.skipif(
    not Path(_ARTIFACT).exists(), reason="streaming artifact not present"
)


# --------------------------------------------------------------------------- #
# canonical_key                                                               #
# --------------------------------------------------------------------------- #
def test_canonical_key_regular_layer():
    k = "language_model.model.layers.7.mlp.switch_mlp.gate_proj.weight"
    assert loader.canonical_key(k) == (False, 7, "gate_proj", "weight")


def test_canonical_key_mtp_both_prefixes():
    # checkpoint form (language_model.mtp....) and bound-module form (mtp....)
    a = "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.scales"
    b = "mtp.layers.0.mlp.switch_mlp.down_proj.scales"
    assert loader.canonical_key(a) == (True, 0, "down_proj", "scales")
    assert loader.canonical_key(b) == (True, 0, "down_proj", "scales")


def test_canonical_key_non_expert_returns_none():
    for k in (
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
        "language_model.model.embed_tokens.weight",
    ):
        assert loader.canonical_key(k) is None


# --------------------------------------------------------------------------- #
# stream_weight_items (no native ext / artifact needed -- uses a fake artifact) #
# --------------------------------------------------------------------------- #
class _FakeArtifact:
    """Stand-in for StreamingArtifact: knows a canon set, wraps to a marker."""

    def __init__(self, canons, shape=(2, 2, 2), dtype=mx.uint32):
        self._canons = set(canons)
        self._shape = shape
        self._dtype = dtype

    def __len__(self):
        return len(self._canons)

    def has(self, canon):
        return canon in self._canons

    def wrap(self, canon):
        # A fresh array; the swap is detected by object identity in the test.
        return mx.zeros(self._shape, dtype=self._dtype)


def test_stream_weight_items_swaps_only_experts():
    canon = (False, 0, "gate_proj", "weight")
    art = _FakeArtifact({canon}, shape=(2, 2, 2), dtype=mx.uint32)
    expert_key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    other_key = "language_model.model.layers.0.self_attn.q_proj.weight"
    items = [
        (expert_key, mx.zeros((2, 2, 2), dtype=mx.uint32)),
        (other_key, mx.ones((3, 3))),
    ]
    new_items, n_swapped, n_missing = loader.stream_weight_items(items, art)
    assert n_swapped == 1 and n_missing == 0
    by_key = dict(new_items)
    # expert entry replaced (object identity differs from the input value)
    assert by_key[expert_key] is not items[0][1]
    # non-expert untouched
    assert by_key[other_key] is items[1][1]


def test_stream_weight_items_missing_and_mismatch_pass_through():
    canon = (False, 0, "gate_proj", "weight")
    # artifact wraps a DIFFERENT shape than the checkpoint value -> keep resident
    art = _FakeArtifact({canon}, shape=(9, 9, 9), dtype=mx.uint32)
    expert_key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    missing_key = "language_model.model.layers.1.mlp.switch_mlp.up_proj.scales"
    orig_val = mx.zeros((2, 2, 2), dtype=mx.uint32)
    items = [
        (expert_key, orig_val),  # shape mismatch -> kept
        (missing_key, mx.zeros((2, 2, 2), dtype=mx.bfloat16)),  # not in artifact
    ]
    new_items, n_swapped, n_missing = loader.stream_weight_items(items, art)
    assert n_swapped == 0
    assert n_missing == 1
    assert dict(new_items)[expert_key] is orig_val  # mismatch left resident


# --------------------------------------------------------------------------- #
# proc_memory registry: name-keyed idempotency + discount arithmetic          #
# --------------------------------------------------------------------------- #
def test_proc_memory_name_keyed_idempotent():
    from omlx.utils import proc_memory

    name = "__test_qwen4_moe_stream__"
    try:
        proc_memory.register_external_wired_provider(name, lambda: 100)
        # re-register same name REPLACES (no accumulation -> no double count)
        proc_memory.register_external_wired_provider(name, lambda: 250)
        assert proc_memory.external_wired_bytes() >= 250
        # discount subtracts the external total, clamped at 0
        assert proc_memory.discount_external_wired(1000) == 1000 - proc_memory.external_wired_bytes()
        assert proc_memory.discount_external_wired(10) == 0  # clamp
    finally:
        proc_memory.unregister_external_wired_provider(name)


def test_proc_memory_provider_exception_swallowed():
    from omlx.utils import proc_memory

    name = "__test_qwen4_moe_stream_bad__"
    try:
        proc_memory.register_external_wired_provider(name, lambda: 1 / 0)
        # a throwing provider is skipped, not propagated
        assert proc_memory.external_wired_bytes() >= 0
    finally:
        proc_memory.unregister_external_wired_provider(name)


# --------------------------------------------------------------------------- #
# Real artifact: StreamingArtifact + bit-exact load_weights round-trip        #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# vlm wire-in context manager: gating / no-op fallbacks                        #
# --------------------------------------------------------------------------- #
def _write_config(tmp_path, model_type):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )
    return tmp_path


def _load_weights_is_unpatched():
    import mlx.nn as nn

    return "_patched_load_weights" not in getattr(
        nn.Module.load_weights, "__name__", ""
    )


def test_wirein_noop_for_non_qwen4_model(tmp_path):
    from omlx.engine import vlm

    _write_config(tmp_path, "llama")
    with vlm._stream_qwen4_exp_experts_on_load(tmp_path):
        assert _load_weights_is_unpatched()  # never touches load_weights


def test_wirein_noop_when_disabled(tmp_path, monkeypatch):
    from omlx.engine import vlm

    _write_config(tmp_path, "qwen4_exp")
    monkeypatch.setenv("OMLX_QWEN4_MOE_STREAM", "0")
    with vlm._stream_qwen4_exp_experts_on_load(tmp_path):
        assert _load_weights_is_unpatched()


def test_wirein_noop_when_artifact_missing(tmp_path, monkeypatch):
    from omlx.engine import vlm

    _write_config(tmp_path, "qwen4_exp")
    monkeypatch.setenv("OMLX_QWEN4_MOE_STREAM", "1")
    # qwen4_exp but no artifact next to the (empty) tmp model dir -> resident.
    with vlm._stream_qwen4_exp_experts_on_load(tmp_path):
        assert _load_weights_is_unpatched()


def test_entry_resident_size_streaming_discount():
    """_entry_resident_size subtracts the streamed payload from the full model
    and this WINS over the stale runtime_estimated_size (LIVE TEST 1 fix B: the
    reload block was that streaming-unaware 50GB estimate)."""
    from types import SimpleNamespace

    from omlx import engine_pool as ep

    entry = SimpleNamespace(
        estimated_size=70 * 1024**3,
        runtime_estimated_size=50 * 1024**3,
    )

    class _FakeStreaming:
        def _qwen4_moe_streaming_offload_bytes(self, e):
            return 40 * 1024**3

        def _distributed_deployment_for_entry(self, e):
            return None

    got = ep.EnginePool._entry_resident_size(_FakeStreaming(), entry)
    assert got == 30 * 1024**3  # 70 full - 40 streamed, not the stale 50

    class _NoStreaming(_FakeStreaming):
        def _qwen4_moe_streaming_offload_bytes(self, e):
            return 0

    # No streaming -> unchanged: falls back to runtime_estimated_size.
    got2 = ep.EnginePool._entry_resident_size(_NoStreaming(), entry)
    assert got2 == 50 * 1024**3


@_native
@_has_artifact
def test_streaming_artifact_index_and_provider():
    art = loader.StreamingArtifact(_ARTIFACT)
    assert len(art) == 441  # 48 layers * 9 + MTP * 9
    # a known 2-bit layer tensor and a 4-bit MTP tensor are both present
    assert art.has((False, 0, "gate_proj", "weight"))
    assert art.has((True, 0, "down_proj", "biases"))
    art.open()
    try:
        assert art.provider()() > 40 * 1024**3  # ~46.87 GB mapped
        w = art.wrap((False, 0, "gate_proj", "weight"))
        assert w.dtype == mx.uint32 and tuple(w.shape) == (512, 640, 160)
    finally:
        art.close()


def _manifest(path):
    with open(path, "rb") as f:
        mlen = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(mlen))["tensors"]


@_native
@_has_artifact
def test_streaming_offload_bytes_gating(monkeypatch):
    """streaming_offload_bytes returns the artifact size when streaming would
    engage and 0 when disabled -- admission's resident estimate must agree with
    whether streaming actually happens (LIVE TEST 1 fix B)."""
    import os

    model_dir = str(Path(_ARTIFACT).parent)
    monkeypatch.setenv("OMLX_QWEN4_MOE_STREAM", "1")
    got = loader.streaming_offload_bytes(model_dir)
    assert got == os.path.getsize(_ARTIFACT) > 40 * 1024**3
    monkeypatch.setenv("OMLX_QWEN4_MOE_STREAM", "0")
    assert loader.streaming_offload_bytes(model_dir) == 0
    # Non-model dir -> 0 even when enabled.
    monkeypatch.setenv("OMLX_QWEN4_MOE_STREAM", "1")
    assert loader.streaming_offload_bytes("/nonexistent/model/dir") == 0


@_native
@_has_artifact
def test_deferred_unmap_completes_on_free():
    """close() with a live wrapped array defers the munmap; once the array is
    freed the deferred unmap fires and mapped_bytes drops to 0 (LIVE TEST 1
    fix C relies on this to recover the ceiling on unload)."""
    import gc

    # Delta-based (other tests may leave mappings settling): baseline first.
    for _ in range(20):
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
    base = fast.mapped_bytes()

    man = _manifest(_ARTIFACT)
    e = man["language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales"]
    aid = fast.mmap_artifact(_ARTIFACT)
    w = fast.wrap_tensor(aid, e["offset"], e["length"], e["shape"], e["dtype"])
    assert fast.mapped_bytes() > base
    fast.close_artifact(aid)  # deferred (w still alive)
    assert fast.mapped_bytes() > base  # not yet unmapped
    del w
    dropped = False
    for _ in range(20):
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        if fast.mapped_bytes() <= base:
            dropped = True
            break
    assert dropped, "deferred unmap did not complete after freeing the array"


@_native
@_has_artifact
def test_close_while_alive_defers_unmap():
    """close_artifact with a live wrapped array must defer the munmap: a GPU
    touch after close must not fault, and a fresh wrap on the closed id is
    refused (Fable review 2, issue 4 -- refcounted deferred unmap)."""
    man = _manifest(_ARTIFACT)
    e = man["language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales"]
    aid = fast.mmap_artifact(_ARTIFACT)
    w = fast.wrap_tensor(aid, e["offset"], e["length"], e["shape"], e["dtype"])
    assert fast.mapped_bytes() > 0
    fast.close_artifact(aid)  # marks closed; unmap deferred (w still alive)
    # GPU touch AFTER close must complete without faulting on unmapped memory.
    total = float(mx.sum(w.astype(mx.float32)))
    assert total == total  # finite / not a fault artifact
    # A new wrap on the closed mapping is refused (no use-after-close).
    with pytest.raises(Exception):
        fast.wrap_tensor(aid, e["offset"], e["length"], e["shape"], e["dtype"])
    del w  # last ref -> deleter drops refcount -> deferred munmap runs


@_native
@_has_artifact
@pytest.mark.parametrize(
    "pfx,is_mtp",
    [
        ("language_model.model.layers.0.mlp.switch_mlp", False),
        ("language_model.mtp.layers.0.mlp.switch_mlp", True),
    ],
)
def test_load_weights_roundtrip_bit_exact(pfx, is_mtp):
    """Wrapped arrays load through the real QuantizedSwitchLinear.load_weights
    seam and produce bit-exact gather_qmm output vs a resident-read reference."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    man = _manifest(_ARTIFACT)
    art = loader.StreamingArtifact(_ARTIFACT)
    art.open()
    try:
        proj = "gate_proj"
        we = man[f"{pfx}.{proj}.weight"]
        se = man[f"{pfx}.{proj}.scales"]
        bits, gs = we["bits"], we["group_size"]
        n_exp, out_dim, _ = we["shape"]
        in_dim = se["shape"][2] * gs

        canon_w = (is_mtp, 0, proj, "weight")
        canon_s = (is_mtp, 0, proj, "scales")
        canon_b = (is_mtp, 0, proj, "biases")

        # Build the module and load the WRAPPED (streamed) arrays into it.
        mod = QuantizedSwitchLinear(
            in_dim, out_dim, n_exp, bias=False, group_size=gs, bits=bits, mode="affine"
        )
        mod.load_weights(
            [
                ("weight", art.wrap(canon_w)),
                ("scales", art.wrap(canon_s)),
                ("biases", art.wrap(canon_b)),
            ],
            strict=True,
        )

        # Reference: GENUINE resident load via numpy read of the artifact bytes
        # (NOT another wrapped array) -- this is what actually proves
        # streamed == resident, closing Fable review-2 issue 2.
        import numpy as np

        def resident(canon):
            e = man[f"{pfx}.{canon[2]}.{canon[3]}"]
            npd = {"U32": np.uint32, "BF16": np.uint16}[e["dtype"]]
            with open(_ARTIFACT, "rb") as fp:
                fp.seek(e["offset"])
                buf = fp.read(e["length"])
            arr = mx.array(np.frombuffer(buf, dtype=npd).reshape(e["shape"]))
            return arr.view(mx.bfloat16) if e["dtype"] == "BF16" else arr

        ref_w = resident(canon_w)
        ref_s = resident(canon_s)
        ref_b = resident(canon_b)

        T = 72
        x = mx.random.normal((T, 1, in_dim)).astype(mx.bfloat16)
        idx = mx.array([(t * 37) % n_exp for t in range(T)], dtype=mx.uint32).reshape(T, 1)

        y_mod = mod(x, idx, sorted_indices=False)
        y_ref = mx.gather_qmm(
            x, ref_w, ref_s, ref_b, rhs_indices=idx, transpose=True,
            group_size=gs, bits=bits, mode="affine", sorted_indices=False,
        )
        assert float(mx.max(mx.abs(y_mod.astype(mx.float32) - y_ref.astype(mx.float32)))) == 0.0
    finally:
        art.close()
