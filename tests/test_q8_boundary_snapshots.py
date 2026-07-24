"""Round-trip tests for int8 boundary-snapshot state storage.

Simulates a Nemotron-3-Super snapshot: 40 ArraysCache layers with bf16 conv
state + fp32 ssm state, plus 8 KVCache layers set to None (sliceable layers
are skipped in snapshots). Exercises both readers: the raw-bytes pending
path (load before flush) and the safetensors disk path (load after flush,
pending cleared)."""

import time

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.cache.boundary_snapshot_store import (  # noqa: E402
    BoundarySnapshotSSDStore,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal is not available"
)


def fake_extract(snapshot_cache):
    extracted = []
    for entry in snapshot_cache:
        if entry is None:
            extracted.append(
                {
                    "state": (),
                    "meta_state": (),
                    "class_name": "KVCache",
                    "cache_type": "KVCache",
                }
            )
        else:
            conv, ssm = entry
            extracted.append(
                {
                    "state": (conv, ssm),
                    "meta_state": ("6",),
                    "class_name": "ArraysCache",
                    "cache_type": "ArraysCache",
                }
            )
    return extracted, {}


def rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() / (mx.abs(b).max() + 1e-8)).item()


def test_q8_round_trip(tmp_path):
    mx.random.seed(11)
    layers = []
    for i in range(8):
        if i % 4 == 3:  # sprinkle "attention" layers
            layers.append(None)
        else:
            conv = (mx.random.normal((1, 3, 2560)) * 0.5).astype(mx.bfloat16)
            ssm = mx.random.normal((1, 32, 64, 128)).astype(mx.float32) * 0.3
            layers.append((conv, ssm))
    mx.eval(*[a for lay in layers if lay for a in lay])

    store = BoundarySnapshotSSDStore(tmp_path)

    ok = store.save("req1", 2048, layers, fake_extract)
    assert ok, "save failed"

    # ---- pending-path load (raw bytes) — force _deserialize, not the
    # cheap 'extracted' echo, by removing it
    with store._pending_lock:
        entry = store._pending_writes[("req1", 2048)]
        raw_bytes = sum(len(v[0]) for v in entry["tensors_raw"].values())
        entry.pop("extracted", None)
    loaded_pending = store.load("req1", 2048)
    assert loaded_pending is not None, "pending load failed"

    # ---- wait for the writer thread to flush, then clear pending -> disk
    for _ in range(100):
        with store._pending_lock:
            if ("req1", 2048) not in store._pending_writes:
                break
        time.sleep(0.05)
    else:
        # some versions keep pending entries until cleanup; drop manually
        with store._pending_lock:
            store._pending_writes.pop(("req1", 2048), None)
    loaded_disk = store.load("req1", 2048)
    assert loaded_disk is not None, "disk load failed"

    files = list(store._snapshot_dir.rglob("*.safetensors"))
    disk_bytes = sum(f.stat().st_size for f in files)

    worst_ssm, worst_conv = 0.0, 0.0
    for name, loaded in (("pending", loaded_pending), ("disk", loaded_disk)):
        for i, layer in enumerate(layers):
            if layer is None:
                assert (
                    loaded[i]["state"] == ()
                    or loaded[i]["state"] is None
                    or len(loaded[i]["state"]) == 0
                )
                continue
            conv, ssm = layer
            lconv, lssm = loaded[i]["state"]
            ec, es = rel(lconv, conv), rel(lssm, ssm)
            worst_conv, worst_ssm = max(worst_conv, ec), max(worst_ssm, es)
            assert ec == 0.0, f"{name} layer {i}: conv changed ({ec})"
            assert es < 1.2e-2, f"{name} layer {i}: ssm err {es}"

    fp32_bytes = sum(
        lay[1].size * 4 + lay[0].size * 2 for lay in layers if lay is not None
    )
    # 4x on the fp32 ssm payload (conv bf16 stays exact and unquantized)
    assert raw_bytes < 0.35 * fp32_bytes
    assert disk_bytes < 0.35 * fp32_bytes
    assert worst_conv == 0.0
    assert worst_ssm < 1.2e-2
    store.cleanup_all()
