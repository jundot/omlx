"""Round-trip tests for int8 boundary-snapshot state storage.

int8 storage is scoped: ``save`` only quantizes fp32 states on layers the
caller has semantically identified as Mamba2-family SSM (via
``ssm_state_shapes``), and only when the tensor matches the expected
per-layer state shape. Everything else — including large fp32 recurrent
states of non-Mamba layers (e.g. gated-deltanet) — is stored bit-exact.

Covers all three read paths (pending fast path, raw-bytes pending path,
safetensors disk path), which must restore the identical representation
regardless of writer timing, plus greedy cache-hit output regressions on a
tiny in-process mlx-lm mamba2 model."""

import time

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.cache.boundary_snapshot_store import (  # noqa: E402
    BoundarySnapshotSSDStore,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal is not available"
)

# Matches the fake Nemotron-style layers built in _make_layers.
_SSM_SHAPE = (32, 64, 128)


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


def _make_layers(n=8, ssm_shape=_SSM_SHAPE):
    layers = []
    for i in range(n):
        if i % 4 == 3:  # sprinkle "attention" layers
            layers.append(None)
        else:
            conv = (mx.random.normal((1, 3, 2560)) * 0.5).astype(mx.bfloat16)
            ssm = mx.random.normal((1, *ssm_shape)).astype(mx.float32) * 0.3
            layers.append((conv, ssm))
    mx.eval(*[a for lay in layers if lay for a in lay])
    return layers


def _ssm_shapes(layers, shape=_SSM_SHAPE):
    return {i: shape for i, lay in enumerate(layers) if lay is not None}


def _drain_pending(store, key):
    for _ in range(100):
        with store._pending_lock:
            if key not in store._pending_writes:
                return
        time.sleep(0.05)
    # some versions keep pending entries until cleanup; drop manually
    with store._pending_lock:
        store._pending_writes.pop(key, None)


def test_q8_round_trip(tmp_path):
    mx.random.seed(11)
    layers = _make_layers()

    store = BoundarySnapshotSSDStore(tmp_path)

    ok = store.save(
        "req1", 2048, layers, fake_extract, ssm_state_shapes=_ssm_shapes(layers)
    )
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
    _drain_pending(store, ("req1", 2048))
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


def test_unscoped_states_stay_fp32(tmp_path):
    """Without ssm_state_shapes, nothing is quantized — every state restores
    bit-exact. This is the non-Mamba hybrid case (e.g. gated-deltanet models
    like Qwen3.5) whose large fp32 recurrent states must NOT be q8."""
    mx.random.seed(12)
    layers = _make_layers()

    store = BoundarySnapshotSSDStore(tmp_path)
    ok = store.save("req1", 2048, layers, fake_extract)
    assert ok, "save failed"

    with store._pending_lock:
        store._pending_writes[("req1", 2048)].pop("extracted", None)
    loaded_pending = store.load("req1", 2048)
    _drain_pending(store, ("req1", 2048))
    loaded_disk = store.load("req1", 2048)

    for loaded in (loaded_pending, loaded_disk):
        assert loaded is not None
        for i, layer in enumerate(layers):
            if layer is None:
                continue
            for orig, rest in zip(layer, loaded[i]["state"]):
                assert rest.dtype == orig.dtype
                assert mx.array_equal(rest, orig).item(), f"layer {i} changed"
    store.cleanup_all()


def test_shape_mismatch_not_quantized(tmp_path):
    """A scoped layer whose fp32 state does not match the expected SSM shape
    (e.g. a deltanet-style recurrent state) is stored bit-exact."""
    mx.random.seed(13)
    # Large fp32 4-D states, but NOT the registered (32, 64, 128) shape.
    layers = []
    conv = (mx.random.normal((1, 3, 2048)) * 0.5).astype(mx.bfloat16)
    delta = mx.random.normal((1, 16, 128, 128)).astype(mx.float32)
    layers.append((conv, delta))
    mx.eval(conv, delta)

    store = BoundarySnapshotSSDStore(tmp_path)
    ok = store.save(
        "req1", 2048, layers, fake_extract, ssm_state_shapes={0: _SSM_SHAPE}
    )
    assert ok, "save failed"

    with store._pending_lock:
        store._pending_writes[("req1", 2048)].pop("extracted", None)
    loaded = store.load("req1", 2048)
    assert loaded is not None
    lconv, ldelta = loaded[0]["state"]
    assert ldelta.dtype == mx.float32
    assert mx.array_equal(ldelta, delta).item()
    assert mx.array_equal(lconv, conv).item()
    store.cleanup_all()


def test_all_read_paths_identical(tmp_path):
    """The pending fast path, raw-bytes pending path, and disk path must
    restore the identical representation — a cache hit cannot depend on
    writer timing."""
    mx.random.seed(14)
    layers = _make_layers()

    store = BoundarySnapshotSSDStore(tmp_path)
    ok = store.save(
        "req1", 2048, layers, fake_extract, ssm_state_shapes=_ssm_shapes(layers)
    )
    assert ok, "save failed"

    # Fast path: 'extracted' still present.
    loaded_fast = store.load("req1", 2048)
    assert loaded_fast is not None

    # Raw-bytes pending path.
    with store._pending_lock:
        store._pending_writes[("req1", 2048)].pop("extracted", None)
    loaded_raw = store.load("req1", 2048)
    assert loaded_raw is not None

    # Disk path.
    _drain_pending(store, ("req1", 2048))
    loaded_disk = store.load("req1", 2048)
    assert loaded_disk is not None

    for i, layer in enumerate(layers):
        if layer is None:
            continue
        for k in range(2):
            fast = loaded_fast[i]["state"][k]
            raw = loaded_raw[i]["state"][k]
            disk = loaded_disk[i]["state"][k]
            assert fast.dtype == raw.dtype == disk.dtype
            assert mx.array_equal(fast, raw).item(), f"layer {i} elem {k}"
            assert mx.array_equal(raw, disk).item(), f"layer {i} elem {k}"
    store.cleanup_all()


# ---------------------------------------------------------------------------
# Greedy cache-hit output regressions on a tiny in-process mamba2 model.
# ---------------------------------------------------------------------------


def _tiny_mamba2():
    mamba2 = pytest.importorskip("mlx_lm.models.mamba2")
    args = mamba2.ModelArgs(
        model_type="mamba2",
        num_heads=16,
        head_dim=64,
        vocab_size=256,
        hidden_size=128,
        intermediate_size=1024,
        state_size=64,
        num_hidden_layers=2,
        layer_norm_epsilon=1e-5,
        conv_kernel=4,
        n_groups=4,
        use_bias=False,
        use_conv_bias=True,
        tie_word_embeddings=True,
        time_step_limit=(0.001, 100.0),
        time_step_rank="auto",
    )
    model = mamba2.Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def _scan_ssm_shapes(model):
    """Mirror of Scheduler._mamba2_ssm_state_shapes' mixer predicate."""
    shapes = {}
    for i, layer in enumerate(model.layers):
        for m in layer.modules():
            num_heads = getattr(m, "num_heads", None)
            head_dim = getattr(m, "head_dim", None)
            state_size = getattr(m, "ssm_state_size", None)
            if (
                isinstance(num_heads, int)
                and isinstance(head_dim, int)
                and isinstance(state_size, int)
                and getattr(m, "conv1d", None) is not None
                and getattr(m, "A_log", None) is not None
            ):
                shapes[i] = (num_heads, head_dim, state_size)
                break
    return shapes


def _extract_from_cache(cache):
    extracted = []
    for c in cache:
        extracted.append(
            {
                "state": (c[0], c[1]),
                "meta_state": (),
                "class_name": "ArraysCache",
                "cache_type": "ArraysCache",
            }
        )
    return extracted, {}


def _restore_cache(model, loaded):
    cache = model.make_cache()
    for c, layer_state in zip(cache, loaded):
        c[0], c[1] = layer_state["state"]
    return cache


def _greedy_continue(model, cache, first_token, n=12):
    tokens = []
    tok = first_token
    for _ in range(n):
        logits = model(mx.array([[tok]]), cache=cache)
        tok = mx.argmax(logits[:, -1, :]).item()
        tokens.append(tok)
    return tokens


def test_cache_hit_greedy_output_parity(tmp_path):
    """With int8 SSM snapshots active, greedy continuation from a cache hit
    must be identical whichever read path served it."""
    mx.random.seed(21)
    model = _tiny_mamba2()
    shapes = _scan_ssm_shapes(model)
    # Predicate must find every mamba2 layer with the real state layout.
    assert shapes == {0: (16, 64, 64), 1: (16, 64, 64)}

    prompt = [7, 42, 99, 3, 58, 120, 14]
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    last = mx.argmax(logits[:, -1, :]).item()

    store = BoundarySnapshotSSDStore(tmp_path)
    ok = store.save("req1", 2048, cache, _extract_from_cache, ssm_state_shapes=shapes)
    assert ok, "save failed"

    loaded_fast = store.load("req1", 2048)
    with store._pending_lock:
        entry = store._pending_writes.get(("req1", 2048))
        if entry is not None:  # writer may have flushed already
            entry.pop("extracted", None)
    loaded_raw = store.load("req1", 2048)
    _drain_pending(store, ("req1", 2048))
    loaded_disk = store.load("req1", 2048)
    assert None not in (loaded_fast, loaded_raw, loaded_disk)

    out_fast = _greedy_continue(model, _restore_cache(model, loaded_fast), last)
    out_raw = _greedy_continue(model, _restore_cache(model, loaded_raw), last)
    out_disk = _greedy_continue(model, _restore_cache(model, loaded_disk), last)

    assert out_fast == out_raw == out_disk
    store.cleanup_all()


def test_cache_hit_greedy_output_exact_when_unscoped(tmp_path):
    """Without SSM scoping (a non-Mamba model), a cache hit must reproduce
    the uninterrupted greedy continuation exactly."""
    mx.random.seed(22)
    model = _tiny_mamba2()

    prompt = [5, 17, 200, 33, 91]
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    last = mx.argmax(logits[:, -1, :]).item()

    store = BoundarySnapshotSSDStore(tmp_path)
    ok = store.save("req1", 2048, cache, _extract_from_cache)
    assert ok, "save failed"

    # Uninterrupted baseline, straight from the live cache.
    baseline = _greedy_continue(model, cache, last)

    with store._pending_lock:
        entry = store._pending_writes.get(("req1", 2048))
        if entry is not None:  # writer may have flushed already
            entry.pop("extracted", None)
    loaded_raw = store.load("req1", 2048)
    _drain_pending(store, ("req1", 2048))
    loaded_disk = store.load("req1", 2048)
    assert None not in (loaded_raw, loaded_disk)

    out_raw = _greedy_continue(model, _restore_cache(model, loaded_raw), last)
    out_disk = _greedy_continue(model, _restore_cache(model, loaded_disk), last)

    assert out_raw == baseline
    assert out_disk == baseline
    store.cleanup_all()
