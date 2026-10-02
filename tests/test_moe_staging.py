# SPDX-License-Identifier: Apache-2.0
"""Production opt-in transport, exact cache state, and asynchronous ownership."""

import gc
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.moe_staging import StagedRead, Staging
from omlx.patches import moe_expert_offload as offload
from tests.test_moe_expert_offload import (
    D,
    _glu_tensors,
    _make_glu,
    _MiniMoE,
    _save_checkpoint,
)


@pytest.fixture
def native():
    return pytest.importorskip("omlx.custom_kernels.moe_staging._ext")


@pytest.fixture(autouse=True)
def fresh_pool(monkeypatch):
    offload._shutdown_io_pool()
    monkeypatch.setenv("OMLX_MOE_OFFLOAD_IO_WORKERS", "12")
    monkeypatch.delenv("OMLX_MOE_OFFLOAD_STAGING", raising=False)
    yield
    offload._shutdown_io_pool()
    mx.synchronize()
    gc.collect()


def bits(array):
    mx.eval(array)
    return np.asarray(array.view(mx.uint8)).tobytes()


def snapshot(cache):
    return (
        list(cache.slot_of.items()),
        list(cache.free),
        cache.warm,
        cache.hits,
        cache.misses,
        bits(cache.map),
        [
            bits(a) if a is not None else None
            for t in cache.resident.values()
            for a in t
        ],
    )


def wrapped(path, glus, kind="qwen4_exp", **kwargs):
    tensors = {}
    for i, glu in enumerate(glus):
        tensors.update(_glu_tensors(glu, f"layers.{i}.experts.switch_glu"))
    _save_checkpoint(path, tensors)
    (path / "config.json").write_text(json.dumps({"model_type": kind}))
    model = _MiniMoE(glus)
    assert offload.apply_moe_expert_offload(model, path, 0.25, **kwargs) == len(glus)
    return [layer.experts.switch_glu for layer in model.layers]


@pytest.mark.parametrize("flag", [None, "0", "true", "garbage", "1"])
def test_explicit_opt_in_is_lazy_and_shared(tmp_path, monkeypatch, flag):
    if flag is not None:
        monkeypatch.setenv("OMLX_MOE_OFFLOAD_STAGING", flag)
    switches = wrapped(tmp_path, [_make_glu(0), _make_glu(1)])
    stores = [s.cache.disk._store for s in switches]
    assert stores[0] is stores[1]
    policy = stores[0]._staging
    assert (policy is not None) == (flag == "1")
    if policy:
        assert policy.pool is None  # loader never allocates the staging pool
        assert len(policy.sizes) == 360
        assert sum((s + 16383) // 16384 * 16384 for s in policy.sizes) <= 128 * 2**20


@pytest.mark.parametrize("kind,mtp", [("other", False), ("qwen4_exp", True)])
def test_unqualified_family_and_mtp_stay_on_bytes(tmp_path, monkeypatch, kind, mtp):
    monkeypatch.setenv("OMLX_MOE_OFFLOAD_STAGING", "1")
    switch = wrapped(tmp_path, [_make_glu()], kind, mtp_resident=mtp)[0]
    assert switch.cache.disk._store._staging is None


def test_large_layout_and_missing_helper_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_MOE_OFFLOAD_STAGING", "1")
    monkeypatch.setattr("omlx.custom_kernels.moe_staging.MAX_BYTES", 1)
    switch = wrapped(tmp_path, [_make_glu()])[0]
    assert switch.cache.disk._store._staging is None
    policy = Staging(
        [64], offload.CheckpointExpertStore.read, offload.CheckpointExpertStore.to_mx
    )
    with patch.dict(sys.modules, {"omlx.custom_kernels.moe_staging._ext": None}):
        assert policy.for_owner() is None
        assert policy.for_owner() is None and policy.disabled


@pytest.mark.parametrize("tag", offload._DTYPES)
def test_transport_bits_zero_copy_and_lazy_lifetime(tmp_path, native, tag):
    dtype, view = offload._DTYPES[tag]
    raw = bytes(range(64))
    path = tmp_path / "raw"
    path.write_bytes(raw)
    with path.open("rb") as file:
        plan = offload._ReadPlan(
            file.fileno(), 0, 64, dtype, view, (64 // np.dtype(dtype).itemsize,)
        )
        policy = Staging(
            [64],
            offload.CheckpointExpertStore.read,
            offload.CheckpointExpertStore.to_mx,
        )
        assert policy.for_owner() is policy
        with ThreadPoolExecutor(max_workers=1) as pool:
            payload = pool.submit(policy.read, plan).result()
            assert isinstance(payload, StagedRead)
            assert pool.submit(policy.for_owner).result() is None
        array = policy.to_mx(plan, payload)
        assert bits(array) == raw
        assert native.address(array) == payload.lease.address
        lazy = mx.concatenate([array, array])
        del array, payload
        gc.collect()
        assert policy.pool.stats()["busy"] == 1
        assert isinstance(policy.read(plan), bytes)  # no wait on an occupied lease
        assert bits(lazy) == raw + raw
        del lazy
        mx.synchronize()
        gc.collect()
        assert policy.pool.stats()["busy"] == 0


@pytest.mark.parametrize("buffers", [0, 27, 90])
@pytest.mark.parametrize("workers", [1, 12])
def test_exact_output_cache_installs_and_nine_reads(
    tmp_path, native, monkeypatch, buffers, workers
):
    monkeypatch.setenv("OMLX_MOE_OFFLOAD_IO_WORKERS", str(workers))
    glu = _make_glu(34)
    mx.random.seed(37)
    steps = [
        (
            mx.random.normal((1, 1, D)),
            mx.array([[[i % 32, (i + 9) % 32, (i + 17) % 32]]]),
        )
        for i in range(12)
    ]

    def run(staged):
        monkeypatch.setenv("OMLX_MOE_OFFLOAD_STAGING", "1" if staged else "0")
        switch = wrapped(tmp_path, [glu])[0]
        cache = switch.cache
        policy = cache.disk._store._staging
        if policy:
            policy.sizes = [p.nbytes for e in range(10) for _, _, p in cache._plans(e)][
                :buffers
            ]
        outputs, states, writes, tasks = [], [], [], []
        write, submit = cache._write, ThreadPoolExecutor.submit

        def recording_write(slot, payload):
            writes.append((slot, [(n, f, p.offset) for n, f, p, _ in payload]))
            return write(slot, payload)

        def recording_submit(pool, function, *args, **kwargs):
            tasks.append((args[0].offset, args[0].nbytes))
            return submit(pool, function, *args, **kwargs)

        with (
            patch.object(cache, "_write", recording_write),
            patch.object(ThreadPoolExecutor, "submit", recording_submit),
        ):
            for x, indices in steps:
                outputs.append(bits(switch(x, indices)))
                states.append(snapshot(cache))
        if workers > 1:
            assert len(tasks) == 9 * cache.misses
        if policy:
            assert (
                policy.pool.stats()["reads"] + policy.pool.stats()["fallbacks"]
                == 9 * cache.misses
            )
            del cache, switch
            mx.synchronize()
            gc.collect()
            assert policy.pool.stats()["busy"] == 0
        return outputs, states, writes, tasks

    assert run(False) == run(True)


@pytest.mark.parametrize("failure", ["short", "bad-fd", "bad-shape"])
def test_error_releases_lease_and_retry(tmp_path, native, failure):
    path = tmp_path / "raw"
    path.write_bytes(bytes(range(64)))
    with path.open("rb") as file:
        plan = offload._ReadPlan(file.fileno(), 0, 64, np.uint8, None, (64,))
        policy = Staging(
            [64],
            offload.CheckpointExpertStore.read,
            offload.CheckpointExpertStore.to_mx,
        ).for_owner()
        if failure == "short":
            with pytest.raises(OSError, match="short read of 64 bytes at 32"):
                policy.read(plan._replace(offset=32))
        elif failure == "bad-fd":
            with pytest.raises(OSError) as error:
                policy.read(plan._replace(fd=-1))
            assert error.value.errno == 9
        else:
            invalid = plan._replace(shape=(63,))
            payload = policy.read(invalid)
            with pytest.raises(ValueError, match="byte length mismatch"):
                policy.to_mx(invalid, payload)
            del payload
        gc.collect()
        assert policy.pool.stats()["busy"] == 0
        assert bits(policy.to_mx(plan, policy.read(plan))) == bytes(range(64))


def test_cross_stream_completion_and_model_release_without_gil_deadlock(native):
    code = """
import tempfile, gc
import mlx.core as mx
import numpy as np
from omlx.custom_kernels.moe_staging import Staging
from omlx.patches.moe_expert_offload import CheckpointExpertStore as Store, _ReadPlan
with tempfile.TemporaryFile() as f:
    raw = bytes(range(64)); f.write(raw); f.flush()
    plan = _ReadPlan(f.fileno(), 0, 64, np.uint8, None, (64,))
    policy = Staging([64], Store.read, Store.to_mx).for_owner()
    payload = policy.read(plan)
    array = policy.to_mx(plan, payload)
    stream = mx.new_stream(mx.gpu)
    with mx.stream(stream):
        output = mx.concatenate([array] * 1024)
        mx.async_eval(output)
    del payload, array, policy
    gc.collect()
    mx.synchronize(stream)
    mx.eval(output)
    assert np.asarray(output).tobytes() == raw * 1024
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


@pytest.mark.parametrize("field", ["weight", "scales", "biases"])
@pytest.mark.parametrize("fault", ["read", "convert"])
def test_partial_install_cleanup_and_retry_are_exact(
    tmp_path, native, monkeypatch, field, fault
):
    glu = _make_glu(38)

    def run(staged):
        monkeypatch.setenv("OMLX_MOE_OFFLOAD_STAGING", "1" if staged else "0")
        cache = wrapped(tmp_path, [glu])[0].cache
        cache.ensure(mx.arange(cache.capacity))
        target = cache.disk.plan("gate_proj", field, cache.capacity + 1)
        policy = cache.disk._store._staging
        owner = policy if staged else offload.CheckpointExpertStore
        name = "read" if fault == "read" else "to_mx"
        original = getattr(owner, name)

        def fail(plan, *args):
            if plan == target:
                raise OSError("injected")
            return original(plan, *args)

        with (
            patch.object(owner, name, fail if staged else staticmethod(fail)),
            pytest.raises(OSError, match="injected"),
        ):
            cache.ensure(
                mx.array([cache.capacity, cache.capacity + 1, cache.capacity + 2])
            )
        failed = snapshot(cache)
        cache.ensure(
            mx.array([cache.capacity, cache.capacity + 1, cache.capacity + 2, 0])
        )
        result = failed, snapshot(cache)
        del cache
        mx.synchronize()
        gc.collect()
        if policy:
            assert policy.pool.stats()["busy"] == 0
        return result

    assert run(False) == run(True)
