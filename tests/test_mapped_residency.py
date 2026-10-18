# SPDX-License-Identifier: Apache-2.0
"""Physical residency bounds and external-buffer lifetime regressions."""

import ctypes
import gc
import weakref
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.utils import mapped_residency as mr
from omlx.utils import mapped_weights as mw


@pytest.fixture(autouse=True)
def isolated_views(monkeypatch):
    monkeypatch.setattr(mr, "_views", weakref.WeakValueDictionary())


def load(path):
    mx.save_safetensors(str(path), {"x": mx.arange(65536, dtype=mx.float32)})
    # MLX's writer may omit header padding; ensure the test exercises mapping.
    import json
    import struct

    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = raw[8 : 8 + n]
    assert json.loads(header)["x"]["data_offsets"] == [0, 65536 * 4]
    padded = header + b" " * ((-n) % 8)
    path.write_bytes(struct.pack("<Q", len(padded)) + padded + raw[8 + n :])
    return mw._load_mapped(path)[0]


def test_live_aliases_count_once_and_last_exporter_releases(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    first = load(path)
    second = mw._load_mapped(path)[0]
    monkeypatch.setattr(
        mr,
        "_resident_file_bytes",
        lambda a, n: (n + mr._PAGE - 1) // mr._PAGE * mr._PAGE,
    )
    assert mr.resident_mapped_bytes() == 65536 * 4
    del first
    gc.collect()
    assert mr.resident_mapped_bytes() == 65536 * 4
    # A derived MLX view must retain the NumPy exporter.
    derived = second["x"][16:32]
    del second
    gc.collect()
    assert mr.resident_mapped_bytes() == 65536 * 4
    assert derived.tolist() == list(range(16, 32))
    del derived
    mx.synchronize()
    mx.clear_cache()
    gc.collect()
    assert mr.resident_mapped_bytes() == 0


def test_sanitized_copy_loses_residency_credit(tmp_path, monkeypatch):
    weights = load(tmp_path / "model.safetensors")
    monkeypatch.setattr(
        mr,
        "_resident_file_bytes",
        lambda a, n: (n + mr._PAGE - 1) // mr._PAGE * mr._PAGE,
    )
    assert mr.resident_mapped_bytes() > 0
    weights["x"] = weights["x"] + 1
    mx.eval(weights)
    mx.synchronize()
    mx.clear_cache()
    gc.collect()
    assert weights["x"][0].item() == 1
    assert mr.resident_mapped_bytes() == 0


def test_fresh_counter_can_drop_to_zero_and_errors_never_reuse_credit(
    tmp_path, monkeypatch
):
    weights = load(tmp_path / "model.safetensors")
    monkeypatch.setattr(
        mr,
        "_resident_file_bytes",
        lambda a, n: (n + mr._PAGE - 1) // mr._PAGE * mr._PAGE,
    )
    assert mr.resident_mapped_bytes() > 0
    monkeypatch.setattr(mr, "_resident_file_bytes", lambda a, n: 0)
    assert mr.resident_mapped_bytes() == 0

    def fail(*args):
        raise OSError("query failed")

    monkeypatch.setattr(mr, "_resident_file_bytes", fail)
    assert mr.resident_mapped_bytes() == 0
    assert weights["x"].shape == (65536,)


def test_native_counter_bound_does_not_exceed_mincore(tmp_path):
    weights = load(tmp_path / "model.safetensors")
    # The view snapshot keeps each mapping alive across both OS queries.
    entries = list(mr._views.items())
    assert entries
    total = 0
    for (_dev, _ino, size, address, lo, hi), _view in entries:
        extent = (size + mr._PAGE - 1) // mr._PAGE * mr._PAGE
        native = mr._resident_file_bytes(address, size)
        vec = np.empty(extent // mr._PAGE, dtype=np.uint8)
        result = ctypes.CDLL(None).mincore(
            ctypes.c_void_p(address),
            ctypes.c_size_t(extent),
            ctypes.c_void_p(vec.ctypes.data),
        )
        assert result == 0
        assert 0 < native <= int(np.count_nonzero(vec & 1)) * mr._PAGE
        total += max(0, native - (extent - (hi - lo)))
    assert mr.resident_mapped_bytes() == total
    assert weights["x"].shape == (65536,)


@pytest.mark.parametrize("resident,expected", [(0, 8), (2, 8), (3, 8), (7, 12)])
def test_guard_counts_only_resident_bytes_outside_reclaimable_overlap(
    monkeypatch, resident, expected
):
    import omlx.process_memory_enforcer as pme

    g = 1024**3
    guard = pme.ProcessMemoryEnforcer(engine_pool=SimpleNamespace())
    guard._tier_reserve_bytes = lambda: 0
    guard._external_process_bytes = lambda: 0
    monkeypatch.setattr(pme, "get_phys_footprint", lambda: 4 * g)
    monkeypatch.setattr(pme, "get_graphics_footprint", lambda: 4 * g)
    monkeypatch.setattr(
        pme,
        "get_macos_vm_stats",
        lambda: dict(free=g, inactive=2 * g, external=3 * g, active=0),
    )
    monkeypatch.setattr(mr, "resident_mapped_bytes", lambda: resident * g)
    releases = []
    guard._release_credit = lambda own, free: releases.append(own) or 0
    terms = guard._dynamic_ceiling_terms()
    assert terms["dynamic"] == expected * g
    assert releases == [4 * g]  # residency never feeds delayed-release credit


def test_partial_file_mapping_is_never_credited(tmp_path):
    path = tmp_path / "partial"
    path.write_bytes(b"a" * (mr._PAGE * 4))
    mapping = np.memmap(
        path, mode="r", dtype=np.uint8, offset=mr._PAGE, shape=(mr._PAGE,)
    )
    mr.track(mapping, mapping, path.stat(), 0, mr._PAGE)
    assert mr.resident_mapped_bytes() == 0
    assert not mr._views


def test_cow_shadow_object_receives_no_credit(tmp_path):
    path = tmp_path / "private"
    path.write_bytes(b"a" * (mr._PAGE * 4))
    mapping = np.memmap(path, mode="c", dtype=np.uint8)
    mapping[0] = 99
    assert mr._resident_file_bytes(mapping.ctypes.data, mapping.nbytes) == 0
    assert path.read_bytes()[0] == ord("a")


def test_partial_live_union_and_disjoint_aliases_use_one_file_budget(
    tmp_path, monkeypatch
):
    path = tmp_path / "ranges"
    p = mr._PAGE
    path.write_bytes(b"a" * (p * 4))
    a = np.memmap(path, mode="r", dtype=np.uint8)
    b = np.memmap(path, mode="r", dtype=np.uint8)
    first, second = a[:p], b[2 * p : 3 * p]
    mr.track(first, a, path.stat(), 0, p)
    mr.track(second, b, path.stat(), 2 * p, 3 * p)
    # Resident pages may all belong to the omitted bytes: grant nothing.
    monkeypatch.setattr(mr, "_resident_file_bytes", lambda a, n: 2 * p)
    assert mr.resident_mapped_bytes() == 0
    monkeypatch.setattr(mr, "_resident_file_bytes", lambda a, n: 3 * p)
    assert mr.resident_mapped_bytes() == p
    monkeypatch.setattr(mr, "_resident_file_bytes", lambda a, n: 4 * p)
    assert mr.resident_mapped_bytes() == 2 * p


def test_in_place_file_growth_invalidates_whole_object_counter(tmp_path, monkeypatch):
    path = tmp_path / "growing"
    p = mr._PAGE
    path.write_bytes(b"a" * (p * 4))
    mapping = np.memmap(path, mode="r", dtype=np.uint8)
    mr.track(mapping, mapping, path.stat(), 0, 4 * p)
    monkeypatch.setattr(mr, "_resident_file_bytes", lambda a, n: n)
    assert mr.resident_mapped_bytes() == 4 * p
    with path.open("ab") as out:
        out.write(b"b" * p)
    assert mr.resident_mapped_bytes() == 0
