# SPDX-License-Identifier: Apache-2.0
"""Real MLX regressions for file-backed weights and scoped loading."""

import gc
import hashlib
import json
import struct
import weakref
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm import utils as lm_utils

from omlx.utils import mapped_weights as mw


def write(path, tensors, shift=0):
    header, data, cursor = {}, [], 0
    for name, (dtype, values) in tensors.items():
        raw = values.tobytes()
        header[name] = dict(
            dtype=dtype,
            shape=list(values.shape),
            data_offsets=[cursor, cursor + len(raw)],
        )
        data.append(raw)
        cursor += len(raw)
    header["__metadata__"] = {"format": "mlx"}
    encoded = json.dumps(header).encode()
    encoded += b" " * ((shift - 8 - len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(data))


def checkpoint(tmp_path, shift=0):
    path = tmp_path / "model.safetensors"
    write(path, {"norm": ("F32", np.arange(64, dtype=np.float32))}, shift)
    return path


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
def test_sanitize_cannot_donate_readonly_weights(tmp_path, device):
    path = checkpoint(tmp_path)
    before = hashlib.sha256(path.read_bytes()).digest()
    with mw.mapped_loading(tmp_path), mx.stream(device):
        # Discard the original mx.array, exactly like sanitize's v + 1.
        weights = lm_utils.mx.load(str(path))
        weights["norm"] = weights["norm"] + 1.0
        mx.eval(weights)
        assert weights["norm"].tolist() == list(range(1, 65))
    assert hashlib.sha256(path.read_bytes()).digest() == before


def test_u8_reinterpreted_as_uint32_is_correct_on_gpu(tmp_path):
    path = tmp_path / "model.safetensors"
    raw = np.arange(32, dtype=np.uint8)
    write(path, {"blocks": ("U8", raw)}, shift=1)
    with mw.mapped_loading(tmp_path):
        weights = lm_utils.mx.load(str(path))
    actual = weights["blocks"].view(mx.uint32) * 1
    mx.eval(actual)
    assert actual.tolist() == raw.view(np.uint32).tolist()


def test_misalignment_copies_without_rewriting(tmp_path, caplog):
    path = checkpoint(tmp_path, shift=3)
    before = path.read_bytes()
    with mw.mapped_loading(tmp_path):
        weights = lm_utils.mx.load(str(path))
    assert (weights["norm"] * 1).tolist() == list(range(64))
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_mapping_released_after_last_array_not_context(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    refs = []
    original = np.memmap

    def track(*args, **kwargs):
        result = original(*args, **kwargs)
        refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(np, "memmap", track)
    with mw.mapped_loading(tmp_path):
        weights = lm_utils.mx.load(str(path))
    gc.collect()
    assert refs[0]() is not None
    assert mx.sum(weights["norm"]).item() == sum(range(64))
    del weights
    mx.synchronize()
    mx.clear_cache()
    gc.collect()
    assert refs[0]() is None


def test_other_thread_and_directory_and_stream_use_original(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    calls = []
    original = mx.load

    def load(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(mx, "load", load)
    other = tmp_path / "other"
    other.mkdir()
    other_path = checkpoint(other)
    with mw.mapped_loading(tmp_path):
        assert mx.load is load
        lm_utils.mx.load(str(path))
        assert not calls
        lm_utils.mx.load(str(other_path))
        lm_utils.mx.load(str(path), stream=mx.cpu)
        with ThreadPoolExecutor(1) as pool:
            pool.submit(lm_utils.mx.load, str(path)).result()
    assert calls == [str(other_path), str(path), str(path)]
    assert mx.load is load


def test_failure_restores_loader(tmp_path):
    original = mx.load
    with pytest.raises(RuntimeError), mw.mapped_loading(tmp_path):
        raise RuntimeError("sanitizer failed")
    assert mx.load is original


def test_metadata_and_positional_signature(tmp_path):
    path = checkpoint(tmp_path)
    with mw.mapped_loading(tmp_path):
        weights, metadata = lm_utils.mx.load(str(path), "safetensors", True)
    assert metadata == {"format": "mlx"}
    assert weights["norm"].shape == (64,)


@pytest.mark.parametrize(
    "kind", ["negative", "overlap", "shape", "trailing", "metadata", "header"]
)
def test_invalid_checkpoint_rejected_before_mapping(tmp_path, monkeypatch, kind):
    path = checkpoint(tmp_path)
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + n])
    data = raw[8 + n :]
    if kind == "negative":
        header["norm"]["data_offsets"] = [-1, 255]
    if kind == "overlap":
        header["other"] = header["norm"].copy()
    if kind == "shape":
        header["norm"]["shape"] = [63]
    if kind == "trailing":
        data += b"junk"
    if kind == "metadata":
        header["__metadata__"] = {"bad": 42}
    encoded = json.dumps(header).encode()
    path.write_bytes(
        struct.pack("<Q", 10**10 if kind == "header" else len(encoded)) + encoded + data
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("mapped invalid file")

    monkeypatch.setattr(np, "memmap", unexpected)
    with mw.mapped_loading(tmp_path), pytest.raises(ValueError):
        lm_utils.mx.load(str(path))


def test_bf16_and_empty_tensor(tmp_path):
    path = tmp_path / "model.safetensors"
    weights = {"bf": mx.arange(64).astype(mx.bfloat16), "empty": mx.zeros((0, 2))}
    mx.save_safetensors(str(path), weights)
    with mw.mapped_loading(tmp_path):
        actual = lm_utils.mx.load(str(path))
    for key in weights:
        assert actual[key].dtype == weights[key].dtype
        assert mx.array_equal(actual[key] * 1, weights[key]).item()


def test_disabled_context_is_a_callable_noop(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from omlx.engine.base import BaseEngine

    engine = SimpleNamespace(_model_name=str(tmp_path), _model_settings=None)
    original = mx.load

    @BaseEngine._mapped_weight_loading(engine)
    def load():
        assert mx.load is original
        return 42

    assert load() == 42


def test_aligned_import_is_zero_copy_and_never_changes_wired_limit(
    tmp_path, monkeypatch
):
    path = checkpoint(tmp_path)

    def unexpected(*args):
        raise AssertionError("changed process-wide residency")

    monkeypatch.setattr(mx, "set_wired_limit", unexpected)
    with mw.mapped_loading(tmp_path):
        weights = lm_utils.mx.load(str(path))
    # Only our generated test file is edited, through a separate descriptor.
    raw = path.read_bytes()
    start = 8 + struct.unpack("<Q", raw[:8])[0]
    with path.open("r+b") as source:
        source.seek(start)
        source.write(np.array([99], dtype=np.float32).tobytes())
    assert (weights["norm"] * 1)[0].item() == 99


def test_old_mlx_is_refused_before_hook_installation(tmp_path, monkeypatch):
    monkeypatch.setattr(mw, "version", lambda _: "0.32.3")
    original = mx.load
    with (
        pytest.raises(RuntimeError, match="MLX >= 0.32.4"),
        mw.mapped_loading(tmp_path),
    ):
        pass
    assert mx.load is original


def test_scoped_proxy_restores_both_modules_and_leaves_core_alone(tmp_path):
    from mlx_vlm import utils as vlm_utils

    core_load = mx.load
    original_lm, original_vlm = lm_utils.mx, vlm_utils.mx
    with pytest.raises(RuntimeError), mw.mapped_loading(tmp_path):
        assert mx.load is core_load
        assert lm_utils.mx is not original_lm
        assert vlm_utils.mx is not original_vlm
        raise RuntimeError("loader failed")
    assert lm_utils.mx is original_lm
    assert vlm_utils.mx is original_vlm
    assert mx.load is core_load


def test_alias_unload_and_lazy_derived_array_keep_their_own_mapping(tmp_path):
    path = checkpoint(tmp_path)
    with mw.mapped_loading(tmp_path):
        first = lm_utils.mx.load(str(path))
        second = lm_utils.mx.load(str(path))
    derived = second["norm"][8:24].reshape(4, 4)
    del first, second
    gc.collect()
    assert (derived + 1).tolist() == np.arange(9, 25).reshape(4, 4).tolist()


def test_unknown_dtype_keeps_native_loader_error(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    write(path, {"x": ("F8_E5M2", np.arange(8, dtype=np.uint8))})
    original = mx.load
    calls = []

    def native(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(mx, "load", native)
    with (
        mw.mapped_loading(tmp_path),
        pytest.raises(RuntimeError, match="unsupported dtype"),
    ):
        lm_utils.mx.load(str(path))
    assert calls == [str(path)]


def test_coverage_reports_mappable_bytes_from_headers(tmp_path):
    aligned, misaligned = tmp_path / "aligned", tmp_path / "misaligned"
    aligned.mkdir()
    misaligned.mkdir()
    checkpoint(aligned)
    checkpoint(misaligned, shift=3)
    assert mw.mapped_coverage(aligned) == (256, 256)
    assert mw.mapped_coverage(misaligned) == (0, 256)
    assert mw.mapped_coverage(tmp_path / "missing") is None
    assert mw.mapped_coverage(tmp_path) is None


def test_coverage_follows_a_rewritten_checkpoint(tmp_path):
    path = checkpoint(tmp_path, shift=3)
    assert mw.mapped_coverage(tmp_path) == (0, 256)
    write(path, {"norm": ("F32", np.arange(128, dtype=np.float32))})
    assert mw.mapped_coverage(tmp_path) == (512, 512)


@pytest.mark.parametrize(
    "header",
    [[], {"x": []}, {"x": {"data_offsets": [8, 0]}}, {"x": {"data_offsets": "0"}}],
)
def test_coverage_returns_unknown_for_invalid_headers(tmp_path, header):
    encoded = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded
    )
    assert mw.mapped_coverage(tmp_path) is None


def _record_native_loads(monkeypatch):
    calls, original = [], mx.load

    def native(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(mx, "load", native)
    return calls


def test_shard_without_aligned_tensors_uses_native_loader(tmp_path, monkeypatch):
    path = checkpoint(tmp_path, shift=3)
    calls = _record_native_loads(monkeypatch)
    with mw.mapped_loading(tmp_path):
        weights = lm_utils.mx.load(str(path))
    assert calls == [str(path)]
    assert weights["norm"].tolist() == list(range(64))


def test_partly_aligned_shard_still_maps(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    write(
        path,
        {
            "a": ("U8", np.arange(3, dtype=np.uint8)),
            "b": ("F32", np.arange(4, dtype=np.float32)),
        },
    )
    calls = _record_native_loads(monkeypatch)
    with mw.mapped_loading(tmp_path):
        weights = lm_utils.mx.load(str(path))
    assert calls == []
    assert weights["a"].tolist() == [0, 1, 2]
    assert (weights["b"] * 1).tolist() == [0, 1, 2, 3]


def test_padding_between_tensors_maps_zero_copy(tmp_path, monkeypatch):
    # Aligned checkpoints pad the data so that every tensor starts at 8 bytes.
    path = tmp_path / "model.safetensors"
    a, b = np.arange(3, dtype=np.uint8), np.arange(4, dtype=np.float32)
    header = {
        "a": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
        "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
    }
    encoded = json.dumps(header).encode()
    encoded += b" " * ((-8 - len(encoded)) % 8)
    data = a.tobytes() + b"\0" * 5 + b.tobytes()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    calls = _record_native_loads(monkeypatch)
    with mw.mapped_loading(tmp_path):
        weights = lm_utils.mx.load(str(path))
    assert calls == []
    assert weights["a"].tolist() == [0, 1, 2]
    assert (weights["b"] * 1).tolist() == [0, 1, 2, 3]
    assert mw.mapped_coverage(tmp_path) == (19, 19)
