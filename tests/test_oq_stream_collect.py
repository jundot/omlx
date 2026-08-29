# SPDX-License-Identifier: Apache-2.0
"""Layer-streaming imatrix collector: parity with the resident collector.

The streaming collector walks rounds on the outside and layers on the
inside, holding one decoder layer resident at a time. The imatrix statistic
is a pure additive token sum, so that traversal must reproduce the
whole-model sample-outer collector bit for bit when both sides see the same
calibration draw and the same micro-batch partitioning.

These tests pin that contract on a truncated bf16 MiniMax-M3: the original
layers 0..3 (three dense layers plus the first MoE layer, which also carries
the sparse-attention index projections), dequantized from the real MXFP8
checkpoint and written as a standalone model that both collectors load
identically. The fixture is cached on disk and reused across runs; the
module skips when neither the cached fixture nor the source checkpoint is
present.
"""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import omlx.oq as oq
from omlx.oq import (
    _OQE_CALIB_DATASET,
    OQImatrixData,
    _collect_imatrix_from_model,
    _collect_imatrix_streaming,
    _imatrix_metadata_path,
    _LazyTensorIndex,
    _load_or_collect_imatrix,
    _lookup_imatrix_importance,
    _resolve_stream_calibration,
    _streamed_source_plan,
)

M3_DIR = Path("/Volumes/Scratch/models/MiniMax-M3-MXFP8")
FIXTURE_DIR = Path("/Volumes/Scratch/omlx-050/fixtures/minimax-m3-trunc4-bf16")
FIXTURE_DONE = FIXTURE_DIR / "fixture_meta.json"

KEEP_LAYERS = 4
LAYER_PREFIX = "language_model.model.layers."
CALIB_SEED = 20260709
NUM_SAMPLES = 32
SEQ_LENGTH = 512
MICRO_BATCH = 8

pytestmark = pytest.mark.skipif(
    not M3_DIR.is_dir() and not FIXTURE_DONE.exists(),
    reason="neither MiniMax-M3-MXFP8 nor the cached truncated fixture is present",
)


# --- fixture model build ----------------------------------------------------

_KEPT_SINGLETONS = (
    "language_model.model.embed_tokens.weight",
    "language_model.model.norm.weight",
    "language_model.lm_head.weight",
)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "chat_template.jinja",
)
_SHARD_BUDGET = 4 * 1024**3


def _keep_key(key: str) -> bool:
    if key in _KEPT_SINGLETONS:
        return True
    if not key.startswith(LAYER_PREFIX):
        return False
    layer = key[len(LAYER_PREFIX) :].split(".", 1)[0]
    return layer.isdigit() and int(layer) < KEEP_LAYERS


def _truncated_config() -> dict:
    """Source config cut down to KEEP_LAYERS, bf16, schedule preserved."""
    config = json.loads((M3_DIR / "config.json").read_text())
    config.pop("quantization_config", None)
    text = config["text_config"]
    text["num_hidden_layers"] = KEEP_LAYERS
    if isinstance(text.get("moe_layer_freq"), list):
        text["moe_layer_freq"] = text["moe_layer_freq"][:KEEP_LAYERS]
    sparse = text.get("sparse_attention_config")
    if isinstance(sparse, dict):
        for key in ("sparse_disable_index_value", "sparse_attention_freq"):
            if isinstance(sparse.get(key), list):
                sparse[key] = sparse[key][:KEEP_LAYERS]
    return config


def _build_fixture() -> None:
    """Dequantize layers 0..3 plus embed/norm/lm_head into a bf16 checkpoint.

    Goes through _LazyTensorIndex so the fp8 pairs take the exact dequant
    path the streaming sourcer uses (weight_scale_inv U8 e8m0 decode). The
    output keeps the original per-expert key layout, so both mlx-vlm load
    and the streamed sanitize plan assemble the MoE stacks themselves.
    """
    idx = _LazyTensorIndex(sorted(M3_DIR.glob("*.safetensors")))
    logical = idx.logical_metadata()
    kept = sorted(k for k in logical if _keep_key(k))
    assert kept, "no fixture keys found in the source checkpoint"

    dtype_bytes = _LazyTensorIndex._DTYPE_BYTES
    sizes = {}
    for key in kept:
        shape, dtype = logical[key]
        n = 1
        for dim in shape:
            n *= dim
        sizes[key] = n * dtype_bytes[dtype]

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key in kept:
        if current and current_bytes + sizes[key] > _SHARD_BUDGET:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(key)
        current_bytes += sizes[key]
    if current:
        shards.append(current)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for i, keys in enumerate(shards, start=1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        tensors = {}
        for key in keys:
            tensor = idx[key]
            mx.eval(tensor)
            tensors[key] = tensor
            weight_map[key] = name
        mx.save_safetensors(str(FIXTURE_DIR / name), tensors)
        del tensors
        mx.clear_cache()

    (FIXTURE_DIR / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(sizes.values())},
                "weight_map": weight_map,
            },
            indent=2,
        )
    )
    (FIXTURE_DIR / "config.json").write_text(json.dumps(_truncated_config(), indent=2))
    for name in _TOKENIZER_FILES:
        src = M3_DIR / name
        if src.exists():
            shutil.copy2(src, FIXTURE_DIR / name)
    FIXTURE_DONE.write_text(
        json.dumps({"source": str(M3_DIR), "keep_layers": KEEP_LAYERS})
    )


@pytest.fixture(scope="session")
def fixture_dir():
    if not FIXTURE_DONE.exists():
        if not M3_DIR.is_dir():
            pytest.skip("fixture not built and the source checkpoint is absent")
        if FIXTURE_DIR.exists():
            shutil.rmtree(FIXTURE_DIR)
        _build_fixture()
    return FIXTURE_DIR


@pytest.fixture(scope="module")
def fixture_config(fixture_dir):
    return json.loads((fixture_dir / "config.json").read_text())


@pytest.fixture(scope="module")
def tokenizer(fixture_dir):
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    return load_tokenizer(fixture_dir)


# --- collection helpers -----------------------------------------------------


@contextmanager
def _pinned_micro_batch(size: int):
    """Pin the calibration micro-batch so both collectors partition alike.

    _oqe_calibration_batch_plan reads live free memory, so an unpinned run
    could legally pick different micro-batch sizes for the two collectors
    and break fp32 summation-grouping parity.
    """
    original = oq._oqe_calibration_batch_plan

    def pinned(config, **kwargs):
        plan = original(config, **kwargs)
        plan["micro_batch_size"] = int(size)
        return plan

    oq._oqe_calibration_batch_plan = pinned
    try:
        yield
    finally:
        oq._oqe_calibration_batch_plan = original


def _load_resident_model(model_dir: Path):
    """Load the fixture the way _collect_imatrix loads a VLM source."""
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(model_dir), for_vlm=True)

    from mlx_vlm.utils import load_model as vlm_load_model

    orig_load_weights = nn.Module.load_weights

    def lenient_load_weights(self, file_or_weights, *args, **kwargs):
        kwargs.pop("strict", None)
        return orig_load_weights(self, file_or_weights, *args, strict=False, **kwargs)

    nn.Module.load_weights = lenient_load_weights
    try:
        return vlm_load_model(Path(model_dir), lazy=True)
    finally:
        nn.Module.load_weights = orig_load_weights


@pytest.fixture(scope="module")
def resident_result(fixture_dir, fixture_config, tokenizer):
    """Entries and metadata from the existing whole-model collector."""
    model = _load_resident_model(fixture_dir)
    try:
        with _pinned_micro_batch(MICRO_BATCH):
            # The subsample permutation inside _load_calibration_data is
            # unseeded; seeding right before the collect pins the draw to
            # the same one the streaming run makes with calib_seed.
            mx.random.seed(CALIB_SEED)
            entries, metadata = _collect_imatrix_from_model(
                model,
                tokenizer,
                fixture_config,
                calib_dataset=_OQE_CALIB_DATASET,
                num_samples=NUM_SAMPLES,
                seq_length=SEQ_LENGTH,
            )
    finally:
        del model
        mx.synchronize()
        mx.clear_cache()
    return entries, metadata


@pytest.fixture(scope="module")
def streaming_result(fixture_dir, fixture_config, tokenizer):
    """Entries and metadata from the layer-streaming collector."""
    with _pinned_micro_batch(MICRO_BATCH):
        entries, metadata = _collect_imatrix_streaming(
            fixture_dir,
            tokenizer,
            fixture_config,
            calib_dataset=_OQE_CALIB_DATASET,
            num_samples=NUM_SAMPLES,
            seq_length=SEQ_LENGTH,
            calib_seed=CALIB_SEED,
        )
    mx.synchronize()
    mx.clear_cache()
    return entries, metadata


# --- tests -------------------------------------------------------------------


def test_stream_calibration_flag_resolution(monkeypatch):
    """Explicit argument beats the env var, env var beats the RAM auto-rule.

    Runs on a supported model_type so the precedence, not the layout gate, is
    what these assertions exercise. The gate itself lives in
    test_oq_stream_resolve.py.
    """
    monkeypatch.delenv("OMLX_OQ_STREAM_CALIBRATION", raising=False)
    mt = "minimax_m3_vl"
    assert (
        _resolve_stream_calibration(True, model_exceeds_ram=False, model_type=mt)
        is True
    )
    assert (
        _resolve_stream_calibration(False, model_exceeds_ram=True, model_type=mt)
        is False
    )
    assert (
        _resolve_stream_calibration(None, model_exceeds_ram=True, model_type=mt) is True
    )
    assert (
        _resolve_stream_calibration(None, model_exceeds_ram=False, model_type=mt)
        is False
    )

    monkeypatch.setenv("OMLX_OQ_STREAM_CALIBRATION", "1")
    assert (
        _resolve_stream_calibration(None, model_exceeds_ram=False, model_type=mt)
        is True
    )
    monkeypatch.setenv("OMLX_OQ_STREAM_CALIBRATION", "off")
    assert (
        _resolve_stream_calibration(None, model_exceeds_ram=True, model_type=mt)
        is False
    )
    monkeypatch.setenv("OMLX_OQ_STREAM_CALIBRATION", "0")
    assert (
        _resolve_stream_calibration(True, model_exceeds_ram=False, model_type=mt)
        is True
    )


def test_streaming_matches_resident_imatrix(resident_result, streaming_result):
    """THE GATE: layer-outer must equal sample-outer bit for bit."""
    res_entries, res_meta = resident_result
    st_entries, st_meta = streaming_result

    # Parity precondition: coverage must be reached inside round one on
    # both sides. Past round one the resident collector may stop mid-round
    # (per micro-batch) while the streaming collector only stops at round
    # boundaries, so the sample multisets legitimately diverge. If this
    # fires, raise NUM_SAMPLES rather than loosening the comparison.
    assert res_meta["coverage_sufficient"], res_meta["coverage"]
    assert len(res_meta["rounds"]) == 1, res_meta["rounds"]
    assert len(st_meta["rounds"]) == 1, st_meta["rounds"]
    assert res_meta["processed_samples"] == st_meta["processed_samples"] == NUM_SAMPLES
    assert res_meta["micro_batch_size"] == st_meta["micro_batch_size"] == MICRO_BATCH

    assert st_meta["load_kind"] == "streaming"
    assert set(st_meta) == set(res_meta) | {"load_kind"}

    assert st_entries, "streaming collector produced no entries"
    assert set(st_entries) == set(res_entries)

    moe_entries = 0
    diffs = []
    for name in sorted(res_entries):
        res, st = res_entries[name], st_entries[name]
        assert res.in_sum2.dtype == np.float32 == st.in_sum2.dtype, name
        assert res.counts.dtype == np.int64 == st.counts.dtype, name
        assert res.in_sum2.shape == st.in_sum2.shape, name
        # Counts are integers (per-expert routing tallies for the MoE
        # entries). No tolerance is acceptable here.
        assert np.array_equal(res.counts, st.counts), f"{name}: routing counts diverge"
        if res.counts.size > 1:
            moe_entries += 1
        if not np.array_equal(res.in_sum2, st.in_sum2):
            rel = np.abs(st.in_sum2 - res.in_sum2) / np.maximum(
                np.abs(res.in_sum2), 1e-30
            )
            diffs.append((float(rel.max()), name))
    assert moe_entries == 2, (
        "expected exactly gate_up_proj and down_proj per-expert entries, "
        f"got {moe_entries}"
    )

    if diffs:
        diffs.sort(reverse=True)
        worst_rel, worst_name = diffs[0]
        print(
            f"\nimatrix parity: bitwise equality failed on {len(diffs)} of "
            f"{len(res_entries)} entries; max rel diff {worst_rel:.3e} at "
            f"{worst_name}. Suspects: eval-order or mask-branch divergence "
            "between the resident and streamed forwards."
        )
        for _, name in diffs:
            res, st = res_entries[name], st_entries[name]
            assert np.allclose(st.in_sum2, res.in_sum2, rtol=1e-5, atol=0.0), (
                f"{name}: in_sum2 rel diff exceeds 1e-5"
            )


def test_streaming_entry_keys_prefixed(fixture_dir, fixture_config, streaming_result):
    """Streamed entries carry full model paths and feed importance lookup."""
    entries, _ = streaming_result

    assert all(name.startswith(LAYER_PREFIX) for name in entries)
    layers_seen = {name[len(LAYER_PREFIX) :].split(".", 1)[0] for name in entries}
    assert layers_seen == {str(i) for i in range(KEEP_LAYERS)}

    dp = _streamed_source_plan(fixture_dir, fixture_config)
    quantizable = []
    for key in dp:
        if not key.startswith(LAYER_PREFIX) or not key.endswith(".weight"):
            continue
        shape = dp.plan_shape(key)
        if len(shape) >= 2:
            quantizable.append((key, shape))
    assert quantizable

    # Exact correspondence: every quantizable layer tensor has an entry,
    # and there are no orphan entries.
    assert {key[: -len(".weight")] for key, _ in quantizable} == set(entries)

    data = OQImatrixData(entries=entries, metadata={}, path="")
    report = {"applied": [], "missing": [], "mismatched": [], "zero_count_experts": 0}
    for key, shape in quantizable:
        importance = _lookup_imatrix_importance(
            data, key, tuple(shape), strict=False, report=report
        )
        assert importance is not None, f"no importance resolved for {key}"
    assert report["missing"] == []
    assert report["mismatched"] == []


def test_streaming_out_none_is_fatal(fixture_dir, fixture_config, monkeypatch):
    """A layer forward returning None must abort the collection loudly."""
    monkeypatch.setattr(oq, "_forward_layer_result", lambda *a, **k: (None, None))
    calib = mx.zeros((2, 8), dtype=mx.int32)
    with pytest.raises(RuntimeError, match="forward"):
        _collect_imatrix_streaming(
            fixture_dir,
            None,
            fixture_config,
            calib_dataset=_OQE_CALIB_DATASET,
            num_samples=2,
            seq_length=8,
            calib_data=calib,
        )


def test_cache_load_kind_busts_legacy(
    fixture_dir, fixture_config, tmp_path, monkeypatch
):
    """Streaming runs reject caches that were not stream-collected."""
    cache_path = tmp_path / "m3-trunc4.npz"
    common = dict(
        cache_path=str(cache_path),
        reuse_cache=True,
        num_samples=2,
        seq_length=64,
        strict=False,
        trust_remote_code=False,
    )

    first = _load_or_collect_imatrix(
        str(fixture_dir), fixture_config, **common, stream_calibration=True
    )
    assert first.reused is False
    assert first.metadata["load_kind"] == "streaming"
    assert first.metadata["collection"]["load_kind"] == "streaming"

    # A streaming-collected npz round-trips.
    second = _load_or_collect_imatrix(
        str(fixture_dir), fixture_config, **common, stream_calibration=True
    )
    assert second.reused is True
    assert set(second.entries) == set(first.entries)

    # Recollections from here on are stubbed; these cases only assert the
    # cache-validation decision.
    calls = []

    def fake_stream(source, tokenizer, config, **kwargs):
        calls.append(kwargs)
        return dict(first.entries), dict(first.metadata["collection"])

    monkeypatch.setattr(oq, "_collect_imatrix_streaming", fake_stream)

    meta_path = _imatrix_metadata_path(cache_path)
    saved = json.loads(meta_path.read_text())

    # Legacy cache: no load_kind anywhere. Must be recollected.
    legacy = json.loads(json.dumps(saved))
    legacy.pop("load_kind", None)
    legacy.get("collection", {}).pop("load_kind", None)
    meta_path.write_text(json.dumps(legacy))
    third = _load_or_collect_imatrix(
        str(fixture_dir), fixture_config, **common, stream_calibration=True
    )
    assert len(calls) == 1, "legacy cache without load_kind must be recollected"
    assert third.reused is False
    assert third.metadata["load_kind"] == "streaming"

    # Mismatching load_kind (a resident-collected cache). Must be recollected.
    mismatched = json.loads(json.dumps(saved))
    mismatched["load_kind"] = "resident"
    mismatched["collection"]["load_kind"] = "resident"
    meta_path.write_text(json.dumps(mismatched))
    fourth = _load_or_collect_imatrix(
        str(fixture_dir), fixture_config, **common, stream_calibration=True
    )
    assert len(calls) == 2, "resident-collected cache must be recollected"
    assert fourth.reused is False

    # The non-streaming path keeps accepting a legacy cache unchanged.
    meta_path.write_text(json.dumps(legacy))

    def no_resident_collect(*args, **kwargs):
        raise AssertionError("resident collection must not run on a cache hit")

    monkeypatch.setattr(oq, "_collect_imatrix", no_resident_collect)
    fifth = _load_or_collect_imatrix(
        str(fixture_dir), fixture_config, **common, stream_calibration=False
    )
    assert fifth.reused is True


def test_streaming_aborts_on_first_layer_zero_install(
    fixture_dir, fixture_config, tokenizer, monkeypatch
):
    """A source whose layout defeats the capture predicate must abort on the
    first layer, not after streaming every layer for an empty imatrix."""
    monkeypatch.setattr(oq.OQImatrixCollector, "install", lambda self, *a, **k: 0)
    with (
        _pinned_micro_batch(MICRO_BATCH),
        pytest.raises(RuntimeError, match="installed 0 capture modules"),
    ):
        _collect_imatrix_streaming(
            fixture_dir,
            tokenizer,
            fixture_config,
            calib_dataset=_OQE_CALIB_DATASET,
            num_samples=NUM_SAMPLES,
            seq_length=SEQ_LENGTH,
            calib_seed=CALIB_SEED,
        )
    mx.synchronize()
    mx.clear_cache()
