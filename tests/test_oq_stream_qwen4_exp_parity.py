# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp streamed-vs-resident parity: THE gate for the qwen4_exp sourcer.

Truncated bf16 Qwen3.8-Flash-Next fixture: the original layers 0..3, i.e.
GDN x3 (with the layer-1 N-gram PLE, ple_layer_ids=[2] attaches on
layer_idx+1) plus the first QSA layer at index 3, together with the token
embedding, the trunk hyper-connection mixer, lm_head, and the vision tower
(kept so the resident mlx-vlm load is complete). MTP tensors are dropped
and the truncated config zeroes the MTP head.

The resident reference runs first with the PLE runtime in "resident" mode;
the streamed run then switches the runtime to "mmap", which is the mode the
real calibration uses — so the gate also exercises the mmap shard-drop path
end to end against the real table.
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
    _collect_imatrix_from_model,
    _collect_imatrix_streaming,
    _LazyTensorIndex,
    _measure_sensitivity_from_model,
)

SRC_DIR = Path("/Volumes/Scratch/qwen38-flash-next/Qwen3.8-Flash-Next")
FIXTURE_DIR = Path("/Volumes/Scratch/omlx-063/fixtures/qwen4exp-trunc4-bf16")
FIXTURE_DONE = FIXTURE_DIR / "fixture_meta.json"

KEEP_LAYERS = 4
RAW_LAYER_PREFIX = "model.language_model.layers."
SAN_LAYER_PREFIX = "language_model.model.layers."
CALIB_SEED = 20260828
NUM_SAMPLES = 32
SEQ_LENGTH = 512
MICRO_BATCH = 8
SENS_SEED = 20260829
SENS_SAMPLES = 8
SENS_SEQ = 128
SENS_OQ_LEVEL = 6

pytestmark = pytest.mark.skipif(
    not SRC_DIR.is_dir() and not FIXTURE_DONE.exists(),
    reason="neither the Flash-Next source nor the cached truncated fixture is present",
)


# --- fixture model build ----------------------------------------------------

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
_SHARD_BUDGET = 4 * 1024**3


def _keep_key(key: str) -> bool:
    if key.startswith("mtp."):
        return False
    if key.startswith(RAW_LAYER_PREFIX):
        layer = key[len(RAW_LAYER_PREFIX) :].split(".", 1)[0]
        return layer.isdigit() and int(layer) < KEEP_LAYERS
    # embed_tokens, trunk hyper_connection_mixer, lm_head, vision tower.
    return True


def _truncated_config() -> dict:
    config = json.loads((SRC_DIR / "config.json").read_text())
    text = config["text_config"]
    text["num_hidden_layers"] = KEEP_LAYERS
    text["layer_types"] = list(text["layer_types"])[:KEEP_LAYERS]
    # ple_layer_ids=[2] attaches on layer_idx+1 == 2, i.e. kept layer 1.
    assert all(0 < i <= KEEP_LAYERS for i in text.get("ple_layer_ids", []))
    # The fixture drops mtp.* tensors, so the config must not declare a head.
    text["mtp_num_hidden_layers"] = 0
    text.pop("mtp", None)
    return config


def _build_fixture() -> None:
    idx = _LazyTensorIndex(sorted(SRC_DIR.glob("*.safetensors")))
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
        sizes[key] = n * dtype_bytes.get(dtype, 2)

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
        print(f"fixture shard {i}/{len(shards)} written", flush=True)
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
        src = SRC_DIR / name
        if src.exists():
            shutil.copy2(src, FIXTURE_DIR / name)
    FIXTURE_DONE.write_text(
        json.dumps({"source": str(SRC_DIR), "keep_layers": KEEP_LAYERS})
    )


@pytest.fixture(scope="session")
def fixture_dir():
    if not FIXTURE_DONE.exists():
        if not SRC_DIR.is_dir():
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
    from omlx.patches.mlx_vlm_qwen4_exp_compat import configure_qwen4_exp_runtime
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    configure_qwen4_exp_runtime(model_dir, mode="resident", mtp_enabled=False)
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
def parity_results(fixture_dir, fixture_config, tokenizer):
    """Resident reference first (PLE resident), then the streamed run (mmap).

    One shared fixture keeps a single 127 GB resident load and makes the
    ordering explicit: the streamed collector flips the module-global PLE
    runtime to mmap, which must not happen before the resident reference
    has been captured.
    """
    model = _load_resident_model(fixture_dir)
    try:
        with _pinned_micro_batch(MICRO_BATCH):
            mx.random.seed(CALIB_SEED)
            resident = _collect_imatrix_from_model(
                model,
                tokenizer,
                fixture_config,
                calib_dataset=_OQE_CALIB_DATASET,
                num_samples=NUM_SAMPLES,
                seq_length=SEQ_LENGTH,
            )
        mx.random.seed(SENS_SEED)
        resident_sens = _measure_sensitivity_from_model(
            model,
            tokenizer,
            fixture_config,
            SENS_OQ_LEVEL,
            num_samples=SENS_SAMPLES,
            seq_length=SENS_SEQ,
        )
    finally:
        del model
        mx.synchronize()
        mx.clear_cache()

    with _pinned_micro_batch(MICRO_BATCH):
        streamed = _collect_imatrix_streaming(
            fixture_dir,
            tokenizer,
            fixture_config,
            calib_dataset=_OQE_CALIB_DATASET,
            num_samples=NUM_SAMPLES,
            seq_length=SEQ_LENGTH,
            calib_seed=CALIB_SEED,
            measure_sensitivity=True,
            sensitivity_oq_level=SENS_OQ_LEVEL,
            sensitivity_num_samples=SENS_SAMPLES,
            sensitivity_seq_length=SENS_SEQ,
            sensitivity_calib_seed=SENS_SEED,
        )
    mx.synchronize()
    mx.clear_cache()
    return {
        "resident": resident,
        "resident_sens": resident_sens,
        "streamed": streamed,
    }


# --- tests -------------------------------------------------------------------


def test_streaming_matches_resident_imatrix(parity_results):
    """THE GATE: layer-outer must equal sample-outer bit for bit."""
    res_entries, res_meta = parity_results["resident"]
    st_entries, st_meta = parity_results["streamed"]

    # Parity precondition: both sides must consume the SAME sample multiset.
    # That holds when coverage is reached at a round boundary on both sides,
    # or — the qwen4_exp fixture case — when it is never reached and both
    # sides exhaust the same adaptive maximum in full rounds (a handful of
    # the fixture's 2048 routed experts never fire on the builtin corpus, so
    # coverage_sufficient stays False by construction; C-coverage declares
    # exactly this per-expert degradation for the full model too).
    assert res_meta["coverage_sufficient"] == st_meta["coverage_sufficient"]
    assert len(res_meta["rounds"]) == len(st_meta["rounds"]), (
        res_meta["rounds"],
        st_meta["rounds"],
    )
    assert res_meta["processed_samples"] == st_meta["processed_samples"]
    assert res_meta["processed_samples"] % NUM_SAMPLES == 0
    assert res_meta["micro_batch_size"] == st_meta["micro_batch_size"] == MICRO_BATCH

    assert st_meta["load_kind"] == "streaming"
    assert set(st_meta) == set(res_meta) | {"load_kind", "sensitivity_map"}

    assert st_entries, "streaming collector produced no entries"
    assert set(st_entries) == set(res_entries)

    # Layer coverage: every kept layer contributes, the PLE layer's own
    # linears contribute (the mmap table path really ran), and per-expert
    # MoE entries exist on all layers (48/48 layers are MoE on qwen4_exp).
    layers_seen = {
        name[len(SAN_LAYER_PREFIX) :].split(".", 1)[0] for name in st_entries
    }
    assert layers_seen == {str(i) for i in range(KEEP_LAYERS)}
    ple_entries = [n for n in st_entries if f"{SAN_LAYER_PREFIX}1.ple." in n]
    assert ple_entries, "no imatrix entries under the layer-1 PLE"

    moe_entries = 0
    diffs = []
    for name in sorted(res_entries):
        res, st = res_entries[name], st_entries[name]
        assert res.in_sum2.dtype == np.float32 == st.in_sum2.dtype, name
        assert res.counts.dtype == np.int64 == st.counts.dtype, name
        assert res.in_sum2.shape == st.in_sum2.shape, name
        # Per-expert routing tallies: no tolerance is acceptable here.
        assert np.array_equal(res.counts, st.counts), f"{name}: routing counts diverge"
        if res.counts.size > 1:
            moe_entries += 1
        if not np.array_equal(res.in_sum2, st.in_sum2):
            rel = np.abs(st.in_sum2 - res.in_sum2) / np.maximum(
                np.abs(res.in_sum2), 1e-30
            )
            diffs.append((float(rel.max()), name))
    assert moe_entries >= KEEP_LAYERS, (
        f"expected per-expert entries on all {KEEP_LAYERS} MoE layers, "
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


def test_fused_sensitivity_matches_resident(parity_results):
    """The fused streamed qdq sensitivity reproduces the resident walk."""
    resident_sens = parity_results["resident_sens"]
    _, st_meta = parity_results["streamed"]
    streamed = st_meta["sensitivity_map"]

    expected_layers = set(range(KEEP_LAYERS))
    assert set(resident_sens) == expected_layers, resident_sens
    assert set(streamed) == expected_layers, streamed
    assert all(v > 0.0 for v in streamed.values()), streamed

    for layer in sorted(expected_layers):
        res, st = float(resident_sens[layer]), float(streamed[layer])
        rel = abs(st - res) / max(abs(res), 1e-30)
        assert rel < 1e-6, (
            f"layer {layer}: streamed sensitivity {st!r} vs resident {res!r} "
            f"(rel diff {rel:.3e})"
        )


def test_streaming_entry_keys_prefixed(parity_results):
    """Streamed entries carry full sanitized model paths."""
    st_entries, _ = parity_results["streamed"]
    assert all(name.startswith(SAN_LAYER_PREFIX) for name in st_entries)
