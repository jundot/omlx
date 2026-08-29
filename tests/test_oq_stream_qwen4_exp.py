"""Unit tests for the qwen4_exp streaming-calibration sourcer pieces.

Everything here runs without model weights: the per-layer mask schedule, the
mmap-mode PLE shard drop, the per-slot forward state, the stream-calibration
gate, and the MTP cache-completeness opt-out. The end-to-end streamed-vs-
resident parity gate lives with the truncated-fixture harness, not here.
"""

import mlx.core as mx
import numpy as np
import pytest

import omlx.oq as oq
from omlx.oq import (
    OQImatrixEntry,
    _qwen4_exp_mmap_skip_key,
    _resolve_stream_calibration,
    _streamed_layer_items,
    _streamed_qwen4_exp_slot_state,
    _streamed_qwen4_exp_state,
)

QWEN4_CONFIG = {
    "model_type": "qwen4_exp",
    "text_config": {
        "hc_count": 2,
        "hidden_size": 8,
        "num_hidden_layers": 4,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
    },
}


# --- stream-calibration gate ------------------------------------------------


def test_resolve_accepts_explicit_qwen4_exp():
    assert _resolve_stream_calibration(
        True, model_exceeds_ram=False, model_type="qwen4_exp"
    )


def test_resolve_auto_rule_streams_qwen4_exp_over_budget():
    assert _resolve_stream_calibration(
        None, model_exceeds_ram=True, model_type="qwen4_exp"
    )
    assert not _resolve_stream_calibration(
        None, model_exceeds_ram=False, model_type="qwen4_exp"
    )


def test_resolve_env_var_streams_qwen4_exp(monkeypatch):
    monkeypatch.setenv("OMLX_OQ_STREAM_CALIBRATION", "1")
    assert _resolve_stream_calibration(
        None, model_exceeds_ram=False, model_type="qwen4_exp"
    )


def test_resolve_still_rejects_unsupported_layout():
    with pytest.raises(ValueError, match="streaming imatrix sourcer"):
        _resolve_stream_calibration(True, model_exceeds_ram=True, model_type="llama")


# --- per-layer mask schedule and forward state ------------------------------


def test_qwen4_exp_state_masks_follow_layer_types():
    calib_data = mx.zeros((3, 5), dtype=mx.int32)
    embedded = mx.random.normal((3, 5, 8)).astype(mx.bfloat16)
    inputs, layer_masks, position_ids = _streamed_qwen4_exp_state(
        QWEN4_CONFIG, calib_data, embedded
    )
    assert inputs.shape == (3, 5, 16)  # hc_count * hidden_size
    assert inputs.dtype == mx.bfloat16
    # The tiled halves are copies of the same stream.
    assert mx.array_equal(inputs[..., :8], inputs[..., 8:])
    assert len(layer_masks) == 4
    assert layer_masks[0] is None
    assert layer_masks[1] is None
    assert layer_masks[3] is None
    causal = layer_masks[2]
    assert causal is not None
    assert causal.shape == (5, 5)
    assert causal.dtype == mx.bfloat16
    assert position_ids.shape == (3, 5)
    assert mx.array_equal(position_ids[0], mx.arange(5, dtype=mx.int32))


def test_qwen4_exp_state_accepts_pretiled_and_rejects_bad_width():
    calib_data = mx.zeros((2, 4), dtype=mx.int32)
    pretiled = mx.zeros((2, 4, 16), dtype=mx.bfloat16)
    inputs, _, _ = _streamed_qwen4_exp_state(QWEN4_CONFIG, calib_data, pretiled)
    assert inputs is pretiled
    with pytest.raises(RuntimeError, match="invalid hidden width"):
        _streamed_qwen4_exp_state(
            QWEN4_CONFIG, calib_data, mx.zeros((2, 4, 12), dtype=mx.bfloat16)
        )


def test_qwen4_exp_state_requires_config_fields():
    with pytest.raises(RuntimeError, match="layer_types"):
        _streamed_qwen4_exp_state(
            {"model_type": "qwen4_exp", "text_config": {"hc_count": 2}},
            mx.zeros((1, 2), dtype=mx.int32),
            mx.zeros((1, 2, 8), dtype=mx.bfloat16),
        )


def test_qwen4_exp_slot_state_slices_like_the_resident_walk():
    calib_data = mx.arange(12, dtype=mx.int32).reshape(4, 3)
    position_ids = mx.broadcast_to(mx.arange(3, dtype=mx.int32)[None], (4, 3))
    state = _streamed_qwen4_exp_slot_state(calib_data, position_ids, 1, 3)
    assert state["kind"] == oq._QWEN4_EXP_LAYER_STATE_KIND
    assert state["input_ids"].shape == (2, 3)
    assert mx.array_equal(state["input_ids"], calib_data[1:3])
    assert state["position_ids"].shape == (2, 3)


# --- mmap-mode PLE shard drop (amendment 5) ---------------------------------


class _PoisonValue:
    """A plan value that fails the test if it is ever materialized (popped)."""

    def __init__(self, key):
        self.key = key


def _fake_plan(layer_idx=1):
    prefix = f"language_model.model.layers.{layer_idx}."
    rel_keys = [
        f"ple.ple_embedding.ngram_embedding.shards.{i}.weight" for i in range(128)
    ] + [
        "ple.ple_embedding.ngram_embedding.weight_scale",
        "ple.ple_embedding.ngram_embedding.ngram_heads_offsets",
        "ple.ple_embedding.ngram_embedding.ngram_heads_vocab_sizes",
        "ple.norm_in.weight",
        "mlp.shared_expert.gate_proj.weight",
    ]
    return {f"{prefix}{rel}": _PoisonValue(rel) for rel in rel_keys}


def test_skip_key_matches_exactly_the_shard_weights():
    assert _qwen4_exp_mmap_skip_key(
        "ple.ple_embedding.ngram_embedding.shards.17.weight"
    )
    for rel in (
        "ple.ple_embedding.ngram_embedding.weight_scale",
        "ple.ple_embedding.ngram_embedding.ngram_heads_offsets",
        "ple.ple_embedding.ngram_embedding.ngram_heads_vocab_sizes",
        "ple.norm_in.weight",
        "mlp.shared_expert.gate_proj.weight",
        "self_attn.q_proj.weight",
    ):
        assert not _qwen4_exp_mmap_skip_key(rel)


def test_layer_items_skip_filters_before_the_pop():
    # The shard tensors must never be popped (never read from disk): the
    # filtered keys stay in the plan and the kept ones come through intact.
    plan = _fake_plan(layer_idx=1)
    items = _streamed_layer_items(plan, 1, skip_key=_qwen4_exp_mmap_skip_key)
    kept_keys = [k for k, _ in items]
    assert len(items) == 5
    assert not any(".shards." in k and k.endswith(".weight") for k in kept_keys)
    assert "ple.ple_embedding.ngram_embedding.weight_scale" in kept_keys
    assert "mlp.shared_expert.gate_proj.weight" in kept_keys
    # Skipped keys were left un-popped in the plan.
    remaining = [k for k in plan if ".shards." in k]
    assert len(remaining) == 128


def test_layer_items_without_skip_is_unchanged():
    # Positive control: a non-PLE layer sources every tensor, strictly.
    prefix = "language_model.model.layers.0."
    plan = {
        f"{prefix}self_attn.q_proj.weight": 1,
        f"{prefix}mlp.gate_proj.weight": 2,
        f"{prefix}input_layernorm.weight": 3,
    }
    items = _streamed_layer_items(plan, 0, skip_key=_qwen4_exp_mmap_skip_key)
    assert sorted(k for k, _ in items) == [
        "input_layernorm.weight",
        "mlp.gate_proj.weight",
        "self_attn.q_proj.weight",
    ]
    assert not plan


# --- table-preserve flag predicate ------------------------------------------


def test_ngram_table_tensor_matches_the_preserve_predicate():
    key = (
        "language_model.model.layers.1.ple.ple_embedding."
        "ngram_embedding.shards.17.weight"
    )
    assert oq._is_qwen4_exp_ngram_embedding_tensor(key, QWEN4_CONFIG)
    assert not oq._is_qwen4_exp_ngram_embedding_tensor(
        "language_model.model.layers.1.mlp.gate_proj.weight", QWEN4_CONFIG
    )
    assert not oq._is_qwen4_exp_ngram_embedding_tensor(key, {"model_type": "llama"})


def test_quantize_oq_streaming_exposes_preserve_ngram_table():
    import inspect

    params = inspect.signature(oq.quantize_oq_streaming).parameters
    assert "preserve_ngram_table" in params
    assert params["preserve_ngram_table"].default is False


# --- MTP cache-completeness opt-out (preserve_mtp=False builds) -------------


def _make_stream_cache(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    mx.save_safetensors(str(src_dir / "model.safetensors"), {"w": mx.zeros((2, 2))})
    config = {"model_type": "qwen4_exp"}
    expected = oq._source_imatrix_signature(
        src_dir,
        config,
        num_samples=4,
        seq_length=8,
        calib_dataset=oq._OQE_CALIB_DATASET,
    )
    metadata = {**expected, "load_kind": "streaming"}
    entries = {
        "language_model.model.layers.0.mlp.gate_proj": OQImatrixEntry(
            in_sum2=np.ones(4, dtype=np.float64),
            counts=np.ones(4, dtype=np.int64),
        )
    }
    cache_path = tmp_path / "imatrix.npz"
    oq._save_oqe_imatrix(cache_path, entries, metadata)
    return src_dir, config, cache_path


def _mtp_source_monkeypatches(monkeypatch):
    import omlx.utils.model_loading as model_loading

    monkeypatch.setattr(model_loading, "_has_mtp_heads", lambda cfg: True)
    monkeypatch.setattr(model_loading, "_checkpoint_has_mtp_weights", lambda path: True)


def test_cache_hit_survives_missing_mtp_entries_when_head_is_dropped(
    tmp_path, monkeypatch
):
    src_dir, config, cache_path = _make_stream_cache(tmp_path)
    _mtp_source_monkeypatches(monkeypatch)

    def _no_recollect(*args, **kwargs):
        raise AssertionError("recollected despite require_mtp_entries=False")

    monkeypatch.setattr(oq, "_collect_imatrix_streaming", _no_recollect)
    got = oq._load_or_collect_imatrix(
        str(src_dir),
        config,
        cache_path=str(cache_path),
        reuse_cache=True,
        num_samples=4,
        seq_length=8,
        strict=False,
        trust_remote_code=False,
        stream_calibration=True,
        require_mtp_entries=False,
    )
    assert got.reused


def test_cache_hit_still_recollects_when_mtp_entries_are_required(
    tmp_path, monkeypatch
):
    src_dir, config, cache_path = _make_stream_cache(tmp_path)
    _mtp_source_monkeypatches(monkeypatch)

    class _RecollectedError(Exception):
        pass

    def _boom(*args, **kwargs):
        raise _RecollectedError()

    monkeypatch.setattr(oq, "_collect_imatrix_streaming", _boom)
    monkeypatch.setattr("mlx_lm.tokenizer_utils.load", lambda p: object())
    with pytest.raises(_RecollectedError):
        oq._load_or_collect_imatrix(
            str(src_dir),
            config,
            cache_path=str(cache_path),
            reuse_cache=True,
            num_samples=4,
            seq_length=8,
            strict=False,
            trust_remote_code=False,
            stream_calibration=True,
            require_mtp_entries=True,
        )
