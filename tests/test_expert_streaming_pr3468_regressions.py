"""Focused regressions for PR #3468 review (4 issues)."""

import json
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _write_supported_checkpoint(tmp: Path) -> None:
    config = {
        "model_type": "qwen4_exp",
        "num_hidden_layers": 2,
        "num_experts": 4,
        "hidden_size": 32,
        "moe_intermediate_size": 16,
    }
    (tmp / "config.json").write_text(json.dumps(config))
    import numpy as np

    tensors = {}
    for layer in range(2):
        for proj, shape in [
            ("gate_proj", (4, 16, 32)),
            ("up_proj", (4, 16, 32)),
            ("down_proj", (4, 32, 16)),
        ]:
            key = f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight"
            tensors[key] = (shape, "BF16", int(np.prod(shape)) * 2)
    header = {}
    offset = 0
    for k, (shape, dtype, size) in tensors.items():
        header[k] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    hb = json.dumps(header).encode()
    with (tmp / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * offset)
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )


def test_ssd_backing_failure_fails_clean_not_ram_fallback():
    """Issue 1: SSD backing failure must raise, never retain banks in RAM."""
    from omlx.patches.expert_streaming import convert_model_to_streaming

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _write_supported_checkpoint(tmp)
        model = SimpleNamespace()  # never reached: backing fails first
        with patch(
            "omlx.patches.expert_streaming.shard_bank.ExpertBackingStore",
            side_effect=OSError("disk gone"),
        ):
            with pytest.raises(RuntimeError, match="SSD backing"):
                convert_model_to_streaming(model, str(tmp), None, use_file_backing=True)


def test_topk_per_model_isolation():
    """Issue 2: one model's threshold must not move a resident block."""
    from omlx.patches.expert_streaming import adaptive_topk as at

    # Resolver must be pure (no global write).
    prev = at.current_threshold()
    try:
        got = at.resolve_threshold_from_settings(
            SimpleNamespace(expert_streaming_topk_threshold=0.6), model_type="qwen4_exp"
        )
        assert got == 0.6
        assert at.current_threshold() == prev
        prior = at.resolve_prior_from_settings(SimpleNamespace(expert_streaming_cache_prior=2.0))
        assert prior == 2.0
        assert at.cache_prior_bonus() == at._CACHE_PRIOR  # global untouched

        # Per-instance stamping isolates resident blocks.
        a = SimpleNamespace()
        b = SimpleNamespace()
        at.set_instance_routing(a, 0.6, 0.0)
        at.set_instance_routing(b, None, 0.0)
        at.configure(0.6)  # global move (e.g. another model's settings)
        try:
            assert at.instance_threshold(a) == 0.6
            assert at.instance_threshold(b) is None
            assert at.instance_prior(a) == 0.0
        finally:
            at.configure(prev)
            at.configure_cache_prior(0.0)
    finally:
        at.configure(prev)


def test_topk_patch_uses_per_instance():
    """Issue 2 (runtime): patched Qwen call prefers the block's own setting."""
    pytest.importorskip("mlx_vlm")
    import mlx.core as mx
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
    from omlx.patches.expert_streaming import adaptive_topk as at

    cfg = SimpleNamespace(
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
    )
    resident = Qwen3_5MoeSparseMoeBlock(cfg)
    other = Qwen3_5MoeSparseMoeBlock(cfg)
    mx.eval(resident.parameters())
    mx.eval(other.parameters())
    assert at.apply_qwen35_moe_topk_patch()
    prev = at.current_threshold()
    try:
        at.set_instance_routing(resident, None, 0.0)
        at.set_instance_routing(other, 0.6, 0.0)
        at.configure(None)
        import numpy as _np

        x = mx.array(_np.random.default_rng(0).standard_normal((1, 2, 32)).astype("float32"))
        out_resident = resident(x)
        mx.eval(out_resident)
        # Global move for "another model" must not change the resident exact block.
        at.configure(0.6)
        out_resident2 = resident(x)
        mx.eval(out_resident2)
        assert bool((out_resident == out_resident2).all())
    finally:
        at.configure(prev)
        at.configure_cache_prior(0.0)


def test_spill_source_change_invalidates_and_empty_manifest_rejected(tmp_path):
    """Issue 3: checkpoint change must not serve previous weights."""
    from omlx.patches.deepseek_v4 import spill as S

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 64)
    conv = S.spill_dir_for(model_dir)
    conv.mkdir(parents=True)
    (conv / "spill_layer_00.safetensors").write_bytes(b"y" * 64)
    S.write_manifest(conv, model_dir, ["spill_layer_00.safetensors"], {"k": "spill_layer_00.safetensors"})
    assert S.spill_is_valid(model_dir) == conv
    # Source change invalidates; stale shard must not validate.
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 65)
    assert S.spill_is_valid(model_dir) is None
    # Empty file list never validates (would otherwise glob stale shards).
    S.write_manifest(conv, model_dir, [], {})
    assert S.spill_is_valid(model_dir) is None


def test_prefill_standin_carries_lru_probe():
    """Issue 4: _record_chunk_transient needs _streaming_lru_heap_growth."""
    from omlx.prefill_transient_tracker import PrefillTransientTracker
    from omlx.scheduler import Scheduler

    tracker = PrefillTransientTracker()
    ns = SimpleNamespace(
        _prefill_min_chunk_tokens=256,
        _prefill_transient_tracker=tracker,
        _streaming_lru_cache=None,
        _streaming_lru_bytes_last=None,
    )
    ns._record_chunk_transient = Scheduler._record_chunk_transient.__get__(ns, Scheduler)
    ns._streaming_lru_heap_growth = Scheduler._streaming_lru_heap_growth.__get__(ns, Scheduler)
    ns._record_chunk_transient(512, 0, 1024, request_id="r", loop_label="t", kv_len=0, requested_step=512)
    # No AttributeError is the regression: the stand-in carries the LRU probe.
