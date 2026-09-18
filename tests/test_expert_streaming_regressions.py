"""Regression coverage: SSD-backing failure, per-model top-k isolation,
prefill stand-in LRU probe."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from streaming_fixtures import write_moe_checkpoint


def test_ssd_backing_failure_fails_clean_not_ram_fallback():
    """SSD backing failure must raise, never retain banks in RAM."""
    from omlx.patches.expert_streaming import convert_model_to_streaming

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        write_moe_checkpoint(tmp)
        model = SimpleNamespace()  # never reached: backing fails first
        with patch(
            "omlx.patches.expert_streaming.shard_bank.ExpertBackingStore",
            side_effect=OSError("disk gone"),
        ):
            with pytest.raises(RuntimeError, match="SSD backing"):
                convert_model_to_streaming(model, str(tmp), None, use_file_backing=True)


def test_topk_per_model_isolation(monkeypatch):
    """One model's threshold must not move a resident block."""
    from omlx.patches.expert_streaming import adaptive_topk as at

    # Resolver must be pure (no global write). monkeypatch restores the
    # globals ``configure``/``configure_cache_prior`` mutate.
    prev = at.current_threshold()
    monkeypatch.setattr(at, "_THRESHOLD", prev)
    monkeypatch.setattr(at, "_CACHE_PRIOR", at.cache_prior_bonus())
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
    assert at.instance_threshold(a) == 0.6
    assert at.instance_threshold(b) is None
    assert at.instance_prior(a) == 0.0


def test_topk_patch_uses_per_instance(monkeypatch):
    """Runtime: patched Qwen call prefers the block's own setting."""
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
    monkeypatch.setattr(at, "_THRESHOLD", prev)
    monkeypatch.setattr(at, "_CACHE_PRIOR", at.cache_prior_bonus())
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

