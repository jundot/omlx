"""Fase M5: effective-config blocks and the A/B comparator tests."""

import json

from bench.bench_expert_streaming import _effective_config
from bench.compare_results import (
    declared_knobs,
    collect_mismatches,
    load_result,
)


def _cfg(**over):
    base = {
        "git_sha": "abc123",
        "model_fingerprint": {"model": "Qwen3.8", "profile_format": 2},
        "single_request": True,
        "decode_tokens": 48,
        "chunk_schedule": {"reference_step": 1024},
        "budget_gib": 0.0,
        "cold_tier": None,
        "hot_fraction": None,
        "ctx_mode_policy": "hybrid",
        "decode_union_rows": 64,
        "ctx_ahead": 3,
        "expert_qd": 16,
        "run_qd": 16,
        "prefill_qd": 0,
        "run_merge_gap": 0,
        "ra_enabled": True,
        "stash_enabled": False,
        "pins_enabled": False,
        "pin_sync_effective": False,
        "pin_regime_effective": None,
        "profile_enabled": False,
        "memtrace_enabled": False,
        "read_sampling_mode": "off",
        "cache_cool_protocol": "warm-page-cache",
        "experiment_knobs": [],
        "active_engines": 1,
    }
    base.update(over)
    return base


def _result(name, cfg, **extra):
    return {
        "effective_config": cfg,
        "bit_exact_kind": "tokens",
        "tok_s": 3.0,
        "ttft_s": 45.0,
        **extra,
    }


def test_result_contains_effective_config(tmp_path):
    import importlib
    import sys

    from types import SimpleNamespace

    cfg = _effective_config(
        git_sha="abc123",
        single_request=True,
        decode_tokens=48,
        chunk_schedule={"reference_step": 1024},
        budget_gib=0.0,
        cold_tier=None,
        hot_fraction=None,
        pins=False,
        pinner=None,
        model_fingerprint={"model": "Qwen3.8", "profile_format": 2},
        run_qd=16,
        expert_qd=16,
        prefill_qd=0,
        knobs=["pins_enabled"],
    )
    for key in (
        "git_sha",
        "model_fingerprint",
        "single_request",
        "decode_tokens",
        "chunk_schedule",
        "budget_gib",
        "cold_tier",
        "ctx_mode_policy",
        "decode_union_rows",
        "ctx_ahead",
        "expert_qd",
        "run_qd",
        "run_merge_gap",
        "ra_enabled",
        "stash_enabled",
        "pins_enabled",
        "profile_enabled",
        "memtrace_enabled",
        "read_sampling_mode",
        "cache_cool_protocol",
        "active_engines",
    ):
        assert key in cfg, key
    assert cfg["experiment_knobs"] == ["pins_enabled"]
    assert cfg["decode_tokens"] == 48
    assert cfg["active_engines"] == 1


def test_result_records_profile_and_memtrace_state():
    """The block reflects the EFFECTIVE profiling state, so a PROFILE arm
    can never be A/B'd against an unprofiled arm silently."""
    cfg = _cfg(profile_enabled=True, memtrace_enabled=True, read_sampling_mode="profile")
    assert cfg["profile_enabled"] is True
    assert cfg["memtrace_enabled"] is True
    assert cfg["read_sampling_mode"] == "profile"


def test_comparator_rejects_chunk_schedule_mismatch():
    a = _result("a", _cfg())
    b = _result(
        "b",
        _cfg(chunk_schedule={"reference_step": 4096}),
    )
    issues = collect_mismatches(a, b)
    assert any("chunk_schedule" in i for i in issues), issues


def test_comparator_rejects_token_gate_kind_mismatch():
    a = _result("a", _cfg())
    b = _result("b", _cfg(), bit_exact_kind="text")
    issues = collect_mismatches(a, b)
    assert any("bit_exact_kind" in i for i in issues), issues


def test_comparator_allows_declared_experiment_knob():
    """pins_enabled declared as the experiment knob by BOTH sides: the
    comparator does not refuse the difference."""
    a = _result(
        "a",
        _cfg(pins_enabled=False, experiment_knobs=["pins_enabled"]),
    )
    b = _result(
        "b",
        _cfg(pins_enabled=True, experiment_knobs=["pins_enabled"]),
    )
    assert collect_mismatches(a, b) == []
    # Without the declaration the mismatch is refused.
    a2 = _result("a", _cfg(pins_enabled=False, experiment_knobs=[]))
    b2 = _result("b", _cfg(pins_enabled=True, experiment_knobs=[]))
    issues = collect_mismatches(a2, b2)
    assert any("pins_enabled" in i for i in issues), issues


def test_comparator_requires_effective_config():
    a = _result("a", _cfg())
    b = {"tok_s": 3.0, "bit_exact_kind": "tokens"}
    issues = collect_mismatches(a, b)
    assert any("missing effective_config" in i for i in issues), issues


def test_comparator_rejects_missing_declared_knob_side():
    """A knob declared by only ONE side cannot legitimize a differ — both
    must declare it for the comparison to be fair."""
    a = _result(
        "a",
        _cfg(pins_enabled=False, experiment_knobs=["pins_enabled"]),
    )
    b = _result(
        "b",
        _cfg(pins_enabled=True, experiment_knobs=[]),
    )
    issues = collect_mismatches(a, b)
    assert any("pins_enabled" in i for i in issues), issues


def test_comparator_rejects_empty_tokens_or_text_kind():
    """Fase A2: an EMPTY token list is a broken gate by construction, and
    a text "gate" proves nothing about token identity — both are refused
    even when both sides share the flaw."""
    a = _result("a", _cfg(), tokens=[])
    b = _result("b", _cfg())
    issues = collect_mismatches(a, b)
    assert any("empty" in i and "tokens" in i for i in issues), issues
    a2 = _result("a2", _cfg(), bit_exact_kind="text")
    b2 = _result("b2", _cfg(), bit_exact_kind="text")
    issues2 = collect_mismatches(a2, b2)
    assert any("text" in i for i in issues2), issues2


def test_comparator_rejects_mixed_stage_vocabulary():
    """Fase A3: read_stats built with a different stage-key vocabulary
    (e.g. the pre-A3 names) cannot be compared stage-wise — refused."""
    a = _result("a", _cfg())
    b = _result("b", _cfg())
    a["read_stats"] = {
        "lifetime": {"stages_us": {"component_e2e_us": {}, "read_duration_us": {}}}
    }
    b["read_stats"] = {
        "lifetime": {"stages_us": {"component_e2e_us": {}, "preadv_us": {}}}
    }
    issues = collect_mismatches(a, b)
    assert any("stage keys" in i for i in issues), issues
    # One-sided stage data never blocks the comparison.
    c = _result("c", _cfg())
    assert collect_mismatches(a, c) == []


def test_comparator_rejects_active_engines_mismatch():
    """Fase A4: comparing a run with a second engine in-process against a
    single-engine run compares different pool worlds — refused."""
    a = _result("a", _cfg())
    b = _result("b", _cfg(active_engines=2))
    issues = collect_mismatches(a, b)
    assert any("active_engines" in i for i in issues), issues

