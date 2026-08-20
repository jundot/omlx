# SPDX-License-Identifier: Apache-2.0
"""OMLX_STEP_PROFILE must be a strict no-op when off and additive when on."""

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "omlx" / "step_profile.py"


def _load(monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv("OMLX_STEP_PROFILE", "1")
    else:
        monkeypatch.delenv("OMLX_STEP_PROFILE", raising=False)
    name = f"_step_profile_{'on' if enabled else 'off'}"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_disabled_is_a_strict_noop(monkeypatch):
    sp = _load(monkeypatch, enabled=False)
    assert sp.enabled() is False
    assert sp.tick() == 0.0
    sp.add("bucket", 1.0)
    sp.add_since("bucket", 0.0)
    assert sp.snapshot() == {}
    sp.maybe_log(10_000)  # must not raise or log


def test_enabled_accumulates_and_resets_on_log(monkeypatch):
    sp = _load(monkeypatch, enabled=True)
    assert sp.enabled() is True
    t0 = sp.tick()
    assert t0 > 0.0
    sp.add_since("a", t0)
    sp.add("b", 0.25)
    sp.add("b", 0.25)
    snap = sp.snapshot()
    assert snap["b"]["seconds"] == 0.5
    assert snap["b"]["samples"] == 2
    assert "a" in snap
    # The window is counted in maybe_log CALLS (buckets are process-wide;
    # per-engine step counters cannot be compared against shared state).
    for step in range(sp._LOG_EVERY):
        sp.maybe_log(step)
    assert sp.snapshot() == {}, "window resets after the summary line"


def test_log_interval_gates_output(monkeypatch):
    sp = _load(monkeypatch, enabled=True)
    sp.add("x", 1.0)
    for step in range(sp._LOG_EVERY - 1):
        sp.maybe_log(step)  # below the call threshold: keeps buckets
    assert sp.snapshot() != {}
