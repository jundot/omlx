# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-model prefill chunk width helpers (#3381).

The three engine seams that apply the override sit inline in ~1000-line async
``start()`` bodies, so the logic lives in ``omlx/engine/base.py`` and is
exercised here with no model, no server and no disk.
"""

import copy
import logging
from pathlib import Path
from types import SimpleNamespace

from omlx.engine.base import (
    _PREFILL_STEP_SIZE_CHOICES,
    _coerce_prefill_step_size,
    log_effective_prefill_step_size,
    resolve_prefill_step_size,
)
from omlx.model_settings import ModelSettings
from omlx.scheduler import SchedulerConfig


def _warnings(caplog) -> list[str]:
    """Warnings from these helpers only, ignoring anything a lazily imported
    patch module logs on first import."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name == "omlx.engine.base"
    ]


def test_coerce_snaps_to_the_nearest_supported_width(caplog):
    with caplog.at_level(logging.WARNING):
        assert _coerce_prefill_step_size(4096, "model-a") == 2048
    assert "4096" in caplog.text
    assert "not a supported chunk width" in caplog.text

    assert _coerce_prefill_step_size(1000, "model-a") == 1024
    assert _coerce_prefill_step_size(300, "model-a") == 256
    # A tie resolves to the smaller width: the memory-safe direction.
    assert _coerce_prefill_step_size(384, "model-a") == 256

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        for choice in _PREFILL_STEP_SIZE_CHOICES:
            assert _coerce_prefill_step_size(choice, "model-a") == choice
    assert caplog.records == []


def test_resolve_is_a_no_op_when_unset(caplog):
    unset = (
        None,
        SimpleNamespace(),
        ModelSettings(),
        ModelSettings(prefill_step_size=None),
    )
    with caplog.at_level(logging.INFO):
        for model_settings in unset:
            config = SchedulerConfig()
            resolve_prefill_step_size(config, model_settings, "model-a")
            assert config.prefill_step_size == 2048
    assert caplog.records == []


def test_resolve_tolerates_a_none_config():
    # DFlash keeps its scheduler config optional, so an override set on a
    # DFlash model must not raise.
    resolve_prefill_step_size(None, ModelSettings(prefill_step_size=512), "model-a")


def test_resolve_tolerates_a_non_numeric_value(caplog):
    config = SchedulerConfig()
    with caplog.at_level(logging.WARNING):
        resolve_prefill_step_size(
            config, SimpleNamespace(prefill_step_size="nope"), "model-a"
        )
    assert config.prefill_step_size == 2048
    assert "nope" in caplog.text

    # A numeric string from a hand-edited settings.json still applies.
    config = SchedulerConfig()
    resolve_prefill_step_size(
        config, SimpleNamespace(prefill_step_size="1024"), "model-a"
    )
    assert config.prefill_step_size == 1024


def test_resolve_mutates_only_the_copy():
    shared = SchedulerConfig()
    per_engine = copy.copy(shared)

    resolve_prefill_step_size(
        per_engine, ModelSettings(prefill_step_size=512), "model-a"
    )

    assert per_engine.prefill_step_size == 512
    assert shared.prefill_step_size == 2048


def test_every_engine_seam_calls_the_helper():
    engine_dir = Path(__file__).resolve().parents[1] / "omlx" / "engine"
    for name in ("vlm.py", "batched.py", "dflash.py"):
        source = (engine_dir / name).read_text()
        assert "resolve_prefill_step_size(" in source, name


def test_effective_width_log_reads_back_the_scheduler(caplog):
    scheduler = SimpleNamespace(
        config=SchedulerConfig(),
        model=SimpleNamespace(model_type="llama"),
        _glm_dsa_adaptive_prefill=None,
        _minimax_m3_adaptive_prefill=None,
        _base_prefill_step_size=lambda processed, remaining: 4096,
    )

    with caplog.at_level(logging.INFO):
        log_effective_prefill_step_size(scheduler, "model-a")

    assert "configured=2048" in caplog.text
    assert "effective=4096" in caplog.text
    assert _warnings(caplog) == []

    # Fail open: a scheduler whose internals moved costs a log line, not start().
    def _boom(processed, remaining):
        raise RuntimeError("scheduler internals moved")

    caplog.clear()
    with caplog.at_level(logging.INFO):
        log_effective_prefill_step_size(
            SimpleNamespace(config=SchedulerConfig(), _base_prefill_step_size=_boom),
            "model-a",
        )

    assert "prefill chunk width" not in caplog.text


def test_widening_warning_names_both_widths(caplog, monkeypatch):
    monkeypatch.setattr(
        "omlx.patches.glm_moe_dsa.generate_patch._glm_dsa_adaptive_prefill_config",
        lambda model, prefill_step_size: SimpleNamespace(step_size=8192),
    )
    scheduler = SimpleNamespace(
        config=SchedulerConfig(prefill_step_size=1024),
        model=SimpleNamespace(model_type="glm_moe_dsa"),
        _glm_dsa_adaptive_prefill=None,
        _minimax_m3_adaptive_prefill=None,
        _base_prefill_step_size=lambda processed, remaining: 1024,
    )

    with caplog.at_level(logging.INFO):
        log_effective_prefill_step_size(scheduler, "glm-model")

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "disables adaptive prefill widening" in warnings[0]
    assert "8192" in warnings[0]
    assert "1024" in warnings[0]
