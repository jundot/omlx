# SPDX-License-Identifier: Apache-2.0
"""Coordinator-loss documentation exists and covers the required semantics."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "coordinator-loss-semantics.md"


def test_document_exists():
    assert DOC.is_file(), f"{DOC} does not exist"


def _read() -> str:
    return DOC.read_text(encoding="utf-8")


def test_covers_fail_stop_behavior():
    text = _read()
    assert "launcher_lost" in text
    assert "os._exit" in text or "_exit(1)" in text


def test_covers_collective_constraint():
    text = _read()
    assert "mx.distributed" in text
    assert "deadlock" in text or "blocks" in text
    assert "world-size" in text or "world_size" in text or "fixed" in text


def test_covers_manual_re_activation():
    text = _read()
    assert "re-activation" in text or "re-activate" in text
    assert "deployments.json" in text
    assert "plan_hash" in text


def test_covers_standalone_fallback():
    text = _read()
    assert "standalone" in text
    assert "HTTP server" in text or "http server" in text


def test_covers_placement_signature():
    text = _read()
    assert "approved_placement" in text or "placement_signature" in text
    assert "409" in text
