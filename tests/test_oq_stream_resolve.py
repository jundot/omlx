# SPDX-License-Identifier: Apache-2.0
"""Model-type gating for the streaming imatrix calibration decision.

_resolve_stream_calibration decides whether oQe calibration streams layers
from the checkpoint instead of building a RAM-safe proxy. The streamed
sourcer only understands the MiniMax-M3 (minimax_m3_vl) layout, so the
decision has to stay inside that supported set:

- the auto rule (source exceeds RAM) must not turn streaming on for a layout
  the sourcer cannot handle, otherwise an oQe build of a big non-MiniMax model
  that completes today through the proxy would start crashing by default
- an explicit request for streaming on an unsupported layout is a
  misconfiguration and must fail early with a clear message, not a late
  AttributeError deep in the sourcer

These are pure decision-logic tests: no checkpoint, no model load, no GPU, so
they run in CI where the MiniMax-M3 weights are absent.
"""

import logging

import pytest

from omlx.oq import (
    _STREAM_CALIBRATION_SUPPORTED_MODEL_TYPES,
    _resolve_stream_calibration,
)

SUPPORTED = "minimax_m3_vl"
UNSUPPORTED = "qwen3_5_moe"


@pytest.fixture(autouse=True)
def _clear_stream_env(monkeypatch):
    """Keep OMLX_OQ_STREAM_CALIBRATION out of the decision unless a test sets it."""
    monkeypatch.delenv("OMLX_OQ_STREAM_CALIBRATION", raising=False)


# --- the auto rule (stream_calibration=None, no env) --------------------------


def test_auto_streams_supported_model_that_exceeds_ram():
    assert (
        _resolve_stream_calibration(None, model_exceeds_ram=True, model_type=SUPPORTED)
        is True
    )


def test_auto_keeps_proxy_for_supported_model_within_ram():
    assert (
        _resolve_stream_calibration(None, model_exceeds_ram=False, model_type=SUPPORTED)
        is False
    )


def test_auto_does_not_stream_unsupported_model_over_ram():
    """The merge blocker: a big non-MiniMax model must keep the proxy path."""
    assert (
        _resolve_stream_calibration(
            None, model_exceeds_ram=True, model_type=UNSUPPORTED
        )
        is False
    )


def test_auto_treats_missing_model_type_as_unsupported():
    assert (
        _resolve_stream_calibration(None, model_exceeds_ram=True, model_type="")
        is False
    )


# --- explicit argument --------------------------------------------------------


def test_explicit_true_streams_supported_even_within_ram():
    assert (
        _resolve_stream_calibration(True, model_exceeds_ram=False, model_type=SUPPORTED)
        is True
    )


def test_explicit_false_never_streams_even_over_ram():
    assert (
        _resolve_stream_calibration(False, model_exceeds_ram=True, model_type=SUPPORTED)
        is False
    )


def test_explicit_true_on_unsupported_raises_early():
    with pytest.raises(ValueError) as exc:
        _resolve_stream_calibration(
            True, model_exceeds_ram=True, model_type=UNSUPPORTED
        )
    msg = str(exc.value)
    assert UNSUPPORTED in msg, "error must name the unsupported model_type"
    assert SUPPORTED in msg, "error must name the supported set"
    assert "stream_calibration" in msg


def test_explicit_true_on_empty_model_type_raises():
    with pytest.raises(ValueError):
        _resolve_stream_calibration(True, model_exceeds_ram=False, model_type="")


# --- environment override -----------------------------------------------------


def test_env_true_streams_supported(monkeypatch):
    monkeypatch.setenv("OMLX_OQ_STREAM_CALIBRATION", "1")
    assert (
        _resolve_stream_calibration(None, model_exceeds_ram=False, model_type=SUPPORTED)
        is True
    )


def test_env_true_on_unsupported_falls_back_to_proxy(monkeypatch, caplog):
    """A session-wide env var is a soft preference: unsupported layouts keep
    the proxy rather than raising, so only the explicit argument is a hard
    per-build assertion. The rejected preference is warned, not silent."""
    monkeypatch.setenv("OMLX_OQ_STREAM_CALIBRATION", "true")
    with caplog.at_level(logging.WARNING):
        result = _resolve_stream_calibration(
            None, model_exceeds_ram=True, model_type=UNSUPPORTED
        )
    assert result is False
    assert UNSUPPORTED in caplog.text
    assert "OMLX_OQ_STREAM_CALIBRATION" in caplog.text


def test_env_false_overrides_auto_for_supported(monkeypatch):
    monkeypatch.setenv("OMLX_OQ_STREAM_CALIBRATION", "0")
    assert (
        _resolve_stream_calibration(None, model_exceeds_ram=True, model_type=SUPPORTED)
        is False
    )


# --- the supported set --------------------------------------------------------


def test_supported_set_is_pinned():
    """Pin the supported layouts: widening the set is a conscious, reviewed change."""
    assert (
        frozenset({"minimax_m3_vl", "qwen4_exp"})
        == _STREAM_CALIBRATION_SUPPORTED_MODEL_TYPES
    )


def test_model_type_match_is_case_insensitive():
    assert (
        _resolve_stream_calibration(
            None, model_exceeds_ram=True, model_type="MiniMax_M3_VL"
        )
        is True
    )


def test_model_type_is_a_required_argument():
    """Every caller must state the layout; omitting it is a programming error,
    not a silent fall-through to the unsupported branch."""
    with pytest.raises(TypeError):
        _resolve_stream_calibration(None, model_exceeds_ram=True)
