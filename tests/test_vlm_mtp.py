# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.speculative.vlm_mtp.

Phase 2A: covers drafter validation, lazy bind, and wrapper-level dispatch
to mlx-vlm's ``_mtp_rounds`` / ``_mtp_rounds_batch``. The actual mlx-vlm
helpers are mocked so this suite stays fast and does not touch model
weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.speculative import vlm_mtp


def _fake_drafter_model(model_type: str = "gemma4_assistant") -> MagicMock:
    """Build a stand-in for Gemma4AssistantDraftModel that satisfies the
    minimum API used by VLMMTPDrafter."""
    drafter = MagicMock()
    drafter.config = MagicMock(model_type=model_type)
    return drafter


def test_load_vlm_mtp_drafter_happy_path():
    """Valid gemma4_assistant artifact returns a populated VLMMTPDrafter."""
    fake_model = _fake_drafter_model("gemma4_assistant")
    with patch.object(
        vlm_mtp, "_vlm_load_drafter", return_value=(fake_model, "mtp")
    ):
        drafter = vlm_mtp.load_vlm_mtp_drafter("/path/to/drafter")
    assert isinstance(drafter, vlm_mtp.VLMMTPDrafter)
    assert drafter.draft_kind == "mtp"
    assert drafter.source_path == "/path/to/drafter"
    assert drafter.model is fake_model


def test_load_vlm_mtp_drafter_rejects_dflash_kind():
    """A drafter that resolves to non-mtp kind is rejected (None + warn)."""
    fake_model = _fake_drafter_model("qwen3_dflash")
    with patch.object(
        vlm_mtp, "_vlm_load_drafter", return_value=(fake_model, "dflash")
    ):
        result = vlm_mtp.load_vlm_mtp_drafter("/path/to/drafter")
    assert result is None


def test_load_vlm_mtp_drafter_rejects_wrong_model_type():
    """Even if kind=='mtp', non-gemma4_assistant model_type is rejected."""
    fake_model = _fake_drafter_model("qwen3_5_assistant")  # hypothetical
    with patch.object(
        vlm_mtp, "_vlm_load_drafter", return_value=(fake_model, "mtp")
    ):
        result = vlm_mtp.load_vlm_mtp_drafter("/path/to/drafter")
    assert result is None


def test_load_vlm_mtp_drafter_swallows_load_exception():
    """Load failures are logged and converted to None — never raised."""
    with patch.object(
        vlm_mtp,
        "_vlm_load_drafter",
        side_effect=RuntimeError("HF repo not found"),
    ):
        result = vlm_mtp.load_vlm_mtp_drafter("not-a-real-drafter")
    assert result is None


def test_vlmmtpdrafter_bind_is_idempotent():
    """``reset(target)`` must be called at most once even if bind_to is
    called repeatedly. The drafter mutates global target state on each
    call, so duplicate binds would corrupt the embed table reference."""
    fake_model = _fake_drafter_model("gemma4_assistant")
    drafter = vlm_mtp.VLMMTPDrafter(fake_model, "mtp", "/p")
    target = MagicMock()
    drafter.bind_to(target)
    drafter.bind_to(target)
    drafter.bind_to(target)
    fake_model.reset.assert_called_once_with(target)


def test_run_vlm_mtp_decode_single_request_dispatches_to_mtp_rounds():
    """Single-int first_bonus routes to ``_mtp_rounds`` and yields ints."""
    fake_model = _fake_drafter_model("gemma4_assistant")
    drafter = vlm_mtp.VLMMTPDrafter(fake_model, "mtp", "/p")
    target = MagicMock()
    sampler = MagicMock()

    yielded = [(11, None), (22, None), (33, None)]
    with (
        patch.object(vlm_mtp, "_mtp_rounds", return_value=iter(yielded)) as m_single,
        patch.object(vlm_mtp, "_mtp_rounds_batch") as m_batch,
    ):
        out = list(
            vlm_mtp.run_vlm_mtp_decode(
                target_language_model=target,
                drafter=drafter,
                prompt_cache=[],
                hidden=mx.zeros((1, 1, 8)),
                shared_kv_states={},
                first_bonus=7,
                max_tokens=4,
                sampler=sampler,
            )
        )

    assert out == [11, 22, 33]
    m_single.assert_called_once()
    m_batch.assert_not_called()
    # Bind happens before the loop runs
    fake_model.reset.assert_called_once_with(target)
    # first_bonus int forwarded as int
    kwargs = m_single.call_args.kwargs
    assert kwargs["first_bonus"] == 7
    assert kwargs["max_tokens"] == 4


def test_run_vlm_mtp_decode_batch_dispatches_to_mtp_rounds_batch():
    """Multi-row mx.array first_bonus routes to ``_mtp_rounds_batch`` and
    yields the per-row list shape."""
    fake_model = _fake_drafter_model("gemma4_assistant")
    drafter = vlm_mtp.VLMMTPDrafter(fake_model, "mtp", "/p")
    target = MagicMock()
    sampler = MagicMock()

    first_bonus = mx.array([1, 2, 3])  # B=3
    yielded = [([1, None, 3], None), ([None, None, None], None)]
    with (
        patch.object(vlm_mtp, "_mtp_rounds_batch", return_value=iter(yielded)) as m_batch,
        patch.object(vlm_mtp, "_mtp_rounds") as m_single,
    ):
        out = list(
            vlm_mtp.run_vlm_mtp_decode(
                target_language_model=target,
                drafter=drafter,
                prompt_cache=[],
                hidden=mx.zeros((3, 1, 8)),
                shared_kv_states={},
                first_bonus=first_bonus,
                max_tokens=4,
                sampler=sampler,
                eos_token_ids={2, 5},
            )
        )

    assert out == [[1, None, 3], [None, None, None]]
    m_batch.assert_called_once()
    m_single.assert_not_called()
    kwargs = m_batch.call_args.kwargs
    # EOS forwarded as a fresh set (function does its own copy)
    assert kwargs["eos_token_ids"] == {2, 5}


def test_run_vlm_mtp_decode_single_scalar_array_unwraps_to_int():
    """B=1 mx.array first_bonus is treated as single-request and unwrapped."""
    fake_model = _fake_drafter_model("gemma4_assistant")
    drafter = vlm_mtp.VLMMTPDrafter(fake_model, "mtp", "/p")
    target = MagicMock()
    sampler = MagicMock()

    first_bonus = mx.array([42])  # B=1 should not take the batch branch
    with (
        patch.object(vlm_mtp, "_mtp_rounds", return_value=iter([])) as m_single,
        patch.object(vlm_mtp, "_mtp_rounds_batch") as m_batch,
    ):
        list(
            vlm_mtp.run_vlm_mtp_decode(
                target_language_model=target,
                drafter=drafter,
                prompt_cache=[],
                hidden=mx.zeros((1, 1, 8)),
                shared_kv_states={},
                first_bonus=first_bonus,
                max_tokens=4,
                sampler=sampler,
            )
        )

    m_single.assert_called_once()
    m_batch.assert_not_called()
    assert m_single.call_args.kwargs["first_bonus"] == 42


@pytest.mark.parametrize(
    "vlm_mtp_kw, other_kw",
    [
        ("dflash_enabled", "dflash_enabled"),
        ("specprefill_enabled", "specprefill_enabled"),
        ("mtp_enabled", "mtp_enabled"),
        ("turboquant_kv_enabled", "turboquant_kv_enabled"),
    ],
)
def test_model_settings_vlm_mtp_mutex(vlm_mtp_kw, other_kw):
    """ModelSettings.__post_init__ raises when vlm_mtp_enabled overlaps
    with any other speculative / cache-mutating toggle."""
    from omlx.model_settings import ModelSettings

    with pytest.raises(ValueError, match="vlm_mtp_enabled"):
        ModelSettings(vlm_mtp_enabled=True, **{other_kw: True})
