# SPDX-License-Identifier: Apache-2.0
"""DFlash/DSpark target allow-list in omlx.utils.model_loading."""

from types import SimpleNamespace

import pytest

from omlx.speculative.dflash import attach_drafter, detach_drafter
from omlx.utils.model_loading import dflash_batched_supported, dflash_compatibility


@pytest.mark.parametrize(
    ("model_type", "engine_type"),
    [
        ("qwen3_5_moe", "vlm"),
        ("minicpmv4_7", "vlm"),
        ("qwen4_exp", "vlm"),
        ("qwen3_moe", "batched"),
    ],
)
def test_supported_targets(model_type, engine_type):
    assert dflash_batched_supported(model_type, engine_type)
    assert dflash_compatibility(model_type, engine_type) == (True, "")


def test_minicpm_v47_is_a_vlm_target_only():
    # MiniCPM-V 4.7 loads on the VLM engine (its text backbone is a Qwen3.5 MoE
    # behind a minicpmv4_7 wrapper); it is not a text-engine DFlash target.
    assert not dflash_batched_supported("minicpmv4_7", "batched")


def test_qwen3_8_flash_next_is_a_vlm_dflash_target():
    # Qwen3.8-Flash-Next ("qwen4_exp") drafts on the VLM engine. Its runtime
    # subclasses qwen3_5, so it exposes the same layer capture and
    # rollback_speculative_cache surface the block drafter binds to, and the
    # DFlash2 draft checkpoint normalizes to model_type "dflash2" through
    # mlx-vlm's architecture remap.
    assert dflash_batched_supported("qwen4_exp", "vlm")
    # mlx-lm has no qwen4_exp loader, so the text-only/force_lm path fails
    # before a drafter could bind.
    assert not dflash_batched_supported("qwen4_exp", "batched")


def test_dflash2_attachment_requests_the_hc_mean_capture():
    """Qwen3.8-Flash-Next-DFlash2 was trained on the target's hc-stream mean;
    v1/Laguna/Muse drafters keep the mixer representation."""
    language_model = SimpleNamespace(model=SimpleNamespace())
    drafter = SimpleNamespace(kind="dflash2", depth=3, clear=lambda: None)
    attach_drafter(language_model, drafter)
    assert language_model._omlx_dflash_capture_mean is True
    assert language_model.model._omlx_dflash_capture_mean is True
    detach_drafter(language_model)
    assert language_model.model._omlx_dflash_capture_mean is False

    language_model = SimpleNamespace(model=SimpleNamespace())
    drafter = SimpleNamespace(kind="dflash", depth=3, clear=lambda: None)
    attach_drafter(language_model, drafter)
    assert language_model._omlx_dflash_capture_mean is False


@pytest.mark.parametrize(
    ("model_type", "engine_type"),
    [("llama", "vlm"), (None, "vlm"), ("minicpmv4_7", None)],
)
def test_unsupported_targets_report_their_model_type(model_type, engine_type):
    supported, reason = dflash_compatibility(model_type, engine_type)
    assert not supported
    assert f"model_type='{model_type or ''}'" in reason
    assert "MiniCPM-V 4.7" in reason
