# SPDX-License-Identifier: Apache-2.0
"""DFlash/DSpark target allow-list in omlx.utils.model_loading."""

import pytest

from omlx.utils.model_loading import dflash_batched_supported, dflash_compatibility


@pytest.mark.parametrize(
    ("model_type", "engine_type"),
    [
        ("qwen3_5_moe", "vlm"),
        ("minicpmv4_7", "vlm"),
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


@pytest.mark.parametrize(
    ("model_type", "engine_type"),
    [("llama", "vlm"), (None, "vlm"), ("minicpmv4_7", None)],
)
def test_unsupported_targets_report_their_model_type(model_type, engine_type):
    supported, reason = dflash_compatibility(model_type, engine_type)
    assert not supported
    assert f"model_type='{model_type or ''}'" in reason
    assert "MiniCPM-V 4.7" in reason
