# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the vendored Unlimited-OCR mlx-vlm compatibility layer."""

from __future__ import annotations


def test_unlimited_ocr_compat_installs_vendor_module():
    from omlx.patches.mlx_vlm_unlimited_ocr_compat import (
        apply_mlx_vlm_unlimited_ocr_compat_patch,
    )

    apply_mlx_vlm_unlimited_ocr_compat_patch()

    import mlx_vlm.models.unlimited_ocr as unlimited_ocr

    assert hasattr(unlimited_ocr, "Model")
    assert hasattr(unlimited_ocr, "ModelConfig")
    assert hasattr(unlimited_ocr, "RingSlidingKVCache")


def test_unlimited_ocr_model_remapping_resolves_module():
    from omlx.patches.mlx_vlm_unlimited_ocr_compat import (
        apply_mlx_vlm_unlimited_ocr_compat_patch,
    )

    apply_mlx_vlm_unlimited_ocr_compat_patch()

    from mlx_vlm.utils import MODEL_REMAPPING, get_model_and_args

    assert MODEL_REMAPPING.get("unlimited-ocr") == "unlimited_ocr"

    # Without the remapping, get_model_and_args would try to import the literal
    # dashed module name and fall through to the speculative-drafter lookup that
    # produced the "No module named ... unlimited-ocr" error in issue #2314.
    module, model_type = get_model_and_args(
        {
            "model_type": "unlimited-ocr",
            "architectures": ["UnlimitedOCRForCausalLM"],
            "vision_config": {},
            "text_config": {},
        }
    )

    assert model_type == "unlimited_ocr"
    assert module.__name__ == "mlx_vlm.models.unlimited_ocr"


def test_unlimited_ocr_prompt_uses_single_image_token():
    from omlx.patches.mlx_vlm_unlimited_ocr_compat import (
        apply_mlx_vlm_unlimited_ocr_compat_patch,
    )

    apply_mlx_vlm_unlimited_ocr_compat_patch()

    from mlx_vlm.prompt_utils import get_message_json

    # Single page: one <image> token.
    single = get_message_json("unlimited-ocr", "document parsing.", num_images=1)
    assert single == {"role": "user", "content": "<image>document parsing."}

    # Multi page: still exactly one <image> token for all pages (infer_multi).
    multi = get_message_json("unlimited-ocr", "Multi page parsing.", num_images=3)
    assert str(multi["content"]).count("<image>") == 1

    # Non-user / skip_image_token: no image token injected.
    assistant = get_message_json("unlimited-ocr", "ok", role="assistant", num_images=1)
    assert assistant == {"role": "assistant", "content": "ok"}


def test_unlimited_ocr_prompt_leaves_other_models_untouched():
    from omlx.patches.mlx_vlm_unlimited_ocr_compat import (
        apply_mlx_vlm_unlimited_ocr_compat_patch,
    )

    apply_mlx_vlm_unlimited_ocr_compat_patch()

    from mlx_vlm.prompt_utils import get_message_json

    # deepseekocr must keep going through the original IMAGE_TOKEN_NEWLINE
    # formatter (newline-suffixed token), not the unlimited-ocr wrapper.
    message = get_message_json("deepseekocr", "hello", num_images=1)
    assert message == {"role": "user", "content": "<image>\nhello"}
