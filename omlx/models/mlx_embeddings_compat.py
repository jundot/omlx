# SPDX-License-Identifier: Apache-2.0
"""Compatibility patches for mlx-embeddings."""

import logging

logger = logging.getLogger(__name__)

_QWEN3_VL_PROCESSOR_PATCHED = False


def _ensure_qwen3_vl_mm_token_ids(processor):
    """Mirror ProcessorMixin multimodal token id fields for manually built processors."""
    defaults = {
        "image_ids": [getattr(processor, "image_token_id", None)],
        "video_ids": [getattr(processor, "video_token_id", None)],
        "audio_ids": [getattr(processor, "audio_token_id", None)],
    }
    for attr_name, value in defaults.items():
        if not hasattr(processor, attr_name):
            setattr(processor, attr_name, value)
    return processor


def patch_qwen3_vl_processor_for_torch_free_image_loading() -> None:
    """Keep mlx-embeddings Qwen3-VL loading off HF AutoImageProcessor."""
    global _QWEN3_VL_PROCESSOR_PATCHED
    if _QWEN3_VL_PROCESSOR_PATCHED:
        return

    try:
        from mlx_embeddings.models.qwen3_vl import processor as qwen3_vl_processor
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
            Qwen3VLImageProcessor,
            _qwen_vl_image_kwargs,
        )
    except Exception as exc:
        logger.debug("Qwen3-VL mlx-embeddings compatibility patch skipped: %s", exc)
        _QWEN3_VL_PROCESSOR_PATCHED = True
        return

    original_auto_image_processor = getattr(
        qwen3_vl_processor, "AutoImageProcessor", None
    )

    class TorchFreeQwen3VLAutoImageProcessor:
        _omlx_original_auto_image_processor = original_auto_image_processor

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            del cls, kwargs
            image_kwargs = _qwen_vl_image_kwargs(
                pretrained_model_name_or_path,
                default_patch_size=16,
            )
            return Qwen3VLImageProcessor(**image_kwargs)

    qwen3_vl_processor.AutoImageProcessor = TorchFreeQwen3VLAutoImageProcessor

    processor_cls = getattr(qwen3_vl_processor, "Processor", None)
    original_build_processor = getattr(processor_cls, "_build_processor", None)
    if (
        processor_cls is not None
        and original_build_processor is not None
        and not getattr(original_build_processor, "_omlx_patched", False)
    ):

        def build_processor_with_mm_token_ids(tokenizer, image_processor):
            processor = original_build_processor(tokenizer, image_processor)
            return _ensure_qwen3_vl_mm_token_ids(processor)

        build_processor_with_mm_token_ids._omlx_patched = True
        processor_cls._build_processor = staticmethod(build_processor_with_mm_token_ids)

    _QWEN3_VL_PROCESSOR_PATCHED = True
    logger.debug("Applied torch-free image loader patch for mlx-embeddings Qwen3-VL")
