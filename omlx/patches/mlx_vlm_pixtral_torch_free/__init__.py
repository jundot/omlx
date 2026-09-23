# SPDX-License-Identifier: Apache-2.0
"""Select the vocabulary-backed tokenizer for Pixtral streaming."""

from functools import wraps


def _pin_tokenizer(processor):
    original = processor.from_pretrained.__func__
    if getattr(original, "_omlx_pixtral_torch_free", False):
        return

    @classmethod
    @wraps(original)
    def from_pretrained(cls, path, **kwargs):
        kwargs.setdefault("fix_mistral_regex", True)
        return original(cls, path, **kwargs)

    from_pretrained.__func__._omlx_pixtral_torch_free = True
    processor.from_pretrained = from_pretrained


def apply_pixtral_torch_free_patch() -> bool:
    from mlx_vlm.models.mistral3.processing_mistral3 import Mistral3Processor
    from mlx_vlm.models.pixtral.processing_pixtral import PixtralProcessor

    _pin_tokenizer(Mistral3Processor)
    _pin_tokenizer(PixtralProcessor)
    return True
