"""Torch-free Pixtral/Mistral3 processor path for the pinned mlx-vlm.

Backports mlx-vlm PR #1502 (upstream e3fec79) and fixes the tokenizer
backend selection so Pixtral-style checkpoints (Devstral, Mistral Small)
load in oMLX's torch-free bundle. Fixes issue #2263, which is two stacked
loading failures:

1. The pinned custom ``Mistral3Processor``/``PixtralProcessor`` build their
   image processor through transformers' ``AutoImageProcessor``, which is
   torch/torchvision-gated since transformers 5.5. The resulting
   ``ImportError`` is swallowed by ``install_auto_processor_patch`` and the
   load falls back to transformers' ``PixtralProcessor``, which is also
   torch-gated and raises the error users see.
2. With the processor unblocked, ``mlx_vlm.utils.load_processor`` still
   dies constructing its streaming detokenizer: transformers 5.12 resolves
   repos shipping ``tekken.json`` to the vocab-less ``MistralCommonBackend``
   whenever mistral-common is installed (oMLX depends on it explicitly).

The vendored modules replace the image-processor path with upstream's local
PIL/NumPy ``PixtralImageProcessor`` and pin the tokenizer back to
``TokenizersBackend`` via the ``fix_mistral_regex`` escape hatch.

Installation transplants ``__call__`` and ``from_pretrained`` onto the
pinned classes in place. The upstream ``install_auto_processor_patch``
registration keeps routing to those classes, so this works regardless of
AutoProcessor patch-chain order. Drop the ``__call__``/``from_pretrained``
transplants once the mlx-vlm pin includes e3fec79; the tokenizer-backend
fix has no upstream equivalent and must stay.
"""

import logging

logger = logging.getLogger(__name__)

_MARKER = "_omlx_pixtral_torch_free"


def _transplant(target_cls, vendor_cls, attr_names):
    for name in attr_names:
        vendored = vendor_cls.__dict__[name]
        setattr(target_cls, name, vendored)
    setattr(target_cls, _MARKER, True)


def apply_pixtral_torch_free_patch() -> bool:
    """Install the torch-free processor bodies onto the pinned mlx-vlm.

    Idempotent. Returns True when the patch is (already) in place.
    """
    try:
        from mlx_vlm.models.mistral3 import processing_mistral3 as pin_m3
        from mlx_vlm.models.pixtral import processing_pixtral as pin_px

        if getattr(pin_m3.Mistral3Processor, _MARKER, False) and getattr(
            pin_px.PixtralProcessor, _MARKER, False
        ):
            return True

        from .vendor.processing_mistral3 import Mistral3Processor
        from .vendor.processing_pixtral import PixtralProcessor

        _transplant(
            pin_m3.Mistral3Processor,
            Mistral3Processor,
            ("__call__", "from_pretrained"),
        )
        _transplant(
            pin_px.PixtralProcessor,
            PixtralProcessor,
            ("__call__", "from_pretrained"),
        )
    except Exception:
        logger.warning(
            "Pixtral torch-free patch not installed; Pixtral/Mistral3 "
            "checkpoints will fail to load without torch",
            exc_info=True,
        )
        return False

    logger.info(
        "Pixtral torch-free processor patch installed (mlx-vlm PR #1502 "
        "backport + TokenizersBackend pin)"
    )
    return True
