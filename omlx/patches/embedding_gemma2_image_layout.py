# SPDX-License-Identifier: Apache-2.0
"""PIL images must not have their channel layout inferred (mlx-vlm #2346).

``EmbeddingGemma2ImageProcessor.preprocess`` converts each PIL image to a
numpy array and calls ``infer_channel_dimension_format`` on it. PIL arrays
are always HWC, but the inference only looks at shape, so two image classes
break:

- 1-pixel-tall: ``(1, W, 3)`` reads as channels-first ``(C=1, H, W)``. The
  single channel survives the resize squeeze, patches become
  ``1 * 16 * 16 = 256`` wide instead of ``3 * 16 * 16 = 768``, and the
  vision tower fails with a matmul shape error.
- 3-pixel-tall: ``(3, W, 3)`` is ambiguous and resolves to channels-first.
  Nothing crashes: the image rows are treated as channels, so patch
  positions, 2-D RoPE and pooling order are all wrong while the soft-token
  count stays 280. Retrieval quality silently degrades.

Fix by forwarding ``input_data_format=ChannelDimension.LAST`` for PIL
inputs, which the processor already honours — the same remedy the upstream
issue suggests and PR #2367 applies to the base Gemma 4 processor. The
EmbeddingGemma 2 subclass overrides ``preprocess`` and does not inherit
that fix, so patch it here.

Upstream tracking: https://github.com/Blaizzy/mlx-vlm/issues/2346
"""

_PATCHED = False


def _all_pil(images) -> bool:
    """True when every leaf of a (possibly nested) image container is a PIL image.

    The HF-style processor call chain hands ``preprocess`` a nested list
    (``[[PIL.Image, ...], ...]``), while direct callers pass a flat list.
    Mixed PIL/numpy batches stay unforced: their layout stays ambiguous and
    the unpatched inference path is kept as-is for them.
    """
    from PIL import Image

    def _leaves(node):
        if isinstance(node, (list, tuple)):
            for child in node:
                yield from _leaves(child)
        else:
            yield node

    leaves = list(_leaves(images))
    return bool(leaves) and all(isinstance(leaf, Image.Image) for leaf in leaves)


def apply_embedding_gemma2_image_layout_patch() -> bool:
    """Return True when the patch was installed, False if already applied
    or the model module is unavailable in this mlx-vlm version."""
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_vlm.models.embedding_gemma2.image_processing_embedding_gemma2 import (
            EmbeddingGemma2ImageProcessor,
        )
    except (ImportError, ModuleNotFoundError):
        return False

    from transformers.image_utils import ChannelDimension

    preprocess = EmbeddingGemma2ImageProcessor.preprocess
    if getattr(EmbeddingGemma2ImageProcessor, "_omlx_pil_layout_patched", False):
        _PATCHED = True
        return False

    def _preprocess(self, images, return_tensors=None, input_data_format=None, **kwargs):
        # Grayscale 2-D arrays never reach the ambiguous branch: the
        # processor promotes them to channels-first before consulting
        # ``input_data_format``. RGB(A) PIL images are the HWC case.
        if input_data_format is None and _all_pil(images):
            input_data_format = ChannelDimension.LAST
        return preprocess(
            self,
            images,
            return_tensors=return_tensors,
            input_data_format=input_data_format,
            **kwargs,
        )

    _preprocess._omlx_pil_layout_patched = True
    EmbeddingGemma2ImageProcessor._omlx_pil_layout_patched = True
    EmbeddingGemma2ImageProcessor.preprocess = _preprocess
    _PATCHED = True
    return True
