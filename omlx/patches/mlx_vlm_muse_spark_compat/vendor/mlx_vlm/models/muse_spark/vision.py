from typing import Sequence

import mlx.core as mx

from ..muse_glimmer.vision import PatchEmbedder as GlimmerPatchEmbedder
from ..muse_glimmer.vision import VisionModel as GlimmerVisionModel
from .config import VisionConfig


class PatchEmbedder(GlimmerPatchEmbedder):
    def __call__(
        self, pixel_values: mx.array, grid: Sequence[Sequence[int]]
    ) -> mx.array:
        embeddings = self.patch_embedding(pixel_values)
        # Muse Spark reads the position table of patch (h, w) at (w, h): the
        # reference interpolates over the (t, w, h) grid.
        swapped = [(t, w, h) for t, h, w in grid]
        positions = self._position_embeddings(swapped).astype(embeddings.dtype)
        return embeddings + positions


class VisionModel(GlimmerVisionModel):
    """Muse Glimmer perception encoder with the Muse Spark position lookup."""

    def __init__(self, config: VisionConfig):
        super().__init__(config)
        self.patch_embedder = PatchEmbedder(config)
