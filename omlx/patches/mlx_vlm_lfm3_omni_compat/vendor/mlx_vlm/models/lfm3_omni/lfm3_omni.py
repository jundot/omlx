# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3-Omni checkpoint reference implementation.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

"""LFM3-Omni: the LFM3-Audio model with the LFM3-VL vision tower and projector."""

from ..base import InputEmbeddingsFeatures
from ..lfm3_audio.lfm3_audio import Model as Lfm3AudioModel
from ..lfm3_vl.lfm3_vl import Model as Lfm3VlModel
from ..lfm3_vl.lfm3_vl import MultiModalProjector
from ..lfm3_vl.vision import VisionModel


class Model(Lfm3AudioModel):
    encode_images = Lfm3VlModel.encode_images
    # LFM3-Omni's model card keeps the earlier speech sampling
    audio_sampling = {"audio_temperature": 0.4, "audio_repetition_penalty": 1.2}

    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config)

    def placeholder_token_ids(self):
        config = self.config
        return super().placeholder_token_ids() + [
            config.image_token_id,
            config.video_token_id,
        ]

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        embeddings = Lfm3VlModel.get_input_embeddings(
            self, input_ids, pixel_values, **kwargs
        ).inputs_embeds
        return InputEmbeddingsFeatures(
            inputs_embeds=self.merge_speech(embeddings, input_ids, **kwargs)
        )

    def sanitize(self, weights):
        weights = Lfm3VlModel.sanitize(self, weights)
        prefix = "audio_encoder."
        return super().sanitize(
            {
                ("audio_tower." + k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in weights.items()
            }
        )
